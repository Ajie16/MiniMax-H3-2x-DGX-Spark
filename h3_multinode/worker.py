"""One-GPU-per-host diffusion worker used by the Ray executor."""

from __future__ import annotations

import os
import socket
import time
import traceback
from typing import Any
from unittest.mock import patch

import torch

from vllm.logger import init_logger

from .lora_catalog import assert_lora_path_allowed, resolve_adapter, resolved_scale
from .lora_mapping import install_h3_lora_packed_mapping

logger = init_logger(__name__)


def _move_tensors_to_cpu(value: Any) -> Any:
    """Move nested tensor payloads off CUDA before Ray serializes them."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_cpu(item) for item in value)

    output = getattr(value, "output", None)
    if output is not None:
        value.output = _move_tensors_to_cpu(output)
    for field in ("trajectory_timesteps", "trajectory_latents", "trajectory_log_probs"):
        item = getattr(value, field, None)
        if item is not None:
            setattr(value, field, _move_tensors_to_cpu(item))
    return value


def _make_worker_class(master_addr: str, master_port: int):
    """Create a DiffusionWorker that honors cross-host rank coordinates."""
    from vllm_omni.diffusion.worker import diffusion_worker as worker_module
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    from vllm_omni.platforms import current_omni_platform

    class MultiNodeDiffusionWorker(DiffusionWorker):
        def init_device(self) -> None:
            original_init = worker_module.init_distributed_environment
            original_get_device = current_omni_platform.get_torch_device

            def cross_host_init(*, world_size: int, rank: int, **_: Any) -> None:
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = str(master_port)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(world_size)
                original_init(
                    world_size=world_size,
                    rank=rank,
                    distributed_init_method="env://",
                    local_rank=self.local_rank,
                )

            def local_device(_: int) -> torch.device:
                return original_get_device(self.local_rank)

            # Upstream currently assumes every diffusion rank is local: it
            # overwrites MASTER_ADDR with localhost and selects cuda:<global
            # rank>. Intercept only those two calls while retaining the rest
            # of its tested device/model-parallel initialization.
            with (
                patch.object(worker_module, "init_distributed_environment", cross_host_init),
                patch.object(current_omni_platform, "get_torch_device", local_device),
            ):
                super().init_device()

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["RANK"] = str(self.rank)
            os.environ["WORLD_SIZE"] = str(self.od_config.num_gpus)

        def init_lora_manager(self) -> None:
            install_h3_lora_packed_mapping()
            super().init_lora_manager()
            manager = getattr(self, "lora_manager", None)
            if manager is None or os.environ.get("H3_LORA_MODE", "off") != "static":
                return
            adapters = getattr(manager, "list_adapters", lambda: [])()
            if adapters:
                manager.pin_adapter(adapters[0])

    return MultiNodeDiffusionWorker


class RayDiffusionWorker:
    """Ray actor implementation; one instance is pinned to each Spark."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        od_config: Any,
        master_addr: str,
        master_port: int,
    ) -> None:
        from vllm_omni.plugins import load_omni_general_plugins

        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self._closed = False

        if int(od_config.num_gpus) != world_size:
            raise ValueError(
                f"od_config.num_gpus={od_config.num_gpus} does not match world_size={world_size}"
            )

        load_omni_general_plugins()
        install_h3_lora_packed_mapping()
        worker_class = _make_worker_class(master_addr, master_port)
        self.worker = worker_class(
            local_rank=0,
            rank=rank,
            od_config=od_config,
        )

    def ready(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "hostname": socket.gethostname(),
            "master": f"{self.master_addr}:{self.master_port}",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "distributed": torch.distributed.is_initialized(),
            "backend": torch.distributed.get_backend() if torch.distributed.is_initialized() else None,
        }

    def execute(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output_rank: int | None,
        execute: bool,
    ) -> dict[str, Any]:
        if not execute:
            return {"rank": self.rank, "ok": True, "result": None, "skipped": True}
        try:
            if method == "execute_model":
                self._prepare_lora_request(args[0] if args else None)
            started = time.monotonic()
            result = getattr(self.worker, method)(*args, **kwargs)
            if method == "execute_model":
                self._log_lora_activation(args[0] if args else None, started)
            if output_rank is not None and self.rank != output_rank:
                result = None
            else:
                result = _move_tensors_to_cpu(result)
            return {
                "rank": self.rank,
                "ok": True,
                "result": result,
                "skipped": False,
            }
        except BaseException as exc:
            return {
                "rank": self.rank,
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "result": None,
                "skipped": False,
            }

    def _prepare_lora_request(self, req: Any) -> None:
        sampling = getattr(req, "sampling_params", None)
        if sampling is None:
            return
        mode = os.environ.get("H3_LORA_MODE", "off")
        lora_request = getattr(sampling, "lora_request", None)
        if mode == "off":
            if lora_request is not None:
                raise RuntimeError("LoRA is disabled (H3_LORA_MODE=off)")
            return
        if mode == "static" and lora_request is None:
            entry = resolve_adapter(os.environ["H3_LORA_NAME"])
            sampling.lora_request = entry.to_request()
            sampling.lora_scale = resolved_scale(catalog_default=entry.default_scale)
            lora_request = sampling.lora_request
        if lora_request is None:
            return
        expected = None
        if mode == "static":
            expected = resolve_adapter(os.environ["H3_LORA_NAME"]).resolved_path
        elif getattr(lora_request, "lora_name", None):
            expected = resolve_adapter(lora_request.lora_name).resolved_path
        assert_lora_path_allowed(str(lora_request.lora_path), expected=expected)

    def _log_lora_activation(self, req: Any, started: float) -> None:
        sampling = getattr(req, "sampling_params", None)
        lora_request = getattr(sampling, "lora_request", None) if sampling is not None else None
        mode = os.environ.get("H3_LORA_MODE", "off")
        name = getattr(lora_request, "lora_name", None) if lora_request is not None else "-"
        int_id = getattr(lora_request, "lora_int_id", None) if lora_request is not None else "-"
        path = getattr(lora_request, "lora_path", None) if lora_request is not None else "-"
        scale = getattr(sampling, "lora_scale", None) if sampling is not None else "-"
        load_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "H3 LoRA rank=%s mode=%s name=%s int_id=%s scale=%s path=%s load_ms=%s compile_cache=n/a",
            self.rank,
            mode,
            name,
            int_id,
            scale,
            path,
            load_ms,
        )

    def ping(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"rank {self.rank} is closed")
        return self.ready()

    def shutdown(self) -> bool:
        if not self._closed:
            self.worker.shutdown()
            self._closed = True
        return True
