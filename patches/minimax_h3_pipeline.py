# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for MiniMax H3 FL2VA and Ref2VA partitions."""

from __future__ import annotations

import functools
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_dit_group,
    init_world_group,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .denoise_loop import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .encoder import MiniMaxH3Qwen3VLEncoder
from .minimax_h3_transformer import MiniMaxH3DiTModel
from .comfy_kitchen_int8 import ComfyKitchenINT8Config
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import (
    minimax_h3_multi_image_presentation_ids,
    minimax_h3_multi_image_presentation_token_tags,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)
from .reference_video import (
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
)
from .time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
    minimax_h3_time_shift_sigmas,
)
from h3_multinode.ref_inputs import coerce_ref_sequence, ref2va_condition_labels, validate_ref_counts

from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

logger = init_logger(__name__)

MINIMAX_H3_FPS = 24
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32


def _minimax_h3_post_process(output, output_type: str = "np"):
    """Convert the joint video/audio output without capturing worker state.

    The callable crosses the multiprocessing result queue, so it must remain a
    module-level function that the standard pickle module can resolve.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    video, audio = output
    if output_type == "latent":
        return output
    if output_type == "np":
        # Export uint8 on the GPU instead of float32 [0, 1]: the serving layer
        # converts float frames back to uint8 with ~5 full CPU passes over the
        # (F, H, W, 3) buffer (~15 s solo for 243x1344x768, far worse under
        # swap), while it passes uint8 through with a single stack copy.
        # Semantics identical to np.round(np.clip(v, 0, 1) * 255) downstream
        # (both round half to even; verified bit-exact).
        video = (
            video.detach().float().mul(255.0).round().clamp(0, 255).to(torch.uint8)
            .permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
        )
        audio = audio.detach().float().cpu().numpy()
        video = [sample for sample in video]
    return {
        "video": video,
        "audio": audio,
        "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        "fps": MINIMAX_H3_FPS,
    }


def get_minimax_h3_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config
    return _minimax_h3_post_process


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"image tensor must be [C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return Image.fromarray(array.clip(0, 255).astype(np.uint8)).convert("RGB")
    raise TypeError(f"unsupported MiniMax H3 image input {type(value)!r}")


def _load_images(value: Any) -> list[Image.Image]:
    return [_load_image(item) for item in coerce_ref_sequence(value)]


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    import torchaudio

    if isinstance(value, (str, os.PathLike)):
        return torchaudio.load(str(value))
    if isinstance(value, tuple) and len(value) == 2:
        waveform, sample_rate = value
        waveform = torch.as_tensor(waveform).float()
        return waveform, int(sample_rate)
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
        if waveform is not None and sample_rate is not None:
            return torch.as_tensor(waveform).float(), int(sample_rate)
    raise TypeError("MiniMax H3 audio input must be a path, (waveform, sample_rate), or a waveform mapping")


def _load_audios(value: Any) -> list[tuple[torch.Tensor, int]]:
    return [_load_audio(item) for item in coerce_ref_sequence(value)]


def _cat_rows(parts: list[torch.Tensor]) -> torch.Tensor | None:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _dit_rank_world() -> tuple[Any, int, int]:
    if not dist.is_initialized():
        return None, 0, 1
    group = get_dit_group()
    return group, dist.get_rank(group), dist.get_world_size(group)


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if tensor is None:
            raise ValueError("source tensor is required for single-rank execution")
        return tensor.to(device=device, dtype=dtype)

    shape = torch.zeros(5, dtype=torch.long, device=device)
    if rank == 0:
        if tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        shape[0] = tensor.ndim
        shape[1 : tensor.ndim + 1] = torch.tensor(
            tensor.shape,
            device=device,
        )
    dist.broadcast(shape, src=0, group=group)
    ndim = int(shape[0].item())
    tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
    if rank == 0:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(tensor_shape, device=device, dtype=dtype)
    dist.broadcast(output, src=0, group=group)
    return output


class _SingleRankEncoderGroup:
    """Lightweight encoder group for ``text_encoder_tp_size == 1``.

    Avoids creating a distributed ``GroupCoordinator`` with a single-member
    rank set, which would assert on every other DiT rank that is not part of
    the group.  The pipeline and encoder only use the attributes below, and
    all ``world_size == 1`` code paths short-circuit before any collective.
    """

    world_size: int = 1
    ranks: list[int] = [0]

    def __init__(self, rank: int) -> None:
        self.rank_in_group = 0 if rank == 0 else -1
        self.device_group = None


def _maybe_torch_profiled(stage: str):
    """Wrap a pipeline stage with torch.profiler, default off (zero overhead).

    Activated per request by either:
    - ``H3_TORCH_PROFILER_DIR`` naming an output directory, or
    - the sentinel file ``/tmp/h3_profiler_on`` existing in the rank's
      container (output then goes to ``/tmp/h3prof``) — this toggle needs no
      restart, so it can be switched on for one smoke and removed.

    Each profiled request writes one gzipped chrome trace per rank
    (hostname/pid in the filename) and logs the top-40 op table by self
    device time. Profiling makes the wrapped request several times slower —
    do not leave it on for production runs.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            out_dir = os.environ.get("H3_TORCH_PROFILER_DIR", "")
            if not out_dir and os.path.exists("/tmp/h3_profiler_on"):
                out_dir = "/tmp/h3prof"
            if not out_dir:
                return fn(self, *args, **kwargs)
            import torch.profiler

            os.makedirs(out_dir, exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                result = fn(self, *args, **kwargs)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(
                out_dir,
                f"{stage}-{stamp}-{os.uname().nodename}-pid{os.getpid()}.json.gz",
            )
            try:  # profiling must never kill a request
                prof.export_chrome_trace(path)
                logger.info("torch profiler %s trace written to %s", stage, path)
            except Exception as exc:
                logger.warning("torch profiler %s trace export failed: %s", stage, exc)
            logger.info(
                "torch profiler %s top-40 ops by self device time:\n%s",
                stage,
                prof.key_averages().table(
                    sort_by="self_device_time_total",
                    row_limit=40,
                    max_name_column_width=80,
                ),
            )
            return result

        return wrapper

    return deco


class MiniMaxH3Pipeline(
    nn.Module,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
):
    """CFG-distilled joint video/audio generation for MiniMax H3."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = [
        "_prepare_reference_videos",
        "encode_prompt",
        "_encode_video_conditions",
        "_encode_video_audio_conditions",
        "diffuse",
        "decode",
    ]
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        if int(self.parallel_config.cfg_parallel_size) != 1:
            raise ValueError("MiniMax-H3 is CFG-distilled and has no negative branch; cfg_parallel_size must be 1")
        self.device = get_local_device()
        model_path = str(od_config.model)
        model_index = json.loads((Path(model_path) / "model_index.json").read_text(encoding="utf-8"))
        release = model_index.get("_minimax_h3") or {}
        self.partition = str(release.get("partition", ""))
        self.supported_tasks = frozenset(release.get("tasks") or ())
        shifts = release.get("sigma_shift_scales") or {}
        self.default_video_shift = float(shifts.get("video", 12.0))
        self.default_audio_shift = float(shifts.get("audio", 3.0))

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        quant_config = od_config.quantization_config
        if os.environ.get("H3_QUANTIZATION") == "int8_convrot":
            quant_config = ComfyKitchenINT8Config()
        self.transformer = MiniMaxH3DiTModel(
            od_config,
            quant_config=quant_config,
        )

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer",
            local_files_only=os.path.isdir(model_path),
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            subfolder="processor",
            local_files_only=os.path.isdir(model_path),
        )

        _, rank, dit_world = _dit_rank_world()
        self._dit_rank = rank
        text_encoder_tp_size = int(getattr(self.parallel_config, "text_encoder_tp_size", 1))
        if text_encoder_tp_size < 1:
            raise ValueError(f"text_encoder_tp_size must be >= 1, got {text_encoder_tp_size}")
        if text_encoder_tp_size > dit_world:
            raise ValueError(
                f"text_encoder_tp_size must not exceed the DiT group size ({dit_world}), got {text_encoder_tp_size}"
            )
        # The Qwen3-VL text model uses 64 attention heads / 8 KV heads; the
        # encoder shards them across the encoder TP ranks.
        if 64 % text_encoder_tp_size or 8 % text_encoder_tp_size:
            raise ValueError(
                "text_encoder_tp_size must divide both Qwen3-VL "
                f"num_attention_heads (64) and num_key_value_heads (8), "
                f"got {text_encoder_tp_size}"
            )
        self.text_encoder_tp_size = text_encoder_tp_size
        self.text_encoder_group = self._build_text_encoder_group(text_encoder_tp_size)
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            os.path.join(model_path, "text_encoder"),
            device=self.device,
            load_model=rank < text_encoder_tp_size,
            encoder_group=self.text_encoder_group,
        )
        self.video_vae = MiniMaxH3VideoVAE(
            os.path.join(model_path, "video_vae"),
            device=self.device,
        )
        self.audio_vae = MiniMaxH3AudioVAE(
            os.path.join(model_path, "audio_vae"),
            device=self.device,
        )
        # Registry-side VAE patch-parallel discovery uses ``pipeline.vae``.
        self.vae = self.video_vae
        self._apply_vae_decode_tuning()

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=(od_config.enable_diffusion_pipeline_profiler)
        )

    def _apply_vae_decode_tuning(self) -> None:
        """Apply optional env-gated VAE decode tuning knobs.

        Stock decode splits every frame into 256px spatial tiles
        (``vae_tile_size``) and, because the checkpoint config leaves
        ``stack_tiling`` unset, decodes them one forward per tile per
        temporal chunk. On SP2 an 832x480 canvas is 12 tiles = 6
        sequential decoder forwards per rank per temporal chunk. These
        knobs trade activation memory for decode latency without editing
        the checkpoint:

        - ``H3_VAE_DECODER_TILE_SIZE``: pixel-domain decoder tile size
          (``decoder_tile_size``). Must leave at least ``sp_size`` tiles
          per decode or the patch-parallel gather fails with "Found
          empty tasks"; e.g. 512 on an 832x480 canvas gives exactly 2.
        - ``H3_VAE_DECODER_TILE_OVERLAP``: minimum pixel overlap between
          decoder tiles (``decoder_tile_overlap_min``).
        - ``H3_VAE_STACK_TILING=1``: batch each rank's tiles into a
          single decoder forward per temporal chunk.
        """
        model = self.video_vae.model
        tile_size = os.environ.get("H3_VAE_DECODER_TILE_SIZE", "")
        if tile_size:
            model.decoder_tile_size = int(tile_size)
        tile_overlap = os.environ.get("H3_VAE_DECODER_TILE_OVERLAP", "")
        if tile_overlap:
            model.decoder_tile_overlap_min = int(tile_overlap)
        stack_tiling = os.environ.get("H3_VAE_STACK_TILING", "").lower() in ("1", "true", "yes")
        if stack_tiling:
            model.stack_tiling = True
        if tile_size or tile_overlap or stack_tiling:
            logger.info(
                "VAE decode tuning: decoder_tile_size=%d decoder_tile_overlap_min=%d stack_tiling=%s",
                model.decoder_tile_size,
                model.decoder_tile_overlap_min,
                model.stack_tiling,
            )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        prefix = "transformer."

        def transformer_weights():
            for name, tensor in weights:
                if name.startswith(prefix):
                    yield name[len(prefix) :], tensor

        loaded = self.transformer.load_weights(transformer_weights())
        self.transformer.post_load_weights()
        loaded_with_prefix = {prefix + name for name in loaded}
        # The text encoder and both VAEs load eagerly in ``__init__`` rather
        # than through ``weights_sources``. Record them for the runner's strict
        # missing-parameter check.
        for component_name in ("text_encoder", "video_vae", "audio_vae"):
            component = getattr(self, component_name)
            loaded_with_prefix.update(f"{component_name}.{name}" for name, _ in component.named_parameters())
        return loaded_with_prefix

    def _resolve_task(
        self,
        requested: str | None,
        multi_modal_data: dict[str, Any],
    ) -> str:
        if requested is None:
            if self.partition == "ref2va":
                requested = "ref2va"
            elif multi_modal_data.get("image") is not None:
                requested = "fl2va"
            else:
                requested = "t2va"
        task = str(requested).lower()
        if task not in self.supported_tasks:
            raise ValueError(
                f"checkpoint partition {self.partition!r} supports {sorted(self.supported_tasks)}, got task={task!r}"
            )
        return task

    def _resolve_shape(
        self,
        task: str,
        sampling: Any,
        image: Image.Image | None,
    ) -> tuple[int, int, int, int, int]:
        fps = int(sampling.fps or MINIMAX_H3_FPS)
        if fps != MINIMAX_H3_FPS:
            raise ValueError(f"MiniMax H3 output fps is fixed at {MINIMAX_H3_FPS}")
        extra = sampling.extra_args or {}
        duration = extra.get("duration")
        if duration is not None:
            requested_frames = int(round(float(duration) * fps))
        elif int(sampling.num_frames or 1) > 1:
            requested_frames = int(sampling.num_frames)
        else:
            requested_frames = 124 if task == "ref2va" else 209
        num_frames = minimax_h3_align_frame_count(requested_frames)

        height = sampling.height
        width = sampling.width
        if height is None or width is None:
            if task == "fl2va" and image is not None:
                ratio = image.width / image.height
                if ratio >= 1:
                    height = 768
                    width = _align_multiple(768 * ratio)
                else:
                    width = 768
                    height = _align_multiple(768 / ratio)
            else:
                height, width = 768, 1344
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        if min(height, width) <= 0:
            raise ValueError(f"invalid MiniMax H3 canvas {width}x{height}")
        if width > 4 * height or height > 4 * width:
            raise ValueError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")

        latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
        audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / fps)
        return height, width, num_frames, latent_t, audio_t

    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | list[Image.Image] | None = None,
        images: list[Image.Image] | None = None,
        prepared_videos: list[dict[str, Any]] | None = None,
        standalone_audio_count: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        image_list = list(images or [])
        if not image_list and image is not None:
            image_list = image if isinstance(image, list) else [image]
        if rank == 0:
            if task == "t2va":
                ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
                tags = torch.ones(ids.shape[0], dtype=torch.long)
                vision_kwargs = {}
            else:
                merge = int(self.processor.image_processor.merge_size) ** 2
                image_token_counts: list[int] = []
                video_has_audio = [bool(item.get("input_has_audio")) for item in (prepared_videos or [])]
                if image_list:
                    vision = self.processor.image_processor(
                        images=image_list,
                        return_tensors="pt",
                    )
                    image_grid = vision["image_grid_thw"]
                    image_token_counts = [
                        int(image_grid[index].prod().item()) // merge for index in range(len(image_list))
                    ]
                    vision_kwargs["pixel_values"] = vision["pixel_values"]
                    vision_kwargs["image_grid_thw"] = image_grid
                block_counts = None
                block_timestamps = None
                if prepared_videos:
                    videos = []
                    sampled_videos = []
                    for index, item in enumerate(prepared_videos):
                        sampled = sample_reference_video_frames(
                            item["prepared_path"],
                            workdir=str(Path(item["prepared_path"]).parent / f"qwen_frames_{index}"),
                        )
                        videos.append(np.stack(sampled["frames"]))
                        sampled_videos.append(sampled)
                    video_vision = self.processor.video_processor(
                        videos=videos,
                        do_sample_frames=False,
                        return_tensors="pt",
                    )
                    video_grid = video_vision["video_grid_thw"]
                    block_counts = []
                    block_timestamps = []
                    for index, sampled in enumerate(sampled_videos):
                        blocks = int(video_grid[index, 0])
                        per_block = int(video_grid[index, 1]) * int(video_grid[index, 2]) // merge
                        timestamps = sampled["block_timestamps"]
                        if len(timestamps) != blocks:
                            raise ValueError(
                                f"video block count mismatch: processor={blocks}, timestamps={len(timestamps)}"
                            )
                        block_counts.append([per_block] * blocks)
                        block_timestamps.append(timestamps)
                    vision_kwargs["pixel_values_videos"] = video_vision["pixel_values_videos"]
                    vision_kwargs["video_grid_thw"] = video_grid
                if not image_list and not prepared_videos:
                    raise ValueError(f"{task} requires at least one image or video reference")
                if task == "fl2va":
                    ids = minimax_h3_multi_image_presentation_ids(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts or [1],
                    )
                    tags = minimax_h3_multi_image_presentation_token_tags(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts or [1],
                    )
                else:
                    ids, tags = minimax_h3_ref2va_video_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=ref2va_condition_labels(
                            n_images=len(image_list),
                            video_has_audio=video_has_audio,
                            n_audios=int(standalone_audio_count),
                        ),
                        image_token_count=image_token_counts or None,
                        video_block_token_counts=block_counts,
                        video_block_timestamps=block_timestamps,
                    )

            logger.info(
                "MiniMax H3 %s Qwen presentation: %d tokens%s%s",
                task,
                int(ids.shape[0]),
                (f", {len(image_list)} reference images" if image_list else ""),
                (f", {len(prepared_videos)} reference videos" if prepared_videos else ""),
            )

        if rank < self.text_encoder_tp_size:
            # Distribute the encode inputs from the DiT main rank to the other
            # encoder TP ranks, then run the distributed encode on all of them.
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)

        hidden = _broadcast_tensor(
            hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )
        tags = _broadcast_tensor(
            tags,
            dtype=torch.long,
            device=self.device,
        )
        return hidden, tags

    def _build_text_encoder_group(self, text_encoder_tp_size: int) -> Any:
        """Create the encoder tensor-parallel process group.

        The encoder group covers the first ``text_encoder_tp_size`` DiT ranks
        (the DiT group is always global ranks ``[0, dit_world)``).  Every rank
        participates in ``new_group`` so the collective completes; ranks
        outside the group never run encoder collectives.  For a single-rank
        encoder we return a lightweight placeholder so non-encoder ranks do
        not need to join a ``GroupCoordinator`` that would assert on ranks
        outside the group.
        """
        if text_encoder_tp_size == 1:
            return _SingleRankEncoderGroup(rank=self._dit_rank)
        ranks = list(range(text_encoder_tp_size))
        return init_world_group(
            ranks=ranks,
            local_rank=envs.LOCAL_RANK,
            backend=current_omni_platform.dist_backend,
        )

    def _encoder_group_broadcast_tensor(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from encoder rank 0 over the encoder TP group."""
        group = self.text_encoder_group
        if group.world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)

        shape = torch.zeros(8, dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            if tensor is None:
                raise ValueError("encoder rank 0 must provide a tensor to broadcast")
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, device=device)
        torch.distributed.broadcast(shape, src=group.ranks[0], group=group.device_group)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(value) for value in shape[1 : ndim + 1].tolist())
        if group.rank_in_group == 0:
            output = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            output = torch.empty(tensor_shape, device=device, dtype=dtype)
        torch.distributed.broadcast(output, src=group.ranks[0], group=group.device_group)
        return output

    def _distribute_encode_inputs(
        self,
        ids: torch.Tensor | None,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fan out encode inputs from encoder rank 0 to the encoder TP ranks.

        Mutates ``vision_kwargs`` in place so every encoder rank ends up with
        the same vision tensors, and returns the broadcast ``input_ids``.
        """
        keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        key_dtypes = {
            "pixel_values": torch.bfloat16,
            "pixel_values_videos": torch.bfloat16,
            "image_grid_thw": torch.long,
            "video_grid_thw": torch.long,
        }
        group = self.text_encoder_group
        device = self.device
        if group.world_size == 1:
            if ids is None:
                raise ValueError("encoder rank 0 must produce input ids")
            return ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(len(keys), dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            for index, key in enumerate(keys):
                mask[index] = 1 if key in vision_kwargs else 0
        torch.distributed.broadcast(mask, src=group.ranks[0], group=group.device_group)

        if group.rank_in_group == 0:
            ids = self._encoder_group_broadcast_tensor(ids, dtype=torch.long, device=device)
        else:
            ids = self._encoder_group_broadcast_tensor(None, dtype=torch.long, device=device)
        for index, key in enumerate(keys):
            if mask[index].item() == 0:
                continue
            source = vision_kwargs.get(key) if group.rank_in_group == 0 else None
            vision_kwargs[key] = self._encoder_group_broadcast_tensor(
                source,
                dtype=key_dtypes[key],
                device=device,
            )
        return ids

    def _prepare_reference_videos(
        self,
        values: Any,
        *,
        target_frame_count: int,
        workdir: str,
    ) -> list[dict[str, Any]] | None:
        _, rank, _ = _dit_rank_world()
        if rank != 0:
            return None
        return prepare_reference_videos(
            values,
            target_frame_count=target_frame_count,
            workdir=workdir,
        )

    def _encode_text_hidden(
        self,
        input_ids: torch.Tensor,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.od_config.enable_cpu_offload:
            # Invoke nn.Module.__call__ so the generic model-level offloader
            # swaps the resident DiT and encoder.
            return self.text_encoder(input_ids, **vision_kwargs)

        if self.od_config.enable_layerwise_offload:
            # Layerwise DiT offload already provides the low-residency encoder
            # phase used by the checkpoint reference.
            self.text_encoder.load_to_device()
            try:
                return self.text_encoder.encode_ids(input_ids, **vision_kwargs)
            finally:
                self.text_encoder.offload_to_cpu()

        # Keep both Qwen and DiT resident across requests. Moving either model
        # here makes encoder latency include a tens-of-gigabytes PCIe transfer,
        # which defeats the no-offload contract.
        self.text_encoder.load_to_device()
        return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    def _encode_visual_condition(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        _, rank, _ = _dit_rank_world()
        rows = self.video_vae.encode_image(image) if rank == 0 else None
        return _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )

    def _encode_audio_condition(
        self,
        audio: tuple[torch.Tensor, int],
    ) -> tuple[torch.Tensor, int]:
        _, rank, _ = _dit_rank_world()
        rows = None
        audio_t = 0
        if rank == 0:
            rows, audio_t = self.audio_vae.encode_waveform(*audio)
        audio_t_tensor = torch.tensor(
            [audio_t],
            dtype=torch.long,
            device=self.device,
        )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(audio_t_tensor, src=0, group=group)
        rows = _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )
        return rows, int(audio_t_tensor.item())

    def _encode_video_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        count: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        group, rank, world_size = _dit_rank_world()
        distributed_encode = self.video_vae.is_distributed_enabled()
        if distributed_encode:
            # Native tiled encode uses collectives, so every VPP rank must
            # enter each reference encode in the same input order.
            prepared_videos_list = [prepared_videos]
            dist.broadcast_object_list(
                prepared_videos_list,
                src=0,
                group=group,
                device=self.device,
            )
            prepared_videos = prepared_videos_list[0]

        rows = None
        shapes = torch.zeros((count, 3), dtype=torch.long, device=self.device)
        if rank == 0 or distributed_encode:
            if prepared_videos is None or len(prepared_videos) != count:
                raise ValueError("reference-video preparation is incomplete")
            encoded = [
                self.video_vae.encode_video(load_video_frames(item["prepared_path"])) for item in prepared_videos
            ]
            rows = torch.cat([item[0] for item in encoded])
            shapes = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        if distributed_encode:
            return (
                rows.to(device=self.device, dtype=torch.float32),
                [tuple(int(value) for value in item) for item in shapes.tolist()],
            )

        if world_size > 1:
            dist.broadcast(shapes, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [tuple(int(value) for value in item) for item in shapes.tolist()],
        )

    def _encode_video_audio_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
    ) -> tuple[torch.Tensor | None, list[int]]:
        _, rank, _ = _dit_rank_world()
        count = sum(has_audio)
        if count == 0:
            return None, []
        rows = None
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        if rank == 0:
            if prepared_videos is None:
                raise ValueError("rank 0 reference-video preparation is incomplete")
            encoded = [
                self.audio_vae.encode_waveform(*load_video_audio(item["original_path"]))
                for item in prepared_videos
                if item["input_has_audio"]
            ]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _initial_noise(
        self,
        *,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_rows = minimax_h3_patchify_video_latent(
            video,
            patch_size=(1, 2, 2),
        )
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_rows = torch.randn(
            audio_t * 2,
            32,
            generator=audio_generator,
            dtype=torch.float32,
        )
        return video_rows, audio_rows

    @_maybe_torch_profiled("diffuse")
    def diffuse(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_video, initial_audio = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None or ref_audio_t is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [
                    {"kind": "image", "latent_h": ref_h, "latent_w": ref_w},
                    {"kind": "audio", "ref_audio_t": ref_audio_t},
                ]
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=[0] if task == "fl2va" else None,
                frame_count=num_frames if task == "fl2va" else None,
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("visual condition shape is missing")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
            full_video = torch.zeros(
                branch.img_pos.shape[0],
                96,
                dtype=torch.float32,
            )
            full_video[branch.update_mask] = initial_video
            initial_video = full_video

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("reference audio length is missing")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )
            full_audio = torch.zeros(
                branch.audio_pos.shape[0],
                32,
                dtype=torch.float32,
            )
            full_audio[branch.audio_update_mask] = initial_audio
            initial_audio = full_audio

        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=video_shift,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=audio_shift,
        )
        with self.progress_bar(total=len(video_sigmas) - 1) as progress:
            video_rows, audio_rows = minimax_h3_denoise_loop(
                model=self.transformer,
                positive=branch,
                initial_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                keyframe_cond_rows=visual_anchor,
                audio_ref_rows=audio_anchor,
                sigmas_video=video_sigmas,
                sigmas_audio=audio_sigmas,
                device=self.device,
                imgvid_cond_noise_aug_for_inference=(MINIMAX_H3_IMGVID_COND_TIMESTEP),
                audio_cond_noise_aug_for_inference=(MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                on_step=lambda step, video, audio: progress.update(),
            )

        target_video = video_rows[branch.update_mask_dev]
        video_latent = minimax_h3_unpatchify_video_tokens(
            target_video,
            latent_shape=(
                latent_t,
                latent_h // 2,
                latent_w // 2,
                24,
            ),
            patch_size=(1, 2, 2),
        )
        target_audio = audio_rows[branch.audio_update_mask_dev]
        audio_latent = minimax_h3_unpack_audio_tokens(
            target_audio,
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    def decode(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_start = time.perf_counter()
        with current_omni_platform.create_autocast_context(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=True,
        ):
            video = self.video_vae.decode_latent(video_latent)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_video = time.perf_counter()
        video = video[..., :height, :width].contiguous()
        audio = self.audio_vae.decode_latent(audio_latent)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.info(
            "VAE decode split: video=%.2fs audio=%.2fs",
            t_video - t_start,
            time.perf_counter() - t_video,
        )
        return video, audio

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise ValueError("MiniMax H3 supports one request at a time")
        raw_prompt = request.prompts[0]
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
            multi_modal_data: dict[str, Any] = {}
        else:
            prompt = str(raw_prompt.get("prompt") or "")
            multi_modal_data = raw_prompt.get("multi_modal_data") or {}
        if not prompt:
            raise ValueError("MiniMax H3 requires a non-empty prompt")

        sampling = request.sampling_params
        extra = sampling.extra_args or {}
        task = self._resolve_task(extra.get("task"), multi_modal_data)

        images = _load_images(multi_modal_data.get("image"))
        audios = _load_audios(multi_modal_data.get("audio")) if task == "ref2va" else []
        raw_videos = multi_modal_data.get("video")
        if task == "fl2va" and not images:
            raise ValueError(f"{task} requires multi_modal_data.image")
        if task == "ref2va" and not images and raw_videos is None:
            raise ValueError("ref2va requires multi_modal_data.image or multi_modal_data.video")
        if task != "ref2va" and raw_videos is not None:
            raise ValueError(f"{task} does not accept a video condition")
        if task != "ref2va" and multi_modal_data.get("audio") is not None:
            raise ValueError(f"{task} does not accept an audio condition")
        if task == "t2va" and images:
            raise ValueError("t2va does not accept an image condition")
        if task == "ref2va":
            n_videos = len(raw_videos) if isinstance(raw_videos, (list, tuple)) else (1 if raw_videos is not None else 0)
            validate_ref_counts(len(images), n_videos, len(audios))

        shape_image = images[0] if images else None
        height, width, num_frames, latent_t, audio_t = self._resolve_shape(task, sampling, shape_image)
        prepared_images: list[Image.Image] = []
        if task == "fl2va":
            prepared_images = [img.resize((width, height), Image.Resampling.LANCZOS) for img in images]
        elif task == "ref2va":
            # ComfyUI ref_image_size parity (comfy_extras/nodes_minimax_h3.py):
            # both modes are aspect-preserving and shrink-only. "match" scales
            # references to the generation's pixel area (fast); "max" keeps up
            # to a 2048px short edge (stronger identity, slower). Default: match.
            ref_image_size = str(extra.get("ref_image_size", "match")).lower()
            if ref_image_size not in ("match", "max"):
                raise ValueError(f"ref_image_size must be 'match' or 'max', got {ref_image_size!r}")
            for img in images:
                img_w, img_h = img.size
                if img_w > 4 * img_h or img_h > 4 * img_w:
                    raise ValueError(f"reference image aspect ratio must be in [1:4, 4:1], got {img_w}x{img_h}")
                if ref_image_size == "match":
                    scale = min(1.0, math.sqrt((width * height) / (img_w * img_h)))
                else:
                    scale = min(1.0, MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(img_w, img_h))
                ref_width = max(
                    MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
                    round(img_w * scale / MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE) * MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
                )
                ref_height = max(
                    MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
                    round(img_h * scale / MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE) * MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
                )
                prepared_images.append(img.resize((ref_width, ref_height), Image.Resampling.LANCZOS))

        visual_condition = None
        visual_shape = None
        visual_shapes = None
        audio_condition = None
        ref_audio_t = None
        audio_lengths = None
        ref_blocks = None
        with tempfile.TemporaryDirectory(prefix="minimax_h3_ref2va_") as workdir:
            prepared_videos = None
            has_audio: list[bool] = []
            video_count = 0
            if raw_videos is not None:
                video_count = len(raw_videos) if isinstance(raw_videos, (list, tuple)) else 1
                prepared_videos = self._prepare_reference_videos(
                    raw_videos,
                    target_frame_count=num_frames,
                    workdir=workdir,
                )
                has_audio_tensor = torch.zeros(
                    video_count,
                    dtype=torch.long,
                    device=self.device,
                )
                _, rank, world_size = _dit_rank_world()
                if rank == 0:
                    has_audio_tensor = torch.tensor(
                        [int(item["input_has_audio"]) for item in prepared_videos or []],
                        dtype=torch.long,
                        device=self.device,
                    )
                if world_size > 1:
                    dist.broadcast(
                        has_audio_tensor,
                        src=0,
                        group=get_dit_group(),
                    )
                has_audio = [bool(value) for value in has_audio_tensor.tolist()]

            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                images=prepared_images,
                prepared_videos=prepared_videos,
                standalone_audio_count=len(audios),
            )

            visual_parts: list[torch.Tensor] = []
            image_ref_blocks: list[dict[str, Any]] = []
            image_shapes: list[tuple[int, int, int]] = []
            for prepared_image in prepared_images:
                visual_parts.append(self._encode_visual_condition(prepared_image))
                image_shapes.append((1, prepared_image.height // 16, prepared_image.width // 16))
                image_ref_blocks.append(
                    {
                        "kind": "image",
                        "latent_h": prepared_image.height // 16,
                        "latent_w": prepared_image.width // 16,
                    }
                )
            video_ref_blocks: list[dict[str, Any]] = []
            video_audio_condition = None
            video_audio_lengths: list[int] = []
            video_shapes: list[tuple[int, int, int]] = []
            if prepared_videos is not None or raw_videos is not None:
                video_visual, video_shapes = self._encode_video_conditions(
                    prepared_videos,
                    count=video_count,
                )
                visual_parts.append(video_visual)
                video_audio_condition, video_audio_lengths = self._encode_video_audio_conditions(
                    prepared_videos,
                    has_audio=has_audio,
                )
                audio_iterator = iter(video_audio_lengths)
                for shape, contributes_audio in zip(
                    video_shapes,
                    has_audio,
                    strict=True,
                ):
                    ref_audio = next(audio_iterator) if contributes_audio else 0
                    video_ref_blocks.append(
                        {
                            "kind": "video",
                            "ref_audio_t": ref_audio,
                            "latent_t": shape[0],
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
            standalone_audio_parts: list[torch.Tensor] = []
            standalone_audio_lengths: list[int] = []
            for audio in audios:
                encoded, encoded_t = self._encode_audio_condition(audio)
                standalone_audio_parts.append(encoded)
                standalone_audio_lengths.append(encoded_t)
            visual_condition = _cat_rows(visual_parts)
            # Noise-aug validates row counts against the shape list, so every
            # reference image and video must appear here in packed row order:
            # images first, then videos.
            if image_shapes or video_shapes:
                visual_shapes = image_shapes + video_shapes
                visual_shape = image_shapes[0] if image_shapes else None
            audio_condition = _cat_rows(
                ([video_audio_condition] if video_audio_condition is not None else []) + standalone_audio_parts
            )
            audio_lengths = list(video_audio_lengths) + standalone_audio_lengths
            if standalone_audio_lengths:
                ref_audio_t = standalone_audio_lengths[0]
            elif video_audio_lengths:
                ref_audio_t = video_audio_lengths[0]
            ref_blocks = (
                image_ref_blocks
                + video_ref_blocks
                + [{"kind": "audio", "ref_audio_t": length} for length in standalone_audio_lengths]
            ) or None

        seed = int(sampling.seed if sampling.seed is not None else 42)
        num_steps = int(sampling.num_inference_steps or 50)
        video_shift = float(extra.get("flow_shift", self.default_video_shift))
        audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
        video_latent, audio_latent = self.diffuse(
            task=task,
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            seed=seed,
            latent_t=latent_t,
            latent_h=height // 16,
            latent_w=width // 16,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            visual_condition=visual_condition,
            visual_condition_shape=visual_shape,
            audio_condition=audio_condition,
            ref_audio_t=ref_audio_t,
            ref_blocks=ref_blocks,
            visual_condition_shapes=visual_shapes,
            audio_condition_lengths=audio_lengths,
        )
        video, audio = self.decode(
            video_latent,
            audio_latent,
            height=height,
            width=width,
        )
        return DiffusionOutput(
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]
