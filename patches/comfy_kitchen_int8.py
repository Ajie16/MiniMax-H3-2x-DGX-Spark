# SPDX-License-Identifier: Apache-2.0
# Modified 2026 for MiniMax H3 INT8 ConvRot inference via comfy-kitchen.
"""INT8 ConvRot linear method using comfy_kitchen's fused W8A8 kernel.

This quantization method is intentionally separate from vLLM's built-in FP8
quantization.  It loads pre-quantized Comfy-Org checkpoints that store
``.weight`` as ``torch.int8`` qdata and ``.weight_scale`` as a per-output-row
float32 scale, then calls ``torch.ops.comfy_kitchen.int8_linear`` at runtime.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import os
import sys

import torch

# Import comfy_kitchen to trigger backend (CUDA/eager/triton) op registration.
# Without this, torch.ops.comfy_kitchen.int8_linear is not available at runtime.
import comfy_kitchen  # noqa: F401
from comfy_kitchen.registry import registry

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    _ColumnvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs


def _dtype_code(dtype: torch.dtype) -> int:
    """Output-dtype code expected by torch.ops.comfy_kitchen.int8_linear."""
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"Unsupported INT8 output dtype: {dtype}")


class ComfyKitchenINT8Config(QuantizationConfig):
    """Quantization config that enables comfy-kitchen INT8 ConvRot GEMMs."""

    def __init__(self) -> None:
        pass

    def get_name(self) -> str:
        return "comfy_kitchen_int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ComfyKitchenINT8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase | None:
        del layer  # unused
        # Comfy-Org INT8 checkpoints use convrot_groupsize=256 for attention/MLP
        # weights and convrot_groupsize=64 for the DiT-block and final-layer
        # AdaLN projections (their input dimension 2688 is divisible by 64 but
        # not by 256).  Match any module name containing ``adaln_proj``.
        group_size = 64 if "adaln_proj" in prefix else 256
        if os.environ.get("H3_INT8_DEBUG") == "1":
            print(
                f"[INT8 DEBUG] get_quant_method prefix={prefix} group_size={group_size}",
                file=sys.stderr,
                flush=True,
            )
        return ComfyKitchenINT8LinearMethod(self, group_size=group_size, prefix=prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []


class ComfyKitchenINT8ScaleParameter(_ColumnvLLMParameter):
    """Per-output-row scale for comfy-kitchen INT8 weights.

    Inherits column-parallel slicing (output_dim=0) so QKV, MergedColumn, and
    ColumnParallel layers shard the scale correctly.  Row-parallel layers do
    not shard the scale along the input dim, so we override the row loader to
    copy the full per-rank scale as-is.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Do not set input_dim; both v1 and v2 row-parallel loaders skip
        # input-dim narrowing when it is absent and copy the full
        # [N_per_rank, 1] scale tensor.
        kwargs.pop("input_dim", None)
        super().__init__(**kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)


@register_weight_loader_v2_supported_method
class ComfyKitchenINT8LinearMethod(LinearMethodBase):
    """Linear method that runs INT8 ConvRot weights through comfy_kitchen."""

    def __init__(
        self,
        quant_config: ComfyKitchenINT8Config,
        group_size: int = 256,
        prefix: str = "",
    ) -> None:
        self.quant_config = quant_config
        self.group_size = group_size
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.pop("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # INT8 qdata: [N_per_rank, K_per_rank].  ModelWeightParameter carries
        # both column- and row-parallel loading logic, so it works for QKV,
        # MergedColumn, ColumnParallel, and RowParallel layers.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Per-output-row scale: [N_per_rank, 1].  The same weight_loader slices
        # along output_dim for column-parallel layers; for row-parallel layers
        # only the input dim is sharded, so the full per-rank scale is loaded.
        scale = ComfyKitchenINT8ScaleParameter(
            data=torch.empty(output_size_per_partition, 1, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # comfy-kitchen expects contiguous weight memory.
        if not layer.weight.is_contiguous():
            layer.weight.data = layer.weight.data.contiguous()
        if not layer.weight_scale.is_contiguous():
            layer.weight_scale.data = layer.weight_scale.data.contiguous()
        # Ensure scale stays float32; some loaders may cast it accidentally.
        if layer.weight_scale.dtype != torch.float32:
            layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = self.prefix or getattr(layer, "prefix", "<unknown>")

        # Defensive: the CUDA kernel assumes contiguous operands on the same
        # device. vLLM's process_weights_after_loading already handles this,
        # but replicate it cheaply here in case a preceding transform changed
        # layout.
        weight = layer.weight if layer.weight.is_contiguous() else layer.weight.contiguous()
        weight_scale = layer.weight_scale if layer.weight_scale.is_contiguous() else layer.weight_scale.contiguous()
        if weight.device != x.device:
            weight = weight.to(x.device)
        if weight_scale.device != x.device:
            weight_scale = weight_scale.to(x.device)

        # The Comfy-Org checkpoint stores convrot_groupsize=256 for most layers
        # and convrot_groupsize=64 for the AdaLN projections whose input dim
        # (2688) is not divisible by 256.  Fall back to a divisor of the actual
        # weight K dimension when the configured group size would not divide it.
        k_features = weight.shape[1]
        group_size = self.group_size
        if k_features % group_size != 0:
            for candidate in (256, 128, 64, 32):
                if k_features % candidate == 0:
                    group_size = candidate
                    break
            else:
                raise ValueError(
                    f"INT8 ConvRot features {k_features} not divisible by any "
                    f"supported group_size for {prefix}"
                )

        if os.environ.get("H3_INT8_DEBUG") == "1":
            print(
                f"[INT8 DEBUG] {prefix} x={tuple(x.shape)}:{x.dtype}:{x.device} "
                f"w={tuple(weight.shape)}:{weight.dtype}:{weight.device} "
                f"s={tuple(weight_scale.shape)}:{weight_scale.dtype}:{weight_scale.device} "
                f"bias={bias is not None} configured_group={self.group_size} "
                f"runtime_group={group_size}",
                file=sys.stderr,
                flush=True,
            )

        if os.environ.get("H3_INT8_EAGER") == "1":
            with registry.use_backend("eager"):
                return torch.ops.comfy_kitchen.int8_linear(
                    x.contiguous(),
                    weight,
                    weight_scale,
                    bias,
                    _dtype_code(x.dtype),
                    True,  # convrot
                    group_size,
                )

        return torch.ops.comfy_kitchen.int8_linear(
            x.contiguous(),
            weight,
            weight_scale,
            bias,
            _dtype_code(x.dtype),
            True,  # convrot
            group_size,
        )
