"""Fused-projection LoRA wrap policy for MiniMax H3.

vLLM's diffusion `from_layer_diffusion` defers to `can_replace_layer`, which
uses `type(source) is QKVParallelLinear`. A subclass (FP8 / PluggableLayer /
Cutlass) is a silent skip: PEFT still reports 312 keys, `_lora_modules` stays
empty, and 4-step turbo looks like an undenoised smear.

The shipped path is **activation-add**: the online FP8 base layer runs
unchanged and the BF16 LoRA delta is added by the wrapper's ``apply()`` after
the quantized GEMM. Baking the delta into the FP8 weight was measured on the
ref2v turbo adapter (2026-08-22) to preserve only ~0.35-0.4% of the delta
(per-tensor requant step ~3e-2 vs delta absmean ~1e-5, correlation ~0), so
the FP8-bake helpers below are diagnostic/reference only and are never called
by serving.
"""

from __future__ import annotations

from typing import Any


def packed_modules_list_for_layer(module: Any) -> list[str]:
    """Packed slice names for a fused linear. Uses isinstance, not type identity."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

    if isinstance(module, QKVParallelLinear):
        return ["q", "k", "v"]
    if isinstance(module, MergedColumnParallelLinear):
        return ["0", "1"]
    return []


def choose_diffusion_lora_wrapper(
    module: Any,
    packed_modules_list: list[str] | None = None,
) -> type | None:
    """Return the diffusion LoRA wrapper class for `module`, or None.

    QKV must be checked before ColumnParallelLinear because QKV subclasses it.
    A `type is` miss on a QKV subclass must still wrap (3-slice MergedQKV).
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm_omni.diffusion.lora.layers.column_parallel_linear import (
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
    )
    from vllm_omni.diffusion.lora.layers.row_parallel_linear import DiffusionRowParallelLinearWithLoRA

    packed = packed_modules_list if packed_modules_list is not None else packed_modules_list_for_layer(module)
    if isinstance(module, QKVParallelLinear):
        if len(packed) == 3:
            return DiffusionMergedQKVParallelLinearWithLoRA
        if len(packed) == 1:
            return DiffusionQKVParallelLinearWithLoRA
        return None
    if isinstance(module, MergedColumnParallelLinear) and len(packed) == 2:
        return DiffusionMergedColumnParallelLinearWithLoRA
    if isinstance(module, RowParallelLinear):
        return DiffusionRowParallelLinearWithLoRA
    if isinstance(module, ColumnParallelLinear) and len(packed) <= 1:
        return DiffusionColumnParallelLinearWithLoRA
    return None


def resolve_packed_sublora_names(
    full_module_name: str,
    packed_modules_mapping: dict[str, list[str]],
    n_slices: int,
) -> list[str] | None:
    """Map fused `...qkv_proj` to split `...to_q` / `...to_k` / `...to_v` names."""
    prefix, _, packed_suffix = full_module_name.rpartition(".")
    sub_suffixes = packed_modules_mapping.get(packed_suffix)
    if not sub_suffixes or len(sub_suffixes) != n_slices:
        return None
    return [f"{prefix}.{sub}" if prefix else sub for sub in sub_suffixes]


def lora_delta_cutlass_layout(lora_a, lora_b):
    """Diagnostic helper; not used by the serving path (see module docstring).

    Comfy merges `W += B @ A` on nn.Linear (out, in).
    Cutlass online FP8 stores `weight` as (in, out) = W.T, so the same delta is
    `(B @ A).T`. `lora_a` is (rank, in), `lora_b` is (out, rank).
    """
    return (lora_b @ lora_a).transpose(0, 1).contiguous()


def dequant_cutlass_fp8(weight, weight_scale):
    """Diagnostic helper; not used by the serving path (see module docstring)."""
    import torch

    dequant = weight.to(torch.bfloat16)
    if weight_scale is None:
        return dequant
    scale = weight_scale.to(dequant.dtype)
    if scale.numel() == 1:
        return dequant * scale
    flat = scale.reshape(-1)
    if flat.numel() == dequant.shape[1]:
        return dequant * flat.reshape(1, -1)
    if flat.numel() == dequant.shape[0]:
        return dequant * flat.reshape(-1, 1)
    return dequant * scale


def cutlass_channel_scale(quant_scale, n_out: int):
    """Diagnostic helper; not used by the serving path (see module docstring)."""
    """Map `scaled_fp8_quant` scales onto Cutlass N (output) channels.

    Per-token dynamic quant of nn.Linear `(out, in)` yields `(out, 1)`. Cutlass
    stores `weight` as `(in, out)`, so the matching scale is length `out`.
    """
    if quant_scale is None:
        return None
    flat = quant_scale.reshape(-1).contiguous()
    if flat.numel() == 1:
        return flat.reshape(1)
    if int(flat.numel()) != int(n_out):
        raise ValueError(
            f"FP8 channel scale length {int(flat.numel())} != Cutlass N {int(n_out)}"
        )
    return flat.reshape(int(n_out)).contiguous()


def write_cutlass_fp8_weight(layer: Any, weight_in_out, scale) -> None:
    """Diagnostic helper; not used by the serving path (see module docstring)."""
    """Copy fused FP8 weights, replacing `weight_scale` when its shape changes."""
    import torch
    from vllm.model_executor.layers.quantization.utils import replace_parameter

    w = weight_in_out.detach()
    current_w = getattr(layer, "weight", None)
    if current_w is not None and tuple(current_w.shape) == tuple(w.shape):
        with torch.no_grad():
            current_w.data.copy_(w)
    else:
        replace_parameter(layer, "weight", w)

    if scale is None:
        return
    s = scale.detach().contiguous()
    current_s = getattr(layer, "weight_scale", None)
    if current_s is None:
        replace_parameter(layer, "weight_scale", s)
        return
    if tuple(current_s.shape) == tuple(s.shape) and current_s.numel() == s.numel():
        with torch.no_grad():
            current_s.data.copy_(s.reshape(current_s.shape))
        return
    replace_parameter(layer, "weight_scale", s)


def fuse_stacked_lora_into_cutlass_weight(
    weight_in_out,
    lora_a_stacked,
    lora_b_stacked,
    output_slices,
    active_slices=None,
):
    """Diagnostic helper; not used by the serving path (see module docstring)."""
    offset = 0
    for slice_idx, slice_size in enumerate(output_slices):
        if active_slices is not None and slice_idx < len(active_slices) and not active_slices[slice_idx]:
            offset += int(slice_size)
            continue
        if slice_idx >= len(lora_a_stacked) or slice_idx >= len(lora_b_stacked):
            offset += int(slice_size)
            continue
        lora_a = lora_a_stacked[slice_idx][0, 0]
        lora_b = lora_b_stacked[slice_idx][0, 0]
        if lora_a.numel() == 0 or lora_b.numel() == 0:
            offset += int(slice_size)
            continue
        delta = lora_delta_cutlass_layout(lora_a, lora_b)
        in_take = min(delta.shape[0], weight_in_out.shape[0])
        out_take = min(delta.shape[1], int(slice_size), weight_in_out.shape[1] - offset)
        if in_take <= 0 or out_take <= 0:
            offset += int(slice_size)
            continue
        weight_in_out[:in_take, offset : offset + out_take].add_(delta[:in_take, :out_take].to(weight_in_out.dtype))
        offset += int(slice_size)
    return weight_in_out


def wrap_diffusion_linear(module: Any, lora_config: Any, max_loras: int = 1) -> Any:
    """Replace a fused linear with the matching LoRA wrapper, or return `module`.

    No FP8 weight baking: ``set_lora`` / ``reset_lora`` only update the BF16
    LoRA buffers, and ``apply()`` adds the delta after the online-FP8 base
    GEMM. This keeps the base weight byte-identical across adapter switches
    (no ~20 GiB snapshot and no restore) and preserves the delta in BF16.
    """
    wrapper_cls = choose_diffusion_lora_wrapper(module)
    if wrapper_cls is None:
        return module
    instance = wrapper_cls(module)
    instance.create_lora_weights(max_loras, lora_config, None)
    return instance
