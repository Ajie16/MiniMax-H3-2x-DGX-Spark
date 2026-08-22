"""Fused QKV wrap must succeed even when vLLM `type is` would skip."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.layers.column_parallel_linear import (
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
)
from vllm_omni.diffusion.lora.layers.row_parallel_linear import DiffusionRowParallelLinearWithLoRA
from vllm_omni.config.lora import LoRAConfig

from h3_multinode.lora_wrap import (
    choose_diffusion_lora_wrapper,
    cutlass_channel_scale,
    dequant_cutlass_fp8,
    fuse_stacked_lora_into_cutlass_weight,
    lora_delta_cutlass_layout,
    packed_modules_list_for_layer,
    resolve_packed_sublora_names,
    write_cutlass_fp8_weight,
)


class _FakeQKV(QKVParallelLinear):
    """Subclass that `type(layer) is QKVParallelLinear` rejects."""


def test_fused_qkv_with_split_targets_is_replaced():
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    packed = packed_modules_list_for_layer(layer)
    assert packed == ["q", "k", "v"]
    chosen = choose_diffusion_lora_wrapper(layer, packed)
    assert chosen is DiffusionMergedQKVParallelLinearWithLoRA


def test_qkv_subclass_type_identity_miss_still_wraps():
    layer = _FakeQKV.__new__(_FakeQKV)
    assert type(layer) is not QKVParallelLinear
    assert isinstance(layer, QKVParallelLinear)
    cfg = LoRAConfig(max_lora_rank=128, max_loras=1, max_cpu_loras=1, fully_sharded_loras=False)
    skipped = DiffusionMergedQKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=layer,
        lora_config=cfg,
        packed_modules_list=["q", "k", "v"],
        model_config=None,
    )
    assert skipped is False
    chosen = choose_diffusion_lora_wrapper(layer, ["q", "k", "v"])
    assert chosen is DiffusionMergedQKVParallelLinearWithLoRA


def test_row_parallel_wraps_and_plain_object_does_not():
    row = RowParallelLinear.__new__(RowParallelLinear)
    assert choose_diffusion_lora_wrapper(row, []) is DiffusionRowParallelLinearWithLoRA
    assert choose_diffusion_lora_wrapper(object(), []) is None


def test_resolve_packed_sublora_names_for_split_qkv():
    mapping = {"qkv_proj": ["to_q", "to_k", "to_v"]}
    names = resolve_packed_sublora_names(
        "transformer.blocks.0.attn.qkv_proj",
        mapping,
        n_slices=3,
    )
    assert names == [
        "transformer.blocks.0.attn.to_q",
        "transformer.blocks.0.attn.to_k",
        "transformer.blocks.0.attn.to_v",
    ]
    assert resolve_packed_sublora_names("transformer.blocks.0.attn.qkv_proj", mapping, 2) is None


def test_fused_fc1_merged_column_wraps_as_two_slices():
    """H3 mlp.fc1 is MergedColumnParallelLinear [ffn, ffn], not a plain Column."""
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    packed = packed_modules_list_for_layer(layer)
    assert packed == ["0", "1"]
    assert choose_diffusion_lora_wrapper(layer, packed) is DiffusionMergedColumnParallelLinearWithLoRA


def test_diffusion_apply_adds_delta_after_quant_gemm():
    """Cutlass/online FP8 GEMM then LoRA add must change the tensor (not a no-op)."""

    class _Quant:
        def apply(self, layer, x, bias=None):  # noqa: ARG002
            return torch.zeros(x.shape[0], 6, dtype=x.dtype)

    wrapper = DiffusionBaseLinearLayerWithLoRA.__new__(DiffusionBaseLinearLayerWithLoRA)
    wrapper.base_layer = type("Base", (), {"quant_method": _Quant()})()
    wrapper.lora_config = None
    wrapper.tp_size = 1
    wrapper.output_slices = (3, 3)
    wrapper._diffusion_lora_active_slices = (True, True)
    r, inn = 2, 4
    wrapper.lora_a_stacked = [
        torch.zeros(1, 1, r, inn),
        torch.zeros(1, 1, r, inn),
    ]
    wrapper.lora_b_stacked = [
        torch.zeros(1, 1, 3, r),
        torch.zeros(1, 1, 3, r),
    ]
    wrapper.lora_a_stacked[0][0, 0].copy_(torch.eye(r, inn)[:r])
    wrapper.lora_b_stacked[0][0, 0].fill_(0.5)
    wrapper.lora_a_stacked[1][0, 0].copy_(torch.eye(r, inn)[:r])
    wrapper.lora_b_stacked[1][0, 0].fill_(-0.25)

    x = torch.ones(5, inn)
    y = DiffusionBaseLinearLayerWithLoRA.apply(wrapper, x)
    assert y.shape == (5, 6)
    assert float(y.abs().mean()) > 1e-5
    assert not torch.allclose(y, torch.zeros_like(y))


def test_lora_delta_matches_comfy_weight_merge_layout():
    """Comfy does W_linear += B @ A; Cutlass W is (in, out) = W_linear.T."""
    inn, out, rank = 4, 6, 2
    lora_a = torch.arange(rank * inn, dtype=torch.float32).reshape(rank, inn)
    lora_b = torch.arange(out * rank, dtype=torch.float32).reshape(out, rank) * 0.1
    comfy = lora_b @ lora_a
    cutlass = lora_delta_cutlass_layout(lora_a, lora_b)
    assert cutlass.shape == (inn, out)
    assert torch.allclose(cutlass, comfy.T)


def test_fuse_stacked_qkv_slices_onto_cutlass_weight():
    inn, rank = 4, 2
    slices = (3, 3, 3)
    weight = torch.zeros(inn, sum(slices), dtype=torch.float32)
    a_stacks, b_stacks = [], []
    for sl in slices:
        a = torch.zeros(1, 1, rank, inn)
        b = torch.zeros(1, 1, sl, rank)
        a[0, 0].copy_(torch.eye(rank, inn)[:rank])
        b[0, 0].fill_(0.5)
        a_stacks.append(a)
        b_stacks.append(b)
    fuse_stacked_lora_into_cutlass_weight(weight, a_stacks, b_stacks, slices, (True, True, False))
    assert float(weight[:, :6].abs().mean()) > 0
    assert torch.equal(weight[:, 6:], torch.zeros(inn, 3))


def test_cutlass_channel_scale_from_per_token_quant():
    n_out = 8
    per_token = torch.arange(n_out, dtype=torch.float32).reshape(n_out, 1) * 0.01
    scaled = cutlass_channel_scale(per_token, n_out)
    assert tuple(scaled.shape) == (n_out,)
    assert torch.equal(scaled, per_token.reshape(n_out))
    scalar = cutlass_channel_scale(torch.tensor([0.5]), n_out)
    assert tuple(scalar.shape) == (1,)
    try:
        cutlass_channel_scale(torch.ones(3), n_out)
    except ValueError as exc:
        assert "channel scale" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched scale length")


def test_dequant_cutlass_fp8_broadcasts_per_n_scale():
    weight = torch.ones(4, 6, dtype=torch.float32)
    per_n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = dequant_cutlass_fp8(weight, per_n)
    assert tuple(out.shape) == (4, 6)
    assert torch.allclose(out[0].float(), per_n)


def test_write_cutlass_fp8_weight_replaces_tensor_scale_with_channel():
    class _Layer:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(4, 6))
            self.weight_scale = torch.nn.Parameter(torch.ones(1))

    captured: dict[str, object] = {}

    def _replace(layer, name, value):
        captured[name] = value
        setattr(layer, name, torch.nn.Parameter(value.detach().clone(), requires_grad=False))

    layer = _Layer()
    new_w = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    new_s = torch.linspace(0.1, 0.6, 6)

    import vllm.model_executor.layers.quantization.utils as quant_utils

    orig_replace = quant_utils.replace_parameter
    quant_utils.replace_parameter = _replace
    try:
        write_cutlass_fp8_weight(layer, new_w, new_s)
    finally:
        quant_utils.replace_parameter = orig_replace
    assert torch.equal(layer.weight.data, new_w)
    assert "weight_scale" in captured
    assert tuple(layer.weight_scale.shape) == (6,)


def test_per_tensor_fp8_bake_erases_small_lora_delta_but_apply_preserves_it():
    """Regression: baking the LoRA into per-tensor FP8 weights drops the delta.

    Measured on the ref2v turbo adapter (2026-08-22): block-0 qkv/fc1 deltas
    have absmean ~1e-5 while the per-tensor FP8 quantization step is ~3e-2, so
    a fused requant preserves only ~0.35-0.4% of the delta (correlation with
    the intended delta ~0.0004), i.e. the adapter is effectively gone. The
    serving path therefore keeps activation-add (FP8 base + BF16 delta), which
    preserves the delta at ~99.9% like ComfyUI's BF16 merge.
    """
    torch.manual_seed(0)
    inn, out, rank = 8, 12, 2
    base = torch.randn(out, inn, dtype=torch.float32)
    lora_a = torch.randn(rank, inn, dtype=torch.float32) * 0.01
    lora_b = torch.randn(out, rank, dtype=torch.float32) * 0.01
    delta = lora_b @ lora_a
    assert delta.abs().max().item() < base.abs().max().item() * 1e-3

    def per_tensor_fp8(x):
        scale = x.abs().max() / 112.0
        return torch.clamp(torch.round(x / scale), -112, 112) * scale

    baked = per_tensor_fp8(base + delta) - per_tensor_fp8(base)
    corr_baked = torch.corrcoef(torch.stack([delta.flatten(), baked.flatten()]))[0, 1]
    assert baked.norm().item() / delta.norm().item() < 0.5
    assert float(corr_baked) < 0.2

    base_bf = base.to(torch.bfloat16).float()
    added = (base_bf + delta.to(torch.bfloat16)).float() - base_bf
    corr_added = torch.corrcoef(torch.stack([delta.flatten(), added.flatten()]))[0, 1]
    assert added.norm().item() / delta.norm().item() > 0.99
    assert float(corr_added) > 0.99
