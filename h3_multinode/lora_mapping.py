"""Packed-module mapping so split Diffusers LoRA keys hit fused H3 linears.

Phase 0A of the on-disk turbo4 adapter (`to_q`/`to_k`/`to_v`, not fused
`qkv_proj`) requires this. Install before DiffusionLoRAManager runs.
Keep the tuples in sync with companion MiniMaxH3Attention/MLP names.
"""

from __future__ import annotations

H3_STACKED_PARAMS_MAPPING = [
    (".qkv_proj", ".to_q", "q"),
    (".qkv_proj", ".to_k", "k"),
    (".qkv_proj", ".to_v", "v"),
]


def install_h3_lora_packed_mapping() -> None:
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    MiniMaxH3DiTModel.stacked_params_mapping = list(H3_STACKED_PARAMS_MAPPING)
