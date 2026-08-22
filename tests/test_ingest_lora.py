from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_INGEST = Path(__file__).resolve().parents[1] / "scripts" / "ingest_lora.py"
_SPEC = importlib.util.spec_from_file_location("ingest_lora", _INGEST)
assert _SPEC and _SPEC.loader
ingest_lora = importlib.util.module_from_spec(_SPEC)
sys.modules["ingest_lora"] = ingest_lora
_SPEC.loader.exec_module(ingest_lora)

FUSED_QKV_OUT = ingest_lora.FUSED_QKV_OUT
apply_qkv_layout = ingest_lora.apply_qkv_layout
ingest = ingest_lora.ingest
remap_key = ingest_lora.remap_key


def test_remap_diffusers_v0_1_keys():
    assert (
        remap_key("transformer_blocks.3.attn.to_q.lora_A.default.weight")
        == "blocks.3.attn.to_q.lora_A.weight"
    )
    assert (
        remap_key("transformer_blocks.3.attn.to_out.0.lora_B.default.weight")
        == "blocks.3.attn.out_proj.lora_B.weight"
    )
    assert (
        remap_key("transformer_blocks.3.ff.net.0.proj.lora_B.default.weight")
        == "blocks.3.mlp.fc1.lora_B.weight"
    )
    assert (
        remap_key("token_refiner.refiner_blocks.1.attn.to_k.lora_A.default.weight")
        == "token_refiner.blocks.1.attn.to_k.lora_A.weight"
    )


def test_missing_layout_is_required(tmp_path):
    with pytest.raises(SystemExit):
        ingest_lora.main(
            ["--src", str(tmp_path / "x.safetensors"), "--name", "turbo4", "--dest", str(tmp_path)]
        )


def test_shape_21504_does_not_auto_reorder():
    tensor = torch.arange(FUSED_QKV_OUT * 4, dtype=torch.float32).reshape(FUSED_QKV_OUT, 4)
    key = "blocks.0.attn.qkv_proj.lora_B.weight"
    out = apply_qkv_layout(key, tensor, "qkv")
    assert torch.equal(out, tensor)
    grouped = apply_qkv_layout(key, tensor, "grouped")
    assert grouped.shape == tensor.shape
    assert not torch.equal(grouped, tensor)


def test_grouped_rejected_for_split_projection():
    tensor = torch.ones(7168, 8)
    with pytest.raises(ValueError, match="split Diffusers"):
        apply_qkv_layout("blocks.0.attn.to_q.lora_B.weight", tensor, "grouped")


def test_forbidden_module_rejected(tmp_path):
    src = tmp_path / "bad.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(8, 16),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(32, 8),
            "blocks.0.adaln.linear.lora_A.default.weight": torch.ones(8, 16),
            "blocks.0.adaln.linear.lora_B.default.weight": torch.ones(16, 8),
        },
        str(src),
    )
    with pytest.raises(ValueError, match="forbidden target"):
        ingest(src=src, dest_root=tmp_path / "out", name="turbo4", qkv_layout="qkv")


def test_ingest_writes_peft_and_manifest(tmp_path):
    src = tmp_path / "src.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(8, 16),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(32, 8),
            "transformer_blocks.0.ff.net.2.lora_A.default.weight": torch.ones(8, 24),
            "transformer_blocks.0.ff.net.2.lora_B.default.weight": torch.ones(16, 8),
        },
        str(src),
    )
    dest = tmp_path / "loras"
    adapter = ingest(src=src, dest_root=dest, name="turbo4", qkv_layout="qkv")
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert config["r"] == 8
    assert config["lora_alpha"] == 8
    assert "to_q" in config["target_modules"]
    assert "fc2" in config["target_modules"]
    assert (adapter / "adapter_model.safetensors").is_file()
    manifest = (dest / "turbo4.sha256").read_text().splitlines()
    assert any(line.endswith("turbo4/adapter_model.safetensors") for line in manifest)


def test_ingest_skips_alpha_allows_mixed_rank_and_bakes_scale(tmp_path):
    src = tmp_path / "ref2v.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(24, 8),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones(48, 24),
            "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(3.0),
            "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": torch.ones(8, 16),
            "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": torch.ones(16, 8),
            "diffusion_model.blocks.0.mlp.fc2.alpha": torch.tensor(1.0),
        },
        str(src),
    )
    dest = tmp_path / "loras"
    adapter = ingest(src=src, dest_root=dest, name="ref2v", qkv_layout="qkv")
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert config["r"] == 24
    assert config["lora_alpha"] == 24
    assert "qkv_proj" in config["target_modules"]
    assert "fc2" in config["target_modules"]
    from safetensors import safe_open

    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        assert all(not k.endswith(".alpha") for k in keys)
        qkv_b = handle.get_tensor("blocks.0.attn.qkv_proj.lora_B.weight")
        fc2_b = handle.get_tensor("blocks.0.mlp.fc2.lora_B.weight")
    # alpha/r baked: 3/24=0.125, 1/8=0.125
    assert torch.allclose(qkv_b, torch.full_like(qkv_b, 0.125))
    assert torch.allclose(fc2_b, torch.full_like(fc2_b, 0.125))
