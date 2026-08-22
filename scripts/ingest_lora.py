#!/usr/bin/env python3
"""Convert a MiniMax H3 ComfyUI/Diffusers LoRA safetensors file into PEFT.

Keep-in-sync: _reorder_grouped_qkv_to_qkv is copied from
MiniMax-H3-DGX-Spark/patches/minimax_h3_transformer.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Allow `python scripts/ingest_lora.py` from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from h3_multinode.lora_catalog import (  # noqa: E402
    ADAPTER_NAME_RE,
    FORBIDDEN_TARGET_MODULES,
)

QKV_LAYOUTS = ("grouped", "qkv", "identity")
NUM_QUERY_GROUPS = 56
HEADS_PER_GROUP = 1
HEAD_DIM = 128
FUSED_QKV_OUT = NUM_QUERY_GROUPS * (HEADS_PER_GROUP + 2) * HEAD_DIM  # 21504

# Diffusers/LightX2V v0.1 keys → H3 serving module names.
_KEY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"^transformer_blocks\.(\d+)\.attn\.to_(q|k|v)\.lora_(A|B)\.default\.weight$"
        ),
        r"blocks.\1.attn.to_\2.lora_\3.weight",
    ),
    (
        re.compile(
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.lora_(A|B)\.default\.weight$"
        ),
        r"blocks.\1.attn.out_proj.lora_\2.weight",
    ),
    (
        re.compile(
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.lora_(A|B)\.default\.weight$"
        ),
        r"blocks.\1.mlp.fc1.lora_\2.weight",
    ),
    (
        re.compile(
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.lora_(A|B)\.default\.weight$"
        ),
        r"blocks.\1.mlp.fc2.lora_\2.weight",
    ),
    (
        re.compile(
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_(q|k|v)\.lora_(A|B)\.default\.weight$"
        ),
        r"token_refiner.blocks.\1.attn.to_\2.lora_\3.weight",
    ),
    (
        re.compile(
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.lora_(A|B)\.default\.weight$"
        ),
        r"token_refiner.blocks.\1.attn.out_proj.lora_\2.weight",
    ),
    (
        re.compile(
            r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj\.lora_(A|B)\.default\.weight$"
        ),
        r"token_refiner.blocks.\1.mlp.fc1.lora_\2.weight",
    ),
    (
        re.compile(
            r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2\.lora_(A|B)\.default\.weight$"
        ),
        r"token_refiner.blocks.\1.mlp.fc2.lora_\2.weight",
    ),
    # Already-PEFT fused keys (optional).
    (
        re.compile(r"^diffusion_model\.(.+\.lora_[AB](?:\.default)?\.weight)$"),
        r"\1",
    ),
)


def _reorder_grouped_qkv_to_qkv(
    weight,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
):
    """Copied from companion minimax_h3_transformer.py. Keep in sync."""
    import torch

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def remap_key(src_key: str) -> str:
    key = src_key
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model.") :]
    for pattern, repl in _KEY_RULES:
        if pattern.match(key):
            return pattern.sub(repl, key)
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")
    return key


def _module_suffix(peft_key: str) -> str:
    # blocks.0.attn.to_q.lora_A.weight → to_q
    parts = peft_key.split(".")
    if len(parts) >= 3 and parts[-1] == "weight" and parts[-2].startswith("lora_"):
        return parts[-3]
    return parts[-1]


def apply_qkv_layout(peft_key: str, tensor, layout: str):
    is_qkv_b = peft_key.endswith(".qkv_proj.lora_B.weight")
    if layout == "identity":
        return tensor
    if layout == "qkv":
        return tensor
    if layout == "grouped":
        if not is_qkv_b:
            raise ValueError(
                "--qkv-layout=grouped only applies to fused *.qkv_proj lora_B; "
                f"{peft_key} is a split Diffusers projection. Use qkv or identity."
            )
        if tensor.shape[0] != FUSED_QKV_OUT:
            raise ValueError(
                f"--qkv-layout=grouped expected fused B rows {FUSED_QKV_OUT}, "
                f"got {tuple(tensor.shape)} for {peft_key}"
            )
        return _reorder_grouped_qkv_to_qkv(
            tensor,
            num_query_groups=NUM_QUERY_GROUPS,
            heads_per_group=HEADS_PER_GROUP,
            head_dim=HEAD_DIM,
        )
    raise ValueError(f"invalid --qkv-layout {layout!r}")


def ingest(
    *,
    src: Path,
    dest_root: Path,
    name: str,
    qkv_layout: str,
    lora_alpha: int | None = None,
) -> Path:
    if qkv_layout not in QKV_LAYOUTS:
        raise ValueError(f"--qkv-layout must be one of {QKV_LAYOUTS}")
    if not ADAPTER_NAME_RE.match(name):
        raise ValueError(f"invalid adapter name {name!r}")
    if not src.is_file():
        raise FileNotFoundError(src)

    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_dir = dest_root / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = dest_root / f"{name}.ingest-log.txt"
    converted: dict[str, object] = {}
    ranks: set[int] = set()
    targets: set[str] = set()
    log_lines = [
        f"src={src}",
        f"name={name}",
        f"qkv_layout={qkv_layout}",
        "keys:",
    ]

    alphas: dict[str, float] = {}
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        for src_key in handle.keys():
            tensor = handle.get_tensor(src_key)
            peft_key = remap_key(src_key)
            if peft_key.endswith(".alpha"):
                if tensor.numel() != 1:
                    raise ValueError(f"alpha tensor must be scalar, got {peft_key} {tuple(tensor.shape)}")
                alphas[peft_key[: -len(".alpha")]] = float(tensor.reshape(-1)[0].item())
                log_lines.append(f"  skip {src_key} alpha={alphas[peft_key[: -len('.alpha')]]}")
                continue
            if not (
                peft_key.endswith(".lora_A.weight") or peft_key.endswith(".lora_B.weight")
            ):
                log_lines.append(f"  skip {src_key} -> {peft_key}")
                continue
            suffix = _module_suffix(peft_key)
            if suffix in FORBIDDEN_TARGET_MODULES:
                raise ValueError(
                    f"refusing forbidden target {suffix} from key {src_key} → {peft_key}"
                )
            if peft_key.endswith(".lora_B.weight"):
                tensor = apply_qkv_layout(peft_key, tensor, qkv_layout)
            if peft_key.endswith(".lora_A.weight"):
                ranks.add(int(tensor.shape[0]))
            targets.add(suffix)
            converted[peft_key] = tensor.contiguous()
            log_lines.append(
                f"  {src_key} {tuple(handle.get_slice(src_key).get_shape())} -> {peft_key} {tuple(tensor.shape)}"
            )

    if not converted:
        raise ValueError(f"no tensors in {src}")
    if not ranks:
        raise ValueError("no LoRA A ranks found")
    rank = max(ranks)
    a_ranks: dict[str, int] = {}
    for peft_key, tensor in converted.items():
        if peft_key.endswith(".lora_A.weight"):
            a_ranks[peft_key[: -len(".lora_A.weight")]] = int(tensor.shape[0])
    baked = 0
    for peft_key, tensor in list(converted.items()):
        if not peft_key.endswith(".lora_B.weight"):
            continue
        module = peft_key[: -len(".lora_B.weight")]
        layer_rank = a_ranks.get(module)
        layer_alpha = alphas.get(module)
        if layer_rank is None or layer_alpha is None or layer_rank <= 0:
            continue
        scale = float(layer_alpha) / float(layer_rank)
        if abs(scale - 1.0) < 1e-8:
            continue
        converted[peft_key] = (tensor * scale).contiguous()
        baked += 1
        log_lines.append(
            f"  bake {peft_key} alpha={layer_alpha} r={layer_rank} scale={scale:.8f}"
        )
    # After per-layer alpha/r is baked into B, PEFT scale must be 1 unless overridden.
    alpha = int(lora_alpha) if lora_alpha is not None else rank
    weight_path = adapter_dir / "adapter_model.safetensors"
    save_file(converted, str(weight_path))
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(targets),
        "lora_dropout": 0.0,
        "bias": "none",
        "base_model_name_or_path": "MiniMax-H3/FL2VA",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest_path = dest_root / f"{name}.sha256"
    rel_files = [
        f"{name}/adapter_config.json",
        f"{name}/adapter_model.safetensors",
    ]
    lines = []
    for rel in rel_files:
        digest = hashlib.sha256((dest_root / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    manifest_path.write_text("\n".join(lines) + "\n")
    log_lines.append(
        f"r={rank} ranks={sorted(ranks)} baked_alpha={baked} lora_alpha={alpha} "
        f"target_modules={sorted(targets)}"
    )
    log_path.write_text("\n".join(log_lines) + "\n")
    return adapter_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--qkv-layout", required=True, choices=QKV_LAYOUTS)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--lora-alpha", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        dest = ingest(
            src=args.src,
            dest_root=args.dest,
            name=args.name,
            qkv_layout=args.qkv_layout,
            lora_alpha=args.lora_alpha,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {args.name} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
