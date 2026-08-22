#!/usr/bin/env python3
"""Download MiniMax H3 Ref2VA via ModelScope (no proxy).

1) MiniMax/MiniMax-H3 Ref2VA transformer (BF16 serving; vLLM online FP8)
   TE/VAE are hardlinked from the existing FL2VA tree when SHA-256 matches.
Comfy-Org files are not downloaded.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Domestic CDN: never go through the local proxy.
for key in list(os.environ):
    if "proxy" in key.lower():
        os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from modelscope.hub.file_download import model_file_download

ROOT = Path("/home/xujie/workspace/dgx-spark-minimax-h3")
FL2VA = ROOT / "models" / "FL2VA"
REF2VA = ROOT / "models" / "Ref2VA"
OFFICIAL = "MiniMax/MiniMax-H3"

TRANSFORMER_SHARDS = [
    f"Ref2VA/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)
] + ["Ref2VA/transformer/model.safetensors.index.json"]

def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "K", "M", "G"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}T"


def download_one(model_id: str, rel: str, dest_file: Path) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.is_file() and dest_file.stat().st_size > 0:
        print(f"[skip] {dest_file} ({human(dest_file.stat().st_size)})", flush=True)
        return
    print(f"[dl] {model_id} :: {rel}", flush=True)
    print(f"     -> {dest_file}", flush=True)
    path = Path(
        model_file_download(model_id=model_id, file_path=rel, local_dir=str(dest_file.parent))
    ).resolve()
    dest = dest_file.resolve()
    if path != dest:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.move(str(path), str(dest))
    print(f"[ok] {dest} ({human(dest.stat().st_size)})", flush=True)


def main() -> int:
    REF2VA.mkdir(parents=True, exist_ok=True)
    print("=== MiniMax/MiniMax-H3 Ref2VA transformer (serving BF16 -> online FP8) ===", flush=True)
    for rel in TRANSFORMER_SHARDS:
        dest = REF2VA / rel[len("Ref2VA/") :]
        download_one(OFFICIAL, rel, dest)

    print("=== verify transformer shards ===", flush=True)
    missing = []
    for rel in TRANSFORMER_SHARDS:
        dest = REF2VA / rel[len("Ref2VA/") :]
        if not dest.is_file() or dest.stat().st_size <= 0:
            missing.append(str(dest))
            print(f"  MISSING {dest}", flush=True)
        else:
            print(f"  OK {dest} ({human(dest.stat().st_size)})", flush=True)
    if missing:
        return 1
    print("SUCCESS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
