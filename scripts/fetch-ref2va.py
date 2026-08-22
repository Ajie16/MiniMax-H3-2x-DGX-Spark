#!/usr/bin/env python3
"""Populate models/Ref2VA: hardlink identical FL2VA files, download the rest."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


REPO_ID = "MiniMaxAI/MiniMax-H3"
REMOTE_PREFIX = "Ref2VA/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl2va", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args(argv)

    fl2va = args.fl2va.resolve()
    dest = args.dest.resolve()
    if not (fl2va / "model_index.json").is_file():
        print(f"error: FL2VA model_index missing at {fl2va}", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    remote_files = [
        item
        for item in api.list_repo_tree(REPO_ID, path_in_repo="Ref2VA", recursive=True, expand=True)
        if getattr(item, "size", None)
    ]
    linked = 0
    skipped = 0
    missing: list[str] = []
    for item in remote_files:
        rel = item.path[len(REMOTE_PREFIX) :] if item.path.startswith(REMOTE_PREFIX) else item.path
        target = dest / rel
        remote_sha = item.lfs.sha256 if item.lfs is not None else None
        source = fl2va / rel
        if source.is_file() and remote_sha:
            local_sha = sha256_file(source)
            if local_sha == remote_sha:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                os.link(source, target)
                linked += 1
                print(f"link {rel}")
                continue
        if target.is_file() and remote_sha and sha256_file(target) == remote_sha:
            skipped += 1
            print(f"have {rel}")
            continue
        missing.append(item.path)
        print(f"need {rel} size={item.size}")

    print(f"linked={linked} have={skipped} need={len(missing)}")
    if not missing:
        print(f"ref2va ready at {dest}")
        return 0
    if not args.download_missing:
        print("pass --download-missing to fetch remaining files", file=sys.stderr)
        return 2

    for remote_path in missing:
        rel = remote_path[len(REMOTE_PREFIX) :]
        print(f"download {rel}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=str(dest.parent),
        )
        print(f"wrote {rel}", flush=True)
    print(f"ref2va ready at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
