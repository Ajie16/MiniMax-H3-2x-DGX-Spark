"""Fail unless the derived image matches the accepted runtime package set."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch


EXPECTED_DISTRIBUTIONS = {
    "diffusers": "0.38.0",
    "flashinfer-python": "0.6.14",
    "ray": "2.56.1",
    "transformers": "5.14.1",
    "vllm": "0.26.0",
    "vllm-omni": "0.1.dev2381+g310b4b477",
}


def main() -> None:
    actual = {
        name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    mismatches = {
        name: (EXPECTED_DISTRIBUTIONS[name], actual[name])
        for name in EXPECTED_DISTRIBUTIONS
        if actual[name] != EXPECTED_DISTRIBUTIONS[name]
    }
    fixed = {
        "architecture": ("aarch64", platform.machine()),
        "python": ("3.12.13", sys.version.split()[0]),
        "torch": ("2.11.0+cu130", torch.__version__),
        "torch_cuda": ("13.0", torch.version.cuda),
    }
    mismatches.update(
        {name: values for name, values in fixed.items() if values[0] != values[1]}
    )
    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected}, got {observed}"
            for name, (expected, observed) in sorted(mismatches.items())
        )
        raise SystemExit(f"runtime provenance check failed: {details}")

    print("runtime provenance passed")
    for name, version in sorted(actual.items()):
        print(f"{name}={version}")
    for name, (_, value) in fixed.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
