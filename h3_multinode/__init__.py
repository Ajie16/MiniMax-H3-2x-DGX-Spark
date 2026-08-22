"""Cross-node diffusion execution for MiniMax H3 on two DGX Sparks."""

from __future__ import annotations

from typing import Any

__all__ = ["RayDiffusionExecutor"]


def __getattr__(name: str) -> Any:
    if name == "RayDiffusionExecutor":
        from .executor import RayDiffusionExecutor

        return RayDiffusionExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
