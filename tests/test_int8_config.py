# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the comfy-kitchen INT8 quantization adapter."""

import torch

from vllm_omni.diffusion.models.minimax_h3.comfy_kitchen_int8 import (
    ComfyKitchenINT8Config,
    ComfyKitchenINT8LinearMethod,
    ComfyKitchenINT8ScaleParameter,
    _dtype_code,
)


def test_dtype_code() -> None:
    assert _dtype_code(torch.float32) == 0
    assert _dtype_code(torch.float16) == 1
    assert _dtype_code(torch.bfloat16) == 2


def test_config_name_and_capability() -> None:
    cfg = ComfyKitchenINT8Config()
    assert cfg.get_name() == "comfy_kitchen_int8"
    assert cfg.get_min_capability() == 80
    assert torch.bfloat16 in cfg.get_supported_act_dtypes()


def test_quant_method_returns_linear_method() -> None:
    cfg = ComfyKitchenINT8Config()
    method = cfg.get_quant_method(None, "")
    assert isinstance(method, ComfyKitchenINT8LinearMethod)


def test_scale_parameter_row_loader_copies_full_tensor() -> None:
    # BasevLLMParameter requires an initialized TP group, so exercise the
    # loader method on a plain tensor-shaped object.
    class FakeParam:
        def __init__(self, data: torch.Tensor) -> None:
            self.data = data

        def _assert_and_load(self, loaded_weight: torch.Tensor) -> None:
            assert self.data.shape == loaded_weight.shape
            self.data.copy_(loaded_weight)

    param = FakeParam(torch.empty(10, 1, dtype=torch.float32))
    ComfyKitchenINT8ScaleParameter.load_row_parallel_weight(param, torch.ones(10, 1))
    assert torch.all(param.data == 1.0)
