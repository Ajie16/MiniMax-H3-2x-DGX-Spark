from dataclasses import dataclass

import torch

from h3_multinode.worker import _move_tensors_to_cpu


@dataclass
class Output:
    output: object
    trajectory_timesteps: object = None
    trajectory_latents: object = None
    trajectory_log_probs: object = None


def test_nested_tensor_payloads_move_to_cpu():
    value = Output(output=(torch.ones(2), {"audio": [torch.zeros(3)]}))
    result = _move_tensors_to_cpu(value)
    assert result.output[0].device.type == "cpu"
    assert result.output[1]["audio"][0].device.type == "cpu"
