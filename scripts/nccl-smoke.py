#!/usr/bin/env python3
"""Verify one Ray GPU actor per Spark and an NCCL all-reduce over RoCE."""

from __future__ import annotations

import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


HEAD_IP = os.environ.get("H3_HEAD_IP") or os.environ["HEAD_IP"]
WORKER_IP = os.environ.get("H3_WORKER_IP") or os.environ["WORKER_IP"]
MASTER_PORT = os.environ.get("H3_NCCL_SMOKE_PORT", "29501")
IFACE = os.environ.get("NCCL_SOCKET_IFNAME", "enp1s0f1np1")
HCA = os.environ.get("NCCL_IB_HCA", "rocep1s0f1")
SHARED_GID = os.environ.get("NCCL_IB_GID_INDEX", "3")
HEAD_GID = os.environ.get("H3_HEAD_GID_INDEX", SHARED_GID)
WORKER_GID = os.environ.get("H3_WORKER_GID_INDEX", SHARED_GID)


@ray.remote(num_gpus=1)
class NcclRank:
    def run(self, rank: int, gid: str) -> dict[str, object]:
        import socket

        import torch
        import torch.distributed as dist

        os.environ.update(
            {
                "MASTER_ADDR": HEAD_IP,
                "MASTER_PORT": MASTER_PORT,
                "RANK": str(rank),
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
                "NCCL_IB_HCA": HCA,
                "NCCL_IB_GID_INDEX": gid,
                "NCCL_IB_ADDR_FAMILY": "AF_INET",
                "NCCL_IB_ROCE_VERSION_NUM": "2",
                "NCCL_SOCKET_IFNAME": IFACE,
                "GLOO_SOCKET_IFNAME": IFACE,
                "NCCL_CUMEM_ENABLE": "0",
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_DEBUG": "WARN",
            }
        )
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=2)
        value = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        result = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "backend": dist.get_backend(),
            "sum": value.item(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        dist.destroy_process_group()
        return result


def main() -> None:
    ray.init(
        address=os.environ.get("H3_RAY_ADDRESS", f"{HEAD_IP}:6379"),
        _node_ip_address=HEAD_IP,
    )
    nodes = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive")
    }
    actors = [
        NcclRank.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip], soft=False)
        ).remote()
        for ip in (HEAD_IP, WORKER_IP)
    ]
    try:
        gids = (HEAD_GID, WORKER_GID)
        results = ray.get(
            [actor.run.remote(rank, gids[rank]) for rank, actor in enumerate(actors)],
            timeout=120,
        )
    finally:
        for actor in actors:
            ray.kill(actor)
    assert sorted(item["rank"] for item in results) == [0, 1]
    assert all(item["sum"] == 3.0 for item in results)
    assert len({item["hostname"] for item in results}) == 2
    print(results)


if __name__ == "__main__":
    main()
