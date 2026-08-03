# Architecture and compatibility boundary

## Why a custom executor exists

The pinned vLLM-Omni image accepts `--distributed-executor-backend ray` at the
CLI layer but its diffusion executor factory rejects that backend. Its local
multiprocess path also assumes every global rank maps to a CUDA device on one
host. That makes global rank 1 select `cuda:1`, which does not exist on a
one-GPU DGX Spark.

The patch in `patches/enable-ray-diffusion-executor.patch` changes only the Ray
factory branch. All model-specific behavior remains in the pinned vLLM-Omni
and companion single-Spark compatibility layers.

## Control and data paths

Ray is the control plane. The head discovers the exact two configured node
addresses and creates one GPU actor on each with hard node affinity. Each actor
uses local rank 0 and local `cuda:0`, while the model keeps global ranks 0 and
1.

PyTorch distributed and NCCL are the tensor data plane. The worker shim keeps
the fabric head address instead of the upstream localhost rendezvous and
preserves the rest of vLLM-Omni's device and model-parallel initialization.
NCCL is explicitly configured for the selected RoCEv2 HCA, GID, and network
interface.

MiniMax H3's DiT uses Ulysses sequence parallel size 2. Both actors execute
every denoising request. Only rank 0 retains the result; tensors are moved to
CPU before Ray serializes the response. Rank 0 also owns the encoders, VAE, and
API response path, which explains the asymmetric memory footprint.

## Failure behavior

- Startup requires exactly two alive Ray nodes at the configured addresses.
- Worker initialization has a bounded timeout.
- A distributed RPC failure marks the executor unhealthy and returns a failed
  diffusion result instead of silently continuing on one rank.
- Ray actors have no automatic restart or task retry.
- Containers have no Docker restart policy.

These choices make failures visible and avoid accidentally presenting a
single-node fallback as a successful two-node run.

## Supported scope

The accepted scope is one synchronous, batch-size-one MiniMax H3 FL2VA request
on two one-GPU DGX Sparks using the exact pinned base stack. Concurrent serving,
other diffusion models, other GPU counts, other interconnects, and updated
vLLM-Omni internals require new validation.
