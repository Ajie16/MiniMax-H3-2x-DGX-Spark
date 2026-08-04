# Reproducibility and image provenance

The accepted two-Spark image is a local derivative, not a standalone image
published by this repository. Its build chain is:

1. Upstream ARM64 image
   `vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e`.
2. Companion single-Spark repository commit
   `8bd7628dbdb51a0ea00c301ddcb1a098874870e4`, which produces the local tag
   `minimax-h3-dgx-spark:sm121-fp8`.
3. This repository's Dockerfile, which installs Ray and adds the cross-host
   executor to produce `minimax-h3-2x-dgx-spark:experimental`.

The local single-Spark tag is only a convenient reference to the image built
from step 2. It is not a registry digest and must not be treated as one. The
accepted local image ID is
`sha256:2383642e221530d3dc26a8f8632c37e00470b051979f0845c2ec0ff9513e04b2`.

`scripts/build-image.sh` fails closed unless that exact local image ID exists,
the accepted upstream digest exists locally, and the upstream layers are the
prefix of the local base image's layers. It also labels the resulting image
with the upstream digest, companion commit, and accepted local base ID. This
does not prove the safety of any image; it prevents an unnoticed tag change
from being presented as the measured stack.

The derived image is then checked against these measured versions:

| Component | Accepted value |
|---|---|
| Architecture | `aarch64` |
| Python | `3.12.13` |
| CUDA reported by PyTorch | `13.0` |
| PyTorch | `2.11.0+cu130` |
| vLLM | `0.26.0` |
| vLLM-Omni | `0.1.dev2381+g310b4b477` |
| Ray | `2.56.1` |
| Transformers | `5.14.1` |
| Diffusers | `0.38.0` |
| FlashInfer Python | `0.6.14` |

Run the documented build script rather than invoking `docker build` directly.
If any digest, commit, image ID, or runtime version changes, treat it as a new
stack and repeat the complete two-node and media acceptance flow before
updating this record.
