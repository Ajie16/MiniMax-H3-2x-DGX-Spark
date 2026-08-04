# MiniMax H3 on two DGX Sparks

[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Platform: 2x DGX Spark](https://img.shields.io/badge/platform-2x%20DGX%20Spark-76B900)](#measured-topology)
[![Architecture: ARM64](https://img.shields.io/badge/architecture-ARM64-informational)](#measured-topology)
[![Interconnect: RoCEv2](https://img.shields.io/badge/interconnect-RoCEv2-76B900)](#how-one-video-spans-both-machines)

One MiniMax H3 FL2VA video, generated cooperatively by two NVIDIA DGX Sparks.

This is a separate, experimental extension of the
[single-Spark recipe](https://github.com/joeynyc/MiniMax-H3-DGX-Spark). It
adds the network-capable diffusion executor that the pinned vLLM-Omni build
does not provide. Ray places one rank on each Spark; NCCL carries the tensor
collectives over the dedicated RoCEv2 link.

> [!IMPORTANT]
> MiniMax H3 is **not** licensed under this repository's Apache-2.0 license.
> Its Community License currently excludes the United States, European Union,
> United Kingdom, and Republic of Korea, and restricts use and display of
> outputs outside its applicable territory. Read [MODEL-LICENSE.md](MODEL-LICENSE.md)
> and obtain any authorization you need from MiniMax before downloading,
> running, or displaying model output. This repository contains no model
> weights and no generated media.

## Measured topology

| Role | Host | Fabric address | Distributed rank |
|---|---|---:|---:|
| Spark 1 / head/API | `spark-head` example | private fabric address | 0 |
| Spark 2 / peer | `spark-peer` example | private fabric address | 1 |

The DiT uses two-way Ulysses sequence parallelism. Both GPUs contribute to the
same denoising trajectory; rank 0 also owns the encoders, VAE, and final API
response.

## Measured result

Verified on the two-Spark lab topology above on August 3, 2026. Two consecutive
fixed-seed T2VA requests returned fully decodable H.264/AAC MP4 files in 68.783
and 64.888 seconds. Both GPUs sustained about 96% utilization during denoising,
and NCCL reported two ranks on two nodes using `NET/IB`.

The comparable single-Spark request took 154.956 seconds, so the observed
client-side speedup was about 2.3x. This is a two-machine lab result, not a
vendor benchmark or an upstream support guarantee. See
[the acceptance record](docs/RESULTS.md) for exact settings, hashes, and limits.

The current verified fast profile uses cuDNN diffusion attention and regional
`torch.compile`. After one compile warm-up, two final acceptance requests took
47.101 and 50.276 seconds. Their 48.689-second average is 24.2% faster than the
64.226-second warm Torch-SDPA/eager baseline. FlashAttention 4 was tested and
rejected after an SM121 CuTe/CUTLASS JIT failure before its first denoising
step; it is not the default.

The same optimization also passed the exact earlier 1344x768, 24 fps,
50-step, four-second quality workload. End-to-end time fell from 2,281.532
seconds to 1,353.506 seconds: 40.7% lower latency, or 1.69x the former
generation rate. The 107-frame H.264/AAC result fully decoded, retained exactly
five adults throughout sampled-frame inspection, and measured 0.9134 video
SSIM against the earlier same-seed output. See the acceptance record for the
quality and bias observations; this is one matched test, not a universal
quality or performance claim.

An optional balanced Cache-DiT profile was then measured against that exact
full-compute baseline. At 1344x768 and 50 steps it completed in 608.991 seconds
(10m 8.991s), 55.0% lower latency and 2.22x the generation rate of the
1,353.506-second no-cache run. The same-seed cached output measured 0.888 SSIM
and 27.04 dB average PSNR against full compute, with effectively unchanged
audio. It is a visually inspected speed/quality tradeoff, not a lossless mode.

## How one video spans both machines

```text
client -> Spark 1 API -> Ray executor
                         |-- rank 0 / Spark 1 / cuda:0 --\
                         |                               | Ulysses SP + NCCL/RoCE
                         `-- rank 1 / Spark 2 / cuda:0 --/
                                      |
                              rank 0 encodes MP4 -> client
```

This is model parallelism, not two independent jobs: every denoising step is
split across the two ranks.

## Prerequisites

- Two ARM64 DGX Sparks with the FL2VA checkpoint available at the same absolute
  path on both nodes.
- Passwordless SSH aliases for the head and peer.
- The dedicated fabric addresses, interface, RoCE HCA, and live GID index for
  each node. Do not assume both nodes use the same GID index after reboot.
- Docker with GPU and `/dev/infiniband` access.
- The known-good `minimax-h3-dgx-spark:sm121-fp8` base image from the
  single-Spark project on the head.
- Authorization to use MiniMax H3 in your territory.

The checkpoint and base image are not redistributed here. Paths, hosts,
addresses, ports, interface, HCA, and GID are configurable in `.env`; the
example uses non-routable documentation addresses that must be replaced.

## Quick start

```bash
git clone https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark.git
cd MiniMax-H3-2x-DGX-Spark
cp .env.example .env
# Edit .env: both hosts, fabric, shared paths, and license acknowledgement.

make audit
make build
./scripts/start-two-sparks.sh
make status
```

Build the `minimax-h3-dgx-spark:sm121-fp8` base image first by following the
[single-Spark repository](https://github.com/joeynyc/MiniMax-H3-DGX-Spark).
`make build` then transfers the derived two-node image to the peer and refuses
to continue unless the accepted base lineage, runtime versions, and both
derived image IDs match. The accepted companion commit, upstream digest, local
base image ID, and exact runtime package versions are recorded in
[the reproducibility guide](docs/REPRODUCIBILITY.md).

Cold startup took about 8 minutes in the accepted run. Wait for `make status`
to confirm both Ray nodes, HTTP health, and exact model identity. The API is
served only on the configured private head-node address on port `8000`;
synchronous generation uses
`POST /v1/videos/sync`.

The default fast profile compiles lazily. Treat the first successful request
after a cold start as warm-up and measure steady-state latency from the second
request onward.

The default launcher keeps cross-step caching off for maximum fidelity. To
start the measured balanced Cache-DiT profile instead:

```bash
./scripts/start-cache-dit-profile.sh
./scripts/wait-ready.sh
```

Run `./scripts/start-two-sparks.sh` to return to the no-cache profile. Both
launchers replace only this experiment's three containers; model data and
benchmark artifacts are preserved.

Only where the model license permits generation and display:

```bash
make smoke
make verify
```

Stop only this experiment with:

```bash
make down
```

## Components

- `h3_multinode/executor.py`: Ray-backed cross-host diffusion executor.
- `h3_multinode/worker.py`: global-rank-aware diffusion worker for one GPU per host.
- `docs/ARCHITECTURE.md`: control path, data path, failure model, and scope.
- `docs/REPRODUCIBILITY.md`: accepted image lineage, commit, and runtime versions.
- `scripts/preflight.sh`: fail-closed host, model, image, port, and fabric checks.
- `scripts/start-two-sparks.sh`: starts Ray rank infrastructure and the API.
- `scripts/start-cache-dit-profile.sh`: starts the measured optional balanced cache profile.
- `scripts/stop-two-sparks.sh`: stops only this experiment's containers.
- `scripts/status.sh`: verifies both containers, Ray nodes, health, and model identity.
- `scripts/wait-ready.sh`: waits for cold loading, then runs the full status check.
- `scripts/smoke-t2va.sh`: fixed one-video acceptance request.
- `scripts/benchmark-smoke.sh`: retains fixed-input performance artifacts and metadata.
- `scripts/benchmark-hq-beach.sh`: repeats the fixed 50-step quality workload without overwriting results.
- `scripts/check-runtime-versions.py`: verifies the accepted image runtime package set.
- `scripts/verify-output.sh`: stream, full-decode, audio, and hash verification.
- `scripts/nccl-smoke.py`: isolated two-rank all-reduce test.
- `scripts/public-audit.sh`: tracked-file, credential-pattern, size, and syntax checks.

The known-good single-Spark deployment is not modified by this repository.

## Experimental boundaries

- The custom executor is deliberately pinned to exactly two one-GPU hosts.
- It has been accepted for synchronous, batch-size-one FL2VA generation, not
  concurrent production traffic or arbitrary diffusion models.
- The API/head rank uses more memory because the encoders and output path are
  not evenly sharded.
- Fixed-seed decoded midpoint frames matched exactly across the two accepted
  runs. MP4 byte hashes differed because the encoded container differed by
  eight bytes, so this is not claimed as bit-for-bit MP4 determinism.
- The accepted cuDNN/compiled profile is deterministic across its measured warm
  runs, but changing attention backends changes the numerical trajectory and
  can change the sampled scene for the same seed.
- Cache-DiT reuses block residuals across selected denoising steps. It is
  deterministic in the measured warm runs but intentionally changes pixels;
  use the default no-cache launcher when exact full-compute fidelity matters.
- This patches an internal vLLM-Omni executor hook and should be revalidated
  against every upstream image change.
- Ray, its dashboard, the API, and dynamic worker traffic use host networking
  without authentication. The dashboard and API bind only to the configured
  head fabric address; the remaining distributed ports still require mutually
  trusted, firewalled nodes. See [SECURITY.md](SECURITY.md).

## Reproducibility language

“Reproducible” here means the documented topology repeatedly started, both
ranks performed one generation, and the output passed the same media checks.
It does not mean byte-identical MP4 output, compatibility with arbitrary
hardware, or an upstream-supported multi-node H3 configuration.

## Upstream and attribution

- [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Companion single-Spark compatibility recipe](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)

Repository code is Apache-2.0. MiniMax H3 weights and outputs remain subject to
MiniMax's separate license. See [NOTICE](NOTICE) for attribution.
