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

## Local deployment (2026-08-19, this fork)

This fork is deployed live on the two-Spark cluster at the workspace root
`/home/xujie/workspace/dgx-spark-minimax-h3`. The roles are **flipped** so the
local work machine (`spark-0d97`) runs only the light rank 1 while the API
host is the second Spark:

| Role | Host | Fabric address | Distributed rank | Memory after load |
|---|---|---:|---:|---:|
| API / rank 0 (encoders, VAE, output) | `xujie2` / `spark-ac8f` | 10.100.65.1 | 0 | ~97 GiB used |
| rank 1 (DiT half) | local / `spark-0d97` | 10.100.65.2 | 1 | ~48 GiB used |

- API base: `http://10.100.65.1:8000/v1`; Ray dashboard:
  `http://10.100.65.1:8265` (private fabric only, no auth by design).
- Fabric: `enP2p1s0f0np0`, HCA `roceP2p1s0f0`, RoCEv2 GID index 3 on both
  nodes; NCCL verified at ~21 GB/s all_gather.
- Images: base `minimax-h3-dgx-spark:sm121-fp8` is locally re-derived from
  companion commit `8bd7628` as `sha256:e498adce…` (the upstream acceptance
  ID is not reproducible across machines because Docker image IDs embed build
  metadata; upstream digest, layer ancestry, and runtime versions are still
  verified fail-closed). The service image
  `minimax-h3-2x-dgx-spark:experimental` is `sha256:81001bb2…` and identical
  on both nodes.
- Build on this cluster needs host networking and a reachable PyPI mirror:

  ```bash
  H3_BUILD_NETWORK=host \
  H3_BUILD_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  H3_BUILD_PIP_TRUSTED_HOST=mirrors.aliyun.com \
  SYNC_WORKER=1 ./scripts/build-image.sh
  ```

- Measured acceptance (fixed 768x448 T2VA, 20 steps): first request
  70.652 s / 73.551 s (lazy regional compile), warm request 47.914 s; outputs
  are byte-stable across role flips (SHA-256 `212a43c8…`).
- The model is byte-identical on both nodes (81/81 files SHA-256 match,
  ~135 GiB). `ffmpeg`/`ffprobe` are installed on the local host for
  `make smoke` / `make verify`.
- Operational instructions are packaged as the Codex skill
  `dgx-spark-2x-h3-api` (startup, usage, rebuild, troubleshooting), see
  `~/.codex/skills/dgx-spark-2x-h3-api`.

### Ref2VA multi-reference inputs (2026-08-22)

The derived image overwrites the upstream H3 pipeline
(`patches/minimax_h3_pipeline.py`) and patches the OpenAI video API
(`patches/allow-mixed-ref-inputs.patch`) so the Ref2VA partition accepts the
full Comfy-style reference set: up to **9 images**, **3 videos**, and **3
standalone audios**, in any combination (`task=ref2va` only; fl2va/t2va keep
the upstream single-reference rules).

- Multipart `input_references` uploads are sniffed by content type/filename
  (`h3_multinode.ref_inputs`): images cross Ray as PIL objects, videos are
  persisted on the API node (only rank 0 reads them), audios are decoded in
  the API process to `(waveform, sample_rate)` tuples so no shared host path
  is needed.
- `audio_reference` (`file://` JSON, via `patches/allow-file-audio-url.patch`)
  still works and is merged with any uploaded audios; a `file://` path must
  exist at the same absolute path on both nodes.
- Requests over the caps are rejected with HTTP 400 before any GPU work.
- Condition-label ordering matches Comfy: images first, then each video
  (soundtrack label before its video label), then standalone audios.
- Reference-image alignment follows ComfyUI's `ref_image_size` (pass via
  `extra_params`): `match` (default) scales references to the generation
  canvas for speed; `max` keeps up to a 2048px short edge for stronger
  identity fidelity at much higher token cost.

  `match` is a **non-uniform** resize to the output canvas: provide reference
  images with the same aspect ratio as `width`/`height`, otherwise the subject
  is stretched/squashed. A 1:1 reference pairs with a square canvas (for
  example 768x768, or up to 992x992, 32-aligned and still within the model's
  native ~1.03MP canvas cap) without any aspect distortion; `max` preserves
  the reference aspect ratio regardless of the canvas.

## Verified results

The final public-release acceptance ran on August 4, 2026 using the same
derived image on both nodes:
`sha256:09e6521356bbbb635048228d30e78a36c65352a48f7620c921d5aeff2d21b90b`
The first request after each cold start includes lazy regional compilation;
the second is the warm measurement.

| Profile | Ready from cold | Compile warm-up | Warm request |
|---|---:|---:|---:|
| cuDNN + regional compile, full compute | 588.98 s | 70.337 s | 46.574 s |
| cuDNN + regional compile, balanced Cache-DiT | 584.91 s | 55.412 s | 30.578 s |

All four fixed-input T2VA requests produced 56-frame, 768x448, 24 fps
H.264/AAC MP4 files. Complete FFmpeg decoding, non-silent audio, and midpoint
visual inspection passed. Both GPUs were active during generation, Ray reported
two healthy nodes, and all three containers remained at zero restarts with no
OOM state. The API and Ray dashboard listened only on the configured private
head address.

Earlier matched tests established the broader performance and quality results:

- The first distributed acceptance completed in 68.783 and 64.888 seconds,
  versus 154.956 seconds for the comparable single-Spark request—about 2.3x
  faster client-side.
- cuDNN attention plus regional compilation averaged 48.689 seconds warm,
  24.2% faster than the 64.226-second warm Torch-SDPA/eager baseline.
- The 1344x768, 50-step full-compute workload fell from 2,281.532 to
  1,353.506 seconds. The output fully decoded and measured 0.9134 video SSIM
  against the earlier same-seed result.
- The balanced Cache-DiT profile completed that same workload in 608.991
  seconds—55.0% lower latency and 2.22x the generation rate of full compute.
  Its same-seed output measured 0.888 SSIM and 27.04 dB average video PSNR
  against full compute, with effectively unchanged audio.

These are measured results from one two-machine lab, not vendor benchmarks or
upstream support guarantees. Cache-DiT is a visually inspected speed/quality
tradeoff, not a lossless mode. See
[the complete acceptance record](docs/RESULTS.md) for request settings, hashes,
quality observations, and experimental limits.

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
git clone git@github.com:Ajie16/MiniMax-H3-2x-DGX-Spark.git
cd MiniMax-H3-2x-DGX-Spark
cp .env.example .env
# Edit .env: both hosts, fabric, shared paths, and license acknowledgement.

make audit
make build
./scripts/start-two-sparks.sh
./scripts/wait-ready.sh
make status
```

Build the `minimax-h3-dgx-spark:sm121-fp8` base image first by following the
[single-Spark repository](https://github.com/joeynyc/MiniMax-H3-DGX-Spark).
`make build` then transfers the derived two-node image to the peer and refuses
to continue unless the accepted base lineage, runtime versions, and both
derived image IDs match. The accepted companion commit, upstream digest, local
base image ID, and exact runtime package versions are recorded in
[the reproducibility guide](docs/REPRODUCIBILITY.md).

Cold readiness measured 584.91 to 588.98 seconds in the final acceptance.
`wait-ready.sh` waits through model loading and then checks both Ray nodes,
HTTP health, all three containers, and exact model identity. The API is served
only on the configured private head-node address on port `8000`; synchronous
generation uses `POST /v1/videos/sync`.

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
