# Two-Spark acceptance results

Verified August 3, 2026 on a two-DGX-Spark lab pair, with the head as rank 0
and the peer as rank 1 over their private fabric addresses.

## Public-release acceptance on August 4, 2026

The release candidate was rebuilt from the public branch and produced image ID
`sha256:09e6521356bbbb635048228d30e78a36c65352a48f7620c921d5aeff2d21b90b`
on both ARM64 nodes. The provenance gate confirmed the pinned upstream digest,
companion commit, accepted local base-image ID, and the runtime versions listed
in `REPRODUCIBILITY.md`.

Both the default full-compute launcher and the optional balanced Cache-DiT
launcher completed a fresh cold start and two fixed-input smoke requests. The
first request after each start includes lazy regional compilation; the second
is the warm measurement.

| Profile | Ready time | Compile warm-up | Warm request |
|---|---:|---:|---:|
| cuDNN + regional compile, full compute | 588.98 s | 70.337 s | 46.574 s |
| cuDNN + regional compile, balanced Cache-DiT | 584.91 s | 55.412 s | 30.578 s |

All four results were 56-frame, 768x448, 24 fps H.264 video with 32 kHz stereo
AAC and passed a complete FFmpeg decode. Midpoint inspection matched the PCB
soldering prompt, and audio was non-silent. During denoising, the two GPUs were
simultaneously measured at 96%/95% utilization for full compute and 96%/94%
for Cache-DiT.

For both profiles, Ray reported two active nodes and two allocated GPUs with no
failures. `/health` returned HTTP 200, `/v1/models` matched the configured
checkpoint exactly, and all three containers remained running with zero
restarts and `OOMKilled=false`. The API and Ray dashboard listened only on the
configured private head address; neither accepted a loopback connection. The
balanced profile remained live after acceptance.

This release check intentionally did not repeat the expensive 50-step quality
workload. The matched benchmark values below remain the published quality and
performance claims.

## Runtime proof

- Ray reported two active nodes and two allocated GPUs with no pending demands
  or node failures.
- Both workers initialized NCCL on local `cuda:0` as global ranks 0 and 1.
- NCCL reported `nRanks 2`, `nNodes 2`, `Using network IB`, and bidirectional
  `NET/IB/0` traffic on `rocep1s0f1`. Each node's live IPv4-mapped RoCEv2 GID
  index is validated independently because the indices can differ after boot.
- During each single request, both GPUs sustained approximately 96% utilization.
- All three experiment containers remained running with zero restarts and
  `OOMKilled=false`; `/health` and exact `/v1/models` identity passed.

## Request

| Setting | Value |
|---|---:|
| Task | T2VA |
| Prompt | `Macro soldering a PCB under warm bench light, soft room tone.` |
| Resolution | 768x448 |
| Frame rate | 24 fps |
| Requested duration | 2.0 s |
| Inference steps | 20 |
| Flow shift | 12 |
| Seed | 42 |

## Results

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| Client elapsed | 68.783 s | 64.888 s |
| Engine E2E | 67.686 s | 62.063 s |
| Mean denoising step | 3.384 s | 3.103 s |
| MP4 size | 698,852 bytes | 698,860 bytes |
| SHA-256 | `854adc79f44ad68629b6eda86570c2afb542ec10c781cf30f1669ff2e564eac2` | `511b3953b41a8d6ded5b138af1fd7cabed0d3a88f7fb7b4fb7e89f673fde76b7` |

Both files contained H.264 video at 768x448 and 24 fps plus 32 kHz stereo AAC
audio. Reported duration was 2.357 seconds. Both passed a complete FFmpeg decode,
and visual inspection showed a coherent macro PCB-soldering scene. The decoded
midpoint PNGs were byte-identical across runs even though the MP4 containers
differed by eight bytes.

The comparable known-good single-Spark request took 154.956 seconds with a
7.748-second mean denoising step. The two-Spark client times are 2.25x and 2.39x
faster (about 2.3x across the two accepted runs).

The acceptance host retained run 2 locally as `output/smoke-t2va-2x.mp4` for
verification. Generated media is intentionally ignored and is not present in
the public repository because model outputs have separate license terms.

## Load profile

| Rank | Model load | Process GPU memory after load |
|---|---:|---:|
| Rank 0 / Spark 1 | 89.1659 GiB in 485.837 s | 89.42 GiB |
| Rank 1 / Spark 2 | 41.1969 GiB in 250.609 s | 41.44 GiB |

The load is intentionally asymmetric: rank 0 retains the shared encoders, VAE,
and API output path while the DiT sequence is split across both ranks.

## Performance profile accepted August 3, 2026

The same fixed prompt, seed, resolution, duration, and 20-step workload was used
to compare attention and execution modes. Client times include MP4 encoding and
HTTP transfer.

| Profile | First request | Warm request 1 | Warm request 2 | Result |
|---|---:|---:|---:|---|
| Torch SDPA + eager | 69.720 s | 64.226 s | - | Baseline |
| cuDNN attention + eager | 58.753 s | 52.872 s | - | Stable |
| cuDNN attention + regional compile | 70.712 s | 48.870 s | 48.068 s | Accepted |
| Final restored cuDNN + regional compile | 72.406 s | 47.101 s | 50.276 s | Live acceptance |
| FlashAttention 4 + regional compile | - | - | - | Rejected before step 1 |

The final two warm requests averaged 48.689 seconds, 24.2% faster than the
64.226-second warm SDPA baseline. A separate accepted compiled pair measured
45.29 and 45.46 seconds inside the engine with 2.265 and 2.273-second mean
denoising steps. The two final warm MP4 files had identical SHA-256 values and
both passed full FFmpeg decoding as 768x448 H.264 video with 32 kHz stereo AAC.

Compilation is lazy: the first request after every cold service start pays the
graph-build cost. Speed claims therefore use subsequent requests. cuDNN changed
the exact fixed-seed visual trajectory relative to SDPA, but inspected midpoint
frames remained coherent and audio was effectively identical. Backend changes
are not claimed to preserve pixel-identical output.

FlashAttention 4 resolved and loaded on SM121, but its first request failed in
the CuTe/CUTLASS DSL because `flash_attn.cute.utils.AuxData` could not be
converted to a JIT argument. It was not promoted. The live service was restored
to cuDNN attention plus regional compilation and warmed with three decoded
acceptance requests.

## Full-quality same-seed comparison

The accepted cuDNN plus regional-compile profile was subsequently tested with
the exact prompt and seed used by the earlier high-quality SDPA run:

| Setting | Value |
|---|---:|
| Resolution | 1344x768 |
| Frame rate | 24 fps |
| Requested duration | 4.0 s |
| Inference steps | 50 |
| Flow shift | 12 |
| Audio flow shift | 3 |
| Seed | 314159 |

The optimized request completed in 1,353.506 seconds client-side (22m 33.506s)
and 1,322.745 seconds inside the engine. Its 49 logged denoising iterations took
21m 17s at a 26.08-second progress-loop mean; the engine reported 26.455
seconds per denoising step. The earlier same-input request took 2,281.532
seconds (38m 1.532s), so the optimized end-to-end run used 928.026 fewer
seconds: 40.7% lower latency, or 1.69x the former generation rate.

The resulting 9,951,160-byte MP4 contains 107 H.264 frames at 1344x768 and
24 fps plus 32 kHz stereo AAC. Its reported duration is 4.482 seconds and its
SHA-256 is
`3fb848190dd8ad0bdb64e4e4176659f06fe735069715fb8aec06c9d4f6049de1`.
A full FFmpeg decode passed.

Nine evenly sampled frames plus full-resolution frames at 0.5, 2.25, and 4.0
seconds were inspected. Exactly five adults remained visible with stable
identities and clothing, plausible walking anatomy, consistent long shadows
and wet-sand reflections, persistent beach footprints, and a coherent moving
shoreline. The prompt's requested demographic breadth was only partly met:
skin tones and gender presentation varied, but the subjects remained mostly
young and conventionally athletic instead of spanning the requested age and
body-type range. This is a useful bias finding, not a claim of physical
simulation or perfect human anatomy.

Against the earlier same-seed SDPA output, decoded video measured 0.9134
overall SSIM and 28.56 dB average PSNR; audio measured approximately 167 dB
PSNR. The outputs are visually close but not pixel-identical, as expected after
changing attention backends.

## Optional Cache-DiT profile

The installed vLLM-Omni image exposes a model-declared Cache-DiT adapter for
MiniMax H3. Three residual thresholds were tested with one front block, four
full-compute warm-up steps, no TaylorSeer, and a one-step cache ceiling. The
default no-cache launcher remained the rollback path.

| Threshold | Warm smoke mean | Change vs 48.689 s | Same-seed video result |
|---:|---:|---:|---|
| 0.05 | 46.134 s | 5.2% lower | Byte-identical; no effective cache reuse |
| 0.15 | 32.561 s | 33.1% lower; 1.50x rate | SSIM 0.823; 23.69 dB PSNR |
| 0.24 | 27.676 s | 43.2% lower; 1.76x rate | SSIM 0.735; 20.72 dB PSNR |

Threshold 0.15 was selected as the balanced profile. Its two warm smoke runs
took 32.334 and 32.787 seconds and produced identical MP4 hashes. Both fully
decoded as 56-frame 768x448 H.264 video with 32 kHz stereo AAC. Audio PSNR
against the no-cache run was approximately 166 dB.

The selected profile then repeated the exact 1344x768, 24 fps, 50-step,
four-second, seed-314159 beach request. Client elapsed time was 608.991 seconds
(10m 8.991s), and engine E2E was 591.584 seconds. The engine reported an
11.832-second mean denoising-step latency, down from 26.455 seconds. Relative
to the matched 1,353.506-second no-cache run, this is 744.515 seconds saved,
55.0% lower end-to-end latency, and 2.22x the generation rate.

The 9,728,247-byte result contains 107 H.264 frames at 1344x768 and 24 fps,
32 kHz stereo AAC, and a 4.482-second reported duration. Its SHA-256 is
`85a402bc08de4bd285c69eb24397cc05063577609a98ee1452d6fb4e88a93207`;
full FFmpeg decoding passed. Against the matched no-cache output, video
measured 0.8879 overall SSIM and 27.04 dB average PSNR, while audio measured
approximately 167.45 dB PSNR.

Nine evenly sampled frames and full-resolution start/middle/end frames were
inspected. Exactly five stable subjects remained visible with coherent surf,
shadows, reflections, footprints, and walking motion. The output is visually
close but not pixel-identical, so this is documented as an optional balanced
speed mode rather than a lossless replacement for full compute.
