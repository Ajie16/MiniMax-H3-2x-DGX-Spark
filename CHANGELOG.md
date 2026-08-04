# Changelog

## Unreleased

- Repeated the complete live public-release acceptance on both Sparks using the
  branch-built image ID
  `sha256:09e6521356bbbb635048228d30e78a36c65352a48f7620c921d5aeff2d21b90b`.
  The default full-compute profile became ready in 588.98 seconds and completed
  compile-warm-up and warm T2VA requests in 70.337 and 46.574 seconds. The
  balanced Cache-DiT profile became ready in 584.91 seconds and completed the
  same requests in 55.412 and 30.578 seconds. All four outputs passed complete
  FFmpeg decoding as H.264/AAC MP4 files; these release measurements do not
  replace the previously published matched 50-step benchmark results.
- Bound the unauthenticated H3 API and Ray dashboard to the configured private
  head fabric address instead of every host interface.
- Replaced internal lab host identities in public examples with generic
  `spark-head` and `spark-peer` placeholders.
- Recorded and enforced the accepted upstream image digest, companion commit,
  local base image ID, ARM64 lineage, and measured runtime version set.
- Added opt-in Cache-DiT launch controls and a measured balanced profile using
  threshold 0.15, four full-compute warm-up steps, no TaylorSeer, and a
  one-step cache ceiling. The default launcher remains no-cache.
- The balanced profile reduced the matched 1344x768, 50-step request from
  1,353.506 to 608.991 seconds (55.0% lower latency, 2.22x generation rate).
  Full decode and multi-frame inspection passed; same-seed video measured
  0.888 SSIM and 27.04 dB PSNR against full compute.
- Made IPv4 RoCEv2 GID selection node-specific so a peer reboot cannot make Ray
  propagate the head node's now-invalid GID index to both diffusion actors.
- Added configurable Torch-SDPA, cuDNN, and FlashAttention diffusion backends,
  configurable eager/compiled execution, readiness waiting, and fixed-input
  benchmark tooling.
- Promoted cuDNN attention plus regional compilation after two final warm runs
  averaged 48.689 seconds, 24.2% faster than the 64.226-second warm baseline.
  FlashAttention 4 was rejected after an SM121 CuTe/CUTLASS JIT failure.
- Repeated the earlier same-seed 1344x768, 50-step, four-second quality request
  with the accepted profile. End-to-end time fell from 2,281.532 to 1,353.506
  seconds (40.7% lower latency, 1.69x generation rate); full media decode and
  multi-frame visual inspection passed.
- Added public-release documentation, model-license warning, security policy,
  contributor guidance, output verification, automated audit, and CI.

## 1.0.0 - 2026-08-03

- Added a Ray-backed two-host diffusion executor for the pinned MiniMax H3
  FL2VA vLLM-Omni stack.
- Added fail-closed two-Spark launch, status, stop, NCCL, and T2VA acceptance
  tooling.
- Recorded two successful one-video runs using Ulysses sequence parallelism
  and NCCL over RoCEv2.
