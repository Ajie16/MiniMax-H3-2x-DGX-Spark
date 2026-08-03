# Two-Spark acceptance results

Verified August 3, 2026 on Spark 1 / JoeyDGX (rank 0) and Spark 2 / gx10
(rank 1) over their private fabric addresses.

## Runtime proof

- Ray reported two active nodes and two allocated GPUs with no pending demands
  or node failures.
- Both workers initialized NCCL on local `cuda:0` as global ranks 0 and 1.
- NCCL reported `nRanks 2`, `nNodes 2`, `Using network IB`, and bidirectional
  `NET/IB/0` traffic on `rocep1s0f1` with RoCEv2 GID index 3.
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
