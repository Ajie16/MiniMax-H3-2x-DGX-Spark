# LoRA serving on two DGX Sparks

This repository can load a **catalogued PEFT adapter** on both ranks. The
default is still `H3_LORA_MODE=off`: no extra mounts, no catalog, same
behavior as the measured two-Spark recipe.

The first lab profile is LightX2V FL2VA **4-step turbo** (`turbo4`), not a
lossless 20-step substitute. Adapter weights inherit the MiniMax H3 Community
License, including territorial restrictions. See [MODEL-LICENSE.md](../MODEL-LICENSE.md).
Do not commit adapters, manifests of real weights, or generated media.

## Modes

| `H3_LORA_MODE` | Meaning |
|---|---|
| `off` (default) | Ignore other `H3_LORA_*` variables. Do not mount a LoRA directory. |
| `static` | Load one catalog name at start. Compile is allowed. Scale is frozen at init. |
| `request` | Per-request catalog `name`. Requires `H3_EXECUTION_MODE=eager`. |

LoRA is mutually exclusive with Cache-DiT, including
`H3_CACHE_PROFILE=balanced` / `H3_CACHE_PROFILE_OVERRIDE=balanced`.

## Lab v1 (turbo4)

Set these in the **local** `.env` (never in git):

```bash
H3_LORA_DIR=/absolute/path/on/both/hosts/loras
H3_LORA_MODE=static
H3_LORA_NAME=turbo4
H3_LORA_ALLOW_TURBO=true
# omit H3_LORA_SCALE → catalog default_scale 1.0
```

Use the same absolute `H3_LORA_DIR` on both Sparks. Copy the PEFT directory
and `sha256` manifest, then let `scripts/preflight.sh` compare hashes.

Turbo requests must use **4** inference steps. This lab ingested LightX2V
**FL2VA turbo 4-step v0.1** (Diffusers split `to_q`/`to_k`/`to_v`, rank 128)
from `ComfyUI/models/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors` with
`--qkv-layout=qkv`. Catalog shifts are **12 / 3**, not the 768p v1.0 6/3
schedule. The existing `make smoke` path stays **20-step, no LoRA**. Do not
treat SSIM/PSNR against that baseline as a quality score.

4-step **T2VA** with this FL2V turbo adapter is not a substitute for 20-step
quality. ComfyUI's sharp 4-step H3 clips use **Ref2VA + `minimax_h3_ref2v_turbo_*`**
(euler, SigmaShift 12/3). The FL2VA workflow keeps the FL2V turbo node at
strength 0 and samples 20-step `res_multistep`.

LoRA is applied by **activation-add**: the online FP8 base GEMM runs unchanged
and the BF16 delta is added per slice in `apply()` (design K9). Do **not** bake
the delta into the FP8 weights: per-tensor requantization was measured
(2026-08-22, ref2v block 0 qkv/fc1) to erase ~99.6% of the delta (only
0.35-0.4% of the norm survives, correlation ~0), because the delta absmean is
~1e-5 while the per-tensor FP8 step is ~3e-2. With activation-add the delta is
preserved at ~99.9%, matching ComfyUI's BF16 merge.

Serving still needs `stacked_params_mapping` so fused `qkv_proj` / `fc1` load
the split `to_q`/`to_k`/`to_v` tensors (Slice 4 / companion mapping). Do not
enable `H3_LORA_MODE=static` until that lands.

`H3_LORA_ALLOW_TURBO` defaults to `false` in `.env.example` so a 20-step smoke
cannot be labeled “turbo works” by accident.

## Catalog

See [lora-catalog.example.json](lora-catalog.example.json). Copy it to
`$H3_LORA_DIR/catalog.json` after ingest. Catalog keys and adapter directory
names are `^[a-z0-9._-]+$`. Clients may send a catalog `name`; they must not
send a filesystem `path`.

Scale resolution: request scale (request mode only) > explicit `H3_LORA_SCALE`
> catalog `default_scale` > `1.0`. Unset is not the same as `1.0`.

## Ingest

ComfyUI / Diffusers single-file turbo weights are not PEFT directories. Convert
them with `scripts/ingest-lora.sh` (Slice 2) and a required `--qkv-layout`
before enabling `static`.
