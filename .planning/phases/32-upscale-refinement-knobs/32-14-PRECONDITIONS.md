---
phase: 32
plan: "14"
slug: fp16-variants
status: complete
created: 2026-05-16
---

# Phase 32 Plan 14 — Preconditions: fp16 Variants of Upscale Models

## Scope

| Model | Status | Reason |
|---|---|---|
| 4xNomos8kSCHAT-L | ✅ already fp16 | Phhofm's official .onnx IS fp16; `UpscaleModelKnobs.is_fp16: true` |
| Real-ESRGAN x4plus | 🔧 needs fp16 export | fp32 was the only existing build |
| Real-ESRGAN x2plus | 🔧 needs fp16 export | fp32 was the only existing build |
| 4x-NMKD-Siax-CX | 🔧 needs fp16 export | Shipped fp32 in plan 32-13 commit `8a970d1` |
| 4x-NMKD-Superscale | 🔧 needs fp16 export | Same |

Both NMKD models were added by Plan 32-13 — Plan 32-14 picks them up
as a bonus since they share the RRDB-23 arch and the same export
pipeline.

## How `engine.rs::optimized_variant_bytes` consumes fp16 variants

1. `OrtEngine::new_with_optimization_level` → `optimized_variant_bytes`.
2. On non-macOS + non-cpu_only → `prunr_models::model_fp16_bytes(id)`.
3. → `load_variant(id, "fp16")` looks up `{name}_fp16.onnx` in:
   - Dev path `{CARGO_MANIFEST_DIR}/../../models/` (debug builds only)
   - **`on_demand_dir()`** (new — added by 32-14 commit `d0e27a9`)
   - Exe-adjacent `{exe_dir}/models/`
4. Returns Some(bytes) → session built on fp16 graph. None → fp32 fallback.

For seg models the file ships via `cargo xtask fetch-models` or the
dev-models build. For upscale models, the fetcher downloads the
`_fp16.onnx` sibling into `on_demand_dir()` automatically — see the
new `ModelSource::OnDemand.fp16: Option<OnDemandVariant>` field and
the patched `kick_off_single` in `download_manager.rs`.

---

## Export pipeline

Built `/tmp/realesrgan-export/export_all_fp16.py` — one script that
exports all four RRDB models in sequence. The script:

1. Loads the `.pth` (already on disk from Phase 30 / 32-05 / 32-13).
2. Real-ESRGAN: reads `params_ema`. NMKD: applies the legacy-key remap
   from `export_nmkd.py` (commit `8a970d1`).
3. Calls `model.half()` on the PyTorch model (the only line that
   distinguishes this from the fp32 export).
4. Runs `torch.onnx.export(... opset=17, dynamic H/W)` with a
   `torch.float16` dummy input.
5. Validates with `onnx.checker.check_model` and asserts input/output
   tensor type is ELEM_TYPE=10 (float16).

Total runtime: ~10 minutes for all four (CPU).

---

## Captured values (uploaded to `aktiwers/prunr/releases` tag `models-v1`)

### RealESRGAN_x4plus_fp16

- URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x4plus_fp16.onnx`
- sha256: `838e4fad9a14a70e96c1953a08da2a0638c23b1dffc5b740351f0bc525b36250`
- Size: `33,654,911` bytes (≈ 33 MB)
- Input dtype: `float16` (ONNX elem_type 10) ✓
- Output dtype: `float16` ✓
- Tensor names: `data` / `output` (matches fp32) ✓

### RealESRGAN_x2plus_fp16

- URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x2plus_fp16.onnx`
- sha256: `700b02d23c0a547441a112ac229424b86039e119c5fefae39ecc74e4ce4b4edc`
- Size: `33,668,892` bytes (≈ 33 MB)
- Input dtype / Output dtype / Tensor names: same as x4plus ✓

### 4x-NMKD-Siax-CX_fp16

- URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX_fp16.onnx`
- sha256: `47645594b73a4e5d563babf4674eb756a5853e3cb5620547c02ad6a9e753d818`
- Size: `33,654,911` bytes (≈ 33 MB) ✓

### 4x-NMKD-Superscale_fp16

- URL: `https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Superscale_fp16.onnx`
- sha256: `a59481435da813b5b1629b047fc7490a22badd8ed09d54831cd2cf297b0c2bb9`
- Size: `33,654,911` bytes (≈ 33 MB) ✓

---

## Visual smoke (pending manual verification)

The fp16 ↔ fp32 perceptual difference is below the noise floor for
8-bit u8 output in theory. Manual verification still recommended on
the user's actual hardware:

- [ ] Process the same source image through `RealEsrganUpscale` on
  CPU EP (forces fp32 path) and a GPU EP (picks fp16).
- [ ] Compare outputs side-by-side. No visible banding, color shift,
  or NaN pixels in flat regions.
- [ ] Spot-check timing: GPU + fp16 should be ~2× faster than GPU + fp32.

Defer to phase verification step.
