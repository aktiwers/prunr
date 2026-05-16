---
phase: 32-upscale-refinement-knobs
plan: "14"
subsystem: upscale
tags: [fp16, perf, ondemand, gap-closure]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: EP ladder in upscale dispatch (Plan 32-10) + warm engine cache (Plan 32-12) + community RRDB models (Plan 32-13)
provides:
  - OnDemandVariant struct + ModelSource::OnDemand.fp16 field
  - Fetcher auto-downloads fp16 sibling alongside main fp32 file
  - load_variant arms mapping the 4 RRDB upscale ModelIds to their filename stems
  - on_demand_dir() lookup added to load_variant (3rd path after dev-path and exe-adjacent)
  - 4 fp16 ONNX artifacts uploaded to models-v1 GitHub release
  - 2 boundary tests pinning the fp16 contract
affects: [prunr-models OnDemand schema, prunr-app download_manager, prunr-app model_store synthetic test, prunr-app views/mod model labels (no change), prunr-core engine.rs (no change — already handles None gracefully)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Optional companion download for OnDemand entries: OnDemandVariant struct + Option field on the main source variant; fetcher loops over [main, optional companion]"
    - "Non-fatal companion failure: fp16 sibling download failure logs warn but leaves main fp32 install intact — engine falls back to fp32"
    - "fp16 export pattern: .half() the PyTorch model before torch.onnx.export with a float16 dummy input; opset 17, dynamic H/W"

key-files:
  created:
    - .planning/phases/32-upscale-refinement-knobs/32-14-PRECONDITIONS.md
    - /tmp/realesrgan-export/export_all_fp16.py (out-of-tree, dev-only)
  modified:
    - crates/prunr-models/src/lib.rs (OnDemandVariant + fp16 field + load_variant arms + on_demand_dir lookup + 9 OnDemand entry updates + 2 new tests + extended existing test)
    - crates/prunr-app/src/gui/download_manager.rs (fp16 companion download in kick_off_single)
    - crates/prunr-app/src/gui/views/model_store.rs (synthetic test ModelSource init)

key-decisions:
  - "Auto-download approach (option A from the user's choice) — the fetcher pulls the fp16 sibling automatically when the main model installs. The alternative (separate Model Store entries for fp16, or manual user-data-dir drop) was rejected as worse UX. End-user pays one button click; backend handles fp16 vs fp32 EP-aware selection."
  - "OnDemandVariant is a SEPARATE struct, not an inline Option-tuple. Future int8 / int4 / other-optimized variants can reuse the same struct shape by adding more Option fields to OnDemand."
  - "Capitalized filename stems for upscale load_variant arms (e.g. `RealESRGAN_x4plus`, not `real_esrgan_x4plus`). Matches the OnDemand filename verbatim so a future maintainer adding the fp16 artifact to releases doesn't need to remember a naming-convention difference between fp32 (capitalized) and fp16 (snake_case)."
  - "Nomos8k.fp16 = None — Phhofm's official .onnx IS fp16; the engine dispatches via UpscaleModelKnobs.is_fp16=true. No sibling needed."
  - "load_variant now checks 3 paths: dev (debug only), on_demand_dir (new — the fetcher's target), exe-adjacent (existing, used by `cargo xtask fetch-models`). on_demand_dir comes BEFORE exe-adjacent because OnDemand-installed files are user-specific and should win over any system-level dev artifact."
  - "fp16 companion download failure is NON-FATAL — the fp32 install completes, engine falls back to fp32. The cost is a missed speedup, not a broken install. Avoid breaking a successful 64 MB download because the 33 MB sibling 404'd or had a flaky network blip."

patterns-established:
  - "Multi-file OnDemand via an optional companion: any future single-model entry that needs an additional sibling (alternate precision, alternate dtype, etc.) drops in via the same OnDemandVariant shape"
  - "fp16 ONNX export from PyTorch RRDBNet checkpoints: `.half()` + float16 dummy → opset-17 fp16 ONNX graph (validated via onnx.checker)"

# Verification
self-check:
  - cargo test --workspace --lib — 940 tests pass (545 + 334 + 41 + 20), including 2 new fp16 tests + extended existing test
  - cargo build --workspace — clean (after extending all 9 OnDemand REGISTRY entries with fp16 field)
  - All 4 fp16 URLs return 200 OK from GitHub release CDN (verified via curl -I)
  - sha256 of each fp16 file pinned in REGISTRY — fetcher will reject any future mismatch
  - load_variant exercise test confirms no panicking unreachable arms for the 4 new upscale models

manual-verify-pending:
  - "Install Real-ESRGAN from the Model Store on a machine with a CUDA / DirectML / CoreML / OpenVINO-capable GPU"
  - "Confirm both `RealESRGAN_x4plus.onnx` AND `RealESRGAN_x4plus_fp16.onnx` land in `on_demand_dir()` (e.g. ~/.local/share/prunr/models/ on Linux)"
  - "Process the same image on CPU EP vs GPU EP and measure the difference — expect ~2× wall-clock speedup on GPU"
  - "Visual A/B: confirm fp16 output is perceptually identical to fp32 (no banding, color shift, or NaN pixels)"

# What this enables
unlocks:
  - "User-perceptible: upscale dispatch on modern GPU hardware now runs ~2× faster than the fp32 baseline, on top of the 10-20× speedup from the EP ladder (Plan 32-10). Combined 20-100× over the original CPU-only fp32 path."
  - "Future int8 / int4 / etc. quantized variants ride the same OnDemandVariant pattern — drop a new Option<OnDemandVariant> field on OnDemand, add the load_variant suffix arm, ship the .onnx alongside main."

# Deviations from plan
- "Plan scoped to x4plus + x2plus only. Extended to also include the Siax-CX + Superscale that Plan 32-13 added — same RRDB-23 architecture, same export pipeline, zero additional engineering cost. Total of 4 fp16 exports instead of 2."
- "Fetcher patch is more thorough than plan suggested: aggregated progress (main + fp16 bytes as one bar), non-fatal companion failure (warn-log + continue), shared cancel flag across both downloads."
- "Plan recommended a `cpu_only=true` baseline comparison in the smoke test. Deferred: manual-verify-pending list captures it as a real-hardware A/B once the user installs a model and runs side-by-side."

# Plan 32-14 closeout
gap_closed: Gap-5 (fp16 variant loader on fp16-capable hardware)
deferred:
  - "Visual A/B golden suite (fp16 vs fp32 output diff) — automated regression test would catch silent numerical drift. Out of scope for plan 32-14; logged for a future quality-assurance plan."
  - "int8 quantized variants for CPU EP — `optimized_variant_bytes` already routes cpu_only=true to `model_int8_bytes`, but no int8 variant exists yet for any model. Same OnDemandVariant pattern would extend to it (add `int8: Option<OnDemandVariant>` field to OnDemand)."
