---
phase: 30-upscale-tab-v1
plan: "05"
subsystem: core-inference
tags: [onnx, ort, upscale, graph-optimization, tiling, hat, esrgan]

requires:
  - phase: 30-01
    provides: upscale module skeleton with TilingConfig, upscale_tiled
  - phase: 30-04
    provides: tile planner + upscale_tiled with overlap-blend; TilingConfig struct

provides:
  - OrtEngine::new_with_optimization_level(model, intra_threads, level)
  - GraphOptimizationLevel re-exported from engine module
  - upscale_rgba public entry point gating Level2 (HAT) vs Level3 (ESRGAN)
  - pick_optimization_level pure helper function under unit test

affects: [30-upscale-tab-v1/30-06, subprocess-worker, inpaint-dispatch]

tech-stack:
  added: []
  patterns:
    - "GraphOptimizationLevel gate: tile_size_multiple.is_some() → Level2 (HAT/Swin); else Level3"
    - "with_session closure: extract Vec<f32> inside lock to satisfy SessionOutputs lifetime"
    - "builder_with_base now parameterized on level; existing callers pass Level3 explicitly"

key-files:
  created:
    - crates/prunr-core/src/upscale/mod.rs
  modified:
    - crates/prunr-core/src/engine.rs
    - crates/prunr-core/src/types.rs
    - crates/prunr-core/src/preprocess.rs
    - crates/prunr-app/src/gui/settings.rs

key-decisions:
  - "Re-export GraphOptimizationLevel via pub use (option a) rather than a custom enum — only caller is upscale::mod in the same crate"
  - "Tensor names hardcoded in upscale_input_name() match arm rather than added to ModelDescriptor — keeps descriptor lean; single use site"
  - "RealEsrganX4Plus + Nomos8kSchatL added to ModelKind enum; preprocess/SettingsModel matches get unreachable! arms — upscale models never enter the seg pipeline"
  - "with_session pattern used (not session() accessor) — extracts Vec<f32> inside closure so SessionOutputs borrow ends before lock releases"
  - "scale=2 path: run 4x model, then Lanczos3 downscale of RgbaImage to half dims; alpha resampled from original input (not 4x alpha) to skip redundant intermediate"

patterns-established:
  - "pick_optimization_level: pure fn on &ModelDescriptor → GraphOptimizationLevel; testable without ONNX load"
  - "builder_with_base(intra_threads, level): level is now a parameter, not hardcoded; all constructors pass Level3 explicitly to preserve prior behavior"

requirements-completed: [Criterion 10]

duration: 90min
completed: 2026-05-14
---

# Phase 30 Plan 05: OrtEngine Level Override + upscale_rgba Entry Point Summary

**ORT Level2/Level3 gate wired to ModelDescriptor.tile_size_multiple: HAT models use Level2 to avoid first-tile shape baking; upscale_rgba public entry point dispatches tiled ONNX inference through the engine.**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-05-14
- **Completed:** 2026-05-14
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- `OrtEngine::new_with_optimization_level` constructor threads an explicit level through `builder_with_base`, preserving Level3 default for all existing paths
- `GraphOptimizationLevel` re-exported from engine module so `upscale::mod` can name it without a direct `ort` dep at the call site
- `upscale_rgba` public entry point: looks up descriptor, picks level, constructs OrtEngine, dispatches to `upscale_tiled` with a `run_tile` closure using `with_session` for safe SessionOutputs lifetime management
- `pick_optimization_level` is a pure fn under unit test — pins the Level2 gate for HAT without requiring any ONNX file load

## Task Commits

1. **Task 1: OrtEngine optimization-level override** - `f558345` (feat)
2. **Task 2: upscale_rgba public entry point** - `cc17a73` (feat)

## Files Created/Modified

- `crates/prunr-core/src/engine.rs` — new_with_optimization_level constructor; builder_with_base parameterized on level; GraphOptimizationLevel re-exported
- `crates/prunr-core/src/upscale/mod.rs` — upscale_rgba entry point; pick_optimization_level helper; upscale_input_name helper; unit test
- `crates/prunr-core/src/types.rs` — RealEsrganX4Plus + Nomos8kSchatL added to ModelKind; From<ModelId> reverse mapping
- `crates/prunr-core/src/preprocess.rs` — unreachable! arms for new ModelKind upscale variants
- `crates/prunr-app/src/gui/settings.rs` — unreachable! arms in SettingsModel From<ModelKind>

## Output Notes (per plan spec)

1. **GraphOptimizationLevel encoding:** re-export via `pub use ort::session::builder::GraphOptimizationLevel` (option a). Custom enum was not used — only caller is `upscale::mod` in the same crate.
2. **Session accessor:** `engine.with_session(|session| { ... })` — no `.session()` accessor added. `SessionOutputs` borrows from Session; extracting to `Vec<f32>` inside the closure resolves the lifetime.
3. **Tensor name encoding:** hardcoded match in `upscale_input_name(id)` in `upscale/mod.rs` — "data" for RealEsrganX4Plus, "input" for Nomos8kSchatL. Descriptor field not used (single use site, descriptor stays lean).
4. **RAM estimate:** unchanged from plan 30-04 doc-comment table. upscale_rgba adds one OrtEngine on the stack (session Mutex) and the run_tile closure captures engine by ref — no additional full-image buffer beyond what upscale_tiled already allocates per tile.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `inputs![...]?` type error**
- **Found during:** Task 2
- **Issue:** Plan sketch applied `?` to the `inputs![]` macro call; macro returns `Vec<...>`, not `Result`
- **Fix:** Moved `?` to `session.run(inputs![...])` — the run call returns `Result`
- **Files modified:** crates/prunr-core/src/upscale/mod.rs
- **Commit:** cc17a73

**2. [Rule 1 - Bug] `DynamicImage::ImageRgb8` type mismatch for scale=2 path**
- **Found during:** Task 2
- **Issue:** `upscale_tiled` returns `RgbaImage`; plan sketch wrapped it in `ImageRgb8`
- **Fix:** Changed to `DynamicImage::ImageRgba8(scale4_result)`
- **Files modified:** crates/prunr-core/src/upscale/mod.rs
- **Commit:** cc17a73

**3. [Rule 1 - Bug] `into_raw_vec()` deprecated**
- **Found during:** Task 2
- **Issue:** ndarray deprecated `into_raw_vec()`
- **Fix:** Used `into_raw_vec_and_offset().0`
- **Files modified:** crates/prunr-core/src/upscale/mod.rs
- **Commit:** cc17a73

**4. [Rule 2 - Missing critical functionality] Exhaustive match failures in preprocess.rs and settings.rs**
- **Found during:** Task 2
- **Issue:** Adding RealEsrganX4Plus + Nomos8kSchatL to ModelKind broke exhaustive matches
- **Fix:** Added `unreachable!` arms — upscale models never flow through seg preprocess or SettingsModel conversion
- **Files modified:** crates/prunr-core/src/preprocess.rs, crates/prunr-app/src/gui/settings.rs
- **Commit:** cc17a73

## Self-Check: PASSED

- FOUND: .planning/phases/30-upscale-tab-v1/30-05-SUMMARY.md
- FOUND: f558345 (Task 1 commit)
- FOUND: cc17a73 (Task 2 commit)
