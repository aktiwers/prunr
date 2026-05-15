---
phase: 32-upscale-refinement-knobs
plan: 05
subsystem: core-pipeline
tags: [rust, prunr-models, prunr-core, upscale, registry, two-pass, onnx]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    plan: 01
    provides: OutputScale enum with X4TwoPass variant; UpscaleRecipe shape
  - phase: 30-upscale-tab-v1
    provides: upscale_rgba entry point, REGISTRY structure, UpscaleModelKnobs

provides:
  - ModelId::RealEsrganX2Plus with real SHA256/URL/size from PRECONDITIONS.md
  - REGISTRY entry for x2plus (tile_size_multiple=Some(2), working_set_mb=1000)
  - is_user_visible() predicate on ModelId — false only for x2plus
  - ModelKind::RealEsrganX2Plus + From/TryFrom impls in prunr-core
  - run_upscale_native() private helper (native-scale dispatch, shared by both callers)
  - upscale_two_pass() public entry point — two-pass 4x scheduler hardcoded to x2plus
  - x4twopass_available(model_id) -> bool predicate for chip UI gating (wave 5)

affects:
  - 32-06 (dispatch wiring uses upscale_two_pass for X4TwoPass OutputScale)
  - 32-07 (chip UI uses x4twopass_available to grey out X4TwoPass for Nomos8k)
  - model_store: x2plus filtered from visible list via is_user_visible()

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "run_upscale_native private helper: native-scale tile dispatch shared by upscale_rgba (4x models) and upscale_two_pass (2x chained); eliminates duplicate ORT dispatch code"
    - "is_user_visible() predicate on ModelId: single source of truth for which models appear in the picker/store vs are internal-only dispatch models"
    - "x4twopass_available(model_id) predicate: chip UI reads this to grey out X4TwoPass for non-ESRGAN models; no model_id parameter on upscale_two_pass itself"

key-files:
  created: []
  modified:
    - crates/prunr-models/src/lib.rs
    - crates/prunr-core/src/types.rs
    - crates/prunr-core/src/preprocess.rs
    - crates/prunr-core/src/upscale/mod.rs
    - crates/prunr-app/src/gui/settings.rs
    - crates/prunr-app/src/gui/views/model_store.rs

key-decisions:
  - "tile_size_multiple=Some(2) for x2plus — PRECONDITIONS.md smoke test confirmed pixel_unshuffle asserts even H/W; plan said None but captured values override"
  - "working_set_mb=1000 — two-pass smoke test peaked at 963 MB RSS; rounded UP to nearest 100 per PRECONDITIONS.md"
  - "run_upscale_native refactor (Option B from plan): private helper with native_scale param eliminates code duplication between upscale_rgba (4x path) and upscale_two_pass (2x chain)"
  - "upscale_rgba keeps native_scale_4x hardcoded at 4 for all user-selectable models; x2plus is NEVER called from upscale_rgba — exclusively via upscale_two_pass"
  - "upscale_two_pass composes alpha from original input (direct 4x Lanczos3), ignoring pass-2 output alpha to avoid double-upscale artifacts on the alpha channel"
  - "is_user_visible() on ModelId: model_store filters with it — x2plus never shown as a standalone choice; installed as a side effect when X4TwoPass is picked"

patterns-established:
  - "Internal model hiding: is_user_visible() on ModelId + model_store filter — single predicate, two enforcement sites"
  - "native_scale parameter on run_upscale_native: the tile closure and upscale_tiled both receive the model's actual output multiplier, preventing ORT output dimension mismatches"

requirements-completed: [Criterion 6, Criterion 7, Criterion 11]

# Metrics
duration: 20min
completed: 2026-05-15
---

# Phase 32 Plan 05: RealEsrganX2Plus Registry + Two-Pass Scheduler Summary

**RealEsrganX2Plus registered with real SHA256 + REGISTRY entry; upscale_two_pass chains x2plus twice for net 4x output; run_upscale_native refactors dispatch to support both 4x and 2x native models**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-15T20:45:00Z
- **Completed:** 2026-05-15T21:07:06Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `ModelId::RealEsrganX2Plus` registered in prunr-models: enum variant, ALL array, stable_name(), bundled_bytes() panic arm, load_variant() return-None arm, exhaustiveness fence
- REGISTRY entry with values verbatim from PRECONDITIONS.md: SHA256 `7e0860bb32d903520a244c327b6e3d5e08d680b95c29b3fa8a02cb6ecd230c60`, URL `models-v1/RealESRGAN_x2plus.onnx`, size_mb=64, working_set_mb=1000, tile_size_multiple=Some(2)
- `is_user_visible()` predicate: false only for x2plus; model_store filters with it; x2plus never appears as a standalone choice
- `ModelKind::RealEsrganX2Plus` with From/TryFrom impls; preprocess.rs, settings.rs match-site fences updated
- `run_upscale_native()` private helper: parameterized on `native_scale` — eliminates duplicate tile dispatch code; upscale_rgba uses it with native_scale=4, upscale_two_pass uses it with native_scale=2
- `upscale_two_pass()`: cancel-before-dispatch guard, single engine for both passes, intermediate moved not cloned, alpha composed from original input
- `x4twopass_available(model_id) -> bool`: returns true only for RealEsrganX4Plus; chip UI wave 5 uses this to dim X4TwoPass for Nomos8k
- 11 new tests across prunr-models and prunr-core; all 900 workspace tests green; clippy clean

## Exact values shipped in REGISTRY

- **SHA256:** `7e0860bb32d903520a244c327b6e3d5e08d680b95c29b3fa8a02cb6ecd230c60`
- **URL:** `https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x2plus.onnx`
- **size_mb:** 64
- **working_set_mb:** 1000 (peak RSS 963 MB during two-pass smoke test, rounded UP to nearest 100)
- **tile_size_multiple:** `Some(2)` (pixel_unshuffle asserts even H/W)
- **input_name:** `"data"` (matches x4plus RRDB export convention)
- **is_fp16:** `false`

## upscale_rgba refactor approach

Option B from the plan was implemented: `run_upscale_native(input, engine, descriptor, native_scale, on_tile_done, cancel)` is the private dispatch core. `upscale_rgba` calls it with `native_scale=4` for all user-selectable models (x4plus / Nomos8k) and post-processes: returns directly for scale=4 requests, Lanczos3-halves for scale=2 requests. `upscale_two_pass` calls `run_upscale_native` with `native_scale=2` directly.

## Task Commits

1. **Task 1: Register RealEsrganX2Plus ModelId + REGISTRY entry** - `d6e76f9` (feat)
2. **Task 2: Add upscale_two_pass + run_upscale_native refactor + x4twopass_available** - `c0a4432` (feat)

**Plan metadata:** `{metadata_commit}` (docs: complete plan)

## Match-site fences updated

| File | What changed |
|------|-------------|
| `crates/prunr-models/src/lib.rs` | `ModelId::ALL`, `stable_name()`, `bundled_bytes()`, `load_variant()`, exhaustiveness fence, REGISTRY |
| `crates/prunr-core/src/types.rs` | `ModelKind` enum, `From<ModelKind>`, `TryFrom<ModelId>` |
| `crates/prunr-core/src/preprocess.rs` | `preprocess()` unreachable arm for upscale variants |
| `crates/prunr-app/src/gui/settings.rs` | `From<ModelKind> for SettingsModel`: unreachable! guard |
| `crates/prunr-app/src/gui/views/model_store.rs` | REGISTRY loop filtered with `is_user_visible()` |

## User-visibility filter wiring

`ModelId::is_user_visible()` (single predicate in prunr-models) returns false only for `RealEsrganX2Plus`. The model_store render loop adds `if !desc.id.is_user_visible() { continue; }` before the category filter — x2plus is installed as a side effect when X4TwoPass is first dispatched, but never surfaced as a standalone download choice.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] tile_size_multiple=Some(2) instead of plan's None**
- **Found during:** Task 1 (reading PRECONDITIONS.md captured values)
- **Issue:** Plan template showed `tile_size_multiple: None` but PRECONDITIONS.md explicitly noted `tile_size_multiple: Some(2) (REQUIRED — pixel_unshuffle asserts even H/W)`. The execution context `<critical_inputs_from_P1>` also specified `Some(2)`. Using None would cause runtime assertion failures.
- **Fix:** Used `Some(2)` as specified in PRECONDITIONS.md captured values
- **Verification:** `x2plus_tile_size_multiple_is_some_2` test passes
- **Committed in:** d6e76f9 (Task 1 commit)

**2. [Rule 2 - Missing Critical] Added run_upscale_native refactor to fix x2plus dispatch**
- **Found during:** Task 2 (analyzing upscale_rgba internals)
- **Issue:** Plan's proposed `upscale_two_pass` pseudocode called `upscale_rgba(input, x2plus, 2, ...)` — this is wrong because `upscale_rgba` hardcodes `out_h = padded_h * 4` in the tile closure, which would produce a dimension mismatch when x2plus (a native-2× model) returns 2× output into a buffer expecting 4×.
- **Fix:** Implemented Option B from the plan: `run_upscale_native` private helper with `native_scale` parameter, used by both `upscale_rgba` (with 4) and `upscale_two_pass` (with 2)
- **Files modified:** `crates/prunr-core/src/upscale/mod.rs`
- **Verification:** `cargo build --workspace` clean; cancel test passes; knobs test passes
- **Committed in:** c0a4432 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 data-correctness, 1 Rule 2 missing-critical dispatch fix)
**Impact on plan:** Both fixes mandatory — None on tile_size_multiple would crash at runtime; wrong scale in dispatch would produce garbled output. No scope creep.

## Issues Encountered

None beyond the deviations documented above.

## Next Phase Readiness

- Plan 32-06: can wire `OutputScale::X4TwoPass` → `upscale_two_pass` in the processor dispatch path
- Plan 32-07: chip UI can call `x4twopass_available(model_id)` to grey out X4TwoPass for Nomos8k
- `every_model_id_has_a_registry_entry` exhaustiveness test passes with x2plus included

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-15*
