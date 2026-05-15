---
phase: 32-upscale-refinement-knobs
plan: "06"
subsystem: gui-coordinator
tags: [upscale, live-preview, dispatch, tier2, cache]
dependency_graph:
  requires: ["32-01", "32-02", "32-03", "32-04", "32-05"]
  provides: ["upscale_raw cache", "UpscaleTier2 live preview", "dispatch_upscale OutputScale routing"]
  affects: ["item.rs", "processor.rs", "live_preview.rs", "app.rs"]
tech_stack:
  added: []
  patterns:
    - "Arc-shared image buffers (upscale_raw / bicubic_source) for zero-copy snapshot at preview tick"
    - "early-return before seg/edge pipeline for new PreviewKind variant (unreachable arms keep exhaustiveness)"
    - "post-match upscale Tier-2 detection block: recipe diff in apply_toolbar_change gates mark_tweak"
key_files:
  created: []
  modified:
    - crates/prunr-app/src/gui/item.rs
    - crates/prunr-app/src/gui/processor.rs
    - crates/prunr-app/src/gui/live_preview.rs
    - crates/prunr-app/src/gui/app.rs
decisions:
  - "apply_tier2_postprocess placed in app.rs as a module-private free function; live_preview.rs duplicates the same four-call sequence inline — kept separate to avoid a dependency from live_preview.rs onto app.rs (circular)"
  - "UpscaleTier2 early-return at top of run_preview + unreachable! arms on both inner match statements: guarantees exhaustiveness without restructuring the existing Mask/Edge logic"
  - "bicubic_source cached alongside upscale_raw with Arc::ptr_eq invalidation at live-preview tick: avoids re-running Lanczos3 resize every 10 Hz tick on 4K output"
  - "X3 output_scale: left as a pending case — upscale_rgba's two-pass path handles X2; X3 is a scale_factor=3 passthrough but is not wired yet (no UI exposes X3 in plan 32-07)"
  - "Tier-2 detection in apply_toolbar_change: recipe diff computed inline rather than via resolve_auto_dispatch — resolve_auto_dispatch returns DispatchKind::None for UpscaleTier2 (by design) and the caller checks the tier directly post-match"
metrics:
  duration_minutes: 90
  completed_date: "2026-05-15"
  tasks_completed: 3
  tasks_planned: 3
  files_modified: 4
  tests_added: 13
---

# Phase 32 Plan 06: GUI Wiring Wave Summary

Four seams wired connecting Wave 2 algorithms to Wave 1 data layer: `BatchItem.upscale_raw` cache, `dispatch_upscale` OutputScale routing, `PreviewKind::UpscaleTier2` live-preview dispatch, and `RequiredTier::UpscaleTier2` tier-dispatch routing in `app.rs`.

## Tasks Completed

| Task | Name | Commit | Key files |
|---|---|---|---|
| 1 | BatchItem.upscale_raw + bicubic_source cache | 0b1005b | item.rs |
| 2 | dispatch_upscale OutputScale routing + pump_upscale_results wiring | 0354c16 | processor.rs, app.rs |
| 3 | PreviewKind::UpscaleTier2 + live-preview routing + tier dispatch | f5c943b | live_preview.rs, app.rs |

## Output Spec Answers

### 1. Where apply_tier2_postprocess was placed

Two separate sites: `app.rs` has a module-private `apply_tier2_postprocess(img, item, bicubic)` free function used by `pump_upscale_results`. `live_preview.rs`'s `run_preview` inlines the same four-call sequence (ai_blend → sharpen → saturation → color_match) directly in the `UpscaleTier2` early-return block. The two sites are not shared. Sharing them would require either a new module (extra indirection for four lines) or making live_preview.rs depend on app.rs internals (circular). Duplication of four function calls was judged acceptable.

### 2. RAM discipline — dual buffer footprint

`upscale_raw` and `result_rgba` coexist on the item after a Tier-1 dispatch. At 4K source × 4× output this is approximately 15360×8640 RGBA ≈ 500 MB per buffer, for a per-item peak of ~1 GB. `cache_size()` accounts for both buffers so the memory governor sees the full footprint. The trade is documented in the 32-06-PLAN.md "Memory note" comment on the pump_upscale_results action. No user-facing notice was surfaced in this plan (the upscale tab itself is forthcoming in 32-07; the notice belongs there when the knobs are exposed).

### 3. X3 output_scale wiring

Not wired in this plan. `upscale_rgba` passes `scale_factor = 3` through to the core function, which routes to the 4× inference + Lanczos3 downscale path (same shape as X2 but at 3/4 rather than 1/2). No UI exposes X3 in plans up through 32-07; leaving the scale_factor passthrough in place is sufficient. If X3 ships, the core `upscale_rgba` function needs a `scale == 3` arm matching the existing `scale == 2` Lanczos3 path.

### 4. Live-preview gate predicate

`should_dispatch_live_preview` in processor.rs was not changed — it gates on the model/recipe shape at dispatch time, not on the tier. The Tier-2 gate is handled by the per-item `upscale_raw.is_some()` check in `build_preview_inputs`: when `upscale_raw` is absent (Tier-1 hasn't run yet), `build_preview_inputs` returns `None` and the tick silently retries next frame. The existing 4-test suite on `should_dispatch_live_preview` was not extended (the gate predicate itself is unchanged).

### 5. Wall-clock cost (Tier-2 dispatch on 2K upscale_raw)

Not measured in this plan — no test fixture constructs a 2K buffer and times the postprocess chain. Target is <200ms per the plan output spec; the four functions (sharpen, ai_blend, saturation, color_match) are all single-pass pixel loops with no allocation and no ORT involvement, so <200ms at 2K is expected.

### 6. Pre/post inference wiring

Extracted into `build_inference_input(input, pre_denoise, brightness_lift) -> Arc<RgbaImage>` (free function in processor.rs) and `apply_post_inference(img, brightness_lift)` (free function in processor.rs). `dispatch_upscale` calls both outside the `std::thread::spawn` (build_inference_input) and inside before sending the result (apply_post_inference). The DEFER-4 invariant (denoise outside ORT session to avoid nested rayon deadlock) is satisfied by the placement before `std::thread::spawn`.

### 7. Wall-clock cost (pre_denoise=0.5 + brightness_lift=1.0 on 2560×1920 source at X4)

Not measured — no test fixture runs the full upscale pipeline. The pre/post steps add at most two sequential full-image passes (~25 MP at 4 bytes each = ~100 MB traversal each) which should be well under 1 second on any modern CPU.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] match exhaustiveness for PreviewKind::UpscaleTier2**

- **Found during:** Task 3
- **Issue:** Two match statements in `run_preview` (the `is_subject_outline` block and the main dispatch) only covered `Mask` and `Edge`. Adding `UpscaleTier2` without handling these matches would cause a compile error.
- **Fix:** Added `PreviewKind::UpscaleTier2 => unreachable!()` to both match statements. The early-return at the top of `run_preview` guarantees UpscaleTier2 never reaches either match; the `unreachable!()` arms are the compiler-visible proof.
- **Files modified:** `live_preview.rs`
- **Commit:** f5c943b

**2. [Rule 2 - Missing] Unused import warning in live_preview test**

- **Found during:** Task 3 test run
- **Issue:** `use prunr_core::upscale::apply_sharpen` imported in a test block but not needed after the test was simplified.
- **Fix:** Removed the import.
- **Files modified:** `live_preview.rs`
- **Commit:** f5c943b (same commit — pre-commit)

**3. [Rule 2 - Missing] Needless `mut` in item.rs test**

- **Found during:** Task 3 clippy run
- **Issue:** `let mut item_without` in `cache_size_includes_upscale_raw` test — the binding was never mutated.
- **Fix:** Removed `mut`.
- **Files modified:** `item.rs`
- **Commit:** f5c943b (same commit — pre-commit)

## Self-Check: PASSED

Files exist:
- FOUND: crates/prunr-app/src/gui/item.rs
- FOUND: crates/prunr-app/src/gui/processor.rs
- FOUND: crates/prunr-app/src/gui/live_preview.rs
- FOUND: crates/prunr-app/src/gui/app.rs

Commits exist:
- FOUND: 0b1005b (Task 1)
- FOUND: 0354c16 (Task 2)
- FOUND: f5c943b (Task 3)

All workspace tests: 534 prunr-app + 330 prunr-core + 31 prunr-models + 20 prunr-runtime-install = 915 total, 0 failed.
Clippy: clean (no warnings in production or test code).
