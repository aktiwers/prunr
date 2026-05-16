---
phase: 32-upscale-refinement-knobs
plan: "09"
subsystem: ui
tags: [upscale, live-preview, recipe-diff, tracing, bug-fix]

requires:
  - phase: 32-upscale-refinement-knobs
    provides: Tier-2 postprocess primitives, live_preview.rs UpscaleTier2 arm, apply_toolbar_change Tier-2 detection block

provides:
  - Diagnostic tracing at four seams of the Tier-2 live-preview chain (mark_tweak, tick, run_preview, apply_completed_previews)
  - Fix for Bug #3: to_model_kind fallback produced BiRefNetLite → recipe diff returned FullPipeline not UpscaleTier2 → mark_tweak never called
  - Fix for secondary bug: applied_recipe.upscale never advanced after Tier-2 preview → infinite per-frame mark_tweak loop after drag settles
  - PreviewResult.kind + applied_tier2_knobs fields for kind-specific post-drain side-effects
  - Three boundary tests pinning the toolbar→mark_tweak→tick→drain→swap chain

affects:
  - 32-upscale-refinement-knobs (Gap-7 closed)
  - future regressions on apply_toolbar_change Tier-2 detection

tech-stack:
  added: []
  patterns:
    - "to_model_id → ModelKind::try_from for upscale model resolution (mirrors dispatch_upscale_intent)"
    - "PreviewResult carries kind + applied_tier2_knobs so apply_completed_previews can patch only the Tier-2 fields"

key-files:
  created:
    - .planning/phases/32-upscale-refinement-knobs/32-09-DIAGNOSTIC.md
  modified:
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/live_preview.rs

key-decisions:
  - "Use to_model_id() → ModelKind::try_from() in Tier-2 detection block (matches dispatch_upscale_intent) instead of to_model_kind() with BiRefNetLite fallback"
  - "Add PreviewResult.kind (PreviewKind) to distinguish UpscaleTier2 results in apply_completed_previews without a separate channel"
  - "Patch only the 4 Tier-2 fields on applied_recipe.upscale (not replace the whole UpscaleRecipe) to avoid drifting Tier-1 fields"

requirements-completed: ["Gap-7"]

duration: 45min
completed: 2026-05-16
---

# Phase 32 Plan 09: Tier-2 Live-Preview Bug #3 Diagnosis and Fix Summary

**Bug #3 closed: `to_model_kind()` returning None for upscale variants caused `resolve_tier` to see a model mismatch (Some(RealEsrganX4Plus) vs None) and return FullPipeline instead of UpscaleTier2, silently gating out all Tier-2 slider live-preview dispatches.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-16T00:00:00Z
- **Completed:** 2026-05-16T00:45:00Z
- **Tasks:** 2
- **Files modified:** 2 + 1 planning doc created

## Accomplishments

- Identified the root cause via static analysis before running code: `SettingsModel::to_model_kind()` returns `None` for `RealEsrganUpscale`/`Nomos8kUpscale`; the `BiRefNetLite` fallback produced `upscale.model = None` in `new_recipe` while `applied_recipe.upscale.model = Some(RealEsrganX4Plus)` from dispatch time — `only_tier2_changed` returned false immediately
- Fixed the model-kind derivation with the same `to_model_id() → ModelKind::try_from()` chain used in `dispatch_upscale_intent`
- Fixed the secondary bug: `apply_completed_previews` was not advancing `applied_recipe.upscale` Tier-2 fields, so `resolve_tier` always saw a diff and `mark_tweak` re-fired every frame after drag settled; added `PreviewResult.kind` + `applied_tier2_knobs` to close the loop
- Added four structured `tracing::debug!` seams (12 total log calls) covering every link in the chain — these stay in the tree for future regression diagnosis
- Landed 3 boundary tests pinning the contract: mismatch test, correct-model test, all-four-knobs test

## Task Commits

1. **Task 1: Instrument Tier-2 live-preview chain** - `3153780` (debug)
2. **Task 2: Fix model-kind derivation + boundary tests** - `21d68d2` (fix)

## Files Created/Modified

- `crates/prunr-app/src/gui/app.rs` — Fixed `to_model_kind()` → `to_model_id() + try_from()` in Tier-2 detection; added Seam 1 (upscale_tier2_check / upscale_tier2_resolve / upscale_tier2_mark) and Seam 4 (upscale_tier2_apply) tracing; `apply_completed_previews` now patches applied_recipe.upscale Tier-2 fields; 3 new boundary tests
- `crates/prunr-app/src/gui/live_preview.rs` — Added `kind: PreviewKind` + `applied_tier2_knobs: Option<UpscaleTier2Knobs>` to `PreviewResult`; Seam 2 (upscale_tier2_mark_tweak / upscale_tier2_tick_ready / upscale_tier2_tick_no_snapshot) and Seam 3 (upscale_tier2_run_start / upscale_tier2_run_done / upscale_tier2_run_abort) tracing in `mark_tweak`, `tick`, and `run_preview`
- `.planning/phases/32-upscale-refinement-knobs/32-09-DIAGNOSTIC.md` — Captured diagnostic findings from static analysis

## Decisions Made

- Used `to_model_id() → ModelKind::try_from()` rather than a direct model-kind mapping to mirror exactly what `dispatch_upscale_intent` does — single source of truth for this conversion
- Added `PreviewResult.kind` (the `PreviewKind` variant) rather than a separate "is_upscale_tier2" bool — kind is already meaningful for callers and doesn't add dead fields for non-upscale results
- Patched only the 4 Tier-2 fields on `applied_recipe.upscale` (not replace the full `UpscaleRecipe`) to leave Tier-1 fields (model, output_scale, pre_denoise, brightness_lift) unchanged — replacing would require shipping the entire UpscaleRecipe through the rayon boundary unnecessarily

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Secondary bug: applied_recipe.upscale never advanced after Tier-2 preview**
- **Found during:** Task 1 (instrumentation analysis)
- **Issue:** `apply_completed_previews` updated `applied_recipe.mask` but not `applied_recipe.upscale` Tier-2 fields. After a Tier-2 preview landed, the next frame's `apply_toolbar_change` still saw a diff → `mark_tweak` re-fired → infinite 150ms dispatch loop after drag settled
- **Fix:** Added `PreviewResult.applied_tier2_knobs: Option<UpscaleTier2Knobs>` carrying the 4 knob values at dispatch time. `apply_completed_previews` patches `applied.upscale.sharpen_bits`, `ai_blend_bits`, `saturation_bits`, `color_match` when `applied_tier2_knobs.is_some()`
- **Files modified:** `crates/prunr-app/src/gui/live_preview.rs`, `crates/prunr-app/src/gui/app.rs`
- **Committed in:** `21d68d2` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — secondary bug discovered during Task 1 analysis)
**Impact on plan:** Secondary fix is a correctness requirement — without it the primary fix would have left an infinite preview loop. No scope creep.

## Issues Encountered

- `tier2_chain_model_kind_mismatch_never_routes_to_upscale_tier2` initially asserted `UpscaleRerun` but the actual result is `FullPipeline` (because `inference.model` also differs when model kinds differ — the FullPipeline gate at `old.inference.model != new.inference.model` fires before the upscale comparison). Fixed the assertion to `assert_ne!(…, UpscaleTier2)` which is the correct boundary: the broken path must NOT produce UpscaleTier2, regardless of which non-Tier2 tier it returns.

## Next Phase Readiness

- Gap-7 (Bug #3) closed: dragging sharpen/ai_blend/saturation/color_match sliders on a Tier-1 upscale result now produces visible canvas updates within ~150 ms (DEBOUNCE)
- Diagnostic tracing remains in tree — future Tier-2 regressions produce an immediately greppable log transcript
- No open blockers from this plan

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-16*
