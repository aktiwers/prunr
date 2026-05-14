---
phase: 30-upscale-tab-v1
plan: 07
subsystem: ui
tags: [egui, toolbar, chip, upscale, progress-bar]

# Dependency graph
requires:
  - phase: 30-03
    provides: ItemSettings.upscale_model + upscale_scale fields
  - phase: 30-06
    provides: SettingsModel upscale variants + is_upscale predicate
provides:
  - upscale_chip::render_scale_chip (scale chip widget, 4x/2x discrete picker)
  - upscale_toolbar::render_upscale_row (full upscale toolbar row)
  - adjustments_toolbar::render_upscale_right_cluster (reset+preset for upscale mode)
  - ToolbarChange.upscale_scale_changed flag
  - Processor::upscale_tile_progress stub (data wired by plan 30-11)
  - is_upscale branch routing in adjustments_toolbar::render
affects:
  - 30-09 (chain-mode trigger on model change)
  - 30-11 (fills upscale_tile_progress body with real tile counter)
  - 30-12 (visual checkpoint — first human verification of upscale toolbar)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "upscale_mode branch in adjustments_toolbar::render — same pattern as inpaint_mode"
    - "source_dims + processor params added to render() for upscale-only needs"
    - "model-change hooks moved outside horizontal block so they fire for all toolbar modes"

key-files:
  created:
    - crates/prunr-app/src/gui/views/upscale_chip.rs
    - crates/prunr-app/src/gui/views/upscale_toolbar.rs
  modified:
    - crates/prunr-app/src/gui/views/mod.rs
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/processor.rs
    - crates/prunr-app/src/gui/app.rs

key-decisions:
  - "source_dims resolved in app.rs using chain_mode + result_rgba.dimensions() fallback to item.dimensions — avoids adding a BatchItem helper"
  - "render_model_dropdown and render_upscale_right_cluster promoted to pub(super) for cross-module reuse"
  - "Model-change hooks (LaMa release, brush prewarm) moved outside the horizontal block so they fire regardless of toolbar mode"
  - "Processor::upscale_tile_progress returns None stub; plan 30-11 fills the body"
  - "No per-task item_id parameter on upscale_tile_progress — plan 30-11 has freedom to add one if needed"

patterns-established:
  - "Upscale toolbar row: model dropdown -> scale chip -> conditional progress bar -> right cluster"
  - "Progress bar gated on Option<(u32,u32)> from Processor — zero height when None, no placeholder"

requirements-completed: [Criterion 5]

# Metrics
duration: 7min
completed: 2026-05-14
---

# Phase 30 Plan 07: Upscale Toolbar (Surfaces 1-3) Summary

**Upscale toolbar row with model dropdown, 4x/2x scale chip, inline tile-progress bar (stub), and reset/preset cluster — wired into adjustments_toolbar via is_upscale branch, Row 3 suppressed in upscale mode**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-14T22:31:25Z
- **Completed:** 2026-05-14T22:38:34Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- `upscale_chip::render_scale_chip` uses all canonical chip primitives (chip_button, popup_for, chip_tooltip, reset_button); UI-SPEC label/tooltip/popover strings verbatim
- `upscale_toolbar::render_upscale_row` provides the complete Surface 1 layout; progress bar gated on `processor.upscale_tile_progress()` — currently returns None (stub for plan 30-11)
- `adjustments_toolbar::render` branches on `upscale_mode`; Row 3 entirely absent when is_upscale(); all existing modes unchanged

## Task Commits

1. **Task 1: Create upscale_chip.rs** - `243c3ab` (feat)
2. **Task 2: Create upscale_toolbar.rs + stub + ToolbarChange field** - `c5dbaa1` (feat)
3. **Task 3: Wire is_upscale branch in adjustments_toolbar::render** - `a975778` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `crates/prunr-app/src/gui/views/upscale_chip.rs` — scale chip widget: chip_button + popup_for + 4x/2x selectable_label rows + reset_button
- `crates/prunr-app/src/gui/views/upscale_toolbar.rs` — render_upscale_row: model dropdown + scale chip + progress bar + right cluster
- `crates/prunr-app/src/gui/views/mod.rs` — module declarations for upscale_chip + upscale_toolbar
- `crates/prunr-app/src/gui/views/adjustments_toolbar.rs` — upscale_mode branch, render() signature extended (source_dims, processor), model-change hooks moved outside horizontal block, render_upscale_right_cluster added
- `crates/prunr-app/src/gui/processor.rs` — upscale_tile_progress() -> Option<(u32,u32)> stub
- `crates/prunr-app/src/gui/app.rs` — passes source_dims (chain-mode-aware) and &self.processor to adjustments_toolbar::render

## Decisions Made

- **source_dims path**: Resolved in `app.rs` using chain-mode logic: `if settings.chain_mode { result_rgba.dimensions() or item.dimensions } else { item.dimensions }`. No new BatchItem helper needed — app.rs already had all the fields for this.
- **ToolbarChange.upscale_scale_changed**: Added as a plain bool field matching the plan spec. Initialized to false in Default.
- **upscale_tile_progress stub**: Returns `None` with no forward-reference comment per CLAUDE.md. Plan 30-11 replaces the body.
- **Model-change hooks location**: Moved outside the `ui.horizontal` block so LaMa session release and brush prewarm fire for model changes made in the upscale toolbar row too. Behaviorally equivalent for existing modes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] render_model_dropdown and model-change hooks need to work in upscale mode**
- **Found during:** Task 3 (wiring is_upscale branch)
- **Issue:** The model-change hooks (LaMa release, inpaint prewarm) were inside the `ui.horizontal` block. Moving the entire horizontal block into an `else` branch meant the hooks wouldn't fire when upscale_mode was true.
- **Fix:** Moved model-change hooks outside both branches so they run after either the upscale or seg/inpaint row renders, reading `change.model_changed` which is set by whichever `render_model_dropdown` ran.
- **Files modified:** `adjustments_toolbar.rs`
- **Verification:** `cargo test --workspace --lib` green (459 tests)
- **Committed in:** `a975778`

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing correctness)
**Impact on plan:** Structural refactor within the single task. No scope creep; existing modes unaffected.

## Issues Encountered

None beyond the model-change hook relocation documented above.

## Self-Check: PASSED

Files confirmed:
- `crates/prunr-app/src/gui/views/upscale_chip.rs` — FOUND
- `crates/prunr-app/src/gui/views/upscale_toolbar.rs` — FOUND

Commits confirmed:
- `243c3ab` — FOUND (feat(30-07): add scale chip widget)
- `c5dbaa1` — FOUND (feat(30-07): add render_upscale_row + ToolbarChange)
- `a975778` — FOUND (feat(30-07): wire is_upscale branch)

## Next Phase Readiness

- Plan 30-09 can wire chain-mode auto-enable on model change (the model dropdown + model_changed flag are in place)
- Plan 30-11 fills `upscale_tile_progress` with real atomic counters from the dispatch path
- Plan 30-12 (visual checkpoint) can now smoke-test the upscale toolbar row end-to-end

---
*Phase: 30-upscale-tab-v1*
*Completed: 2026-05-14*
