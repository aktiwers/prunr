---
phase: 30-upscale-tab-v1
plan: 09
subsystem: ui
tags: [egui, toolbar, chain-mode, upscale, intent-pattern]

# Dependency graph
requires:
  - phase: 30-06
    provides: SettingsModel::is_upscale() predicate and upscale model variants
  - phase: 30-07
    provides: upscale toolbar row and render_model_dropdown wiring
provides:
  - auto_chain_on intent field on ToolbarChange
  - resolve_auto_chain_on pure helper with boundary tests
  - BatchItem::has_result() method
  - apply_toolbar_change gates chain_mode flip on item.has_result()
affects: [30-11, any plan touching chain_mode logic]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "View emits intent on ToolbarChange; app decides whether to apply — Phase 17 view→app intent pattern"
    - "Pure helper extraction for testable chain-mode logic: resolve_auto_chain_on(signal, has_result, current)"

key-files:
  created: []
  modified:
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/item.rs

key-decisions:
  - "Pure helper extraction used for test surface — resolve_auto_chain_on avoids PrunrApp harness, three boundary tests cover all three scenarios"
  - "has_result() added to BatchItem in item.rs (result_rgba.is_some()) — existing usage pattern confirmed field name"
  - "auto_chain_on fires only on upscale-entry transition (new is_upscale() && !prev.is_upscale()), never on every frame while upscale is selected"
  - "No inverse flip-off: switching away from upscale leaves chain_mode unchanged, confirmed by absence of any old_model.is_upscale() branch in apply_toolbar_change"

patterns-established:
  - "auto_chain_on: view-side intent sets flag; app-side gates application on per-item state"

requirements-completed: ["Criterion 9"]

# Metrics
duration: 12min
completed: 2026-05-14
---

# Phase 30 Plan 09: Chain-Mode Auto-On Intent Summary

**`auto_chain_on` ToolbarChange intent auto-sets chain_mode when switching to upscale with a prior result, gated by pure helper `resolve_auto_chain_on` with three boundary tests**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-14T22:30:00Z
- **Completed:** 2026-05-14T22:42:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Added `auto_chain_on: bool` to `ToolbarChange` struct with correct Default
- Wired upscale-entry detection in `render_model_dropdown` (fires only when `new.is_upscale() && !prev.is_upscale()`)
- Added `BatchItem::has_result()` predicate (wraps `result_rgba.is_some()`)
- Added `resolve_auto_chain_on` pure helper in `app.rs` with 3 boundary tests; `apply_toolbar_change` uses it
- 462 tests pass, clippy clean, workspace builds clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Add auto_chain_on intent + wire view and app** - `1c3c52b` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `crates/prunr-app/src/gui/views/adjustments_toolbar.rs` - `auto_chain_on` field + Default + view trigger + test
- `crates/prunr-app/src/gui/app.rs` - `resolve_auto_chain_on` pure helper + 3 tests + `apply_toolbar_change` handler
- `crates/prunr-app/src/gui/item.rs` - `BatchItem::has_result()` method

## Decisions Made
- Pure helper extraction chosen over PrunrApp harness — `resolve_auto_chain_on(signal, has_result, current)` is a free function with zero dependencies, cleanly testable, and called inline in `apply_toolbar_change`.
- `has_result()` location: `item.rs` alongside `cache_size()` and other pure item predicates. No new file needed.
- View-side test added to existing `toolbar_change_default_is_empty` module: asserts `auto_chain_on` starts false.

## Deviations from Plan

None — plan executed exactly as written. The pure-helper approach was the plan's preferred option and was straightforward to implement.

## Issues Encountered
None.

## Self-Check

- [x] `crates/prunr-app/src/gui/views/adjustments_toolbar.rs` exists and contains `auto_chain_on`
- [x] `crates/prunr-app/src/gui/app.rs` exists and contains `resolve_auto_chain_on`
- [x] `crates/prunr-app/src/gui/item.rs` exists and contains `has_result`
- [x] Commit `1c3c52b` verified in git log

## Self-Check: PASSED

## Next Phase Readiness
- `auto_chain_on` intent is wired end-to-end; chain mode auto-sets silently when switching to upscale with a prior result
- No visible UI indicator for chain mode in this phase (deferred per UI-SPEC scope boundary)
- 30-11 (upscale dispatch wiring) can rely on chain_mode being correctly set

---
*Phase: 30-upscale-tab-v1*
*Completed: 2026-05-14*
