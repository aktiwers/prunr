---
phase: 05-gui-feature-completeness
plan: 01
subsystem: ui
tags: [egui, zoom, pan, before-after, serde, settings, canvas, state-machine]

# Dependency graph
requires:
  - phase: 04-gui-foundation
    provides: BgPrunrApp struct, canvas.rs, statusbar.rs, app.rs, worker, theme constants
provides:
  - Settings struct with serde derives and Default impl
  - BatchItem and BatchStatus types for multi-image workflow
  - AppState::Animating variant
  - Phase 5 theme constants (SIDEBAR_WIDTH, ZOOM_MIN/MAX/STEP, ANIM_*, SETTINGS_DIALOG_*, sidebar colors)
  - BgPrunrApp zoom/pan/before-after/animation/batch/settings fields
  - Canvas zoom-to-cursor (scroll wheel), Space+drag pan, Ctrl+0/Ctrl+1 fit/actual-size
  - B-key before/after toggle with Original/Result indicator label
  - Status bar zoom percentage display
affects: [05-02, 05-03, downstream phase 5 plans]

# Tech tracking
tech-stack:
  added: [serde 1.0 (derive), serde_json 1.0, num_cpus 1 (already in workspace)]
  patterns:
    - pending_fit_zoom/pending_actual_size flags decouple keyboard shortcut handling in logic() from canvas rendering
    - zoom/pan state lives on BgPrunrApp; canvas.rs mutates it directly via &mut BgPrunrApp
    - BatchItem per-file isolation pattern for multi-image support

key-files:
  created:
    - crates/bgprunr-app/src/gui/settings.rs
    - crates/bgprunr-app/src/gui/tests/zoom_pan_tests.rs
  modified:
    - crates/bgprunr-app/src/gui/state.rs
    - crates/bgprunr-app/src/gui/theme.rs
    - crates/bgprunr-app/src/gui/app.rs
    - crates/bgprunr-app/src/gui/mod.rs
    - crates/bgprunr-app/src/gui/views/canvas.rs
    - crates/bgprunr-app/src/gui/views/statusbar.rs
    - crates/bgprunr-app/src/gui/tests/mod.rs
    - crates/bgprunr-app/src/gui/tests/state_tests.rs
    - Cargo.toml
    - crates/bgprunr-app/Cargo.toml

key-decisions:
  - "canvas::render takes &mut BgPrunrApp so zoom/pan state is mutated during the render pass itself"
  - "pending_fit_zoom and pending_actual_size flags bridge logic()-side keyboard detection to canvas-side geometry computation"
  - "zoom cursor-centering uses cursor_rel / old_zoom - cursor_rel / new_zoom + pan formula (pan-adjusted)"
  - "fit_zoom() caps at 1.0 to never upscale smaller images beyond native pixels"

patterns-established:
  - "Pending-flag pattern: logic() sets flag, canvas.rs consumes it — avoids canvas_rect dependency in logic()"
  - "canvas::render signature uses &mut App when rendering requires app state mutation"

requirements-completed: [VIEW-01, VIEW-02, VIEW-03, VIEW-04, VIEW-05]

# Metrics
duration: 25min
completed: 2026-04-07
---

# Phase 05 Plan 01: Foundation Types, Zoom/Pan Canvas, and Before/After Toggle Summary

**egui canvas with cursor-centered scroll-wheel zoom, Space+drag pan, B-key before/after toggle, and serde-enabled Settings type for Phase 5 feature foundation**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments
- Full zoom/pan canvas: scroll-wheel zooms centered on cursor, Space+drag pans, Ctrl+0 fits to window, Ctrl+1 shows 1:1 with previous-zoom restore on second press
- Before/after B-key toggle in Done state with "Original"/"Result" indicator label; checkerboard renders behind transparent result
- Settings struct with serde derives, SettingsModel↔ModelKind From impls, Default using num_cpus; BatchItem/BatchStatus types; AppState::Animating variant
- Zoom percentage displayed in status bar right side; all Phase 5 theme constants defined

## Task Commits

1. **Task 1: Foundation types, deps, theme constants, and app fields** - `c3d4f2d` (feat)
2. **Task 2: Canvas zoom/pan/before-after rendering and status bar zoom display** - `3d2cf69` (feat)
3. **Task 3: Unit tests for zoom/pan/before-after state logic** - `afc59e9` (test)

## Files Created/Modified
- `crates/bgprunr-app/src/gui/settings.rs` — Settings, SettingsModel with serde and Default
- `crates/bgprunr-app/src/gui/state.rs` — Added Animating variant to AppState
- `crates/bgprunr-app/src/gui/theme.rs` — Phase 5 constants: SIDEBAR_WIDTH, ZOOM_*, ANIM_*, SETTINGS_DIALOG_*, sidebar/status colors
- `crates/bgprunr-app/src/gui/app.rs` — New fields (zoom, pan_offset, show_original, batch_items, settings, etc.); B/Ctrl+0/Ctrl+1 shortcuts; Animating cancel handling
- `crates/bgprunr-app/src/gui/mod.rs` — Added `pub mod settings`
- `crates/bgprunr-app/src/gui/views/canvas.rs` — Full rewrite: &mut BgPrunrApp, zoom/pan event handling, compute_img_rect, fit_zoom, render_done with before/after
- `crates/bgprunr-app/src/gui/views/statusbar.rs` — Zoom percentage display; Animating match arm
- `crates/bgprunr-app/src/gui/tests/zoom_pan_tests.rs` — 7 new tests
- `crates/bgprunr-app/src/gui/tests/state_tests.rs` — animating_state_exists_and_is_distinct test
- `crates/bgprunr-app/src/gui/tests/mod.rs` — zoom_pan_tests module registered
- `Cargo.toml` — serde, serde_json workspace deps
- `crates/bgprunr-app/Cargo.toml` — serde, serde_json, num_cpus deps

## Decisions Made
- `canvas::render` takes `&mut BgPrunrApp` so zoom/pan mutation happens in-place during render, avoiding a separate update pass
- `pending_fit_zoom`/`pending_actual_size` flags bridge logic()-side keyboard handling to canvas-side geometry (canvas_rect is only available at render time)
- Zoom cursor-centering formula: `cursor_rel / old_zoom - cursor_rel / new_zoom + pan_offset` keeps the pixel under the cursor stationary
- `fit_zoom()` caps at `1.0` to avoid upscaling images smaller than the canvas

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Animating match arms to canvas.rs and statusbar.rs**
- **Found during:** Task 1 verification (cargo check)
- **Issue:** Adding `AppState::Animating` to state.rs caused non-exhaustive match errors in canvas.rs and statusbar.rs — both used explicit match without a wildcard
- **Fix:** Added `AppState::Animating => render_done(ui, app)` to canvas.rs match and `AppState::Animating | AppState::Done => app.status_text.clone()` to statusbar.rs match
- **Files modified:** crates/bgprunr-app/src/gui/views/canvas.rs, crates/bgprunr-app/src/gui/views/statusbar.rs
- **Verification:** cargo check passed after fix
- **Committed in:** c3d4f2d (Task 1 commit, inline with those files)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary consequence of adding the Animating variant; both files were already scheduled for modification in Task 2, handled in Task 1 instead.

## Issues Encountered
None beyond the auto-fixed Animating match exhaustion above.

## Next Phase Readiness
- All Phase 5 foundation types available: Settings, BatchItem, BatchStatus, Animating state
- Zoom/pan/before-after canvas fully functional
- Theme constants ready for sidebar (Plan 02) and settings dialog (Plan 02)
- pending_fit_zoom/pending_actual_size pattern ready for Ctrl+0/Ctrl+1 in later plans
- 16 lib tests passing

---
*Phase: 05-gui-feature-completeness*
*Completed: 2026-04-07*
