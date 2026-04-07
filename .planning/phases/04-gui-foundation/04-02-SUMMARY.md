---
phase: 04-gui-foundation
plan: 02
subsystem: ui
tags: [egui, eframe, arboard, rfd, desktop-gui, drag-and-drop, clipboard, keyboard-shortcuts]

# Dependency graph
requires:
  - phase: 04-01
    provides: AppState enum, spawn_worker() with mpsc channels + AtomicBool cancel, theme constants, test scaffolding
  - phase: 03
    provides: bgprunr-core process_image(), encode_rgba_png(), ModelKind, ProcessResult
provides:
  - Complete egui GUI application: BgPrunrApp struct with eframe::App impl
  - All 4 view modules: toolbar, canvas, statusbar, shortcuts overlay
  - Keyboard shortcuts via bool-flag pattern (Ctrl/Cmd+O/R/S/C, Escape, ?)
  - Drag-and-drop image loading
  - File open/save dialogs via rfd
  - Clipboard copy via arboard
  - Cancel inference via AtomicBool cancel_flag
  - Window title update with filename
  - Checkerboard transparency background for Done state
  - 7 unit tests covering cancel flag, open path error handling, clipboard no-panic
affects: [04-03, phase-05, phase-06]

# Tech tracking
tech-stack:
  added: [image crate added to bgprunr-app Cargo.toml for direct texture loading]
  patterns:
    - "Bool-flag keyboard shortcut pattern: detect in ctx.input() closure, act AFTER closure returns (avoids blocking rfd inside egui lock)"
    - "eframe 0.34 logic()+ui() split: channel polling in logic(), widget rendering in ui()"
    - "Temporary status text with 2-second auto-clear via status_set_at: Option<Instant>"
    - "Source texture loaded lazily in logic() when source_bytes present but source_texture is None"

key-files:
  created:
    - crates/bgprunr-app/src/gui/app.rs
    - crates/bgprunr-app/src/gui/views/mod.rs
    - crates/bgprunr-app/src/gui/views/toolbar.rs
    - crates/bgprunr-app/src/gui/views/canvas.rs
    - crates/bgprunr-app/src/gui/views/statusbar.rs
    - crates/bgprunr-app/src/gui/views/shortcuts.rs
  modified:
    - crates/bgprunr-app/src/gui/mod.rs
    - crates/bgprunr-app/src/gui/tests/input_tests.rs
    - crates/bgprunr-app/src/gui/tests/clipboard_tests.rs
    - crates/bgprunr-app/src/main.rs
    - crates/bgprunr-app/Cargo.toml

key-decisions:
  - "egui 0.34 API renames applied: Panel::top/bottom (not TopBottomPanel), exact_size (not exact_height), corner_radius (not rounding), CornerRadius (not Rounding), global_style/set_global_style (not style/set_style)"
  - "image crate added to bgprunr-app deps for direct RgbaImage loading without going through bgprunr-core"
  - "BgPrunrApp::new_for_test() constructor added for unit tests since eframe::CreationContext unavailable outside eframe::run_native"
  - "cancel_flag made pub(crate) to allow test assertions on AtomicBool state"
  - "Source texture loaded lazily in logic() rather than at open time since egui Context unavailable in handle_open_path()"

patterns-established:
  - "Pattern 1: Bool-flag shortcuts — set flags inside ctx.input() closure, call handlers (rfd/arboard) AFTER closure returns"
  - "Pattern 2: Temporary status — set_temporary_status() records Instant; logic() clears after 2 seconds"
  - "Pattern 3: Lazy texture loading — check for None texture in logic() each frame, decode+upload if source bytes present"

requirements-completed: [LOAD-01, LOAD-02, OUT-01, OUT-02, UX-01, UX-03, UX-04]

# Metrics
duration: 4min
completed: 2026-04-07
---

# Phase 4 Plan 02: GUI Foundation — BgPrunrApp Summary

**Full egui application with toolbar/canvas/statusbar/shortcuts views, bool-flag keyboard shortcuts, drag-and-drop, file dialogs, clipboard copy, and cancel via AtomicBool**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-07T07:40:14Z
- **Completed:** 2026-04-07T07:44:00Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- BgPrunrApp struct with full eframe 0.34 logic()+ui() split — no deprecated update() method
- All 4 view modules render toolbar, canvas (drop zone + image + checkerboard), status bar, and shortcuts overlay
- Keyboard shortcuts wired via bool-flag pattern; rfd file dialogs never called inside ctx.input() closure
- 7 unit tests pass: cancel flag, open path error, clipboard no-panic scenarios

## Task Commits

Each task was committed atomically:

1. **Task 1: BgPrunrApp struct, eframe boilerplate, gui::run()** - `6f952fa` (feat)
2. **Task 2: All view modules and test stubs** - `13116a3` (feat)

**Plan metadata:** (committed with docs after this summary)

## Files Created/Modified
- `crates/bgprunr-app/src/gui/app.rs` - BgPrunrApp struct, eframe::App impl (logic+ui split, all handlers)
- `crates/bgprunr-app/src/gui/mod.rs` - Updated with app, views modules and run() function
- `crates/bgprunr-app/src/main.rs` - None arm now calls gui::run()
- `crates/bgprunr-app/src/gui/views/mod.rs` - Views module root
- `crates/bgprunr-app/src/gui/views/toolbar.rs` - Open/Remove BG/Save/Copy buttons + model selector ComboBox
- `crates/bgprunr-app/src/gui/views/canvas.rs` - Drop zone, image display, processing overlay, checkerboard
- `crates/bgprunr-app/src/gui/views/statusbar.rs` - State text, progress bar, dimensions, backend badge
- `crates/bgprunr-app/src/gui/views/shortcuts.rs` - Centered modal with 6 platform-aware shortcuts
- `crates/bgprunr-app/src/gui/tests/input_tests.rs` - handle_cancel, open_path_error, show_shortcuts tests
- `crates/bgprunr-app/src/gui/tests/clipboard_tests.rs` - handle_copy no-panic tests
- `crates/bgprunr-app/Cargo.toml` - Added image crate dependency

## Decisions Made

- **egui 0.34 API renames:** Several methods renamed in 0.34: `TopBottomPanel` -> `Panel::top/bottom`, `exact_height` -> `exact_size`, `.rounding()` -> `.corner_radius()`, `Rounding` -> `CornerRadius`, `ctx.style()` -> `ctx.global_style()`, `ctx.set_style()` -> `ctx.set_global_style()`. All applied to eliminate deprecation warnings.
- **image crate in bgprunr-app:** Added `image = { workspace = true }` to bgprunr-app Cargo.toml since app.rs needs `image::load_from_memory()` and `image::RgbaImage` for texture loading and clipboard tests, without routing through bgprunr-core.
- **BgPrunrApp::new_for_test():** Added `#[cfg(test)]` constructor that creates mock mpsc channels without eframe::CreationContext, enabling unit tests for state-machine logic methods.
- **cancel_flag pub(crate):** Made `cancel_flag: Arc<AtomicBool>` field pub(crate) so input_tests.rs can assert the AtomicBool value directly.
- **Lazy source texture:** Source texture is created in `logic()` rather than in `handle_open_path()` because the egui `Context` is not available when loading from file picker or drag-and-drop path handling.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed egui 0.34 renamed API methods**
- **Found during:** Task 1 (compile verification)
- **Issue:** Plan code used deprecated/renamed egui 0.34 APIs: `TopBottomPanel`, `exact_height`, `.rounding()`, `Rounding`, `ctx.style()`, `ctx.screen_rect()`, `Margin::same(f32)`
- **Fix:** Updated all call sites to use new names; `Margin::same()` takes `i8` so cast `SPACE_MD as i8`
- **Files modified:** app.rs, toolbar.rs, shortcuts.rs
- **Verification:** `cargo check` exits 0 with only unused-constant warnings
- **Committed in:** 6f952fa, 13116a3

**2. [Rule 1 - Bug] Added image crate to bgprunr-app Cargo.toml**
- **Found during:** Task 1 (compile verification)
- **Issue:** `app.rs` uses `image::load_from_memory()` and `image::RgbaImage` directly, but `image` was not listed in `bgprunr-app/Cargo.toml` dependencies (only available transitively via bgprunr-core)
- **Fix:** Added `image = { workspace = true }` to bgprunr-app Cargo.toml
- **Files modified:** Cargo.toml, Cargo.lock
- **Committed in:** 6f952fa

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs from API rename + missing direct dep)
**Impact on plan:** Both fixes necessary for compilation. No scope changes.

## Issues Encountered
- egui 0.34 has several renamed APIs vs what the plan code assumed. All renames resolved by checking compiler deprecation messages.

## Next Phase Readiness
- GUI application fully wired: run `cargo run -p bgprunr-app --features dev-models` to launch
- Phase 04-03 (integration/polish) can now add image zoom/pan, window state persistence, and error display improvements
- All 7 unit tests passing for core state machine logic

---
*Phase: 04-gui-foundation*
*Completed: 2026-04-07*
