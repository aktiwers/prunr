---
phase: 05-gui-feature-completeness
plan: 03
subsystem: ui
tags: [egui, batch-processing, rayon, sidebar, dnd, thumbnails, keyboard-nav]

# Dependency graph
requires:
  - phase: 05-01
    provides: BatchItem/BatchStatus structs, app fields (batch_items, show_sidebar, next_batch_id), theme constants for sidebar
  - phase: 05-02
    provides: settings (auto_remove_on_import, parallel_jobs), AppState, worker channel types
provides:
  - Batch sidebar with thumbnails, status icons, drag-to-reorder
  - Worker BatchProcess/BatchItemDone/BatchComplete message handling with rayon parallel inference
  - [/] keyboard navigation between batch items with wrapping
  - Tab key sidebar toggle
  - Process All button in toolbar
  - Auto-remove on import (fires handle_process_all after drop)
  - Batch-aware window title and status bar
  - 14-shortcut overlay covering all Phase 4+5 shortcuts
  - 39 unit tests (10 new batch tests)
affects: [phase-06-packaging]

# Tech tracking
tech-stack:
  added: [rayon (prunr-app Cargo.toml)]
  patterns:
    - "rayon::ThreadPoolBuilder::new().num_threads(jobs) for batch parallelism — mirrors prunr-core pattern"
    - "Each rayon worker creates its own OrtEngine::new(model, intra_threads) — no Mutex contention"
    - "sync_selected_batch_textures() bridges batch item state to app-level canvas textures"
    - "add_to_batch() migrates existing single-image to first batch slot when transitioning to batch mode"

key-files:
  created:
    - crates/prunr-app/src/gui/views/sidebar.rs
    - crates/prunr-app/src/gui/tests/batch_tests.rs
  modified:
    - crates/prunr-app/src/gui/worker.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/views/mod.rs
    - crates/prunr-app/src/gui/views/toolbar.rs
    - crates/prunr-app/src/gui/views/statusbar.rs
    - crates/prunr-app/src/gui/views/shortcuts.rs
    - crates/prunr-app/src/gui/tests/mod.rs
    - crates/prunr-app/Cargo.toml

key-decisions:
  - "rayon ThreadPoolBuilder with num_threads(jobs) used for batch — same pattern as prunr-core batch processing"
  - "add_to_batch() migrates the existing single image to batch slot 0 when first batch item is added — preserves loaded state"
  - "sync_selected_batch_textures() lazily loads source/result textures for selected batch item and syncs app-level fields for canvas"
  - "Drop handler: single file + empty batch uses single-image flow; multiple files OR existing batch uses add_to_batch() for all"
  - "BatchProcess cancel propagates via AtomicBool; on cancel each Processing item reverts to Pending"

patterns-established:
  - "Batch item status lifecycle: Pending -> Processing -> Done | Error(String)"
  - "Lazy texture loading pattern: source_texture None until item selected, then loaded via sync_selected_batch_textures()"
  - "DnD reorder: dnd_set_drag_payload(i) + dnd_release_payload::<usize>() with post-iteration swap to avoid borrow conflict"

requirements-completed: [BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05, BATCH-06, UX-05]

# Metrics
duration: 35min
completed: 2026-04-07
---

# Phase 05 Plan 03: Batch Sidebar, Worker Extension, and Queue Management Summary

**Rayon-parallel batch processing with egui sidebar queue, DnD thumbnail reorder, per-item result caching, and [/] keyboard navigation**

## Performance

- **Duration:** 35 min
- **Started:** 2026-04-07T15:30:00Z
- **Completed:** 2026-04-07T16:05:00Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Extended worker.rs with BatchProcess/BatchItemDone/BatchComplete — rayon thread pool creates one OrtEngine per item, sends results as they complete
- Wired full batch workflow in app.rs: multi-drop populates queue, sidebar auto-shows at 2+ items, [/] navigation with wrapping, Tab toggle, auto-remove on import
- Created sidebar.rs with thumbnail rendering, status icon overlay (○/◆/✓/✗), selection highlight, and egui DnD drag-to-reorder
- Added Process All button to toolbar, batch progress text to status bar, batch-aware window title
- Extended shortcuts overlay to 14 rows covering all Phase 4+5 shortcuts
- Added 10 batch unit tests; full workspace test suite passes (39 app tests, 34 core tests, 1 models test)

## Task Commits

Each task was committed atomically:

1. **Task 1: Worker batch extension and sidebar view** - `81066b6` (feat)
2. **Task 2: Batch wiring in app.rs — drop handler, navigation, Process All, auto-remove** - `d7ed9e0` (feat)
3. **Task 3: Shortcuts overlay update and batch unit tests** - `824f881` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `crates/prunr-app/src/gui/views/sidebar.rs` — New: thumbnail list with DnD reorder, status icons, selection highlight
- `crates/prunr-app/src/gui/worker.rs` — Extended with BatchProcess/BatchItemDone/BatchComplete + rayon parallel handler
- `crates/prunr-app/src/gui/app.rs` — add_to_batch(), handle_process_all(), sync_selected_batch_textures(); batch-aware drop handler, keyboard shortcuts, window title, sidebar panel wiring
- `crates/prunr-app/src/gui/views/toolbar.rs` — Process All button for 2+ batch items
- `crates/prunr-app/src/gui/views/statusbar.rs` — Batch progress display (N/M images processing)
- `crates/prunr-app/src/gui/views/shortcuts.rs` — 14-row shortcuts grid (was 6 rows)
- `crates/prunr-app/src/gui/views/mod.rs` — Added `pub mod sidebar`
- `crates/prunr-app/src/gui/tests/batch_tests.rs` — New: 10 batch unit tests
- `crates/prunr-app/src/gui/tests/mod.rs` — Added `mod batch_tests`
- `crates/prunr-app/Cargo.toml` — Added rayon dependency

## Decisions Made

- rayon ThreadPoolBuilder with num_threads(jobs) used for batch — mirrors prunr-core batch processing pattern exactly
- add_to_batch() migrates the existing single image to batch slot 0 when first batch item is added — existing work is preserved
- sync_selected_batch_textures() centralizes lazy loading and app-level texture sync for the canvas rendering path
- Drop handler splits on: single file + empty batch = single-image flow; otherwise batch flow for all files
- BatchProcess cancel: AtomicBool propagates; on Escape each Processing item reverts to Pending status

## Deviations from Plan

None — plan executed exactly as written. Task 1 and Task 2 were committed separately as planned despite needing both to compile (Task 1 added new WorkerResult variants; Task 2 added match arms for them).

## Issues Encountered

Task 1 alone triggered a compile error (non-exhaustive match on WorkerResult) because the new BatchItemDone/BatchComplete variants were added to worker.rs but app.rs still had the old match. This is expected for incremental delivery — Task 2 completed the match arms. Both tasks compile cleanly when applied together.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All BATCH-01 through BATCH-06 and UX-05 requirements satisfied
- Phase 05 plan 03 is the last plan in Phase 05 — phase is complete pending final verification
- Phase 06 (packaging) can begin; no blockers

---
*Phase: 05-gui-feature-completeness*
*Completed: 2026-04-07*
