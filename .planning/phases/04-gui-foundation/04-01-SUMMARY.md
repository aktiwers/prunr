---
phase: 04-gui-foundation
plan: 01
subsystem: ui
tags: [egui, eframe, arboard, rfd, worker-thread, state-machine, atomic, mpsc]

requires:
  - phase: 02-inference-core
    provides: OrtEngine, process_image, ProcessResult, ProgressStage, ModelKind
  - phase: 03-cli-integration
    provides: prunr-app crate structure, cli.rs, main.rs

provides:
  - gui/state.rs with AppState enum (Empty/Loaded/Processing/Done)
  - gui/worker.rs with WorkerMessage/WorkerResult and spawn_worker()
  - gui/theme.rs with full UI-SPEC color, spacing, layout constants
  - gui/mod.rs with module declarations and test wiring
  - gui/tests/ with state_tests (2 passing), input_tests stub, clipboard_tests stub
  - prunr_app lib target enabling --lib test runs

affects: [04-gui-foundation plan 02, 04-gui-foundation plan 03]

tech-stack:
  added: [egui 0.34, eframe 0.34, arboard 3.6, rfd 0.15]
  patterns:
    - Worker thread owns OrtEngine per ProcessImage invocation (no shared state)
    - mpsc channel pair for bidirectional GUI-worker communication
    - Arc<AtomicBool> cancel flag threaded through ProcessImage message
    - ctx.request_repaint() called after progress AND after final result
    - lib.rs + [lib] target added to binary crate for --lib test compatibility

key-files:
  created:
    - crates/prunr-app/src/gui/mod.rs
    - crates/prunr-app/src/gui/state.rs
    - crates/prunr-app/src/gui/worker.rs
    - crates/prunr-app/src/gui/theme.rs
    - crates/prunr-app/src/gui/tests/mod.rs
    - crates/prunr-app/src/gui/tests/state_tests.rs
    - crates/prunr-app/src/gui/tests/input_tests.rs
    - crates/prunr-app/src/gui/tests/clipboard_tests.rs
    - crates/prunr-app/src/lib.rs
  modified:
    - crates/prunr-app/Cargo.toml

key-decisions:
  - "lib.rs + [lib] section added to prunr-app so cargo test --lib works; plan used --lib flag but crate was binary-only"
  - "OrtEngine::new(model, 1) used in worker — plan showed 1-arg call but actual signature requires intra_threads param"
  - "Worker creates OrtEngine per ProcessImage invocation (consistent with Phase 2/3 each-worker-creates-own pattern)"

patterns-established:
  - "AppState enum drives all UI rendering decisions in Plan 02"
  - "WorkerResult::Cancelled returned when cancel flag set before final result send"
  - "Theme constants are compile-time Color32 values, not runtime config"

requirements-completed: [UX-03]

duration: 6min
completed: 2026-04-07
---

# Phase 4 Plan 01: GUI Foundation Modules Summary

**egui worker-thread architecture with AtomicBool cancel, mpsc channels, AppState machine, and UI-SPEC theme constants**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-07T07:31:12Z
- **Completed:** 2026-04-07T07:37:11Z
- **Tasks:** 3 (Task 0 + Task 1 + Task 2)
- **Files modified:** 10

## Accomplishments
- AppState enum with 4 variants and Default impl, verified by 2 passing state tests
- spawn_worker() returning (Sender<WorkerMessage>, Receiver<WorkerResult>) with named thread, cancel via Arc<AtomicBool>, and ctx.request_repaint() for immediate UI updates
- Full UI-SPEC color palette (11 Color32 constants) plus spacing scale, layout dims, typography sizes, and overlay/checkerboard constants in theme.rs
- Test scaffolding in place: state_tests pass now; input_tests and clipboard_tests are stubs ready for Plan 02

## Task Commits

1. **Task 0: Create test scaffolding** - `db9d8b0` (test)
2. **Task 1: Add GUI deps and gui/mod.rs skeleton** - `2b6970c` (feat)
3. **Task 2: Implement state.rs, worker.rs, theme.rs** - `ee9f0fd` (feat)

**Plan metadata:** (created after this summary)

## Files Created/Modified
- `crates/prunr-app/src/gui/mod.rs` - Module declarations for state, worker, theme + test module
- `crates/prunr-app/src/gui/state.rs` - AppState enum (Empty/Loaded/Processing/Done) with Default impl
- `crates/prunr-app/src/gui/worker.rs` - WorkerMessage/WorkerResult enums, spawn_worker() with cancel support
- `crates/prunr-app/src/gui/theme.rs` - All UI-SPEC color/spacing/layout constants as Color32 and f32
- `crates/prunr-app/src/gui/tests/mod.rs` - Test module declarations
- `crates/prunr-app/src/gui/tests/state_tests.rs` - 2 passing AppState tests
- `crates/prunr-app/src/gui/tests/input_tests.rs` - Stub for Plan 02 keyboard tests
- `crates/prunr-app/src/gui/tests/clipboard_tests.rs` - Stub for Plan 02 clipboard tests
- `crates/prunr-app/src/lib.rs` - Exposes gui module as lib target
- `crates/prunr-app/Cargo.toml` - Added egui, eframe, arboard, rfd deps + [lib] section

## Decisions Made
- Added `[lib]` section to Cargo.toml and `lib.rs` so `cargo test --lib` works. The plan specified `--lib` flag but prunr-app was binary-only. Adding a lib target is the standard Rust solution (no API changes needed).
- `OrtEngine::new(model, 1)` used in worker.rs — the plan showed a 1-arg call but the actual engine signature requires `intra_threads: usize`. Using `1` matches single-image GUI use.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected OrtEngine::new() call signature in worker.rs**
- **Found during:** Task 2 (worker.rs implementation)
- **Issue:** Plan showed `OrtEngine::new(model)` with 1 argument but actual signature is `OrtEngine::new(model, intra_threads: usize)`
- **Fix:** Used `OrtEngine::new(model, 1)` — single thread appropriate for GUI use (no rayon batch context)
- **Files modified:** crates/prunr-app/src/gui/worker.rs
- **Verification:** `cargo check -p prunr-app --features dev-models` exits 0
- **Committed in:** ee9f0fd (Task 2 commit)

**2. [Rule 3 - Blocking] Added lib target so --lib test flag works**
- **Found during:** Task 2 (verification step)
- **Issue:** Plan's verification command `cargo test --lib gui::tests::state_tests` fails on binary-only crate
- **Fix:** Added `[lib]` section to Cargo.toml and `src/lib.rs` declaring `pub mod gui`
- **Files modified:** crates/prunr-app/Cargo.toml, crates/prunr-app/src/lib.rs
- **Verification:** `cargo test -p prunr-app --features dev-models --lib gui::tests::state_tests` passes 2 tests
- **Committed in:** ee9f0fd (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug fix, 1 blocking)
**Impact on plan:** Both fixes required for compilation and test execution. No scope creep.

## Issues Encountered
- Cargo registry permission warning (`failed to remove file .cache/it/oa/itoa — Permission denied`) on every cargo invocation — pre-existing system issue, not caused by this plan.

## Next Phase Readiness
- gui/state.rs, gui/worker.rs, gui/theme.rs all compile cleanly
- spawn_worker() is ready for PrunrApp struct integration in Plan 02
- Test targets exist and state tests pass; input/clipboard stubs ready for Plan 02 population
- No blockers for Plan 02

---
*Phase: 04-gui-foundation*
*Completed: 2026-04-07*
