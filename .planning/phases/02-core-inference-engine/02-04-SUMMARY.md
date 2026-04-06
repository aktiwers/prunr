---
phase: 02-core-inference-engine
plan: "04"
subsystem: inference
tags: [rayon, batch, parallel, ort, thread-pool, num_cpus]

# Dependency graph
requires:
  - phase: 02-03
    provides: OrtEngine::new() and process_image() pipeline API
provides:
  - batch_process() parallel batch processing API with rayon thread pool
  - ort_intra_threads() formula for CPU oversubscription prevention
  - Per-worker OrtEngine pattern for thread-safe inference
affects:
  - bgprunr-cli
  - bgprunr-gui

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-worker OrtEngine: each rayon worker creates its own Session — sessions never shared"
    - "Index-preserve pattern: collect (idx, result) pairs then assign to pre-allocated Vec"
    - "Thread balancing: ort_intra_threads = (num_cpus / rayon_workers).max(1)"

key-files:
  created:
    - crates/bgprunr-core/src/batch.rs
  modified:
    - crates/bgprunr-core/src/lib.rs

key-decisions:
  - "Each rayon worker creates its own OrtEngine::new() — no Arc<Mutex<Session>> sharing, avoids contention"
  - "ort_intra_threads formula normalizes total thread count: num_cpus / jobs, minimum 1, prevents CPU oversubscription"
  - "Results collected as (idx, result) pairs then assigned to pre-allocated Vec to preserve input order despite rayon work-stealing"
  - "Unit tests (ort_intra_threads, empty input, bad image error path) pass without model files; integration tests gated behind dev-models feature"

patterns-established:
  - "Per-worker OrtEngine: create session inside rayon closure, not outside — never share sessions across threads"
  - "Order-preserving parallel batch: collect indexed tuples, assign to pre-allocated Vec by index"

requirements-completed:
  - CORE-02
  - CORE-03
  - CORE-05

# Metrics
duration: 2min
completed: 2026-04-06
---

# Phase 02 Plan 04: Batch Processing Summary

**batch_process() with rayon thread pool where each worker owns its OrtEngine, preventing session sharing and CPU oversubscription via num_cpus/jobs intra-thread formula**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-06T23:27:27Z
- **Completed:** 2026-04-06T23:29:02Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- batch_process() accepts image byte slices, model kind, job count, and optional indexed progress callback
- Results returned in input order via (idx, result) indexed collection pattern — safe with rayon work-stealing
- ort_intra_threads() formula (num_cpus / workers).max(1) prevents CPU oversubscription in mixed ORT+rayon environments
- 4 unit tests pass without model files; 2 integration tests correctly gated behind dev-models feature

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement batch.rs with rayon parallel dispatch** - `0f07225` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `crates/bgprunr-core/src/batch.rs` - batch_process(), ort_intra_threads(), build_batch_pool() with embedded tests
- `crates/bgprunr-core/src/lib.rs` - Added `pub use batch::batch_process` re-export

## Decisions Made
- Each rayon worker creates its own OrtEngine::new() inside the closure — avoids Arc<Mutex<Session>> sharing, removes lock contention during parallel inference
- Results collected as Vec<(usize, Result<...>)> then assigned to pre-allocated Vec by index — preserves input order despite rayon work-stealing execution order
- Unit tests scoped to avoid model dependency: ort_intra_threads formula, empty input, and bad-image error path are model-free; model-requiring tests are cfg(feature = "dev-models") gated

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Running `cargo test -p bgprunr-core batch` (without any feature flag) fails because the non-dev-models path uses `include_bytes_zstd!` which requires model files at compile time — this is a pre-existing project constraint, not caused by batch.rs. Tests run correctly with `--features dev-models`.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- batch_process() is exported from bgprunr_core public API and ready for CLI and GUI consumption
- Integration tests (test_batch_process_jobs_1_sequential, test_batch_process_preserves_order) will pass once `cargo xtask fetch-models` is run
- Phase 02-05 can proceed: batch_process() provides the parallel batch API the CLI batch queue depends on

## Self-Check: PASSED

- FOUND: crates/bgprunr-core/src/batch.rs
- FOUND: .planning/phases/02-core-inference-engine/02-04-SUMMARY.md
- FOUND: commit 0f07225 (feat(02-04): implement batch_process())

---
*Phase: 02-core-inference-engine*
*Completed: 2026-04-06*
