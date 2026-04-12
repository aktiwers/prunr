---
phase: 07-iterative-processing
plan: 02
subsystem: gui, worker, cli
tags: [chain-mode, iterative-processing, egui, ort, image-pipeline]

# Dependency graph
requires:
  - phase: 07-iterative-processing plan 01
    provides: history stack with result_rgba on BatchItem
provides:
  - chain_mode setting (Settings.chain_mode bool, persisted)
  - chain_mode toggle in General settings tab
  - chain_input plumbing through WorkerMessage and process_items
  - worker chain_input handling for all three LineMode arms
  - --chain CLI flag for multi-pass scripting
affects: [07-iterative-processing plan 03]

# Tech tracking
tech-stack:
  added: []
  patterns: [chain input via Option<Arc<RgbaImage>> in worker message tuple, Cow<[u8]> for zero-copy fallback to source bytes, encode_rgba_png for chain input to process_image_with_mask]

key-files:
  created: []
  modified:
    - crates/prunr-app/src/gui/settings.rs
    - crates/prunr-app/src/gui/views/settings.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/worker.rs
    - crates/prunr-app/src/cli.rs

key-decisions:
  - "chain_input encoded to PNG bytes via encode_rgba_png for Off mode since process_image_with_mask takes &[u8]"
  - "AfterBgRemoval with chain_input skips BG removal entirely and goes straight to edge detection"
  - "Cow<[u8]> used in Off arm to avoid unnecessary clone of source_bytes when not chaining"

patterns-established:
  - "Chain input pattern: Option<Arc<RgbaImage>> in WorkerMessage tuple, converted per-arm in worker"

requirements-completed: [ITER-01, ITER-04]

# Metrics
duration: 5min
completed: 2026-04-12
---

# Phase 7 Plan 2: Chain Mode Summary

**Chain mode toggle wired through Settings, GUI, worker pipeline, and CLI -- process results feed back as input for iterative effect stacking**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-12T23:45:53Z
- **Completed:** 2026-04-12T23:50:37Z
- **Tasks:** 5
- **Files modified:** 5

## Accomplishments
- Added chain_mode boolean to Settings with serde persistence and default false
- Chain mode checkbox in General settings tab with hint text explaining effect stacking
- WorkerMessage carries Option<Arc<RgbaImage>> chain_input; process_items passes result_rgba when enabled
- Worker handles chain_input in all three LineMode arms: LinesOnly uses chain image directly, AfterBgRemoval skips BG removal on chain, Off encodes chain to PNG for process_image_with_mask
- CLI --chain flag added for future multi-pass scripting workflows

## Task Commits

Each task was committed atomically:

1. **Task 1: Add chain_mode setting** - `d5b1968` (feat)
2. **Task 2: Add chain_mode toggle to General settings tab** - `4da6d71` (feat)
3. **Task 3: Add chain_input to WorkerMessage and pass result data** - `2cfbb70` (feat)
4. **Task 4: Worker uses chain_input when provided** - `19bd04a` (feat)
5. **Task 5: Add --chain CLI flag** - `a64cec3` (feat)

## Files Created/Modified
- `crates/prunr-app/src/gui/settings.rs` - Added chain_mode: bool field with default false
- `crates/prunr-app/src/gui/views/settings.rs` - Chain mode checkbox in General tab, reset handler updated
- `crates/prunr-app/src/gui/app.rs` - process_items passes result_rgba as chain_input when chain_mode enabled
- `crates/prunr-app/src/gui/worker.rs` - All three LineMode arms handle chain_input with fallback to source
- `crates/prunr-app/src/cli.rs` - --chain flag for multi-pass CLI workflows

## Decisions Made
- chain_input encoded to PNG bytes via encode_rgba_png for the Off (normal BG removal) mode, since process_image_with_mask takes &[u8] encoded image bytes
- AfterBgRemoval with chain_input skips BG removal entirely and runs edge detection directly on the chain image
- Used Cow<[u8]> in the Off arm to avoid cloning source_bytes when not chaining

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added chain_mode to General tab reset handler**
- **Found during:** Task 2
- **Issue:** Reset General button would not reset chain_mode to default
- **Fix:** Added `app.settings.chain_mode = defaults.chain_mode;` to the reset handler
- **Files modified:** crates/prunr-app/src/gui/views/settings.rs
- **Verification:** Reset General button now resets chain_mode to false
- **Committed in:** 4da6d71 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Necessary for correct reset behavior. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Chain mode fully wired for iterative processing
- Plan 07-03 can build on this to add UI feedback and refinements

---
## Self-Check: PASSED

All 5 modified files exist on disk. All 5 task commits verified in git log.

*Phase: 07-iterative-processing*
*Completed: 2026-04-12*
