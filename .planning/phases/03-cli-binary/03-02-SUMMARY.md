---
phase: 03-cli-binary
plan: 02
subsystem: cli
tags: [rust, clap, indicatif, bgprunr-core, ort, rayon]

# Dependency graph
requires:
  - phase: 03-01
    provides: CLI struct definitions (Cli, Commands, RemoveArgs, CliModel, LargeImagePolicy) and clap/indicatif deps
  - phase: 02-cli-binary
    provides: bgprunr-core public API (process_image, batch_process, OrtEngine, formats)

provides:
  - Full CLI execution engine: main.rs dispatch + run_remove() orchestration
  - Single-image path with stage spinner, overwrite protection, large-image policy
  - Batch path with MultiProgress (overall bar + per-image spinners), exit codes 0/1/2
  - --quiet suppresses all non-error output
  - --force bypasses overwrite protection
  - --jobs N controls rayon parallelism via batch_process

affects:
  - Phase 04 (GUI): main.rs None arm will be replaced with eframe::run_native()
  - Phase 06 (release): binary produced here is the shipped artifact

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Arc-wrapped spinner Vec for Send + Sync batch progress callback
    - load_with_policy() centralizes LargeImagePolicy dispatch before PNG encode roundtrip
    - valid_indices mapping preserves correspondence between batch_process results and original inputs
    - Output path computed via output_path() helper (single -o, --output-dir, or alongside input)

key-files:
  created: []
  modified:
    - crates/bgprunr-app/src/main.rs
    - crates/bgprunr-app/src/cli.rs

key-decisions:
  - "load_with_policy() encodes DynamicImage to PNG bytes before passing to process_image — adds a round-trip encode but keeps process_image signature as &[u8] (no core API change)"
  - "run_batch() loads all image bytes upfront, marks failures before calling batch_process — fail-fast per-image, no global abort"
  - "Both tasks implemented in a single atomic commit because run_batch() was written alongside run_single() in one edit session"

patterns-established:
  - "Arc<Vec<Option<ProgressBar>>> pattern: wrap spinner vec in Arc to pass Send + Sync closure to batch_process"
  - "valid_indices Vec maps batch_process slot index back to original args.inputs index for correct spinner update"
  - "Exit code 0/1/2 contract: 0 = all success, 1 = all failed, 2 = partial — encoded at call site in main.rs via std::process::exit"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-04, CLI-05]

# Metrics
duration: 2min
completed: 2026-04-06
---

# Phase 03 Plan 02: CLI Execution Engine Summary

**Full bgprunr remove CLI with stage-spinner single-image path, MultiProgress batch path, overwrite protection, --quiet/--force/--jobs flags, and 0/1/2 exit code contract**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-07T02:16:38Z
- **Completed:** 2026-04-07T02:18:36Z
- **Tasks:** 2 (implemented in 1 commit — both written in same edit)
- **Files modified:** 2

## Accomplishments

- Rewrote main.rs to dispatch Commands::Remove to run_remove(), printing a Phase 4 GUI stub hint when no subcommand given
- Implemented run_remove(), run_single(), run_batch(), output_path(), check_overwrite(), load_with_policy(), and stage_label() in cli.rs
- Single-image path: load -> large-image policy (downscale/process) -> OrtEngine::new -> indicatif spinner updated per ProgressStage -> write -> exit 0/1
- Batch path: MultiProgress with overall bar + per-image spinners; Arc-wrapped spinner refs satisfy Send + Sync for batch_process callback; per-image overwrite check; completion lines with elapsed time; exit 0/1/2

## Task Commits

Each task was committed atomically:

1. **Task 1 + Task 2: main.rs dispatch and full run_remove() with single and batch paths** - `d5815a2` (feat)

**Plan metadata:** (pending — created in this summary commit)

_Note: Both tasks were written in a single edit session and committed together. The plan split was logical (single then batch), but the implementation was linear and required no separate verification round._

## Files Created/Modified

- `crates/bgprunr-app/src/main.rs` - Rewrote placeholder: dispatch Commands::Remove, GUI stub hint on no args
- `crates/bgprunr-app/src/cli.rs` - Added run_remove(), all helpers, run_single(), and run_batch() after Plan 01 struct definitions

## Decisions Made

- `load_with_policy()` converts `DynamicImage` to PNG bytes via `encode_rgba_png()` before passing to `process_image()`. This adds a round-trip encode for non-large images but avoids changing the `process_image(&[u8], ...)` core API signature. Acceptable tradeoff noted in SUMMARY as planned.
- Both tasks committed in a single commit because `run_batch()` was written as part of the same contiguous edit — splitting would have required a partial-state intermediate build verification that added no value.

## Deviations from Plan

None — plan executed exactly as written. Both task implementations followed the plan's code blocks faithfully. The `ProcessResult` import was removed from the use statement (it is only needed as a return type, which Rust resolves without an explicit import when `batch_process` returns `Vec<Result<ProcessResult, CoreError>>`).

## Issues Encountered

None. Build succeeded on first attempt after removing the unnecessary `ProcessResult` import and a spurious `use ProcessResult as _;` line that would not compile.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `bgprunr remove --help` and `bgprunr --help` both work
- Binary exits 0 with GUI stub message when called with no args
- All five CLI requirements (CLI-01 through CLI-05) are met
- Phase 4 (GUI): replace the `None =>` arm in main.rs with `eframe::run_native()` call — no other changes to the CLI path required

---
*Phase: 03-cli-binary*
*Completed: 2026-04-06*
