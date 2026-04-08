---
phase: 03-cli-binary
plan: 01
subsystem: cli
tags: [clap, indicatif, rust, cli, pipeline]

# Dependency graph
requires:
  - phase: 02-inference-engine
    provides: process_image, batch_process, OrtEngine, ModelKind, ProgressStage, CoreError, formats
provides:
  - Cli, Commands, RemoveArgs, CliModel, LargeImagePolicy clap derive structs in prunr-app/src/cli.rs
  - clap 4.5 and indicatif 0.17 deps in prunr-app/Cargo.toml
  - process_image_unchecked in prunr-core for --large-image=process bypass
  - dev-models feature propagation in prunr-app
affects: [03-02-executor, cli, batch]

# Tech tracking
tech-stack:
  added: [clap 4.5 (derive, ValueEnum), indicatif 0.17]
  patterns: [clap derive macros for CLI struct, ValueEnum for enum flags, From impl for core type conversion, process_image_from_decoded private helper for DRY pipeline]

key-files:
  created:
    - crates/prunr-app/src/cli.rs
  modified:
    - crates/prunr-app/Cargo.toml
    - crates/prunr-app/src/main.rs
    - crates/prunr-core/src/pipeline.rs
    - crates/prunr-core/src/lib.rs

key-decisions:
  - "dev-models feature added to prunr-app (propagates prunr-core/dev-models) so cargo check works without OOM from model embedding proc macro"
  - "process_image refactored: shared logic extracted to private process_image_from_decoded, process_image_unchecked bypasses check_large_image guard"
  - "CliModel (not ModelKind) defined in cli.rs with From<CliModel> for prunr_core::ModelKind — keeps CLI layer decoupled from core types"

patterns-established:
  - "ValueEnum pattern: CLI enum flags (CliModel, LargeImagePolicy) implement clap::ValueEnum, core type conversion via From impl"
  - "Pipeline bypass pattern: process_image_from_decoded private helper shared between checked and unchecked variants"
  - "Feature propagation: prunr-app dev-models = prunr-core/dev-models for consistent dev/prod mode switching"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-04]

# Metrics
duration: 77min
completed: 2026-04-07
---

# Phase 3 Plan 01: CLI Argument Structure and Pipeline Bypass Summary

**clap 4.5 derive structs for prunr remove subcommand (Cli/Commands/RemoveArgs/CliModel/LargeImagePolicy) plus process_image_unchecked for --large-image=process bypass**

## Performance

- **Duration:** 77 min
- **Started:** 2026-04-07T00:56:44Z
- **Completed:** 2026-04-07T02:14:08Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Defined complete clap CLI contract (Cli, Commands, RemoveArgs) with all 7 flags: --model, --jobs, --large-image, --output-dir, --force, --quiet, -o
- CliModel and LargeImagePolicy implement ValueEnum for clap parsing with From<CliModel> for prunr_core::ModelKind conversion
- Refactored prunr-core pipeline to extract shared logic into process_image_from_decoded, exposing process_image_unchecked as a public bypass for large images
- Added dev-models feature to prunr-app so the crate can be checked without triggering the model-embedding proc macro (OOM risk)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add clap/indicatif deps and create cli.rs** - `7c9eca2` (feat)
2. **Task 2: Add process_image_unchecked to pipeline.rs and re-export** - `0521d0c` (feat)

**Plan metadata:** (see docs commit below)

## Files Created/Modified
- `crates/prunr-app/src/cli.rs` - Complete clap derive structs: Cli, Commands, RemoveArgs, CliModel, LargeImagePolicy
- `crates/prunr-app/Cargo.toml` - Added clap, indicatif workspace deps; dev-models feature
- `crates/prunr-app/src/main.rs` - Added `pub mod cli;` declaration
- `crates/prunr-core/src/pipeline.rs` - Refactored into process_image, process_image_unchecked, process_image_from_decoded
- `crates/prunr-core/src/lib.rs` - Re-exported process_image_unchecked

## Decisions Made
- Added `dev-models` feature to prunr-app: without it, cargo check triggers the include-bytes-zstd proc macro that compresses 174MB of model files, causing SIGKILL due to memory pressure. The feature propagates prunr-core/dev-models and skips the embedding.
- Defined `CliModel` as a separate enum from `prunr_core::ModelKind`: keeps the CLI layer decoupled from core types. The `From<CliModel>` conversion bridges them at the dispatch layer in Plan 02.
- Extracted `process_image_from_decoded` as private helper: avoids duplicating the full inference pipeline body between the checked and unchecked variants.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added dev-models feature to prunr-app/Cargo.toml**
- **Found during:** Task 1 (cargo check verification)
- **Issue:** `cargo check -p prunr-app` without dev-models feature triggers prunr-models proc macro to embed 174MB ONNX files, causing SIGKILL. The plan's verification command would fail on any machine without adequate memory.
- **Fix:** Added `[features] dev-models = ["prunr-core/dev-models"]` to prunr-app/Cargo.toml and adjusted verification to use `--features dev-models`
- **Files modified:** crates/prunr-app/Cargo.toml
- **Verification:** `cargo check -p prunr-app --features dev-models` exits 0
- **Committed in:** 7c9eca2 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Auto-fix necessary for correctness — without it the verification step fails every time. No scope creep.

## Issues Encountered
- Background cargo processes from concurrent invocations caused file lock contention on the build directory. Resolved by killing stale processes before the definitive verification run.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `crates/prunr-app/src/cli.rs` is the locked CLI contract for Plan 02 executor
- `prunr_core::process_image_unchecked` is callable from prunr-app
- Plan 02 can use `prunr::cli::Cli::parse()`, dispatch on `Commands::Remove(args)`, and call either `process_image` or `process_image_unchecked` based on `args.large_image`
- 43 prunr-core tests (34 unit + 9 integration) pass with no regression

## Self-Check: PASSED

- crates/prunr-app/src/cli.rs: FOUND
- .planning/phases/03-cli-binary/03-01-SUMMARY.md: FOUND
- commit 7c9eca2: FOUND
- commit 0521d0c: FOUND

---
*Phase: 03-cli-binary*
*Completed: 2026-04-07*
