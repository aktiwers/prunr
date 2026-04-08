---
phase: 01-workspace-scaffolding
plan: "03"
subsystem: infra
tags: [github-actions, ci, rust, cargo, matrix-build, caching]

requires:
  - phase: 01-01
    provides: Cargo workspace with xtask (cargo xtask fetch-models command)

provides:
  - GitHub Actions CI workflow with 4-platform matrix
  - Model file caching keyed by xtask/src/main.rs hash
  - Cargo artifact caching via Swatinem/rust-cache
  - Conditional model fetch (only on cache miss)

affects:
  - All future phases (CI validates every push/PR)
  - Phase 02+ (CI will catch build regressions on all platforms)

tech-stack:
  added:
    - actions/checkout@v4
    - dtolnay/rust-toolchain@stable
    - Swatinem/rust-cache@v2
    - actions/cache@v4
  patterns:
    - Matrix CI with fail-fast disabled for independent platform failure reporting
    - Model cache keyed by hashFiles to invalidate when model URLs/checksums change
    - dev-models feature flag used in test step to avoid model file embed dependency

key-files:
  created:
    - .github/workflows/ci.yml
  modified: []

key-decisions:
  - "dtolnay/rust-toolchain used instead of deprecated actions-rs/toolchain"
  - "fail-fast: false so each platform failure is independently reported"
  - "Build step runs without dev-models (uses real fetched models), test step uses dev-models (faster, no embed dependency)"
  - "Model cache key tied to hashFiles('xtask/src/main.rs') — invalidates when SHA256 constants change"

patterns-established:
  - "CI cache pattern: restore models first, fetch only on miss, then build/test"

requirements-completed:
  - DIST-03

duration: 3min
completed: 2026-04-06
---

# Phase 01 Plan 03: CI Matrix Workflow Summary

**GitHub Actions CI with 4-platform native matrix (Linux x86_64, macOS x86_64/aarch64, Windows x86_64), model caching keyed to xtask SHA256 hash, and Cargo artifact caching via Swatinem/rust-cache**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-06T21:03:31Z
- **Completed:** 2026-04-06T21:06:30Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `.github/workflows/ci.yml` with a 4-runner matrix covering all DIST-03 platform targets
- Model cache invalidation tied to `hashFiles('xtask/src/main.rs')` — changes to model URLs/checksums automatically bust the cache
- Build step uses real fetched models; test step uses `--features prunr-models/dev-models` to run fast without embed-time model files

## Task Commits

Each task was committed atomically:

1. **Task 1: Create GitHub Actions CI matrix workflow** - `aee69c0` (feat)

**Plan metadata:** _(pending — created with docs commit below)_

## Files Created/Modified

- `.github/workflows/ci.yml` - 4-target CI matrix with model caching, Cargo caching, conditional model fetch, build and test steps

## Decisions Made

- Used `dtolnay/rust-toolchain@stable` (not `actions-rs/toolchain` which has been unmaintained since 2022)
- `fail-fast: false` ensures a Windows failure doesn't cancel the macOS jobs — each platform's failure report is independently valuable
- Build step intentionally does NOT use `--features prunr-models/dev-models` so the CI validates the real model embedding pipeline; test step uses `dev-models` to avoid embed-time model dependency in tests

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required beyond pushing to GitHub to trigger the workflow.

## Next Phase Readiness

- CI workflow is in place and will validate all future commits on all four platforms
- Phase 02 (inference engine) will benefit immediately from the CI catching build regressions
- Full DIST-03 validation requires a first push to GitHub and observing the green CI badge

---
*Phase: 01-workspace-scaffolding*
*Completed: 2026-04-06*
