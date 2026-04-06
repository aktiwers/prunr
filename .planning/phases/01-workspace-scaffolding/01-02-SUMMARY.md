---
phase: 01-workspace-scaffolding
plan: "02"
subsystem: infra
tags: [xtask, reqwest, sha2, onnx, model-fetching, sha256, bootstrap]

# Dependency graph
requires:
  - phase: 01-01
    provides: xtask crate stub with Cargo.toml dependencies (anyhow, reqwest, sha2, hex)
provides:
  - cargo xtask fetch-models subcommand with SHA256-verified ONNX model download
  - Bootstrap mode (empty sha256 constant) for first-run without known checksums
  - Idempotent model caching: skips re-download when checksum matches existing file
affects:
  - 01-03 (CI workflow — uses cargo xtask fetch-models before build)
  - 01-04 (bgprunr-models embed — depends on models/ populated by xtask)
  - Phase 2 (inference engine — depends on silueta.onnx and u2net.onnx being present)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bootstrap SHA256 pattern: empty constant skips verification and prints hash for hardcoding"
    - "reqwest blocking client with rustls for build-tool HTTP — no async overhead"
    - "sha2::Sha256::digest over accumulated bytes for post-download integrity check"
    - "Idempotent download: verify existing file first, skip or re-download based on checksum"

key-files:
  created: []
  modified:
    - xtask/src/main.rs

key-decisions:
  - "Bootstrap SHA256 mode: empty string constant skips verification and prints hash — developer hardens after first run"
  - "rembg GitHub releases as model source (danielgatis/rembg/releases/download/v0.0.0/) — exact same models rembg uses"

patterns-established:
  - "Bootstrap-then-harden SHA256 pattern for xtask model fetching"

requirements-completed: [DIST-02]

# Metrics
duration: 1min
completed: 2026-04-06
---

# Phase 01 Plan 02: fetch-models xtask Summary

**reqwest-based cargo xtask fetch-models with bootstrap SHA256 mode — downloads silueta.onnx and u2net.onnx from rembg GitHub releases with idempotent checksum-gated caching**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-06T21:03:23Z
- **Completed:** 2026-04-06T21:04:16Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Replaced xtask stub with complete fetch-models implementation using reqwest blocking + sha2
- Bootstrap mode: empty sha256 constant skips verification and prints computed hash so developer can hardcode it after first run
- Idempotent: verifies existing model files before downloading — "OK (cached)" on match, re-downloads on mismatch
- Strict mode: non-empty sha256 verifies after download, exits non-zero with clear "SHA256 mismatch for {name}" message
- Usage fallthrough: unknown subcommands print structured usage and exit 1

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement fetch-models with SHA256 verification** - `9942364` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `xtask/src/main.rs` - Complete fetch-models implementation with bootstrap SHA256 mode, reqwest blocking download, sha2 verification, and idempotent caching

## Decisions Made
- Bootstrap SHA256 mode: empty string constant skips verification and prints hash — developer hardens after first run by replacing empty strings with the printed SHA256 values. This makes the xtask work immediately after cloning without requiring the developer to know the checksums in advance.
- Used rembg's own GitHub releases as model source (same URLs rembg itself uses), ensuring model compatibility.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None - the implementation matched the plan specification exactly. Build produced one cache permission warning for the global cargo registry (pre-existing environment issue, not caused by this task).

## User Setup Required
None - no external service configuration required.

Developers must run `cargo xtask fetch-models` after cloning to populate `models/silueta.onnx` and `models/u2net.onnx`. On first run, the tool will print computed SHA256 values with instructions to hardcode them in `xtask/src/main.rs`. After hardcoding, subsequent runs verify checksums before use.

## Next Phase Readiness
- `cargo xtask fetch-models` is ready for use by CI (Plan 01-03) and developers
- The bootstrap SHA256 constants in `xtask/src/main.rs` still need to be hardened (replaced with actual SHA256 values after first download) — this is expected and the tool guides the developer through it
- Models directory will be populated on first `cargo xtask fetch-models` run, enabling bgprunr-models embed (Plan 01-04)

## Self-Check: PASSED

- xtask/src/main.rs: FOUND
- .planning/phases/01-workspace-scaffolding/01-02-SUMMARY.md: FOUND
- Commit 9942364: FOUND

---
*Phase: 01-workspace-scaffolding*
*Completed: 2026-04-06*
