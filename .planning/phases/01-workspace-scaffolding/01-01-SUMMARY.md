---
phase: 01-workspace-scaffolding
plan: 01
subsystem: infra
tags: [cargo, workspace, rust, thiserror, include-bytes-zstd, xtask, onnx]

# Dependency graph
requires: []
provides:
  - Cargo workspace with four members: prunr-core, prunr-models, prunr-app, xtask
  - InferenceEngine trait stub (Send+Sync) in prunr-core
  - CoreError thiserror enum with Io and Model variants in prunr-core
  - dev-models feature gate in prunr-models with fs-read functions
  - Production zstd model embedding guards (cfg not dev-models) in prunr-models
  - Placeholder binary printing CARGO_PKG_VERSION in prunr-app
  - xtask stub with fetch-models not-yet-implemented message
  - .cargo/config.toml with xtask alias
  - .gitignore excluding /target and /models/
  - Cargo.lock committed (binary project)
affects: [02-xtask-model-fetcher, 03-inference-pipeline, 04-cli, 05-gui, 06-distribution]

# Tech tracking
tech-stack:
  added:
    - thiserror 2.0 (structured error enums)
    - include-bytes-zstd (git, build-time zstd model compression)
    - anyhow 1 (xtask error handling)
    - reqwest 0.12 blocking+rustls-tls (xtask model download — stub for Plan 02)
    - sha2 0.10 + hex 0.4 (xtask SHA256 verification — stub for Plan 02)
  patterns:
    - Cargo workspace with shared [workspace.dependencies] for consistent pinning
    - prunr-models isolation: zero deps on other workspace crates, model recompile is isolated
    - dev-models feature gate: dev uses fs::read, prod uses include_bytes_zstd! static
    - xtask alias in .cargo/config.toml for developer tooling

key-files:
  created:
    - Cargo.toml (workspace root with 4 members and shared dep declarations)
    - .cargo/config.toml (xtask alias)
    - .gitignore (/target and /models/)
    - crates/prunr-core/Cargo.toml
    - crates/prunr-core/src/lib.rs
    - crates/prunr-core/src/engine.rs (InferenceEngine trait)
    - crates/prunr-core/src/types.rs (CoreError enum)
    - crates/prunr-models/Cargo.toml (dev-models feature, include-bytes-zstd build dep)
    - crates/prunr-models/src/lib.rs (model embedding with cfg gates)
    - crates/prunr-app/Cargo.toml
    - crates/prunr-app/src/main.rs (placeholder binary)
    - xtask/Cargo.toml
    - xtask/src/main.rs (stub)
    - Cargo.lock
  modified: []

key-decisions:
  - "Added #[allow(unused_imports)] to engine.rs use crate::types::CoreError — import required by spec but unused in stub; will be used in Phase 2 inference backend"

patterns-established:
  - "Workspace dependency pattern: all shared deps declared under [workspace.dependencies], crates reference via { workspace = true }"
  - "Model isolation pattern: prunr-models has zero path deps to other workspace crates; dependency arrow is app -> core -> models (never reverse)"
  - "Feature gate pattern: #[cfg(not(feature = 'dev-models'))] guards prod statics, #[cfg(feature = 'dev-models')] guards dev fs-read fns"
  - "xtask pattern: cargo xtask alias in .cargo/config.toml routes to xtask package"

requirements-completed: [DIST-02, DIST-04]

# Metrics
duration: 8min
completed: 2026-04-06
---

# Phase 1 Plan 1: Workspace Scaffolding Summary

**Cargo workspace skeleton with InferenceEngine/CoreError trait stubs in prunr-core, dev-models feature gate in prunr-models, and placeholder binary in prunr-app — all four packages build and test clean**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-06T21:17:54Z
- **Completed:** 2026-04-06T21:25:54Z
- **Tasks:** 2
- **Files modified:** 14 created, 0 modified

## Accomplishments

- Four-crate Cargo workspace builds clean with `cargo build --workspace --features prunr-models/dev-models`
- Three tests pass: `test_inference_engine_trait_is_object_safe`, `test_core_error_model_variant`, `test_model_api_compiles`
- prunr-models crate is fully isolated (zero workspace crate deps) so model recompilation is independent of source changes
- `cargo xtask` alias resolves and provides usage help

## Task Commits

Each task was committed atomically:

1. **Task 1: Workspace manifests and crate skeletons** - `37d8b46` (feat)
2. **Task 2: Source stubs — traits, types, model embedding, placeholder binary** - `044436a` (feat)

**Plan metadata:** `8add3d5` (docs: complete workspace-scaffolding plan 01)

## Files Created/Modified

- `Cargo.toml` — workspace root with 4 members, shared dep declarations, release/dist profiles
- `.cargo/config.toml` — xtask alias: `run --package xtask --`
- `.gitignore` — excludes `/target` and `/models/`
- `crates/prunr-core/Cargo.toml` — depends on prunr-models and thiserror
- `crates/prunr-core/src/lib.rs` — exports InferenceEngine and CoreError
- `crates/prunr-core/src/engine.rs` — InferenceEngine trait (Send+Sync) with mock test
- `crates/prunr-core/src/types.rs` — CoreError enum with Io and Model variants + test
- `crates/prunr-models/Cargo.toml` — dev-models feature, include-bytes-zstd build dep only
- `crates/prunr-models/src/lib.rs` — cfg-gated model statics (prod) and fs-read fns (dev) + API test
- `crates/prunr-app/Cargo.toml` — depends on prunr-core, defines prunr binary
- `crates/prunr-app/src/main.rs` — placeholder printing CARGO_PKG_VERSION
- `xtask/Cargo.toml` — anyhow, reqwest, sha2, hex (ready for Plan 02 fetch-models)
- `xtask/src/main.rs` — stub with not-yet-implemented fetch-models command
- `Cargo.lock` — committed (binary project)

## Decisions Made

- Added `#[allow(unused_imports)]` to `engine.rs` for `use crate::types::CoreError` — the plan spec requires the import but it is unused in the Phase 1 stub. Suppressing the warning preserves the intended interface specification without warnings.

## Deviations from Plan

None — plan executed exactly as written, with one minor auto-fix (Rule 1) for the unused import warning.

### Auto-fixed Issues

**1. [Rule 1 - Bug] Suppressed unused import warning in engine.rs**
- **Found during:** Task 2 (build verification)
- **Issue:** `use crate::types::CoreError` in engine.rs stub triggers `unused_imports` warning; plan spec requires the import for the intended interface
- **Fix:** Added `#[allow(unused_imports)]` to silence the warning while preserving the import as architectural documentation
- **Files modified:** `crates/prunr-core/src/engine.rs`
- **Verification:** `cargo build --workspace --features prunr-models/dev-models` completes with no warnings
- **Committed in:** `044436a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 warning suppression)
**Impact on plan:** No scope change; preserves intended interface specification.

## Issues Encountered

None — all four packages compiled on first attempt after source files were created.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Workspace fully established; Plan 02 can implement `cargo xtask fetch-models` with SHA256-verified ONNX downloads
- prunr-models isolation confirmed; model recompilation will not cascade to core/app on source changes
- InferenceEngine trait ready for ORT backend implementation in Phase 2/3
- CoreError ready for error propagation throughout the inference pipeline

---
*Phase: 01-workspace-scaffolding*
*Completed: 2026-04-06*
