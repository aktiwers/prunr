---
phase: 02-core-inference-engine
plan: "01"
subsystem: inference
tags: [rust, ort, ndarray, image, rayon, num_cpus, types, error-handling]

# Dependency graph
requires:
  - phase: 01-workspace-scaffolding
    provides: workspace structure, prunr-core crate scaffold, prunr-models crate with model bytes API
provides:
  - CoreError with 5 variants (Io, Model, Inference, ImageFormat, LargeImage)
  - ModelKind enum (Silueta, U2net)
  - ProgressStage enum (Decode, Resize, Normalize, Infer, Postprocess, Alpha)
  - ProcessResult struct (rgba_bytes, active_provider)
  - LARGE_IMAGE_LIMIT and DOWNSCALE_TARGET constants
  - Phase 2 Cargo dependencies declared (ort, ndarray, image, rayon, num_cpus)
  - Stub modules for pipeline, preprocess, postprocess, batch, formats
affects:
  - 02-02-preprocess (imports PreprocessStage and NCHW tensor type from this)
  - 02-03-postprocess (imports ProcessResult, ProgressStage from this)
  - 02-04-engine (imports CoreError, ModelKind, ProcessResult from this)
  - 02-05-batch (imports ProcessResult, ProgressStage from this)
  - All Phase 3+ plans via public re-exports in lib.rs

# Tech tracking
tech-stack:
  added:
    - ort 2.0.0-rc.12 (ONNX Runtime Rust bindings — CUDA/CoreML/DirectML/CPU)
    - ndarray 0.16 (4D tensor construction for ORT input/output)
    - image 0.25 (image decode/encode with jpeg, png, webp, bmp features)
    - rayon 1.11 (work-stealing thread pool for batch parallelism)
    - num_cpus 1.x (CPU count for ORT thread balancing formula)
  patterns:
    - "thiserror-based CoreError enum pattern extended with domain-specific variants"
    - "Manual From impls for ort::Error and image::ImageError (not #[from] — deps were not in scope at Phase 1)"
    - "Stub module pattern — declare pub mod in lib.rs before implementation exists"
    - "dev-models feature in dev-dependencies for tests without model files"

key-files:
  created:
    - crates/prunr-core/src/pipeline.rs (stub — implemented in 02-02)
    - crates/prunr-core/src/preprocess.rs (stub — implemented in 02-02)
    - crates/prunr-core/src/postprocess.rs (stub — implemented in 02-03)
    - crates/prunr-core/src/batch.rs (stub — implemented in 02-05)
    - crates/prunr-core/src/formats.rs (stub — implemented in 02-06)
  modified:
    - crates/prunr-core/src/types.rs (5 CoreError variants, ModelKind, ProgressStage, ProcessResult, constants)
    - crates/prunr-core/src/lib.rs (stub module declarations, full re-exports)
    - crates/prunr-core/Cargo.toml (Phase 2 deps added)
    - Cargo.toml (num_cpus added to workspace deps)
    - crates/prunr-models/Cargo.toml (include-bytes-zstd moved to [dependencies])

key-decisions:
  - "include-bytes-zstd moved from [build-dependencies] to [dependencies] in prunr-models — it is a proc-macro crate, not a build script"
  - "cargo check passes only with --features prunr-models/dev-models because model files are gitignored and require xtask fetch-models; this is correct by design"
  - "ProgressStage uses Postprocess not Sigmoid — matches rembg's actual min-max normalization (not sigmoid activation) per RESEARCH.md"

patterns-established:
  - "All Phase 2 types imported via prunr_core::{CoreError, ModelKind, ProgressStage, ProcessResult}"
  - "Dev builds always use --features prunr-models/dev-models to avoid embedding 174MB model files during development"

requirements-completed: [CORE-01, CORE-02, CORE-03, CORE-04, LOAD-03, LOAD-04]

# Metrics
duration: 4min
completed: 2026-04-06
---

# Phase 2 Plan 01: Type System and Dependency Foundation Summary

**CoreError extended to 5 variants (Inference, ImageFormat, LargeImage), ModelKind/ProgressStage/ProcessResult types declared, and ort/ndarray/image/rayon/num_cpus Cargo deps wired into prunr-core**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-06T23:03:30Z
- **Completed:** 2026-04-06T23:07:24Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Extended `CoreError` with Inference, ImageFormat, and LargeImage variants including manual `From` impls for `ort::Error` and `image::ImageError`
- Declared all Phase 2 domain types: `ModelKind`, `ProgressStage` (6 stages), `ProcessResult`, `LARGE_IMAGE_LIMIT`, `DOWNSCALE_TARGET`
- Added all Phase 2 Cargo dependencies (ort, ndarray, image, rayon, num_cpus) to prunr-core and workspace; all 8 unit tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend types.rs with Phase 2 error variants and new types** - `e949599` (feat)
2. **Task 2: Add Phase 2 Cargo dependencies and update lib.rs re-exports** - `7917513` (feat)

## Files Created/Modified
- `crates/prunr-core/src/types.rs` - CoreError (5 variants), ModelKind, ProgressStage, ProcessResult, LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET
- `crates/prunr-core/src/lib.rs` - Added stub module declarations and full type re-exports
- `crates/prunr-core/Cargo.toml` - ort, ndarray, image, rayon, num_cpus deps; dev-models in dev-deps
- `Cargo.toml` - num_cpus added to [workspace.dependencies]
- `crates/prunr-models/Cargo.toml` - Fixed: include-bytes-zstd moved to [dependencies]
- `crates/prunr-core/src/pipeline.rs` - Stub
- `crates/prunr-core/src/preprocess.rs` - Stub
- `crates/prunr-core/src/postprocess.rs` - Stub
- `crates/prunr-core/src/batch.rs` - Stub
- `crates/prunr-core/src/formats.rs` - Stub

## Decisions Made
- `include-bytes-zstd` was in `[build-dependencies]` of prunr-models but is a proc-macro crate that must be in `[dependencies]`; fixed as a Rule 1 (bug) auto-fix
- `ProgressStage::Postprocess` used instead of `Sigmoid` to match rembg's actual min-max normalization behavior (per RESEARCH.md critical finding)
- Production `cargo check -p prunr-core` (no feature flags) cannot pass without the `.onnx` model files; `--features prunr-models/dev-models` is the correct dev workflow

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed include-bytes-zstd in wrong Cargo section in prunr-models**
- **Found during:** Task 2 (running cargo check -p prunr-core)
- **Issue:** `prunr-models/Cargo.toml` declared `include-bytes-zstd` under `[build-dependencies]` but it is a proc-macro crate (not a build script dependency). The macro was unresolvable at compile time, failing with "use of unresolved module or unlinked crate"
- **Fix:** Moved `include-bytes-zstd = { workspace = true }` from `[build-dependencies]` to `[dependencies]` in `prunr-models/Cargo.toml`
- **Files modified:** `crates/prunr-models/Cargo.toml`
- **Verification:** `cargo check -p prunr-core --features prunr-models/dev-models` passes; `cargo test -p prunr-core --lib` (8 tests) passes
- **Committed in:** 7917513 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix — prunr-models was non-compilable. No scope creep.

## Issues Encountered
- Production `cargo check` (no feature flags) fails because model files are gitignored; only `--features prunr-models/dev-models` works. This is correct by design — the plan's verification command implicitly requires the dev-models feature. All dev workflows use this flag.

## Next Phase Readiness
- All Phase 2 types are in place and publicly exported via `prunr_core::{CoreError, ModelKind, ProgressStage, ProcessResult}`
- Phase 2 Plan 02 (preprocess module) can import from types.rs and use ort/ndarray/image deps immediately
- No blockers for subsequent plans

---
*Phase: 02-core-inference-engine*
*Completed: 2026-04-06*
