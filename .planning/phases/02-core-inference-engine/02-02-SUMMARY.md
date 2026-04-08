---
phase: 02-core-inference-engine
plan: "02"
subsystem: inference
tags: [ndarray, image, onnx, preprocessing, postprocessing, rembg]

# Dependency graph
requires:
  - phase: 02-01
    provides: CoreError, types.rs, module stubs for preprocess/postprocess/formats
provides:
  - preprocess(): NCHW Array4<f32> tensor with rembg-exact Lanczos3 + max-pixel normalization
  - postprocess(): min-max alpha mask upscaling, RgbaImage at original dimensions
  - formats.rs: PNG/JPEG/WebP/BMP loading, large image detection, downscaling, PNG encoding
affects:
  - 02-03 (engine.rs integration uses preprocess/postprocess)
  - 02-04 (pipeline.rs uses formats for input loading)
  - 02-06 (reference test validates these exact functions)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - rembg-exact preprocessing: Lanczos3 resize, max(max_pixel, 1e-6) normalization, NCHW layout
    - rembg-exact postprocessing: min-max normalization only (no sigmoid, no threshold)
    - From<image::ImageError> for CoreError via ImageFormat variant

key-files:
  created:
    - crates/prunr-core/src/preprocess.rs
    - crates/prunr-core/src/postprocess.rs
    - crates/prunr-core/src/formats.rs
  modified:
    - crates/prunr-core/src/types.rs

key-decisions:
  - "From<image::ImageError> for CoreError auto-added to types.rs — required for formats.rs image loading (Rule 3 fix)"
  - "DOWNSCALE_TARGET imported in test module directly (not re-exported from formats.rs) to avoid unused import warning"

patterns-established:
  - "preprocess(): always [1, 3, 320, 320] NCHW layout, Lanczos3 resize, max(max_pixel, 1e-6) norm"
  - "postprocess(): min-max only — (val - mi) / (ma - mi).max(1e-6), no sigmoid"
  - "formats.rs: image loading returns Result<DynamicImage, CoreError> via From<image::ImageError>"

requirements-completed: [CORE-05, LOAD-03, LOAD-04]

# Metrics
duration: 5min
completed: 2026-04-06
---

# Phase 02 Plan 02: Preprocessing, Postprocessing, and Format Loading Summary

**rembg-exact NCHW preprocessing (Lanczos3 + max-pixel normalization) and min-max postprocessing (no sigmoid) implemented as pure functions, with PNG/JPEG/WebP/BMP format loading, large image detection, and RGBA PNG encoding — 22 tests all green**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-06T23:09:37Z
- **Completed:** 2026-04-06T23:14:04Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- preprocess(): rembg-exact pipeline — Lanczos3 resize to 320x320, max(max_pixel, 1e-6) normalization, NCHW Array4<f32> [1,3,320,320] with ImageNet mean/std
- postprocess(): min-max normalization only (no sigmoid, no threshold), Lanczos3 mask upscale, alpha channel merge to RgbaImage at original dimensions
- formats.rs: load_image_from_path, load_image_from_bytes (magic-byte format detection), check_large_image, downscale_image (aspect-preserving Lanczos3), encode_rgba_png

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement preprocess.rs with rembg-exact pipeline** - `3c4bdb6` (feat)
2. **Task 2: Implement postprocess.rs and formats.rs** - `bf2714a` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `crates/prunr-core/src/preprocess.rs` - Pure function preprocess(): DynamicImage -> Array4<f32> [1,3,320,320]
- `crates/prunr-core/src/postprocess.rs` - Pure function postprocess(): ArrayView4<f32> + DynamicImage -> RgbaImage
- `crates/prunr-core/src/formats.rs` - Image loading (path/bytes), large image check, downscale, PNG encode
- `crates/prunr-core/src/types.rs` - Added From<image::ImageError> for CoreError

## Decisions Made
- `From<image::ImageError>` added to types.rs as an inline impl (not `#[from]` on a new variant) — maps to existing `ImageFormat(String)` variant, preserves the error enum shape from Plan 01
- `DOWNSCALE_TARGET` imported explicitly in test module rather than re-exported from formats.rs — avoids unused import lint in production code while still being available in tests

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added From<image::ImageError> for CoreError to types.rs**
- **Found during:** Task 2 (formats.rs implementation)
- **Issue:** formats.rs uses `.map_err(CoreError::from)` for image decoding errors, but `types.rs` had no `From<image::ImageError>` impl — compile error blocked Task 2 completion
- **Fix:** Added `impl From<image::ImageError> for CoreError` to types.rs, mapping to `CoreError::ImageFormat(e.to_string())`
- **Files modified:** `crates/prunr-core/src/types.rs`
- **Verification:** `cargo test -p prunr-core` passes all 22 tests with no errors
- **Committed in:** `bf2714a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The From impl was referenced in the plan's interface block as expected but missing from Plan 01's types.rs output. Auto-fix necessary for compilation. No scope creep.

## Issues Encountered
- `DOWNSCALE_TARGET` was imported in formats.rs production code but only used in tests — removed from production import, added explicit import in test module to avoid unused import warning

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- preprocess() and postprocess() ready for integration into engine.rs (Plan 02-03)
- formats.rs image loading ready for pipeline.rs (Plan 02-04)
- All pure functions have unit tests verifying rembg-exact behavior
- No blockers for Plan 02-03

---
*Phase: 02-core-inference-engine*
*Completed: 2026-04-06*
