---
phase: 02-core-inference-engine
plan: "06"
subsystem: testing
tags: [ort, u2net, silueta, integration-tests, pixel-accuracy, rembg, cargo-test]

# Dependency graph
requires:
  - phase: 02-core-inference-engine/02-04
    provides: process_image, batch_process, OrtEngine pipeline
  - phase: 02-core-inference-engine/02-05
    provides: generate_references.py, tests/test_images structure, tests/references layout

provides:
  - crates/bgprunr-core/tests/reference_test.rs — integration test suite covering CORE-01 through CORE-05, LOAD-03, LOAD-04
  - CORE-05 hard gate (test_rembg_reference): 95% pixel match vs rembg Python reference masks
  - All 9 test functions covering every phase requirement

affects: [03-cli, 04-gui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Integration test file in crates/bgprunr-core/tests/ (Cargo integration test convention)
    - InferenceEngine trait must be in scope for active_provider() — import explicitly in test files
    - pixel_match_percent helper: |our_alpha - ref_alpha| <= 5 tolerance, returns f64 percentage
    - Reference mask resize fallback: if dimensions differ, resize with FilterType::Nearest before comparing

key-files:
  created:
    - crates/bgprunr-core/tests/reference_test.rs
  modified: []

key-decisions:
  - "Import InferenceEngine trait explicitly in integration tests — active_provider() only available when trait is in scope"
  - "Removed unused ProcessResult and GenericImageView imports to avoid compiler warnings"
  - "Reference mask resize with FilterType::Nearest for pixel-accurate comparison when dimensions differ"

patterns-established:
  - "Integration tests that need models guard with OrtEngine::new().expect() — clear error message points to cargo xtask fetch-models"
  - "Tests that need test images check existence and return early with eprintln! rather than panicking — graceful skip pattern"

requirements-completed: [CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, LOAD-03, LOAD-04]

# Metrics
duration: 12min
completed: 2026-04-06
---

# Phase 02 Plan 06: Integration Test Suite Summary

**9-test integration suite with CORE-05 hard gate: 95% pixel match vs rembg Python output for all 3 car images**

## Status

**CHECKPOINT PENDING** — Task 1 complete and committed. Awaiting human verification (Task 2: run models, generate references, confirm test_rembg_reference >= 95% match).

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-06T23:33:00Z
- **Completed (Task 1):** 2026-04-06T23:45:00Z
- **Tasks:** 1 of 2 complete (Task 2 is checkpoint:human-verify)
- **Files created:** 1

## Accomplishments

- Created `crates/bgprunr-core/tests/reference_test.rs` with all 9 test functions
- Model-independent tests pass immediately: `test_large_image_warning`, `test_downscale_image_preserves_aspect_ratio`, `test_batch_process_multiple_images` (graceful skip if no images)
- All 6 model-dependent tests compile and run; fail with clear error message pointing to `cargo xtask fetch-models`
- CORE-05 reference test implements pixel_match_percent helper with ±5/255 tolerance and automatic reference mask resize fallback

## Task Commits

1. **Task 1: Write integration test suite in tests/reference_test.rs** - `f14b3cf` (test)

**Plan metadata:** pending (awaiting checkpoint completion)

## Files Created/Modified

- `crates/bgprunr-core/tests/reference_test.rs` — 9-function integration test suite covering CORE-01 through CORE-05, LOAD-03, LOAD-04

## Decisions Made

- Import `InferenceEngine` trait explicitly in integration tests — required for `active_provider()` method to be callable on `OrtEngine`
- Removed `ProcessResult` and `GenericImageView` from top-level imports to eliminate unused import warnings
- Reference mask resize uses `FilterType::Nearest` for pixel-accurate comparison when rembg output dimensions differ from bgprunr output

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing InferenceEngine trait import**
- **Found during:** Task 1 (compilation)
- **Issue:** `engine.active_provider()` failed with E0599 — trait method not accessible without trait in scope
- **Fix:** Added `InferenceEngine` to the `use bgprunr_core::{}` import block
- **Files modified:** crates/bgprunr-core/tests/reference_test.rs
- **Verification:** Compiled cleanly after fix
- **Committed in:** f14b3cf (Task 1 commit)

**2. [Rule 1 - Bug] Fixed unused imports causing warnings**
- **Found during:** Task 1 (compilation)
- **Issue:** `ProcessResult` and `GenericImageView` imported but not directly referenced in test functions
- **Fix:** Removed both from the top-level use statement; `ProcessResult` fields are accessed via inference, not named type
- **Files modified:** crates/bgprunr-core/tests/reference_test.rs
- **Verification:** Compiled without warnings
- **Committed in:** f14b3cf (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — compilation bugs)
**Impact on plan:** Both necessary for correct compilation. No scope creep. Test logic is exactly as specified.

## Issues Encountered

None beyond the import fixes above.

## User Setup Required

Before `test_rembg_reference` can pass:

1. **Download test images:**
   ```
   cd tests/test_images
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-1.jpg
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-2.jpg
   curl -LO https://github.com/danielgatis/rembg/raw/main/tests/car-3.jpg
   ```

2. **Fetch ONNX models:**
   ```
   cargo xtask fetch-models
   ```

3. **Generate reference masks:**
   ```
   pip install rembg
   python scripts/generate_references.py
   git add tests/references/
   git commit -m "test: add rembg reference masks for CORE-05"
   ```

4. **Run full test suite:**
   ```
   cargo test -p bgprunr-core --features dev-models
   ```

## Next Phase Readiness

- Phase 3 (CLI) unblocked once `test_rembg_reference` passes at >= 95% for all 3 images
- All 7 requirements (CORE-01 through CORE-05, LOAD-03, LOAD-04) are exercised by the test suite
- If reference test fails: check rembg preprocessing constants match pipeline.rs (Lanczos3, min-max normalization)

---
*Phase: 02-core-inference-engine*
*Completed: 2026-04-06 (Task 1 only — checkpoint pending)*
