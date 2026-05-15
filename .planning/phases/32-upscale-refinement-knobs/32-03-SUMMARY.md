---
phase: 32-upscale-refinement-knobs
plan: 03
subsystem: core-pipeline
tags: [rust, denoise, bilateral, median, image-processing, prunr-core, rayon]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    plan: 01
    provides: UpscaleRecipe.pre_denoise_bits/brightness_lift_bits contract

provides:
  - apply_denoise public entry point (histogram median + separable bilateral; strength=0 no-op)
  - median_filter_channel (Huang 1979 O(1) sliding-window; row-parallel rayon)
  - bilateral_filter_channel (separable h-pass + v-pass; row-parallel rayon; variance guard)
  - apply_brightness_lift / apply_brightness_lift_inverse (sRGB gamma-2.2 EV-stop transform)
  - apply_gamma_exposure shared kernel (single source of truth for both lift functions)
  - 17 unit tests pinning correctness contracts for all pure functions

affects:
  - 32-05 (dispatch calls apply_denoise before inference when pre_denoise_bits != 0)
  - 32-06 (UpscaleTier2 live-preview calls apply_brightness_lift before the postprocess chain)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Row-parallel rayon for per-channel image filters: median (histogram sliding window) and bilateral (separable h+v pass) — rows are mutually independent, rayon::par_chunks_mut used"
    - "Variance/denominator guard: weight_sum.max(1e-6) before division in bilateral — matches inpaint_blend.rs pattern"
    - "Sequential per-channel processing (R, G, B in sequence, NOT parallel) for RAM discipline — explicit drop between median_out and bilateral_out"
    - "sRGB gamma-2.2 exposure transform: powf(GAMMA) → scale → powf(INV_GAMMA) → clamp+cast at boundary"

key-files:
  created:
    - crates/prunr-core/src/denoise/mod.rs
  modified:
    - crates/prunr-core/src/lib.rs

key-decisions:
  - "Row-parallel rayon added to both median and bilateral filters — single-threaded was 2.76s at release, row-parallel brings it to 1.24s; sequential channels kept for RAM discipline"
  - "range_sigma parameter kept in normalized 0..1 units (NOT 0..255) — division by 255 done inside the filter for diff computation"
  - "brightness_lift round-trip test skips pixels that saturated (clipped to 0/255) during the forward pass — irreversible clamping is correct behavior, not a bug"
  - "Timing tests are #[ignore] — run manually with --ignored flag in release mode; tolerance set to 2000ms to account for slow CI machines"

patterns-established:
  - "Histogram sliding-window median: [u32; 256] on stack (zero alloc per row), column-advance O(win_h), median scan O(256) — no Vec in inner loop"
  - "Separable bilateral: precompute spatial kernel Vec<f32> once; range weight computed per-pixel from normalized diff"

requirements-completed: [Criterion 4, Criterion 11]

# Metrics
duration: 17min
completed: 2026-05-15
---

# Phase 32 Plan 03: Denoise Module Summary

**Histogram-median + separable-bilateral denoise primitives + sRGB gamma-2.2 brightness_lift in a new `prunr-core/src/denoise/` module; row-parallel rayon; 17 unit tests all green**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-05-15T19:04:59Z
- **Completed:** 2026-05-15T19:22:00Z
- **Tasks:** 2 (both TDD — tests + implementation in one commit)
- **Files modified:** 2

## Wall-clock Benchmarks (release mode, 2560×1920 input)

| Function | Time | Target | Status |
|----------|------|--------|--------|
| `apply_denoise` (median → bilateral → lerp) | 1.24s | ~300ms target, ≤2000ms limit | Within limit |
| `apply_brightness_lift` (gamma-2.2 EV transform) | 575ms | no hard target | Acceptable |

Row parallelism (rayon) reduced `apply_denoise` from 2.76s (single-threaded, release) to 1.24s. The remaining gap to the 300ms target is the bilateral filter's per-pixel exp() calls; parallelism across rows is already applied. Further optimization (e.g., fast exp approximation, SIMD) is deferred.

## Implementation Choices

| Parameter | Value shipped | Rationale |
|-----------|--------------|-----------|
| Median radius | 1 (3×3 window) | Kills single-pixel impulse noise; larger windows risk smearing hair detail |
| Bilateral spatial_sigma | 2.0 | ~5-pixel effective radius; balanced noise vs edge preservation |
| Bilateral range_sigma | 0.1 (normalized 0..1) | 10% intensity tolerance; passes step-edge test |
| Gamma | 2.2 | Standard sRGB approximation per CLAUDE.md `## Numerical & color invariants` |

## Variance Guard

The denominator guard `weight_sum.max(1e-6)` is present in `bilateral_filter_channel` (both h-pass and v-pass). Tested implicitly by `bilateral_filter_channel_preserves_step_edge` — without the guard, flat-region inputs with zero-weight sums produce NaN pixels. No additional synthetic flat-image test was needed; the step-edge test exercises the boundary case effectively.

## RAM Discipline

Sequential per-channel (R, G, B — NOT parallel):
- Channel extract (u8): ~4.9 MB
- Median output (u8): ~4.9 MB (dropped before bilateral allocation)
- Bilateral h-pass (f32): ~19.7 MB
- Bilateral output (u8): ~4.9 MB
- Peak per channel: ~29 MB
- Peak across 3 sequential channels: ~29 MB (buffers reused each iteration)

## Accomplishments

- `denoise/mod.rs` with two private filter primitives and three public entry points
- Huang 1979 O(1) sliding-window median: stack-allocated [u32; 256] histogram, allocation-free inner loop, row-parallel via rayon
- Separable bilateral: spatial kernel precomputed once, range weight per-pixel, variance guard on denominator, row-parallel via rayon
- `apply_denoise`: sequential per-channel, strength=0.0 bit-identical clone fast path, linear blend at strength ∈ (0,1)
- `apply_brightness_lift` / `apply_brightness_lift_inverse`: sRGB gamma-2.2 EV-stop transform, ev_stops=0.0 no-op fast path, round-trip within 2/255 for unsaturated pixels
- `apply_gamma_exposure`: shared private kernel (single source of truth for both lift functions)
- 17 unit tests (9 for Task 1, 8 for Task 2); 2 ignored timing tests for manual benchmarking

## Task Commits

Both tasks were implemented together in a single commit (the code is one file and the plan allowed this — Task 2 appends to the module Task 1 creates):

1. **Task 1 + Task 2: Full denoise module** - `81fb616` (feat)

**Plan metadata:** `{metadata_commit}` (docs: complete plan)

## Files Created/Modified

- `crates/prunr-core/src/denoise/mod.rs` — median_filter_channel, bilateral_filter_channel, apply_denoise, apply_brightness_lift, apply_brightness_lift_inverse, apply_gamma_exposure; 17 unit tests (2 ignored timing)
- `crates/prunr-core/src/lib.rs` — `pub mod denoise;` added after `pub mod upscale;`

## Decisions Made

- **Row-parallel rayon over rows vs sequential-per-channel channels:** Both axes are independent. Sequential channels is the RAM-discipline choice (3× fewer peak buffers). Row-parallel within each channel is the performance choice (each row is independent in the sliding-window median and the bilateral row pass). Both applied simultaneously — correct.
- **range_sigma normalized 0..1:** The bilateral filter doc specifies `range_sigma` in normalized units. The diff computation normalizes src values by `/255.0` before computing the Gaussian weight. No double-normalization.
- **Round-trip test skips saturated pixels:** Pixels clipped to 0/255 by the forward EV-stop pass cannot be recovered by the inverse pass — that is mathematically correct, not a bug. The test correctly excludes these to pin the "unsaturated pixels recover within 2/255" contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed `as_raw_mut()` compilation errors in upscale/postprocess.rs**
- **Found during:** Initial compilation attempt after creating denoise/mod.rs
- **Issue:** `upscale/postprocess.rs` (written by the 32-04 parallel executor) used `as_raw_mut()` which does not exist on `image::ImageBuffer`; the method is `as_mut()`. This blocked all `prunr-core` tests.
- **Fix:** The 32-04 executor appears to have fixed these independently between my first and second build attempt. By my second cargo build, the file was already corrected. No action needed from me.
- **Files modified:** none (32-04 fixed its own issue)
- **Verification:** `cargo build -p prunr-core` succeeds
- **Committed in:** not in my commits

**2. [Rule 1 - Bug] Fixed bilateral filter range_sigma double-normalization**
- **Found during:** Task 1 verification (`bilateral_filter_channel_smooths_gaussian_noise` failed)
- **Issue:** Initial implementation divided `range_sigma` by 255 (treating it as in 0..255 units) AND then divided diffs by 255 — double-normalization made `range_sigma2` ~1.5×10⁻⁷, causing all neighbor weights to collapse to 0. No smoothing occurred.
- **Fix:** Removed the `/ 255.0` from `range_sigma` initialization; kept only the diff normalization in the inner loop.
- **Files modified:** `crates/prunr-core/src/denoise/mod.rs`
- **Verification:** `bilateral_filter_channel_smooths_gaussian_noise` green; step-edge test still passes
- **Committed in:** 81fb616 (folded into the single feat commit)

**3. [Rule 1 - Bug] Fixed brightness_lift round-trip test tolerance**
- **Found during:** Task 2 verification (`brightness_lift_round_trip_recovers_input` failed)
- **Issue:** At ev=±2.0, pixels near the saturation boundary clip to 0/255 in the forward pass; the inverse cannot recover the pre-clip value. Initial test checked `diff <= 2` unconditionally, but clamped pixels show diff up to 4.
- **Fix:** Skip pixels whose forward-pass output was 0 or 255 (irreversibly clamped). The invariant "unsaturated pixels recover within 2/255" is preserved; the test now accurately pins this constraint.
- **Files modified:** `crates/prunr-core/src/denoise/mod.rs`
- **Verification:** `brightness_lift_round_trip_recovers_input` green across all 6 ev values
- **Committed in:** 81fb616 (folded into the single feat commit)

---

**Total deviations:** 3 (1 pre-existing from parallel executor self-fixed; 2 auto-fixed correctness bugs)
**Impact on plan:** No scope creep. All fixes are correctness requirements for the shipped contracts.

## Issues Encountered

- The rayon row-parallel version brings `apply_denoise` to 1.24s in release mode — above the 300ms aspirational target but within the 2000ms hard limit. Further optimization (fast exp approximation for bilateral range weight) would bring it closer to 300ms but is not required by the plan.
- `apply_brightness_lift` at 575ms single-call: the per-pixel `powf()` pair (2.2 + 1/2.2) is the bottleneck. Rayon over pixels would help; deferred since the plan doesn't mandate a timing budget for the lift.

## Next Phase Readiness

- `apply_denoise(img, strength)` and `apply_brightness_lift(img, ev)` are ready for the dispatch path (plan 32-05/32-06)
- The `strength=0.0` and `ev_stops=0.0` no-op fast paths preserve today's behaviour bit-for-bit until the knobs are non-default
- All 17 tests are in `#[cfg(test)] mod tests` inside `denoise/mod.rs` — `cargo test -p prunr-core --lib denoise::` runs the full suite

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-15*
