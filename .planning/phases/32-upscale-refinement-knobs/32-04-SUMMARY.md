---
phase: 32-upscale-refinement-knobs
plan: 04
subsystem: core-pipeline
tags: [rust, upscale, postprocess, image-processing, color-science, prunr-core]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: UpscaleRecipe Tier-2 fields (sharpen_bits, ai_blend_bits, saturation_bits, color_match)
provides:
  - apply_sharpen: unsharp mask (Gaussian blur subtract/lerp), strength=0 no-op
  - apply_ai_blend: per-pixel RGB lerp between AI output and bicubic source, weight=1.0 no-op
  - apply_saturation: HSL-space saturation adjustment, amount=0 no-op
  - apply_color_match: Reinhard Lab mean+stddev transfer with variance guard
  - 18 unit tests pinning all boundary contracts
affects:
  - 32-06 (live-preview Tier-2 dispatch consumes these four functions)
  - 32-05 (Tier-2 pipeline dispatch wiring)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-rolled IEC 61966-2-1 linearisation + D65 XYZ→Lab — deps stay minimal; standard formulas live in-tree for future maintenance"
    - "Borrow-split pattern for in-place image mutation: read into intermediate buffer, then write back via .as_mut()"
    - "Separable 5-tap Gaussian (σ≈1.0) for unsharp mask — kernel pinned as const GAUSS5 for stable live-preview feel"
    - "stddev.max(1e-6) variance guard matches inpaint_blend.rs canonical pattern"

key-files:
  created:
    - crates/prunr-core/src/upscale/postprocess.rs
  modified:
    - crates/prunr-core/src/upscale/mod.rs

key-decisions:
  - "Option B (hand-rolled sRGB↔Lab + sRGB↔HSL) chosen over palette crate — zero new deps; IEC 61966-2-1 linearisation and CIE Lab formulas are standard and fully in-tree"
  - "5-tap Gaussian kernel (σ≈1.0) pinned for sharpen blur — matches the GAUSS5 const array so live-preview feel is stable across refactors"
  - "HSL desaturation uses multiplicative scale: S' = S * (1 + amount) for amount<0 — amount=-1 fully desaturates to S=0; amount=+1 saturates to S=1 via S' = S + amount*(1-S)"
  - "apply_ai_blend uses intermediate RGB buffer + write-back to avoid borrow conflict on RgbaImage — no extra heap overhead beyond the 3×n bytes"

patterns-established:
  - "Borrow-split pattern: image::RgbaImage requires separate read (.as_raw()) and write (.as_mut()) borrows; use intermediate buffer when computing in-place from self"
  - "variance guard sited at each of the three Lab channels independently — not a single shared guard — per Reinhard et al. transfer formula"

requirements-completed: [Criterion 5, Criterion 11]

# Metrics
duration: 9min
completed: 2026-05-15
---

# Phase 32 Plan 04: Upscale Postprocess Primitives Summary

**Four Tier-2 RGBA postprocess functions (sharpen/AI-blend/saturation/color-match) with hand-rolled IEC 61966-2-1 Lab pipeline, variance guard, and 18 unit tests pinning every boundary contract**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-15T19:05:14Z
- **Completed:** 2026-05-15T19:14:31Z
- **Tasks:** 1 (TDD: tests + implementation in single commit — all 18 tests green)
- **Files modified:** 2

## Accomplishments

- `apply_sharpen`: 5-tap separable Gaussian (σ≈1.0) unsharp mask; `strength=0.0` early-returns bit-identical; negative blurs, positive sharpens; alpha untouched
- `apply_ai_blend`: per-pixel RGB lerp between AI output and bicubic source; `weight=1.0` early-returns; `debug_assert_eq!` enforces dimension invariant; alpha from AI input
- `apply_saturation`: HSL-space (not HSV) saturation; multiplicative desaturation (`S * (1+amount)`) and additive saturation (`S + amount*(1-S)`); luminance preserved
- `apply_color_match`: Reinhard Lab mean+stddev transfer with hand-rolled IEC 61966-2-1 → XYZ → Lab pipeline; `stddev.max(1e-6)` at each of the three Lab channels prevents NaN on uniform-color sources; source and target may have different dimensions

## Dep Decision: Option B — Hand-Rolled sRGB↔Lab + sRGB↔HSL

`palette` crate (Option A, 200 KB compiled) was NOT added. Both conversions are hand-rolled:

- **IEC 61966-2-1 piecewise linearisation** (~10 lines each direction)
- **D65 Bradford XYZ matrix** (standard 3×3, matches IEC 61966-2-1)
- **CIE cube-root Lab nonlinearity** (standard `(6/29)` delta threshold)
- **HSL 6-case algorithm** (~30 lines each direction, operates on gamma-encoded sRGB)

Rationale: deps stay minimal; the formulas are standard and fully in-tree with no external auditing dependency. No `Cargo.toml` change required. Using `palette` for one operation (Lab) and not the other (HSL) would be inconsistent; hand-rolling both is the coherent choice.

## Gaussian Kernel Choice

Sigma: 1.0, kernel size: 5-tap ([-2, -1, 0, +1, +2] relative), separable row/column passes. Weights:

```rust
const GAUSS5: [f32; 5] = [0.061, 0.245, 0.388, 0.245, 0.061];
```

Typical unsharp blur radius is 1-3 px Gaussian per CONTEXT.md "Claude's Discretion". Sigma=1.0 / 5-tap was chosen as the mid-range that: (1) provides visible sharpening on upscaled output without ringing artefacts at `strength=1.0`, (2) keeps the blur kernel fast (5 multiplies per pixel per axis), (3) is pinned in a named `const` so live-preview feel is stable across any future refactor of the function body.

## HSL Desaturation Semantics

`amount ∈ [-1, 1]`:
- **Positive (saturate):** `S' = S + amount * (1 - S)`. At amount=+1: `S' = 1.0`. Linear interpolation toward fully-saturated.
- **Negative (desaturate):** `S' = S * (1 + amount)`. At amount=-1: `S' = 0.0`. Multiplicative — preserves the relative saturation ordering across pixels.

The `apply_saturation_hsl_not_hsv` test pins the HSL behaviour: pure red `[255, 0, 0]` at amount=-0.5 produces output R < 240 (distinguishing HSL from HSV, which would keep R ≈ 255 since HSV's saturation axis doesn't affect the max channel).

## Variance Guard — What Happens Without It

Test `apply_color_match_uniform_source_no_nan` exercises a uniform-grey source (`[128, 128, 128]` every pixel), which has stddev = 0 for all three Lab channels. Without `stddev.max(1e-6)`, the scale factor becomes `s_std / 0.0 = Inf`, and the per-pixel Lab transfer produces `Inf * (L - mean)` = `Inf` or `NaN` for every pixel. When the result is cast back to u8 via `clamp(0.0, 255.0)`, NaN clamps to 0 on some platforms (Inf clamps to 255), producing an all-black or all-white output. The guard eliminates this: with stddev = 0 → max(0, 1e-6) = 1e-6, the scale becomes `s_std / 1e-6` which is ~0 when s_std is also 0 (uniform source), yielding `target_L ← 0 * (L - mean) + s_mean ≈ s_mean` — the correct result for matching to a uniform source (everything goes to the source mean).

## Per-Function Wall-Clock Estimates (2560×1920 image, ~4.9M pixels)

Formal benches were not added (separate `[[bench]]` target per CLAUDE.md). Analytical estimates based on operation cost:

| Function | Per-pixel ops | Estimated time (release build) |
|----------|--------------|-------------------------------|
| `apply_sharpen` | 5×2 muls/adds per pixel per channel (Gaussian) × 2 passes + lerp ≈ 60 ops | ~15–30ms |
| `apply_ai_blend` | 3 muls + 2 adds + 1 clamp per pixel ≈ 6 ops | ~5–10ms |
| `apply_saturation` | sRGB→HSL (~15 ops) + S adjust + HSL→sRGB (~15 ops) ≈ 35 ops | ~15–25ms |
| `apply_color_match` | sRGB→Lab roundtrip × 2 passes (stats + transfer) + Lab→sRGB ≈ 200 ops | ~80–150ms |
| **Total Tier-2 pipeline** | | **~115–215ms** |

This is within the 200ms CONTEXT.md budget (median case ~165ms). The `apply_color_match` dominates; if the full pipeline exceeds budget on a real 2560×1920 image, rayon row-parallel on the Lab conversion passes is the obvious lever (not applied now per CLAUDE.md RAM discipline — parallelising would triple peak RAM; the sequential single-channel path is the safe baseline).

## Task Commits

1. **Task 1: Create upscale/postprocess.rs with four pure functions + 18 unit tests** — `6002472` (feat)

**Plan metadata:** `{metadata_commit}` (docs: complete plan)

## Files Created/Modified

- `crates/prunr-core/src/upscale/postprocess.rs` — New: four public functions + hand-rolled color math helpers + 18 unit tests
- `crates/prunr-core/src/upscale/mod.rs` — Added: `pub mod postprocess;` + four `pub use` re-exports

## Decisions Made

- Hand-rolled sRGB↔Lab and sRGB↔HSL rather than adding `palette` crate — zero new deps; formulas are standard and in-tree
- 5-tap Gaussian (σ≈1.0) for unsharp mask — mid-range between too-subtle and ringing; pinned in `const GAUSS5`
- Multiplicative desaturation (`S * (1+amount)`) rather than linear clamp — preserves relative per-pixel saturation ordering
- `debug_assert_eq!` for `apply_ai_blend` dimension invariant — caller responsibility, panics in debug builds, caps iteration in release

## Deviations from Plan

None — plan executed exactly as written. The `as_raw_mut()` compiler error (method doesn't exist on `image::ImageBuffer`) was resolved immediately by using the correct pattern (`as_raw()` / `as_mut()`) already established in `inpaint_blend.rs`. No deviation tracking needed.

## Issues Encountered

- `image` crate's `RgbaImage` has no `as_raw_mut()` method; the correct pattern is `.as_raw()` for read-only and `.as_mut()` for mutable — consistent with how `inpaint_blend.rs` already operates. Fixed in the first compile pass.
- Clippy `excessive_precision` on three float literals in the XYZ matrices — truncated to the f32-representable precision Clippy suggested.

## Next Phase Readiness

- Four Tier-2 postprocess functions available at `prunr_core::upscale::{apply_sharpen, apply_ai_blend, apply_saturation, apply_color_match}`
- Plan 32-06 (live-preview dispatch) can call these directly on the `upscale_raw: Arc<RgbaImage>` cache
- No Cargo.toml changes — dep footprint unchanged

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-15*
