---
phase: 32-upscale-refinement-knobs
plan: 02
subsystem: gui-settings
tags: [rust, item-settings, upscale, serde, byte-budget, niche-optimization]

# Dependency graph
requires:
  - phase: 32-upscale-refinement-knobs
    provides: "Plan 32-01: OutputScale enum + UpscaleRecipe with 7 new knob fields"
provides:
  - "ItemSettings with seven new upscale knobs (pre_denoise, brightness_lift, output_scale, sharpen, ai_blend, saturation, color_match)"
  - "byte-budget compliant (size_of::<ItemSettings>() == 128)"
  - "current_recipe() wires every new field into UpscaleRecipe"
  - "Option<NonZeroU64> niche-optimised correction_hash and bg_image_hash"
affects:
  - 32-05 (dispatch reads output_scale from ItemSettings)
  - 32-06 (live-preview reads Phase-32 knob fields)
  - 32-07 (GUI chips bind to pre_denoise/brightness_lift/sharpen/ai_blend/saturation/color_match)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Option<NonZeroU64> niche optimisation for hash fields: 8 bytes vs 16 for Option<u64>, serde-transparent"
    - "Narrowing u32 → u8 for bounded UI fields (edge_thickness 0-10, guided_radius 4-32) with as u32 cast at MaskSettings/EdgeSettings boundary"
    - "default_ai_blend() = 1.0 serde helper for non-zero default f32 fields"

key-files:
  created: []
  modified:
    - crates/prunr-app/src/gui/item_settings.rs
    - crates/prunr-app/src/gui/item.rs
    - crates/prunr-app/src/gui/app.rs
    - crates/prunr-app/src/gui/views/upscale_chip.rs
    - crates/prunr-app/src/gui/views/upscale_toolbar.rs
    - crates/prunr-app/src/gui/views/adjustments_toolbar.rs
    - crates/prunr-app/src/gui/presets.rs

key-decisions:
  - "Option<NonZeroU64> chosen for correction_hash and bg_image_hash — saves 16 bytes (niche optimisation), preserves Option serde semantics (null = None), transparent at mask_settings()/current_recipe() boundary via .map(|nz| nz.get())"
  - "edge_thickness: u32 → u8 and guided_radius: u32 → u8; cast to u32 at MaskSettings/EdgeSettings boundaries; bridged in GUI via local u32 temps in slider/chip callers"
  - "upscale_scale: u32 field removed entirely; output_scale: OutputScale replaces it; old presets with 'upscale_scale' key silently dropped (serde unknown field ignore)"
  - "ai_blend defaults to 1.0 via serde default helper — 0.0 would be pure bicubic, silently regressing Phase-30 upscale output for all existing presets"

patterns-established:
  - "Option<NonZeroU64> pattern: use for hash fields that need Option semantics but where 0 hash is pathologically impossible (fnv1a of any non-empty input is non-zero)"
  - "Stub-then-wire task split: Task 1 introduced temporary literal zeros in current_recipe(), Task 2 replaced them with field refs — clean bisect boundary"

requirements-completed: [Criterion 1, Criterion 9]

# Metrics
duration: 35min
completed: 2026-05-15
---

# Phase 32 Plan 02: ItemSettings Byte-Budget Refactor + Seven Upscale Knobs Summary

**ItemSettings extended with seven Phase-32 upscale knobs, 128-byte budget maintained via NonZeroU64 niche packing + u8 narrowing; all fields wired through current_recipe() into UpscaleRecipe**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-15T19:10:00Z
- **Completed:** 2026-05-15T19:45:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- `upscale_scale: u32` removed; replaced by `output_scale: prunr_core::OutputScale` (#[serde(default)] → X4 for Phase-30 presets with `upscale_scale` key)
- `edge_thickness: u32` → `u8` and `guided_radius: u32` → `u8`; cast at `MaskSettings`/`EdgeSettings` boundaries; GUI sliders bridged via local `u32` temps
- Six new Phase-32 knob fields added: `pre_denoise`, `brightness_lift`, `sharpen`, `ai_blend`, `saturation`, `color_match` with `#[serde(default)]`; `ai_blend` defaults to 1.0 (full AI)
- `current_recipe()` wires all seven new fields into `UpscaleRecipe` via `.to_bits()` (matching the `MaskRecipe::gamma_bits` pattern)
- Budget packing: `correction_hash` and `bg_image_hash` narrowed from `Option<u64>` (16 bytes each) to `Option<NonZeroU64>` (8 bytes each, niche optimisation) — saves 16 bytes, exactly meeting the 128-byte ceiling
- 10 new tests (5 per task); all 519 prunr-app + 325 prunr-core tests green; workspace clippy clean

## Exact `size_of::<ItemSettings>()` after refactor

**128 bytes** — exactly at the 128-byte ceiling. Verified by the `size_under_cache_line_budget` and `budget_still_under_128_bytes` tests.

## Packing options applied

| Option | Description | Bytes saved |
|--------|-------------|-------------|
| A | `upscale_scale: u32` → `output_scale: OutputScale` (#[repr(u8)]) | 3 |
| B | `edge_thickness: u32` → `u8` | 3 |
| C | `guided_radius: u32` → `u8` | 3 |
| NonZeroU64 | `correction_hash: Option<u64>` → `Option<NonZeroU64>` | 8 |
| NonZeroU64 | `bg_image_hash: Option<u64>` → `Option<NonZeroU64>` | 8 |
| **Total** | | **25 bytes freed** |

Phase-32 additions: 5×f32 (20) + bool (1) + 4 bytes alignment overhead = 25 bytes added.
Net: 128 + 25 - 25 = **128 bytes** (options D and E from RESEARCH.md were not needed).

## Callers of `upscale_scale` updated

| File | Caller | Strategy |
|------|--------|----------|
| `upscale_chip.rs` | `render_scale_chip(&mut u32, ...)` | Rewrote to take `&mut OutputScale`; inline `output_scale_to_factor()` match |
| `upscale_toolbar.rs` | `item_settings.upscale_scale` multiply | Inline match to `factor: u32`, then passed to chip |
| `app.rs` | `item.settings.upscale_scale` in dispatch | Inline match `output_scale` → `scale: u32` before `dispatch_upscale` call |
| `adjustments_toolbar.rs` | `slider_row_u32(&mut guided_radius)` | Local `radius_u32` temp; `guided_radius as u32` in, `min(255) as u8` out |
| `adjustments_toolbar.rs` | `chip_u32(&mut edge_thickness, ...)` | Wrapped in block with local `thickness_u32` temp; same in/out pattern |
| `presets.rs` tests | `upscale_scale: 2` | Updated to `output_scale: OutputScale::X2` |

## Existing serde round-trip test extended

`serde_json_roundtrip_all_fields_populated` now includes all seven Phase-32 fields at non-default values. Old test used `upscale_scale: 2` (no longer a field).

## Exact field list and defaults shipped

| Field | Type | Default | Serde |
|-------|------|---------|-------|
| `output_scale` | `OutputScale` | `X4` | `#[serde(default)]` — old presets with `upscale_scale` key silently ignored |
| `pre_denoise` | `f32` | `0.0` | `#[serde(default)]` |
| `brightness_lift` | `f32` | `0.0` | `#[serde(default)]` |
| `sharpen` | `f32` | `0.0` | `#[serde(default)]` |
| `ai_blend` | `f32` | `1.0` | `#[serde(default = "default_ai_blend")]` |
| `saturation` | `f32` | `0.0` | `#[serde(default)]` |
| `color_match` | `bool` | `false` | `#[serde(default)]` |

## Task Commits

1. **Task 1: Replace upscale_scale with output_scale + shrink edge_thickness/guided_radius to u8** - `07f69d6` (feat)
2. **Task 2: Add six new ItemSettings fields + wire current_recipe + serde round-trip tests** - `04f232b` (feat)

**Plan metadata:** `{metadata_commit}` (docs: complete plan)

## Files Created/Modified

- `crates/prunr-app/src/gui/item_settings.rs` — struct, Default, mask_settings/edge_settings/current_recipe, 10 new tests
- `crates/prunr-app/src/gui/item.rs` — NonZeroU64::new() at correction_hash and bg_image_hash write sites
- `crates/prunr-app/src/gui/app.rs` — output_scale dispatch match + reconcile_bg_image NonZeroU64 unwrap
- `crates/prunr-app/src/gui/views/upscale_chip.rs` — render_scale_chip signature changed to &mut OutputScale
- `crates/prunr-app/src/gui/views/upscale_toolbar.rs` — inline scale factor match
- `crates/prunr-app/src/gui/views/adjustments_toolbar.rs` — u32 temp bridges for guided_radius and edge_thickness
- `crates/prunr-app/src/gui/presets.rs` — updated roundtrip test to use output_scale: OutputScale::X2

## Decisions Made

- **NonZeroU64 over sentinel u64:** Option<NonZeroU64> preserves the exact same serde JSON format (null vs number) as Option<u64>. Sentinel approach would silently break old presets with `"correction_hash": null` (null cannot deserialize as u64). NonZeroU64::new() at write sites handles the hash=0 edge case by treating it as None, which is correct (fnv1a of any real content is non-zero).
- **Options D and E not needed:** After A+B+C+NonZeroU64×2 = 25 bytes savings vs 25 bytes added, the budget landed at exactly 128. D (bool packing) and E (Option<f32> flattening) were not required.
- **No `scale_factor()` method on OutputScale in recipe.rs:** The plan suggested adding a helper to prunr-core; to keep this plan file-disjoint from 32-01's prunr-core changes, inline match blocks were used in the 3 GUI call sites. Plan 32-07 will refactor the chip to use OutputScale natively.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed `adjustments_toolbar.rs` callers for u8-narrowed fields**
- **Found during:** Task 1 (workspace build after narrowing guided_radius and edge_thickness)
- **Issue:** `slider_row_u32` and `chip_u32` take `&mut u32` but fields are now `u8`
- **Fix:** Local u32 temp in `refine_edges` popover for `guided_radius`; block with temp for `edge_thickness` chip
- **Files modified:** `crates/prunr-app/src/gui/views/adjustments_toolbar.rs`
- **Verification:** `cargo build --workspace` clean
- **Committed in:** 07f69d6 (Task 1 commit)

**2. [Rule 3 - Blocking] Additional budget packing via Option<NonZeroU64> niche optimisation**
- **Found during:** Task 2 (budget test reported 144 bytes after adding 6 fields)
- **Issue:** D+E from RESEARCH.md (flags packing + threshold flattening) only saves 5 bytes; needed 16
- **Fix:** Changed `correction_hash` and `bg_image_hash` from `Option<u64>` to `Option<NonZeroU64>` — saves 16 bytes via Rust's niche optimisation. Updated 3 write sites in item.rs, 1 read site in app.rs
- **Files modified:** `item_settings.rs`, `item.rs`, `app.rs`
- **Verification:** `size_under_cache_line_budget` and `budget_still_under_128_bytes` both green at exactly 128 bytes
- **Committed in:** 04f232b (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking compilation and budget failures)
**Impact on plan:** Both fixes were necessary. The adjustments_toolbar bridge is the correct approach (plan 32-07 rewrites the chip properly). The NonZeroU64 packing is strictly superior to the RESEARCH.md options D+E — same budget savings, no API churn on threshold or booleans, transparent serde semantics.

## Issues Encountered

None beyond the two auto-fixed Rule 3 deviations above.

## Next Phase Readiness

- `ItemSettings` data layer complete: all seven Phase-32 knobs are stored, serialized, and wired through `current_recipe()` to `UpscaleRecipe`
- Plans 32-03 (denoise module) and 32-04 (postprocess.rs) are unblocked — they read from `UpscaleRecipe` fields, not from `ItemSettings` directly
- Plan 32-05 (dispatch): reads `output_scale` from the recipe, not from `ItemSettings`; unblocked
- Plan 32-07 (GUI chips): can bind directly to `pre_denoise`, `brightness_lift`, `sharpen`, `ai_blend`, `saturation`, `color_match` on `ItemSettings`

---
*Phase: 32-upscale-refinement-knobs*
*Completed: 2026-05-15*
