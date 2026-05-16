# 32-09 Diagnostic: Tier-2 Live-Preview Signal Drop

## Method

Static code analysis of the five candidate failure modes (a)–(e) from
32-VERIFICATION.md. All four seams are now instrumented with structured
`tracing::debug!` calls (committed in 32-09 Task 1). This file captures
the findings from that analysis.

## Failure Mode Identified: (b) — wrong `ModelKind` in recipe diff

**Location:** `crates/prunr-app/src/gui/app.rs`, `apply_toolbar_change`,
the Tier-2 detection block (previously lines 3402–3424, now updated).

**Root cause:**

```rust
// BEFORE (broken):
let model = self.settings.model
    .to_model_kind()
    .unwrap_or(prunr_core::ModelKind::BiRefNetLite);
```

`SettingsModel::to_model_kind()` returns `None` for upscale variants
(`RealEsrganUpscale`, `Nomos8kUpscale`). The `unwrap_or` falls back to
`BiRefNetLite`. This causes `item.settings.current_recipe(BiRefNetLite, …)`
to produce an `UpscaleRecipe` with `model: None` (the `current_recipe`
match arm for non-upscale model kinds).

At dispatch time (in `dispatch_upscale_intent`), the recipe is correctly
built with `model_kind = RealEsrganX4Plus`, producing `applied_recipe.upscale.model = Some(RealEsrganX4Plus)`.

On the next toolbar render, `resolve_tier` compares:
- `old.upscale.model = Some(RealEsrganX4Plus)`  (from `applied_recipe`)
- `new.upscale.model = None`                     (from the broken path)

`only_tier2_changed(old, new)` checks `old.model == new.model` first:
`Some(RealEsrganX4Plus) != None` → returns **false**.

`resolve_tier` therefore returns `UpscaleRerun`, never `UpscaleTier2`.
The `if matches!(tier, RequiredTier::UpscaleTier2)` gate never passes.
`mark_tweak` is never called. Live preview is silently dead.

## Secondary finding: `applied_recipe.upscale` never advanced

After a Tier-2 live preview lands, `apply_completed_previews` only updates
`applied_recipe.mask` (for the drift-tripwire). The `applied_recipe.upscale`
Tier-2 fields (sharpen/ai_blend/saturation/color_match) stay at their
Tier-1 dispatch values permanently.

This means: even after the model-kind fix, every frame that
`apply_toolbar_change` runs would see a non-zero diff in the four Tier-2
fields, re-fire `mark_tweak`, and produce a continuous preview loop at
DEBOUNCE rate after the user stops dragging. The loop is benign (previews
are correct) but wastes CPU and prevents `is_final = true` from settling.

**Fix (landed in Task 2):** Patch the four Tier-2 fields on
`applied_recipe.upscale` when a `PreviewResult` with
`applied_tier2_knobs = Some(…)` is drained. The `kind: PreviewKind`
field added to `PreviewResult` allows `apply_completed_previews` to
distinguish Tier-2 upscale results without an additional enum.

## Candidates ruled out

| Mode | Verdict | Reason |
|---|---|---|
| (a) refinement_row returns default ToolbarChange | Ruled out | `render_refinement_row` mutates `&mut ItemSettings` directly; `apply_toolbar_change` runs every frame unconditionally |
| (b) recipe diff resolves to wrong tier | **CONFIRMED** | `to_model_kind()` returns None for upscale → fallback BiRefNetLite → wrong upscale.model in new_recipe |
| (c) tick never selects the item | Not reached | mark_tweak never fires, so there is nothing to tick |
| (d) upscale_raw reset before tick | Not reached | same — mark_tweak never fires |
| (e) result_rgba swap doesn't fire | Not reached | dispatch never reaches run_preview |

## Tracing output if a user runs `RUST_LOG=prunr=debug` after the fix

With the fix applied, dragging a sharpen slider should produce:

```
upscale_tier2_check  has_upscale_raw=true  has_applied_recipe=true
upscale_tier2_resolve  resolved_tier=UpscaleTier2  old_sharpen_bits=X  new_sharpen_bits=Y
upscale_tier2_mark  "calling mark_tweak"
upscale_tier2_mark_tweak  kind=UpscaleTier2  is_new_entry=true
upscale_tier2_tick_ready  kind=UpscaleTier2
upscale_tier2_run_start  sharpen=…  ai_blend=…  saturation=…  color_match=…
upscale_tier2_run_done  elapsed_ms=…
upscale_tier2_apply  is_upscale_tier2=true  replaced_result_rgba=true  tex_rebuild_scheduled=true
```

Without the fix, only `upscale_tier2_check` and `upscale_tier2_resolve`
fire (with `resolved_tier=UpscaleRerun`). Nothing downstream executes.
