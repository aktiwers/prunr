//! Recipe types for tiered pipeline skip/re-apply.
//!
//! A "recipe" captures the exact settings used to produce a result.
//! When settings change, `resolve_tier()` determines which pipeline
//! tier needs to re-run (or if processing can be skipped entirely).

use serde::{Serialize, Deserialize};
use crate::types::{ModelKind, EdgeScale, ComposeMode, LineStyle, FillStyle, BgEffect, InputTransform};

/// Tier 1: settings that require AI model inference.
///
/// `line_strength` and `solid_line_color` are NOT here — those are applied
/// AFTER DexiNed in the edge postprocess stage and live in `EdgeRecipe`.
/// This lets a `line_strength` tweak fire a Tier 2 (EdgeRerun) instead of
/// a full re-inference, as long as the cached edge tensor is still valid
/// (i.e. `uses_edge_detection` hasn't changed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InferenceRecipe {
    pub model: ModelKind,
    /// Whether segmentation model runs (false for EdgesOnly mode).
    pub uses_segmentation: bool,
    /// Whether edge detection model runs (true when line_mode is not Off).
    pub uses_edge_detection: bool,
    /// Pre-inference image transform. Changing this invalidates the edge
    /// tensor cache because DexiNed sees different input.
    pub input_transform: InputTransform,
}

/// Tier 2 (edge variant): settings re-applied to the cached DexiNed tensor.
/// Changes here trigger an EdgeRerun, not a full pipeline — ~20-100ms vs 200ms-10s.
///
/// `edge_scale` lives here (not in `InferenceRecipe`) because all 4 scales are
/// extracted from a single DexiNed run; switching between them is a tensor
/// lookup, not a new inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EdgeRecipe {
    pub line_strength_bits: u32,
    pub solid_line_color: Option<[u8; 3]>,
    pub edge_thickness: u32,
    pub edge_scale: EdgeScale,
    pub compose_mode: ComposeMode,
    pub line_style: LineStyle,
}

impl From<&crate::EdgeSettings> for EdgeRecipe {
    fn from(e: &crate::EdgeSettings) -> Self {
        Self {
            line_strength_bits: e.line_strength.to_bits(),
            solid_line_color: e.solid_line_color,
            edge_thickness: e.edge_thickness,
            edge_scale: e.edge_scale,
            compose_mode: e.compose_mode,
            line_style: e.line_style,
        }
    }
}

/// Tier 2: mask postprocessing settings (gamma, threshold, edge refinement).
/// Uses `f32::to_bits()` for exact float comparison.
/// All fields are `Copy` types (u32, bool, enums), so `MaskRecipe: Copy`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaskRecipe {
    gamma_bits: u32,
    threshold_bits: Option<u32>,
    edge_shift_bits: u32,
    pub refine_edges: bool,
    guided_radius: u32,
    guided_epsilon_bits: u32,
    feather_bits: u32,
    pub fill_style: FillStyle,
    pub bg_effect: BgEffect,
    #[serde(default)]
    correction_hash: Option<u64>,
}

impl From<&crate::MaskSettings> for MaskRecipe {
    fn from(m: &crate::MaskSettings) -> Self {
        Self {
            gamma_bits: m.gamma.to_bits(),
            threshold_bits: m.threshold.map(|t| t.to_bits()),
            edge_shift_bits: m.edge_shift.to_bits(),
            refine_edges: m.refine_edges,
            guided_radius: m.guided_radius,
            guided_epsilon_bits: m.guided_epsilon.to_bits(),
            feather_bits: m.feather.to_bits(),
            fill_style: m.fill_style,
            bg_effect: m.bg_effect,
            correction_hash: m.correction_hash,
        }
    }
}

impl PartialEq for MaskRecipe {
    fn eq(&self, other: &Self) -> bool {
        self.gamma_bits == other.gamma_bits
            && self.threshold_bits == other.threshold_bits
            && self.edge_shift_bits == other.edge_shift_bits
            && self.refine_edges == other.refine_edges
            && self.guided_radius == other.guided_radius
            && self.guided_epsilon_bits == other.guided_epsilon_bits
            && self.feather_bits == other.feather_bits
            && self.fill_style == other.fill_style
            && self.bg_effect == other.bg_effect
            && self.correction_hash == other.correction_hash
    }
}
impl Eq for MaskRecipe {}

impl std::hash::Hash for MaskRecipe {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.gamma_bits.hash(state);
        self.threshold_bits.hash(state);
        self.edge_shift_bits.hash(state);
        self.refine_edges.hash(state);
        self.guided_radius.hash(state);
        self.guided_epsilon_bits.hash(state);
        self.feather_bits.hash(state);
        self.fill_style.hash(state);
        self.bg_effect.hash(state);
        self.correction_hash.hash(state);
    }
}

/// User-facing output scale choice for upscale dispatches.
///
/// `X4TwoPass` is Real-ESRGAN-only (runs `RealEsrganX2Plus` twice);
/// the dispatch layer rejects it for non-x4plus models. `#[repr(u8)]`
/// keeps the enum at 1 byte so it fits inside the `ItemSettings`
/// cache-line budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum OutputScale {
    X2 = 0,
    X3 = 1,
    X4 = 2,
    X4TwoPass = 3,
}

impl Default for OutputScale {
    /// `X4` = native model output. This is the no-op default (the upscale model is trained at 4×).
    fn default() -> Self { OutputScale::X4 }
}

impl OutputScale {
    /// User-facing multiplicative output factor. `X4TwoPass` runs the model
    /// twice but still produces a 4× image — the factor is the *display*
    /// factor the user sees, not the inference count.
    #[inline]
    pub fn factor(self) -> u32 {
        match self {
            OutputScale::X2 => 2,
            OutputScale::X3 => 3,
            OutputScale::X4 | OutputScale::X4TwoPass => 4,
        }
    }
}

/// Upscale settings. `None` model means upscale is disabled.
///
/// All f32 knobs are stored as `u32::from_bits` for safe `PartialEq` + `Hash`
/// (matches the `MaskRecipe::gamma_bits` pattern).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UpscaleRecipe {
    /// `None` = upscale disabled (no upscale model selected).
    pub model: Option<prunr_models::ModelId>,
    /// User-facing output scale (replaces the prior `scale: u32`).
    pub output_scale: OutputScale,
    // ----- Tier-1 knobs (re-inference) -----
    /// Pre-inference denoise blend strength, range [0.0, 1.0].
    pub pre_denoise_bits: u32,
    /// Pre-inference EV-stop exposure adjustment, range [-2.0, 2.0].
    pub brightness_lift_bits: u32,
    // ----- Tier-2 knobs (postprocess only) -----
    /// Unsharp-mask strength, range [-1.0, 1.0]. Negative = blur.
    pub sharpen_bits: u32,
    /// Lerp weight toward AI output vs bicubic source, range [0.0, 1.0].
    /// 1.0 = full AI (default), 0.0 = pure bicubic.
    pub ai_blend_bits: u32,
    /// HSL-space saturation adjustment, range [-1.0, 1.0].
    pub saturation_bits: u32,
    /// Reinhard Lab mean+stddev color match against source.
    pub color_match: bool,
}

impl Default for UpscaleRecipe {
    fn default() -> Self {
        Self {
            model: None,
            output_scale: OutputScale::X4,
            pre_denoise_bits: 0_u32,          // 0.0_f32.to_bits() == 0
            brightness_lift_bits: 0_u32,
            sharpen_bits: 0.2_f32.to_bits(),  // mild default sharpen counters Real-ESRGAN softness
            ai_blend_bits: 1.0_f32.to_bits(), // default = full AI; 0 = pure bicubic (disables AI output)
            saturation_bits: 0_u32,
            color_match: false,
        }
    }
}

impl UpscaleRecipe {
    #[inline] pub fn pre_denoise(&self) -> f32 { f32::from_bits(self.pre_denoise_bits) }
    #[inline] pub fn brightness_lift(&self) -> f32 { f32::from_bits(self.brightness_lift_bits) }
    #[inline] pub fn sharpen(&self) -> f32 { f32::from_bits(self.sharpen_bits) }
    #[inline] pub fn ai_blend(&self) -> f32 { f32::from_bits(self.ai_blend_bits) }
    #[inline] pub fn saturation(&self) -> f32 { f32::from_bits(self.saturation_bits) }
}

/// Tier 3: compositing settings (bg color, bg image).
/// These can be applied without re-running inference or masking.
///
/// `solid_line_color` deliberately lives in `EdgeRecipe`, not here — a
/// line-colour change is an EdgeRerun (re-threshold the cached DexiNed
/// tensor), never a composite-only repaint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct CompositeRecipe {
    pub bg_color: Option<[u8; 3]>,
    /// `Some` means a bg image is active; `None` means no image bg. The
    /// caller hashes the image once on pick so the recipe diff can detect
    /// change with a u64 compare instead of re-hashing every dispatch.
    #[serde(default)]
    pub bg_image_hash: Option<u64>,
    /// How the bg image is positioned in the frame (Cover / Contain /
    /// Stretch / Tile / Center). Independent of `bg_image_hash` so a fit
    /// change with the same image still flips the recipe.
    #[serde(default)]
    pub bg_image_fit: crate::types::BgImageFit,
}

/// Complete recipe — the full fingerprint of settings that produced a result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProcessingRecipe {
    pub inference: InferenceRecipe,
    pub edge: EdgeRecipe,
    pub mask: MaskRecipe,
    pub composite: CompositeRecipe,
    /// True if this result was produced in chain mode (previous result as input).
    pub was_chain: bool,
    #[serde(default)]
    pub upscale: UpscaleRecipe,
}

/// What processing tier is needed to go from old settings to new settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredTier {
    /// Nothing changed — skip entirely.
    Skip,
    /// Only compositing changed (bg_color) — parent-local, instant.
    CompositeOnly,
    /// Only Tier-2 upscale fields (sharpen / ai_blend / saturation /
    /// color_match) changed — re-apply postprocess on the cached
    /// `upscale_raw` buffer; no re-inference needed.
    UpscaleTier2,
    /// Only the upscale recipe changed (Tier-1 fields) — re-composite then
    /// re-upscale; no mask or inference work needed.
    UpscaleRerun,
    /// Mask settings changed — re-run postprocess from cached segmentation tensor (~200ms).
    MaskRerun,
    /// Edge settings changed — re-threshold cached DexiNed tensor (~20-100ms).
    EdgeRerun,
    /// Turning edge detection on while the segmentation side is unchanged.
    /// Reuse the cached seg tensor, run only DexiNed on the masked image,
    /// compose (~DexiNed inference time, no seg re-inference).
    AddEdgeInference,
    /// Model / mode changed — full pipeline needed.
    FullPipeline,
}

/// True when two `UpscaleRecipe` values differ ONLY in Tier-2 fields
/// (sharpen, ai_blend, saturation, color_match). Used by `resolve_tier`
/// to pick between `UpscaleTier2` (cheap, postprocess-only) and
/// `UpscaleRerun` (re-inference).
fn only_tier2_changed(old: &UpscaleRecipe, new: &UpscaleRecipe) -> bool {
    // Tier-1 fields MUST be identical to qualify as Tier-2.
    old.model == new.model
        && old.output_scale == new.output_scale
        && old.pre_denoise_bits == new.pre_denoise_bits
        && old.brightness_lift_bits == new.brightness_lift_bits
}

/// Determine what processing tier is needed when changing from old to new recipe.
///
/// Cost ordering (cheapest → most expensive), matching the `ordered()` table
/// in `knob_catalog.rs` and the top-down "most-expensive-wins" checks below:
/// Skip < CompositeOnly < UpscaleTier2 < UpscaleRerun < EdgeRerun < MaskRerun < AddEdgeInference < FullPipeline.
pub fn resolve_tier(old: &ProcessingRecipe, new: &ProcessingRecipe) -> RequiredTier {
    if old == new {
        return RequiredTier::Skip;
    }
    // Chain mode changes the input image, not settings — always needs full pipeline.
    if old.was_chain != new.was_chain {
        return RequiredTier::FullPipeline;
    }
    if old.inference != new.inference {
        // Model change forces everything (seg tensor from old model is wrong shape / scale).
        if old.inference.model != new.inference.model {
            return RequiredTier::FullPipeline;
        }
        // Seg-flag flip swaps between modes whose cached tensors aren't
        // interchangeable (e.g. EdgesOnly's edge tensor is on the raw input,
        // SubjectOutline's is on the masked input — mixing them produces
        // nonsense). Only safe for the edge-flag-only case below.
        if old.inference.uses_segmentation != new.inference.uses_segmentation {
            return RequiredTier::FullPipeline;
        }
        match (old.inference.uses_edge_detection, new.inference.uses_edge_detection) {
            // Enabling edges (or already on with `input_transform` flipped):
            // seg tensor cached → skip seg inference, run DexiNed on the
            // masked image.
            (false, true) | (true, true) => return RequiredTier::AddEdgeInference,
            // Disabling edges: seg tensor still cached. Concurrent edge-recipe
            // or input_transform drift is dead state — regenerate the mask
            // without lines instead of dispatching wasted EdgeRerun work.
            (true, false) => return RequiredTier::MaskRerun,
            // Both edges off: only `input_transform` could differ here, and
            // it's dead state with edges off (the field's only effect is on
            // the DexiNed input). Fall through to mask/edge/composite checks
            // so a simultaneous mask/edge/composite change still wins; if
            // nothing else changed the bottom of the function returns Skip.
            (false, false) => {}
        }
    }
    if old.mask != new.mask {
        return RequiredTier::MaskRerun;
    }
    if old.edge != new.edge {
        return RequiredTier::EdgeRerun;
    }
    if old.upscale != new.upscale {
        if only_tier2_changed(&old.upscale, &new.upscale) {
            return RequiredTier::UpscaleTier2;
        }
        return RequiredTier::UpscaleRerun;
    }
    // Composite changes (bg color) do not invalidate the cached upscale — user re-runs Process for upscale-on-new-bg.
    if old.composite != new.composite {
        return RequiredTier::CompositeOnly;
    }
    // Reachable only via the (false, false) input_transform fall-through:
    // recipes differ but only in dead-state input_transform. Output is
    // bit-identical so the cheapest tier — Skip — is correct.
    RequiredTier::Skip
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a MaskRecipe for tests — other MaskSettings fields default.
    fn mask(gamma: f32, threshold: Option<f32>, edge_shift: f32, refine: bool) -> MaskRecipe {
        MaskRecipe::from(&crate::MaskSettings {
            gamma,
            threshold,
            edge_shift,
            refine_edges: refine,
            ..Default::default()
        })
    }

    fn make_recipe(model: ModelKind, gamma: f32, bg: Option<[u8; 3]>) -> ProcessingRecipe {
        ProcessingRecipe {
            inference: InferenceRecipe {
                model,
                uses_segmentation: true,
                uses_edge_detection: false,
                input_transform: InputTransform::None,
            },
            edge: EdgeRecipe {
                line_strength_bits: 0.5f32.to_bits(),
                solid_line_color: None,
                edge_thickness: 0,
                edge_scale: EdgeScale::Fused,
                compose_mode: ComposeMode::LinesOnly,
                line_style: LineStyle::Solid,
            },
            mask: mask(gamma, None, 0.0, false),
            composite: CompositeRecipe {
                bg_color: bg,
                bg_image_hash: None,
                bg_image_fit: crate::types::BgImageFit::default(),
            },
            upscale: UpscaleRecipe::default(),
            was_chain: false,
        }
    }

    #[test]
    fn identical_recipes_skip() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = a.clone();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::Skip);
    }

    #[test]
    fn bg_color_change_composite_only() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = make_recipe(ModelKind::Silueta, 1.0, Some([255, 0, 0]));
        assert_eq!(resolve_tier(&a, &b), RequiredTier::CompositeOnly);
    }

    #[test]
    fn bg_image_hash_change_composite_only() {
        // bg_image is a render-time / save-time texture — same tier
        // class as bg_color (no inference, no postprocess).
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = make_recipe(ModelKind::Silueta, 1.0, None);
        b.composite.bg_image_hash = Some(0xdeadbeef);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::CompositeOnly);
    }

    #[test]
    fn bg_image_fit_change_composite_only() {
        // Same image, different fit (Cover → Tile) — pure render math change.
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = make_recipe(ModelKind::Silueta, 1.0, None);
        b.composite.bg_image_fit = crate::types::BgImageFit::Tile;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::CompositeOnly);
    }

    #[test]
    fn gamma_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = make_recipe(ModelKind::Silueta, 1.5, None);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn model_change_full_pipeline() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = make_recipe(ModelKind::BiRefNetLite, 1.0, None);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::FullPipeline);
    }

    #[test]
    fn chain_mode_change_full_pipeline() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.was_chain = true;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::FullPipeline);
    }

    #[test]
    fn float_bits_equality() {
        let a = mask(1.0, Some(0.5), 0.0, true);
        let b = mask(1.0, Some(0.5), 0.0, true);
        assert_eq!(a, b);
    }

    #[test]
    fn float_bits_inequality() {
        let a = mask(1.0, Some(0.5), 0.0, true);
        let b = mask(1.0, Some(0.50001), 0.0, true);
        assert_ne!(a, b);
    }

    #[test]
    fn line_strength_change_edge_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.edge.line_strength_bits = 0.8f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::EdgeRerun);
    }

    fn mask_with_correction(hash: Option<u64>) -> MaskRecipe {
        MaskRecipe::from(&crate::MaskSettings {
            correction_hash: hash,
            ..Default::default()
        })
    }

    #[test]
    fn correction_hash_some_to_other_some_triggers_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask_with_correction(Some(42));
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn equal_correction_hashes_skip() {
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.mask = mask_with_correction(Some(7));
        let b = a.clone();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::Skip);
    }

    #[test]
    fn correction_hash_none_to_some_triggers_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask_with_correction(Some(1));
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn correction_hash_some_to_none_triggers_mask_rerun() {
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.mask = mask_with_correction(Some(99));
        let b = make_recipe(ModelKind::Silueta, 1.0, None);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn solid_line_color_change_edge_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.edge.solid_line_color = Some([255, 0, 0]);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::EdgeRerun);
    }

    #[test]
    fn edge_scale_change_edge_rerun() {
        // Scale change picks a different cached output from the same DexiNed
        // run — an EdgeRerun (pick a different tensor), never FullPipeline.
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.edge.edge_scale = EdgeScale::Bold;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::EdgeRerun);
    }

    #[test]
    fn edge_thickness_change_edge_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.edge.edge_thickness = 3;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::EdgeRerun);
    }

    #[test]
    fn threshold_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask(1.0, Some(0.5), 0.0, false);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn edge_shift_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask(1.0, None, 2.0, false);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn refine_edges_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask(1.0, None, 0.0, true);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn enable_edge_detection_with_seg_stable_is_add_edge_inference() {
        // Off → SubjectOutline: uses_segmentation stays true, uses_edge_detection
        // flips false → true. Seg tensor cached → skip seg re-inference.
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.inference.uses_edge_detection = true;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::AddEdgeInference);
    }

    #[test]
    fn disable_edge_detection_with_seg_stable_is_mask_rerun() {
        // SubjectOutline → Off: uses_segmentation stays true, uses_edge_detection
        // flips true → false. Seg tensor still cached; MaskRerun regenerates the
        // result without edges.
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.inference.uses_edge_detection = true;
        let mut b = a.clone();
        b.inference.uses_edge_detection = false;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn seg_flag_flip_still_full_pipeline() {
        // EdgesOnly ↔ SubjectOutline (or Off ↔ EdgesOnly): uses_segmentation
        // flips — cached tensors from one mode aren't valid for the other, so
        // the tier must drop all caches and re-infer.
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.inference.uses_segmentation = false;
        a.inference.uses_edge_detection = true; // EdgesOnly
        let mut b = a.clone();
        b.inference.uses_segmentation = true;
        b.inference.uses_edge_detection = true; // SubjectOutline
        assert_eq!(resolve_tier(&a, &b), RequiredTier::FullPipeline);
    }

    #[test]
    fn enable_edge_with_mask_change_still_add_edge_inference() {
        // AddEdgeInference takes priority over MaskRerun when both apply —
        // the worker handles this by re-masking from the cached seg as part
        // of the AddEdgeInference flow anyway, so a separate MaskRerun would
        // be redundant.
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.inference.uses_edge_detection = true;
        b.mask = mask(1.5, None, 0.0, false); // mask also changed
        assert_eq!(resolve_tier(&a, &b), RequiredTier::AddEdgeInference);
    }

    #[test]
    fn edges_only_mode_still_honours_edge_rerun() {
        // EdgesOnly: uses_segmentation = false. Mask fields become no-ops at
        // the ItemSettings → ProcessingRecipe boundary, but if a recipe directly
        // toggles line_strength, resolve_tier still returns EdgeRerun.
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.inference.uses_segmentation = false;
        a.inference.uses_edge_detection = true;
        let mut b = a.clone();
        b.edge.line_strength_bits = 0.2f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::EdgeRerun);
    }

    #[test]
    fn input_transform_change_with_edges_on_is_add_edge_inference() {
        // Per InferenceRecipe.input_transform doc-comment: changing the
        // pre-inference image transform invalidates the edge tensor cache
        // because DexiNed sees different input. With seg + edges both on
        // and the model unchanged, the tier must re-run DexiNed only —
        // seg tensor stays valid because seg is run on the original image
        // before the transform reaches the edge leg.
        let mut a = make_recipe(ModelKind::Silueta, 1.0, None);
        a.inference.uses_edge_detection = true;
        let mut b = a.clone();
        b.inference.input_transform = InputTransform::Grayscale;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::AddEdgeInference);
    }

    /// Table-driven test: single source of truth for tier priority ordering.
    /// When multiple things change, the highest-cost tier wins.
    #[test]
    fn tier_resolution_priority_table() {
        let base = make_recipe(ModelKind::Silueta, 1.0, None);

        let with_gamma = |g: f32| {
            let mut r = base.clone();
            r.mask = mask(g, None, 0.0, false);
            r
        };
        let with_line = |s: f32| {
            let mut r = base.clone();
            r.edge.line_strength_bits = s.to_bits();
            r
        };
        let with_bg = |rgb: [u8; 3]| {
            let mut r = base.clone();
            r.composite.bg_color = Some(rgb);
            r
        };
        // Edges-on baseline for input_transform cases (the transform only
        // affects the edge leg; with edges off the transform is dead state).
        let edges_on = || {
            let mut r = base.clone();
            r.inference.uses_edge_detection = true;
            r
        };
        let with_input_transform = |it: InputTransform| {
            let mut r = edges_on();
            r.inference.input_transform = it;
            r
        };

        // (name, old, new, expected)
        let cases: Vec<(&str, ProcessingRecipe, ProcessingRecipe, RequiredTier)> = vec![
            ("identical", base.clone(), base.clone(), RequiredTier::Skip),
            ("bg only", base.clone(), with_bg([255, 0, 0]), RequiredTier::CompositeOnly),
            ("gamma only", base.clone(), with_gamma(1.5), RequiredTier::MaskRerun),
            ("line only", base.clone(), with_line(0.9), RequiredTier::EdgeRerun),
            ("model only", base.clone(), make_recipe(ModelKind::BiRefNetLite, 1.0, None), RequiredTier::FullPipeline),
            ("input_transform+edges → add_edge", edges_on(), with_input_transform(InputTransform::Grayscale), RequiredTier::AddEdgeInference),
            // input_transform with edges off both sides: dead state (the
            // field only affects the DexiNed input), so output is bit-
            // identical → Skip rather than MaskRerun.
            ("input_transform alone, edges off → skip", base.clone(), {
                let mut r = base.clone();
                r.inference.input_transform = InputTransform::Grayscale;
                r
            }, RequiredTier::Skip),
            // Edges on→off with concurrent edge-recipe drift: the edge
            // recipe is dead state once edges are off, so MaskRerun beats
            // EdgeRerun (no point re-composing edges that won't render).
            ("edges on→off + line_strength → mask", edges_on(), {
                let mut r = edges_on();
                r.inference.uses_edge_detection = false;
                r.edge.line_strength_bits = 0.9f32.to_bits();
                r
            }, RequiredTier::MaskRerun),
            // Priority: mask changes dominate composite changes.
            ("gamma+bg → mask", base.clone(), {
                let mut r = with_gamma(1.5);
                r.composite.bg_color = Some([9, 9, 9]);
                r
            }, RequiredTier::MaskRerun),
            // Priority: edge changes dominate composite changes.
            ("line+bg → edge", base.clone(), {
                let mut r = with_line(0.2);
                r.composite.bg_color = Some([9, 9, 9]);
                r
            }, RequiredTier::EdgeRerun),
            // Priority: AddEdgeInference dominates mask + edge + composite.
            ("input_transform+gamma+line+bg → add_edge", edges_on(), {
                let mut r = with_input_transform(InputTransform::Grayscale);
                r.mask = mask(1.5, None, 0.0, false);
                r.edge.line_strength_bits = 0.2f32.to_bits();
                r.composite.bg_color = Some([9, 9, 9]);
                r
            }, RequiredTier::AddEdgeInference),
            // Priority: model change dominates everything.
            ("model+gamma+line+bg → full", base.clone(), {
                let mut r = make_recipe(ModelKind::U2net, 1.5, Some([9, 9, 9]));
                r.edge.line_strength_bits = 0.2f32.to_bits();
                r
            }, RequiredTier::FullPipeline),
        ];

        for (name, old, new, expected) in &cases {
            assert_eq!(
                resolve_tier(old, new),
                *expected,
                "table row {name:?}: expected {expected:?}",
            );
        }
    }

    #[test]
    fn guided_radius_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = MaskRecipe::from(&crate::MaskSettings {
            guided_radius: 16,
            ..Default::default()
        });
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn guided_epsilon_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = MaskRecipe::from(&crate::MaskSettings {
            guided_epsilon: 2e-3,
            ..Default::default()
        });
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn feather_change_mask_rerun() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = MaskRecipe::from(&crate::MaskSettings {
            feather: 1.5,
            ..Default::default()
        });
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    fn upscale_with_scale(model: Option<prunr_models::ModelId>, scale: OutputScale) -> UpscaleRecipe {
        UpscaleRecipe { model, output_scale: scale, ..UpscaleRecipe::default() }
    }

    #[test]
    fn upscale_rerun_when_only_upscale_recipe_changed() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.upscale = upscale_with_scale(Some(prunr_models::ModelId::RealEsrganX4Plus), OutputScale::X4);
        let mut c = a.clone();
        c.upscale = upscale_with_scale(Some(prunr_models::ModelId::RealEsrganX4Plus), OutputScale::X2);
        assert_eq!(resolve_tier(&b, &c), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn upscale_rerun_when_only_upscale_model_changed() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut old = a.clone();
        old.upscale = upscale_with_scale(Some(prunr_models::ModelId::Nomos8kSchatL), OutputScale::X4);
        let mut new = a.clone();
        new.upscale = upscale_with_scale(Some(prunr_models::ModelId::RealEsrganX4Plus), OutputScale::X4);
        assert_eq!(resolve_tier(&old, &new), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn mask_rerun_when_mask_and_upscale_both_changed() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = a.clone();
        b.mask = mask(2.2, None, 0.0, false);
        b.upscale = upscale_with_scale(Some(prunr_models::ModelId::RealEsrganX4Plus), OutputScale::X2);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::MaskRerun);
    }

    #[test]
    fn composite_only_when_only_composite_changed_upscale_unchanged() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = make_recipe(ModelKind::Silueta, 1.0, Some([0, 0, 0]));
        assert_eq!(resolve_tier(&a, &b), RequiredTier::CompositeOnly);
    }

    #[test]
    fn skip_when_upscale_disabled_and_other_recipe_identical() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let b = a.clone();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::Skip);
    }

    #[test]
    fn upscale_rerun_outranks_composite_when_both_changed() {
        let a = make_recipe(ModelKind::Silueta, 1.0, None);
        let mut b = make_recipe(ModelKind::Silueta, 1.0, Some([255, 0, 0]));
        b.upscale = upscale_with_scale(Some(prunr_models::ModelId::RealEsrganX4Plus), OutputScale::X4);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn output_scale_size_is_one_byte() {
        assert_eq!(std::mem::size_of::<OutputScale>(), 1);
    }

    #[test]
    fn output_scale_serde_roundtrip() {
        for variant in [OutputScale::X2, OutputScale::X3, OutputScale::X4, OutputScale::X4TwoPass] {
            let json = serde_json::to_string(&variant).unwrap();
            let recovered: OutputScale = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, recovered, "{variant:?}");
        }
    }

    #[test]
    fn upscale_recipe_default_matches_existing_behaviour() {
        let r = UpscaleRecipe::default();
        assert_eq!(r.output_scale, OutputScale::X4);
        assert_eq!(r.pre_denoise_bits, 0_u32);
        assert_eq!(r.brightness_lift_bits, 0_u32);
        assert_eq!(r.sharpen_bits, 0.2_f32.to_bits());
        assert_eq!(r.ai_blend_bits, 1.0_f32.to_bits());
        assert_eq!(r.saturation_bits, 0_u32);
        assert!(!r.color_match);
    }

    #[test]
    fn upscale_recipe_default_sharpen_is_0_2() {
        let r = UpscaleRecipe::default();
        assert_eq!(r.sharpen_bits, 0.2_f32.to_bits(),
            "default sharpen=0.2 is the perceived-quality baseline for Real-ESRGAN's slightly soft output");
        assert_eq!(r.sharpen(), 0.2);
    }

    #[test]
    fn upscale_recipe_bit_fields_eq_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = UpscaleRecipe {
            pre_denoise_bits: 0.5_f32.to_bits(),
            ..UpscaleRecipe::default()
        };
        let b = UpscaleRecipe {
            pre_denoise_bits: 0.5_f32.to_bits(),
            ..UpscaleRecipe::default()
        };
        assert_eq!(a, b);

        let hash_of = |r: &UpscaleRecipe| {
            let mut h = DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(&a), hash_of(&b));
    }


    fn make_upscale_recipe_with_model() -> ProcessingRecipe {
        let mut r = make_recipe(ModelKind::Silueta, 1.0, None);
        r.upscale = UpscaleRecipe {
            model: Some(prunr_models::ModelId::RealEsrganX4Plus),
            ..UpscaleRecipe::default()
        };
        r
    }

    #[test]
    fn upscale_tier1_change_returns_upscale_rerun() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.pre_denoise_bits = 0.6_f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn brightness_lift_change_returns_upscale_rerun() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.brightness_lift_bits = (-0.5_f32).to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn output_scale_change_returns_upscale_rerun() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.output_scale = OutputScale::X2;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn sharpen_change_returns_upscale_tier2() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.sharpen_bits = 0.5_f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleTier2);
    }

    #[test]
    fn ai_blend_change_returns_upscale_tier2() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.ai_blend_bits = 0.4_f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleTier2);
    }

    #[test]
    fn saturation_change_returns_upscale_tier2() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.saturation_bits = 0.3_f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleTier2);
    }

    #[test]
    fn color_match_toggle_returns_upscale_tier2() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.color_match = true;
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleTier2);
    }

    #[test]
    fn mixed_tier1_and_tier2_change_returns_upscale_rerun() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.pre_denoise_bits = 0.4_f32.to_bits();
        b.upscale.sharpen_bits = 0.5_f32.to_bits();
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }

    #[test]
    fn model_swap_within_upscale_returns_upscale_rerun() {
        let a = make_upscale_recipe_with_model();
        let mut b = a.clone();
        b.upscale.model = Some(prunr_models::ModelId::Nomos8kSchatL);
        assert_eq!(resolve_tier(&a, &b), RequiredTier::UpscaleRerun);
    }
}
