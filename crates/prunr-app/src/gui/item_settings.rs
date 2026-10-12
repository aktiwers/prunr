//! Per-image processing settings.
//!
//! Each `BatchItem` owns an `ItemSettings`. The toolbar binds to the current
//! image's settings; tweaking a knob updates only that image. `AppSettings`
//! holds app-wide config (parallel jobs, history depth, hotkeys, presets).
//!
//! `ItemSettings` is `Copy` and contains no heap allocations — diffing and
//! snapshotting are branchless and free. Building a `ProcessingRecipe` from
//! an `ItemSettings` is a trivial field copy.
//!
//! Target size: under 40 bytes to fit in a single cache line alongside the
//! BatchItem's lightweight metadata fields.

use prunr_core::{LineMode, MaskSettings, EdgeSettings, EdgeScale, BgImageFit};
use serde::{Deserialize, Serialize};

/// Per-image processing settings. Edited via the adjustments toolbar.
///
/// Invariants:
/// - All fields are `Copy`. Diff is a branchless struct compare.
/// - `bg` uses RGBA for UI parity with egui color pickers; only RGB is sent
///   into the pipeline (alpha is always treated as 255 during compositing).
///
/// Serialization contract — IMPORTANT when adding fields:
/// Preset files on disk carry one `ItemSettings` as JSON. Users share them,
/// so files written by old builds MUST keep loading when we add fields.
/// The `#[serde(default)]` attribute on this struct makes missing fields
/// fall back to `Default::default()`, so adding a new field is safe *as
/// long as `Default` covers it*. Do NOT add `#[serde(deny_unknown_fields)]`
/// — the tripwire tests in `presets_fs::tests` enforce this contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ItemSettings {
    /// Gamma curve applied to the mask. >1.0 = more aggressive removal, <1.0 = gentler.
    pub gamma: f32,
    /// Edge contrast (0.0-1.0): 0 keeps the soft edge, 1 cuts it hard.
    pub contrast: f32,
    /// Edge shift in pixels: >0 erodes, <0 dilates.
    pub edge_shift: f32,
    /// Guided-filter refinement of mask edges.
    pub refine_edges: bool,
    /// Guided filter window radius (pixels). Only used when refine_edges.
    /// Range 4-32 in the UI; stored as u8 to fit the ItemSettings byte budget.
    pub guided_radius: u8,
    /// Guided filter regularization. Only used when refine_edges.
    pub guided_epsilon: f32,
    /// Gaussian blur sigma applied to mask (softens edges, color-agnostic).
    pub feather: f32,

    /// Line extraction mode.
    pub line_mode: LineMode,
    /// DexiNed edge sensitivity (0.0-1.0).
    pub line_strength: f32,
    /// Solid color override for edges. `None` = preserve original RGB.
    pub solid_line_color: Option<[u8; 3]>,
    /// Line width: the edge mask grown by N pixels. Range 0-20 in the UI;
    /// stored as u8 to fit the ItemSettings byte budget.
    pub edge_thickness: u8,
    /// Which DexiNed output scale to read from the cached tensor set.
    pub edge_scale: EdgeScale,

    /// Fill transparent areas with a solid color. `None` = transparent.
    /// Stored as RGBA for UI parity; pipeline only uses RGB.
    pub bg: Option<[u8; 4]>,

    /// How the subject mask and edge mask combine in SubjectOutline mode.
    /// Ignored otherwise. Serde-defaulted for backwards compat with presets
    /// saved before this field existed.
    #[serde(default)]
    pub compose_mode: prunr_core::ComposeMode,
    /// Line colouring: solid (via `solid_line_color`) or a gradient.
    #[serde(default)]
    pub line_style: prunr_core::LineStyle,
    /// RGB transformation applied to the masked subject before compose.
    #[serde(default)]
    pub fill_style: prunr_core::FillStyle,
    /// Sticker outline and shadow around the cut-out.
    #[serde(default)]
    pub decoration: prunr_core::Decoration,
    /// A picture made from the photo, shown behind the result.
    #[serde(default)]
    pub bg_effect: prunr_core::BgEffect,
    /// Transform applied to the image before DexiNed inference. A change
    /// drops the edge tensor cache and reruns the detector at once (on the
    /// cached cut-out when there is one).
    #[serde(default)]
    pub input_transform: prunr_core::InputTransform,
    /// Drives the CompositeOnly recipe diff when the user picks a different
    /// background image. `None` when no image bg is set.
    ///
    /// `NonZeroU64` niche saves 8 bytes vs `Option<u64>`. Hash=0 is treated
    /// as None (content_hash never yields 0 in practice).
    #[serde(default)]
    pub bg_image_hash: Option<std::num::NonZeroU64>,
    /// How the background image is positioned/scaled inside the result frame.
    /// Default `Cover` matches CSS `background-size: cover`.
    #[serde(default)]
    pub bg_image_fit: BgImageFit,

    /// User-facing output scale. Older presets that carried `upscale_scale: u32`
    /// silently drop the field via `#[serde(default)]` — they reload at
    /// `OutputScale::X4` (the no-op default matching the prior native upscale
    /// model behaviour).
    #[serde(default)]
    pub output_scale: prunr_core::OutputScale,
    /// Pre-inference classical denoise blend strength, range [0.0, 1.0].
    /// 0.0 = no denoise (output bit-identical to the un-touched source).
    #[serde(default)]
    pub pre_denoise: f32,
    /// Unsharp-mask strength applied after inference, range [-1.0, 1.0].
    /// Negative = blur, 0.0 = no-op, positive = sharpen.
    #[serde(default = "default_sharpen")]
    pub sharpen: f32,
    /// Lerp weight: 1.0 = full AI upscale (default), 0.0 = pure bicubic.
    /// Default is 1.0 so a preset that omits this field reads as full AI —
    /// the no-op state where the upscale model's output passes through.
    #[serde(default = "default_ai_blend")]
    pub ai_blend: f32,
    /// Reinhard Lab mean+stddev color match against source RGB.
    #[serde(default)]
    pub color_match: bool,
    /// Restore faces: on or off, with which model and how strongly.
    #[serde(default)]
    pub faces: FaceSettings,
}

/// The face restorer a run uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceModel {
    #[default]
    Gfpgan,
    RestoreFormerPp,
}

impl FaceModel {
    pub const ALL: [Self; 2] = [Self::Gfpgan, Self::RestoreFormerPp];

    pub fn id(self) -> prunr_models::ModelId {
        match self {
            Self::Gfpgan => prunr_models::ModelId::FaceGfpgan,
            Self::RestoreFormerPp => prunr_models::ModelId::FaceRestoreFormerPp,
        }
    }
}

/// Restore faces. `on` and `model` decide whether faces are found and
/// redrawn (a model run); the rest only how they go back, which is live.
/// Strength is whole percent to fit the ItemSettings byte budget.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceSettings {
    pub on: bool,
    pub model: FaceModel,
    pub strength_pct: u8,
    pub largest_only: bool,
    pub hair: bool,
}

impl Default for FaceSettings {
    fn default() -> Self {
        Self { on: false, model: FaceModel::default(), strength_pct: 50, largest_only: false, hair: false }
    }
}

impl FaceSettings {
    pub fn blend(&self) -> prunr_core::face::FaceBlend {
        prunr_core::face::FaceBlend {
            strength: f32::from(self.strength_pct.min(100)) / 100.0,
            largest_only: self.largest_only,
            hair: self.hair,
        }
    }

    /// The faces to find, when any: the model to find them with.
    pub fn wanted(&self) -> Option<prunr_models::ModelId> {
        self.on.then(|| self.model.id())
    }
}

fn default_sharpen() -> f32 { prunr_core::DEFAULT_SHARPEN }
fn default_ai_blend() -> f32 { 1.0 }

impl Default for ItemSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            contrast: 0.0,
            edge_shift: 0.0,
            refine_edges: false,
            guided_radius: 8,
            guided_epsilon: 1e-4,
            feather: 0.0,
            line_mode: LineMode::Off,
            line_strength: 0.5,
            solid_line_color: None,
            edge_thickness: 0,
            edge_scale: EdgeScale::Fused,
            bg: None,
            compose_mode: prunr_core::ComposeMode::default(),
            line_style: prunr_core::LineStyle::default(),
            fill_style: prunr_core::FillStyle::default(),
            decoration: prunr_core::Decoration::default(),
            bg_effect: prunr_core::BgEffect::default(),
            input_transform: prunr_core::InputTransform::default(),
            bg_image_hash: None,
            bg_image_fit: BgImageFit::default(),
            output_scale: prunr_core::OutputScale::X4,
            pre_denoise: 0.0,
            sharpen: prunr_core::DEFAULT_SHARPEN,
            ai_blend: 1.0,
            color_match: false,
            faces: FaceSettings::default(),
        }
    }
}

/// What sits behind a result: shown under it on screen and baked in on
/// export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backdrop {
    Transparent,
    Color([u8; 3]),
    Image(std::num::NonZeroU64),
    Effect(prunr_core::BgEffect),
}

impl Backdrop {
    /// How a picture backdrop sits in the frame: an effect is made from the
    /// photo itself, so it covers exactly; an image follows its Fit.
    pub fn fit(self, image_fit: BgImageFit) -> BgImageFit {
        if matches!(self, Backdrop::Image(_)) { image_fit } else { BgImageFit::Cover }
    }

    /// What a picture backdrop shows, to tell whether a built texture is
    /// still current; `None` for no picture.
    pub fn picture_key(self) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        match self {
            Backdrop::Image(hash) => Some(hash.get()),
            Backdrop::Effect(effect) => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                effect.hash(&mut h);
                Some(h.finish())
            }
            Backdrop::Transparent | Backdrop::Color(_) => None,
        }
    }
}

impl ItemSettings {
    /// Contrast for a hard edge, what the old hard threshold was.
    pub const HARD_EDGE: f32 = 1.0;

    /// What sits behind the result. One choice; settings saved while the
    /// fields could hold several read an effect first, then an image,
    /// then a colour.
    pub fn backdrop(&self) -> Backdrop {
        if self.bg_effect != prunr_core::BgEffect::None {
            Backdrop::Effect(self.bg_effect)
        } else if let Some(hash) = self.bg_image_hash {
            Backdrop::Image(hash)
        } else if let Some([r, g, b, _]) = self.bg {
            Backdrop::Color([r, g, b])
        } else {
            Backdrop::Transparent
        }
    }

    /// Convert mask-related fields into the core `MaskSettings`.
    pub fn mask_settings(&self) -> MaskSettings {
        MaskSettings {
            gamma: self.gamma,
            contrast: self.contrast,
            edge_shift: self.edge_shift,
            refine_edges: self.refine_edges,
            guided_radius: self.guided_radius as u32,
            guided_epsilon: self.guided_epsilon,
            feather: self.feather,
            fill_style: self.fill_style,
            decoration: self.decoration,
        }
    }

    /// Convert edge-postprocess fields into the core `EdgeSettings`.
    pub fn edge_settings(&self) -> EdgeSettings {
        EdgeSettings {
            line_strength: self.line_strength,
            solid_line_color: self.solid_line_color,
            edge_thickness: self.edge_thickness as u32,
            edge_scale: self.edge_scale,
            compose_mode: self.compose_mode,
            line_style: self.line_style,
            input_transform: self.input_transform,
        }
    }

    /// Build a `ProcessingRecipe` snapshot for tier routing.
    /// Pins mask fields to defaults in modes that skip segmentation so unrelated
    /// knob changes don't trigger reprocessing when the mask isn't used.
    ///
    /// `model` is app-global, not per-item — the caller provides it.
    pub fn current_recipe(&self, model: prunr_core::ModelKind) -> prunr_core::ProcessingRecipe {
        let uses_segmentation = self.line_mode != LineMode::EdgesOnly;
        let uses_edge_detection = self.line_mode != LineMode::Off;
        let mask = if uses_segmentation {
            prunr_core::MaskRecipe::from(&self.mask_settings())
        } else {
            prunr_core::MaskRecipe::from(&MaskSettings::default())
        };
        let backdrop = self.backdrop();

        let mut edge = prunr_core::EdgeRecipe::from(&self.edge_settings());
        // compose_mode only affects SubjectOutline (combine subject + line
        // outline). Pin elsewhere so a preset value can't dirty the recipe.
        if self.line_mode != LineMode::SubjectOutline {
            edge.compose_mode = prunr_core::ComposeMode::default();
        }

        prunr_core::ProcessingRecipe {
            inference: prunr_core::InferenceRecipe {
                model,
                uses_segmentation,
                uses_edge_detection,
                input_transform: if uses_edge_detection {
                    self.input_transform
                } else {
                    prunr_core::InputTransform::default()
                },
            },
            edge,
            mask,
            composite: prunr_core::CompositeRecipe {
                bg_color: match backdrop { Backdrop::Color(c) => Some(c), _ => None },
                bg_image_hash: match backdrop { Backdrop::Image(h) => Some(h.get()), _ => None },
                bg_image_fit: self.bg_image_fit,
                bg_effect: self.bg_effect,
            },
            upscale: prunr_core::UpscaleRecipe {
                model: model.is_upscale().then(|| model.into()),
                output_scale: self.output_scale,
                pre_denoise_bits: self.pre_denoise.to_bits(),
                sharpen_bits: self.sharpen.to_bits(),
                ai_blend_bits: self.ai_blend.to_bits(),
                color_match: self.color_match,
            },
        }
    }
}

/// Test fixture: `ItemSettings::default()` with `gamma` overridden.
#[cfg(test)]
pub(crate) fn item_with_gamma(gamma: f32) -> ItemSettings {
    ItemSettings { gamma, ..ItemSettings::default() }
}

/// Test fixture: `ItemSettings::default()` with `line_mode` overridden.
#[cfg(test)]
pub(crate) fn item_with_line_mode(line_mode: prunr_core::LineMode) -> ItemSettings {
    ItemSettings { line_mode, ..ItemSettings::default() }
}

/// Rewrite item settings saved by older versions: a `threshold` (a
/// number when it was on) becomes `contrast` at a hard edge, a retired
/// line style becomes solid, a retired composition lines only and a
/// retired effect none. Only
/// item settings are touched: each preset entry's `item_settings`, and a
/// v1 preset file, which is item settings at the top (it has no
/// `format_version`).
pub(crate) fn migrate_old_item_settings(value: &mut serde_json::Value) {
    fn item(map: &mut serde_json::Map<String, serde_json::Value>) {
        if let Some(old) = map.remove("threshold") {
            if old.is_number() && !map.contains_key("contrast") {
                map.insert("contrast".into(), serde_json::json!(ItemSettings::HARD_EDGE));
            }
        }
        // The Chromatic and Noise line styles are gone; their lines draw solid.
        if map.get("line_style").and_then(|v| v.as_object())
            .is_some_and(|o| o.contains_key("chromatic") || o.contains_key("noise"))
        {
            map.insert("line_style".into(), serde_json::json!("solid"));
        }
        // Inverse mask is gone; Lines only is the nearest composition.
        if map.get("compose_mode").and_then(|v| v.as_str()) == Some("inverse_mask") {
            map.insert("compose_mode".into(), serde_json::json!("lines_only"));
        }
        // The Channel swap effect is gone.
        if map.get("fill_style").and_then(|v| v.as_object()).is_some_and(|o| o.contains_key("channel_swap")) {
            map.insert("fill_style".into(), serde_json::json!("none"));
        }
    }
    fn walk(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::Object(settings)) = map.get_mut("item_settings") {
                    item(settings);
                }
                map.values_mut().for_each(walk);
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(walk),
            _ => {}
        }
    }
    if let serde_json::Value::Object(map) = value {
        if map.contains_key("gamma") && !map.contains_key("format_version") {
            item(map);
        }
    }
    walk(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What is behind a result is one choice. Settings saved while the
    /// fields could hold several read an effect first, then an image,
    /// then a colour, and the recipe records only that one.
    #[test]
    fn the_backdrop_is_one_choice() {
        use prunr_core::BgEffect;
        let image = std::num::NonZeroU64::new(7);
        let blur = BgEffect::BlurredSource { radius: 9 };
        let rows = [
            (None, None, BgEffect::None, Backdrop::Transparent),
            (Some([1, 2, 3, 255]), None, BgEffect::None, Backdrop::Color([1, 2, 3])),
            (Some([1, 2, 3, 255]), image, BgEffect::None, Backdrop::Image(image.unwrap())),
            (Some([1, 2, 3, 255]), image, blur, Backdrop::Effect(blur)),
        ];
        for (bg, bg_image_hash, bg_effect, want) in rows {
            let s = ItemSettings { bg, bg_image_hash, bg_effect, ..ItemSettings::default() };
            assert_eq!(s.backdrop(), want);
            let c = s.current_recipe(prunr_core::ModelKind::Silueta).composite;
            assert_eq!(c.bg_color.is_some(), matches!(want, Backdrop::Color(_)), "{want:?}");
            assert_eq!(c.bg_image_hash.is_some(), matches!(want, Backdrop::Image(_)), "{want:?}");
        }
    }

    /// A preset saved with the hard threshold on loads as a hard edge;
    /// one saved with it off loads soft.
    #[test]
    fn an_old_hard_threshold_loads_as_full_contrast() {
        let mut v = serde_json::json!({ "models": { "silueta": { "item_settings": { "gamma": 1.2, "threshold": 0.3 } },
                                                     "birefnet": { "item_settings": { "gamma": 1.0, "threshold": null } } } });
        v["brush"] = serde_json::json!({ "threshold": 0.4 });
        migrate_old_item_settings(&mut v);
        let hard: ItemSettings = serde_json::from_value(v["models"]["silueta"]["item_settings"].clone()).unwrap();
        let soft: ItemSettings = serde_json::from_value(v["models"]["birefnet"]["item_settings"].clone()).unwrap();
        assert_eq!((hard.contrast, hard.gamma), (1.0, 1.2));
        assert_eq!(soft.contrast, 0.0);
        assert_eq!(v["brush"]["threshold"], 0.4, "a threshold outside item settings is left alone");

        let mut styled = serde_json::json!({ "item_settings": { "line_style": { "noise": { "amount": 80 } } } });
        migrate_old_item_settings(&mut styled);
        let s: ItemSettings = serde_json::from_value(styled["item_settings"].clone()).unwrap();
        assert_eq!(s.line_style, prunr_core::LineStyle::Solid, "a retired line style loads as solid");
        let mut inverse = serde_json::json!({ "item_settings": { "compose_mode": "inverse_mask" } });
        migrate_old_item_settings(&mut inverse);
        let s: ItemSettings = serde_json::from_value(inverse["item_settings"].clone()).unwrap();
        assert_eq!(s.compose_mode, prunr_core::ComposeMode::LinesOnly, "a retired composition loads as lines only");
        let mut swapped = serde_json::json!({ "item_settings": { "fill_style": { "channel_swap": { "variant": "grb" } } } });
        migrate_old_item_settings(&mut swapped);
        let s: ItemSettings = serde_json::from_value(swapped["item_settings"].clone()).unwrap();
        assert_eq!(s.fill_style, prunr_core::FillStyle::None, "a retired effect loads as none");

        let mut v1 = serde_json::json!({ "gamma": 1.0, "threshold": 0.5 });
        migrate_old_item_settings(&mut v1);
        assert_eq!(v1["contrast"], 1.0, "a v1 preset is item settings at the top");
    }

    #[test]
    fn default_matches_v1_defaults() {
        let s = ItemSettings::default();
        assert_eq!(s.gamma, 1.0);
        assert_eq!(s.contrast, 0.0);
        assert_eq!(s.edge_shift, 0.0);
        assert!(!s.refine_edges);
        assert_eq!(s.line_mode, LineMode::Off);
        assert_eq!(s.line_strength, 0.5);
        assert!(s.solid_line_color.is_none());
        assert!(s.bg.is_none());
    }

    #[test]
    fn size_under_cache_line_budget() {
        // Budget is sized to absorb incremental FillStyle / LineStyle growth.
        // It sits around two cache lines. Still Copy-friendly; the cost is one extra cache
        // miss on cold access, never on the per-pixel hot path. Bump in
        // coarse steps to avoid a churn of test-only commits.
        assert!(
            std::mem::size_of::<ItemSettings>() <= 128,
            "ItemSettings is {} bytes, budget is 128",
            std::mem::size_of::<ItemSettings>()
        );
    }

    #[test]
    fn is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<ItemSettings>();
    }

    #[test]
    fn mask_settings_roundtrip() {
        let s = ItemSettings {
            gamma: 1.5,
            contrast: 0.3,
            edge_shift: 2.0,
            refine_edges: true,
            ..ItemSettings::default()
        };
        let m = s.mask_settings();
        assert_eq!(m.gamma, 1.5);
        assert_eq!(m.contrast, 0.3);
        assert_eq!(m.edge_shift, 2.0);
        assert!(m.refine_edges);
    }

    #[test]
    fn bg_rgb_stripped_in_recipe() {
        let s = ItemSettings { bg: Some([10, 20, 30, 200]), ..ItemSettings::default() };
        let r = s.current_recipe(prunr_core::ModelKind::Silueta);
        assert_eq!(r.composite.bg_color, Some([10, 20, 30]));
    }

    #[test]
    fn edges_only_pins_mask_defaults() {
        let s = ItemSettings {
            line_mode: LineMode::EdgesOnly,
            gamma: 2.0, // should NOT affect recipe in EdgesOnly mode
            contrast: 0.8,
            ..ItemSettings::default()
        };
        let r = s.current_recipe(prunr_core::ModelKind::Silueta);
        let defaults = prunr_core::MaskRecipe::from(&prunr_core::MaskSettings::default());
        assert_eq!(r.mask, defaults);
    }

    #[test]
    fn serde_loads_old_preset_missing_edge_scale_as_fused() {
        // Backwards-compat: preset files written before Task 5 lack
        // edge_scale. `#[serde(default)]` on ItemSettings must fall back
        // to `EdgeScale::Fused` (the previous hardcoded block_cat output).
        let old_json = r#"{
            "gamma": 1.2,
            "contrast": 0.0,
            "edge_shift": 0.0,
            "refine_edges": false,
            "guided_radius": 8,
            "guided_epsilon": 0.0001,
            "feather": 0.0,
            "line_mode": "EdgesOnly",
            "line_strength": 0.5,
            "solid_line_color": null,
            "edge_thickness": 0,
            "bg": null
        }"#;
        let loaded: ItemSettings = serde_json::from_str(old_json).unwrap();
        assert_eq!(loaded.edge_scale, EdgeScale::Fused);
        assert_eq!(loaded.line_mode, LineMode::EdgesOnly); // sanity: others still parse
    }

    #[test]
    fn serde_json_roundtrip_all_fields_populated() {
        let s = ItemSettings {
            gamma: 2.3,
            contrast: 0.75,
            edge_shift: -1.5,
            refine_edges: true,
            guided_radius: 12,
            guided_epsilon: 5e-4,
            feather: 1.5,
            line_mode: LineMode::EdgesOnly,
            line_strength: 0.3,
            solid_line_color: Some([10, 20, 30]),
            edge_thickness: 2,
            edge_scale: EdgeScale::Bold,
            bg: Some([100, 150, 200, 240]),
            compose_mode: prunr_core::ComposeMode::Ghost,
            line_style: prunr_core::LineStyle::GradientY { top: [255, 0, 0], bottom: [0, 0, 255] },
            fill_style: prunr_core::FillStyle::Duotone { dark: [10, 10, 40], light: [240, 240, 200] },
            decoration: prunr_core::Decoration {
                outline_width: 6.0, outline_color: [1, 2, 3], shadow: prunr_core::ShadowKind::Drop,
                shadow_opacity: 0.3, shadow_blur: 9.0, shadow_distance: 4.0,
            },
            bg_effect: prunr_core::BgEffect::BlurredSource { radius: 8 },
            input_transform: prunr_core::InputTransform::ContrastBoost { percent: 150 },
            bg_image_hash: std::num::NonZeroU64::new(0xfeedface),
            bg_image_fit: prunr_core::BgImageFit::Tile,
            output_scale: prunr_core::OutputScale::X2,
            pre_denoise: 0.4,
            sharpen: 0.3,
            ai_blend: 0.6,
            color_match: true,
            faces: FaceSettings { on: true, model: FaceModel::RestoreFormerPp, strength_pct: 70, largest_only: true, hair: true },
        };
        let json = serde_json::to_string(&s).unwrap();
        let recovered: ItemSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, recovered);
    }

    #[test]
    fn serde_loads_old_preset_missing_upscale_fields() {
        // Old preset JSON that carried `upscale_model` or `upscale_scale` from
        // the pre-refactor schema — serde should ignore unknown keys and load
        // cleanly. Missing `output_scale` defaults to X4 (native).
        let old_json = r#"{
            "gamma": 1.0,
            "contrast": 0.0,
            "edge_shift": 0.0,
            "refine_edges": false,
            "guided_radius": 8,
            "guided_epsilon": 0.0001,
            "feather": 0.0,
            "line_mode": "Off",
            "line_strength": 0.5,
            "solid_line_color": null,
            "edge_thickness": 0,
            "bg": null
        }"#;
        let loaded: ItemSettings = serde_json::from_str(old_json).unwrap();
        assert_eq!(loaded.output_scale, prunr_core::OutputScale::X4);
    }

    #[test]
    fn current_recipe_derives_upscale_model_from_active_model_kind() {
        let s = ItemSettings { output_scale: prunr_core::OutputScale::X2, ..ItemSettings::default() };
        // Active upscale model → recipe slot populated.
        let r = s.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus);
        assert_eq!(r.upscale.model, Some(prunr_models::ModelId::RealEsrganX4Plus));
        assert_eq!(r.upscale.output_scale, prunr_core::OutputScale::X2);
        // Active seg model → upscale slot empty.
        let r = s.current_recipe(prunr_core::ModelKind::Silueta);
        assert_eq!(r.upscale.model, None);
    }


    #[test]
    fn output_scale_replaces_upscale_scale_default() {
        assert_eq!(ItemSettings::default().output_scale, prunr_core::OutputScale::X4);
    }

    #[test]
    fn edge_thickness_shrunk_to_u8() {
        assert_eq!(std::mem::size_of::<u8>(), 1);
        assert_eq!(ItemSettings::default().edge_thickness, 0u8);
    }

    #[test]
    fn guided_radius_shrunk_to_u8() {
        assert_eq!(std::mem::size_of::<u8>(), 1);
        assert_eq!(ItemSettings::default().guided_radius, 8u8);
    }

    #[test]
    fn current_recipe_passes_output_scale_through() {
        let s = ItemSettings {
            output_scale: prunr_core::OutputScale::X2,
            ..ItemSettings::default()
        };
        let r = s.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus);
        assert_eq!(r.upscale.output_scale, prunr_core::OutputScale::X2);
    }

    #[test]
    fn serde_loads_old_preset_with_upscale_scale_int() {
        // Phase-30 preset JSON with `"upscale_scale": 2` — the field is
        // unknown to ItemSettings after this plan; serde silently drops it
        // and defaults `output_scale` to X4.
        let old_json = r#"{
            "gamma": 1.0,
            "contrast": 0.0,
            "edge_shift": 0.0,
            "refine_edges": false,
            "guided_radius": 8,
            "guided_epsilon": 0.0001,
            "feather": 0.0,
            "line_mode": "Off",
            "line_strength": 0.5,
            "solid_line_color": null,
            "edge_thickness": 0,
            "bg": null,
            "upscale_scale": 2
        }"#;
        let loaded: ItemSettings = serde_json::from_str(old_json).unwrap();
        assert_eq!(loaded.output_scale, prunr_core::OutputScale::X4);
    }


    #[test]
    fn new_knobs_default_values() {
        let s = ItemSettings::default();
        assert_eq!(s.pre_denoise, 0.0);
        assert_eq!(s.sharpen, prunr_core::DEFAULT_SHARPEN);
        assert_eq!(s.ai_blend, 1.0);
        assert!(!s.color_match);
    }

    #[test]
    fn default_current_recipe_carries_sharpen_default() {
        let r = ItemSettings::default().current_recipe(prunr_core::ModelKind::RealEsrganX4Plus);
        assert_eq!(r.upscale.sharpen(), prunr_core::DEFAULT_SHARPEN);
    }

    #[test]
    fn current_recipe_wires_all_new_fields() {
        let s = ItemSettings {
            pre_denoise: 0.4,
            sharpen: 0.3,
            ai_blend: 0.6,
            color_match: true,
            output_scale: prunr_core::OutputScale::X2,
            ..ItemSettings::default()
        };
        let r = s.current_recipe(prunr_core::ModelKind::RealEsrganX4Plus);
        assert_eq!(r.upscale.pre_denoise_bits, 0.4_f32.to_bits());
        assert_eq!(r.upscale.sharpen_bits, 0.3_f32.to_bits());
        assert_eq!(r.upscale.ai_blend_bits, 0.6_f32.to_bits());
        assert!(r.upscale.color_match);
        assert_eq!(r.upscale.output_scale, prunr_core::OutputScale::X2);
    }

    #[test]
    fn serde_roundtrip_all_phase32_fields() {
        let s = ItemSettings {
            pre_denoise: 0.4,
            sharpen: 0.3,
            ai_blend: 0.6,
            color_match: true,
            output_scale: prunr_core::OutputScale::X2,
            ..ItemSettings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let recovered: ItemSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, recovered);
    }

    #[test]
    fn serde_loads_old_preset_missing_phase32_fields() {
        // Phase-30 preset JSON missing all Phase-32 fields must load with
        // sensible defaults: pre_denoise=0, sharpen off (from default_sharpen()),
        // ai_blend=1 (the critical one — zero here would disable AI output).
        let old_json = r#"{
            "gamma": 1.0,
            "contrast": 0.0,
            "edge_shift": 0.0,
            "refine_edges": false,
            "guided_radius": 8,
            "guided_epsilon": 0.0001,
            "feather": 0.0,
            "line_mode": "Off",
            "line_strength": 0.5,
            "solid_line_color": null,
            "edge_thickness": 0,
            "bg": null
        }"#;
        let loaded: ItemSettings = serde_json::from_str(old_json).unwrap();
        assert_eq!(loaded.pre_denoise, 0.0);
        assert_eq!(loaded.sharpen, prunr_core::DEFAULT_SHARPEN);
        assert_eq!(loaded.ai_blend, 1.0);
        assert!(!loaded.color_match);
    }

    /// The recipe names the upscale model for every upscaler (the NMKD
    /// ones read as "no upscale"), and none for a background-removal model.
    #[test]
    fn the_recipe_names_every_upscale_model() {
        use prunr_core::ModelKind;
        for kind in [ModelKind::RealEsrganX4Plus, ModelKind::Nomos8kSchatL, ModelKind::FourXNmkdSiaxCx, ModelKind::FourXNmkdSuperscale] {
            assert_eq!(ItemSettings::default().current_recipe(kind).upscale.model, Some(kind.into()), "{kind:?}");
        }
        assert_eq!(ItemSettings::default().current_recipe(ModelKind::Silueta).upscale.model, None);
    }

    /// Brightness lift and Saturation are gone; presets that saved them
    /// still load, ignoring both.
    #[test]
    fn presets_with_the_retired_upscale_knobs_still_load() {
        let loaded: ItemSettings = serde_json::from_str(r#"{ "sharpen": 0.4, "brightness_lift": 1.5, "saturation": -0.3 }"#).unwrap();
        assert_eq!(loaded.sharpen, 0.4);
    }

    #[test]
    fn budget_still_under_128_bytes() {
        assert!(
            std::mem::size_of::<ItemSettings>() <= 128,
            "ItemSettings is {} bytes after Phase-32 fields, budget is 128",
            std::mem::size_of::<ItemSettings>()
        );
    }
}
