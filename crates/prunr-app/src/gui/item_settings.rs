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

use prunr_core::{LineMode, MaskSettings, EdgeSettings, EdgeScale};
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
    /// Optional binary threshold (0.0-1.0). `None` = soft mask.
    pub threshold: Option<f32>,
    /// Edge shift in pixels: >0 erodes, <0 dilates.
    pub edge_shift: f32,
    /// Guided-filter refinement of mask edges.
    pub refine_edges: bool,
    /// Guided filter window radius (pixels). Only used when refine_edges.
    pub guided_radius: u32,
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
    /// Dilate edge mask by N pixels after threshold — thickens thin lines.
    pub edge_thickness: u32,
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
}

impl Default for ItemSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            threshold: None,
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
        }
    }
}

impl ItemSettings {
    /// bg color as RGB triple (alpha stripped) for pipeline/compositing.
    /// `None` when bg is disabled.
    pub fn bg_rgb(&self) -> Option<[u8; 3]> {
        self.bg.map(|[r, g, b, _]| [r, g, b])
    }

    /// Convert mask-related fields into the core `MaskSettings`.
    pub fn mask_settings(&self) -> MaskSettings {
        MaskSettings {
            gamma: self.gamma,
            threshold: self.threshold,
            edge_shift: self.edge_shift,
            refine_edges: self.refine_edges,
            guided_radius: self.guided_radius,
            guided_epsilon: self.guided_epsilon,
            feather: self.feather,
        }
    }

    /// Convert edge-postprocess fields into the core `EdgeSettings`.
    pub fn edge_settings(&self) -> EdgeSettings {
        EdgeSettings {
            line_strength: self.line_strength,
            solid_line_color: self.solid_line_color,
            edge_thickness: self.edge_thickness,
            edge_scale: self.edge_scale,
            compose_mode: self.compose_mode,
        }
    }

    /// Build a `ProcessingRecipe` snapshot for tier routing.
    /// Pins mask fields to defaults in modes that skip segmentation so unrelated
    /// knob changes don't trigger reprocessing when the mask isn't used.
    ///
    /// `model` and `chain_mode` are app-global, not per-item — caller provides them.
    pub fn current_recipe(&self, model: prunr_core::ModelKind, chain_mode: bool) -> prunr_core::ProcessingRecipe {
        let uses_segmentation = self.line_mode != LineMode::EdgesOnly;
        let uses_edge_detection = self.line_mode != LineMode::Off;
        let bg_rgb = self.bg.map(|[r, g, b, _]| [r, g, b]);

        let mask = if uses_segmentation {
            prunr_core::MaskRecipe::from(&self.mask_settings())
        } else {
            prunr_core::MaskRecipe::from(&MaskSettings::default())
        };

        prunr_core::ProcessingRecipe {
            inference: prunr_core::InferenceRecipe {
                model,
                uses_segmentation,
                uses_edge_detection,
            },
            edge: prunr_core::EdgeRecipe::from(&self.edge_settings()),
            mask,
            composite: prunr_core::CompositeRecipe {
                bg_color: bg_rgb,
                solid_line_color: self.solid_line_color,
            },
            was_chain: chain_mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_v1_defaults() {
        let s = ItemSettings::default();
        assert_eq!(s.gamma, 1.0);
        assert!(s.threshold.is_none());
        assert_eq!(s.edge_shift, 0.0);
        assert!(!s.refine_edges);
        assert_eq!(s.line_mode, LineMode::Off);
        assert_eq!(s.line_strength, 0.5);
        assert!(s.solid_line_color.is_none());
        assert!(s.bg.is_none());
    }

    #[test]
    fn size_under_cache_line_budget() {
        // 64 B = one x86 cache line. Growing past a line still keeps the struct
        // Copy-friendly; it just means an extra cache miss on read. The
        // ComposeMode u8 tipped us over 48 B in Phase 1; stop when we reach 64.
        assert!(
            std::mem::size_of::<ItemSettings>() <= 64,
            "ItemSettings is {} bytes, budget is 64",
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
        let mut s = ItemSettings::default();
        s.gamma = 1.5;
        s.threshold = Some(0.3);
        s.edge_shift = 2.0;
        s.refine_edges = true;
        let m = s.mask_settings();
        assert_eq!(m.gamma, 1.5);
        assert_eq!(m.threshold, Some(0.3));
        assert_eq!(m.edge_shift, 2.0);
        assert!(m.refine_edges);
    }

    #[test]
    fn bg_rgb_stripped_in_recipe() {
        let mut s = ItemSettings::default();
        s.bg = Some([10, 20, 30, 200]);
        let r = s.current_recipe(prunr_core::ModelKind::Silueta, false);
        assert_eq!(r.composite.bg_color, Some([10, 20, 30]));
    }

    #[test]
    fn edges_only_pins_mask_defaults() {
        let mut s = ItemSettings::default();
        s.line_mode = LineMode::EdgesOnly;
        s.gamma = 2.0; // should NOT affect recipe in EdgesOnly mode
        s.threshold = Some(0.8);
        let r = s.current_recipe(prunr_core::ModelKind::Silueta, false);
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
            "threshold": null,
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
            threshold: Some(0.75),
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
        };
        let json = serde_json::to_string(&s).unwrap();
        let recovered: ItemSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, recovered);
    }
}
