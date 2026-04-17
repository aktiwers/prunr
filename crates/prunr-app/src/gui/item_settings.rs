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

use prunr_core::{LineMode, MaskSettings};
use serde::{Deserialize, Serialize};

/// Per-image processing settings. Edited via the adjustments toolbar.
///
/// Invariants:
/// - All fields are `Copy`. Diff is a branchless struct compare.
/// - `bg` uses RGBA for UI parity with egui color pickers; only RGB is sent
///   into the pipeline (alpha is always treated as 255 during compositing).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ItemSettings {
    /// Gamma curve applied to the mask. >1.0 = more aggressive removal, <1.0 = gentler.
    pub gamma: f32,
    /// Optional binary threshold (0.0-1.0). `None` = soft mask.
    pub threshold: Option<f32>,
    /// Edge shift in pixels: >0 erodes, <0 dilates.
    pub edge_shift: f32,
    /// Guided-filter refinement of mask edges.
    pub refine_edges: bool,

    /// Line extraction mode.
    pub line_mode: LineMode,
    /// DexiNed edge sensitivity (0.0-1.0).
    pub line_strength: f32,
    /// Solid color override for edges. `None` = preserve original RGB.
    pub solid_line_color: Option<[u8; 3]>,

    /// Fill transparent areas with a solid color. `None` = transparent.
    /// Stored as RGBA for UI parity; pipeline only uses RGB.
    pub bg: Option<[u8; 4]>,
}

impl Default for ItemSettings {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            threshold: None,
            edge_shift: 0.0,
            refine_edges: false,
            line_mode: LineMode::Off,
            line_strength: 0.5,
            solid_line_color: None,
            bg: None,
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
            prunr_core::MaskRecipe::new(self.gamma, self.threshold, self.edge_shift, self.refine_edges)
        } else {
            prunr_core::MaskRecipe::new(1.0, None, 0.0, false)
        };

        prunr_core::ProcessingRecipe {
            inference: prunr_core::InferenceRecipe {
                model,
                uses_segmentation,
                uses_edge_detection,
            },
            edge: prunr_core::EdgeRecipe {
                line_strength_bits: self.line_strength.to_bits(),
                solid_line_color: self.solid_line_color,
            },
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
        // Hard limit: must stay under 40 bytes so BatchItem doesn't bloat.
        assert!(
            std::mem::size_of::<ItemSettings>() <= 40,
            "ItemSettings is {} bytes, budget is 40",
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
        let defaults = prunr_core::MaskRecipe::new(1.0, None, 0.0, false);
        assert_eq!(r.mask, defaults);
    }
}
