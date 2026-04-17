//! Recipe types for tiered pipeline skip/re-apply.
//!
//! A "recipe" captures the exact settings used to produce a result.
//! When settings change, `resolve_tier()` determines which pipeline
//! tier needs to re-run (or if processing can be skipped entirely).

use serde::{Serialize, Deserialize};
use crate::types::ModelKind;

/// Tier 1: settings that require AI model inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InferenceRecipe {
    pub model: ModelKind,
    /// Whether segmentation model runs (false for EdgesOnly mode).
    pub uses_segmentation: bool,
    /// Whether edge detection model runs (true when line_mode is not Off).
    pub uses_edge_detection: bool,
    /// Edge detection sensitivity (affects DexiNed output).
    pub line_strength_bits: u32,
    /// Solid line color override (affects edge output).
    pub solid_line_color: Option<[u8; 3]>,
}

/// Tier 2: mask postprocessing settings (gamma, threshold, edge refinement).
/// Uses `f32::to_bits()` for exact float comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskRecipe {
    gamma_bits: u32,
    threshold_bits: Option<u32>,
    edge_shift_bits: u32,
    pub refine_edges: bool,
}

impl MaskRecipe {
    pub fn new(gamma: f32, threshold: Option<f32>, edge_shift: f32, refine_edges: bool) -> Self {
        Self {
            gamma_bits: gamma.to_bits(),
            threshold_bits: threshold.map(|t| t.to_bits()),
            edge_shift_bits: edge_shift.to_bits(),
            refine_edges,
        }
    }
}

impl PartialEq for MaskRecipe {
    fn eq(&self, other: &Self) -> bool {
        self.gamma_bits == other.gamma_bits
            && self.threshold_bits == other.threshold_bits
            && self.edge_shift_bits == other.edge_shift_bits
            && self.refine_edges == other.refine_edges
    }
}
impl Eq for MaskRecipe {}

impl std::hash::Hash for MaskRecipe {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.gamma_bits.hash(state);
        self.threshold_bits.hash(state);
        self.edge_shift_bits.hash(state);
        self.refine_edges.hash(state);
    }
}

/// Tier 3: compositing settings (bg color, line settings).
/// These can be applied without re-running inference or masking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CompositeRecipe {
    pub bg_color: Option<[u8; 3]>,
    pub solid_line_color: Option<[u8; 3]>,
}

/// Complete recipe — the full fingerprint of settings that produced a result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProcessingRecipe {
    pub inference: InferenceRecipe,
    pub mask: MaskRecipe,
    pub composite: CompositeRecipe,
    /// True if this result was produced in chain mode (previous result as input).
    pub was_chain: bool,
}

/// What processing tier is needed to go from old settings to new settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredTier {
    /// Nothing changed — skip entirely.
    Skip,
    /// Only compositing changed (bg_color) — parent-local, instant.
    CompositeOnly,
    /// Mask settings changed — re-run postprocess from cached tensor (~200ms).
    MaskRerun,
    /// Model changed — full pipeline needed.
    FullPipeline,
}

/// Determine what processing tier is needed when changing from old to new recipe.
pub fn resolve_tier(old: &ProcessingRecipe, new: &ProcessingRecipe) -> RequiredTier {
    if old == new {
        return RequiredTier::Skip;
    }
    // Chain mode changes the input image, not settings — always needs full pipeline
    if old.was_chain != new.was_chain || old.inference != new.inference {
        return RequiredTier::FullPipeline;
    }
    if old.mask != new.mask {
        return RequiredTier::MaskRerun;
    }
    RequiredTier::CompositeOnly
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recipe(model: ModelKind, gamma: f32, bg: Option<[u8; 3]>) -> ProcessingRecipe {
        ProcessingRecipe {
            inference: InferenceRecipe {
                model,
                uses_segmentation: true,
                uses_edge_detection: false,
                line_strength_bits: 0.5f32.to_bits(),
                solid_line_color: None,
            },
            mask: MaskRecipe::new(gamma, None, 0.0, false),
            composite: CompositeRecipe { bg_color: bg, solid_line_color: None },
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
        let a = MaskRecipe::new(1.0, Some(0.5), 0.0, true);
        let b = MaskRecipe::new(1.0, Some(0.5), 0.0, true);
        assert_eq!(a, b);
    }

    #[test]
    fn float_bits_inequality() {
        let a = MaskRecipe::new(1.0, Some(0.5), 0.0, true);
        let b = MaskRecipe::new(1.0, Some(0.50001), 0.0, true);
        assert_ne!(a, b);
    }
}
