//! Declarative per-knob routing catalog: each user-facing adjustment maps
//! to its pipeline tier, cache invalidation, and runtime dispatch. Chips
//! read this table instead of hand-setting `ToolbarChange` flags, so one
//! table drives all routing decisions downstream.
//!
//! `resolve_tier` operates on recipe diffs; this catalog operates on knobs.
//! Both must agree on single-knob mutations — cross-check tests enforce it.
//!
//! **Type discipline.** `StaticKnob` covers knobs whose spec is fully
//! determined by the knob identity. `LineModeChange` carries transition
//! context (from/to). `spec()` takes only `StaticKnob`, so context-
//! sensitive dispatch must go through `line_mode_spec` or
//! `input_transform_spec` with item state — "silent worst-case fallback"
//! perf bugs can't exist.

use prunr_core::{LineMode, ProcessingRecipe, RequiredTier};

/// Knobs whose `KnobSpec` is a pure function of the knob identity.
/// Context-sensitive knobs (line_mode transition, input_transform) live
/// in their own types so callers can't accidentally ask the catalog for
/// a static spec and get a conservative over-approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StaticKnob {
    // Mask tier — cached seg tensor reused.
    Gamma,
    Threshold,
    EdgeShift,
    RefineEdges,
    GuidedRadius,
    GuidedEpsilon,
    Feather,
    FillStyle,
    BgEffect,

    // Edge tier — cached DexiNed tensor reused.
    LineStrength,
    EdgeThickness,
    EdgeScale,
    SolidLineColor,
    ComposeMode,
    LineStyle,

    // Composite — render-time only.
    BgColor,

    // Inference-level — needs worker subprocess.
    Model,
    ChainMode,
}

impl StaticKnob {
    /// Stable bit position for `KnobSet`. Order must match `ALL`.
    const fn ordinal(self) -> u8 {
        match self {
            StaticKnob::Gamma => 0,
            StaticKnob::Threshold => 1,
            StaticKnob::EdgeShift => 2,
            StaticKnob::RefineEdges => 3,
            StaticKnob::GuidedRadius => 4,
            StaticKnob::GuidedEpsilon => 5,
            StaticKnob::Feather => 6,
            StaticKnob::FillStyle => 7,
            StaticKnob::BgEffect => 8,
            StaticKnob::LineStrength => 9,
            StaticKnob::EdgeThickness => 10,
            StaticKnob::EdgeScale => 11,
            StaticKnob::SolidLineColor => 12,
            StaticKnob::ComposeMode => 13,
            StaticKnob::LineStyle => 14,
            StaticKnob::BgColor => 15,
            StaticKnob::Model => 16,
            StaticKnob::ChainMode => 17,
        }
    }

    /// All variants, for table-driven tests and catalog validation.
    pub const ALL: &'static [StaticKnob] = &[
        StaticKnob::Gamma,
        StaticKnob::Threshold,
        StaticKnob::EdgeShift,
        StaticKnob::RefineEdges,
        StaticKnob::GuidedRadius,
        StaticKnob::GuidedEpsilon,
        StaticKnob::Feather,
        StaticKnob::FillStyle,
        StaticKnob::BgEffect,
        StaticKnob::LineStrength,
        StaticKnob::EdgeThickness,
        StaticKnob::EdgeScale,
        StaticKnob::SolidLineColor,
        StaticKnob::ComposeMode,
        StaticKnob::LineStyle,
        StaticKnob::BgColor,
        StaticKnob::Model,
        StaticKnob::ChainMode,
    ];
}

/// Context-sensitive event: line-mode transition. Dispatch depends on
/// whether the DexiNed tensor is cached — `line_mode_spec` consumes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineModeChange {
    pub from: LineMode,
    pub to: LineMode,
}


/// Bitset over `StaticKnob` variants. `Copy`, zero-alloc; a single u32
/// covers all 18 static knobs with room to spare.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KnobSet(u32);

impl KnobSet {
    pub fn insert(&mut self, k: StaticKnob) {
        self.0 |= 1u32 << (k.ordinal() as u32);
    }
    pub fn contains(self, k: StaticKnob) -> bool {
        (self.0 >> (k.ordinal() as u32)) & 1 == 1
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheImpact {
    Nothing,
    EdgeCache,
    SegCache,
    Both,
}

impl CacheImpact {
    /// Combine impacts from multiple knobs touched in one frame.
    /// Seg + Edge promotes to Both (the two tensors are independent).
    pub fn union(self, other: Self) -> Self {
        use CacheImpact::*;
        match (self, other) {
            (Both, _) | (_, Both) => Both,
            (SegCache, EdgeCache) | (EdgeCache, SegCache) => Both,
            (SegCache, _) | (_, SegCache) => SegCache,
            (EdgeCache, _) | (_, EdgeCache) => EdgeCache,
            (Nothing, Nothing) => Nothing,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchKind {
    None,
    Render,
    LivePreviewMask,
    LivePreviewEdge,
    SubprocessAddEdge,
    SubprocessFullPipeline,
}

impl DispatchKind {
    fn cost(self) -> u8 {
        match self {
            DispatchKind::None => 0,
            DispatchKind::Render => 1,
            DispatchKind::LivePreviewMask => 2,
            DispatchKind::LivePreviewEdge => 2,
            DispatchKind::SubprocessAddEdge => 3,
            DispatchKind::SubprocessFullPipeline => 4,
        }
    }

    /// Strongest of two dispatches. When both are live-preview, `Mask` wins
    /// over `Edge` — a mask rerun recomposes edges, the reverse doesn't.
    pub fn max(self, other: Self) -> Self {
        use DispatchKind::*;
        match (self, other) {
            (LivePreviewMask, LivePreviewEdge) | (LivePreviewEdge, LivePreviewMask) => {
                LivePreviewMask
            }
            _ => {
                if self.cost() >= other.cost() {
                    self
                } else {
                    other
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnobSpec {
    pub tier: RequiredTier,
    pub cache_impact: CacheImpact,
    pub dispatch: DispatchKind,
    /// Whether a committed change to this knob auto-fires its dispatch.
    /// True for LineMode + InputTransform (intent is "flip and go"); false
    /// for Model / ChainMode (user should review and click Process).
    pub auto_trigger_on_commit: bool,
}

/// Static per-knob spec. Only defined for `StaticKnob` — context-sensitive
/// knobs have their own resolvers, so silent worst-case fallbacks are
/// unrepresentable.
pub fn spec(knob: StaticKnob) -> KnobSpec {
    use CacheImpact::*;
    use DispatchKind::*;
    use RequiredTier::*;
    let s = |tier, cache, dispatch, auto| KnobSpec {
        tier,
        cache_impact: cache,
        dispatch,
        auto_trigger_on_commit: auto,
    };

    match knob {
        StaticKnob::Gamma
        | StaticKnob::Threshold
        | StaticKnob::EdgeShift
        | StaticKnob::RefineEdges
        | StaticKnob::GuidedRadius
        | StaticKnob::GuidedEpsilon
        | StaticKnob::Feather
        | StaticKnob::FillStyle
        | StaticKnob::BgEffect => s(MaskRerun, Nothing, LivePreviewMask, true),

        StaticKnob::LineStrength
        | StaticKnob::EdgeThickness
        | StaticKnob::EdgeScale
        | StaticKnob::SolidLineColor
        | StaticKnob::ComposeMode
        | StaticKnob::LineStyle => s(EdgeRerun, Nothing, LivePreviewEdge, true),

        StaticKnob::BgColor => s(CompositeOnly, Nothing, Render, true),

        // Model / ChainMode: user must click Process.
        StaticKnob::Model => s(FullPipeline, SegCache, SubprocessFullPipeline, false),
        StaticKnob::ChainMode => s(FullPipeline, Both, SubprocessFullPipeline, false),
    }
}

/// Precise spec for a line-mode transition.
///
/// `cached_edge` gates the `Off → SubjectOutline` fast path: with the
/// DexiNed tensor in memory we recompose via live-preview; otherwise we
/// rerun DexiNed on the cached seg (AddEdgeInference subprocess).
///
/// Transitions to/from `Off` preserve the DexiNed tensor — `Off` doesn't
/// use it. `SubjectOutline ↔ EdgesOnly` invalidates edge cache because
/// DexiNed sees different input (masked-subject vs raw scene).
pub fn line_mode_spec(change: LineModeChange, cached_edge: bool) -> KnobSpec {
    use CacheImpact::*;
    use DispatchKind::*;
    use LineMode::*;
    use RequiredTier::*;
    let s = |tier, cache, dispatch| KnobSpec {
        tier,
        cache_impact: cache,
        dispatch,
        auto_trigger_on_commit: true,
    };

    match (change.from, change.to) {
        (Off, Off) | (EdgesOnly, EdgesOnly) | (SubjectOutline, SubjectOutline) => {
            s(Skip, Nothing, DispatchKind::None)
        }
        (Off, SubjectOutline) if cached_edge => s(MaskRerun, Nothing, LivePreviewMask),
        (Off, SubjectOutline) => s(AddEdgeInference, Nothing, SubprocessAddEdge),
        (Off, EdgesOnly) => s(FullPipeline, EdgeCache, SubprocessFullPipeline),
        (SubjectOutline, Off) => s(MaskRerun, Nothing, LivePreviewMask),
        (SubjectOutline, EdgesOnly) => s(FullPipeline, EdgeCache, SubprocessFullPipeline),
        (EdgesOnly, Off) => s(FullPipeline, Nothing, SubprocessFullPipeline),
        (EdgesOnly, SubjectOutline) => s(FullPipeline, EdgeCache, SubprocessFullPipeline),
    }
}

/// Precise spec for an input-transform change. With a cached seg tensor
/// we reuse it and rerun DexiNed only; otherwise a full pipeline is needed.
pub fn input_transform_spec(cached_seg: bool) -> KnobSpec {
    use CacheImpact::EdgeCache;
    use DispatchKind::{SubprocessAddEdge, SubprocessFullPipeline};
    use RequiredTier::{AddEdgeInference, FullPipeline};

    let (tier, dispatch) = if cached_seg {
        (AddEdgeInference, SubprocessAddEdge)
    } else {
        (FullPipeline, SubprocessFullPipeline)
    };
    KnobSpec {
        tier,
        cache_impact: EdgeCache,
        dispatch,
        auto_trigger_on_commit: true,
    }
}

/// Aggregate `CacheImpact` implied by a recipe diff — union of per-knob
/// impacts for every field that actually changed. Used by `classify_candidates`
/// to invalidate only the caches whose INPUT changed (seg stays valid on an
/// input_transform change, edge stays valid on a model swap).
pub fn cache_impact_for_recipe_diff(old: &ProcessingRecipe, new: &ProcessingRecipe) -> CacheImpact {
    let mut impact = CacheImpact::Nothing;
    if old.was_chain != new.was_chain {
        impact = impact.union(spec(StaticKnob::ChainMode).cache_impact);
    }
    if old.inference.model != new.inference.model {
        impact = impact.union(spec(StaticKnob::Model).cache_impact);
    }
    if old.inference.input_transform != new.inference.input_transform {
        impact = impact.union(input_transform_spec(false).cache_impact);
    }
    let old_mode = line_mode_from_flags(
        old.inference.uses_segmentation,
        old.inference.uses_edge_detection,
    );
    let new_mode = line_mode_from_flags(
        new.inference.uses_segmentation,
        new.inference.uses_edge_detection,
    );
    if old_mode != new_mode {
        let change = LineModeChange { from: old_mode, to: new_mode };
        impact = impact.union(line_mode_spec(change, false).cache_impact);
    }
    impact
}

/// Reconstruct the LineMode from the (seg, edge) recipe flags.
fn line_mode_from_flags(uses_seg: bool, uses_edge: bool) -> LineMode {
    match (uses_seg, uses_edge) {
        (_, false) => LineMode::Off,
        (false, true) => LineMode::EdgesOnly,
        (true, true) => LineMode::SubjectOutline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_static_knob_has_a_spec() {
        for k in StaticKnob::ALL {
            let _ = spec(*k);
        }
    }

    #[test]
    fn no_static_knob_resolves_to_skip() {
        for k in StaticKnob::ALL {
            assert_ne!(spec(*k).tier, RequiredTier::Skip, "{k:?}");
        }
    }

    #[test]
    fn specs_are_internally_consistent() {
        // With context-sensitive knobs split into their own types, the
        // (tier, dispatch) pairing on `StaticKnob` is total.
        for k in StaticKnob::ALL {
            let s = spec(*k);
            match (s.tier, s.dispatch) {
                (RequiredTier::CompositeOnly, DispatchKind::Render)
                | (RequiredTier::MaskRerun, DispatchKind::LivePreviewMask)
                | (RequiredTier::EdgeRerun, DispatchKind::LivePreviewEdge)
                | (RequiredTier::FullPipeline, DispatchKind::SubprocessFullPipeline)
                | (RequiredTier::AddEdgeInference, DispatchKind::SubprocessAddEdge) => {}
                other => panic!("knob {k:?} has inconsistent (tier, dispatch) = {other:?}"),
            }
        }
    }

    #[test]
    fn cache_impact_union_exhaustive() {
        use CacheImpact::*;
        // All 16 ordered pairs; `union` must be commutative.
        let cases: [(CacheImpact, CacheImpact, CacheImpact); 16] = [
            (Nothing, Nothing, Nothing),
            (Nothing, EdgeCache, EdgeCache),
            (Nothing, SegCache, SegCache),
            (Nothing, Both, Both),
            (EdgeCache, Nothing, EdgeCache),
            (EdgeCache, EdgeCache, EdgeCache),
            (EdgeCache, SegCache, Both),
            (EdgeCache, Both, Both),
            (SegCache, Nothing, SegCache),
            (SegCache, EdgeCache, Both),
            (SegCache, SegCache, SegCache),
            (SegCache, Both, Both),
            (Both, Nothing, Both),
            (Both, EdgeCache, Both),
            (Both, SegCache, Both),
            (Both, Both, Both),
        ];
        for (a, b, expected) in cases {
            assert_eq!(a.union(b), expected, "union({a:?}, {b:?})");
            assert_eq!(b.union(a), expected, "commutativity {a:?} vs {b:?}");
        }
    }

    #[test]
    fn dispatch_max_prefers_stronger() {
        use DispatchKind::*;
        assert_eq!(SubprocessFullPipeline.max(LivePreviewMask), SubprocessFullPipeline);
        assert_eq!(LivePreviewMask.max(SubprocessAddEdge), SubprocessAddEdge);
        assert_eq!(LivePreviewMask.max(LivePreviewEdge), LivePreviewMask);
        assert_eq!(LivePreviewEdge.max(LivePreviewMask), LivePreviewMask);
        assert_eq!(Render.max(DispatchKind::None), Render);
        assert_eq!(DispatchKind::None.max(DispatchKind::None), DispatchKind::None);
    }

    fn lmc(from: LineMode, to: LineMode) -> LineModeChange {
        LineModeChange { from, to }
    }

    #[test]
    fn line_mode_identity_is_skip() {
        for m in [LineMode::Off, LineMode::EdgesOnly, LineMode::SubjectOutline] {
            let s = line_mode_spec(lmc(m, m), false);
            assert_eq!(s.tier, RequiredTier::Skip);
            assert_eq!(s.dispatch, DispatchKind::None);
        }
    }

    #[test]
    fn line_mode_off_to_subject_respects_cached_edge() {
        let warm = line_mode_spec(lmc(LineMode::Off, LineMode::SubjectOutline), true);
        assert_eq!(warm.dispatch, DispatchKind::LivePreviewMask);
        let cold = line_mode_spec(lmc(LineMode::Off, LineMode::SubjectOutline), false);
        assert_eq!(cold.dispatch, DispatchKind::SubprocessAddEdge);
    }

    #[test]
    fn line_mode_transitions_match_table() {
        use CacheImpact::*;
        use DispatchKind::*;
        use LineMode::*;
        let cases = [
            (Off, SubjectOutline, false, Nothing, SubprocessAddEdge),
            (Off, SubjectOutline, true, Nothing, LivePreviewMask),
            (Off, EdgesOnly, false, EdgeCache, SubprocessFullPipeline),
            (SubjectOutline, Off, false, Nothing, LivePreviewMask),
            (SubjectOutline, EdgesOnly, false, EdgeCache, SubprocessFullPipeline),
            (EdgesOnly, Off, false, Nothing, SubprocessFullPipeline),
            (EdgesOnly, SubjectOutline, false, EdgeCache, SubprocessFullPipeline),
        ];
        for (old, new, cached, cache, dispatch) in cases {
            let s = line_mode_spec(lmc(old, new), cached);
            assert_eq!(s.cache_impact, cache, "{old:?} -> {new:?}");
            assert_eq!(s.dispatch, dispatch, "{old:?} -> {new:?}");
        }
    }

    /// Regression: the cold-path (cached_edge=false) spec for Off→Subject is
    /// `SubprocessAddEdge`, which outranks `LivePreviewMask` under `.max()`.
    /// If a caller folds the cold dispatch into an aggregate and the
    /// dispatcher then `.max()`es in the refined warm dispatch, the warm
    /// fast path is silently poisoned — every Off→Subject toggle takes the
    /// subprocess path, spawning a cold DexiNed rerun on each flip. Guard
    /// by asserting the ordering explicitly so the bug can't return.
    #[test]
    fn cold_line_mode_dispatch_outranks_warm_under_max() {
        use DispatchKind::*;
        let cold = line_mode_spec(lmc(LineMode::Off, LineMode::SubjectOutline), false).dispatch;
        let warm = line_mode_spec(lmc(LineMode::Off, LineMode::SubjectOutline), true).dispatch;
        assert_eq!(cold, SubprocessAddEdge);
        assert_eq!(warm, LivePreviewMask);
        assert_eq!(
            cold.max(warm),
            SubprocessAddEdge,
            "`.max()` keeps the cold dispatch — callers must NOT fold the \
             cold spec into an aggregate before the warm refinement runs.",
        );
    }

    #[test]
    fn input_transform_spec_reuses_cached_seg() {
        let warm = input_transform_spec(true);
        assert_eq!(warm.tier, RequiredTier::AddEdgeInference);
        assert_eq!(warm.dispatch, DispatchKind::SubprocessAddEdge);
        assert_eq!(warm.cache_impact, CacheImpact::EdgeCache);

        let cold = input_transform_spec(false);
        assert_eq!(cold.tier, RequiredTier::FullPipeline);
        assert_eq!(cold.dispatch, DispatchKind::SubprocessFullPipeline);
        assert_eq!(cold.cache_impact, CacheImpact::EdgeCache);
    }

    /// Apply a single-knob mutation to `ItemSettings`. Picks a value that
    /// differs from the default so `resolve_tier` detects the change.
    fn mutate_static(knob: StaticKnob, base: crate::gui::item_settings::ItemSettings)
        -> crate::gui::item_settings::ItemSettings
    {
        use prunr_core::{ComposeMode, EdgeScale, FillStyle, LineStyle};
        let mut new = base;
        match knob {
            StaticKnob::Gamma => new.gamma = base.gamma + 0.5,
            StaticKnob::Threshold => new.threshold = Some(base.threshold.unwrap_or(0.5) + 0.1),
            StaticKnob::EdgeShift => new.edge_shift = base.edge_shift + 1.0,
            StaticKnob::RefineEdges => new.refine_edges = !base.refine_edges,
            StaticKnob::GuidedRadius => new.guided_radius = base.guided_radius + 1,
            StaticKnob::GuidedEpsilon => new.guided_epsilon = base.guided_epsilon * 2.0,
            StaticKnob::Feather => new.feather = base.feather + 0.5,
            StaticKnob::FillStyle => {
                new.fill_style = match base.fill_style {
                    FillStyle::None => FillStyle::Invert,
                    _ => FillStyle::None,
                };
            }
            StaticKnob::BgEffect => {
                new.bg_effect = match base.bg_effect {
                    prunr_core::BgEffect::None => prunr_core::BgEffect::InvertedSource,
                    _ => prunr_core::BgEffect::None,
                };
            }
            StaticKnob::LineStrength => new.line_strength = (base.line_strength + 0.2).min(1.0),
            StaticKnob::EdgeThickness => new.edge_thickness = base.edge_thickness + 1,
            StaticKnob::EdgeScale => {
                new.edge_scale = match base.edge_scale {
                    EdgeScale::Fused => EdgeScale::Fine,
                    _ => EdgeScale::Fused,
                };
            }
            StaticKnob::SolidLineColor => {
                new.solid_line_color = match base.solid_line_color {
                    None => Some([255, 0, 0]),
                    Some(_) => None,
                };
            }
            StaticKnob::ComposeMode => {
                new.compose_mode = match base.compose_mode {
                    ComposeMode::LinesOnly => ComposeMode::SubjectFilled,
                    _ => ComposeMode::LinesOnly,
                };
            }
            StaticKnob::LineStyle => {
                new.line_style = match base.line_style {
                    LineStyle::Solid => LineStyle::Rainbow { cycles: 3 },
                    _ => LineStyle::Solid,
                };
            }
            StaticKnob::BgColor => {
                new.bg = match base.bg {
                    None => Some([128, 128, 128, 255]),
                    Some(_) => None,
                };
            }
            StaticKnob::Model | StaticKnob::ChainMode => {
                // Model + ChainMode live outside ItemSettings; tested via
                // dedicated recipe-level mutations in the integration tests.
            }
        }
        new
    }

    #[test]
    fn static_knob_tiers_and_cache_agree_with_resolve_tier() {
        use prunr_core::{LineMode as Lm, ModelKind};
        let base = crate::gui::item_settings::ItemSettings {
            // Edge-tier + line knobs are only active in this mode; mutations
            // otherwise pin to defaults at the recipe boundary.
            line_mode: Lm::SubjectOutline,
            ..Default::default()
        };
        let ordered = |t: prunr_core::RequiredTier| -> u8 {
            use prunr_core::RequiredTier::*;
            match t {
                Skip => 0,
                CompositeOnly => 1,
                EdgeRerun => 2,
                MaskRerun => 3,
                AddEdgeInference => 4,
                FullPipeline => 5,
            }
        };
        for knob in StaticKnob::ALL {
            if matches!(knob, StaticKnob::Model | StaticKnob::ChainMode) {
                continue;
            }
            let new = mutate_static(*knob, base);
            let model = ModelKind::Silueta;
            let old_recipe = base.current_recipe(model, false);
            let new_recipe = new.current_recipe(model, false);

            let tier_from_resolve = prunr_core::resolve_tier(&old_recipe, &new_recipe);
            let impact_from_diff = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);
            let cat_spec = spec(*knob);

            assert_eq!(
                cat_spec.cache_impact, impact_from_diff,
                "knob {knob:?}: catalog cache_impact {:?} != diff {:?}",
                cat_spec.cache_impact, impact_from_diff,
            );
            assert!(
                ordered(cat_spec.tier) >= ordered(tier_from_resolve),
                "knob {knob:?}: catalog tier {:?} < resolve_tier {:?}",
                cat_spec.tier,
                tier_from_resolve,
            );
        }
    }

    #[test]
    fn line_mode_cache_impact_matches_recipe_diff() {
        use prunr_core::ModelKind;
        let cases = [
            (LineMode::Off, LineMode::SubjectOutline),
            (LineMode::Off, LineMode::EdgesOnly),
            (LineMode::SubjectOutline, LineMode::Off),
            (LineMode::SubjectOutline, LineMode::EdgesOnly),
            (LineMode::EdgesOnly, LineMode::Off),
            (LineMode::EdgesOnly, LineMode::SubjectOutline),
        ];
        for (from, to) in cases {
            let old = crate::gui::item_settings::ItemSettings {
                line_mode: from,
                ..Default::default()
            };
            let new = crate::gui::item_settings::ItemSettings {
                line_mode: to,
                ..Default::default()
            };
            let old_recipe = old.current_recipe(ModelKind::Silueta, false);
            let new_recipe = new.current_recipe(ModelKind::Silueta, false);
            let cat = line_mode_spec(lmc(from, to), false);
            let diff = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);
            assert_eq!(cat.cache_impact, diff, "{from:?} -> {to:?}");
        }
    }

    #[test]
    fn input_transform_cache_impact_matches_recipe_diff() {
        use prunr_core::{InputTransform as It, ModelKind};
        let old = crate::gui::item_settings::ItemSettings {
            line_mode: LineMode::SubjectOutline,
            ..Default::default()
        };
        let new = crate::gui::item_settings::ItemSettings {
            line_mode: LineMode::SubjectOutline,
            input_transform: It::Grayscale,
            ..Default::default()
        };
        let diff = cache_impact_for_recipe_diff(
            &old.current_recipe(ModelKind::Silueta, false),
            &new.current_recipe(ModelKind::Silueta, false),
        );
        assert_eq!(
            input_transform_spec(true).cache_impact,
            diff,
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(1000))]

        /// Any recipe compared to itself must resolve to Skip.
        #[test]
        fn identity_recipe_is_skip(
            gamma in 0.01f32..10.0,
            threshold_on in proptest::prelude::any::<bool>(),
            threshold_val in 0.001f32..0.999,
            edge_shift in -50.0f32..50.0,
            line_mode in proptest::prelude::prop_oneof![
                proptest::prelude::Just(LineMode::Off),
                proptest::prelude::Just(LineMode::EdgesOnly),
                proptest::prelude::Just(LineMode::SubjectOutline),
            ],
            chain in proptest::prelude::any::<bool>(),
        ) {
            let s = crate::gui::item_settings::ItemSettings {
                gamma,
                threshold: if threshold_on { Some(threshold_val) } else { None },
                edge_shift,
                line_mode,
                ..Default::default()
            };
            let r = s.current_recipe(prunr_core::ModelKind::Silueta, chain);
            proptest::prop_assert_eq!(prunr_core::resolve_tier(&r, &r), RequiredTier::Skip);
        }

        /// Cache impact is additive: applying two independent knobs in sequence
        /// matches the union of their individual impacts.
        #[test]
        fn cache_impact_is_additive(
            chain_flip in proptest::prelude::any::<bool>(),
            transform_flip in proptest::prelude::any::<bool>(),
            model_flip in proptest::prelude::any::<bool>(),
        ) {
            use prunr_core::{InputTransform, ModelKind};
            let base = crate::gui::item_settings::ItemSettings {
                line_mode: LineMode::SubjectOutline,
                ..Default::default()
            };
            let old_recipe = base.current_recipe(ModelKind::Silueta, false);

            let mut new_settings = base;
            if transform_flip {
                new_settings.input_transform = InputTransform::Grayscale;
            }
            let new_model = if model_flip { ModelKind::BiRefNetLite } else { ModelKind::Silueta };
            let new_recipe = new_settings.current_recipe(new_model, chain_flip);

            let actual = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);

            let mut expected = CacheImpact::Nothing;
            if chain_flip {
                expected = expected.union(spec(StaticKnob::ChainMode).cache_impact);
            }
            if model_flip {
                expected = expected.union(spec(StaticKnob::Model).cache_impact);
            }
            if transform_flip {
                expected = expected.union(
                    input_transform_spec(false).cache_impact,
                );
            }
            proptest::prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn cache_impact_for_recipe_diff_covers_common_knobs() {
        use prunr_core::{ComposeMode, EdgeScale, FillStyle, InputTransform, LineStyle, ModelKind};
        use prunr_core::{CompositeRecipe, EdgeRecipe, InferenceRecipe, MaskSettings};

        let base = ProcessingRecipe {
            inference: InferenceRecipe {
                model: ModelKind::Silueta,
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
            mask: (&MaskSettings {
                fill_style: FillStyle::None,
                ..Default::default()
            }).into(),
            composite: CompositeRecipe { bg_color: None, solid_line_color: None },
            was_chain: false,
        };

        assert_eq!(
            cache_impact_for_recipe_diff(&base, &base),
            CacheImpact::Nothing,
        );

        let mut model_swap = base.clone();
        model_swap.inference.model = ModelKind::BiRefNetLite;
        assert_eq!(
            cache_impact_for_recipe_diff(&base, &model_swap),
            CacheImpact::SegCache,
        );

        let mut transform_change = base.clone();
        transform_change.inference.input_transform = InputTransform::Grayscale;
        assert_eq!(
            cache_impact_for_recipe_diff(&base, &transform_change),
            CacheImpact::EdgeCache,
        );

        let mut chain_flip = base.clone();
        chain_flip.was_chain = true;
        assert_eq!(
            cache_impact_for_recipe_diff(&base, &chain_flip),
            CacheImpact::Both,
        );

        // Off → SubjectOutline: seg valid, edge tensor still valid (Off
        // didn't run DexiNed). Cache stays Nothing.
        let mut to_subject = base.clone();
        to_subject.inference.uses_edge_detection = true;
        assert_eq!(
            cache_impact_for_recipe_diff(&base, &to_subject),
            CacheImpact::Nothing,
        );
    }
}
