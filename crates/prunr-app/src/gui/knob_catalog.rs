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

use crate::gui::settings::SettingsModel;

/// Knobs whose `KnobSpec` is a pure function of the knob identity.
/// Context-sensitive knobs (line_mode transition, input_transform) live
/// in their own types so callers can't accidentally ask the catalog for
/// a static spec and get a conservative over-approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StaticKnob {
    // Mask tier — cached seg tensor reused.
    Gamma,
    Contrast,
    EdgeShift,
    RefineEdges,
    GuidedRadius,
    GuidedEpsilon,
    Feather,
    SubjectEffect,
    BgEffect,

    // Edge tier — cached TEED tensor reused.
    LineStrength,
    EdgeThickness,
    EdgeScale,
    SolidLineColor,
    ComposeMode,
    LineStyle,

    // Composite — render-time only.
    BgColor,
    BgImageFit,

    // Inference-level — needs worker subprocess.
    Model,

    // Mask tier.
    Decoration,
    /// The Depth chip: lens blur, fog, aerial perspective and focus.
    /// Re-renders the photo from the kept depth map.
    DepthLook,
    /// Sketch's Fade with distance.
    LineFade,
}

impl StaticKnob {
    /// All variants, for table-driven tests and catalog validation.
    pub const ALL: &'static [StaticKnob] = &[
        StaticKnob::Gamma,
        StaticKnob::Contrast,
        StaticKnob::EdgeShift,
        StaticKnob::RefineEdges,
        StaticKnob::GuidedRadius,
        StaticKnob::GuidedEpsilon,
        StaticKnob::Feather,
        StaticKnob::SubjectEffect,
        StaticKnob::BgEffect,
        StaticKnob::LineStrength,
        StaticKnob::EdgeThickness,
        StaticKnob::EdgeScale,
        StaticKnob::SolidLineColor,
        StaticKnob::ComposeMode,
        StaticKnob::LineStyle,
        StaticKnob::BgColor,
        StaticKnob::BgImageFit,
        StaticKnob::Model,
        StaticKnob::Decoration,
        StaticKnob::DepthLook,
        StaticKnob::LineFade,
    ];
}

/// Context-sensitive event: line-mode transition. Dispatch depends on
/// whether the TEED tensor is cached — `line_mode_spec` consumes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineModeChange {
    pub from: LineMode,
    pub to: LineMode,
}


/// What the selected model does to the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelRole {
    /// Background removal: cuts the subject out.
    Cutout,
    /// No model: filters and lines on the photo.
    Filters,
    /// Fills a painted region.
    Erase,
    /// Enlarges the image.
    Upscale,
    /// Colours a black-and-white photo.
    Colorize,
}

impl ModelRole {
    /// Brush strokes edit a cut-out, each with a direction (restore or
    /// erase), rather than shape a selection.
    pub fn edits_cutout(self) -> bool {
        self == ModelRole::Cutout
    }

    /// Runs only on the image picked, when asked (an eraser also needs a
    /// painted region), so never on import.
    pub fn runs_on_demand(self) -> bool {
        matches!(self, ModelRole::Erase | ModelRole::Upscale | ModelRole::Colorize)
    }
}

/// A surface or control that exists only for some models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Mask,
    Lines,
    Effect,
    /// The cut-out's sticker outline and shadow.
    Decoration,
    Background,
    /// The upscaler's Scale and Refine chips.
    Upscale,
    /// The colorizer's Strength chip.
    Colorize,
    /// The Stable Diffusion eraser's Quality, Prompt and Advanced chips.
    SdChips,
    /// The RORem eraser's Advanced chip: steps and seed.
    RoremChips,
    /// Paint and Magic Brush, the selection on the image and its actions.
    Brushes,
    /// The Paint Brush's mode (Restore/Erase or Add/Subtract), Hardness
    /// and Opacity: strokes that edit a cut-out or a selection by degree.
    /// The eraser's region is all or nothing, so none of them act there.
    SelectionStrokes,
    /// Expand region, Feather and Sharpen for the filled region.
    FillRegionKnobs,
    /// Restore faces: every model whose output is the photo.
    Faces,
    /// The Depth chip: lens blur, fog and aerial perspective on the photo.
    Depth,
}

/// Whether `surface` shows for `model`: only where it has an effect. A
/// surface that shows but cannot act now (Mask with Sketch on Whole
/// photo, before Process) is greyed with a reason by `surface_gate`.
pub fn shows(model: SettingsModel, surface: Surface) -> bool {
    use ModelRole::*;
    let role = model.role();
    match surface {
        Surface::Mask | Surface::Decoration => role == Cutout,
        Surface::Lines | Surface::Effect | Surface::Background => matches!(role, Cutout | Filters),
        Surface::Upscale => role == Upscale,
        Surface::Colorize => role == Colorize,
        Surface::SdChips => model == SettingsModel::SdInpaint,
        Surface::RoremChips => model == SettingsModel::RoremInpaint,
        Surface::Brushes => !matches!(role, Upscale | Colorize),
        Surface::SelectionStrokes => matches!(role, Cutout | Filters),
        Surface::FillRegionKnobs => role == Erase,
        Surface::Faces => matches!(role, Cutout | Filters | Upscale),
        Surface::Depth => role == Filters,
    }
}

/// What a surface's gate reads: the model, the image's facts, and
/// whether a run is under way.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GateContext {
    pub model: SettingsModel,
    pub facts: crate::gui::capability::ItemFacts,
    /// A run is under way anywhere in the app, not only on this image.
    pub processing: bool,
}

/// Why a shown surface's controls are off right now, `None` when they
/// work: the one gate the toolbar greys a surface by. Controls that need
/// the cut-out wait for Process; everything else works on what the image
/// shows.
pub(crate) fn surface_gate(surface: Surface, g: &GateContext) -> Option<&'static str> {
    use crate::gui::capability::{check, Need};
    let role = g.model.role();
    let needs = |need: Need| check(&g.facts, need, role).reason().map(crate::gui::capability::Reason::text);
    // A knob an upscale reads changed mid-run makes the landing result
    // stale and drops it.
    let upscale_lock = (g.processing && role == ModelRole::Upscale).then_some("Wait for the upscale to finish, or cancel it.");
    match surface {
        Surface::Mask | Surface::Decoration => needs(Need::Cutout),
        Surface::Effect => needs(Need::Subject),
        Surface::Background => needs(Need::Transparency),
        Surface::Upscale => upscale_lock,
        // On an upscaler Faces is a setting of the run; elsewhere it works
        // on the image shown.
        Surface::Faces if role == ModelRole::Upscale => upscale_lock,
        Surface::Faces => needs(Need::ShownImage),
        Surface::Colorize => g.processing.then_some("Wait for Colorize to finish, or cancel it."),
        Surface::Lines => needs(Need::Processed),
        Surface::SdChips
        | Surface::RoremChips
        | Surface::Brushes
        | Surface::SelectionStrokes
        | Surface::FillRegionKnobs
        | Surface::Depth => None,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheImpact {
    #[default]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchKind {
    #[default]
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
    /// for Model (user should review and click Process).
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
        | StaticKnob::Contrast
        | StaticKnob::EdgeShift
        | StaticKnob::RefineEdges
        | StaticKnob::GuidedRadius
        | StaticKnob::GuidedEpsilon
        | StaticKnob::Feather
        | StaticKnob::SubjectEffect
        | StaticKnob::Decoration
        | StaticKnob::DepthLook => s(MaskRerun, Nothing, LivePreviewMask, true),

        StaticKnob::LineStrength
        | StaticKnob::EdgeThickness
        | StaticKnob::EdgeScale
        | StaticKnob::SolidLineColor
        | StaticKnob::ComposeMode
        | StaticKnob::LineStyle
        | StaticKnob::LineFade => s(EdgeRerun, Nothing, LivePreviewEdge, true),

        StaticKnob::BgColor
        | StaticKnob::BgEffect
        | StaticKnob::BgImageFit => s(CompositeOnly, Nothing, Render, true),

        // Model: user must click Process.
        StaticKnob::Model => s(FullPipeline, SegCache, SubprocessFullPipeline, false),
    }
}

/// Precise spec for a line-mode transition.
///
/// `cached_edge` gates the `Off → SubjectOutline` fast path: with the
/// TEED tensor in memory we recompose via live-preview; otherwise we
/// rerun TEED on the cached seg (AddEdgeInference subprocess).
///
/// Transitions to/from `Off` preserve the TEED tensor — `Off` doesn't
/// use it. `SubjectOutline ↔ EdgesOnly` invalidates edge cache because
/// TEED sees different input (masked-subject vs raw scene).
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
        // The cached lines were traced from the whole photo; Subject later
        // must not reuse them.
        (EdgesOnly, Off) => s(FullPipeline, EdgeCache, SubprocessFullPipeline),
        (EdgesOnly, SubjectOutline) => s(FullPipeline, EdgeCache, SubprocessFullPipeline),
    }
}

/// Precise spec for an input-transform change. With a cached seg tensor
/// we reuse it and rerun TEED only; otherwise a full pipeline is needed.
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
/// impacts for every field that actually changed. Used by `BatchItem::run_tier`
/// to invalidate only the caches whose INPUT changed (seg stays valid on an
/// input_transform change, edge stays valid on a model swap).
pub fn cache_impact_for_recipe_diff(old: &ProcessingRecipe, new: &ProcessingRecipe) -> CacheImpact {
    let mut impact = CacheImpact::Nothing;
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
    /// The toolbar's one gate: cut-out surfaces wait for Process on a
    /// cut-out model, work on No model's photo, and every surface's
    /// reason reads as a sentence.
    #[test]
    fn surface_gates_follow_what_the_image_shows() {
        use crate::gui::capability::{Activity, ItemFacts, Shown};
        use crate::gui::item::Engine;
        let photo = ItemFacts::photo();
        let cut = ItemFacts { shown: Shown::Run(Engine::Segment), cutout_mask: true, ..ItemFacts::photo() };
        let g = |model, facts, processing| GateContext { model, facts, processing };
        let cutout_surfaces = [Surface::Mask, Surface::Decoration, Surface::Effect, Surface::Background];
        for s in cutout_surfaces {
            assert_eq!(surface_gate(s, &g(SettingsModel::Silueta, photo, false)), Some("Process first."), "{s:?} before Process");
            assert_eq!(surface_gate(s, &g(SettingsModel::Silueta, cut, false)), None, "{s:?} on a cut-out");
        }
        assert_eq!(surface_gate(Surface::Effect, &g(SettingsModel::None, photo, false)), None, "Effect on No model's photo");
        assert!(surface_gate(Surface::Background, &g(SettingsModel::None, photo, false)).is_some_and(|r| r.starts_with("Nothing is transparent")),
            "No model with Sketch off has nothing transparent to fill");
        assert_eq!(surface_gate(Surface::Faces, &g(SettingsModel::Silueta, photo, false)), None, "Faces work before Process");
        let upscaling = ItemFacts { activity: Activity::Running(Engine::Upscale), ..photo };
        assert!(surface_gate(Surface::Faces, &g(SettingsModel::RealEsrganUpscale, upscaling, true)).is_some(), "Faces wait for the upscale");
        assert!(surface_gate(Surface::Upscale, &g(SettingsModel::RealEsrganUpscale, upscaling, true)).is_some());
        let whole = |facts: ItemFacts| ItemFacts { line_mode: LineMode::EdgesOnly, ..facts };
        assert_eq!(surface_gate(Surface::Mask, &g(SettingsModel::Silueta, whole(cut), false)), Some("Sketch is on Whole photo, so nothing is cut out."));
        assert_eq!(surface_gate(Surface::Effect, &g(SettingsModel::None, whole(photo), false)), Some("Sketch is on Whole photo, so there is no subject to restyle."));
        assert_eq!(surface_gate(Surface::Background, &g(SettingsModel::Silueta, whole(cut), false)), None, "lines over a cut-out leave a background to fill");
        assert_eq!(surface_gate(Surface::Background, &g(SettingsModel::None, whole(photo), false)), None, "lines on No model's photo leave the rest transparent");
    }

    use super::*;

    #[test]
    fn every_static_knob_has_a_spec() {
        for k in StaticKnob::ALL {
            let _ = spec(*k);
        }
    }

    /// Every model shows exactly the surfaces that act on its image. One
    /// row per model, written out rather than derived, so a new model or
    /// a moved surface has to be decided here.
    #[test]
    fn each_model_shows_only_what_works_with_it() {
        use Surface::*;
        let all = [Mask, Lines, Effect, Decoration, Background, Upscale, Colorize, SdChips, Brushes, SelectionStrokes, FillRegionKnobs, Faces, Depth];
        let cutout: &[Surface] = &[Mask, Lines, Effect, Decoration, Background, Brushes, SelectionStrokes, Faces];
        let filters: &[Surface] = &[Lines, Effect, Background, Brushes, SelectionStrokes, Faces, Depth];
        let erase: &[Surface] = &[Brushes, FillRegionKnobs];
        let sd: &[Surface] = &[Brushes, FillRegionKnobs, SdChips];
        let upscale: &[Surface] = &[Upscale, Faces];
        let colorize: &[Surface] = &[Colorize];
        let rows: &[(SettingsModel, &[Surface])] = &[
            (SettingsModel::Silueta, cutout),
            (SettingsModel::U2net, cutout),
            (SettingsModel::BiRefNetLite, cutout),
            (SettingsModel::None, filters),
            (SettingsModel::Inpaint, erase),
            (SettingsModel::BigInpaint, erase),
            (SettingsModel::MiganInpaint, erase),
            (SettingsModel::SdInpaint, sd),
            (SettingsModel::RoremInpaint, erase),
            (SettingsModel::RealEsrganUpscale, upscale),
            (SettingsModel::Nomos8kUpscale, upscale),
            (SettingsModel::FourXNmkdSiaxCxUpscale, upscale),
            (SettingsModel::FourXNmkdSuperscaleUpscale, upscale),
            (SettingsModel::Nomos2Upscale, upscale),
            (SettingsModel::ColorizeDdcolor, colorize),
        ];
        assert_eq!(rows.len(), SettingsModel::ALL.len(), "every model has a row");
        for &(model, shown) in rows {
            for surface in all {
                assert_eq!(shows(model, surface), shown.contains(&surface), "{model:?} {surface:?}");
            }
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
            (EdgesOnly, Off, false, EdgeCache, SubprocessFullPipeline),
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
    /// subprocess path, spawning a cold TEED rerun on each flip. Guard
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
        use prunr_core::{ComposeMode, EdgeScale, SubjectEffect, LineStyle};
        let mut new = base;
        match knob {
            StaticKnob::Gamma => new.gamma = base.gamma + 0.5,
            StaticKnob::Contrast => new.contrast = base.contrast + 0.1,
            StaticKnob::EdgeShift => new.edge_shift = base.edge_shift + 1.0,
            StaticKnob::RefineEdges => new.refine_edges = !base.refine_edges,
            StaticKnob::GuidedRadius => new.guided_radius = base.guided_radius + 1,
            StaticKnob::GuidedEpsilon => new.guided_epsilon = base.guided_epsilon * 2.0,
            StaticKnob::Feather => new.feather = base.feather + 0.5,
            StaticKnob::SubjectEffect => {
                new.effect = match base.effect {
                    SubjectEffect::None => SubjectEffect::Invert,
                    _ => SubjectEffect::None,
                };
            }
            StaticKnob::Decoration => new.decoration.outline_width = base.decoration.outline_width + 4.0,
            StaticKnob::BgEffect => {
                new.bg_effect = match base.bg_effect {
                    prunr_core::BgEffect::None => prunr_core::BgEffect::DesaturatedSource,
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
            StaticKnob::BgImageFit => {
                new.bg_image_fit = match base.bg_image_fit {
                    prunr_core::BgImageFit::Cover => prunr_core::BgImageFit::Tile,
                    _ => prunr_core::BgImageFit::Cover,
                };
            }
            StaticKnob::DepthLook => new.depth.fog_pct = base.depth.fog_pct + 30,
            StaticKnob::LineFade => new.line_fade_pct = base.line_fade_pct + 30,
            StaticKnob::Model => {
                // Model lives outside ItemSettings; tested via
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
        for knob in StaticKnob::ALL {
            if matches!(knob, StaticKnob::Model) {
                continue;
            }
            let new = mutate_static(*knob, base);
            let model = ModelKind::Silueta;
            let old_recipe = base.current_recipe(model);
            let new_recipe = new.current_recipe(model);

            let tier_from_resolve = prunr_core::resolve_tier(&old_recipe, &new_recipe);
            let impact_from_diff = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);
            let cat_spec = spec(*knob);

            assert_eq!(
                cat_spec.cache_impact, impact_from_diff,
                "knob {knob:?}: catalog cache_impact {:?} != diff {:?}",
                cat_spec.cache_impact, impact_from_diff,
            );
            assert!(
                cat_spec.tier >= tier_from_resolve,
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
            let old_recipe = old.current_recipe(ModelKind::Silueta);
            let new_recipe = new.current_recipe(ModelKind::Silueta);
            let cat = line_mode_spec(lmc(from, to), false);
            let diff = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);
            assert_eq!(cat.cache_impact, diff, "{from:?} -> {to:?}");
        }
    }

    #[test]
    fn input_transform_cache_impact_matches_recipe_diff() {
        use prunr_core::{InputTransform as It, ModelKind};
        let old = crate::gui::item_settings::item_with_line_mode(LineMode::SubjectOutline);
        let new = crate::gui::item_settings::ItemSettings {
            line_mode: LineMode::SubjectOutline,
            input_transform: It::Grayscale,
            ..Default::default()
        };
        let diff = cache_impact_for_recipe_diff(
            &old.current_recipe(ModelKind::Silueta),
            &new.current_recipe(ModelKind::Silueta),
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
            contrast in 0.0f32..=1.0,
            edge_shift in -50.0f32..50.0,
            line_mode in proptest::prelude::prop_oneof![
                proptest::prelude::Just(LineMode::Off),
                proptest::prelude::Just(LineMode::EdgesOnly),
                proptest::prelude::Just(LineMode::SubjectOutline),
            ],
        ) {
            let s = crate::gui::item_settings::ItemSettings {
                gamma,
                contrast,
                edge_shift,
                line_mode,
                ..Default::default()
            };
            let r = s.current_recipe(prunr_core::ModelKind::Silueta);
            proptest::prop_assert_eq!(prunr_core::resolve_tier(&r, &r), RequiredTier::Skip);
        }

        /// Cache impact is additive: applying two independent knobs in sequence
        /// matches the union of their individual impacts.
        #[test]
        fn cache_impact_is_additive(
            transform_flip in proptest::prelude::any::<bool>(),
            model_flip in proptest::prelude::any::<bool>(),
        ) {
            use prunr_core::{InputTransform, ModelKind};
            let base = crate::gui::item_settings::item_with_line_mode(LineMode::SubjectOutline);
            let old_recipe = base.current_recipe(ModelKind::Silueta);

            let mut new_settings = base;
            if transform_flip {
                new_settings.input_transform = InputTransform::Grayscale;
            }
            let new_model = if model_flip { ModelKind::BiRefNetLite } else { ModelKind::Silueta };
            let new_recipe = new_settings.current_recipe(new_model);

            let actual = cache_impact_for_recipe_diff(&old_recipe, &new_recipe);

            let mut expected = CacheImpact::Nothing;
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
        use prunr_core::{ComposeMode, EdgeScale, SubjectEffect, InputTransform, LineStyle, ModelKind};
        use prunr_core::{CompositeRecipe, EdgeRecipe, InferenceRecipe, MaskSettings, UpscaleRecipe};

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
                line_fade_pct: 0,
            },
            mask: (&MaskSettings {
                effect: SubjectEffect::None,
                ..Default::default()
            }).into(),
            composite: CompositeRecipe::default(),
            upscale: UpscaleRecipe::default(),
            colorize: Default::default(),
            depth: Default::default(),
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

        // Off → SubjectOutline: seg valid, edge tensor still valid (Off
        // didn't run TEED). Cache stays Nothing.
        let mut to_subject = base.clone();
        to_subject.inference.uses_edge_detection = true;
        assert_eq!(
            cache_impact_for_recipe_diff(&base, &to_subject),
            CacheImpact::Nothing,
        );
    }
}
