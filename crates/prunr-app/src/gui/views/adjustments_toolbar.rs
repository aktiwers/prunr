//! Rows 2 + 3 of the persistent toolbar. Row 2 holds the mask / composite
//! adjustments (gamma, threshold, edge shift, refine edges, bg color).
//! Row 3 holds the line-specific knobs (line strength, solid line color)
//! and is only visible when `line_mode != Off`.
//!
//! View Component discipline: `render` takes `&mut ItemSettings` + a `&AppSettings`
//! reference for defaults / live-preview flag lookups. Never `&mut PrunrApp`.
//!
//! Returns a `ToolbarChange` summarizing WHAT changed so the caller can
//! invalidate the right textures and schedule live-preview reruns.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use crate::gui::brush_state::BrushState;
use crate::gui::item_settings::ItemSettings;
use crate::gui::knob_catalog::{
    self, CacheImpact, DispatchKind, KnobSet, LineModeChange, StaticKnob,
};
use crate::gui::settings::{Settings, SettingsModel};
use crate::gui::theme;
use crate::gui::views::{chip, hint, preset_dropdown};
use prunr_core::LineMode;

use super::model_label;

/// Append the "Press F3 for the full pipeline." hint to a chip tooltip at
/// compile time. Single-point of truth for the hint suffix so an F3 rebind
/// or rewording doesn't require touching every chip.
macro_rules! tip {
    ($body:literal) => {
        concat!($body, "\n\nPress F3 for the full pipeline.")
    };
}

/// Summary of what a toolbar render cycle changed.
///
/// Tier flags drive live-preview dispatch; cache-invalidation flags are
/// granular so we don't clear a still-valid cache (e.g. seg tensor stays
/// good after a line_mode toggle — only the edge tensor is stale then).
/// Keeping unrelated caches alive means the user's next mask/edge tweak
/// can still live-preview without a full Process.
#[derive(Debug, Clone, Copy)]
pub struct ToolbarChange {
    /// Any chip signalled a committed value (slider released, toggle
    /// flipped, color picked). Flushes pending debounced previews.
    pub commit: bool,
    /// The model dropdown flipped — triggers settings save + toast. Does
    /// not auto-reprocess; user must click Process.
    pub model_changed: bool,
    /// A preset was applied — archive the pre-apply snapshot for undo.
    /// The dispatcher resolves subprocess vs skip from the recipe diff.
    pub preset_applied: bool,
    /// Which static knobs were touched (`Copy` bitset). Provides a
    /// single-allocation record for downstream passes that want to
    /// audit "was X touched this frame?" without a salad of booleans.
    pub touched: KnobSet,
    /// Previous `line_mode` when it changed this frame. Drives the
    /// context-sensitive `line_mode_spec(from, current, cached_edge)` path.
    /// `None` means no transition this frame.
    pub line_mode_from: Option<LineMode>,
    /// `input_transform` flipped this frame. Drives `input_transform_spec`
    /// for precise dispatch. The bool is here because `StaticKnob` (and
    /// thus `KnobSet`) intentionally excludes context-sensitive knobs.
    pub input_transform_changed: bool,

    /// Catalog-derived aggregate cache invalidation — union over all knobs.
    pub cache_impact: CacheImpact,
    /// Strongest auto-fire dispatch (live-preview or subprocess) from knobs
    /// with `auto_trigger_on_commit = true`. `DispatchKind::None` if no
    /// auto-dispatchable knob was touched. Routed unconditionally when
    /// the item is Done (live-preview fires regardless of Done status).
    pub auto_dispatch: DispatchKind,
    /// A render-only knob (bg color) fired — request a repaint even when
    /// no other dispatch kicks in.
    pub render_repaint: bool,
    /// User clicked "Clear strokes" in the brush popover.
    pub clear_correction_requested: bool,
    /// Brush popover settled a change AND `app_settings.brush` was synced.
    pub brush_settings_committed: bool,
    /// Set when the user clicked "More models…" or a not-yet-installed
    /// dropdown entry. `filter = None` means "show everything"; the
    /// dropdown's not-installed click pre-filters to the entry's category.
    pub open_model_store: Option<ModelStoreRequest>,
    /// User chose Image in the bg chip — `apply_toolbar_change` opens the
    /// file picker, decodes, and calls `BatchItem::set_bg_image`.
    pub pick_bg_image: bool,
    /// User picked a non-image bg kind (color or effect) while a bg image
    /// was active — drop the image so the chosen kind takes over.
    pub clear_bg_image: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelStoreRequest {
    pub filter: Option<prunr_models::ModelCategory>,
}

impl Default for ToolbarChange {
    fn default() -> Self {
        Self {
            commit: false,
            model_changed: false,
            preset_applied: false,
            touched: KnobSet::default(),
            line_mode_from: None,
            input_transform_changed: false,
            cache_impact: CacheImpact::Nothing,
            auto_dispatch: DispatchKind::None,
            render_repaint: false,
            clear_correction_requested: false,
            brush_settings_committed: false,
            open_model_store: None,
            pick_bg_image: false,
            clear_bg_image: false,
        }
    }
}

/// Factory default values for per-chip reset. Per-chip reset sends the value
/// back to ItemSettings::default() (not the user's default preset) — single
/// knobs should reset predictably, independent of whatever preset is loaded.
/// The separate "Reset all knobs" button below goes back to the default
/// preset instead; that's the per-user "what I usually want" anchor.
struct Defaults {
    template: ItemSettings,
    /// "Pick this when user toggles enabled" fallback for Option chips that
    /// have no factory value (threshold/bg/line_color are None by default but
    /// the color/slider inside the popover needs a starting value).
    threshold_value: f32,
    bg_value: [u8; 4],
    solid_line_color_value: [u8; 3],
}

impl Defaults {
    fn new() -> Self {
        Self {
            template: ItemSettings::default(),
            threshold_value: 0.5,
            bg_value: [255, 255, 255, 255],
            solid_line_color_value: [0, 0, 0],
        }
    }
}

/// Render rows 2 + 3. Returns a `ToolbarChange` summarizing what was edited.
/// `app_settings` exposes model + preset map (Row 2 hosts both dropdowns).
/// `applied_preset` is read for the button's modified/clean icon and written
/// in place when the user applies or saves a preset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    app_settings: &mut Settings,
    applied_preset: &mut String,
    brush_state: &mut BrushState,
    brush_available: bool,
    processing: bool,
    has_bg_image: bool,
    bg_image_label: Option<&str>,
) -> ToolbarChange {
    let mut change = ToolbarChange::default();
    let defaults = Defaults::new();

    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;

    // Snapshots for smart cache invalidation. We compare before/after so we
    // only clear caches whose INPUT actually changed, not every cache on
    // every apply. Matters for live preview: a preset apply that keeps
    // line_mode the same leaves the edge tensor valid, so subsequent
    // line_strength tweaks still live-preview without needing a Process.
    let before_line_mode = item_settings.line_mode;

    // Knob-enablement rules:
    // - mask_active:   mask chips (gamma/threshold/edge_shift/refine/
    //                  feather) operate on the seg mask. Dead when model
    //                  skips seg (No model), or when line_mode is
    //                  EdgesOnly without chain mode (edges-only doesn't
    //                  produce a mask).
    // - fill_style:    transforms subject RGB. Dead when there IS no
    //                  subject — i.e. EdgesOnly (lines only, transparent
    //                  bg).
    // - bg_active:     fills transparent areas. Dead only when there's no
    //                  transparency at all — filter-only mode (No model +
    //                  Off) outputs a full-RGB image with nothing to fill.
    let model_uses_seg = app_settings.model.uses_segmentation();
    // Snapshot pre-dropdown — the model dropdown below can mutate
    // `app_settings.model`. Branches that need post-dropdown state
    // (e.g. the auto-flip-to-Off + brush prewarm at line 218) re-read
    // `app_settings.model.is_inpaint()` directly.
    let inpaint_mode = app_settings.model.is_inpaint();
    let mask_active = model_uses_seg
        && (item_settings.line_mode != LineMode::EdgesOnly || app_settings.chain_mode);
    let fill_style_active = item_settings.line_mode != LineMode::EdgesOnly;
    let bg_active = !(matches!(app_settings.model, SettingsModel::None)
        && item_settings.line_mode == LineMode::Off);

    ui.horizontal(|ui| {
        render_model_dropdown(ui, app_settings, processing, mask_active, &mut change);

        // If the user just picked "No model" while line_mode was
        // SubjectOutline, auto-flip to Off — the invalid combination (no
        // seg but compose-over-subject) would just silently render as
        // filter-only anyway.
        if change.model_changed
            && !app_settings.model.uses_segmentation()
            && item_settings.line_mode == LineMode::SubjectOutline
        {
            item_settings.line_mode = LineMode::Off;
        }

        // Inpaint mode: paint is the only input — auto-enable brush so
        // the user doesn't have to click two buttons. Settings stays
        // pinned subtract-equivalent (mode picker is hidden in the
        // popover anyway when Inpaint is active). Also pre-warm the
        // LaMa session in the background so the first stroke doesn't
        // pay the 5-10s zstd-decompress + ORT-session-build cost.
        if change.model_changed && app_settings.model.is_inpaint() {
            if !brush_state.is_enabled() {
                brush_state.toggle();
            }
            // Pre-warm the specific backend the dispatch will actually
            // use. For SD with FAST mode active, dispatch routes through
            // `lcm_routing_active()` to the LCM bundle — prewarming the
            // standard SD bundle here would build TWO bundles (one
            // standard via prewarm, one LCM via dispatch), each ~15 GB,
            // tipping a low-RAM machine into the SD-bundle guard's
            // refusal path.
            let raw_id = app_settings.model.to_model_id();
            let id = match raw_id {
                Some(prunr_models::ModelId::SdV15InpaintFp16)
                    if app_settings.lcm_routing_active(prunr_models::ModelId::SdV15InpaintFp16) =>
                {
                    Some(prunr_models::ModelId::SdV15LcmInpaintFp16)
                }
                other => other,
            };
            if let Some(id) = id {
                rayon::spawn(move || {
                    if let Err(e) = prunr_core::inpaint::prewarm(id) {
                        tracing::warn!(?id, %e, "Inpaint prewarm failed");
                    }
                });
            }
        }

        if !inpaint_mode {
            render_seg_mask_chips(
                ui,
                item_settings,
                &SegRowFlags {
                    defaults: &defaults,
                    mask_active,
                    fill_style_active,
                    bg_active,
                    has_bg_image,
                    bg_image_label,
                },
                &mut change,
            );
        }

        // Right-aligned cluster: reset, preset. Right-to-left layout fills
        // from the right edge so items stack: [..free space..] [preset] [↺].
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                // Reset-all-knobs button visible directly (per user feedback —
                // was previously hidden in the kebab overflow menu).
                let reset_target = app_settings.default_preset.clone();
                let reset_tooltip = format!(
                    "Reset all knobs to the \"{reset_target}\" preset (your default)"
                );
                let reset_resp = chip::icon_toggle_button(ui, ICON_RESTART_ALT.codepoint, false);
                if reset_resp.on_hover_text(reset_tooltip).clicked() {
                    *item_settings = app_settings.preset_values(&reset_target);
                    *applied_preset = reset_target;
                    mark_preset_apply(&mut change);
                }

                if let Some(name) = preset_dropdown::render(ui, app_settings, item_settings, applied_preset) {
                    *applied_preset = name;
                    mark_preset_apply(&mut change);
                }

                let brush_active = brush_state.is_enabled();
                let brush_tooltip = if !brush_available {
                    "Brush is available after processing — run the image through a model first."
                } else if brush_active {
                    "Brush mode ON — click on canvas to add (positive) / subtract (negative) mask. Click here to disable."
                } else {
                    "Toggle brush mode: paint corrections onto the mask"
                };
                ui.add_enabled_ui(brush_available, |ui| {
                    let brush_resp = chip::icon_toggle_button(ui, ICON_BRUSH.codepoint, brush_active);
                    if brush_resp.on_hover_text(brush_tooltip).clicked() {
                        brush_state.toggle();
                    }
                });

                // Settings chip — only visible when brush is on AND has
                // somewhere to paint. In the right-to-left layout this
                // appears LEFT of the toggle.
                if brush_available && brush_state.is_enabled() {
                    let is_sd = matches!(app_settings.model, crate::gui::settings::SettingsModel::SdInpaint);
                    // Same predicate as the dispatch path so UI greying
                    // and routing can't disagree.
                    let sd_fast_mode = is_sd
                        && app_settings.lcm_routing_active(prunr_models::ModelId::SdV15InpaintFp16);
                    let outcome = super::brush_chip::render(
                        ui, brush_state, app_settings.model.is_inpaint(), is_sd, sd_fast_mode,
                    );
                    if outcome.clear_requested {
                        change.clear_correction_requested = true;
                    }
                    if outcome.committed && app_settings.brush != *brush_state.settings() {
                        app_settings.brush = brush_state.settings().clone();
                        change.brush_settings_committed = true;
                    }
                }
            },
        );
    });

    // ── Row 3: Lines mode selector (always visible) + line knobs (when Lines
    // mode is on). The Lines dropdown is the gate for enabling line
    // extraction; keeping it visible at all times is how the user turns
    // lines on. `lines_popover::render` mutates `item_settings.line_mode`
    // directly — cache invalidation below picks up the change.
    //
    // Eraser models hide Row 3 entirely — DexiNed isn't part of the
    // inpaint pipeline.
    if !inpaint_mode {
        render_lines_row(
            ui,
            item_settings,
            &LinesRowFlags {
                app_settings,
                defaults: &defaults,
                processing,
                mask_active,
            },
            &mut change,
        );
    }

    // Line-mode transition: record the signal + cache impact (deterministic
    // in `(from, to)`). The dispatcher owns dispatch resolution — folding a
    // worst-case here would dominate the refined fast path via `.max()`
    // (e.g. Off→Subject with a warm edge cache would be routed to
    // SubprocessAddEdge instead of LivePreviewMask).
    if item_settings.line_mode != before_line_mode {
        let spec = knob_catalog::line_mode_spec(
            LineModeChange { from: before_line_mode, to: item_settings.line_mode },
            false,
        );
        change.cache_impact = change.cache_impact.union(spec.cache_impact);
        change.line_mode_from = Some(before_line_mode);
        change.commit = true;
    }

    change
}

/// Read-only bundle for the seg/filter row 2 chip cluster. Bundles the
/// inputs that don't need mutation to keep `render_seg_mask_chips` at
/// 4 params (under the 6-param alarm).
struct SegRowFlags<'a> {
    defaults: &'a Defaults,
    mask_active: bool,
    fill_style_active: bool,
    bg_active: bool,
    has_bg_image: bool,
    bg_image_label: Option<&'a str>,
}

fn render_seg_mask_chips(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    flags: &SegRowFlags<'_>,
    change: &mut ToolbarChange,
) {
    let defaults = flags.defaults;
    ui.add_enabled_ui(flags.mask_active, |ui| {
        aggregate_knob(chip::chip_f32(
            ui,
            chip::ChipMeta {
                id_salt: "gamma",
                icon: "γ",
                label: "Gamma",
                description: "How hard the mask cuts. >1 is more aggressive; <1 is gentler.",
                tooltip: tip!("Stage 1 of 5. How hard the mask cuts. >1 removes more aggressively, <1 is gentler on fine edges. Feeds every stage below."),
            },
            &mut item_settings.gamma,
            0.01..=10.0, defaults.template.gamma,
            true, // log scale — matches perceptual symmetry around 1.0
            |v| format!("{v:.2}"),
        ), StaticKnob::Gamma, change);

        aggregate_knob(chip::chip_option_f32(
            ui,
            chip::ChipMeta {
                id_salt: "threshold",
                icon: ICON_BOLT.codepoint,
                label: "Hard threshold",
                description: "Snap the mask to fully opaque or fully transparent at this cutoff.",
                tooltip: tip!("Stage 2 of 5. Snaps the mask to fully opaque or fully transparent at this cutoff. Soft = smooth alpha, on = crisp silhouette. When on, downstream stages lose the gradient — Refine can only clean up stairsteps."),
            },
            &mut item_settings.threshold,
            0.001..=0.999, defaults.threshold_value, "Soft",
            |v| format!("{:.1}%", v * 100.0),
        ), StaticKnob::Threshold, change);

        aggregate_knob(chip::chip_f32(
            ui,
            chip::ChipMeta {
                id_salt: "edge_shift",
                icon: ICON_SWAP_HORIZ.codepoint,
                label: "Edge shift",
                description: "Shrink or grow the mask outline. Positive erodes; negative dilates.",
                tooltip: tip!("Stage 3 of 5. Shrink or grow the mask outline. Positive = erode (trim fringe pixels), negative = dilate (keep more edge detail). Refine Edges then snaps the shifted boundary to image color."),
            },
            &mut item_settings.edge_shift,
            -50.0..=50.0, defaults.template.edge_shift,
            false,
            |v| {
                if v > 0.05 { format!("erode {v:.1}px") }
                else if v < -0.05 { format!("dilate {:.1}px", v.abs()) }
                else { "0px".to_string() }
            },
        ), StaticKnob::EdgeShift, change);

        aggregate_knob(chip::chip_bool_with_extras(
            ui,
            chip::ChipMeta {
                id_salt: "refine_edges",
                icon: ICON_AUTO_FIX_HIGH.codepoint,
                label: "Refine edges",
                description: "Use the original image's colors to sharpen the mask around fine detail like hair or leaves.",
                tooltip: tip!("Stage 4 of 5. Uses the original image's colors to sharpen the mask around fine detail like hair or leaves. Sees whatever threshold + edge shift produced, so tighter upstream input gives a tighter result. Slower but higher quality."),
            },
            &mut item_settings.refine_edges,
            |ui| {
                let mut inner = chip::ChipChange::default();
                let r = chip::slider_row_u32(
                    ui, "Refine radius (px)",
                    &mut item_settings.guided_radius,
                    1..=64,
                );
                if r.changed { inner.changed = true; }
                if r.commit  { inner.commit  = true; }
                let e = chip::slider_row_f32(
                    ui, "Refine strength (ε)",
                    &mut item_settings.guided_epsilon,
                    1e-6..=1e-2,
                    true,
                    |v| format!("{v:.1e}"),
                );
                if e.changed { inner.changed = true; }
                if e.commit  { inner.commit  = true; }
                inner
            },
        ), StaticKnob::RefineEdges, change);

        aggregate_knob(chip::chip_f32(
            ui,
            chip::ChipMeta {
                id_salt: "feather",
                icon: ICON_BLUR_LINEAR.codepoint,
                label: "Feather",
                description: "Soften mask edges with a Gaussian blur.",
                tooltip: tip!("Stage 5 of 5. Final softening pass — Gaussian blur over the finished mask. Runs last so it smooths whatever Refine Edges sharpened; reach for Feather when Refine can't pick up the right detail."),
            },
            &mut item_settings.feather,
            0.0..=10.0, defaults.template.feather,
            false,
            |v| if v < 0.1 { "off".into() } else { format!("σ {v:.1}") },
        ), StaticKnob::Feather, change);
    });

    // FillStyle is independent of the seg mask — it also works in
    // filter-only mode (No model + Off), where it applies to the raw
    // source. Only EdgesOnly (no subject) kills it.
    ui.add_enabled_ui(flags.fill_style_active, |ui| {
        let changed = render_fill_style_chip(ui, &mut item_settings.fill_style);
        aggregate_bool(changed, StaticKnob::FillStyle, change);
    });

    ui.separator();

    // Unified Background chip — Transparent / Solid colour / source-derived
    // effects. The two underlying fields (`bg`, `bg_effect`) stay orthogonal;
    // the chip enforces mutual exclusivity at the UI layer.
    ui.add_enabled_ui(flags.bg_active, |ui| {
        render_background_chip(ui, BgChipState {
            bg: &mut item_settings.bg,
            bg_effect: &mut item_settings.bg_effect,
            bg_image_fit: &mut item_settings.bg_image_fit,
            default_color: defaults.bg_value,
            has_bg_image: flags.has_bg_image,
            bg_image_label: flags.bg_image_label,
        }, change);
    });
}

struct LinesRowFlags<'a> {
    app_settings: &'a Settings,
    defaults: &'a Defaults,
    processing: bool,
    mask_active: bool,
}

fn render_lines_row(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    flags: &LinesRowFlags<'_>,
    change: &mut ToolbarChange,
) {
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        let seg_model_name = super::model_name(flags.app_settings.model);
        let subject_available = flags.app_settings.model.uses_segmentation();
        ui.add_enabled_ui(!flags.processing, |ui| {
            let _ = super::lines_popover::render(ui, item_settings, seg_model_name, subject_available);
        });
        if !flags.mask_active {
            ui.label(
                RichText::new(format!(
                    "{}  DexiNed only",
                    ICON_BLOCK.codepoint,
                ))
                .color(theme::TEXT_SECONDARY)
                .size(theme::FONT_SIZE_MONO),
            );
        }
        if item_settings.line_mode != LineMode::Off {
            render_row2_right_cluster(ui, item_settings, flags.defaults, change);
        }
    });
}

/// Compose-mode is hidden outside SubjectOutline because the worker
/// ignores it there — no point showing a dead control.
fn render_row2_right_cluster(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    defaults: &Defaults,
    change: &mut ToolbarChange,
) {
    // DualScale generates Fine + Bold internally and ignores `edge_scale`
    // (see `prunr-core/src/edge.rs::finalize_dual_scale`).
    let scale_active = !matches!(item_settings.line_style, prunr_core::LineStyle::DualScale { .. });
    let scale_changed = chip::guarded(
        ui,
        scale_active,
        "DualScale uses Fine + Bold internally; the scale chip has no effect under this line style.",
        |ui| super::lines_popover::render_scale_chip(ui, item_settings),
    );
    aggregate_bool(scale_changed, StaticKnob::EdgeScale, change);

    aggregate_knob(chip::chip_f32(
        ui,
        chip::ChipMeta {
            id_salt: "line_strength",
            icon: ICON_TUNE.codepoint,
            label: "Line strength",
            description: "How much edge detail to capture. Lower = bold outlines only; higher = fine texture.",
            tooltip: "Stage 2 of 4 in the lines pipeline. Threshold on DexiNed's raw edge tensor. Lower = bold outlines only; higher = fine texture and subtle edges. Feeds edge thickness + solid color.",
        },
        &mut item_settings.line_strength,
        0.0..=1.0, defaults.template.line_strength,
        false,
        |v| format!("{v:.2}"),
    ), StaticKnob::LineStrength, change);

    aggregate_knob(chip::chip_u32(
        ui,
        chip::ChipMeta {
            id_salt: "edge_thickness",
            icon: ICON_LINE_WEIGHT.codepoint,
            label: "Edge thickness",
            description: "Thicken edges by dilating the mask. 0 = native DexiNed width; higher = bolder outlines.",
            tooltip: "Stage 3 of 4 in the lines pipeline. Dilates the thresholded edge mask by N pixels. 0 = native DexiNed width; higher = bolder outlines that stay readable at display resolution. Runs before solid color, so bolder edges still inherit the paint choice.",
        },
        &mut item_settings.edge_thickness,
        0..=20, defaults.template.edge_thickness,
        |v| if v == 0 { "off".into() } else { format!("+{v}px") },
    ), StaticKnob::EdgeThickness, change);

    ui.separator();

    if item_settings.line_mode == LineMode::SubjectOutline {
        let compose_changed = render_compose_mode_chip(ui, &mut item_settings.compose_mode);
        aggregate_bool(compose_changed, StaticKnob::ComposeMode, change);
    }
    let style_changed = render_line_style_chip(ui, &mut item_settings.line_style);
    aggregate_bool(style_changed, StaticKnob::LineStyle, change);
    if render_input_transform_chip(ui, &mut item_settings.input_transform) {
        mark_input_transform_change(change);
    }

    // Edge.rs forces `solid_tint = None` for any non-`Solid` LineStyle, so
    // the chip's value is silently ignored otherwise.
    let solid_tint_active = matches!(item_settings.line_style, prunr_core::LineStyle::Solid);
    let solid_change = chip::guarded(
        ui,
        solid_tint_active,
        "Only takes effect when line style is Solid. Other styles use the source RGB beneath each edge pixel.",
        |ui| chip::chip_option_rgb(
            ui,
            chip::ChipMeta {
                id_salt: "solid_line_color",
                icon: ICON_BRUSH.codepoint,
                label: "Solid line color",
                description: "Paint every edge the same color.",
                tooltip: "Stage 4 of 4 in the lines pipeline. Paint every visible edge the same color, or leave unset to keep the original RGB beneath the mask. Runs after edge thickness.",
            },
            &mut item_settings.solid_line_color,
            defaults.solid_line_color_value,
        ),
    );
    aggregate_knob(solid_change, StaticKnob::SolidLineColor, change);
}

/// Fold a static chip's `ChipChange` into the aggregate, routing via the
/// catalog. Populates `auto_dispatch` only for knobs whose spec marks them
/// `auto_trigger_on_commit` — Model / ChainMode fall through so the user
/// has to click Process.
fn aggregate_knob(ch: chip::ChipChange, knob: StaticKnob, acc: &mut ToolbarChange) {
    if ch.commit {
        acc.commit = true;
    }
    if !ch.changed {
        return;
    }
    let spec = knob_catalog::spec(knob);
    acc.touched.insert(knob);
    acc.cache_impact = acc.cache_impact.union(spec.cache_impact);
    if matches!(spec.dispatch, DispatchKind::Render) {
        acc.render_repaint = true;
    }
    if spec.auto_trigger_on_commit {
        acc.auto_dispatch = acc.auto_dispatch.max(spec.dispatch);
    }
}

/// Shorthand for chips that return a bool (commit-on-change). Builds a
/// `ChipChange` where `changed == commit == b` and folds via the catalog.
fn aggregate_bool(b: bool, knob: StaticKnob, acc: &mut ToolbarChange) {
    aggregate_knob(chip::ChipChange { changed: b, commit: b }, knob, acc);
}

/// Flag a preset application. Dispatch is deferred — the caller checks the
/// actual recipe diff so a no-op preset pick doesn't spawn a subprocess.
fn mark_preset_apply(acc: &mut ToolbarChange) {
    acc.preset_applied = true;
    acc.commit = true;
    acc.render_repaint = true;
}

/// Flag an `InputTransform` change. Dispatch is deferred to the caller's
/// `resolve_auto_dispatch`, which has item state (cached_seg) for the
/// precise AddEdgeInference vs FullPipeline choice. Cache impact is the
/// same in both warm and cold paths (EdgeCache), so we fold it here.
fn mark_input_transform_change(acc: &mut ToolbarChange) {
    acc.input_transform_changed = true;
    acc.commit = true;
    acc.cache_impact = acc.cache_impact.union(CacheImpact::EdgeCache);
}

/// Input-transform picker. Changes invalidate the DexiNed edge tensor cache
/// (and seg cache for conservatism) — not live-previewable. Label accent
/// reminds the user their next Process will re-run inference.
fn render_input_transform_chip(ui: &mut Ui, transform: &mut prunr_core::InputTransform) -> bool {
    use prunr_core::InputTransform;
    let accent = !matches!(transform, InputTransform::None);
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_TUNE.codepoint, transform.name(), accent),
        "Pre-inference transform",
        "Transform applied to the image BEFORE edge detection. Changing this\
         invalidates the edge cache and re-runs DexiNed on the next Process.",
    );

    let popup_id = ui.make_persistent_id("input_transform_popup");
    let mut changed = false;
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.label(RichText::new("Pre-inference transform").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        for option in InputTransform::ALL {
            let selected = std::mem::discriminant(option) == std::mem::discriminant(transform);
            if ui.selectable_label(selected, option.name()).clicked() && !selected {
                *transform = *option;
                changed = true;
            }
        }
        ui.separator();
        match transform {
            InputTransform::None | InputTransform::Grayscale => {
                ui.label(RichText::new("No parameters.").color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_MONO));
            }
            InputTransform::ContrastBoost { percent } => {
                if ui.add(egui::Slider::new(percent, 50..=300).text("Percent")).changed() {
                    changed = true;
                }
            }
            InputTransform::Posterize { levels } => {
                if ui.add(egui::Slider::new(levels, 2..=8).text("Levels")).changed() {
                    changed = true;
                }
            }
        }
    });
    changed
}

/// Line-style picker. Solid defers to the user's `solid_line_color` chip;
/// every other variant carries its own colours / params. Picking a variant
/// keeps the popover open so the user can tune the params below — click
/// outside to dismiss.
fn render_line_style_chip(ui: &mut Ui, style: &mut prunr_core::LineStyle) -> bool {
    use prunr_core::LineStyle;
    let accent = !matches!(style, LineStyle::Solid);
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_GRADIENT.codepoint, style.name(), accent),
        "Line style",
        "How line pixels are coloured.",
    );

    let popup_id = ui.make_persistent_id("line_style_popup");
    let mut changed = false;
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.label(RichText::new("Line style").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        for option in LineStyle::ALL {
            let selected = std::mem::discriminant(option) == std::mem::discriminant(style);
            if ui.selectable_label(selected, option.name()).clicked() && !selected {
                *style = *option;
                changed = true;
            }
        }
        ui.separator();
        if line_style_params(ui, style) {
            changed = true;
        }
    });
    changed
}

fn line_style_params(ui: &mut Ui, style: &mut prunr_core::LineStyle) -> bool {
    use prunr_core::LineStyle;
    let mut changed = false;
    match style {
        LineStyle::Solid => {
            ui.label(RichText::new("Uses Solid line color chip.").color(theme::TEXT_SECONDARY)
                .size(theme::FONT_SIZE_MONO));
        }
        LineStyle::GradientY { top, bottom } => {
            changed |= rgb_picker_row(ui, "Top", top);
            changed |= rgb_picker_row(ui, "Bottom", bottom);
        }
        LineStyle::GradientX { left, right } => {
            changed |= rgb_picker_row(ui, "Left", left);
            changed |= rgb_picker_row(ui, "Right", right);
        }
        LineStyle::RadialGradient { center, inner, outer } => {
            changed |= rgb_picker_row(ui, "Inner", inner);
            changed |= rgb_picker_row(ui, "Outer", outer);
            ui.horizontal(|ui| {
                ui.label("Centre");
                if ui.add(egui::Slider::new(&mut center[0], 0..=255).text("X")).changed() {
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("");
                if ui.add(egui::Slider::new(&mut center[1], 0..=255).text("Y")).changed() {
                    changed = true;
                }
            });
        }
        LineStyle::Rainbow { cycles } => {
            if ui.add(egui::Slider::new(cycles, 1..=10).text("Cycles")).changed() {
                changed = true;
            }
        }
        LineStyle::Chromatic { offset } => {
            if ui.add(egui::Slider::new(offset, 1..=16).text("Offset px")).changed() {
                changed = true;
            }
        }
        LineStyle::Noise { amount } => {
            if ui.add(egui::Slider::new(amount, 0..=255).text("Amount")).changed() {
                changed = true;
            }
        }
        LineStyle::DualScale { fine_color, bold_color } => {
            changed |= rgb_picker_row(ui, "Fine (detail)", fine_color);
            changed |= rgb_picker_row(ui, "Bold (structure)", bold_color);
        }
    }
    changed
}

/// Fill-style picker. Variant list + inline param editor. Pick a variant to
/// switch, tune the params below, click outside to dismiss.
fn render_fill_style_chip(ui: &mut Ui, style: &mut prunr_core::FillStyle) -> bool {
    use prunr_core::FillStyle;
    let accent = !matches!(style, FillStyle::None);
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_FORMAT_PAINT.codepoint, style.name(), accent),
        "Fill style",
        "How the subject RGB is transformed before compose.",
    );

    let popup_id = ui.make_persistent_id("fill_style_popup");
    let mut changed = false;
    chip::popup_for(ui, popup_id, &resp, |ui| {
        // Wider popover so the variant list sits next to the parameter column
        // instead of stacking above it — otherwise 4-stop GradientMap makes
        // the popover taller than most screens.
        ui.set_min_width(FILL_STYLE_POPOVER_WIDTH);
        ui.label(RichText::new("Fill style").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(FILL_STYLE_LIST_WIDTH);
                for option in FillStyle::ALL {
                    let selected = std::mem::discriminant(option) == std::mem::discriminant(style);
                    if ui.selectable_label(selected, option.name()).clicked() && !selected {
                        *style = *option;
                        changed = true;
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                if fill_style_params(ui, style) {
                    changed = true;
                }
            });
        });
    });
    changed
}

const FILL_STYLE_POPOVER_WIDTH: f32 = 560.0;
const FILL_STYLE_LIST_WIDTH: f32 = 140.0;
const GRADIENT_MAP_COL_WIDTH: f32 = 190.0;

fn fill_style_params(ui: &mut Ui, style: &mut prunr_core::FillStyle) -> bool {
    use prunr_core::FillStyle;
    let mut changed = false;
    match style {
        FillStyle::None | FillStyle::Desaturate | FillStyle::Invert | FillStyle::Sepia => {
            ui.label(RichText::new("No parameters.").color(theme::TEXT_SECONDARY)
                .size(theme::FONT_SIZE_MONO));
        }
        FillStyle::Threshold { level } => {
            if ui.add(egui::Slider::new(level, 0..=255).text("Level")).changed() {
                changed = true;
            }
        }
        FillStyle::Posterize { levels } => {
            if ui.add(egui::Slider::new(levels, 2..=8).text("Levels")).changed() {
                changed = true;
            }
        }
        FillStyle::Solarize { pivot } => {
            if ui.add(egui::Slider::new(pivot, 0..=255).text("Pivot")).changed() {
                changed = true;
            }
        }
        FillStyle::HueShift { degrees } => {
            if ui.add(egui::Slider::new(degrees, -180..=180).text("Degrees")).changed() {
                changed = true;
            }
        }
        FillStyle::Saturate { percent } => {
            if ui.add(egui::Slider::new(percent, 0..=300).text("Percent")).changed() {
                changed = true;
            }
        }
        FillStyle::ColorSplash { keep_hue, tolerance } => {
            if ui.add(egui::Slider::new(keep_hue, 0..=359).text("Hue°")).changed() {
                changed = true;
            }
            if ui.add(egui::Slider::new(tolerance, 0..=180).text("Tolerance°")).changed() {
                changed = true;
            }
        }
        FillStyle::Pixelate { block_size } => {
            if ui.add(egui::Slider::new(block_size, 2..=64).text("Block size")).changed() {
                changed = true;
            }
        }
        FillStyle::Duotone { dark, light } => {
            changed |= rgb_picker_row(ui, "Dark", dark);
            changed |= rgb_picker_row(ui, "Light", light);
        }
        FillStyle::CrossProcess { shadow, highlight } => {
            changed |= rgb_picker_row(ui, "Shadow", shadow);
            changed |= rgb_picker_row(ui, "Highlight", highlight);
        }
        FillStyle::ChannelSwap { variant } => {
            use prunr_core::ChannelSwapVariant;
            ui.label(RichText::new("Channel order").color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
            // 5 three-letter codes fit comfortably on one horizontal row —
            // a stacked column wasted the right-hand column of the popover.
            ui.horizontal_wrapped(|ui| {
                for option in ChannelSwapVariant::ALL {
                    let selected = *variant == *option;
                    if ui.selectable_label(selected, option.name()).clicked() && !selected {
                        *variant = *option;
                        changed = true;
                    }
                }
            });
        }
        FillStyle::Halftone { dot_spacing } => {
            if ui.add(egui::Slider::new(dot_spacing, 2..=32).text("Dot spacing px")).changed() {
                changed = true;
            }
        }
        FillStyle::GradientMap { stops } => {
            // 2×2 grid keeps the 4-stop popover within one screen height; a
            // linear stack would exceed the viewport on most displays.
            let [s0, s1, s2, s3] = stops;
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_max_width(GRADIENT_MAP_COL_WIDTH);
                    changed |= rgb_picker_row(ui, "Shadow", s0);
                });
                ui.vertical(|ui| {
                    ui.set_max_width(GRADIENT_MAP_COL_WIDTH);
                    changed |= rgb_picker_row(ui, "Dark mid", s1);
                });
            });
            ui.add_space(theme::SPACE_XS);
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_max_width(GRADIENT_MAP_COL_WIDTH);
                    changed |= rgb_picker_row(ui, "Light mid", s2);
                });
                ui.vertical(|ui| {
                    ui.set_max_width(GRADIENT_MAP_COL_WIDTH);
                    changed |= rgb_picker_row(ui, "Highlight", s3);
                });
            });
        }
    }
    changed
}

/// The five backgrounds a user can choose. Derived from the two
/// orthogonal data fields (`bg`, `bg_effect`) — effects take precedence
/// over solid colour at render time, and the chip mirrors that precedence.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BgKind { Transparent, Solid, Image, BlurredSource, InvertedSource, DesaturatedSource }

impl BgKind {
    const ALL: [Self; 6] = [
        Self::Transparent, Self::Solid, Self::Image,
        Self::BlurredSource, Self::InvertedSource, Self::DesaturatedSource,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Transparent => "Transparent",
            Self::Solid => "Solid colour",
            Self::Image => "Image",
            Self::BlurredSource => "Blurred source",
            Self::InvertedSource => "Inverted source",
            Self::DesaturatedSource => "Desaturated source",
        }
    }

    /// Derive the current kind from the underlying fields. Effects win over
    /// solid colour (render-time precedence); image bg sits between effects
    /// and solid — picking image clears bg_color and effect, picking effect
    /// or color clears the image (mutual exclusion enforced by the chip).
    fn current(bg: &Option<[u8; 4]>, effect: &prunr_core::BgEffect, has_image: bool) -> Self {
        use prunr_core::BgEffect;
        match effect {
            BgEffect::BlurredSource { .. } => Self::BlurredSource,
            BgEffect::InvertedSource => Self::InvertedSource,
            BgEffect::DesaturatedSource => Self::DesaturatedSource,
            BgEffect::None => {
                if has_image { Self::Image }
                else if bg.is_some() { Self::Solid }
                else { Self::Transparent }
            }
        }
    }

    /// Effects require a postprocess rerun (baked into the output RGBA);
    /// Transparent / Solid / Image only change the render-time bg fill.
    fn needs_postprocess(self) -> bool {
        matches!(self, Self::BlurredSource | Self::InvertedSource | Self::DesaturatedSource)
    }
}

/// Unified Background chip: one control for "what fills the transparent
/// area behind the subject" — solid colour or a source-derived effect.
/// Editable + display state for the background chip. Grouped because the
/// chip needs three live-edit fields plus image-availability metadata —
/// past the param-count alarm without this grouping.
struct BgChipState<'a> {
    bg: &'a mut Option<[u8; 4]>,
    bg_effect: &'a mut prunr_core::BgEffect,
    bg_image_fit: &'a mut prunr_core::BgImageFit,
    default_color: [u8; 4],
    has_bg_image: bool,
    bg_image_label: Option<&'a str>,
}

fn render_background_chip(
    ui: &mut Ui,
    state: BgChipState<'_>,
    change: &mut ToolbarChange,
) {
    let BgChipState {
        bg,
        bg_effect,
        bg_image_fit,
        default_color,
        has_bg_image,
        bg_image_label,
    } = state;
    use egui::widgets::color_picker::{color_picker_color32, Alpha};
    use prunr_core::BgEffect;

    let current = BgKind::current(bg, bg_effect, has_bg_image);
    let accent = current != BgKind::Transparent;
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_PALETTE.codepoint, current.name(), accent),
        "Background",
        "What fills transparent areas behind the subject: a solid colour, \
         a chosen image, or a source-derived effect (blurred / inverted / desaturated).",
    );

    let popup_id = ui.make_persistent_id("background_popup");
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.set_min_width(BACKGROUND_POPOVER_WIDTH);
        ui.label(RichText::new("Background").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(BACKGROUND_LIST_WIDTH);
                for kind in BgKind::ALL {
                    let selected = kind == current;
                    if ui.selectable_label(selected, kind.name()).clicked() && !selected {
                        // Image is owned by BatchItem (Arc bytes don't fit in
                        // the Copy ItemSettings) — emit an intent and let the
                        // app handle the file dialog + decode side effects.
                        // We ALSO clear bg + bg_effect here, mirroring the
                        // clear_bg_image branch below: BgKind::current
                        // resolves bg_effect first, so leaving the previous
                        // effect set would keep the result texture's baked-in
                        // effect masking the freshly-loaded image.
                        if matches!(kind, BgKind::Image) {
                            *bg = None;
                            *bg_effect = prunr_core::BgEffect::None;
                            let knob = if current.needs_postprocess() {
                                StaticKnob::BgEffect
                            } else {
                                StaticKnob::BgColor
                            };
                            aggregate_bool(true, knob, change);
                            change.pick_bg_image = true;
                            continue;
                        }
                        apply_bg_kind(bg, bg_effect, kind, default_color);
                        if has_bg_image {
                            // Picking any non-image kind drops the image so
                            // the new choice owns the bg surface alone.
                            change.clear_bg_image = true;
                        }
                        let knob = if kind.needs_postprocess() || current.needs_postprocess() {
                            StaticKnob::BgEffect
                        } else {
                            StaticKnob::BgColor
                        };
                        aggregate_bool(true, knob, change);
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                match current {
                    BgKind::Transparent => {
                        hint(ui, "No background — alpha is preserved on export.");
                    }
                    BgKind::Solid => {
                        if let Some(rgba) = bg.as_mut() {
                            let mut c = egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
                            if color_picker_color32(ui, &mut c, Alpha::OnlyBlend) {
                                let [r, g, b, a] = c.to_srgba_unmultiplied();
                                *rgba = [r, g, b, a];
                                aggregate_bool(true, StaticKnob::BgColor, change);
                            }
                            hint(ui, "Solid colour fills transparent areas at render / export.");
                        }
                    }
                    BgKind::Image => {
                        if let Some(label) = bg_image_label {
                            ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
                            ui.add_space(theme::SPACE_XS);
                        }
                        if ui.button("Choose image\u{2026}").clicked() {
                            change.pick_bg_image = true;
                        }
                        if has_bg_image && ui.button("Remove image").clicked() {
                            change.clear_bg_image = true;
                        }
                        if has_bg_image {
                            ui.add_space(theme::SPACE_XS);
                            ui.label(RichText::new("Fit").color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
                            for fit in prunr_core::BgImageFit::ALL {
                                let selected = *fit == *bg_image_fit;
                                if ui.selectable_label(selected, fit.name()).clicked() && !selected {
                                    *bg_image_fit = *fit;
                                    aggregate_bool(true, StaticKnob::BgImageFit, change);
                                }
                            }
                        }
                        hint(ui, "Picked image fills transparent areas at render / export. Fit follows CSS conventions (Cover / Contain / Stretch / Tile / Center).");
                    }
                    BgKind::BlurredSource => {
                        if let BgEffect::BlurredSource { radius } = bg_effect {
                            let r = ui.add(egui::Slider::new(radius, 1..=64).text("Blur radius"));
                            aggregate_bool(r.changed(), StaticKnob::BgEffect, change);
                        }
                        hint(ui, "Transparent areas filled with a Gaussian-blurred copy of the source image.");
                    }
                    BgKind::InvertedSource => {
                        hint(ui, "Transparent areas filled with the RGB-inverted source image.");
                    }
                    BgKind::DesaturatedSource => {
                        hint(ui, "Transparent areas filled with the luma-grayscale source.");
                    }
                }
            });
        });
    });
}

const BACKGROUND_POPOVER_WIDTH: f32 = 480.0;
const BACKGROUND_LIST_WIDTH: f32 = 150.0;

/// Mutate the two bg fields to match `kind`. Transparent clears both; Solid
/// sets `bg` (picking up an existing colour if present, falling back to the
/// default) and clears the effect; effects set `bg_effect` without touching
/// `bg` so the user's colour survives a round-trip through an effect.
fn apply_bg_kind(
    bg: &mut Option<[u8; 4]>,
    bg_effect: &mut prunr_core::BgEffect,
    kind: BgKind,
    default_color: [u8; 4],
) {
    use prunr_core::BgEffect;
    match kind {
        BgKind::Transparent => {
            *bg = None;
            *bg_effect = BgEffect::None;
        }
        BgKind::Solid => {
            if bg.is_none() { *bg = Some(default_color); }
            *bg_effect = BgEffect::None;
        }
        BgKind::BlurredSource => {
            *bg_effect = BgEffect::BlurredSource { radius: 12 };
        }
        BgKind::InvertedSource => {
            *bg_effect = BgEffect::InvertedSource;
        }
        BgKind::DesaturatedSource => {
            *bg_effect = BgEffect::DesaturatedSource;
        }
        // Image bg lives on BatchItem, not on these two ItemSettings fields —
        // the chip emits `change.pick_bg_image` and skips this fn for Image.
        BgKind::Image => unreachable!("Image kind is handled via change.pick_bg_image, not apply_bg_kind"),
    }
}

/// Label + inline colour picker on two stacked rows. Thin wrapper over
/// `chip::rgb_picker` — the label is purely visual context so the user
/// knows which colour they're editing.
fn rgb_picker_row(ui: &mut Ui, label: &str, rgb: &mut [u8; 3]) -> bool {
    ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    chip::rgb_picker(ui, rgb)
}

/// SubjectOutline compose-mode picker — dropdown chip listing the 5 modes.
/// Returns true when the user changed the mode.
fn render_compose_mode_chip(ui: &mut Ui, mode: &mut prunr_core::ComposeMode) -> bool {
    use prunr_core::ComposeMode;
    let accent = *mode != ComposeMode::default();
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_LAYERS.codepoint, &mode.to_string(), accent),
        "Style",
        "How the subject mask and outline combine.\n\
         • Lines only — outline inside the subject, transparent bg.\n\
         • Subject filled — solid subject with outline on top.\n\
         • Engraving — outline cut through the filled subject.\n\
         • Ghost — faded subject with a strong outline.\n\
         • Inverse mask — outline in the background, subject invisible.",
    );

    let popup_id = ui.make_persistent_id("compose_mode_popup");
    let mut changed = false;
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.label(RichText::new("Style").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        for option in ComposeMode::ALL {
            let selected = *option == *mode;
            if ui.selectable_label(selected, option.to_string()).clicked() {
                if !selected {
                    *mode = *option;
                    changed = true;
                }
                egui::Popup::close_id(ui.ctx(), popup_id);
            }
        }
    });
    changed
}

/// Row 2 leftmost: model dropdown. Edits `app_settings.model` directly and
/// sets `change.model_changed` + `commit` when the selection flips so caller
/// can invalidate tensor caches and fire a fresh Tier 1.
fn render_model_dropdown(
    ui: &mut Ui,
    app_settings: &mut Settings,
    processing: bool,
    mask_active: bool,
    change: &mut ToolbarChange,
) {
    let prev_model = app_settings.model;
    // Always enabled (except mid-processing) so the user can flip to
    // `No model` from any mode — previously gated on mask_active, which
    // made the dropdown unreachable in EdgesOnly.
    let enabled = !processing;
    ui.add_enabled_ui(enabled, |ui| {
        // Match the combobox visuals used by row 1's other dropdowns.
        let vis = ui.visuals_mut();
        vis.widgets.inactive.weak_bg_fill = theme::BG_SECONDARY;
        vis.widgets.inactive.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(0x30, 0x2e, 0x32);
        vis.widgets.hovered.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.open.weak_bg_fill = theme::BG_SECONDARY;
        vis.widgets.open.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.active.fg_stroke.color = theme::TEXT_PRIMARY;
        vis.widgets.noninteractive.fg_stroke.color = theme::TEXT_SECONDARY;

        ui.spacing_mut().interact_size.y = theme::CHIP_HEIGHT;
        // Inpaint doesn't use seg ⇒ mask_active is false, but it's not
        // "bypassed" — it's the Eraser model. Show the model name.
        let selected_text = if mask_active || app_settings.model.is_inpaint() {
            model_label(app_settings.model, true)
        } else {
            format!("{}  Bypassed", ICON_BLOCK.codepoint)
        };
        egui::ComboBox::from_id_salt("adjustments_model")
            .selected_text(
                RichText::new(selected_text)
                    .color(theme::TEXT_PRIMARY),
            )
            .height(420.0)
            .show_ui(ui, |ui| {
                ui.label(RichText::new("Models").strong().color(theme::TEXT_PRIMARY));
                ui.add_space(theme::SPACE_XS);
                ui.separator();
                ui.add_space(theme::SPACE_XS);
                // Filter to installed models only — `None` (filter-only)
                // and Bundled descriptors are always available; OnDemand
                // entries appear only after the user has downloaded them.
                // `None` is pinned last regardless of position in `ALL`.
                let installed: Vec<SettingsModel> = SettingsModel::ALL.iter()
                    .copied()
                    .filter(|v| *v != SettingsModel::None)
                    .filter(|v| v.to_model_id().is_none_or(prunr_models::is_available))
                    .collect();
                for variant in &installed {
                    let model_id = variant.to_model_id();
                    let desc = model_id.and_then(prunr_models::descriptor);
                    let advisory = desc.and_then(|d| d.hardware_advisory(&app_settings.active_backend));
                    let resp = ui.selectable_value(
                        &mut app_settings.model,
                        *variant,
                        RichText::new(model_label(*variant, false))
                            .color(theme::TEXT_PRIMARY),
                    );
                    if let Some(tip) = advisory {
                        resp.on_hover_text(tip);
                    }
                }
                ui.separator();
                ui.selectable_value(
                    &mut app_settings.model,
                    SettingsModel::None,
                    RichText::new(model_label(SettingsModel::None, false))
                        .color(theme::TEXT_PRIMARY),
                );
                ui.separator();
                if ui.button(
                    RichText::new("More models…").color(theme::TEXT_PRIMARY),
                ).clicked() {
                    change.open_model_store = Some(ModelStoreRequest::default());
                }
            })
            .response
            .on_hover_ui(|ui| {
                let (heading, body) = if app_settings.model.is_inpaint() {
                    (
                        "Eraser (LaMa inpaint)",
                        "Object-removal mode. Paint over an unwanted area with the brush; LaMa fills it in. Brush is auto-enabled in this mode.",
                    )
                } else if mask_active {
                    (
                        "Segmentation model",
                        "Which AI model extracts the subject. Trade quality, speed, and memory footprint — per-row labels show each option's position on those three axes.",
                    )
                } else {
                    (
                        "Mask model bypassed",
                        "Sketch is set to Full, so DexiNed runs over the whole image and the subject-extraction model isn't needed. Switch Sketch to Off or Subject to re-enable.",
                    )
                };
                ui.label(RichText::new(heading).strong().color(theme::TEXT_PRIMARY));
                ui.add_space(theme::SPACE_XS);
                ui.label(
                    RichText::new(body)
                        .color(theme::TEXT_PRIMARY)
                        .size(theme::FONT_SIZE_MONO),
                );
            });
    });

    if app_settings.model != prev_model {
        // Clamp parallel jobs to the new model's safe maximum — a correctness
        // invariant on app_settings that mustn't leave an invalid value even
        // if downstream skips persistence.
        let max = app_settings.max_jobs();
        if app_settings.parallel_jobs > max {
            app_settings.parallel_jobs = max;
        }
        change.model_changed = true;
        aggregate_bool(true, StaticKnob::Model, change);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbar_change_default_is_empty() {
        let c = ToolbarChange::default();
        assert_eq!(c.cache_impact, CacheImpact::Nothing);
        assert_eq!(c.auto_dispatch, DispatchKind::None);
        assert!(c.touched.is_empty());
        assert!(!c.render_repaint);
        assert!(c.line_mode_from.is_none());
    }
}
