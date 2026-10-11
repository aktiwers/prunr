//! The adjustments toolbar: `Model`, then one group chip per
//! pipeline stage for the active mode, then the tool cluster and the
//! preset controls. Each group popover lists its knobs in the order the
//! pipeline applies them.
//!
//! View Component discipline: `render` takes `&mut ItemSettings` + a `&AppSettings`
//! reference for defaults / live-preview flag lookups. Never `&mut PrunrApp`.
//!
//! Returns a `ToolbarChange` summarizing WHAT changed so the caller can
//! invalidate the right textures and schedule live-preview reruns.

use egui::{RichText, Ui};
use egui_material_icons::icons::*;

use super::selection_action_bar::{render_selection_actions, SelectionAction};
use super::shortcuts::Action;

use crate::gui::brush_state::BrushState;
use crate::gui::item_settings::{Backdrop, ItemSettings};
use crate::gui::knob_catalog::{
    self, shows, CacheImpact, DispatchKind, KnobContext, KnobRequirement, LineModeChange, StaticKnob, Surface,
};
use crate::gui::settings::{Settings, SettingsModel};
use crate::gui::theme;
use crate::gui::views::{chip, fmt, hint, preset_dropdown};
use prunr_core::{EdgeScale, LineMode};

use super::lines_popover::{compose_description, mode_description, mode_label, scale_description, scale_label};
use super::{differs, installed_models, model_info};

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
    /// A preset was applied; the app records it as an undo step.
    /// The dispatcher resolves subprocess vs skip from the recipe diff.
    pub preset_applied: bool,
    /// Previous `line_mode` when it changed this frame. Drives the
    /// context-sensitive `line_mode_spec(from, current, cached_edge)` path.
    /// `None` means no transition this frame.
    pub line_mode_from: Option<LineMode>,
    /// `input_transform` flipped this frame. Drives `input_transform_spec`
    /// for precise dispatch; `StaticKnob` excludes context-sensitive knobs.
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
    /// User clicked a selection action button (Delete/Copy/Cut/Invert/Clear).
    /// `None` when no action was clicked this frame.
    pub(crate) selection_action: Option<SelectionAction>,
    /// User clicked the Paint Brush toggle button.
    pub(crate) toggle_paint: bool,
    /// User clicked the Magic Brush toggle button.
    pub(crate) toggle_magic: bool,
    /// User clicked the Compare toggle (show the original over the result).
    pub(crate) toggle_compare: bool,
}

/// Read-only per-frame facts the toolbar renders against. Built by the
/// app from the selected item and the coordinators.
pub(crate) struct ToolbarState<'a> {
    pub magic_brush_active: bool,
    pub brush_available: bool,
    pub processing: bool,
    pub has_bg_image: bool,
    pub bg_image_label: Option<&'a str>,
    /// `(w, h)` of the image the next upscale starts from.
    pub source_dims: (u32, u32),
    pub has_selection: bool,
    /// On-disk install state, refreshed on events rather than per frame.
    pub installed: crate::gui::hardware_cache::HardwareInstallCache,
    pub show_original: bool,
    pub can_compare: bool,
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
            line_mode_from: None,
            input_transform_changed: false,
            cache_impact: CacheImpact::Nothing,
            auto_dispatch: DispatchKind::None,
            render_repaint: false,
            brush_settings_committed: false,
            open_model_store: None,
            pick_bg_image: false,
            clear_bg_image: false,
            selection_action: None,
            toggle_paint: false,
            toggle_magic: false,
            toggle_compare: false,
        }
    }
}

// Starting values for knobs whose factory default is "off": the slider
// or picker needs something to show the moment the user switches them on.
const BG_COLOR_FALLBACK: [u8; 4] = [255, 255, 255, 255];
const LINE_COLOR_FALLBACK: [u8; 3] = [0, 0, 0];

/// Render the adjustments toolbar. Returns a `ToolbarChange` summarizing what was edited.
/// `app_settings` exposes model + preset map. `applied_preset` is read for
/// the button's modified/clean icon and written in place when the user
/// applies or saves a preset.
pub(crate) fn render(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    app_settings: &mut Settings,
    mut presets: super::preset_dropdown::PresetUi<'_>,
    brush_state: &mut BrushState,
    state: ToolbarState<'_>,
) -> ToolbarChange {
    let mut change = ToolbarChange::default();
    // A group's Reset returns to the user's default preset, like the
    // toolbar Reset, not to factory values.
    let defaults = app_settings.preset_values(&app_settings.default_preset);

    ui.spacing_mut().item_spacing.x = theme::SPACE_SM;

    // Compared after the render so the line-mode transition is recorded
    // once, whichever row changed it.
    let before_line_mode = item_settings.line_mode;

    let model_uses_seg = app_settings.model.uses_segmentation();
    let knob_ctx = KnobContext {
        model_uses_seg,
        model_is_none: matches!(app_settings.model, SettingsModel::None),
        line_mode: item_settings.line_mode,
    };
    let mask_active = knob_catalog::knob_enabled(KnobRequirement::MaskProduced, knob_ctx);
    let fill_style_active = knob_catalog::knob_enabled(KnobRequirement::SubjectPresent, knob_ctx);
    let bg_active = knob_catalog::knob_enabled(KnobRequirement::TransparencyProduced, knob_ctx);

    ui.horizontal(|ui| {
        render_model_dropdown(ui, app_settings, state.processing, mask_active, state.installed, &mut change);

        let model = app_settings.model;
        if shows(model, Surface::Upscale) {
            super::upscale_toolbar::render_upscale_chips(ui, app_settings, item_settings, state.source_dims, state.installed.x2plus);
        }
        if shows(model, Surface::SdChips) {
            let outcome = super::eraser_chip::render(ui, app_settings, state.installed);
            if outcome.committed {
                change.brush_settings_committed = true;
            }
        }
        if shows(model, Surface::Mask) {
            let mask_reason = (!mask_active).then_some("Sketch is on Whole photo, so nothing is cut out.");
            chip::gated(ui, mask_reason, |ui| {
                render_mask_group(ui, item_settings, &defaults, &mut change);
            });
        }
        if shows(model, Surface::Lines) {
            render_lines_group(ui, item_settings, &defaults, model_uses_seg, &mut change);
        }
        if shows(model, Surface::Effect) {
            // Fill works without a mask (filter-only mode); only Full sketch
            // has no subject to fill.
            let fill_reason = (!fill_style_active).then_some("Sketch is on Whole photo, so there is no subject to restyle.");
            chip::gated(ui, fill_reason, |ui| {
                let changed = render_fill_style_chip(ui, &mut item_settings.fill_style);
                aggregate_bool(changed, StaticKnob::FillStyle, &mut change);
            });
        }
        if shows(model, Surface::Background) {
            let bg_reason = (!bg_active).then_some("Nothing is transparent in this mode, so there is no background to fill.");
            chip::gated(ui, bg_reason, |ui| {
                let backdrop = item_settings.backdrop();
                render_background_chip(ui, BgChipState {
                    backdrop,
                    bg: &mut item_settings.bg,
                    bg_effect: &mut item_settings.bg_effect,
                    bg_image_fit: &mut item_settings.bg_image_fit,
                    default_color: BG_COLOR_FALLBACK,
                    has_bg_image: state.has_bg_image,
                    bg_image_label: state.bg_image_label,
                }, &mut change);
            });
        }

        // Right-aligned cluster. Right-to-left layout fills from the right
        // edge: [Magic][Paint][tool chip][selection actions][Compare] | [Preset][Reset].
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            render_reset_preset_cluster(ui, app_settings, item_settings, &mut presets, &mut change);
            ui.separator();

            chip::gated(ui, (!state.can_compare).then_some("Process the image first."), |ui| {
                let resp = chip::tooltip(
                    chip::icon_toggle_button(ui, ICON_VISIBILITY.codepoint, state.show_original),
                    "Compare",
                    "Show the original image instead of the result.",
                    Some(Action::BeforeAfter),
                );
                if resp.clicked() {
                    change.toggle_compare = true;
                }
            });

            if !shows(model, Surface::Brushes) {
                return;
            }

            if state.has_selection {
                if let Some(action) = render_selection_actions(ui) {
                    change.selection_action = Some(action);
                }
            }

            // Tool cluster, right-to-left: Paint, Magic. Their settings
            // live in the strip over the canvas while a tool is on.
            let paint_active = brush_state.is_enabled();
            chip::gated(ui, (!state.brush_available).then_some("Open an image first."), |ui| {
                let paint_resp = chip::tooltip(
                    chip::icon_toggle_button(ui, ICON_BRUSH.codepoint, paint_active),
                    "Paint Brush",
                    "Paint the selection by hand.",
                    None,
                );
                if paint_resp.clicked() {
                    change.toggle_paint = true;
                }
                let magic_resp = chip::tooltip(
                    chip::icon_toggle_button(ui, ICON_AUTO_AWESOME.codepoint, state.magic_brush_active),
                    "Magic Brush",
                    super::shortcuts::magic_brush_tip(app_settings.model.role()),
                    None,
                );
                if magic_resp.clicked() {
                    change.toggle_magic = true;
                }
            });
        });
    });

    // Model-change hooks: fire regardless of which row is active.
    // Drop any cached LaMa sessions on model change — fires for
    // every direction (LaMa→LaMa-FP32, LaMa→SD, Silueta→LaMa, …)
    // so the previous backend's ~700 MB–2 GB resident set isn't
    // pinned across the user's "I'm done with that tool" signal.
    // Safe because in-flight strokes hold their own Arc — we only
    // drop the cache's ref. Rebuild on next paint is 5–15 s
    // (well below user perception threshold for an explicit
    // model switch).
    if change.model_changed {
        prunr_core::inpaint::release_all_lama_sessions();
    }

    // Inpaint mode: paint is the only input — auto-enable brush so
    // the user doesn't have to click two buttons. Settings stays
    // pinned subtract-equivalent (mode picker is hidden in the
    // popover anyway when Inpaint is active). Pre-warm the LaMa /
    // MI-GAN session in the background so the first stroke doesn't
    // pay the 5-10 s zstd-decompress + ORT-session-build cost.
    //
    // SD-family models are NOT prewarmed here. They run in the
    // dedicated `inpaint_only` subprocess via `inpaint_bridge`, so
    // building an `SdSession` in this (GUI) process loads ~9 GB
    // into RAM that no dispatch path will ever read — the GUI
    // never calls `process_inpaint_with` for SD models. The
    // observable bug: GUI process bloats to 15 GB on model
    // switch, then the subprocess loads its own 9 GB on first
    // stroke → ~24 GB peak → memory-pressure abort on any
    // 16 GB box.
    if change.model_changed && app_settings.model.is_inpaint() {
        if !brush_state.is_enabled() {
            brush_state.toggle();
        }
        let raw_id = app_settings.model.to_model_id();
        let in_process = matches!(
            raw_id,
            Some(prunr_models::ModelId::LaMaFp32)
                | Some(prunr_models::ModelId::BigLaMa)
                | Some(prunr_models::ModelId::Migan)
        );
        if in_process {
            if let Some(id) = raw_id {
                rayon::spawn(move || {
                    if let Err(e) = prunr_core::inpaint::prewarm(id) {
                        tracing::warn!(?id, %e, "Inpaint prewarm failed");
                    }
                });
            }
        }
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

const LINES_POPOVER_WIDTH: f32 = 560.0;
const LINES_LEFT_COLUMN_WIDTH: f32 = 260.0;
const LINES_COLOR_LIST_WIDTH: f32 = 170.0;

/// Every mask knob, in processing order.
const MASK_FIELDS: &[chip::Field<ItemSettings>] = &[
    chip::Field { knob: Some(StaticKnob::Gamma), differs: |s, d| differs(s.gamma, d.gamma), reset: |s, d| s.gamma = d.gamma },
    chip::Field { knob: Some(StaticKnob::EdgeShift), differs: |s, d| differs(s.edge_shift, d.edge_shift), reset: |s, d| s.edge_shift = d.edge_shift },
    chip::Field { knob: Some(StaticKnob::RefineEdges), differs: |s, d| s.refine_edges != d.refine_edges, reset: |s, d| s.refine_edges = d.refine_edges },
    chip::Field { knob: Some(StaticKnob::GuidedRadius), differs: |s, d| s.refine_edges && s.guided_radius != d.guided_radius, reset: |s, d| s.guided_radius = d.guided_radius },
    chip::Field { knob: Some(StaticKnob::GuidedEpsilon), differs: |s, d| s.refine_edges && differs(s.guided_epsilon, d.guided_epsilon), reset: |s, d| s.guided_epsilon = d.guided_epsilon },
    chip::Field { knob: Some(StaticKnob::Feather), differs: |s, d| differs(s.feather, d.feather), reset: |s, d| s.feather = d.feather },
    chip::Field { knob: Some(StaticKnob::Contrast), differs: |s, d| differs(s.contrast, d.contrast), reset: |s, d| s.contrast = d.contrast },
];

/// Every line knob except the mode, which the chip face shows instead.
const LINES_FIELDS: &[chip::Field<ItemSettings>] = &[
    chip::Field { knob: Some(StaticKnob::EdgeScale), differs: |s, d| s.edge_scale != d.edge_scale, reset: |s, d| s.edge_scale = d.edge_scale },
    chip::Field { knob: Some(StaticKnob::LineStrength), differs: |s, d| differs(s.line_strength, d.line_strength), reset: |s, d| s.line_strength = d.line_strength },
    chip::Field { knob: Some(StaticKnob::EdgeThickness), differs: |s, d| s.edge_thickness != d.edge_thickness, reset: |s, d| s.edge_thickness = d.edge_thickness },
    chip::Field { knob: Some(StaticKnob::ComposeMode), differs: |s, d| s.compose_mode != d.compose_mode, reset: |s, d| s.compose_mode = d.compose_mode },
    chip::Field { knob: Some(StaticKnob::LineStyle), differs: |s, d| s.line_style != d.line_style, reset: |s, d| s.line_style = d.line_style },
    chip::Field { knob: Some(StaticKnob::SolidLineColor), differs: |s, d| s.solid_line_color != d.solid_line_color, reset: |s, d| s.solid_line_color = d.solid_line_color },
    chip::Field { knob: None, differs: |s, d| s.input_transform != d.input_transform, reset: |s, d| s.input_transform = d.input_transform },
];

/// Put a group back to its defaults and dispatch every knob that changed.
fn reset_group(fields: &[chip::Field<ItemSettings>], s: &mut ItemSettings, d: &ItemSettings, change: &mut ToolbarChange) {
    chip::reset_fields(fields, s, d);
    for knob in fields.iter().filter_map(|f| f.knob) {
        aggregate_bool(true, knob, change);
    }
}

/// The five mask stages, in the order the pipeline applies them.
fn render_mask_group(
    ui: &mut Ui,
    s: &mut ItemSettings,
    d: &ItemSettings,
    change: &mut ToolbarChange,
) {
    let tuned = chip::tuned_count(MASK_FIELDS, s, d);
    let summary = chip::tuned_summary(tuned);
    let group = chip::GroupChip {
        id_salt: "mask",
        icon: ICON_TONALITY.codepoint,
        label: "Mask",
        summary: &summary,
        tooltip: "How the subject is cut out, in five steps. Each step works on the result of the one above it.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: true,
    };
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            reset_group(MASK_FIELDS, s, d, change);
        }

        aggregate_knob(
            chip::slider_row_f32(ui, "Gamma", &mut s.gamma, 0.01..=10.0, true, |v| fmt::plain(v, 2)),
            StaticKnob::Gamma, change,
        );
        hint(ui, "How hard the mask cuts. Above 1 removes more; below 1 keeps more of the edge.");
        ui.add_space(theme::SPACE_XS);


        let mut grow = prunr_core::MaskSettings::edge_shift_for_grow(s.edge_shift);
        let shift = chip::slider_row_f32(ui, "Shift edge", &mut grow, -20.0..=20.0, false, |v| fmt::signed_px(v, 1));
        if shift.changed {
            s.edge_shift = prunr_core::MaskSettings::edge_shift_for_grow(grow);
        }
        aggregate_knob(shift, StaticKnob::EdgeShift, change);
        hint(ui, "Grow (+) or shrink (\u{2212}) the cut-out's edge.");
        ui.add_space(theme::SPACE_XS);

        aggregate_knob(chip::toggle_row(ui, "Refine edges", &mut s.refine_edges), StaticKnob::RefineEdges, change);
        hint(ui, "Snap the mask to color edges in the photo, for hair and leaves. Slower.");
        if s.refine_edges {
            let mut radius = s.guided_radius as u32;
            let r = chip::slider_row(ui, "Edge width (px)", &mut radius, 1..=64);
            s.guided_radius = radius.min(255) as u8;
            aggregate_knob(r, StaticKnob::GuidedRadius, change);
            hint(ui, "How far from the edge refinement looks for hair and detail.");
            egui::CollapsingHeader::new("Advanced").id_salt("refine_advanced").show(ui, |ui| {
                let mut detail = detail_from_epsilon(s.guided_epsilon);
                let d = chip::slider_row_f32(ui, "Detail", &mut detail, 0.0..=1.0, false, fmt::percent);
                if d.changed {
                    s.guided_epsilon = epsilon_from_detail(detail);
                }
                aggregate_knob(d, StaticKnob::GuidedEpsilon, change);
                hint(ui, "Higher follows finer color edges.");
            });
        }
        ui.add_space(theme::SPACE_XS);

        aggregate_knob(
            chip::slider_row_f32(ui, "Feather", &mut s.feather, 0.0..=10.0, false, |v| fmt::off_or(v, 0.05, |v| fmt::px(v, 1))),
            StaticKnob::Feather, change,
        );
        hint(ui, "Soften the mask edge.");
        ui.add_space(theme::SPACE_XS);

        aggregate_knob(
            chip::slider_row_f32(ui, "Contrast", &mut s.contrast, 0.0..=1.0, false, fmt::percent),
            StaticKnob::Contrast, change,
        );
        hint(ui, "Sharpen the edge of the cut-out. 100% is a hard edge.");
        ui.add_space(theme::SPACE_SM);

        hint(ui, "Press F3 for the pipeline diagram.");
    });
}

/// Lines take the photo's own colors: the "Original" line color.
fn lines_in_photo_colors(s: &ItemSettings) -> bool {
    s.line_style == prunr_core::LineStyle::Solid && s.solid_line_color.is_none()
}

const fn step<T>(value: T, name: &'static str) -> chip::Choice<T> {
    chip::Choice { value, name, description: "", enabled: true }
}

const CONTRAST_STEPS: [chip::Choice<u16>; 4] = [step(125, "125%"), step(150, "150%"), step(200, "200%"), step(300, "300%")];
const LEVEL_STEPS: [chip::Choice<u8>; 5] = [step(2, "2"), step(3, "3"), step(4, "4"), step(6, "6"), step(8, "8")];

/// What a line color style looks like, for its row in the list.
fn line_style_description(style: &prunr_core::LineStyle) -> &'static str {
    use prunr_core::LineStyle;
    match style {
        LineStyle::Solid => "one color for every line",
        LineStyle::GradientY { .. } => "one color at the top fading to another at the bottom",
        LineStyle::GradientX { .. } => "one color on the left fading to another on the right",
        LineStyle::RadialGradient { .. } => "one color at a center point fading outward",
        LineStyle::Rainbow { .. } => "colors cycling across the image",
        LineStyle::DualScale { .. } => "fine detail and bold shapes, each in its own color",
    }
}

/// Refine's precision (the guided filter's epsilon, 1e-2 to 1e-6 on a log
/// scale) as a 0-1 Detail amount: more detail is a smaller epsilon.
fn detail_from_epsilon(epsilon: f32) -> f32 {
    ((-epsilon.max(1e-6).log10() - 2.0) / 4.0).clamp(0.0, 1.0)
}

fn epsilon_from_detail(detail: f32) -> f32 {
    10f32.powf(-2.0 - 4.0 * detail.clamp(0.0, 1.0))
}

/// Sketch mode plus every line knob, in processing order; the line color
/// list sits in its own column.
fn render_lines_group(
    ui: &mut Ui,
    s: &mut ItemSettings,
    d: &ItemSettings,
    subject_available: bool,
    change: &mut ToolbarChange,
) {
    use prunr_core::{ComposeMode, InputTransform, LineStyle};
    let on = s.line_mode != LineMode::Off;
    let tuned = on && chip::tuned_count(LINES_FIELDS, s, d) > 0;
    let group = chip::GroupChip {
        id_salt: "lines",
        icon: ICON_DRAW.codepoint,
        label: "Sketch",
        summary: mode_label(s.line_mode),
        tooltip: "Trace the outlines of the image or of the subject, then style the lines.",
        tuned,
        width: if on { LINES_POPOVER_WIDTH } else { theme::POPOVER_WIDTH },
        live: true,
    };
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            // The mode transition is recorded by the caller after the render.
            s.line_mode = d.line_mode;
            reset_group(LINES_FIELDS, s, d, change);
            mark_input_transform_change(change);
        }

        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(LINES_LEFT_COLUMN_WIDTH);
                ui.set_max_width(LINES_LEFT_COLUMN_WIDTH);

                let modes = [LineMode::Off, LineMode::SubjectOutline, LineMode::EdgesOnly].map(|m| {
                    let enabled = m != LineMode::SubjectOutline || subject_available;
                    chip::Choice {
                        value: m,
                        name: mode_label(m),
                        description: if enabled { mode_description(m) } else { "Needs a background-removal model to find the subject" },
                        enabled,
                    }
                });
                // The caller records the mode transition after the render.
                chip::choice_row(ui, "Sketch", &modes, &mut s.line_mode);
                if !on {
                    hint(ui, "Off: no lines are drawn.");
                    return;
                }
                ui.add_space(theme::SPACE_XS);
                chip::section_label(ui, "Pre-filter");
                let mut filter_changed = false;
                ui.horizontal_wrapped(|ui| {
                    for option in InputTransform::ALL {
                        let selected = std::mem::discriminant(option) == std::mem::discriminant(&s.input_transform);
                        if ui.selectable_label(selected, option.name()).clicked() && !selected {
                            s.input_transform = *option;
                            filter_changed = true;
                        }
                    }
                });
                // Steps, not sliders: each pick reruns the line detector once.
                match &mut s.input_transform {
                    InputTransform::None | InputTransform::Grayscale | InputTransform::Smooth => {}
                    InputTransform::ContrastBoost { percent } => {
                        filter_changed |= chip::choice_row(ui, "Contrast", &CONTRAST_STEPS, percent);
                    }
                    InputTransform::Posterize { levels } => {
                        filter_changed |= chip::choice_row(ui, "Levels", &LEVEL_STEPS, levels);
                    }
                }
                if filter_changed {
                    mark_input_transform_change(change);
                }
                hint(ui, "Applied to the image before the lines are traced. Requires reprocessing.");
                ui.add_space(theme::SPACE_XS);

                // Dual scale draws Fine and Bold itself, so the scale has no effect.
                let scale_active = !matches!(s.line_style, LineStyle::DualScale { .. });
                let scales = [EdgeScale::Fine, EdgeScale::Balanced, EdgeScale::Bold, EdgeScale::Fused].map(|sc| chip::Choice {
                    value: sc,
                    name: scale_label(sc),
                    description: scale_description(sc),
                    enabled: scale_active,
                });
                let scale_changed = chip::choice_row(ui, "Scale", &scales, &mut s.edge_scale);
                aggregate_bool(scale_changed, StaticKnob::EdgeScale, change);
                if !scale_active {
                    hint(ui, "Dual scale draws Fine and Bold itself.");
                }
                ui.add_space(theme::SPACE_XS);

                aggregate_knob(
                    chip::slider_row_f32(ui, "Line detail", &mut s.line_strength, 0.0..=1.0, false, |v| fmt::plain(v, 2)),
                    StaticKnob::LineStrength, change,
                );
                hint(ui, "Lower keeps bold outlines only; higher adds fine texture.");
                ui.add_space(theme::SPACE_XS);

                let mut thickness = s.edge_thickness as u32;
                let t = chip::slider_row(ui, "Line width (px)", &mut thickness, 0..=20);
                s.edge_thickness = thickness.min(255) as u8;
                aggregate_knob(t, StaticKnob::EdgeThickness, change);
                ui.add_space(theme::SPACE_XS);

                if s.line_mode == LineMode::SubjectOutline {
                    let compositions = [
                        ComposeMode::LinesOnly, ComposeMode::SubjectFilled, ComposeMode::Engraving,
                        ComposeMode::Ghost,
                    ].map(|c| chip::Choice { value: c, name: compose_label(c), description: compose_description(c), enabled: true });
                    let changed = chip::choice_row(ui, "Composition", &compositions, &mut s.compose_mode);
                    aggregate_bool(changed, StaticKnob::ComposeMode, change);
                    // Lines in the photo's own colors vanish on a filled subject.
                    if changed && s.compose_mode == ComposeMode::SubjectFilled && lines_in_photo_colors(s) {
                        s.solid_line_color = Some([0, 0, 0]);
                        aggregate_bool(true, StaticKnob::SolidLineColor, change);
                    }
                    ui.add_space(theme::SPACE_XS);
                }

            });

            if !on {
                return;
            }
            ui.separator();
            ui.vertical(|ui| {
                ui.set_min_width(LINES_COLOR_LIST_WIDTH);
                chip::section_label(ui, "Line color");
                let is_solid = matches!(s.line_style, LineStyle::Solid);
                if chip::picker_row(ui, lines_in_photo_colors(s), "Original", "the photo's own colors").clicked() {
                    s.line_style = LineStyle::Solid;
                    s.solid_line_color = None;
                    aggregate_bool(true, StaticKnob::LineStyle, change);
                    aggregate_bool(true, StaticKnob::SolidLineColor, change);
                }
                if chip::picker_row(ui, is_solid && s.solid_line_color.is_some(), "Solid color", "one color for every line").clicked() {
                    s.line_style = LineStyle::Solid;
                    s.solid_line_color = Some(LINE_COLOR_FALLBACK);
                    aggregate_bool(true, StaticKnob::LineStyle, change);
                    aggregate_bool(true, StaticKnob::SolidLineColor, change);
                }
                for option in &LineStyle::ALL[1..] {
                    let selected = std::mem::discriminant(option) == std::mem::discriminant(&s.line_style);
                    if chip::picker_row(ui, selected, option.name(), line_style_description(option)).clicked() && !selected {
                        s.line_style = *option;
                        aggregate_bool(true, StaticKnob::LineStyle, change);
                    }
                }
                ui.add_space(theme::SPACE_XS);
                if is_solid {
                    if let Some(rgb) = s.solid_line_color.as_mut() {
                        let changed = rgb_picker_row(ui, "Color", rgb);
                        aggregate_bool(changed, StaticKnob::SolidLineColor, change);
                    }
                } else {
                    let changed = line_style_params(ui, &mut s.line_style);
                    aggregate_bool(changed, StaticKnob::LineStyle, change);
                }
            });
        });
    });
}

fn compose_label(mode: prunr_core::ComposeMode) -> &'static str {
    use prunr_core::ComposeMode;
    match mode {
        ComposeMode::LinesOnly => "Lines only",
        ComposeMode::SubjectFilled => "Subject filled",
        ComposeMode::Engraving => "Engraving",
        ComposeMode::Ghost => "Ghost",
    }
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

fn line_style_params(ui: &mut Ui, style: &mut prunr_core::LineStyle) -> bool {
    use prunr_core::LineStyle;
    let mut changed = false;
    match style {
        LineStyle::Solid => {}
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
            changed |= chip::slider_row(ui, "Center X", &mut center[0], 0..=255).changed;
            changed |= chip::slider_row(ui, "Center Y", &mut center[1], 0..=255).changed;
        }
        LineStyle::Rainbow { cycles } => {
            changed |= chip::slider_row(ui, "Cycles", cycles, 1..=10).changed;
        }
        LineStyle::DualScale { fine_color, bold_color } => {
            changed |= rgb_picker_row(ui, "Fine (detail)", fine_color);
            changed |= rgb_picker_row(ui, "Bold (structure)", bold_color);
        }
    }
    changed
}

/// Effect picker ("Fill style" in settings and code). Variant list + inline
/// param editor; "None" heads the list, so the header has no reset.
fn render_fill_style_chip(ui: &mut Ui, style: &mut prunr_core::FillStyle) -> bool {
    use prunr_core::FillStyle;
    let accent = !matches!(style, FillStyle::None);
    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_FORMAT_PAINT.codepoint, style.name(), accent),
        "Effect",
        "Restyle the subject's colors, or the whole photo with No model.",
        None,
    );

    let popup_id = ui.make_persistent_id("fill_style_popup");
    let mut changed = false;
    chip::flyout_for(popup_id, &resp, |ui| {
        // Wider popover so the variant list sits next to the parameter column
        // instead of stacking above it — otherwise 4-stop GradientMap makes
        // the popover taller than most screens.
        ui.set_min_width(FILL_STYLE_POPOVER_WIDTH);
        chip::popover_header(ui, "Effect", None);
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
            changed |= chip::slider_row(ui, "Level", level, 0..=255).changed;
        }
        FillStyle::Posterize { levels } => {
            changed |= chip::slider_row(ui, "Levels", levels, 2..=8).changed;
        }
        FillStyle::Solarize { pivot } => {
            changed |= chip::slider_row(ui, "Pivot", pivot, 0..=255).changed;
        }
        FillStyle::HueShift { degrees } => {
            changed |= chip::slider_row(ui, "Degrees", degrees, -180..=180).changed;
        }
        FillStyle::Saturate { percent } => {
            changed |= chip::slider_row(ui, "Percent", percent, 0..=300).changed;
        }
        FillStyle::ColorSplash { keep_hue, tolerance } => {
            changed |= chip::slider_row(ui, "Hue (degrees)", keep_hue, 0..=359).changed;
            changed |= chip::slider_row(ui, "Tolerance (degrees)", tolerance, 0..=180).changed;
        }
        FillStyle::Pixelate { block_size } => {
            changed |= chip::slider_row(ui, "Block size (px)", block_size, 2..=64).changed;
        }
        FillStyle::Duotone { dark, light } => {
            changed |= rgb_picker_row(ui, "Dark", dark);
            changed |= rgb_picker_row(ui, "Light", light);
        }
        FillStyle::CrossProcess { shadow, highlight } => {
            changed |= rgb_picker_row(ui, "Shadow", shadow);
            changed |= rgb_picker_row(ui, "Highlight", highlight);
        }
        FillStyle::Halftone { dot_spacing } => {
            changed |= chip::slider_row(ui, "Dot spacing (px)", dot_spacing, 2..=32).changed;
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
/// over solid color at render time, and the chip mirrors that precedence.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BgKind { Transparent, Solid, Image, BlurredSource, DesaturatedSource }

impl BgKind {
    const ALL: [Self; 5] = [
        Self::Transparent, Self::Solid, Self::Image,
        Self::BlurredSource, Self::DesaturatedSource,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Transparent => "Transparent",
            Self::Solid => "Solid color",
            Self::Image => "Image",
            Self::BlurredSource => "Blurred photo",
            Self::DesaturatedSource => "Black & white photo",
        }
    }

    fn of(backdrop: Backdrop) -> Self {
        use prunr_core::BgEffect;
        match backdrop {
            Backdrop::Transparent => Self::Transparent,
            Backdrop::Color(_) => Self::Solid,
            Backdrop::Image(_) => Self::Image,
            Backdrop::Effect(BgEffect::BlurredSource { .. }) => Self::BlurredSource,
            Backdrop::Effect(BgEffect::DesaturatedSource) => Self::DesaturatedSource,
            Backdrop::Effect(BgEffect::None) => Self::Transparent,
        }
    }
}

/// Unified Background chip: one control for "what fills the transparent
/// area behind the subject" — solid color or a source-derived effect.
/// Editable + display state for the background chip. Grouped because the
/// chip needs three live-edit fields plus image-availability metadata —
/// past the param-count alarm without this grouping.
struct BgChipState<'a> {
    backdrop: Backdrop,
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
        backdrop,
        bg,
        bg_effect,
        bg_image_fit,
        default_color,
        has_bg_image,
        bg_image_label,
    } = state;
    use prunr_core::BgEffect;

    let current = BgKind::of(backdrop);
    let accent = current != BgKind::Transparent;
    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_PALETTE.codepoint, current.name(), accent),
        "Background",
        "What fills the transparent area behind the subject: a solid color, an image, or a blurred or black-and-white copy of the photo.",
    None,
);

    let popup_id = ui.make_persistent_id("background_popup");
    chip::flyout_for(popup_id, &resp, |ui| {
        ui.set_min_width(BACKGROUND_POPOVER_WIDTH);
        chip::popover_header(ui, "Background", None);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_min_width(BACKGROUND_LIST_WIDTH);
                for kind in BgKind::ALL {
                    let selected = kind == current;
                    if ui.selectable_label(selected, kind.name()).clicked() && !selected {
                        // The picture lives on the item and loads off the UI
                        // thread; it replaces the background when it lands.
                        if matches!(kind, BgKind::Image) {
                            change.pick_bg_image = true;
                            continue;
                        }
                        apply_bg_kind(bg, bg_effect, kind, default_color);
                        if has_bg_image {
                            // Picking any non-image kind drops the image so
                            // the new choice owns the bg surface alone.
                            change.clear_bg_image = true;
                        }
                        aggregate_bool(true, StaticKnob::BgColor, change);
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                match current {
                    BgKind::Transparent => {
                        hint(ui, "No background; the exported PNG keeps its transparency.");
                    }
                    BgKind::Solid => {
                        if let Some(rgba) = bg.as_mut() {
                            let mut rgb = [rgba[0], rgba[1], rgba[2]];
                            if chip::rgb_picker(ui, &mut rgb) {
                                *rgba = [rgb[0], rgb[1], rgb[2], 255];
                                aggregate_bool(true, StaticKnob::BgColor, change);
                            }
                            hint(ui, "One color fills the transparent area on screen and in the export.");
                        }
                    }
                    BgKind::Image => {
                        if let Some(label) = bg_image_label {
                            ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
                            ui.add_space(theme::SPACE_XS);
                        }
                        if chip::button(ui, chip::ButtonKind::Secondary, "Choose image\u{2026}").clicked() {
                            change.pick_bg_image = true;
                        }
                        if has_bg_image
                            && chip::button(ui, chip::ButtonKind::Secondary, "Remove image").clicked()
                        {
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
                        hint(ui, "The image fills the transparent area on screen and in the export.");
                    }
                    BgKind::BlurredSource => {
                        if let BgEffect::BlurredSource { radius } = bg_effect {
                            let r = chip::slider_row(ui, "Blur radius (px)", radius, 1..=64);
                            aggregate_bool(r.changed, StaticKnob::BgEffect, change);
                        }
                        hint(ui, "A blurred copy of the photo fills the transparent area.");
                    }
                    BgKind::DesaturatedSource => {
                        hint(ui, "A black-and-white copy of the photo fills the transparent area.");
                    }
                }
            });
        });
    });
}

const BACKGROUND_POPOVER_WIDTH: f32 = 480.0;
const BACKGROUND_LIST_WIDTH: f32 = 150.0;

/// Set the two bg fields to `kind` alone: the background is one choice.
/// Solid keeps a colour picked earlier, or starts from the default.
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
            *bg = None;
            *bg_effect = BgEffect::BlurredSource { radius: 12 };
        }
        BgKind::DesaturatedSource => {
            *bg = None;
            *bg_effect = BgEffect::DesaturatedSource;
        }
        // Image bg lives on BatchItem, not on these two ItemSettings fields —
        // the chip emits `change.pick_bg_image` and skips this fn for Image.
        BgKind::Image => unreachable!("Image kind is handled via change.pick_bg_image, not apply_bg_kind"),
    }
}

/// Label + inline color picker on two stacked rows. Thin wrapper over
/// `chip::rgb_picker` — the label is purely visual context so the user
/// knows which color they're editing.
fn rgb_picker_row(ui: &mut Ui, label: &str, rgb: &mut [u8; 3]) -> bool {
    ui.label(RichText::new(label).color(theme::TEXT_SECONDARY).size(theme::FONT_SIZE_MONO));
    chip::rgb_picker(ui, rgb)
}

/// Reset-to-default-preset button + preset dropdown.
pub(super) fn render_reset_preset_cluster(
    ui: &mut Ui,
    app_settings: &mut Settings,
    item_settings: &mut ItemSettings,
    presets: &mut super::preset_dropdown::PresetUi<'_>,
    change: &mut ToolbarChange,
) {
    let reset_resp = chip::tooltip(
        chip::icon_action_button(ui, ICON_RESTART_ALT.codepoint, theme::TEXT_PRIMARY),
        "Reset",
        "Return every knob to your default preset.",
        None,
    );
    if reset_resp.clicked() {
        let reset_target = app_settings.default_preset.clone();
        let resolved = app_settings.resolve_active_preset(None);
        *item_settings = resolved.item_settings;
        app_settings.brush = resolved.brush;
        *presets.applied = reset_target;
        change.brush_settings_committed = true;
        mark_preset_apply(change);
    }

    if let Some(name) = preset_dropdown::render(ui, app_settings, item_settings, presets) {
        *presets.applied = name;
        change.brush_settings_committed = true;
        mark_preset_apply(change);
    }
}

/// Leftmost on the toolbar: the model picker. Edits `app_settings.model` directly and
/// sets `change.model_changed` + `commit` when the selection flips so caller
/// can invalidate tensor caches and fire a fresh Tier 1.
pub(super) fn render_model_dropdown(
    ui: &mut Ui,
    app_settings: &mut Settings,
    processing: bool,
    mask_active: bool,
    installed: crate::gui::hardware_cache::HardwareInstallCache,
    change: &mut ToolbarChange,
) {
    let prev_model = app_settings.model;
    // Always enabled (except mid-processing) so the user can flip to
    // `No model` from any mode — previously gated on mask_active, which
    // made the dropdown unreachable in EdgesOnly.
    let enabled = !processing;
    ui.add_enabled_ui(enabled, |ui| {
        let info = model_info(app_settings.model);
        let resp = chip::chip_button(ui, info.icon, info.name, false);
        // The chip is always named "Model"; the hint says what this one does.
        use crate::gui::knob_catalog::ModelRole;
        let body = match app_settings.model.role() {
            ModelRole::Cutout if !mask_active => "Sketch is on Whole photo, so the lines cover the whole photo and nothing is cut out. Set Sketch to Off or Subject to cut out the subject again.",
            ModelRole::Cutout => "Cuts the subject out of the photo. Each model in the list shows its strength and download size.",
            ModelRole::Erase => "Removes what you paint over and fills the area in. Paint with the brush, then Process.",
            ModelRole::Upscale => "Enlarges the photo. Each model in the list shows what it is best at and its download size.",
            ModelRole::Filters => "No model runs: only the effect, lines and background apply to the photo.",
        };
        let resp = chip::tooltip(resp, "Model", body, None);
        let pop_id = egui::Id::new("adjustments_model_popup");
        chip::popup_for(ui, pop_id, &resp, |ui| {
            chip::popover_header(ui, "Model", None);
            for variant in installed_models(installed) {
                if variant == SettingsModel::None {
                    ui.separator();
                }
                let advisory = variant
                    .to_model_id()
                    .and_then(prunr_models::descriptor)
                    .and_then(|d| d.hardware_advisory(&app_settings.active_backend));
                let info = model_info(variant);
                let row = chip::picker_row(ui, app_settings.model == variant, info.name, info.blurb);
                let row = match advisory {
                    Some(tip) => row.on_hover_text(tip),
                    None => row,
                };
                if row.clicked() {
                    app_settings.model = variant;
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
            }
            ui.separator();
            if chip::button(ui, chip::ButtonKind::Secondary, "More models…").clicked() {
                change.open_model_store = Some(ModelStoreRequest::default());
                egui::Popup::close_id(ui.ctx(), pop_id);
            }
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


    /// Detail maps the guided filter's epsilon onto 0-100 % and back.
    #[test]
    fn refine_detail_round_trips_through_epsilon() {
        assert!((detail_from_epsilon(1e-2) - 0.0).abs() < 1e-4);
        assert!((detail_from_epsilon(1e-6) - 1.0).abs() < 1e-4);
        assert!((detail_from_epsilon(1e-4) - 0.5).abs() < 1e-4, "the default sits mid-way");
        for d in [0.0, 0.25, 0.5, 0.9, 1.0] {
            assert!((detail_from_epsilon(epsilon_from_detail(d)) - d).abs() < 1e-4);
        }
    }

    #[test]
    fn toolbar_change_default_is_empty() {
        let c = ToolbarChange::default();
        assert_eq!(c.cache_impact, CacheImpact::Nothing);
        assert_eq!(c.auto_dispatch, DispatchKind::None);
        assert!(!c.render_repaint);
        assert!(c.line_mode_from.is_none());
    }
}
