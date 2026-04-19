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

use crate::gui::item_settings::ItemSettings;
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
#[derive(Default, Debug, Clone, Copy)]
pub struct ToolbarChange {
    pub mask: bool,
    pub edge: bool,
    pub bg: bool,
    pub commit: bool,
    pub model_changed: bool,
    pub preset_applied: bool,
    /// Segmentation tensor is stale (produced by a now-swapped model).
    pub seg_cache_invalid: bool,
    /// DexiNed edge tensor is stale (different line_mode means DexiNed
    /// would see a different input — full scene vs subject-on-white).
    pub edge_cache_invalid: bool,
    /// Line mode (Off / EdgesOnly / SubjectOutline) was toggled. The caller
    /// auto-triggers a reprocess on a Done item so the user doesn't have to
    /// click Process — tier routing keeps this cheap (AddEdgeInference when
    /// the seg tensor is still valid).
    pub line_mode_changed: bool,
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
pub fn render(
    ui: &mut Ui,
    item_settings: &mut ItemSettings,
    app_settings: &mut Settings,
    applied_preset: &mut String,
    processing: bool,
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

    let aggregate = |ch: chip::ChipChange, tier: Tier, acc: &mut ToolbarChange| {
        if ch.changed {
            match tier {
                Tier::Mask => acc.mask = true,
                Tier::Edge => acc.edge = true,
            }
        }
        if ch.commit { acc.commit = true; }
    };

    // EdgesOnly w/o chain bypasses the mask tier — don't let the user
    // tweak dead knobs. Chain mode re-runs mask on a later pass.
    let mask_active = !(item_settings.line_mode == LineMode::EdgesOnly
        && !app_settings.chain_mode);

    ui.horizontal(|ui| {
        render_model_dropdown(ui, app_settings, processing, mask_active, &mut change);

        ui.add_enabled_ui(mask_active, |ui| {
            aggregate(chip::chip_f32(
                ui, "gamma", "γ", "Gamma",
                "How hard the mask cuts. >1 is more aggressive; <1 is gentler.",
                tip!("Stage 1 of 5. How hard the mask cuts. >1 removes more aggressively, <1 is gentler on fine edges. Feeds every stage below."),
                &mut item_settings.gamma,
                0.01..=10.0, defaults.template.gamma,
                true, // log scale — matches perceptual symmetry around 1.0
                |v| format!("{v:.2}"),
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_option_f32(
                ui, "threshold",
                &ICON_BOLT.codepoint.to_string(), "Hard threshold",
                "Snap the mask to fully opaque or fully transparent at this cutoff.",
                tip!("Stage 2 of 5. Snaps the mask to fully opaque or fully transparent at this cutoff. Soft = smooth alpha, on = crisp silhouette. When on, downstream stages lose the gradient — Refine can only clean up stairsteps."),
                &mut item_settings.threshold,
                0.001..=0.999, defaults.threshold_value, "Soft",
                |v| format!("{:.1}%", v * 100.0),
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_f32(
                ui, "edge_shift",
                &ICON_SWAP_HORIZ.codepoint.to_string(), "Edge shift",
                "Shrink or grow the mask outline. Positive erodes; negative dilates.",
                tip!("Stage 3 of 5. Shrink or grow the mask outline. Positive = erode (trim fringe pixels), negative = dilate (keep more edge detail). Refine Edges then snaps the shifted boundary to image color."),
                &mut item_settings.edge_shift,
                -50.0..=50.0, defaults.template.edge_shift,
                false,
                |v| {
                    if v > 0.05 { format!("erode {v:.1}px") }
                    else if v < -0.05 { format!("dilate {:.1}px", v.abs()) }
                    else { "0px".to_string() }
                },
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_bool_with_extras(
                ui, "refine_edges",
                &ICON_AUTO_FIX_HIGH.codepoint.to_string(), "Refine edges",
                "Use the original image's colors to sharpen the mask around fine detail like hair or leaves.",
                tip!("Stage 4 of 5. Uses the original image's colors to sharpen the mask around fine detail like hair or leaves. Sees whatever threshold + edge shift produced, so tighter upstream input gives a tighter result. Slower but higher quality."),
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
            ), Tier::Mask, &mut change);

            aggregate(chip::chip_f32(
                ui, "feather",
                &ICON_BLUR_LINEAR.codepoint.to_string(), "Feather",
                "Soften mask edges with a Gaussian blur.",
                tip!("Stage 5 of 5. Final softening pass — Gaussian blur over the finished mask. Runs last so it smooths whatever Refine Edges sharpened; reach for Feather when Refine can't pick up the right detail."),
                &mut item_settings.feather,
                0.0..=10.0, defaults.template.feather,
                false,
                |v| if v < 0.1 { "off".into() } else { format!("σ {v:.1}") },
            ), Tier::Mask, &mut change);

            if render_fill_style_chip(ui, &mut item_settings.fill_style) {
                change.mask = true;
                change.commit = true;
            }
        });

        // Divider between mask and composite groups.
        ui.separator();

        // Unified Background chip: Transparent / Solid colour / source-derived
        // effects (Blurred / Inverted / Desaturated). Behind the scenes the
        // two fields (`bg`, `bg_effect`) stay orthogonal — the chip enforces
        // mutual exclusivity at the UI so users pick one kind at a time.
        render_background_chip(ui, &mut item_settings.bg, &mut item_settings.bg_effect, defaults.bg_value, &mut change);

        // Right-aligned cluster: reset, preset. Right-to-left layout fills
        // from the right edge so items stack: [..free space..] [preset] [↺].
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                // Reset-all-knobs button visible directly (per user feedback —
                // was previously hidden in the kebab overflow menu).
                let reset_btn = egui::Button::new(
                    RichText::new(ICON_RESTART_ALT.codepoint)
                        .color(theme::TEXT_SECONDARY)
                        .size(theme::ICON_SIZE_SMALL),
                )
                .fill(theme::BG_SECONDARY)
                .corner_radius(theme::BUTTON_ROUNDING)
                .min_size(egui::vec2(theme::CHIP_HEIGHT, theme::CHIP_HEIGHT));
                let reset_target = app_settings.default_preset.clone();
                let reset_tooltip = format!(
                    "Reset all knobs to the \"{reset_target}\" preset (your default)"
                );
                if ui.add(reset_btn).on_hover_text(reset_tooltip).clicked() {
                    *item_settings = app_settings.preset_values(&reset_target);
                    *applied_preset = reset_target;
                    change.preset_applied = true;
                    change.mask = true;
                    change.edge = true;
                    change.bg = true;
                    change.commit = true;
                }

                if let Some(name) = preset_dropdown::render(ui, app_settings, item_settings, applied_preset) {
                    *applied_preset = name;
                    change.preset_applied = true;
                    change.commit = true;
                    change.mask = true;
                    change.edge = true;
                    change.bg = true;
                }
            },
        );
    });

    // ── Row 3: Lines mode selector (always visible) + line knobs (when Lines
    // mode is on). The Lines dropdown is the gate for enabling line
    // extraction; keeping it visible at all times is how the user turns
    // lines on. `lines_popover::render` mutates `item_settings.line_mode`
    // directly — cache invalidation below picks up the change.
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        let seg_model_name = super::model_name(app_settings.model);
        ui.add_enabled_ui(!processing, |ui| {
            let _ = super::lines_popover::render(ui, item_settings, seg_model_name);
        });
        if !mask_active {
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
            if super::lines_popover::render_scale_chip(ui, item_settings) {
                change.edge = true;
                change.commit = true;
            }

            aggregate(chip::chip_f32(
                ui, "line_strength",
                &ICON_TUNE.codepoint.to_string(), "Line strength",
                "How much edge detail to capture. Lower = bold outlines only; higher = fine texture.",
                "Stage 2 of 4 in the lines pipeline. Threshold on DexiNed's raw edge tensor. Lower = bold outlines only; higher = fine texture and subtle edges. Feeds edge thickness + solid color.",
                &mut item_settings.line_strength,
                0.0..=1.0, defaults.template.line_strength,
                false,
                |v| format!("{v:.2}"),
            ), Tier::Edge, &mut change);

            aggregate(chip::chip_u32(
                ui, "edge_thickness",
                &ICON_LINE_WEIGHT.codepoint.to_string(), "Edge thickness",
                "Thicken edges by dilating the mask. 0 = native DexiNed width; higher = bolder outlines.",
                "Stage 3 of 4 in the lines pipeline. Dilates the thresholded edge mask by N pixels. 0 = native DexiNed width; higher = bolder outlines that stay readable at display resolution. Runs before solid color, so bolder edges still inherit the paint choice.",
                &mut item_settings.edge_thickness,
                0..=20, defaults.template.edge_thickness,
                |v| if v == 0 { "off".into() } else { format!("+{v}px") },
            ), Tier::Edge, &mut change);

            // Divider between threshold+shape knobs and the paint-time composite choice.
            ui.separator();

            // ComposeMode picker — only meaningful in SubjectOutline mode.
            // Ignored by the worker / live preview in EdgesOnly + Off, so we
            // hide the chip there to avoid presenting a dead control.
            if item_settings.line_mode == LineMode::SubjectOutline {
                if render_compose_mode_chip(ui, &mut item_settings.compose_mode) {
                    change.edge = true;
                    change.commit = true;
                }
            }
            if render_line_style_chip(ui, &mut item_settings.line_style) {
                change.edge = true;
                change.commit = true;
            }
            if render_input_transform_chip(ui, &mut item_settings.input_transform) {
                // Changes to the input transform invalidate BOTH caches —
                // DexiNed sees a different image → signal a full pipeline
                // rerun via seg + edge cache invalidation.
                change.edge_cache_invalid = true;
                change.seg_cache_invalid = true;
                change.commit = true;
            }

            aggregate(chip::chip_option_rgb(
                ui, "solid_line_color",
                &ICON_BRUSH.codepoint.to_string(), "Solid line color",
                "Paint every edge the same color.",
                "Stage 4 of 4 in the lines pipeline. Paint every visible edge the same color, or leave unset to keep the original RGB beneath the mask. Runs after edge thickness.",
                &mut item_settings.solid_line_color,
                defaults.solid_line_color_value,
            ), Tier::Edge, &mut change);
        }
    });

    // Granular cache invalidation:
    // - Segmentation tensor depends on the model. Only stale when the model
    //   swapped. Mask-tier live preview keeps working across preset applies
    //   and line_mode toggles.
    // - Edge tensor depends on what DexiNed saw: full scene (EdgesOnly)
    //   vs subject-on-white (SubjectOutline). Switching between modes with
    //   different inputs invalidates it; switching from/to Off doesn't.
    if change.model_changed {
        change.seg_cache_invalid = true;
    }
    if item_settings.line_mode != before_line_mode {
        // Edge cache is only stale when DexiNed would see a DIFFERENT input.
        // SubjectOutline's edges were computed on the masked-subject image;
        // EdgesOnly's edges on the raw image. Off doesn't run DexiNed at
        // all, so transitions to/from Off preserve whatever edge cache
        // exists (letting a SubjectOutline → Off → SubjectOutline round
        // trip keep live-preview working for line_strength tweaks).
        let invalidates = matches!((before_line_mode, item_settings.line_mode),
            (LineMode::SubjectOutline, LineMode::EdgesOnly)
            | (LineMode::EdgesOnly, LineMode::SubjectOutline));
        if invalidates {
            change.edge_cache_invalid = true;
        }
        change.line_mode_changed = true;
    }

    change
}

/// Which tier a chip's change lifts into on the aggregate ToolbarChange.
/// `Bg` is written directly by `render_background_chip` (not via aggregate)
/// so the enum no longer carries a Bg variant.
#[derive(Copy, Clone)]
enum Tier { Mask, Edge }

/// Input-transform picker. Changes invalidate the DexiNed edge tensor cache
/// (and seg cache for conservatism) — not live-previewable. Label accent
/// reminds the user their next Process will re-run inference.
fn render_input_transform_chip(ui: &mut Ui, transform: &mut prunr_core::InputTransform) -> bool {
    use prunr_core::InputTransform;
    let accent = !matches!(transform, InputTransform::None);
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, &ICON_TUNE.codepoint.to_string(), transform.name(), accent),
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
        chip::chip_button(ui, &ICON_GRADIENT.codepoint.to_string(), style.name(), accent),
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
        chip::chip_button(ui, &ICON_FORMAT_PAINT.codepoint.to_string(), style.name(), accent),
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
enum BgKind { Transparent, Solid, BlurredSource, InvertedSource, DesaturatedSource }

impl BgKind {
    const ALL: [Self; 5] = [
        Self::Transparent, Self::Solid,
        Self::BlurredSource, Self::InvertedSource, Self::DesaturatedSource,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Transparent => "Transparent",
            Self::Solid => "Solid colour",
            Self::BlurredSource => "Blurred source",
            Self::InvertedSource => "Inverted source",
            Self::DesaturatedSource => "Desaturated source",
        }
    }

    /// Derive the current kind from the two underlying fields. Effects win
    /// over solid colour — matches the render-time precedence.
    fn current(bg: &Option<[u8; 4]>, effect: &prunr_core::BgEffect) -> Self {
        use prunr_core::BgEffect;
        match effect {
            BgEffect::BlurredSource { .. } => Self::BlurredSource,
            BgEffect::InvertedSource => Self::InvertedSource,
            BgEffect::DesaturatedSource => Self::DesaturatedSource,
            BgEffect::None => if bg.is_some() { Self::Solid } else { Self::Transparent },
        }
    }

    /// Effects require a postprocess rerun (baked into the output RGBA);
    /// Transparent / Solid only change the render-time bg fill.
    fn needs_postprocess(self) -> bool {
        matches!(self, Self::BlurredSource | Self::InvertedSource | Self::DesaturatedSource)
    }
}

/// Unified Background chip: one control for "what fills the transparent
/// area behind the subject" — solid colour or a source-derived effect.
fn render_background_chip(
    ui: &mut Ui,
    bg: &mut Option<[u8; 4]>,
    bg_effect: &mut prunr_core::BgEffect,
    default_color: [u8; 4],
    change: &mut ToolbarChange,
) {
    use egui::widgets::color_picker::{color_picker_color32, Alpha};
    use prunr_core::BgEffect;

    let current = BgKind::current(bg, bg_effect);
    let accent = current != BgKind::Transparent;
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, &ICON_PALETTE.codepoint.to_string(), current.name(), accent),
        "Background",
        "What fills transparent areas behind the subject: a solid colour, \
         or a source-derived effect (blurred / inverted / desaturated).",
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
                        apply_bg_kind(bg, bg_effect, kind, default_color);
                        // Effect transitions need a postprocess rerun (bake
                        // the effect into the output RGBA); Transparent/Solid
                        // toggles are render-time only.
                        if kind.needs_postprocess() || current.needs_postprocess() {
                            change.mask = true;
                        } else {
                            change.bg = true;
                        }
                        change.commit = true;
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
                                change.bg = true;
                                change.commit = true;
                            }
                            hint(ui, "Solid colour fills transparent areas at render / export.");
                        }
                    }
                    BgKind::BlurredSource => {
                        if let BgEffect::BlurredSource { radius } = bg_effect {
                            if ui.add(egui::Slider::new(radius, 1..=64).text("Blur radius")).changed() {
                                change.mask = true;
                                change.commit = true;
                            }
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
        chip::chip_button(ui, &ICON_LAYERS.codepoint.to_string(), &mode.to_string(), accent),
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
    let enabled = !processing && mask_active;
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
        let selected_text = if mask_active {
            model_label(app_settings.model, true)
        } else {
            format!("{}  Bypassed", ICON_BLOCK.codepoint)
        };
        egui::ComboBox::from_id_salt("adjustments_model")
            .selected_text(
                RichText::new(selected_text)
                    .color(theme::TEXT_PRIMARY),
            )
            .show_ui(ui, |ui| {
                ui.label(RichText::new("Models").strong().color(theme::TEXT_PRIMARY));
                ui.add_space(theme::SPACE_XS);
                ui.separator();
                ui.add_space(theme::SPACE_XS);
                for variant in SettingsModel::ALL {
                    ui.selectable_value(
                        &mut app_settings.model,
                        variant,
                        RichText::new(model_label(variant, false))
                            .color(theme::TEXT_PRIMARY),
                    );
                }
            })
            .response
            .on_hover_ui(|ui| {
                let (heading, body) = if mask_active {
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
        // Clamp parallel jobs to the new model's safe maximum. Keeping this
        // here (vs the caller) because it's a correctness invariant on
        // app_settings — we mustn't leave an invalid parallel_jobs value.
        // Disk persistence, toasts, and cache invalidation all live on the
        // caller via `change.model_changed`.
        let max = app_settings.max_jobs();
        if app_settings.parallel_jobs > max {
            app_settings.parallel_jobs = max;
        }
        change.model_changed = true;
        change.commit = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbar_change_default_invalidates_nothing() {
        let c = ToolbarChange::default();
        assert!(!c.seg_cache_invalid);
        assert!(!c.edge_cache_invalid);
    }
}
