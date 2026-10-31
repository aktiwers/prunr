//! Eraser chips: the SD eraser's Quality, Prompt and Advanced (the
//! fill's scale and sharpness, steps, guidance, denoising strength,
//! sampler, seed, the fast decoder and the edge), RORem's Advanced, and
//! the other erasers' Advanced (the fill's scale, sharpness and edge).

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{
    default_sd_strength, default_sharpen, BrushSettings, SdScheduler, DEFAULT_SD_NEGATIVE_PROMPT, ROREM_STEPS_RANGE,
};
use prunr_core::inpaint::FillScale;
use prunr_core::inpaint_blend::EdgeColour;
use prunr_core::inpaint_sd::ROREM_STEPS;
use crate::gui::presets::SdSchedulerBundle;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::chip::{self, GroupChip};
use super::{differs, fmt, get_model, hint};

/// Render the SD-eraser chips. Caller decides placement. Returns true
/// when a setting was committed.
pub(crate) fn render(
    ui: &mut egui::Ui,
    app_settings: &mut Settings,
    installed: crate::gui::hardware_cache::HardwareInstallCache,
    shown_seed: Option<u64>,
) -> bool {
    let mut committed = render_quality_chip(ui, app_settings, installed.lcm_bundle);
    committed |= render_prompt_chip(ui, &mut app_settings.brush);
    committed |= render_advanced_group(ui, &mut app_settings.brush, installed.taesd, shown_seed);
    committed
}

fn render_advanced_group(ui: &mut egui::Ui, brush: &mut BrushSettings, taesd_installed: bool, shown_seed: Option<u64>) -> bool {
    let quality = brush.sd_scheduler;
    let shipped = SdSchedulerBundle::default_for(quality);
    let tuned = usize::from(brush.sd_steps != shipped.steps)
        + usize::from(differs(brush.sd_guidance_scale, shipped.guidance_scale))
        + usize::from(differs(brush.sd_strength, default_sd_strength()))
        + usize::from(brush.sd_seed.is_some())
        + usize::from(brush.sd_use_taesd.is_some())
        + usize::from(brush.sd_sampler.is_some())
        + generated_edge_tuned(brush)
        + fill_tuned(brush, true);
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "eraser_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "How the eraser generates the fill: its scale and sharpness, steps, guidance, denoising strength, sampler, seed and how it meets the photo.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: false,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            brush.sd_steps = shipped.steps;
            brush.sd_guidance_scale = shipped.guidance_scale;
            brush.sd_strength = default_sd_strength();
            brush.sd_seed = None;
            brush.sd_use_taesd = None;
            brush.sd_sampler = None;
            reset_generated_edge(brush);
            reset_fill(brush, true);
            committed = true;
        }

        committed |= fill_rows(ui, brush);
        committed |= chip::slider_row(ui, "Steps", &mut brush.sd_steps, quality.steps_range()).commit;
        hint(ui, "More steps refine the fill and take longer.");
        ui.add_space(theme::SPACE_XS);

        committed |= chip::slider_row_f32(
            ui, "Guidance", &mut brush.sd_guidance_scale, quality.guidance_range(), false,
            |v| if v <= 1.0 + 1e-3 { "Off".to_string() } else { fmt::plain(v, 1) },
        ).commit;
        hint(ui, "How closely the fill follows the prompt and avoids the negative prompt. 1 turns both off; higher follows closer but burns colors.");
        ui.add_space(theme::SPACE_XS);

        committed |= chip::slider_row_f32(ui, "Denoising strength", &mut brush.sd_strength, 0.0..=1.0, false, fmt::percent).commit;
        hint(ui, "How much of the painted area is drawn anew. 100% replaces it; lower keeps its shapes and colors, to change something rather than remove it.");
        ui.add_space(theme::SPACE_XS);

        committed |= seed_row(ui, brush, shown_seed);
        ui.add_space(theme::SPACE_XS);

        let mut fast = brush.sd_use_taesd_effective_with_avail(taesd_installed);
        let reason = (!taesd_installed).then(|| get_model::needs("The fast decoder", prunr_models::ModelId::TaesdFp16));
        if chip::gated(ui, reason.as_deref(), |ui| chip::toggle_row(ui, "Fast decoder", &mut fast)).changed {
            brush.sd_use_taesd = Some(fast);
            committed = true;
        }
        hint(ui, "About three times faster decoding at a slight quality cost.");
        ui.add_space(theme::SPACE_XS);

        committed |= chip::choice_row(ui, "Sampler", &SAMPLERS, &mut brush.sd_sampler);
        hint(ui, "How each step is taken. DPM++ 2M Karras suits Quality's full model, LCM the Fast one.");
        ui.add_space(theme::SPACE_XS);

        committed |= generated_edge_rows(ui, brush);
    });
    committed
}

const SAMPLERS: [chip::Choice<Option<SdScheduler>>; 3] = [
    chip::Choice { value: None, name: "As Quality", description: "The sampler the chosen quality runs", enabled: true },
    chip::Choice { value: Some(SdScheduler::DpmPlusPlus2MKarras), name: "DPM++ 2M Karras", description: "Detailed, for 20 steps or more", enabled: true },
    chip::Choice { value: Some(SdScheduler::Lcm), name: "LCM", description: "A few steps; made for the Fast model", enabled: true },
];

const EDGE_COLOURS: [chip::Choice<EdgeColour>; 3] = [
    chip::Choice { value: EdgeColour::AlongEdge, name: "Along the edge", description: "Matches the colors around each part of the edge", enabled: true },
    chip::Choice { value: EdgeColour::WholeArea, name: "Whole area", description: "One color shift for the whole area", enabled: true },
    chip::Choice { value: EdgeColour::Off, name: "Off", description: "The fill's own colors", enabled: true },
];

fn edge_colour_row(ui: &mut egui::Ui, brush: &mut BrushSettings, generated: bool) -> bool {
    let mut colour = brush.edge_colour(generated);
    let changed = chip::choice_row(ui, "Edge color", &EDGE_COLOURS, &mut colour);
    if changed {
        brush.inpaint_edge_colour = Some(colour);
    }
    hint(ui, "How the fill's colors meet the photo around it.");
    changed
}

/// Whether the edge colour differs from the eraser's own.
fn edge_colour_tuned(brush: &BrushSettings, generated: bool) -> bool {
    brush.edge_colour(generated) != crate::gui::brush_state::default_edge_colour(generated)
}

/// The edge of a generated fill (SD and RORem): Soft edge, then Edge colour.
fn generated_edge_rows(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let mut committed = chip::toggle_row(ui, "Soft edge", &mut brush.sd_soft_edge).changed;
    hint(ui, "Draws the Feather band last, against the photo around it, so the fill blends in.");
    ui.add_space(theme::SPACE_XS);
    committed |= edge_colour_row(ui, brush, true);
    committed
}

/// How many of the generated edge's settings differ from the shipped ones.
fn generated_edge_tuned(brush: &BrushSettings) -> usize {
    usize::from(!brush.sd_soft_edge) + usize::from(edge_colour_tuned(brush, true))
}

fn reset_generated_edge(brush: &mut BrushSettings) {
    brush.sd_soft_edge = true;
    brush.inpaint_edge_colour = None;
}

/// LaMa, Big-LaMa and MI-GAN's Advanced group: the fill's scale and
/// sharpness, and how it meets the photo.
pub(crate) fn render_fill_advanced(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let tuned = usize::from(edge_colour_tuned(brush, false)) + fill_tuned(brush, false);
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "fill_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "The fill's scale and sharpness, and how it meets the photo around it.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: false,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            brush.inpaint_edge_colour = None;
            reset_fill(brush, false);
            committed = true;
        }
        committed |= fill_rows(ui, brush);
        committed |= edge_colour_row(ui, brush, false);
    });
    committed
}

const FILL_SCALES: [chip::Choice<FillScale>; 4] = [
    chip::Choice { value: FillScale::Auto, name: "Auto", description: "Full size for small areas; large ones shrink to fit", enabled: true },
    chip::Choice { value: FillScale::Full, name: "Full", description: "Every area at full size: sharpest, blotchy on large ones", enabled: true },
    chip::Choice { value: FillScale::Half, name: "1/2", description: "Half size", enabled: true },
    chip::Choice { value: FillScale::Quarter, name: "1/4", description: "Quarter size: softest, most even on large areas", enabled: true },
];

/// The fill's scale and sharpness, at the top of every eraser's Advanced.
fn fill_rows(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let mut committed = chip::choice_row(ui, "Fill scale", &FILL_SCALES, &mut brush.inpaint_fill_scale);
    hint(ui, "A large area fills more convincingly at a smaller scale, a little softer. Auto keeps small areas at full size.");
    ui.add_space(theme::SPACE_XS);
    // Sharpen displays as 0-100% on a 0-2 internal range.
    committed |= chip::slider_row_f32(ui, "Sharpen", &mut brush.inpaint_sharpen, 0.0..=2.0, false, |v| fmt::percent(v / 2.0)).commit;
    hint(ui, "Sharpen the filled area, which comes out slightly soft.");
    ui.add_space(theme::SPACE_XS);
    committed
}

/// How many of the fill's scale and sharpness differ from the shipped ones.
fn fill_tuned(brush: &BrushSettings, generated: bool) -> usize {
    usize::from(brush.inpaint_fill_scale != FillScale::default()) + usize::from(differs(brush.inpaint_sharpen, default_sharpen(generated)))
}

fn reset_fill(brush: &mut BrushSettings, generated: bool) {
    brush.inpaint_fill_scale = FillScale::default();
    brush.inpaint_sharpen = default_sharpen(generated);
}

/// Pin the seed of the fill on screen, or let each fill draw its own.
fn seed_row(ui: &mut egui::Ui, brush: &mut BrushSettings, shown_seed: Option<u64>) -> bool {
    let mut pinned = brush.sd_seed.is_some();
    let seed_label = match brush.sd_seed {
        Some(seed) => format!("Pin seed  \u{2026}{:06}", seed % 1_000_000),
        None => "Pin seed".to_string(),
    };
    let changed = chip::toggle_row(ui, &seed_label, &mut pinned).changed;
    if changed {
        brush.sd_seed = pinned.then(|| shown_seed.unwrap_or_else(prunr_core::inpaint_sd::fresh_seed));
    }
    hint(ui, "Keeps the seed of the fill on screen, so the same settings give that fill again.");
    changed
}

/// RORem's Advanced group: steps, seed and the edge. Its prompt is baked
/// in, so there is nothing else to tell it.
pub(crate) fn render_rorem(ui: &mut egui::Ui, brush: &mut BrushSettings, shown_seed: Option<u64>) -> bool {
    let tuned = usize::from(brush.rorem_steps != ROREM_STEPS) + usize::from(brush.sd_seed.is_some()) + generated_edge_tuned(brush) + fill_tuned(brush, true);
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "rorem_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "How the eraser generates the fill: its scale and sharpness, steps, seed and how it meets the photo.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: false,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            brush.rorem_steps = ROREM_STEPS;
            brush.sd_seed = None;
            reset_generated_edge(brush);
            reset_fill(brush, true);
            committed = true;
        }
        committed |= fill_rows(ui, brush);
        committed |= chip::slider_row(ui, "Steps", &mut brush.rorem_steps, ROREM_STEPS_RANGE).commit;
        hint(ui, "Fewer steps are faster but rougher; past 4, the number the model was trained for, the fill barely changes.");
        ui.add_space(theme::SPACE_XS);
        committed |= seed_row(ui, brush, shown_seed);
        ui.add_space(theme::SPACE_XS);
        committed |= generated_edge_rows(ui, brush);
    });
    committed
}

fn render_quality_chip(
    ui: &mut egui::Ui,
    app_settings: &mut Settings,
    lcm_bundle_installed: bool,
) -> bool {
    let pop_id = egui::Id::new("eraser_preset_popover");
    let active = app_settings.brush.sd_scheduler;
    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_AUTO_AWESOME.codepoint, active.label(), false),
        "Quality",
        "Fast fills in a few seconds; Quality takes longer for the best fill.",
        None,
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        chip::popover_header(ui, "Quality", None);
        for quality in SdScheduler::ALL {
            let (label, desc) = (quality.label(), quality.description());
            if quality == SdScheduler::Lcm && !lcm_bundle_installed {
                let reason = get_model::needs(label, prunr_models::ModelId::SdV15LcmInpaintFp16);
                chip::gated(ui, Some(&reason), |ui| chip::picker_row(ui, false, label, desc));
                continue;
            }
            let selected = active == quality;
            if chip::picker_row(ui, selected, label, desc).clicked() {
                if !selected {
                    app_settings.on_scheduler_change_resolve_sd(quality);
                    changed = true;
                }
                egui::Popup::close_id(ui.ctx(), pop_id);
            }
        }
    });
    changed
}

fn render_prompt_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_prompt_popover");
    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_EDIT_NOTE.codepoint, "Prompt", !brush.sd_prompt.trim().is_empty()),
        "Prompt",
        "Describe what should fill the painted area. Empty continues the surroundings.",
        None,
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        ui.set_min_width(theme::POPOVER_WIDTH_WIDE);
        let already_default = brush.sd_prompt.is_empty()
            && brush.sd_negative_prompt == DEFAULT_SD_NEGATIVE_PROMPT;
        if chip::popover_header(ui, "Prompt", Some(("Clear the prompt and restore the shipped negative prompt", already_default))) {
            brush.sd_prompt.clear();
            brush.sd_negative_prompt = DEFAULT_SD_NEGATIVE_PROMPT.to_string();
            changed = true;
        }
        let p = chip::named(ui.add(
            egui::TextEdit::multiline(&mut brush.sd_prompt)
                .hint_text("Empty: continue the surroundings")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        ), "Prompt text");
        if p.lost_focus() { changed = true; }
        hint(ui, "What should fill the painted area. Be specific: \"wooden park bench in autumn forest\" works better than \"bench\".");

        ui.add_space(theme::SPACE_SM);
        ui.label(RichText::new("Negative prompt").strong().color(theme::TEXT_PRIMARY));
        let np = chip::named(ui.add(
            egui::TextEdit::multiline(&mut brush.sd_negative_prompt)
                .hint_text("e.g. blurry, watermark, low quality")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        ), "Negative prompt");
        if np.lost_focus() { changed = true; }
        hint(ui, "What to keep out of the fill. Used when Guidance (under Advanced) is above 1.");
    });
    changed
}
