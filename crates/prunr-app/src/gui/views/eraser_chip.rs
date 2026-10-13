//! SD-eraser chips: Quality, Prompt, and an Advanced group holding the
//! steps, guidance, seed and the fast decoder.

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{
    BrushSettings, SdScheduler, DEFAULT_SD_NEGATIVE_PROMPT, ROREM_STEPS_RANGE,
};
use prunr_core::inpaint_sd::ROREM_STEPS;
use crate::gui::presets::SdSchedulerBundle;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::chip::{self, GroupChip};
use super::{differs, fmt, get_model, hint};


#[derive(Default)]
pub struct EraserRowChange {
    pub committed: bool,
}

/// Render the SD-eraser chips. Caller decides placement.
pub(crate) fn render(
    ui: &mut egui::Ui,
    app_settings: &mut Settings,
    installed: crate::gui::hardware_cache::HardwareInstallCache,
    shown_seed: Option<u64>,
) -> EraserRowChange {
    let mut change = EraserRowChange::default();
    change.committed |= render_quality_chip(ui, app_settings, installed.lcm_bundle);
    change.committed |= render_prompt_chip(ui, &mut app_settings.brush);
    change.committed |= render_advanced_group(ui, &mut app_settings.brush, installed.taesd, shown_seed);
    change
}

fn render_advanced_group(ui: &mut egui::Ui, brush: &mut BrushSettings, taesd_installed: bool, shown_seed: Option<u64>) -> bool {
    let quality = brush.sd_scheduler;
    let shipped = SdSchedulerBundle::default_for(quality);
    let tuned = usize::from(brush.sd_steps != shipped.steps)
        + usize::from(differs(brush.sd_guidance_scale, shipped.guidance_scale))
        + usize::from(brush.sd_seed.is_some())
        + usize::from(brush.sd_use_taesd.is_some());
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "eraser_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "How the eraser generates the fill: steps, guidance, seed.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: false,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            brush.sd_steps = shipped.steps;
            brush.sd_guidance_scale = shipped.guidance_scale;
            brush.sd_seed = None;
            brush.sd_use_taesd = None;
            committed = true;
        }

        committed |= chip::slider_row(ui, "Steps", &mut brush.sd_steps, quality.steps_range()).commit;
        hint(ui, "More steps refine the fill and take longer.");
        ui.add_space(theme::SPACE_XS);

        committed |= chip::slider_row_f32(
            ui, "Guidance", &mut brush.sd_guidance_scale, quality.guidance_range(), false,
            |v| if v <= 1.0 + 1e-3 { "Off".to_string() } else { fmt::plain(v, 1) },
        ).commit;
        hint(ui, "How closely the fill follows the prompt and avoids the negative prompt. 1 turns both off; higher follows closer but burns colors.");
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
    });
    committed
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

/// RORem's Advanced group: steps and seed. Its prompt is baked in, so
/// there is nothing else to tell it.
pub(crate) fn render_rorem(ui: &mut egui::Ui, brush: &mut BrushSettings, shown_seed: Option<u64>) -> bool {
    let tuned = usize::from(brush.rorem_steps != ROREM_STEPS) + usize::from(brush.sd_seed.is_some());
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "rorem_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "How the eraser generates the fill: steps and seed.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
        live: false,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            brush.rorem_steps = ROREM_STEPS;
            brush.sd_seed = None;
            committed = true;
        }
        committed |= chip::slider_row(ui, "Steps", &mut brush.rorem_steps, ROREM_STEPS_RANGE).commit;
        hint(ui, "Fewer steps are faster but rougher; above the 4 the model was trained for, the fill barely changes.");
        ui.add_space(theme::SPACE_XS);
        committed |= seed_row(ui, brush, shown_seed);
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
