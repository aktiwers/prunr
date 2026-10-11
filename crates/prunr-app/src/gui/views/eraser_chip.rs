//! SD-eraser chips: Quality, Prompt, and an Advanced group holding the
//! steps, guidance, seed and the fast decoder.

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{
    fresh_sd_seed, BrushSettings, SdScheduler, DEFAULT_SD_NEGATIVE_PROMPT,
};
use crate::gui::presets::SdSchedulerBundle;
use crate::gui::settings::Settings;
use crate::gui::theme;

use super::chip::{self, GroupChip};
use super::{differs, fmt, hint};

const LCM_DOWNLOAD_HINT: &str =
    "Download Eraser (SD 1.5 LCM, fast) in the Model Store to enable.";
const TAESD_DOWNLOAD_HINT: &str =
    "Download the fast decoder in the Model Store to enable.";

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
    brush.fit_sd_to_quality();
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

        let mut pinned = brush.sd_seed.is_some();
        let seed_label = match brush.sd_seed {
            Some(seed) => format!("Pin seed  \u{2026}{:06}", seed % 1_000_000),
            None => "Pin seed".to_string(),
        };
        if chip::toggle_row(ui, &seed_label, &mut pinned).changed {
            brush.sd_seed = pinned.then(|| shown_seed.unwrap_or_else(fresh_sd_seed));
            committed = true;
        }
        hint(ui, "Keeps the seed of the fill on screen, so the same prompt and settings give that fill again.");
        ui.add_space(theme::SPACE_XS);

        let mut fast = brush.sd_use_taesd_effective_with_avail(taesd_installed);
        let reason = (!taesd_installed).then_some(TAESD_DOWNLOAD_HINT);
        if chip::gated(ui, reason, |ui| chip::toggle_row(ui, "Fast decoder", &mut fast)).changed {
            brush.sd_use_taesd = Some(fast);
            committed = true;
        }
        hint(ui, "About three times faster decoding at a slight quality cost.");
    });
    committed
}

fn render_quality_chip(
    ui: &mut egui::Ui,
    app_settings: &mut Settings,
    lcm_bundle_installed: bool,
) -> bool {
    let pop_id = egui::Id::new("eraser_preset_popover");
    app_settings.fit_sd_quality_to_install(lcm_bundle_installed);
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
                chip::gated(ui, Some(LCM_DOWNLOAD_HINT), |ui| chip::picker_row(ui, false, label, desc));
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
        let p = ui.add(
            egui::TextEdit::multiline(&mut brush.sd_prompt)
                .hint_text("Empty: continue the surroundings")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        if p.lost_focus() { changed = true; }
        hint(ui, "What should fill the painted area. Be specific: \"wooden park bench in autumn forest\" works better than \"bench\".");

        ui.add_space(theme::SPACE_SM);
        ui.label(RichText::new("Negative prompt").strong().color(theme::TEXT_PRIMARY));
        let np = ui.add(
            egui::TextEdit::multiline(&mut brush.sd_negative_prompt)
                .hint_text("e.g. blurry, watermark, low quality")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        if np.lost_focus() { changed = true; }
        hint(ui, "What to keep out of the fill. Used when Guidance (under Advanced) is above 1.");
    });
    changed
}
