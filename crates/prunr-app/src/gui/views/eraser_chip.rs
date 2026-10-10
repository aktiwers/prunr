//! SD-eraser chips: Quality, Prompt, and an Advanced group
//! holding the scheduler, steps, denoising strength, Karras, seed and the
//! fast decoder.

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{
    BrushSettings, SdQualityPreset, SdScheduler,
    DEFAULT_SD_NEGATIVE_PROMPT, DEFAULT_SD_PROMPT,
    default_cfg,
};
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
) -> EraserRowChange {
    let mut change = EraserRowChange::default();
    change.committed |= render_quality_preset_chip(ui, app_settings, installed.lcm_bundle);
    change.committed |= render_prompt_chip(ui, &mut app_settings.brush);
    change.committed |= render_advanced_group(ui, app_settings, installed.lcm_bundle, installed.taesd);
    change
}

/// Steps the scheduler is trained for: LCM caps at 8, standard SD runs
/// to 30. Clamps a stale count left behind by a scheduler switch.
fn steps_range(brush: &mut BrushSettings) -> (u32, u32) {
    let lcm = brush.sd_scheduler == SdScheduler::Lcm;
    let max = if lcm { 8 } else { 30 };
    if brush.sd_steps > max {
        brush.sd_steps = max;
    }
    (max, if lcm { 8 } else { 20 })
}

fn render_advanced_group(ui: &mut egui::Ui, app_settings: &mut Settings, lcm_bundle_installed: bool, taesd_installed: bool) -> bool {
    let (max_steps, default_steps) = steps_range(&mut app_settings.brush);
    let brush = &app_settings.brush;
    let tuned = usize::from(brush.sd_steps != default_steps)
        + usize::from(differs(brush.sd_strength, 1.0))
        + usize::from(brush.sd_use_karras_sigmas)
        + usize::from(brush.sd_seed.is_some())
        + usize::from(brush.sd_use_taesd == Some(false));
    let summary = chip::tuned_summary(tuned);
    let group = GroupChip {
        id_salt: "eraser_advanced",
        icon: ICON_TUNE.codepoint,
        label: "Advanced",
        summary: &summary,
        tooltip: "How the eraser generates the fill: scheduler, steps, strength, seed.",
        tuned: tuned > 0,
        width: theme::POPOVER_WIDTH,
    };
    let mut committed = false;
    chip::group_chip(ui, group, |ui, reset| {
        if reset {
            let b = &mut app_settings.brush;
            b.sd_steps = default_steps;
            b.sd_strength = 1.0;
            b.sd_use_karras_sigmas = false;
            b.sd_seed = None;
            b.sd_use_taesd = None;
            committed = true;
        }

        chip::section_label(ui, "Scheduler");
        for sched in SdScheduler::ALL {
            let label = sched.label();
            let desc = sched.description();
            let bundle_gated = matches!(sched, SdScheduler::Lcm) && !lcm_bundle_installed;
            if !sched.is_available() {
                chip::gated(ui, Some("Not available in this build yet."), |ui| chip::picker_row(ui, false, label, desc));
            } else if bundle_gated {
                chip::gated(ui, Some(LCM_DOWNLOAD_HINT), |ui| chip::picker_row(ui, false, label, desc));
            } else {
                let selected = app_settings.brush.sd_scheduler == sched;
                if chip::picker_row(ui, selected, label, desc).clicked() && !selected {
                    app_settings.on_scheduler_change_resolve_sd(sched);
                    committed = true;
                }
            }
        }
        ui.add_space(theme::SPACE_XS);

        let brush = &mut app_settings.brush;
        committed |= chip::slider_row(ui, "Steps", &mut brush.sd_steps, 1..=max_steps).commit;
        hint(ui, "More steps refine the fill and take longer.");
        ui.add_space(theme::SPACE_XS);
        committed |= chip::slider_row_f32(ui, "Denoising strength", &mut brush.sd_strength, 0.0..=1.0, false, fmt::percent).commit;
        hint(ui, "100% invents the fill from scratch; lower keeps more of the original.");
        ui.add_space(theme::SPACE_XS);

        // Only LCM honours the Karras toggle; the other schedulers ship with
        // their own fixed schedule.
        if brush.sd_scheduler == SdScheduler::Lcm {
            committed |= chip::toggle_row(ui, "Karras schedule", &mut brush.sd_use_karras_sigmas).changed;
            hint(ui, "An alternative step spacing. Try both on your photo.");
            ui.add_space(theme::SPACE_XS);
        }

        let mut pinned = brush.sd_seed.is_some();
        let seed_label = match brush.sd_seed {
            Some(seed) => format!("Pin seed  \u{2026}{:06}", seed % 1_000_000),
            None => "Pin seed".to_string(),
        };
        if chip::toggle_row(ui, &seed_label, &mut pinned).changed {
            brush.sd_seed = pinned.then(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
            });
            committed = true;
        }
        hint(ui, "Pinned: the same prompt and settings give the same fill every time.");
        ui.add_space(theme::SPACE_XS);

        let mut fast = brush.sd_use_taesd.unwrap_or(true) && taesd_installed;
        let reason = (!taesd_installed).then_some(TAESD_DOWNLOAD_HINT);
        if chip::gated(ui, reason, |ui| chip::toggle_row(ui, "Fast decoder", &mut fast)).changed {
            brush.sd_use_taesd = Some(fast);
            committed = true;
        }
        hint(ui, "About three times faster decoding at a slight quality cost.");
    });
    committed
}

fn render_quality_preset_chip(
    ui: &mut egui::Ui,
    app_settings: &mut Settings,
    lcm_bundle_installed: bool,
) -> bool {
    let pop_id = egui::Id::new("eraser_preset_popover");
    let active = SdQualityPreset::detect_from(&app_settings.brush);
    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_AUTO_AWESOME.codepoint, active.label(), false),
        "Quality",
        "Fast, Balanced or Quality picks the scheduler and steps for you. Changing them by hand shows Custom.",
        None,
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        chip::popover_header(ui, "Quality", None);
        for preset in [SdQualityPreset::Fast, SdQualityPreset::Balanced, SdQualityPreset::Quality] {
            let preset_scheduler = match preset {
                SdQualityPreset::Quality => SdScheduler::DpmPlusPlus2MKarras,
                _ => SdScheduler::Lcm,
            };
            let label = preset.label();
            let preset_bundle_gated =
                matches!(preset_scheduler, SdScheduler::Lcm) && !lcm_bundle_installed;
            if !preset_scheduler.is_available() {
                chip::gated(ui, Some("Not available in this build yet."), |ui| chip::picker_row(ui, false, label, ""));
            } else if preset_bundle_gated {
                chip::gated(ui, Some(LCM_DOWNLOAD_HINT), |ui| chip::picker_row(ui, false, label, ""));
            } else {
                let selected = active == preset;
                if chip::picker_row(ui, selected, label, "").clicked() {
                    preset.apply_to_settings(app_settings);
                    changed = true;
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
            }
        }
    });
    changed
}

fn render_prompt_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_prompt_popover");
    let lcm = brush.sd_scheduler == SdScheduler::Lcm;
    let neg_color = if lcm { theme::TEXT_SECONDARY } else { theme::TEXT_PRIMARY };

    let resp = chip::tooltip(
        chip::chip_button(ui, ICON_EDIT_NOTE.codepoint, "Prompt", !brush.sd_prompt.is_empty()),
        "Prompt",
        "Describe what should fill the painted area. Leave it empty for a plain fill, which can look noisy on flat areas.",
        None,
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        ui.set_min_width(theme::POPOVER_WIDTH_WIDE);
        let already_default = brush.sd_prompt == DEFAULT_SD_PROMPT
            && brush.sd_negative_prompt == DEFAULT_SD_NEGATIVE_PROMPT
            && (brush.sd_guidance_scale - default_cfg()).abs() < 1e-3;
        if chip::popover_header(ui, "Prompt", Some(("Back to the shipped prompt, negative prompt and guidance", already_default))) {
            brush.sd_prompt = DEFAULT_SD_PROMPT.to_string();
            brush.sd_negative_prompt = DEFAULT_SD_NEGATIVE_PROMPT.to_string();
            brush.sd_guidance_scale = default_cfg();
            changed = true;
        }
        let p = ui.add(
            egui::TextEdit::multiline(&mut brush.sd_prompt)
                .hint_text("e.g. wooden park bench in autumn forest")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        if p.lost_focus() { changed = true; }
        super::hint(ui, "What should fill the painted area. Be specific: \"wooden park bench in autumn forest\" works better than \"bench\".");

        ui.add_space(theme::SPACE_SM);

        ui.add_enabled_ui(!lcm, |ui| {
            ui.label(RichText::new("Negative prompt").strong().color(neg_color));
            let np = ui.add(
                egui::TextEdit::multiline(&mut brush.sd_negative_prompt)
                    .hint_text("e.g. blurry, watermark, low quality")
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            if np.lost_focus() { changed = true; }
            super::hint(ui, "What to push away from. Only used when Guidance > 1.");
            ui.add_space(theme::SPACE_SM);
            let cfg = chip::slider_row_f32(
                ui, "Guidance", &mut brush.sd_guidance_scale, 1.0..=15.0, false,
                |v| if v <= 1.0 + 1e-3 { "Off".to_string() } else { fmt::plain(v, 1) },
            );
            if cfg.commit { changed = true; }
            super::hint(ui, "How closely to follow the prompt. 1 ignores it; 7 to 8 is typical; higher matches closer but burns colors.");
        });

        if lcm {
            ui.add_space(theme::SPACE_SM);
            super::hint(ui, "The LCM scheduler ignores the negative prompt and guidance. Pick another scheduler under Advanced to use them.");
        }
    });
    changed
}

