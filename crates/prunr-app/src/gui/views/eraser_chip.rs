//! SD-eraser toolbar chips: Quality / Scheduler / Steps / Strength /
//! EdgeSoftness / Karras / Seed / Prompt. Renders as a horizontal
//! cluster inline in Row 2 next to the model dropdown — dropdown chips
//! mirror `lines_popover`'s pattern (chip-button + popover with
//! selectable rows). Karras only renders for schedulers that accept the
//! toggle (LCM).

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{
    BrushSettings, SdQualityPreset, SdScheduler,
    DEFAULT_SD_NEGATIVE_PROMPT, DEFAULT_SD_PROMPT,
    DEFAULT_MASK_BLUR,
    default_cfg,
};
use crate::gui::theme;
use prunr_core::inpaint_sd::{MASK_BLUR_MAX, MASK_BLUR_OFF_THRESHOLD};

use super::chip::{self, ChipMeta};

#[derive(Default)]
pub struct EraserRowChange {
    pub committed: bool,
}

/// Render the SD-eraser chip cluster. Caller decides placement.
pub fn render(ui: &mut egui::Ui, brush: &mut BrushSettings) -> EraserRowChange {
    let mut change = EraserRowChange::default();
    ui.horizontal(|ui| {
        change.committed |= render_quality_preset_chip(ui, brush);
        change.committed |= render_scheduler_chip(ui, brush);
        change.committed |= render_steps_chip(ui, brush);
        change.committed |= render_strength_chip(ui, brush);
        change.committed |= render_mask_blur_chip(ui, brush);
        // Karras toggle: LCM (user-toggleable), UniPC, Euler-A.
        // DDIM and DPM++ 2M Karras are pinned to one setting in this build.
        // Karras chip is only meaningful for schedulers whose dispatch
        // honors the toggle. LCM is the only one wired today; UniPC and
        // Euler-A both ship Karras-on hardcoded (matches Diffusers'
        // SD-1.5 reference) and ignore the field. Showing a no-op
        // toggle confuses users — hide it instead.
        if matches!(brush.sd_scheduler, SdScheduler::Lcm) {
            change.committed |= render_karras_chip(ui, brush);
        }
        change.committed |= render_seed_chip(ui, brush);
        change.committed |= render_prompt_chip(ui, brush);
    });
    change
}

fn render_strength_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let change = chip::chip_f32(
        ui,
        ChipMeta {
            id_salt: "eraser_strength",
            icon: ICON_AUTO_FIX_HIGH.codepoint,
            label: "Strength",
            description: "Inpaint aggressiveness. 100% = pure noise init, fully creative rewrite. 70-85% = preserve structure / lighting and make targeted edits. <50% = subtle nudges. 0% = preserve original.",
            tooltip: "Inpaint strength",
        },
        &mut brush.sd_strength,
        0.0..=1.0,
        1.0,
        false,
        |v| format!("{:.0}%", v * 100.0),
    );
    change.commit
}

fn render_mask_blur_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let change = chip::chip_f32(
        ui,
        ChipMeta {
            id_salt: "eraser_mask_blur",
            icon: ICON_BLUR_ON.codepoint,
            label: "Edge softness",
            description: "Soft gradient at the mask boundary during inference. Higher = smoother \
                lighting/color blend with surroundings, fewer visible seams. 0 = hard edge (faster, \
                more precise but visible boundary). 4-6 px is the typical SD recommendation.",
            tooltip: "Mask edge softness (px)",
        },
        &mut brush.sd_mask_blur,
        0.0..=MASK_BLUR_MAX,
        DEFAULT_MASK_BLUR,
        false,
        |v| if v < MASK_BLUR_OFF_THRESHOLD { "Off".to_string() } else { format!("{v:.0} px") },
    );
    change.commit
}

fn render_quality_preset_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_preset_popover");
    let active = SdQualityPreset::detect_from(brush);
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_AUTO_AWESOME.codepoint, active.label(), false),
        "Quality",
        "Picks scheduler + steps + CFG + Karras. Tweaking individual sliders flips to Custom.",
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new("Quality").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        for preset in [SdQualityPreset::Fast, SdQualityPreset::Balanced, SdQualityPreset::Quality] {
            let preset_scheduler = match preset {
                SdQualityPreset::Quality => SdScheduler::DpmPlusPlus2MKarras,
                _ => SdScheduler::Lcm,
            };
            let label = preset.label();
            if preset_scheduler.is_available() {
                let selected = active == preset;
                if ui.selectable_label(selected, label).clicked() {
                    preset.apply_to(brush);
                    changed = true;
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
            } else {
                ui.add_enabled_ui(false, |ui| {
                    let _ = ui.selectable_label(false, format!("{label} (coming soon)"));
                });
            }
        }
    });
    changed
}

fn render_prompt_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_prompt_popover");
    let lcm = brush.sd_scheduler == SdScheduler::Lcm;
    let neg_color = if lcm { theme::TEXT_SECONDARY } else { theme::TEXT_PRIMARY };

    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_EDIT_NOTE.codepoint, "Prompt", !brush.sd_prompt.is_empty()),
        "Prompt",
        "Text prompt + negative + guidance. Empty prompt = unconditional inpaint (often noisy on flat surrounds).",
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        ui.set_min_width(theme::POPOVER_WIDTH_WIDE);
        ui.label(RichText::new("Prompt").strong().color(theme::TEXT_PRIMARY));
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
                |v| if v <= 1.0 + 1e-3 { "off".to_string() } else { format!("{v:.1}") },
            );
            if cfg.commit { changed = true; }
            super::hint(ui, "Prompt strength. 1 = ignore prompt (single UNet pass). 7-8 = typical SD strength (UNet runs twice per step). Higher = closer match but oversaturated/burnt.");
        });

        if lcm {
            ui.add_space(theme::SPACE_SM);
            super::hint(ui, "LCM bakes guidance into training and ignores Negative + Guidance. Switch the Scheduler chip to DDIM or DPM++ to use them.");
        }

        ui.add_space(theme::SPACE_SM);
        ui.separator();
        ui.add_space(theme::SPACE_XS);
        let already_default = brush.sd_prompt == DEFAULT_SD_PROMPT
            && brush.sd_negative_prompt == DEFAULT_SD_NEGATIVE_PROMPT
            && (brush.sd_guidance_scale - default_cfg()).abs() < 1e-3;
        ui.add_enabled_ui(!already_default, |ui| {
            if chip::reset_button(ui, "Restore Prompt + Negative + Guidance to the shipped defaults.") {
                brush.sd_prompt = DEFAULT_SD_PROMPT.to_string();
                brush.sd_negative_prompt = DEFAULT_SD_NEGATIVE_PROMPT.to_string();
                brush.sd_guidance_scale = default_cfg();
                changed = true;
            }
        });
    });
    changed
}

fn render_scheduler_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_scheduler_popover");
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_TUNE.codepoint, brush.sd_scheduler.label(), false),
        "Scheduler",
        "Denoise math. LCM = fast (4-8 steps); DDIM = conservative; DPM++ 2M Karras = quality at 15-25 steps; Euler-A = creative per-seed variation; UniPC = best quality at 8-12 steps.",
    );
    let mut changed = false;
    chip::popup_for(ui, pop_id, &resp, |ui| {
        ui.label(RichText::new("Scheduler").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        for sched in [
            SdScheduler::Lcm,
            SdScheduler::Ddim,
            SdScheduler::DpmPlusPlus2MKarras,
            SdScheduler::UniPc,
            SdScheduler::EulerA,
        ] {
            let label = sched.label();
            let desc = sched.description();
            if sched.is_available() {
                let selected = brush.sd_scheduler == sched;
                if ui.selectable_label(selected, label).clicked() {
                    if brush.sd_scheduler != sched {
                        brush.sd_scheduler = sched;
                        changed = true;
                    }
                    egui::Popup::close_id(ui.ctx(), pop_id);
                }
                super::hint(ui, desc);
                ui.add_space(theme::SPACE_XS);
            } else {
                ui.add_enabled_ui(false, |ui| {
                    let _ = ui.selectable_label(false, format!("{label} (coming soon)"));
                });
                super::hint(ui, desc);
                ui.add_space(theme::SPACE_XS);
            }
        }
    });
    changed
}

fn render_steps_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let lcm = brush.sd_scheduler == SdScheduler::Lcm;
    // LCM caps at 8 by training (community consensus is no benefit
    // beyond 8); standard SD ranges 1-30. Clamp first to handle a
    // scheduler-switch that left a too-large step count behind.
    let max: u32 = if lcm { 8 } else { 30 };
    if brush.sd_steps > max { brush.sd_steps = max; }
    let default_steps: u32 = if lcm { 8 } else { 20 };
    let change = chip::chip_u32(
        ui,
        ChipMeta {
            id_salt: "eraser_steps",
            icon: ICON_REPLAY.codepoint,
            label: "Steps",
            description: "Denoise iteration count. LCM: 4-8 typical; Standard SD: 15-25.",
            tooltip: "Denoise iteration count",
        },
        &mut brush.sd_steps,
        1..=max,
        default_steps,
        |v| format!("{v}"),
    );
    change.commit
}

fn render_karras_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let resp = chip::icon_toggle_button(ui, ICON_BLUR_LINEAR.codepoint, brush.sd_use_karras_sigmas)
        .on_hover_text(
            "Karras sigma schedule. LCM was distilled against linear \
             spacing — Karras shifts the inference timestep distribution \
             away from training. Toggle to A/B compare on your content.",
        );
    if resp.clicked() {
        brush.sd_use_karras_sigmas = !brush.sd_use_karras_sigmas;
        return true;
    }
    false
}

fn render_seed_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pinned = brush.sd_seed.is_some();
    let icon = if pinned { ICON_LOCK.codepoint } else { ICON_LOCK_OPEN.codepoint };
    // Format the seed inline only when pinned — the unpinned arm
    // uses a `&'static str` to avoid allocating each frame.
    let pinned_label;
    let label: &str = match brush.sd_seed {
        Some(s) => {
            // Truncate to last 6 digits so the chip stays narrow.
            pinned_label = format!("…{:06}", s % 1_000_000);
            &pinned_label
        }
        None => "random",
    };
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, icon, label, pinned),
        "Seed",
        "Random by default — every stroke explores a different fill. Click to pin a single seed; the same prompt + scheduler + steps will then produce the exact same fill across strokes (useful for tweaking the prompt while comparing to a previous result, or for re-running an inpaint reproducibly).",
    );
    if resp.clicked() {
        brush.sd_seed = if pinned {
            None
        } else {
            Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0))
        };
        return true;
    }
    false
}
