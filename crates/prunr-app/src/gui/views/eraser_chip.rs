//! SD-eraser toolbar chips: Quality preset, Scheduler, Steps,
//! Karras toggle, Seed pin. Renders as a horizontal cluster on the
//! inpaint-mode Row 2 — dropdown chips mirror `lines_popover`'s
//! pattern (chip-button + popover with selectable rows).
//!
//! Schedulers without a dispatch backend wired yet appear as greyed
//! "(coming soon)" entries in the dropdowns. UI gating belt-and-
//! braces with the `is_available()` check inside dispatch.

use egui::RichText;
use egui_material_icons::icons::*;

use crate::gui::brush_state::{BrushSettings, SdQualityPreset, SdScheduler};
use crate::gui::theme;

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
        change.committed |= render_karras_chip(ui, brush);
        change.committed |= render_seed_chip(ui, brush);
    });
    change
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

fn render_scheduler_chip(ui: &mut egui::Ui, brush: &mut BrushSettings) -> bool {
    let pop_id = egui::Id::new("eraser_scheduler_popover");
    let resp = chip::chip_tooltip(
        chip::chip_button(ui, ICON_TUNE.codepoint, brush.sd_scheduler.label(), false),
        "Scheduler",
        "Denoise math. LCM = fast (4-8 steps); DDIM = conservative; DPM++ / UniPC / Euler-A coming soon.",
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
            if sched.is_available() {
                let selected = brush.sd_scheduler == sched;
                if ui.selectable_label(selected, label).clicked() {
                    if brush.sd_scheduler != sched {
                        brush.sd_scheduler = sched;
                        changed = true;
                    }
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

fn render_karras_chip(ui: &mut egui::Ui, brush: &BrushSettings) -> bool {
    // Karras schedule is orthogonal to scheduler choice (DDIM /
    // DPM++ / UniPC / Euler-A all support it; LCM has its own
    // fixed schedule and ignores the toggle). Greyed until the
    // Karras sigma helper has a dispatch backend.
    //
    // Pure placeholder: render disabled-visual + tooltip but never
    // mutate. egui's `add_enabled_ui` greys the appearance but
    // doesn't suppress `Response::clicked()` for every widget type;
    // unconditional `return false` is the correctness guarantee.
    ui.add_enabled_ui(false, |ui| {
        chip::icon_toggle_button(ui, ICON_BLUR_LINEAR.codepoint, brush.sd_use_karras_sigmas)
            .on_hover_text("Karras sigma schedule (coming soon)");
    });
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
        "Click to pin / un-pin the RNG seed. Pinned = same output across strokes; useful for A/B testing prompts.",
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
