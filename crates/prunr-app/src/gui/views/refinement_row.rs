//! Refinement chip row for the upscale toolbar.
//!
//! Renders six chips (pre-denoise, brightness-lift, sharpen, ai-blend,
//! saturation, color-match) in a dedicated row below `render_upscale_row`.
//! The output-scale chip ships in `upscale_chip.rs` and is rendered by
//! `upscale_toolbar.rs`; this file covers the other six.
//!
//! Tier-1 chips (re-run inference): pre-denoise, brightness-lift.
//! Tier-2 chips (real-time): sharpen, ai-blend, saturation, color-match.
//!
//! A 4-px gap between brightness-lift and sharpen visually groups the
//! two tiers without a separator line.
//!
//! Conventions match the other adjustment rows (mask / edge / lines):
//!   - `chip::chip_button` for every chip face
//!   - `chip::popup_for` for popovers
//!   - `chip::slider_row_f32` for slider rows inside the popover
//!   - `chip::chip_tooltip` for hover tooltips
//!   - `chip::reset_button` for reset-to-default
//!
//! Mutations update `item_settings.<field>` directly — the recipe diff
//! at dispatch time picks the change up; no per-chip ToolbarChange signal
//! needed (UpscaleTier2/Tier-1 are derived by the diff).

use egui::{RichText, Ui};
use egui_material_icons::icons::{
    ICON_BLUR_ON, ICON_BRIGHTNESS_6, ICON_DEBLUR, ICON_PALETTE, ICON_PSYCHOLOGY, ICON_TUNE,
};

use super::chip;
use super::hint;
use crate::gui::item_settings::ItemSettings;
use crate::gui::theme;

/// Render the Refinement row (six chips). Returns `true` if any value changed.
///
/// Caller uses the return value to schedule a repaint; recipe-diff at
/// the next dispatch tick determines the actual tier (Tier-1 vs Tier-2).
pub(crate) fn render_refinement_row(ui: &mut Ui, item_settings: &mut ItemSettings) -> bool {
    let mut changed = false;
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        // Tier-1 chips (re-run inference on change)
        changed |= render_pre_denoise_chip(ui, &mut item_settings.pre_denoise);
        changed |= render_brightness_lift_chip(ui, &mut item_settings.brightness_lift);

        // Visual gap separating Tier-1 from Tier-2 chips
        ui.add_space(4.0);

        // Tier-2 chips (real-time postprocess)
        changed |= render_sharpen_chip(ui, &mut item_settings.sharpen);
        changed |= render_ai_blend_chip(ui, &mut item_settings.ai_blend);
        changed |= render_saturation_chip(ui, &mut item_settings.saturation);
        changed |= render_color_match_chip(ui, &mut item_settings.color_match);
    });
    changed
}

// ── Individual chip helpers ──────────────────────────────────────────────────

fn render_pre_denoise_chip(ui: &mut Ui, value: &mut f32) -> bool {
    let label = if *value == 0.0 {
        "Off".to_string()
    } else {
        format!("{:.2}", *value)
    };
    let accent = *value != 0.0;
    let resp = chip::chip_button(ui, ICON_BLUR_ON.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Pre-denoise",
        "Classical denoise applied BEFORE inference. Median + bilateral. \
         Kills chromatic dot speckles from low-light input. 0.0 = off. \
         Re-runs inference on change (Tier-1).",
    );
    let mut changed = false;
    chip::popup_for(ui, egui::Id::new("refinement_pre_denoise"), &resp, |ui| {
        ui.label(RichText::new("Pre-denoise").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let r = chip::slider_row_f32(ui, "Strength", value, 0.0..=1.0, false, |v| {
            if v == 0.0 {
                "off".to_string()
            } else {
                format!("{:.2}", v)
            }
        });
        if r.changed {
            changed = true;
        }
        ui.add_space(theme::SPACE_XS);
        hint(ui, "Re-runs inference on change (Tier-1).");
        ui.add_space(theme::SPACE_XS);
        if chip::reset_button(ui, "Reset to off") && *value != 0.0 {
            *value = 0.0;
            changed = true;
        }
    });
    changed
}

fn render_brightness_lift_chip(ui: &mut Ui, value: &mut f32) -> bool {
    let label = if *value == 0.0 {
        "0 EV".to_string()
    } else {
        format!("{:+.1} EV", *value)
    };
    let accent = *value != 0.0;
    let resp = chip::chip_button(ui, ICON_BRIGHTNESS_6.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Brightness Lift",
        "Pre-inference exposure adjustment in EV stops. Lifts dark inputs \
         so the model has more signal; reciprocal tone-map applied post-\
         inference. 0 = off. Range: -2 to +2 EV. Re-runs inference on change (Tier-1).",
    );
    let mut changed = false;
    chip::popup_for(
        ui,
        egui::Id::new("refinement_brightness_lift"),
        &resp,
        |ui| {
            ui.label(
                RichText::new("Brightness Lift")
                    .strong()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_XS);
            let r = chip::slider_row_f32(ui, "EV stops", value, -2.0..=2.0, false, |v| {
                if v == 0.0 {
                    "0 EV".to_string()
                } else {
                    format!("{:+.1} EV", v)
                }
            });
            if r.changed {
                changed = true;
            }
            ui.add_space(theme::SPACE_XS);
            hint(ui, "Re-runs inference on change (Tier-1).");
            ui.add_space(theme::SPACE_XS);
            if chip::reset_button(ui, "Reset to 0 EV") && *value != 0.0 {
                *value = 0.0;
                changed = true;
            }
        },
    );
    changed
}

fn render_sharpen_chip(ui: &mut Ui, value: &mut f32) -> bool {
    let label = if *value == 0.0 {
        "Off".to_string()
    } else {
        format!("{:+.2}", *value)
    };
    let accent = *value != 0.0;
    let resp = chip::chip_button(ui, ICON_DEBLUR.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Sharpen",
        "Unsharp mask applied after inference. Negative = blur, 0 = off, \
         positive = sharpen. Real-time (Tier-2). Range: -1 to +1.",
    );
    let mut changed = false;
    chip::popup_for(ui, egui::Id::new("refinement_sharpen"), &resp, |ui| {
        ui.label(RichText::new("Sharpen").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let r = chip::slider_row_f32(ui, "Amount", value, -1.0..=1.0, false, |v| {
            if v == 0.0 {
                "off".to_string()
            } else {
                format!("{:+.2}", v)
            }
        });
        if r.changed {
            changed = true;
        }
        ui.add_space(theme::SPACE_XS);
        hint(ui, "Real-time (Tier-2) \u{2014} no re-inference.");
        ui.add_space(theme::SPACE_XS);
        if chip::reset_button(ui, "Reset to off") && *value != 0.0 {
            *value = 0.0;
            changed = true;
        }
    });
    changed
}

fn render_ai_blend_chip(ui: &mut Ui, value: &mut f32) -> bool {
    let label = if *value >= 1.0 - f32::EPSILON {
        "Full AI".to_string()
    } else {
        format!("{:.0}% AI", *value * 100.0)
    };
    let accent = *value < 1.0 - f32::EPSILON;
    let resp = chip::chip_button(ui, ICON_PSYCHOLOGY.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "AI Blend",
        "Lerp between AI upscale output (1.0 = full AI) and a bicubic-of-\
         source baseline (0.0 = pure bicubic). Ease back the plastic look \
         from over-aggressive AI smoothing. Real-time (Tier-2).",
    );
    let mut changed = false;
    chip::popup_for(ui, egui::Id::new("refinement_ai_blend"), &resp, |ui| {
        ui.label(RichText::new("AI Blend").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);
        let r = chip::slider_row_f32(ui, "AI weight", value, 0.0..=1.0, false, |v| {
            if v >= 1.0 - f32::EPSILON {
                "Full AI".to_string()
            } else {
                format!("{:.0}% AI", v * 100.0)
            }
        });
        if r.changed {
            changed = true;
        }
        ui.add_space(theme::SPACE_XS);
        hint(ui, "1.0 = full AI, 0.0 = pure bicubic. Real-time (Tier-2).");
        ui.add_space(theme::SPACE_XS);
        if chip::reset_button(ui, "Reset to full AI") && *value != 1.0 {
            *value = 1.0;
            changed = true;
        }
    });
    changed
}

fn render_saturation_chip(ui: &mut Ui, value: &mut f32) -> bool {
    let label = if *value == 0.0 {
        "Off".to_string()
    } else {
        format!("{:+.2}", *value)
    };
    let accent = *value != 0.0;
    let resp = chip::chip_button(ui, ICON_PALETTE.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Saturation",
        "HSL-space saturation adjustment. Negative = desaturate toward \
         grey, 0 = off, positive = saturate. Real-time (Tier-2).",
    );
    let mut changed = false;
    chip::popup_for(ui, egui::Id::new("refinement_saturation"), &resp, |ui| {
        ui.label(
            RichText::new("Saturation")
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        ui.add_space(theme::SPACE_XS);
        let r = chip::slider_row_f32(ui, "Amount", value, -1.0..=1.0, false, |v| {
            if v == 0.0 {
                "off".to_string()
            } else {
                format!("{:+.2}", v)
            }
        });
        if r.changed {
            changed = true;
        }
        ui.add_space(theme::SPACE_XS);
        hint(ui, "Real-time (Tier-2) \u{2014} no re-inference.");
        ui.add_space(theme::SPACE_XS);
        if chip::reset_button(ui, "Reset to off") && *value != 0.0 {
            *value = 0.0;
            changed = true;
        }
    });
    changed
}

fn render_color_match_chip(ui: &mut Ui, value: &mut bool) -> bool {
    let label = if *value { "On" } else { "Off" };
    let resp = chip::chip_button(ui, ICON_TUNE.codepoint, label, *value);
    let resp = chip::chip_tooltip(
        resp,
        "Color Match",
        "Reinhard Lab-space mean+stddev color transfer from the source \
         image. Snaps the upscale's color statistics back to source \u{2014} \
         cancels model color drift. Real-time (Tier-2).",
    );
    let mut changed = false;
    chip::popup_for(ui, egui::Id::new("refinement_color_match"), &resp, |ui| {
        ui.label(
            RichText::new("Color Match")
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        ui.add_space(theme::SPACE_XS);
        let before = *value;
        ui.checkbox(value, "Match source colors");
        if before != *value {
            changed = true;
        }
        ui.add_space(theme::SPACE_XS);
        hint(
            ui,
            "Transfers source RGB mean+stddev in Lab space. Real-time (Tier-2).",
        );
        ui.add_space(theme::SPACE_XS);
        if chip::reset_button(ui, "Reset to off") && *value {
            *value = false;
            changed = true;
        }
    });
    changed
}
