//! Refinement chip row for the upscale toolbar.
//!
//! Mutations land on `item_settings` directly; the recipe diff at the
//! next dispatch decides Tier-1 (re-inference) vs Tier-2 (real-time).

use egui::Ui;
use egui_material_icons::icons::{
    ICON_BLUR_ON, ICON_BRIGHTNESS_6, ICON_DEBLUR, ICON_PALETTE, ICON_PSYCHOLOGY, ICON_TUNE,
};

use super::chip::{self, ChipMeta};
use crate::gui::item_settings::ItemSettings;
use crate::gui::theme;

pub(crate) fn render_refinement_row(ui: &mut Ui, item_settings: &mut ItemSettings) {
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        // Tier-1 (re-runs inference)
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_pre_denoise",
                icon: ICON_BLUR_ON.codepoint,
                label: "Pre-denoise",
                description: "Re-runs inference on change (Tier-1).",
                tooltip: "Classical denoise applied BEFORE inference. Median + bilateral. \
                          Kills chromatic dot speckles from low-light input. 0.0 = off.",
            },
            &mut item_settings.pre_denoise,
            0.0..=1.0,
            0.0,
            false,
            |v| if v == 0.0 { "Off".to_string() } else { format!("{:.2}", v) },
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_brightness_lift",
                icon: ICON_BRIGHTNESS_6.codepoint,
                label: "Brightness Lift",
                description: "Re-runs inference on change (Tier-1).",
                tooltip: "Pre-inference exposure adjustment in EV stops. Lifts dark inputs \
                          so the model has more signal; reciprocal tone-map applied post-\
                          inference. 0 = off. Range: -2 to +2 EV.",
            },
            &mut item_settings.brightness_lift,
            -2.0..=2.0,
            0.0,
            false,
            |v| if v == 0.0 { "0 EV".to_string() } else { format!("{:+.1} EV", v) },
        );

        ui.add_space(4.0);

        // Tier-2 (real-time postprocess)
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_sharpen",
                icon: ICON_DEBLUR.codepoint,
                label: "Sharpen",
                description: "Real-time (Tier-2) \u{2014} no re-inference.",
                tooltip: "Unsharp mask applied after inference. Negative = blur, 0 = off, \
                          positive = sharpen. Range: -1 to +1.",
            },
            &mut item_settings.sharpen,
            -1.0..=1.0,
            0.0,
            false,
            |v| if v == 0.0 { "Off".to_string() } else { format!("{:+.2}", v) },
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_ai_blend",
                icon: ICON_PSYCHOLOGY.codepoint,
                label: "AI Blend",
                description: "1.0 = full AI, 0.0 = pure bicubic. Real-time (Tier-2).",
                tooltip: "Lerp between AI upscale output (1.0 = full AI) and a bicubic-of-\
                          source baseline (0.0 = pure bicubic). Ease back the plastic look \
                          from over-aggressive AI smoothing.",
            },
            &mut item_settings.ai_blend,
            0.0..=1.0,
            1.0,
            false,
            |v| if v >= 1.0 - f32::EPSILON { "Full AI".to_string() } else { format!("{:.0}% AI", v * 100.0) },
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_saturation",
                icon: ICON_PALETTE.codepoint,
                label: "Saturation",
                description: "Real-time (Tier-2) \u{2014} no re-inference.",
                tooltip: "HSL-space saturation adjustment. Negative = desaturate toward \
                          grey, 0 = off, positive = saturate.",
            },
            &mut item_settings.saturation,
            -1.0..=1.0,
            0.0,
            false,
            |v| if v == 0.0 { "Off".to_string() } else { format!("{:+.2}", v) },
        );
        chip::chip_bool(
            ui,
            ChipMeta {
                id_salt: "refinement_color_match",
                icon: ICON_TUNE.codepoint,
                label: "Color Match",
                description: "Transfers source RGB mean+stddev in Lab space. Real-time (Tier-2).",
                tooltip: "Reinhard Lab-space mean+stddev color transfer from the source \
                          image. Snaps the upscale's color statistics back to source \u{2014} \
                          cancels model color drift.",
            },
            &mut item_settings.color_match,
        );
    });
}
