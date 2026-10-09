//! Refinement chip row for the upscale toolbar.
//!
//! Mutations land on `item_settings` directly; the recipe diff at the
//! next dispatch decides whether to reprocess or update live.

use egui::Ui;
use egui_material_icons::icons::{
    ICON_BLUR_ON, ICON_BRIGHTNESS_6, ICON_DEBLUR, ICON_PALETTE, ICON_PSYCHOLOGY, ICON_TUNE,
};

use super::chip::{self, ChipMeta};
use super::fmt;
use crate::gui::item_settings::ItemSettings;
use crate::gui::theme;

pub(crate) fn render_refinement_row(ui: &mut Ui, item_settings: &mut ItemSettings) {
    ui.add_space(theme::SPACE_XS);
    ui.horizontal(|ui| {
        // Pre-process knobs (reprocess on change)
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_pre_denoise",
                icon: ICON_BLUR_ON.codepoint,
                label: "Pre-denoise",
                description: "Strength of the noise smoother. Higher values reduce fine detail along with the noise. Requires reprocessing.",
                tooltip: "Smooth chromatic speckles and grain before upscaling. Useful for low-light or high-ISO photos where the AI would otherwise amplify the noise. 0 leaves the image untouched.",
            },
            &mut item_settings.pre_denoise,
            0.0..=1.0,
            0.0,
            false,
            |v| fmt::off_or(v, f32::EPSILON, |v| fmt::plain(v, 2)),
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_brightness_lift",
                icon: ICON_BRIGHTNESS_6.codepoint,
                label: "Brightness Lift",
                description: "Pre-upscale exposure boost in EV stops (camera units). The inverse curve restores the original brightness after. Requires reprocessing.",
                tooltip: "Brighten dark areas before upscaling so the AI sees more shadow detail, then return to the original brightness. Especially helpful on underexposed photos. \u{00b1}2 EV range.",
            },
            &mut item_settings.brightness_lift,
            -2.0..=2.0,
            0.0,
            false,
            fmt::ev,
        );

        ui.add_space(4.0);

        // Post-process knobs (live, no reprocess)
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_sharpen",
                icon: ICON_DEBLUR.codepoint,
                label: "Sharpen",
                description: "Sharpening amount. Negative values blur, positive values sharpen.",
                tooltip: "Unsharp-mask pass on the upscaled image. Real-ESRGAN intentionally outputs slightly soft so you can choose how crisp the result looks; lift this above 0 to taste.",
            },
            &mut item_settings.sharpen,
            -1.0..=1.0,
            0.0,
            false,
            |v| fmt::off_or(v, f32::EPSILON, |v| fmt::signed_plain(v, 2)),
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_ai_blend",
                icon: ICON_PSYCHOLOGY.codepoint,
                label: "AI Blend",
                description: "Blend between the AI upscale (100%) and a plain enlargement (0%). Lower values dial back the AI's plastic look.",
                tooltip: "AI upscalers sometimes smooth faces, skin, or fine texture into a plastic look. Drop this below 100% to mix in the original photo's grain and detail. 100% is pure AI.",
            },
            &mut item_settings.ai_blend,
            0.0..=1.0,
            1.0,
            false,
            fmt::percent,
        );
        chip::chip_f32(
            ui,
            ChipMeta {
                id_salt: "refinement_saturation",
                icon: ICON_PALETTE.codepoint,
                label: "Saturation",
                description: "Colour intensity. Negative fades toward grey, positive saturates.",
                tooltip: "Boost or mute colour without touching brightness (HSL-based, so bright reds stay bright when you saturate). Range -1 to +1.",
            },
            &mut item_settings.saturation,
            -1.0..=1.0,
            0.0,
            false,
            |v| fmt::off_or(v, f32::EPSILON, |v| fmt::signed_plain(v, 2)),
        );
        chip::chip_bool(
            ui,
            ChipMeta {
                id_salt: "refinement_color_match",
                icon: ICON_TUNE.codepoint,
                label: "Color Match",
                description: "Snap the upscale's colour balance back to the source photo.",
                tooltip: "Some AI upscalers shift hues or saturation subtly during the run. This compares the upscale's overall colour statistics to the source image and corrects the drift, so the result matches the original tone.",
            },
            &mut item_settings.color_match,
        );
    });
}
