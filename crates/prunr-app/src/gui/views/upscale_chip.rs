//! Scale chip widget for the upscale toolbar.
//!
//! Renders `[icon  {scale}x · {w}×{h}]` as a chip button. Clicking opens
//! a popover with two `selectable_label` rows (4x / 2x) and a reset button.
//! Returns `true` when the caller's `scale` value changed so the upstream
//! `ToolbarChange` can track the edit.

use egui::{RichText, Ui};
use egui_material_icons::icons::ICON_OPEN_IN_FULL;
use prunr_core::OutputScale;

use super::chip;
use super::hint;
use crate::gui::theme;

fn output_scale_to_factor(scale: OutputScale) -> u32 {
    match scale {
        OutputScale::X2 => 2,
        OutputScale::X3 => 3,
        OutputScale::X4 | OutputScale::X4TwoPass => 4,
    }
}

/// Render the scale chip. Returns `true` if `scale` changed.
///
/// `output_dims` is `(w, h)` of the upscale's projected output —
/// computed by the caller from source dimensions × scale factor.
pub(crate) fn render_scale_chip(
    ui: &mut Ui,
    scale: &mut OutputScale,
    output_dims: (u32, u32),
) -> bool {
    let factor = output_scale_to_factor(*scale);
    let label = format!("{}x · {}×{}", factor, output_dims.0, output_dims.1);
    let accent = *scale != OutputScale::X4;
    let resp = chip::chip_button(ui, ICON_OPEN_IN_FULL.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Scale",
        "Output size after upscaling. 4x is native model quality. 2x runs at 4x then downscales with Lanczos3.",
    );

    let mut changed = false;
    let popup_id = egui::Id::new("upscale_scale_popup");
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.label(RichText::new("Scale").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);

        if ui
            .selectable_label(*scale == OutputScale::X4, "4x  —  native model output")
            .clicked()
            && *scale != OutputScale::X4
        {
            *scale = OutputScale::X4;
            changed = true;
        }
        if ui
            .selectable_label(*scale == OutputScale::X2, "2x  —  Lanczos3 downscale after model")
            .clicked()
            && *scale != OutputScale::X2
        {
            *scale = OutputScale::X2;
            changed = true;
        }

        ui.add_space(theme::SPACE_XS);
        hint(ui, "4x is the native model output. 2x runs at 4x internally then halves the result.");
        ui.add_space(theme::SPACE_XS);

        if chip::reset_button(ui, "Reset to 4x (native)") && *scale != OutputScale::X4 {
            *scale = OutputScale::X4;
            changed = true;
        }
    });
    changed
}
