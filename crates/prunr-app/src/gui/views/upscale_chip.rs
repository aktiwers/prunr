//! Output-scale chip widget for the upscale toolbar.
//!
//! Renders `[icon  {scale} · {w}×{h}]` as a chip button. Clicking opens
//! a popover with the three scales (2× / 3× / 4×) and a reset button.

use egui::Ui;
use egui_material_icons::icons::ICON_OPEN_IN_FULL;
use prunr_core::OutputScale;

use super::chip;

/// User-visible label for each OutputScale variant.
pub fn output_scale_label(s: OutputScale) -> &'static str {
    match s {
        OutputScale::X2 => "2\u{00d7}",
        OutputScale::X3 => "3\u{00d7}",
        OutputScale::X4 => "4\u{00d7}",
    }
}

/// Render the output-scale chip. Returns `true` if `value` changed.
///
/// `output_dims` is the projected output size at the current scale
/// (caller computes `(source.0 * factor, source.1 * factor)` using
/// `value.factor()`).
pub(crate) fn render_output_scale_chip(ui: &mut Ui, value: &mut OutputScale, output_dims: (u32, u32)) -> bool {
    let label = format!(
        "{} \u{00b7} {}\u{00d7}{}",
        output_scale_label(*value),
        output_dims.0,
        output_dims.1,
    );
    let accent = !matches!(*value, OutputScale::X4);
    let resp = chip::chip_button(ui, ICON_OPEN_IN_FULL.codepoint, &label, accent);
    let resp = chip::tooltip(
        resp,
        "Scale",
        "How much to enlarge the image. The models produce 4\u{00d7}; 2\u{00d7} and \
         3\u{00d7} resample that with Lanczos, except Real-ESRGAN, which runs its own \
         2\u{00d7} model for 2\u{00d7} when it is installed. Requires reprocessing.",
        None,
    );

    let mut changed = false;
    let popup_id = egui::Id::new("upscale_output_scale_popup");
    chip::popup_for(ui, popup_id, &resp, |ui| {
        if chip::popover_header(ui, "Scale", Some(("Back to 4\u{00d7}, the model's native size", *value == OutputScale::X4))) {
            *value = OutputScale::X4;
            changed = true;
        }
        for option in [OutputScale::X2, OutputScale::X3, OutputScale::X4] {
            if chip::picker_row(ui, *value == option, output_scale_label(option), "").clicked() && *value != option {
                *value = option;
                changed = true;
                egui::Popup::close_id(ui.ctx(), popup_id);
            }
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_scale_label_pinned() {
        assert_eq!(output_scale_label(OutputScale::X2), "2\u{00d7}");
        assert_eq!(output_scale_label(OutputScale::X3), "3\u{00d7}");
        assert_eq!(output_scale_label(OutputScale::X4), "4\u{00d7}");
    }

    #[test]
    fn output_scale_factor_via_method() {
        assert_eq!(OutputScale::X2.factor(), 2);
        assert_eq!(OutputScale::X3.factor(), 3);
        assert_eq!(OutputScale::X4.factor(), 4);
    }
}
