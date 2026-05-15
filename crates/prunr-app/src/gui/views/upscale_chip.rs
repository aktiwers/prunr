//! Output-scale chip widget for the upscale toolbar.
//!
//! Renders `[icon  {scale} · {w}×{h}]` as a chip button. Clicking opens
//! a popover with four selectable rows (2× / 3× / 4× / 4× (two-pass)) and a
//! reset button. The X4TwoPass row dims when the active model is not
//! Real-ESRGAN x4plus (Nomos8k has no native 2× variant).

use egui::{RichText, Ui};
use egui_material_icons::icons::ICON_OPEN_IN_FULL;
use prunr_core::upscale::x4twopass_available;
use prunr_core::OutputScale;
use prunr_models::ModelId;

use super::chip;
use super::hint;
use crate::gui::theme;

/// User-visible label for each OutputScale variant.
/// Single source of truth — both the chip face and the popover's
/// selectable_label rows read this function.
pub fn output_scale_label(s: OutputScale) -> &'static str {
    match s {
        OutputScale::X2 => "2\u{00d7}",
        OutputScale::X3 => "3\u{00d7}",
        OutputScale::X4 => "4\u{00d7}",
        OutputScale::X4TwoPass => "4\u{00d7} (two-pass)",
    }
}

/// Render the output-scale chip. Returns `true` if `value` changed.
///
/// `output_dims` is the projected output size at the current scale
/// (caller computes `(source.0 * factor, source.1 * factor)` using
/// `value.factor()`). `model_id` drives X4TwoPass availability —
/// dimmed for non-x4plus models.
pub(crate) fn render_output_scale_chip(
    ui: &mut Ui,
    value: &mut OutputScale,
    output_dims: (u32, u32),
    model_id: ModelId,
) -> bool {
    let label = format!(
        "{} \u{00b7} {}\u{00d7}{}",
        output_scale_label(*value),
        output_dims.0,
        output_dims.1,
    );
    let accent = !matches!(*value, OutputScale::X4);
    let resp = chip::chip_button(ui, ICON_OPEN_IN_FULL.codepoint, &label, accent);
    let resp = chip::chip_tooltip(
        resp,
        "Output Scale",
        "Output size relative to the source. 4\u{00d7} is native model output. \
         2\u{00d7} and 3\u{00d7} run the model at 4\u{00d7} then downscale with Lanczos3. \
         4\u{00d7} (two-pass) runs Real-ESRGAN x2plus twice \u{2014} better for \
         noisy / low-light input. Available with Real-ESRGAN only.",
    );

    let mut changed = false;
    let popup_id = egui::Id::new("upscale_output_scale_popup");
    chip::popup_for(ui, popup_id, &resp, |ui| {
        ui.label(RichText::new("Output Scale").strong().color(theme::TEXT_PRIMARY));
        ui.add_space(theme::SPACE_XS);

        for &option in &[
            OutputScale::X2,
            OutputScale::X3,
            OutputScale::X4,
            OutputScale::X4TwoPass,
        ] {
            let available = match option {
                OutputScale::X4TwoPass => x4twopass_available(model_id),
                _ => true,
            };
            let row_label = if matches!(option, OutputScale::X4TwoPass) && !available {
                "4\u{00d7} (two-pass) \u{2014} Real-ESRGAN only"
            } else {
                output_scale_label(option)
            };
            ui.add_enabled_ui(available, |ui| {
                if ui
                    .selectable_label(*value == option, row_label)
                    .clicked()
                    && *value != option
                {
                    *value = option;
                    changed = true;
                }
            });
        }

        ui.add_space(theme::SPACE_XS);
        hint(ui, "4\u{00d7} (two-pass) handles noise better but uses Real-ESRGAN x2plus internally. Nomos8k cannot use this variant.");
        ui.add_space(theme::SPACE_XS);

        if chip::reset_button(ui, "Reset to 4\u{00d7} (native)") && *value != OutputScale::X4 {
            *value = OutputScale::X4;
            changed = true;
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_core::upscale::x4twopass_available;

    #[test]
    fn x4twopass_available_only_for_real_esrgan() {
        assert!(x4twopass_available(ModelId::RealEsrganX4Plus));
        assert!(!x4twopass_available(ModelId::Nomos8kSchatL));
        assert!(!x4twopass_available(ModelId::Silueta));
    }

    #[test]
    fn output_scale_label_pinned() {
        assert_eq!(output_scale_label(OutputScale::X2), "2\u{00d7}");
        assert_eq!(output_scale_label(OutputScale::X3), "3\u{00d7}");
        assert_eq!(output_scale_label(OutputScale::X4), "4\u{00d7}");
        assert_eq!(output_scale_label(OutputScale::X4TwoPass), "4\u{00d7} (two-pass)");
    }

    #[test]
    fn output_scale_factor_via_method() {
        assert_eq!(OutputScale::X2.factor(), 2);
        assert_eq!(OutputScale::X3.factor(), 3);
        assert_eq!(OutputScale::X4.factor(), 4);
        assert_eq!(OutputScale::X4TwoPass.factor(), 4);
    }
}
