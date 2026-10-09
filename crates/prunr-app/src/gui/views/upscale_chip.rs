//! Output-scale chip widget for the upscale toolbar.
//!
//! Renders `[icon  {scale} · {w}×{h}]` as a chip button. Clicking opens
//! a popover with four selectable rows (2× / 3× / 4× / 4× (two-pass)) and a
//! reset button. The X4TwoPass row dims when the active model is not
//! Real-ESRGAN x4plus (Nomos8k has no native 2× variant).

use egui::Ui;
use egui_material_icons::icons::ICON_OPEN_IN_FULL;
use prunr_core::upscale::x4twopass_available;
use prunr_core::OutputScale;
use prunr_models::ModelId;

use super::chip;
use super::hint;
use crate::gui::theme;

/// User-visible label for each OutputScale variant.
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
/// `value.factor()`). `model_id` drives X4TwoPass architectural
/// availability; `is_x2plus_installed` gates the install-state half —
/// both must hold for the X4TwoPass row to be selectable.
pub(crate) fn render_output_scale_chip(
    ui: &mut Ui,
    value: &mut OutputScale,
    output_dims: (u32, u32),
    model_id: ModelId,
    is_x2plus_installed: bool,
) -> bool {
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
        "How much to enlarge the image. Most models produce 4\u{00d7}; other \
         factors resample that output with high-quality Lanczos. 4\u{00d7} \
         (two-pass) runs the 2\u{00d7} model twice for cleaner results on noisy \
         or low-light photos \u{2014} Real-ESRGAN only.",
    None,
);

    let mut changed = false;
    let popup_id = egui::Id::new("upscale_output_scale_popup");
    chip::popup_for(ui, popup_id, &resp, |ui| {
        if chip::popover_header(ui, "Scale", Some(("Back to 4\u{00d7}, the model's native size", *value == OutputScale::X4))) {
            *value = OutputScale::X4;
            changed = true;
        }

        let arch_ok = x4twopass_available(model_id);
        for &option in &[
            OutputScale::X2,
            OutputScale::X3,
            OutputScale::X4,
            OutputScale::X4TwoPass,
        ] {
            let reason = match option {
                OutputScale::X4TwoPass if !arch_ok => Some("Only the Real-ESRGAN model can run two passes."),
                OutputScale::X4TwoPass if !is_x2plus_installed => Some("Install Real-ESRGAN x2plus from the Model Store first."),
                _ => None,
            };
            let clicked = chip::gated(ui, reason, |ui| {
                chip::picker_row(ui, *value == option, output_scale_label(option), "").clicked()
            });
            if clicked && *value != option {
                *value = option;
                changed = true;
            }
        }

        ui.add_space(theme::SPACE_XS);
        hint(ui, "Two-pass runs the upscaler twice for a cleaner result on noisy photos, in about twice the time.");
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
