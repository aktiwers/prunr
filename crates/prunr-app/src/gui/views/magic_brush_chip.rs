//! The Magic Brush panel: the shared cursor block and selection look,
//! plus the Confidence threshold.

use egui::Ui;

use crate::gui::brush_state::BrushSettings;
use crate::gui::theme;
use crate::gui::views::{chip, fmt};

use super::brush_chip::BrushChipOutcome;

/// Wider than a popover so the two-column cursor block fits.
const MAGIC_POPOVER_MIN_WIDTH: f32 = 320.0;

/// `encoder_pending` shows the "Preparing…" spinner while the encoder runs.
pub(crate) fn flyout_body(ui: &mut Ui, bs: &mut BrushSettings, encoder_pending: bool) -> BrushChipOutcome {
    let mut outcome = BrushChipOutcome::default();
    ui.set_min_width(MAGIC_POPOVER_MIN_WIDTH);
    chip::popover_header(ui, "Magic Brush", None);

    if encoder_pending {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                egui::RichText::new("Preparing\u{2026}")
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_BODY),
            );
        });
        ui.add_space(6.0);
    }

    outcome.committed |= super::brush_chip::render_cursor_section(ui, bs);
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    outcome.committed |= super::brush_chip::render_shared_selection_section(ui, bs);

    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    ui.label(
        egui::RichText::new("Magic Brush")
            .strong()
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_SIZE_BODY),
    );
    ui.add_space(2.0);

    let ct = chip::slider_row_f32(ui, "Confidence", &mut bs.magic_confidence_threshold, 0.0..=1.0, false, fmt::percent);
    outcome.committed |= ct.commit;
    super::hint(ui, "Keep only the parts the model is sure about. Lower selects more.");
    outcome
}
