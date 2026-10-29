//! The Magic Brush panel: what its strip does not show.

use egui::Ui;

use crate::gui::brush_state::{BrushSettings, StrokeReach};
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::theme;
use crate::gui::views::chip;

use super::brush_chip::BrushChipOutcome;

/// `encoder_pending` shows a spinner while the encoder runs.
pub(crate) fn flyout_body(ui: &mut Ui, bs: &mut BrushSettings, reach: &mut StrokeReach, encoder_pending: bool, model: crate::gui::settings::SettingsModel) -> BrushChipOutcome {
    let mut outcome = BrushChipOutcome::default();
    chip::popover_header(ui, "Magic Brush", None);
    super::brush_chip::reach_row(ui, reach);

    if encoder_pending {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                egui::RichText::new(prunr_core::Step::LoadingMagicBrush.label())
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::FONT_SIZE_BODY),
            );
        });
        ui.add_space(6.0);
    }

    let specks = chip::toggle_row(ui, "Remove specks", &mut bs.magic_remove_specks);
    outcome.committed |= specks.commit;
    super::hint(ui, "Drop tiny islands and fill pinholes where the model is unsure.");
    outcome.committed |= super::brush_chip::fill_scale_row(ui, bs, model);
    if shows(model, Surface::SelectionStrokes) {
        outcome.committed |= super::brush_chip::render_shared_selection_section(ui, bs);
    }
    outcome
}
