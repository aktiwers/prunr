//! The Magic Brush panel: the shared cursor block and selection look,
//! plus the Confidence threshold. Opens from the tool strip.

use egui::Ui;

use crate::gui::brush_state::BrushSettings;
use crate::gui::theme;
use crate::gui::views::{chip, fmt};

/// Minimum popover width so the two-column cursor block fits.
const MAGIC_POPOVER_MIN_WIDTH: f32 = 320.0;

/// Outcome reported back to the toolbar's change accumulator.
#[derive(Default, Clone, Copy)]
pub(crate) struct MagicChipOutcome {
    /// True on slider release / toggle — caller persists brush settings.
    pub(crate) committed: bool,
    /// New value of `Settings::protect_selection` when its switch flipped.
    pub(crate) protect_selection: Option<bool>,
}

/// The full panel behind the strip's dots. `encoder_pending` shows the
/// "Preparing…" spinner while the encoder runs.
pub(crate) fn flyout_body(
    ui: &mut Ui,
    bs: &mut BrushSettings,
    encoder_pending: bool,
    protect: Option<bool>,
) -> MagicChipOutcome {
    let mut outcome = MagicChipOutcome::default();
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

    // One brush, two tools: the cursor block is shared with Paint.
    outcome.committed |= super::brush_chip::render_cursor_section(ui, bs);
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    outcome.committed |= super::brush_chip::render_shared_selection_section(ui, bs);
    outcome.protect_selection = super::brush_chip::render_auto_apply_row(ui, protect);

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
