//! Magic Brush settings chip + popover.
//!
//! Rendered next to the [ Magic ] toggle while Magic Brush mode is
//! on. Contains the shared Selection section (delegated to the brush-chip
//! helper) plus a Magic-only Confidence threshold slider.
//!
//! The "Preparing..." spinner shows in the popover while `encoder_pending`
//! is true; the canvas overlay (separate render path) shows the same.

use egui::Ui;
use egui_material_icons::icons::ICON_AUTO_AWESOME;

use crate::gui::brush_state::BrushSettings;
use crate::gui::theme;
use crate::gui::views::{chip, fmt};

/// Width budget for the chip label, padded so 1- to 3-digit sizes don't
/// reflow the popover anchor while the slider is dragged.
const LABEL_PAD_WIDTH: f32 = 72.0;

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

/// Render the Magic Brush chip button + popover.
///
/// `encoder_pending` — show "Preparing…" spinner when true (encoder in flight).
/// Returns `MagicChipOutcome` so the caller can flush brush settings.
pub(crate) fn render(
    ui: &mut Ui,
    bs: &mut BrushSettings,
    encoder_pending: bool,
    protect: Option<bool>,
) -> MagicChipOutcome {
    let label = format!("{:>3} px", bs.radius as u32);
    let resp = ui
        .scope(|ui| {
            ui.set_min_width(LABEL_PAD_WIDTH);
            chip::chip_button(ui, ICON_AUTO_AWESOME.codepoint, &label, /*accent=*/ true)
        })
        .inner;
    let resp = chip::tooltip(
        resp,
        "Magic Brush settings",
        "Cursor size and shape, confidence, and how the selection is shown.",
        None,
    );

    let mut outcome = MagicChipOutcome::default();
    chip::popup_for(
        ui,
        egui::Id::new("magic_brush_chip_popup"),
        &resp,
        |ui| {
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

            let sel_committed = super::brush_chip::render_shared_selection_section(ui, bs);
            outcome.committed |= sel_committed;
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

            let ct = chip::slider_row_f32(
                ui,
                "Confidence",
                &mut bs.magic_confidence_threshold,
                0.0..=1.0,
                false,
                fmt::percent,
            );
            outcome.committed |= ct.commit;
            super::hint(
                ui,
                "Keep only the parts the model is sure about. Lower selects more.",
            );
        },
    );

    outcome
}
