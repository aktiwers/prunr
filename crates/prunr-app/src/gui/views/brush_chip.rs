//! The Paint Brush panel (what its strip does not show), and the brush
//! knobs the strip and the panel share.

use egui::Ui;

use crate::gui::brush_state::BrushSettings;
use crate::gui::theme;
use crate::gui::knob_catalog::{shows, Surface};
use crate::gui::settings::SettingsModel;
use prunr_core::brush::BrushShape;
use prunr_core::selection::BrushMode;

use super::chip;
use super::fmt;

/// What a brush panel or the tool strip changed this frame.
#[derive(Default, Clone, Copy)]
pub(crate) struct BrushChipOutcome {
    pub reset_brush_requested: bool,
    /// A slider settled or a choice was made: the brush settings persist.
    pub committed: bool,
}

impl BrushChipOutcome {
    pub fn merge(&mut self, other: Self) {
        self.reset_brush_requested |= other.reset_brush_requested;
        self.committed |= other.committed;
    }
}

/// On a background-removal model a stroke edits the cut-out.
pub(super) const CUTOUT_MODES: [chip::Choice<BrushMode>; 2] = [
    chip::Choice { value: BrushMode::Add, name: "Restore", description: "Bring back what the model took away", enabled: true },
    chip::Choice { value: BrushMode::Subtract, name: "Erase", description: "Remove what the model left behind", enabled: true },
];

/// Elsewhere a stroke shapes the selection or the region to fill.
pub(super) const REGION_MODES: [chip::Choice<BrushMode>; 2] = [
    chip::Choice { value: BrushMode::Add, name: "Add", description: "Add to the selection", enabled: true },
    chip::Choice { value: BrushMode::Subtract, name: "Subtract", description: "Take away from the selection", enabled: true },
];

/// The mode switch; Erase on a cut-out is picked in warm red, as its
/// strokes are drawn.
pub(super) fn mode_buttons(ui: &mut Ui, model: SettingsModel, mode: &mut BrushMode) -> bool {
    let picked = if model.erases_with(*mode) { theme::ERASE_PICKED } else { theme::ACCENT_BRIGHT };
    chip::choice_buttons_picked(ui, mode_choices(model), mode, picked)
}

/// Restore / Erase on a cut-out; Add / Subtract on a selection or region.
pub(super) fn mode_choices(model: SettingsModel) -> &'static [chip::Choice<BrushMode>; 2] {
    if model.edits_cutout() { &CUTOUT_MODES } else { &REGION_MODES }
}

const BRUSH_SHAPES: [chip::Choice<BrushShape>; 3] = [
    chip::Choice { value: BrushShape::Circle, name: "Circle", description: "", enabled: true },
    chip::Choice { value: BrushShape::Square, name: "Square", description: "", enabled: true },
    chip::Choice { value: BrushShape::Line, name: "Line", description: "", enabled: true },
];

/// The Paint Brush panel: what the strip does not show.
pub(super) fn flyout_body(ui: &mut Ui, s: &mut BrushSettings, model: SettingsModel) -> BrushChipOutcome {
    let mut outcome = BrushChipOutcome::default();
    if chip::popover_header(ui, "Paint Brush", Some(("Reset the brush", false))) {
        outcome.reset_brush_requested = true;
    }
    outcome.committed |= chip::choice_row(ui, "Shape", &BRUSH_SHAPES, &mut s.shape);
    if shows(model, Surface::FillRegionKnobs) {
        // Sharpen displays as 0-100% on a 0-2 internal range.
        let sh = chip::slider_row_f32(ui, "Sharpen", &mut s.inpaint_sharpen, 0.0..=2.0, false, |v| fmt::percent(v / 2.0));
        outcome.committed |= sh.commit;
        super::hint(ui, "Sharpen the filled area, which comes out slightly soft.");
    }
    if shows(model, Surface::SelectionStrokes) {
        outcome.committed |= render_shared_selection_section(ui, s);
    }
    outcome
}

/// Selection visualization knobs shared between the Paint Brush chip and
/// the Magic Brush chip. Returns `true` if any slider committed this frame.
pub(super) fn render_shared_selection_section(ui: &mut egui::Ui, s: &mut BrushSettings) -> bool {
    let mut committed = false;
    chip::panel_section(ui, "SELECTION", None);
    let fe = chip::slider_row_f32(
        ui, "Feather", &mut s.edge_feather, 0.0..=20.0, false,
        |v| fmt::off_or(v, 0.1, |v| fmt::px(v, 0)),
    );
    committed |= fe.commit;
    super::hint(ui, "Soften the selection edge.");
    committed
}


/// The eraser's Expand region slider, in the flyout and the tool strip.
pub(super) fn expand_region_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0))
}

/// The eraser's Feather slider, in the flyout and the tool strip.
pub(super) fn feather_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Feather", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::off_or(v, 0.5, |v| fmt::px(v, 0)))
}
