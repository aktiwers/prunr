//! The Paint Brush panel (what its strip does not show), and the brush
//! knobs the strip and the panel share.

use egui::Ui;

use crate::gui::brush_state::{BrushSettings, CutoutAction, StrokeReach};
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
    /// Restore / Erase / Select changed: the selection is applied anew.
    pub action_changed: bool,
}

impl BrushChipOutcome {
    pub fn merge(&mut self, other: Self) {
        self.reset_brush_requested |= other.reset_brush_requested;
        self.committed |= other.committed;
        self.action_changed |= other.action_changed;
    }
}

/// On a background-removal model: what the selection does to the cut-out.
const CUTOUT_ACTIONS: [chip::Choice<CutoutAction>; 3] = [
    chip::Choice { value: CutoutAction::Restore, name: "Restore", description: "Bring back what the model took away, as you paint", enabled: true },
    chip::Choice { value: CutoutAction::Erase, name: "Erase", description: "Remove what the model left behind, as you paint", enabled: true },
    chip::Choice { value: CutoutAction::Select, name: "Select", description: "Only select; Restore and Delete in the selection bar act on it", enabled: true },
];

/// Elsewhere a stroke shapes the selection or the region to fill.
const REGION_MODES: [chip::Choice<BrushMode>; 2] = [
    chip::Choice { value: BrushMode::Add, name: "Add", description: "Add to the selection", enabled: true },
    chip::Choice { value: BrushMode::Subtract, name: "Subtract", description: "Take away from the selection", enabled: true },
];

/// The brush switch: on a cut-out what the selection does (Erase picked
/// in warm red, as its marks are drawn), elsewhere Add / Subtract.
pub(super) fn mode_buttons(ui: &mut Ui, model: SettingsModel, s: &mut BrushSettings) -> bool {
    if model.edits_cutout() {
        let picked = if s.cutout_action == CutoutAction::Erase { theme::ERASE_PICKED } else { theme::ACCENT_BRIGHT };
        chip::choice_buttons_picked(ui, &CUTOUT_ACTIONS, &mut s.cutout_action, picked)
    } else {
        chip::choice_buttons(ui, &REGION_MODES, &mut s.mode)
    }
}

const BRUSH_SHAPES: [chip::Choice<BrushShape>; 3] = [
    chip::Choice { value: BrushShape::Circle, name: "Circle", description: "", enabled: true },
    chip::Choice { value: BrushShape::Square, name: "Square", description: "", enabled: true },
    chip::Choice { value: BrushShape::Line, name: "Line", description: "", enabled: true },
];

/// The Paint Brush panel: what the strip does not show.
pub(super) fn flyout_body(ui: &mut Ui, s: &mut BrushSettings, reach: &mut StrokeReach, model: SettingsModel) -> BrushChipOutcome {
    let mut outcome = BrushChipOutcome::default();
    if chip::popover_header(ui, "Paint Brush", Some(("Reset the brush", false))) {
        outcome.reset_brush_requested = true;
    }
    reach_row(ui, reach);
    outcome.committed |= chip::choice_row(ui, "Shape", &BRUSH_SHAPES, &mut s.shape);
    if shows(model, Surface::SelectionStrokes) {
        outcome.committed |= render_shared_selection_section(ui, s);
    }
    outcome
}

/// The selection's Feather, in both brushes' panels. True when it
/// committed this frame.
pub(super) fn render_shared_selection_section(ui: &mut egui::Ui, s: &mut BrushSettings) -> bool {
    chip::panel_section(ui, "Selection", None);
    let feather = chip::slider_row_f32(ui, "Feather", &mut s.edge_feather, 0.0..=20.0, false, |v| fmt::off_or(v, 0.1, |v| fmt::px(v, 0)));
    super::hint(ui, "Soften the selection edge.");
    feather.commit
}


const REACHES: [chip::Choice<StrokeReach>; 3] = [
    chip::Choice { value: StrokeReach::Next, name: "Next", description: "", enabled: true },
    chip::Choice { value: StrokeReach::Last, name: "Last", description: "", enabled: true },
    chip::Choice { value: StrokeReach::All, name: "All", description: "", enabled: true },
];

/// What the sliders adjust, the same in both brushes' panels.
pub(super) fn reach_row(ui: &mut Ui, reach: &mut StrokeReach) {
    chip::choice_row(ui, "Sliders adjust", &REACHES, reach);
    super::hint(ui, super::tool_strip::ADJUST_REACH);
}

/// The eraser's Expand region slider, in the flyout and the tool strip.
pub(super) fn expand_region_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Expand region", &mut s.inpaint_grow, -16.0..=16.0, false, |v| fmt::signed_px(v, 0))
}


/// The eraser's Feather slider, in the flyout and the tool strip.
pub(super) fn feather_row(ui: &mut egui::Ui, s: &mut BrushSettings) -> chip::ChipChange {
    chip::slider_row_f32(ui, "Feather", &mut s.inpaint_feather, 0.0..=32.0, false, |v| fmt::off_or(v, 0.5, |v| fmt::px(v, 0)))
}
