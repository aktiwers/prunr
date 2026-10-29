//! The brushes' mode switch and the Paint Brush's shapes, on the tool
//! strip.

use egui::Ui;

use crate::gui::brush_state::{BrushSettings, CutoutAction, Tool};
use crate::gui::theme;
use crate::gui::settings::SettingsModel;
use prunr_core::brush::BrushShape;
use prunr_core::selection::BrushMode;

use super::chip;

/// What the tool strip changed this frame.
#[derive(Default, Clone, Copy)]
pub(crate) struct BrushChipOutcome {
    /// That brush's settings go back to the default preset.
    pub reset: Option<Tool>,
    /// A slider settled or a choice was made: the brush settings persist.
    pub committed: bool,
    /// Restore / Erase / Select changed: the selection is applied anew.
    pub action_changed: bool,
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
    chip::Choice { value: BrushShape::Circle, name: "Circle", description: "A round brush", enabled: true },
    chip::Choice { value: BrushShape::Square, name: "Square", description: "A square brush", enabled: true },
    chip::Choice { value: BrushShape::Line, name: "Line", description: "A straight line from where you press to where you let go", enabled: true },
];

const SHAPE_ICONS: [chip::IconPaint; 3] = [circle_icon, square_icon, line_icon];

// The shapes on a 24-unit grid, as the strip's board draws them.
fn circle_icon(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    painter.circle_stroke(rect.center(), rect.width() * 7.0 / 24.0, egui::Stroke::new(rect.width() / 12.0, color));
}

fn square_icon(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let square = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(rect.width() * 13.0 / 24.0));
    painter.rect_stroke(square, 1.0, egui::Stroke::new(rect.width() / 12.0, color), egui::StrokeKind::Middle);
}

fn line_icon(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let at = |x: f32, y: f32| rect.lerp_inside(egui::vec2(x / 24.0, y / 24.0));
    painter.line_segment([at(5.0, 19.0), at(19.0, 5.0)], egui::Stroke::new(rect.width() * 2.4 / 24.0, color));
}

/// The Paint Brush's shape, as three icons.
pub(super) fn shape_buttons(ui: &mut Ui, s: &mut BrushSettings) -> bool {
    chip::icon_choice_buttons(ui, &BRUSH_SHAPES, &SHAPE_ICONS, &mut s.shape)
}
