//! The Magic Brush's own knobs on the tool strip: what a click picks and
//! whether specks go, how sure a pick must be or how far around it reaches.

use egui::Ui;

use crate::gui::brush_state::{BrushSettings, DEPTH_RANGE};
use crate::gui::views::{chip, fmt};

/// What a Magic Brush click picks, which its slider then tunes.
const PICK_BY: [chip::Choice<bool>; 2] = [
    chip::Choice { value: false, name: "Object", description: "Pick the object under the click; Confidence sets how sure it must be", enabled: true },
    chip::Choice { value: true, name: "Depth", description: "Pick everything about as far away as the click; Range sets how far around it", enabled: true },
];

/// Object or Depth, and the Remove specks icon beside them.
pub(super) fn pick(ui: &mut Ui, s: &mut BrushSettings) -> bool {
    let picked = chip::choice_buttons(ui, &PICK_BY, &mut s.magic_depth);
    let specks = chip::icon_toggle(ui, "Remove specks", "Drop tiny islands and fill pinholes where the model is unsure.", specks_icon, &mut s.magic_remove_specks);
    picked | specks
}

// Scattered dots, laid out on a 24-unit grid scaled to `rect`.
fn specks_icon(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    for (x, y, r) in [(6.0, 7.0, 1.8), (17.0, 6.0, 1.4), (11.0, 12.0, 2.2), (18.0, 16.0, 1.8), (7.0, 18.0, 1.4)] {
        painter.circle_filled(rect.lerp_inside(egui::vec2(x / 24.0, y / 24.0)), rect.width() * r / 24.0, color);
    }
}

/// The reading slider's names: an object pick's, a depth pick's.
pub(super) const CONFIDENCE: &str = "Confidence";
pub(super) const RANGE: &str = "Range";

/// Confidence for an object pick, Range for a depth pick, in one place.
pub(super) fn reading(ui: &mut Ui, s: &mut BrushSettings, name_width: f32) -> bool {
    if s.magic_depth {
        chip::slider_cell_f32(ui, RANGE, name_width, &mut s.magic_depth_range, DEPTH_RANGE, false, |v| fmt::plain(v, 2)).commit
    } else {
        chip::slider_cell_f32(ui, CONFIDENCE, name_width, &mut s.magic_confidence_threshold, crate::gui::brush_state::UNIT_RANGE, false, fmt::percent).commit
    }
}
