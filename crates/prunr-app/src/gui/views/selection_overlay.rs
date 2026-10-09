//! Selection overlay: one textured quad per frame. Fill, outline and
//! feather are baked into `BatchItem.selection_texture` off-thread (see
//! `background_io::build_selection_image`); the only per-frame knob is
//! the tint alpha. RENDER-ONLY — no I/O, no decode, no GPU upload.

use egui::{Color32, Pos2, Rect, Ui};

use crate::gui::background_io::SelectionStyle;
use crate::gui::brush_state::BrushSettings;
use crate::gui::item::BatchItem;

/// Called from canvas render after the item image is painted. `img_rect`
/// is the on-screen rect of the displayed texture (post-zoom).
pub(crate) fn render_selection_overlay(ui: &Ui, item: &BatchItem, brush: &BrushSettings, img_rect: Rect) {
    let Some(tex) = item.selection_texture.as_ref() else { return };
    let alpha = SelectionStyle::from_brush(brush).tint_alpha();
    if alpha <= 0.001 {
        return;
    }
    ui.painter().image(
        tex.id(),
        img_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::from_white_alpha((alpha * 255.0).round() as u8),
    );
}
