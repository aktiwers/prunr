//! Selection overlay: one untinted textured quad per frame. Fill,
//! outline and feather are already baked into the item's selection
//! texture off-thread. RENDER-ONLY — no I/O, no decode, no GPU upload.

use egui::{Color32, Pos2, Rect, Ui};

use crate::gui::item::BatchItem;

/// Called from canvas render after the item image is painted. `img_rect`
/// is the on-screen rect of the displayed texture (post-zoom).
pub(crate) fn render_selection_overlay(ui: &Ui, item: &BatchItem, img_rect: Rect) {
    let Some(tex) = item.selection_texture.as_ref() else { return };
    ui.painter().image(
        tex.handle.id(),
        img_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}
