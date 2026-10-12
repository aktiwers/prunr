//! Selection overlay: one untinted textured quad per frame. Fill,
//! outline and feather are already baked into the item's selection
//! texture off-thread. RENDER-ONLY — no I/O, no decode, no GPU upload.

use egui::{Color32, Rect, Ui};

use crate::gui::item::BatchItem;

/// Called from canvas render after the item image is painted. `img_rect`
/// is the on-screen rect of the displayed texture (post-zoom).
/// `working`: a run is filling this region, which then breathes in soft
/// light with a sheen crossing it, inside the brushed shape.
pub(crate) fn render_selection_overlay(ui: &Ui, item: &BatchItem, img_rect: Rect, working: bool) {
    let Some(tex) = item.selection_texture.as_ref() else { return };
    if working {
        let t = crate::gui::theme::motion_clock(ui.ctx());
        super::effects::paint_working_masked(ui.painter(), img_rect, tex.handle.id(), t);
    } else {
        ui.painter().image(tex.handle.id(), img_rect, super::effects::UNIT_UV, Color32::WHITE);
    }
}
