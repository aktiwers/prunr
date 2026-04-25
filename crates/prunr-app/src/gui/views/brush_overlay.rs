//! Canvas-side brush input + cursor render.
//!
//! Called from `canvas::render_done` when `brush_state.is_enabled()` and
//! the result/source texture is showing. Owns:
//! - the soft circle cursor
//! - left-press/drag/release pointer wiring → BrushState
//! - on commit: write the new correction onto the active BatchItem.
//!
//! Does NOT own coordinate math beyond the screen→model transform
//! (canvas owns `compute_img_rect` and zoom/pan).

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::gui::brush_state::BrushState;
use crate::gui::item::BatchItem;
use crate::gui::theme;

/// Minimum brush radius in screen pixels — clamps against zoom-out
/// shrinking the cursor below visibility.
const MIN_SCREEN_RADIUS_PX: f32 = 2.0;

/// Result of `handle_input`. Caller (canvas) reacts after the painter
/// has been used so it can route the commit through the app's normal
/// dispatch / cache-invalidation path.
pub(crate) enum BrushAction {
    /// User finished a stroke. Caller writes it to the BatchItem.
    Committed(prunr_core::brush::MaskCorrection),
    /// User clicked or moved without committing a stroke yet.
    None,
}

/// Returns the model-space resolution of the active item's mask, or
/// `None` if no inference has run yet (brush has nothing to correct).
pub(crate) fn model_dims(item: &BatchItem) -> Option<(u16, u16)> {
    let t = item.cached_tensor.as_ref()?;
    Some((t.width as u16, t.height as u16))
}

/// Handle pointer input + paint cursor for one frame. The canvas calls
/// this after rendering the image. `img_rect` is the on-screen rect of
/// the displayed texture (post-zoom, post-pan). `tex_size` is the
/// underlying texture pixel dimensions.
pub(crate) fn handle_input(
    ui: &mut Ui,
    brush_state: &mut BrushState,
    item: &BatchItem,
    img_rect: Rect,
    tex_size: Vec2,
) -> BrushAction {
    let Some((model_w, model_h)) = model_dims(item) else {
        // No cached tensor → brush has nothing to write into. Render a
        // muted cursor so the user gets feedback that brush is ON, but
        // skip pointer wiring.
        draw_cursor(ui, img_rect, brush_state, /*armed=*/ false);
        return BrushAction::None;
    };

    // `Sense::click_and_drag` claims pointer events on this rect so the
    // panel below (zoom/pan) doesn't also see them.
    let response = ui.interact(img_rect, ui.id().with("brush_overlay"), Sense::click_and_drag());

    let pointer = response.hover_pos().or_else(|| response.interact_pointer_pos());
    let pointer_on_img = pointer.filter(|p| img_rect.contains(*p));

    // Cursor circle — armed only when the pointer is over the image.
    draw_cursor(ui, img_rect, brush_state, pointer_on_img.is_some());

    let Some(p) = pointer_on_img else {
        if !response.dragged() && brush_state.has_active_stroke() {
            // Pointer left the image mid-drag — keep the stroke alive;
            // user might come back. The stroke commits on full release.
        }
        return BrushAction::None;
    };

    let model_pos = screen_to_model(p, img_rect, tex_size, model_w, model_h);

    if response.drag_started() {
        brush_state.begin_stroke(model_w, model_h);
    }

    if response.dragged() || response.drag_started() {
        // Convert the screen-space brush radius to model space using
        // the same scale (img_rect→tex→model). One scale factor is
        // enough — strokes paint isotropically at the final resolution.
        let model_radius = brush_state.settings().radius
            * (model_w as f32 / img_rect.width().max(1.0));
        brush_state.extend_stroke_with_radius(model_pos.x, model_pos.y, model_radius);
    }

    if response.drag_stopped() {
        if let Some(strokes) = brush_state.commit_stroke() {
            return BrushAction::Committed(strokes);
        }
    }

    BrushAction::None
}

/// Convert a screen-space pointer to model-grid coordinates.
fn screen_to_model(p: Pos2, img_rect: Rect, tex_size: Vec2, model_w: u16, model_h: u16) -> Pos2 {
    let in_img_x = (p.x - img_rect.min.x) / img_rect.width().max(1.0);
    let in_img_y = (p.y - img_rect.min.y) / img_rect.height().max(1.0);
    let _ = tex_size;
    Pos2::new(in_img_x * model_w as f32, in_img_y * model_h as f32)
}

fn draw_cursor(ui: &Ui, img_rect: Rect, brush_state: &BrushState, armed: bool) {
    let pointer = ui.input(|i| i.pointer.hover_pos());
    let Some(p) = pointer else { return };
    if !img_rect.contains(p) {
        return;
    }
    let r = brush_state.settings().radius.max(MIN_SCREEN_RADIUS_PX);
    let color = if armed {
        theme::ACCENT
    } else {
        Color32::from_rgba_premultiplied(160, 160, 160, 180)
    };
    ui.painter()
        .circle_stroke(p, r, Stroke::new(1.5, color));
    // Inner dot for precision targeting.
    ui.painter().circle_filled(p, 1.5, color);
}
