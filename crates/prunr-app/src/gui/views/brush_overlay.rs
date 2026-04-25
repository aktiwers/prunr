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

use egui::{Color32, Pos2, Rect, Stroke, Ui};

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
/// the displayed texture (post-zoom, post-pan); the brush works in that
/// rect's normalized coordinate space, so the underlying texture size
/// is irrelevant here.
pub(crate) fn handle_input(
    ui: &mut Ui,
    brush_state: &mut BrushState,
    item: &BatchItem,
    img_rect: Rect,
) -> BrushAction {
    let Some((model_w, model_h)) = model_dims(item) else {
        // No cached tensor → brush has nothing to write into. Render a
        // muted cursor so the user gets feedback that brush is ON, but
        // skip pointer wiring.
        tracing::debug!(item_id = item.id, "brush active but no cached_tensor — cursor only");
        draw_cursor(ui, img_rect, brush_state, /*armed=*/ false);
        return BrushAction::None;
    };

    // No `ui.interact` here: it would set `egui_wants_pointer_input`,
    // which the canvas pan handler reads to decide whether to ignore
    // events. With pan on secondary in brush mode, primary belongs to
    // the brush and secondary to pan — they don't conflict, so neither
    // needs widget-level claim. Reading raw input below is enough.
    let (hover, primary_pressed, primary_down, primary_released) = ui.input(|i| {
        (
            i.pointer.hover_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.primary_released(),
        )
    });
    let pointer_on_img = hover.filter(|p| img_rect.contains(*p));

    // In-progress stroke trail (drawn first so the cursor renders on top).
    draw_trail(ui, brush_state);

    // Cursor circle — armed only when the pointer is over the image.
    draw_cursor(ui, img_rect, brush_state, pointer_on_img.is_some());

    // Brush radius in model-pixel space — derived from the on-screen
    // img_rect width vs model width. One isotropic factor is enough:
    // letterboxed images keep proportional w/h.
    let model_radius_for = |screen_radius: f32| -> f32 {
        screen_radius * (model_w as f32 / img_rect.width().max(1.0))
    };

    let screen_radius = brush_state.settings().radius;

    if primary_pressed {
        if let Some(p) = pointer_on_img {
            tracing::debug!(model_w, model_h, "brush press — begin stroke");
            brush_state.begin_stroke(model_w, model_h);
            let m = screen_to_model(p, img_rect, model_w, model_h);
            brush_state.extend_stroke_with_radius(m.x, m.y, model_radius_for(screen_radius));
            brush_state.record_trail_stamp(p.x, p.y, screen_radius);
        }
    }

    if primary_down && brush_state.has_active_stroke() {
        if let Some(p) = pointer_on_img {
            let m = screen_to_model(p, img_rect, model_w, model_h);
            brush_state.extend_stroke_with_radius(m.x, m.y, model_radius_for(screen_radius));
            brush_state.record_trail_stamp(p.x, p.y, screen_radius);
        }
    }

    if primary_released && brush_state.has_active_stroke() {
        if let Some(strokes) = brush_state.commit_stroke() {
            tracing::debug!("brush release — commit");
            return BrushAction::Committed(strokes);
        }
    }

    BrushAction::None
}

/// Convert a screen-space pointer to model-grid coordinates.
fn screen_to_model(p: Pos2, img_rect: Rect, model_w: u16, model_h: u16) -> Pos2 {
    let in_img_x = (p.x - img_rect.min.x) / img_rect.width().max(1.0);
    let in_img_y = (p.y - img_rect.min.y) / img_rect.height().max(1.0);
    Pos2::new(in_img_x * model_w as f32, in_img_y * model_h as f32)
}

/// Translucent ACCENT-purple stamps along the in-progress stroke's
/// trail. Drawn on a foreground layer so the image rendered earlier
/// in the same frame can't accidentally cover them, and tinted
/// brighter than the theme ACCENT so the trail is legible on dark
/// images too.
fn draw_trail(ui: &Ui, brush_state: &BrushState) {
    let s = brush_state.settings();
    if s.strength <= 0.0 {
        return;
    }
    let accent = theme::ACCENT;
    let center_alpha = (110.0 * s.strength.clamp(0.0, 1.0)) as u8;
    if center_alpha == 0 {
        return;
    }
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("brush_trail"),
    ));
    let hardness = s.hardness.clamp(0.0, 1.0);
    let solid = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), center_alpha);

    let Some(shape) = brush_state.active_shape() else { return };
    match shape {
        prunr_core::brush::BrushShape::Line => {
            // Collect only when needed — the Line branch wants first + last.
            let stamps: Vec<_> = brush_state.trail_stamps().collect();
            let (Some(&first), Some(&last)) = (stamps.first(), stamps.last()) else { return };
            let stroke = egui::Stroke::new(first.2 * 2.0, solid);
            painter.line_segment([Pos2::new(first.0, first.1), Pos2::new(last.0, last.1)], stroke);
        }
        prunr_core::brush::BrushShape::Circle => {
            for (sx, sy, outer_r) in brush_state.trail_stamps() {
                draw_round_stamp(&painter, sx, sy, outer_r, hardness, accent, center_alpha, solid);
            }
        }
        prunr_core::brush::BrushShape::Square => {
            for (sx, sy, outer_r) in brush_state.trail_stamps() {
                draw_square_stamp(&painter, sx, sy, outer_r, hardness, accent, center_alpha);
            }
        }
    }
}

fn draw_round_stamp(
    painter: &egui::Painter,
    sx: f32, sy: f32, outer_r: f32,
    hardness: f32,
    accent: Color32, center_alpha: u8, solid: Color32,
) {
    let center = Pos2::new(sx, sy);
    let inner_r = outer_r * hardness;
    if inner_r >= 0.5 {
        painter.circle_filled(center, inner_r, solid);
    }
    let span = (outer_r - inner_r).max(0.001);
    let steps = 8;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let dist = inner_r + span * t;
        let intensity = prunr_core::math::smoothstep(1.0 - t);
        let a = (center_alpha as f32 * intensity) as u8;
        if a == 0 {
            continue;
        }
        let stroke_color = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a);
        painter.circle_stroke(center, dist, egui::Stroke::new(span / steps as f32 * 1.4, stroke_color));
    }
}

fn draw_square_stamp(
    painter: &egui::Painter,
    sx: f32, sy: f32, half_size: f32,
    hardness: f32,
    accent: Color32, center_alpha: u8,
) {
    let inner = half_size * hardness;
    if inner >= 0.5 {
        let solid = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), center_alpha);
        painter.rect_filled(
            Rect::from_center_size(Pos2::new(sx, sy), egui::vec2(inner * 2.0, inner * 2.0)),
            0.0,
            solid,
        );
    }
    let span = (half_size - inner).max(0.001);
    let steps = 6;
    for i in 0..steps {
        let t = (i as f32 + 0.5) / steps as f32;
        let dist = inner + span * t;
        let intensity = prunr_core::math::smoothstep(1.0 - t);
        let a = (center_alpha as f32 * intensity) as u8;
        if a == 0 {
            continue;
        }
        let stroke_color = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a);
        painter.rect_stroke(
            Rect::from_center_size(Pos2::new(sx, sy), egui::vec2(dist * 2.0, dist * 2.0)),
            0.0,
            egui::Stroke::new(span / steps as f32 * 1.4, stroke_color),
            egui::StrokeKind::Outside,
        );
    }
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
    let stroke = Stroke::new(1.5, color);
    match brush_state.settings().shape {
        prunr_core::brush::BrushShape::Circle => {
            ui.painter().circle_stroke(p, r, stroke);
        }
        prunr_core::brush::BrushShape::Square => {
            ui.painter().rect_stroke(
                Rect::from_center_size(p, egui::vec2(r * 2.0, r * 2.0)),
                0.0,
                stroke,
                egui::StrokeKind::Outside,
            );
        }
        prunr_core::brush::BrushShape::Line => {
            ui.painter().line_segment([Pos2::new(p.x - r, p.y), Pos2::new(p.x + r, p.y)], stroke);
            ui.painter().line_segment([Pos2::new(p.x, p.y - r), Pos2::new(p.x, p.y + r)], stroke);
        }
    }
    // Inner dot for precision targeting.
    ui.painter().circle_filled(p, 1.5, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the screen→model coordinate transform. A bug here flips an
    /// axis silently — strokes paint mirrored or off-by-half-image.
    #[test]
    fn screen_to_model_corners_and_center() {
        let img_rect = Rect::from_min_size(Pos2::new(100.0, 200.0), egui::vec2(400.0, 300.0));
        let (mw, mh) = (320u16, 240u16);

        let top_left = screen_to_model(img_rect.min, img_rect, mw, mh);
        assert!((top_left.x - 0.0).abs() < 1e-3);
        assert!((top_left.y - 0.0).abs() < 1e-3);

        let bottom_right = screen_to_model(img_rect.max, img_rect, mw, mh);
        assert!((bottom_right.x - 320.0).abs() < 1e-3);
        assert!((bottom_right.y - 240.0).abs() < 1e-3);

        let center = screen_to_model(img_rect.center(), img_rect, mw, mh);
        assert!((center.x - 160.0).abs() < 1e-3);
        assert!((center.y - 120.0).abs() < 1e-3);
    }

    #[test]
    fn screen_to_model_handles_zero_width_safely() {
        let img_rect = Rect::from_min_size(Pos2::new(0.0, 0.0), egui::vec2(0.0, 0.0));
        let p = screen_to_model(Pos2::new(50.0, 50.0), img_rect, 320, 240);
        assert!(p.x.is_finite() && p.y.is_finite(), "must not return NaN/inf on degenerate rect");
    }
}
