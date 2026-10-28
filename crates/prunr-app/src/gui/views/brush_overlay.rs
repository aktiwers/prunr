//! Canvas-side brush input + cursor render. A finished stroke goes back to
//! the canvas as `BrushAction::Committed`.

use egui::{Color32, Pos2, Rect, Stroke, Ui};

use crate::gui::brush_state::{BrushSettings, BrushState};
use prunr_core::brush::Stamp;
use crate::gui::item::BatchItem;
use crate::gui::settings::Settings;
use crate::gui::theme;

/// Minimum brush radius in screen pixels — clamps against zoom-out
/// shrinking the cursor below visibility.
const MIN_SCREEN_RADIUS_PX: f32 = 2.0;

/// Result of `handle_input`. Caller (canvas) reacts after the painter
/// has been used so it can route the commit through the app's normal
/// dispatch / cache-invalidation path.
pub(crate) enum BrushAction {
    /// User finished a stroke: the signed, soft stroke at source-image
    /// resolution, ready to merge into the item's selection, and its path.
    Committed(prunr_core::selection::MaskArtifact, crate::gui::brush_state::PaintPath),
    /// User clicked or moved without committing a stroke yet.
    None,
}

/// Handle pointer input + paint cursor for one frame. The brush works in
/// the normalized space of `img_rect` (the on-screen rect after zoom and
/// pan), so the texture size is irrelevant here.
pub(crate) fn handle_input(
    ui: &mut Ui,
    brush_state: &mut BrushState,
    app_settings: &Settings,
    stamp: Stamp,
    item: &BatchItem,
    img_rect: Rect,
) -> BrushAction {
    let settings = &app_settings.brush;
    let look = CursorLook::of(settings, app_settings.brush_erases());
    // Strokes are authored at SOURCE resolution because the selection lives
    // there; painting at tensor resolution misaligned strokes once they were
    // resampled back.
    let (model_w, model_h) = item.dimensions;
    if model_w == 0 || model_h == 0 {
        // Render a muted cursor so the user gets feedback that brush is
        // ON, but skip wiring.
        tracing::debug!(item_id = item.id, "brush active but item has no dimensions — cursor only");
        draw_cursor(ui, img_rect, look, /*armed=*/ false);
        return BrushAction::None;
    }

    // No `ui.interact` here: it would set `egui_wants_pointer_input`, which
    // the canvas pan handler reads. In brush mode primary is the brush's and
    // secondary pans, so raw input is enough.
    let (hover, primary_pressed, primary_down, primary_released) = ui.input(|i| {
        (
            i.pointer.hover_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.primary_released(),
        )
    });
    let pointer_on_img = hover.filter(|&p| on_shown_image(ui, img_rect, p));

    // In-progress stroke trail (drawn first so the cursor renders on top).
    draw_trail(ui, brush_state, look);

    // Cursor circle — armed only when the pointer is over the image.
    draw_cursor(ui, img_rect, look, pointer_on_img.is_some());

    // Brush radius in model-pixel space — derived from the on-screen
    // img_rect width vs model width. One isotropic factor is enough:
    // letterboxed images keep proportional w/h.
    let screen_radius = settings.radius;
    let model_radius_for = |screen_radius: f32| -> f32 {
        screen_radius * (model_w as f32 / img_rect.width().max(1.0))
    };

    if primary_pressed {
        if let Some(p) = pointer_on_img {
            tracing::debug!(model_w, model_h, "brush press — begin stroke");
            brush_state.begin_stroke(model_w, model_h, settings.shape, screen_radius);
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
        if let Some((correction, path)) = brush_state.commit_stroke(stamp) {
            tracing::debug!("brush release — commit");
            return BrushAction::Committed(correction, path);
        }
    }

    BrushAction::None
}

pub(super) fn screen_to_model(p: Pos2, img_rect: Rect, model_w: u32, model_h: u32) -> Pos2 {
    let in_img_x = (p.x - img_rect.min.x) / img_rect.width().max(1.0);
    let in_img_y = (p.y - img_rect.min.y) / img_rect.height().max(1.0);
    Pos2::new(in_img_x * model_w as f32, in_img_y * model_h as f32)
}

/// How a brush's cursor and trail are drawn.
#[derive(Clone, Copy)]
pub(crate) struct CursorLook {
    pub(crate) radius: f32,
    pub(crate) hardness: f32,
    pub(crate) strength: f32,
    pub(crate) shape: prunr_core::brush::BrushShape,
    /// Drawn in Erase's warm red.
    pub(crate) erase: bool,
}

impl CursorLook {
    pub(crate) fn of(s: &BrushSettings, erase: bool) -> Self {
        Self { radius: s.radius, hardness: s.hardness, strength: s.strength, shape: s.shape, erase }
    }
}

/// Magic Brush picks objects, so its size and shape change nothing: it
/// draws a small fixed cursor and trail.
pub(crate) const MAGIC_CURSOR: CursorLook = CursorLook {
    radius: 10.0,
    hardness: 1.0,
    strength: 1.0,
    shape: prunr_core::brush::BrushShape::Circle,
    erase: false,
};

/// Translucent stamps in the mark colour along the in-progress stroke's
/// trail, on a foreground layer so the image rendered earlier in the
/// same frame can't cover them.
fn draw_trail(ui: &Ui, brush_state: &BrushState, look: CursorLook) {
    let Some(shape) = brush_state.active_shape() else { return };
    draw_trail_for(ui, look, shape, brush_state.trail_stamps());
}

/// Paint the in-progress trail from any stamp iterator; Paint Brush and
/// Magic Brush differ only in what they commit on release.
pub(crate) fn draw_trail_for(
    ui: &Ui,
    s: CursorLook,
    shape: prunr_core::brush::BrushShape,
    stamps: impl IntoIterator<Item = (f32, f32, f32)>,
) {
    if s.strength <= 0.0 {
        return;
    }
    let accent = theme::mark_color(s.erase);
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

    match shape {
        prunr_core::brush::BrushShape::Line => {
            let stamps: Vec<_> = stamps.into_iter().collect();
            let (Some(&first), Some(&last)) = (stamps.first(), stamps.last()) else { return };
            let stroke = egui::Stroke::new(first.2 * 2.0, solid);
            painter.line_segment([Pos2::new(first.0, first.1), Pos2::new(last.0, last.1)], stroke);
        }
        prunr_core::brush::BrushShape::Circle | prunr_core::brush::BrushShape::Square => {
            let key = TrailKey { shape, hardness, accent, alpha: center_alpha, ppp: ui.ctx().pixels_per_point() };
            let options = ui.ctx().tessellation_options(|o| *o);
            let mesh = trail_mesh(ui.ctx(), key, options, stamps.into_iter());
            painter.add(egui::Shape::mesh(mesh));
        }
    }
}

/// What a trail's tessellated stamps were made with.
#[derive(Clone, Copy, PartialEq)]
struct TrailKey {
    shape: prunr_core::brush::BrushShape,
    hardness: f32,
    accent: Color32,
    alpha: u8,
    ppp: f32,
}

/// A stroke's stamps tessellated once each and kept across frames: a long
/// stroke with a small brush is thousands of stamps of a fill and rings
/// each, tessellated anew every frame. New stamps are appended; another
/// stroke, look or scale starts over.
#[derive(Clone, Default)]
struct TrailMesh {
    key: Option<TrailKey>,
    first: Option<(f32, f32, f32)>,
    done: usize,
    mesh: std::sync::Arc<egui::Mesh>,
}

fn trail_mesh(
    ctx: &egui::Context,
    key: TrailKey,
    options: egui::epaint::TessellationOptions,
    stamps: impl Iterator<Item = (f32, f32, f32)>,
) -> std::sync::Arc<egui::Mesh> {
    let id = egui::Id::new("brush_trail_mesh");
    let stamps: Vec<(f32, f32, f32)> = stamps.collect();
    // Taken out, not copied: a second holder would make `make_mut` copy
    // the whole trail each frame it grows.
    let mut kept: TrailMesh = ctx.data_mut(|d| d.remove_temp(id)).unwrap_or_default();
    if kept.key != Some(key) || kept.first != stamps.first().copied() || stamps.len() < kept.done {
        kept = TrailMesh { key: Some(key), first: stamps.first().copied(), ..TrailMesh::default() };
    }
    if stamps.len() > kept.done {
        let mut tessellator = egui::epaint::Tessellator::new(key.ppp, options, [1, 1], Vec::new());
        let mesh = std::sync::Arc::make_mut(&mut kept.mesh);
        for &(sx, sy, outer_r) in &stamps[kept.done..] {
            let center = Pos2::new(sx, sy);
            let shapes: Vec<egui::Shape> = match key.shape {
                prunr_core::brush::BrushShape::Square => super::chip::falloff_square_shapes(center, outer_r, key.hardness, key.accent, key.alpha, 6).collect(),
                _ => super::chip::falloff_circle_shapes(center, outer_r, key.hardness, key.accent, key.alpha, 8).collect(),
            };
            for shape in shapes {
                tessellator.tessellate_shape(shape, mesh);
            }
        }
        kept.done = stamps.len();
    }
    let mesh = std::sync::Arc::clone(&kept.mesh);
    ctx.data_mut(|d| d.insert_temp(id, kept));
    mesh
}

/// Whether `p` is on the part of the image the canvas shows, and not on a
/// control drawn over it (the strip). Zoomed in, the image reaches under
/// the toolbars, which share the canvas's layer: there the pointer is on
/// the toolbar.
pub(crate) fn on_shown_image(ui: &Ui, img_rect: Rect, p: Pos2) -> bool {
    img_rect.intersect(ui.clip_rect()).contains(p) && on_canvas_layer(ui, p)
}

fn on_canvas_layer(ui: &Ui, p: Pos2) -> bool {
    ui.ctx().layer_id_at(p).is_none_or(|l| l == ui.layer_id())
}

pub(crate) fn draw_cursor(ui: &Ui, img_rect: Rect, s: CursorLook, armed: bool) {
    let pointer = ui.input(|i| i.pointer.hover_pos());
    let Some(p) = pointer else { return };
    if !img_rect.intersect(ui.clip_rect()).contains(p) {
        return;
    }
    // Ours replaces the system's (an arrow, or a cross on GNOME), except
    // over a control floating on the image, which keeps its own.
    if on_canvas_layer(ui, p) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
    }
    let r = s.radius.max(MIN_SCREEN_RADIUS_PX);
    let color = if armed {
        theme::mark_color(s.erase)
    } else {
        Color32::from_rgba_premultiplied(160, 160, 160, 180)
    };
    let stroke = Stroke::new(1.5, color);
    match s.shape {
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
    /// A stroke's stamps are tessellated once: a frame with no new stamp
    /// reuses the mesh, a new stamp only adds to it, another stroke
    /// starts over.
    #[test]
    fn a_trail_tessellates_each_stamp_once() {
        let ctx = egui::Context::default();
        let key = super::TrailKey { shape: prunr_core::brush::BrushShape::Circle, hardness: 0.5, accent: egui::Color32::RED, alpha: 100, ppp: 1.0 };
        let options = egui::epaint::TessellationOptions::default();
        let one = [(10.0, 10.0, 6.0)];
        let two = [(10.0, 10.0, 6.0), (14.0, 10.0, 6.0)];
        let first = super::trail_mesh(&ctx, key, options, one.into_iter());
        let again = super::trail_mesh(&ctx, key, options, one.into_iter());
        assert!(std::sync::Arc::ptr_eq(&first, &again), "nothing new: the same mesh");
        let grown = super::trail_mesh(&ctx, key, options, two.into_iter());
        assert!(grown.vertices.len() > first.vertices.len() && grown.vertices.starts_with(&first.vertices), "the new stamp is appended");
        let other = super::trail_mesh(&ctx, key, options, [(50.0, 50.0, 6.0)].into_iter());
        assert_eq!(other.vertices.len(), first.vertices.len(), "another stroke starts over");
    }

    /// A new stamp adds to the kept mesh in place: once the frame that
    /// painted it lets go, nothing else holds it, so a long stroke is not
    /// copied whole every frame it grows.
    #[test]
    fn a_growing_trail_is_not_copied() {
        let ctx = egui::Context::default();
        let key = super::TrailKey { shape: prunr_core::brush::BrushShape::Circle, hardness: 0.5, accent: egui::Color32::RED, alpha: 100, ppp: 1.0 };
        let options = egui::epaint::TessellationOptions::default();
        let painted = super::trail_mesh(&ctx, key, options, [(10.0, 10.0, 6.0)].into_iter());
        let kept = std::sync::Arc::as_ptr(&painted);
        drop(painted);
        let grown = super::trail_mesh(&ctx, key, options, [(10.0, 10.0, 6.0), (14.0, 10.0, 6.0)].into_iter());
        assert_eq!(std::sync::Arc::as_ptr(&grown), kept, "the same mesh, added to");
    }

    use super::*;

    /// Pin the screen→model coordinate transform. A bug here flips an
    /// axis silently — strokes paint mirrored or off-by-half-image.
    #[test]
    fn screen_to_model_corners_and_center() {
        let img_rect = Rect::from_min_size(Pos2::new(100.0, 200.0), egui::vec2(400.0, 300.0));
        let (mw, mh) = (320u32, 240u32);

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
