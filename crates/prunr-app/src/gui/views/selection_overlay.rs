//! Selection visualization overlay. 60 Hz, zero-alloc in render closure.
//!
//! Reads pre-computed `BatchItem.selection_texture` + `BatchItem.selection_outline`.
//! Texture build + outline polyline run off-thread (see `background_io.rs`
//! `request_selection_visualization`). This file is RENDER-ONLY — no I/O,
//! no decode, no GPU upload.

use egui::{Color32, Pos2, Rect, Stroke, Ui};

use crate::gui::brush_state::BrushSettings;
use crate::gui::item::BatchItem;
use crate::gui::theme::ACCENT;

/// Called from canvas render after the item image is painted.
/// No allocation except the per-frame Vec<Pos2> transform of the pre-built
/// outline (O(N) over existing data). No decode, no GPU upload, no I/O.
///
/// `img_rect` — the on-screen rect of the displayed texture (post-zoom).
/// `src_w / src_h` — source image dimensions for the pixel→screen transform.
pub(crate) fn render_selection_overlay(
    ui: &Ui,
    item: &BatchItem,
    brush: &BrushSettings,
    img_rect: Rect,
    src_w: u32,
    src_h: u32,
) {
    let Some(_) = item.selection_mask.as_ref() else { return };

    let painter = ui.painter();

    // Pre-built selection_texture carries ACCENT pixels at opacity 255;
    // fill_opacity is applied at render time via the image tint alpha so
    // this closure stays alloc-free.
    if brush.fill_opacity > 0.001 {
        if let Some(tex) = item.selection_texture.as_ref() {
            let alpha = (brush.fill_opacity * 255.0).clamp(0.0, 255.0) as u8;
            let tint = Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), alpha);
            painter.image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                tint,
            );
        }
    }

    if brush.outline_opacity > 0.001 && brush.outline_thickness > 0.001 {
        if let Some(outline) = item.selection_outline.as_ref() {
            if !outline.is_empty() {
                let alpha = (brush.outline_opacity * 255.0).clamp(0.0, 255.0) as u8;
                let stroke_color = Color32::from_rgba_unmultiplied(
                    ACCENT.r(), ACCENT.g(), ACCENT.b(), alpha,
                );
                let stroke = Stroke::new(brush.outline_thickness, stroke_color);

                // Source → screen transform: image occupies `img_rect` at
                // `src_w x src_h` logical pixels; each source pixel maps
                // linearly. The outline is capped at OUTLINE_MAX_POINTS
                // in `background_io::request_selection_visualization`, so
                // this per-frame Vec<Pos2> is bounded.
                let iw = img_rect.width() / src_w.max(1) as f32;
                let ih = img_rect.height() / src_h.max(1) as f32;
                let ox = img_rect.min.x;
                let oy = img_rect.min.y;

                let pts: Vec<Pos2> = outline.iter()
                    .map(|&(px, py)| Pos2::new(ox + px as f32 * iw, oy + py as f32 * ih))
                    .collect();

                painter.add(egui::Shape::line(pts, stroke));
            }
        }
    }
}
