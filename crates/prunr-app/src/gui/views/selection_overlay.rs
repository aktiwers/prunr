//! Selection visualization overlay. 60 Hz, zero-alloc in render closure.
//!
//! Reads pre-computed `BatchItem.selection_texture` + `BatchItem.selection_outline`.
//! Texture build + outline polyline run off-thread (see `background_io.rs`
//! `request_selection_visualization`). This file is RENDER-ONLY — no I/O,
//! no decode, no GPU upload.

use egui::{Color32, Pos2, Rect, Ui};

use crate::gui::brush_state::BrushSettings;
use crate::gui::item::BatchItem;
use crate::gui::theme::ACCENT;

/// Called from canvas render after the item image is painted.
/// No allocation in the render closure — outline rendering iterates the
/// pre-built `Vec<(u32, u32)>` and emits one `rect_filled` shape per
/// boundary pixel directly into the painter.
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
                let color = Color32::from_rgba_unmultiplied(
                    ACCENT.r(), ACCENT.g(), ACCENT.b(), alpha,
                );

                // Source → screen transform: image occupies `img_rect` at
                // `src_w x src_h` logical pixels; each source pixel maps
                // linearly.
                let iw = img_rect.width() / src_w.max(1) as f32;
                let ih = img_rect.height() / src_h.max(1) as f32;
                let ox = img_rect.min.x;
                let oy = img_rect.min.y;

                // One filled rect per boundary pixel. Replacing the old
                // `Shape::line` polyline rendering — the polyline drew a
                // continuous line between EVERY consecutive pair of boundary
                // pixels in row-major scan order, which produced spurious
                // connections in two pathological cases:
                //   1. Multiple disjoint selected regions: the polyline
                //      bridged them with a line through empty space.
                //   2. Inverted mask (most pixels selected): the row-major
                //      scan jumps left-to-right on every row, producing a
                //      spider-web of full-image-width lines that flooded the
                //      canvas with purple at outline_opacity=1.0.
                // Per-pixel rects have no connection geometry at all — they
                // form a continuous outline because boundary pixels are
                // adjacent in source space; disjoint regions stay visually
                // disjoint with no extra lines.
                //
                // Rect side length: outline_thickness, but never smaller than
                // the source-pixel-to-screen-pixel scale, so adjacent
                // boundary pixels' rects touch (or overlap) at any zoom.
                // Without this floor, zooming in beyond 1.0× would produce
                // a visibly dotted outline.
                let pixel_scale = iw.max(ih).max(1.0);
                let side = brush.outline_thickness.max(pixel_scale);
                let half = side * 0.5;

                for &(px, py) in outline.iter() {
                    let cx = ox + (px as f32 + 0.5) * iw;
                    let cy = oy + (py as f32 + 0.5) * ih;
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(cx - half, cy - half),
                            Pos2::new(cx + half, cy + half),
                        ),
                        0.0,
                        color,
                    );
                }
            }
        }
    }
}
