//! A sticker outline and a shadow around a cut-out, drawn under it.
//!
//! Works on straight-alpha sRGB like the rest of the pipeline. The outline
//! is the subject's alpha grown by a disc of its width (round corners, and
//! its rim keeps the subject's own soft edge), filled with the outline
//! colour. The shadow is the alpha of subject and outline blurred by three
//! box passes (close to a Gaussian), moved down for a Drop shadow, and
//! filled with black at the chosen opacity.
//!
//! Only the subject's bbox plus the decoration's reach is touched; pixels
//! farther out cannot change.
//!
//! Peak working set for a bbox of n pixels: the RGBA crop (4n), the
//! grown alpha (n) and two f32 planes for the shadow (8n), freed before
//! the crop is written back.

use image::{GrayImage, RgbaImage};

use crate::types::{Decoration, ShadowKind};

pub fn apply_decoration(rgba: &mut RgbaImage, d: &Decoration) {
    if !d.has_outline() && !d.has_shadow() {
        return;
    }
    let (w, h) = rgba.dimensions();
    let Some((x0, y0, x1, y1)) = alpha_bbox(rgba) else { return };
    let reach = d.reach_px();
    let (bx, by) = (x0.saturating_sub(reach), y0.saturating_sub(reach));
    let (ex, ey) = ((x1 + reach + 1).min(w), (y1 + reach + 1).min(h));
    let mut crop = image::imageops::crop_imm(rgba, bx, by, ex - bx, ey - by).to_image();
    if d.has_outline() {
        add_outline(&mut crop, d.outline_width, d.outline_color);
    }
    if d.has_shadow() {
        add_shadow(&mut crop, d);
    }
    image::imageops::replace(rgba, &crop, bx as i64, by as i64);
}

/// Inclusive bounds of the pixels with any alpha.
fn alpha_bbox(rgba: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let w = rgba.width() as usize;
    let mut bounds: Option<(usize, usize, usize, usize)> = None;
    for (i, px) in rgba.as_raw().chunks_exact(4).enumerate() {
        if px[3] == 0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        bounds = Some(match bounds {
            None => (x, y, x, y),
            Some((a, b, c, e)) => (a.min(x), b.min(y), c.max(x), e.max(y)),
        });
    }
    bounds.map(|(a, b, c, e)| (a as u32, b as u32, c as u32, e as u32))
}

fn alpha_plane(rgba: &RgbaImage) -> GrayImage {
    let alpha = rgba.as_raw().chunks_exact(4).map(|px| px[3]).collect();
    // invariant: one byte per pixel of the same image.
    GrayImage::from_raw(rgba.width(), rgba.height(), alpha).expect("one alpha byte per pixel")
}

/// `rgba` composited over a layer of `color` at coverage `under`
/// (straight alpha, 0–1 per pixel).
fn composite_over(rgba: &mut RgbaImage, color: [u8; 3], under: impl Fn(usize) -> f32) {
    for (i, px) in rgba.as_mut().chunks_exact_mut(4).enumerate() {
        let top = px[3] as f32 / 255.0;
        let below = under(i) * (1.0 - top);
        let out = top + below;
        if out <= 0.0 {
            continue;
        }
        for c in 0..3 {
            px[c] = ((px[c] as f32 * top + color[c] as f32 * below) / out).round().clamp(0.0, 255.0) as u8;
        }
        px[3] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

fn add_outline(rgba: &mut RgbaImage, width: f32, color: [u8; 3]) {
    let grown = crate::morphology::dilate_round(&alpha_plane(rgba), width.round() as u32);
    let grown = grown.as_raw();
    composite_over(rgba, color, |i| grown[i] as f32 / 255.0);
}

fn add_shadow(rgba: &mut RgbaImage, d: &Decoration) {
    let (w, h) = rgba.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let fall = if d.shadow == ShadowKind::Drop { d.shadow_distance.round().max(0.0) as usize } else { 0 };
    let raw = rgba.as_raw();
    // The silhouette moved down by `fall`; rows above it start empty.
    let mut cast = vec![0.0f32; wu * hu];
    for y in fall..hu {
        for x in 0..wu {
            cast[y * wu + x] = raw[((y - fall) * wu + x) * 4 + 3] as f32 / 255.0;
        }
    }
    let radius = (d.shadow_blur / 2.0).round() as u32;
    if radius > 0 {
        let mut tmp = vec![0.0f32; cast.len()]; // reused as each pass's output
        for _ in 0..3 {
            crate::guided_filter::box_filter_into(&cast, w, h, radius, &mut tmp);
            std::mem::swap(&mut cast, &mut tmp);
        }
    }
    let opacity = d.shadow_opacity.clamp(0.0, 1.0);
    composite_over(rgba, [0, 0, 0], |i| cast[i] * opacity);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An opaque red square in the middle of a transparent canvas.
    fn square() -> RgbaImage {
        let mut img = RgbaImage::new(64, 64);
        for y in 24..40 {
            for x in 24..40 {
                img.put_pixel(x, y, image::Rgba([200, 0, 0, 255]));
            }
        }
        img
    }

    #[test]
    fn no_decoration_leaves_the_image_untouched() {
        let mut img = square();
        apply_decoration(&mut img, &Decoration::default());
        assert_eq!(img, square());
    }

    /// The outline is an opaque ring of its width and colour around the
    /// subject, rounded at the corners, and the subject stays as it was.
    #[test]
    fn the_outline_rings_the_subject_in_its_color() {
        let mut img = square();
        let d = Decoration { outline_width: 4.0, outline_color: [255, 255, 255], ..Decoration::default() };
        apply_decoration(&mut img, &d);
        assert_eq!(img.get_pixel(30, 30).0, [200, 0, 0, 255], "inside is the subject");
        assert_eq!(img.get_pixel(21, 30).0, [255, 255, 255, 255], "3 px out is outline");
        assert_eq!(img.get_pixel(18, 30).0[3], 0, "past the width is clear");
        assert_eq!(img.get_pixel(20, 20).0[3], 0, "the corner is round, not square");
    }

    /// A Drop shadow darkens below the subject and not above it; a Soft
    /// one reaches every side.
    #[test]
    fn a_drop_shadow_falls_below_and_a_soft_one_all_around() {
        let drop = Decoration { shadow: ShadowKind::Drop, shadow_opacity: 0.5, shadow_blur: 4.0, shadow_distance: 8.0, ..Decoration::default() };
        let mut img = square();
        apply_decoration(&mut img, &drop);
        assert!(img.get_pixel(32, 45).0[3] > 60, "below the subject is shadow");
        assert_eq!(img.get_pixel(32, 20).0[3], 0, "above it is clear");
        assert_eq!(img.get_pixel(32, 45).0[..3], [0, 0, 0], "the shadow is black");
        assert_eq!(img.get_pixel(30, 30).0, [200, 0, 0, 255], "the subject covers its shadow");

        let soft = Decoration { shadow: ShadowKind::Soft, ..drop };
        let mut img = square();
        apply_decoration(&mut img, &soft);
        for (x, y) in [(32, 21), (32, 42), (21, 32), (42, 32)] {
            assert!(img.get_pixel(x, y).0[3] > 0, "soft shadow at ({x}, {y})");
        }
    }

    /// The bbox crop changes nothing: the same subject shifted within a
    /// larger canvas gets the same decoration.
    #[test]
    fn the_result_does_not_depend_on_where_the_subject_sits() {
        let d = Decoration { outline_width: 3.0, shadow: ShadowKind::Drop, shadow_opacity: 0.6, shadow_blur: 6.0, shadow_distance: 5.0, ..Decoration::default() };
        let mut small = square();
        apply_decoration(&mut small, &d);
        let mut big = RgbaImage::new(128, 128);
        image::imageops::replace(&mut big, &square(), 40, 40);
        apply_decoration(&mut big, &d);
        let back = image::imageops::crop_imm(&big, 40, 40, 64, 64).to_image();
        assert_eq!(back, small);
    }
}
