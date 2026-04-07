use egui::{Color32, ColorImage, Vec2};
use image::RgbaImage;

use crate::gui::theme;

/// Build an animation frame: subject pixels at full opacity, background pixels
/// fading from original alpha to 0 based on `progress` (0.0 = start, 1.0 = done).
///
/// `anim_mask` contains the alpha channel of the result image:
///   - alpha > ANIM_MASK_THRESHOLD = subject (keep sharp)
///   - alpha <= ANIM_MASK_THRESHOLD = background (dissolve)
///
/// For performance, the output is capped to `max_size` pixels.
pub fn build_animation_frame(
    source_rgba: &RgbaImage,
    result_rgba: &RgbaImage,
    anim_mask: &[u8],
    progress: f32,
    max_width: u32,
    max_height: u32,
) -> ColorImage {
    let (src_w, src_h) = (source_rgba.width(), source_rgba.height());

    // Determine output size (downscale if source is larger than canvas)
    let scale = (max_width as f32 / src_w as f32)
        .min(max_height as f32 / src_h as f32)
        .min(1.0);
    let out_w = ((src_w as f32 * scale) as u32).max(1);
    let out_h = ((src_h as f32 * scale) as u32).max(1);

    // Ease-out cubic: t = 1 - (1-t)^3
    let t = 1.0 - (1.0 - progress).powi(3);

    let mut pixels = Vec::with_capacity((out_w * out_h) as usize);

    for y in 0..out_h {
        for x in 0..out_w {
            // Map output pixel to source pixel
            let sx = ((x as f32 / scale) as u32).min(src_w - 1);
            let sy = ((y as f32 / scale) as u32).min(src_h - 1);
            let idx = (sy * src_w + sx) as usize;

            let mask_alpha = anim_mask[idx];

            if mask_alpha > theme::ANIM_MASK_THRESHOLD {
                // Subject pixel — always show at full opacity from result
                let p = result_rgba.get_pixel(sx, sy);
                pixels.push(Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]));
            } else {
                // Background pixel — fade from source to transparent
                let p = source_rgba.get_pixel(sx, sy);
                let faded_alpha = ((p[3] as f32) * (1.0 - t)) as u8;
                pixels.push(Color32::from_rgba_unmultiplied(p[0], p[1], p[2], faded_alpha));
            }
        }
    }

    ColorImage {
        size: [out_w as usize, out_h as usize],
        source_size: Vec2::new(out_w as f32, out_h as f32),
        pixels,
    }
}
