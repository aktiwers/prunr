use image::{GrayImage, RgbaImage};
use crate::formats::resize_gray_lanczos3;

/// Sized to the tiler's RGB output, so (R, G, B, A) recombine without
/// further resampling.
pub fn upscale_alpha_lanczos3(rgba: &RgbaImage, out_w: u32, out_h: u32) -> GrayImage {
    resize_gray_lanczos3(&crate::formats::alpha_plane(rgba), out_w, out_h)
}

/// Give `out` the alpha of `input`, the image it was upscaled from,
/// resampled to its size. An opaque input stays opaque without one: the
/// resample of a flat plane the output's size costs seconds on a large
/// upscale and gives the same 255.
pub(crate) fn take_input_alpha(out: &mut RgbaImage, input: &RgbaImage) {
    if crate::formats::is_opaque(input) {
        out.pixels_mut().for_each(|p| p.0[3] = u8::MAX);
        return;
    }
    let alpha = upscale_alpha_lanczos3(input, out.width(), out.height());
    for (p, &a) in out.pixels_mut().zip(alpha.as_raw()) {
        p.0[3] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An opaque input's upscale is opaque everywhere, edges and all; a
    /// see-through one still takes its alpha resampled.
    #[test]
    fn an_opaque_input_stays_opaque() {
        let opaque = RgbaImage::from_pixel(37, 23, image::Rgba([9, 8, 7, 255]));
        let mut out = RgbaImage::new(37 * 4, 23 * 4);
        take_input_alpha(&mut out, &opaque);
        assert!(out.pixels().all(|p| p[3] == 255));
        let mut half = opaque.clone();
        half.put_pixel(0, 0, image::Rgba([9, 8, 7, 0]));
        take_input_alpha(&mut out, &half);
        assert!(out.get_pixel(0, 0)[3] < 128 && out.get_pixel(100, 60)[3] == 255);
    }

    #[test]
    fn alpha_lanczos3_preserves_uniform_input() {
        let rgba = RgbaImage::from_pixel(64, 32, image::Rgba([0, 0, 0, 200]));
        let out = upscale_alpha_lanczos3(&rgba, 256, 128);
        assert_eq!(out.dimensions(), (256, 128));
        // Lanczos3 rings a little at the edges; the flat centre is exact
        // within u8 rounding.
        let center = out.get_pixel(128, 64).0[0];
        assert!((center as i32 - 200).abs() <= 2, "expected ~200, got {center}");
    }
}
