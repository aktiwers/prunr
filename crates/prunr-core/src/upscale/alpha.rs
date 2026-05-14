use image::{GrayImage, Luma, RgbaImage};
use crate::formats::resize_gray_lanczos3;

/// Upscale the alpha channel of an RGBA image to (out_w, out_h) using
/// Lanczos3. The result matches the dimensions of the RGB output produced
/// by the tiler, so the caller can recombine (R, G, B, A) without further
/// resampling.
pub fn upscale_alpha_lanczos3(rgba: &RgbaImage, out_w: u32, out_h: u32) -> GrayImage {
    let alpha = GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        Luma([rgba.get_pixel(x, y).0[3]])
    });
    resize_gray_lanczos3(&alpha, out_w, out_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_lanczos3_produces_correct_output_dimensions() {
        let rgba = RgbaImage::from_pixel(200, 100, image::Rgba([0, 0, 0, 128]));
        let out = upscale_alpha_lanczos3(&rgba, 800, 400);
        assert_eq!(out.width(), 800);
        assert_eq!(out.height(), 400);
    }

    #[test]
    fn alpha_lanczos3_preserves_uniform_input() {
        let rgba = RgbaImage::from_pixel(64, 64, image::Rgba([0, 0, 0, 200]));
        let out = upscale_alpha_lanczos3(&rgba, 256, 256);
        // Sample center; Lanczos3 has minor ringing at edges but center is
        // flat-region exact within u8 rounding.
        let center = out.get_pixel(128, 128).0[0];
        assert!((center as i32 - 200).abs() <= 2, "expected ~200, got {center}");
    }
}
