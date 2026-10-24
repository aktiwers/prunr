use image::{GrayImage, Luma, RgbaImage};
use crate::formats::resize_gray_lanczos3;

/// Sized to the tiler's RGB output, so (R, G, B, A) recombine without
/// further resampling.
pub fn upscale_alpha_lanczos3(rgba: &RgbaImage, out_w: u32, out_h: u32) -> GrayImage {
    let alpha = GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        Luma([rgba.get_pixel(x, y).0[3]])
    });
    resize_gray_lanczos3(&alpha, out_w, out_h)
}

/// Give `out` the alpha of `input`, the image it was upscaled from,
/// resampled to its size.
pub(crate) fn take_input_alpha(out: &mut RgbaImage, input: &RgbaImage) {
    let alpha = upscale_alpha_lanczos3(input, out.width(), out.height());
    for (p, &a) in out.pixels_mut().zip(alpha.as_raw()) {
        p.0[3] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
