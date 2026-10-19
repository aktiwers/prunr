use std::mem::MaybeUninit;

use image::DynamicImage;
use ndarray::Array4;

use crate::formats::resize_rgb_lanczos3;
use crate::types::ModelKind;

const REMBG_SIZE: u32 = 320;
const BIREFNET_SIZE: u32 = 1024;

/// Standard ImageNet normalization used by every vision backbone in this crate.
pub(crate) const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub(crate) const IMAGENET_STD:  [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess for the given model. Returns NCHW tensor at the model's expected resolution.
pub fn preprocess(img: &DynamicImage, model: ModelKind) -> Array4<f32> {
    match model {
        ModelKind::Silueta | ModelKind::U2net => preprocess_rembg(img),
        ModelKind::BiRefNetLite => preprocess_birefnet(img),
        // Upscale models run through a separate tiled RGB pipeline; they
        // never enter the segmentation preprocess path.
        ModelKind::RealEsrganX4Plus
        | ModelKind::Nomos8kSchatL
        | ModelKind::RealEsrganX2Plus
        | ModelKind::FourXNmkdSiaxCx
        | ModelKind::FourXNmkdSuperscale
        | ModelKind::FourXNomos2RealPlksr => {
            unreachable!("upscale models are dispatched via upscale_rgba, not the seg pipeline")
        }
        ModelKind::DdcolorTiny => unreachable!("the colorizer runs through colorize_plane, not the seg pipeline"),
        ModelKind::DepthAnythingV2Small => unreachable!("depth runs through depth::estimate_depth, not the seg pipeline"),
    }
}

/// Build NCHW tensor from a resized image with the given divisor.
fn to_nchw(resized: &image::RgbImage, size: u32, divisor: f32) -> Array4<f32> {
    let s = size as usize;
    let raw = resized.as_raw();
    let inv_div = 1.0 / divisor;
    let scale: [f32; 3] = std::array::from_fn(|c| inv_div / IMAGENET_STD[c]);
    let bias: [f32; 3] = std::array::from_fn(|c| IMAGENET_MEAN[c] / IMAGENET_STD[c]);

    // `Array4::zeros` writes 12 MB at BiRefNet 1024² before the loop
    // overwrites every element. `uninit` skips the zero-fill — saves
    // ~3 ms of memory bandwidth per dispatch (~3% of the 100 ms live
    // preview budget at 1024²). Loop body is interleaved (one source
    // pixel → three plane writes per iter) so `raw[base..base+3]` is
    // read once while it's still hot in L1; the previously-attempted
    // split-pass form did 3× source traversal for no vectorization
    // gain (stride-3 u8→f32 gather defeats LLVM's loop vectorizer
    // either way — verified with `--emit=asm`).
    let mut out: Array4<MaybeUninit<f32>> = Array4::uninit((1, 3, s, s));
    let plane_size = s * s;
    // invariant: `out` is a freshly allocated `Array4` (standard layout, contiguous).
    let out_slice = out.as_slice_mut().unwrap();
    for i in 0..plane_size {
        let base = i * 3;
        out_slice[i].write(raw[base] as f32 * scale[0] - bias[0]);
        out_slice[plane_size + i].write(raw[base + 1] as f32 * scale[1] - bias[1]);
        out_slice[plane_size * 2 + i].write(raw[base + 2] as f32 * scale[2] - bias[2]);
    }
    // Safety: the loop above writes every element of all three planes
    // (`plane_size * 3 == out.len()`). No element is skipped or read
    // before this point.
    unsafe { out.assume_init() }
}

fn preprocess_rembg(img: &DynamicImage) -> Array4<f32> {
    let resized = resize_rgb_lanczos3(img, REMBG_SIZE, REMBG_SIZE);
    // Clamp to 1e-6 so an all-black image doesn't divide by zero.
    let max_u8 = resized.as_raw().iter().copied().max().unwrap_or(0);
    let max_val = (max_u8 as f32).max(1e-6_f32);
    to_nchw(&resized, REMBG_SIZE, max_val)
}

fn preprocess_birefnet(img: &DynamicImage) -> Array4<f32> {
    let resized = resize_rgb_lanczos3(img, BIREFNET_SIZE, BIREFNET_SIZE);
    to_nchw(&resized, BIREFNET_SIZE, 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, Rgb};

    fn solid_rgb_image(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    /// Silueta takes 320², BiRefNet 1024²; values are finite and in the
    /// normalized range, black included (max_val clamps to 1e-6, not 0),
    /// and channel 0 is red.
    #[test]
    fn test_preprocess_shape_range_and_channel_order() {
        let silueta = preprocess(&solid_rgb_image(128, 64, 32, 640, 480), ModelKind::Silueta);
        assert_eq!(silueta.shape(), &[1, 3, 320, 320]);
        let birefnet = preprocess(&solid_rgb_image(128, 64, 32, 640, 480), ModelKind::BiRefNetLite);
        assert_eq!(birefnet.shape(), &[1, 3, 1024, 1024]);
        assert!(birefnet.iter().all(|v| v.is_finite()), "BiRefNet tensor contains NaN/Inf");
        for (r, g, b) in [(100, 150, 200), (0, 0, 0)] {
            let tensor = preprocess(&solid_rgb_image(r, g, b, 32, 32), ModelKind::Silueta);
            assert!(tensor.iter().all(|&v| v.is_finite() && v > -5.0 && v < 5.0), "({r}, {g}, {b})");
        }
        let red = preprocess(&solid_rgb_image(255, 0, 0, 64, 64), ModelKind::Silueta);
        assert!(red[[0, 0, 160, 160]] > red[[0, 1, 160, 160]], "channel 0 is R");
    }
}
