use image::DynamicImage;
use ndarray::Array4;

use crate::formats::resize_rgb_lanczos3;
use crate::types::ModelKind;

const REMBG_SIZE: u32 = 320;
const BIREFNET_SIZE: u32 = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD:  [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess for the given model. Returns NCHW tensor at the model's expected resolution.
pub fn preprocess(img: &DynamicImage, model: ModelKind) -> Array4<f32> {
    match model {
        ModelKind::Silueta | ModelKind::U2net => preprocess_rembg(img),
        ModelKind::BiRefNetLite => preprocess_birefnet(img),
    }
}

/// Build NCHW tensor from a resized image with the given divisor.
fn to_nchw(resized: &image::RgbImage, size: u32, divisor: f32) -> Array4<f32> {
    let s = size as usize;
    let raw = resized.as_raw();
    let inv_div = 1.0 / divisor;
    let scale: [f32; 3] = std::array::from_fn(|c| inv_div / STD[c]);
    let bias: [f32; 3] = std::array::from_fn(|c| MEAN[c] / STD[c]);

    let mut out = Array4::<f32>::zeros((1, 3, s, s));
    let plane_size = s * s;
    // invariant: `out` is a freshly allocated `Array4` (standard layout, contiguous).
    let out_slice = out.as_slice_mut().unwrap();
    for i in 0..plane_size {
        let base = i * 3;
        out_slice[i] = raw[base] as f32 * scale[0] - bias[0];
        out_slice[plane_size + i] = raw[base + 1] as f32 * scale[1] - bias[1];
        out_slice[plane_size * 2 + i] = raw[base + 2] as f32 * scale[2] - bias[2];
    }
    out
}

fn preprocess_rembg(img: &DynamicImage) -> Array4<f32> {
    let resized = resize_rgb_lanczos3(img, REMBG_SIZE, REMBG_SIZE);
    let max_val = resized
        .pixels()
        .flat_map(|p| p.0.iter().copied())
        .map(|v| v as f32)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-6_f32);
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

    #[test]
    fn test_preprocess_output_shape() {
        let img = solid_rgb_image(128, 64, 32, 640, 480);
        let tensor = preprocess(&img, ModelKind::Silueta);
        assert_eq!(tensor.shape(), &[1, 3, 320, 320]);
    }

    #[test]
    fn test_preprocess_birefnet_shape() {
        let img = solid_rgb_image(128, 64, 32, 640, 480);
        let tensor = preprocess(&img, ModelKind::BiRefNetLite);
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
        for &val in tensor.iter() {
            assert!(val.is_finite(), "BiRefNet tensor contains NaN/Inf: {val}");
        }
    }

    #[test]
    fn test_preprocess_values_normalized() {
        let img = solid_rgb_image(100, 150, 200, 320, 320);
        let tensor = preprocess(&img, ModelKind::Silueta);
        for &val in tensor.iter() {
            assert!(val.is_finite(), "Tensor contains NaN or Inf: {val}");
            assert!(val > -5.0 && val < 5.0, "Value out of expected range: {val}");
        }
    }

    #[test]
    fn test_preprocess_channel_order() {
        // Solid red image: R=255, G=0, B=0
        // After normalization, channel 0 (R) should be much larger than channels 1,2
        let img = solid_rgb_image(255, 0, 0, 64, 64);
        let tensor = preprocess(&img, ModelKind::Silueta);
        let r_center = tensor[[0, 0, 160, 160]];
        let g_center = tensor[[0, 1, 160, 160]];
        assert!(r_center > g_center, "Channel 0 (R) should be normalized higher than channel 1 (G) for red image");
    }

    #[test]
    fn test_preprocess_black_image_no_nan() {
        // All-black image: max_val = 1e-6 (clamped), pixel/max_val = 0.0
        // Result = (0.0 - MEAN[c]) / STD[c] — all finite
        let img = solid_rgb_image(0, 0, 0, 32, 32);
        let tensor = preprocess(&img, ModelKind::Silueta);
        for &val in tensor.iter() {
            assert!(val.is_finite(), "Black image produced NaN or Inf: {val}");
        }
    }
}
