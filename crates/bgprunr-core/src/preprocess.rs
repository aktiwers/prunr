use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;

pub const TARGET_SIZE: u32 = 320;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD:  [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess a dynamic image into a 4D NCHW float tensor for rembg-compatible ONNX models.
///
/// Matches rembg's normalize() from rembg/sessions/base.py exactly:
/// 1. Convert to RGB8
/// 2. Resize to 320x320 using Lanczos3 (rembg: Image.Resampling.LANCZOS)
/// 3. Divide by max(max_pixel, 1e-6)  — NOT 255.0 unconditionally
/// 4. Subtract ImageNet mean per channel, divide by std
/// 5. Arrange in NCHW order: [1, 3, 320, 320]
pub fn preprocess(img: &DynamicImage) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(&rgb, TARGET_SIZE, TARGET_SIZE, FilterType::Lanczos3);

    // rembg: im_ary = im_ary / max(np.max(im_ary), 1e-6)
    let max_val = resized
        .pixels()
        .flat_map(|p| p.0.iter().copied())
        .map(|v| v as f32)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-6_f32);

    let s = TARGET_SIZE as usize;
    let mut out = Array4::<f32>::zeros((1, 3, s, s));
    for y in 0..s {
        for x in 0..s {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                out[[0, c, y, x]] = (p[c] as f32 / max_val - MEAN[c]) / STD[c];
            }
        }
    }
    out
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
        let tensor = preprocess(&img);
        assert_eq!(tensor.shape(), &[1, 3, 320, 320]);
    }

    #[test]
    fn test_preprocess_values_normalized() {
        let img = solid_rgb_image(100, 150, 200, 320, 320);
        let tensor = preprocess(&img);
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
        let tensor = preprocess(&img);
        let r_center = tensor[[0, 0, 160, 160]];
        let g_center = tensor[[0, 1, 160, 160]];
        assert!(r_center > g_center, "Channel 0 (R) should be normalized higher than channel 1 (G) for red image");
    }

    #[test]
    fn test_preprocess_black_image_no_nan() {
        // All-black image: max_val = 1e-6 (clamped), pixel/max_val = 0.0
        // Result = (0.0 - MEAN[c]) / STD[c] — all finite
        let img = solid_rgb_image(0, 0, 0, 32, 32);
        let tensor = preprocess(&img);
        for &val in tensor.iter() {
            assert!(val.is_finite(), "Black image produced NaN or Inf: {val}");
        }
    }
}
