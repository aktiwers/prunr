//! Guided filter for alpha mask refinement.
//!
//! Uses the original image as a guide to refine the AI-generated mask,
//! producing better edges around fine detail (hair, leaves, etc.).
//!
//! Reference: He, Sun, Tang — "Guided Image Filtering" (2013)

use image::{GrayImage, RgbaImage};

/// Refine an alpha mask using a guided filter.
///
/// - `guide`: original RGBA image (used for color-based edge guidance)
/// - `mask`: AI-generated grayscale alpha mask (same dimensions as guide)
/// - `radius`: filter window radius in pixels (larger = smoother)
/// - `epsilon`: regularization (smaller = sharper edges, larger = smoother)
///
/// Returns a refined GrayImage mask with the same dimensions.
pub fn guided_filter_alpha(
    guide: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
) -> GrayImage {
    let (w, h) = (mask.width(), mask.height());
    let n = (w * h) as usize;

    // Convert guide to grayscale float [0,1] and mask to float [0,1]
    let mut guide_f = vec![0.0f32; n];
    let mut mask_f = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let gp = guide.get_pixel(x, y);
            // Luminance from RGB
            guide_f[idx] = (0.299 * gp[0] as f32 + 0.587 * gp[1] as f32 + 0.114 * gp[2] as f32) / 255.0;
            mask_f[idx] = mask.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    // Box filter helper (integral image based)
    let mean_i = box_filter(&guide_f, w, h, radius);
    let mean_p = box_filter(&mask_f, w, h, radius);

    // I*I and I*p
    let mut ii = vec![0.0f32; n];
    let mut ip = vec![0.0f32; n];
    for i in 0..n {
        ii[i] = guide_f[i] * guide_f[i];
        ip[i] = guide_f[i] * mask_f[i];
    }
    let mean_ii = box_filter(&ii, w, h, radius);
    let mean_ip = box_filter(&ip, w, h, radius);

    // Compute a and b for each pixel
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        let var_i = mean_ii[i] - mean_i[i] * mean_i[i];
        let cov_ip = mean_ip[i] - mean_i[i] * mean_p[i];
        a[i] = cov_ip / (var_i + epsilon);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }

    // Average a and b over the window
    let mean_a = box_filter(&a, w, h, radius);
    let mean_b = box_filter(&b, w, h, radius);

    // Compute output: q = mean_a * I + mean_b
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let val = (mean_a[idx] * guide_f[idx] + mean_b[idx]).clamp(0.0, 1.0);
            out.put_pixel(x, y, image::Luma([(val * 255.0) as u8]));
        }
    }
    out
}

/// O(1) box filter using integral image (cumulative sum).
fn box_filter(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let r = radius as i64;
    let n = w * h;

    // Build integral image
    let mut integral = vec![0.0f64; n];
    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += src[y * w + x] as f64;
            integral[y * w + x] = row_sum + if y > 0 { integral[(y - 1) * w + x] } else { 0.0 };
        }
    }

    let get = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 { return 0.0; }
        let x = (x as usize).min(w - 1);
        let y = (y as usize).min(h - 1);
        integral[y * w + x]
    };

    let mut out = vec![0.0f32; n];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let x1 = (x - r - 1).max(-1);
            let y1 = (y - r - 1).max(-1);
            let x2 = (x + r).min(w as i64 - 1);
            let y2 = (y + r).min(h as i64 - 1);
            let area = (x2 - x1) as f64 * (y2 - y1) as f64;
            let sum = get(x2, y2) - get(x1, y2) - get(x2, y1) + get(x1, y1);
            out[y as usize * w + x as usize] = (sum / area.max(1.0)) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, Luma};

    #[test]
    fn test_guided_filter_preserves_dimensions() {
        let guide = RgbaImage::from_pixel(64, 48, Rgba([128, 128, 128, 255]));
        let mask = GrayImage::from_pixel(64, 48, Luma([200]));
        let result = guided_filter_alpha(&guide, &mask, 4, 0.01);
        assert_eq!(result.width(), 64);
        assert_eq!(result.height(), 48);
    }

    #[test]
    fn test_guided_filter_uniform_mask_unchanged() {
        let guide = RgbaImage::from_pixel(32, 32, Rgba([100, 150, 200, 255]));
        let mask = GrayImage::from_pixel(32, 32, Luma([255]));
        let result = guided_filter_alpha(&guide, &mask, 4, 0.01);
        // Uniform white mask should stay ~white everywhere
        for p in result.pixels() {
            assert!(p[0] >= 250, "Expected ~255, got {}", p[0]);
        }
    }

    #[test]
    fn test_box_filter_uniform_is_identity() {
        let data = vec![0.5f32; 16];
        let result = box_filter(&data, 4, 4, 1);
        for &v in &result {
            assert!((v - 0.5).abs() < 0.01, "Expected ~0.5, got {v}");
        }
    }
}
