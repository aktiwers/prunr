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
pub fn guided_filter_alpha(
    guide: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
) -> GrayImage {
    let (w, h) = (mask.width(), mask.height());
    let n = (w * h) as usize;

    // Two reusable scratchpads — avoids allocating 6 separate Vecs
    let mut buf_a = vec![0.0f32; n];
    let mut buf_b = vec![0.0f32; n];
    let mut integral = vec![0.0f64; n]; // reused by all box_filter calls

    // Convert guide to grayscale float [0,1] and mask to float [0,1]
    let mut guide_f = vec![0.0f32; n];
    let mut mask_f = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let gp = guide.get_pixel(x, y);
            guide_f[idx] = (0.299 * gp[0] as f32 + 0.587 * gp[1] as f32 + 0.114 * gp[2] as f32) / 255.0;
            mask_f[idx] = mask.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    // mean_I and mean_p
    box_filter(&guide_f, w, h, radius, &mut integral, &mut buf_a);
    let mean_i = buf_a.clone(); // need to keep for later
    box_filter(&mask_f, w, h, radius, &mut integral, &mut buf_b);
    let mean_p = buf_b.clone();

    // I*I → buf_a, I*p → buf_b, then box filter each
    for i in 0..n {
        buf_a[i] = guide_f[i] * guide_f[i];
        buf_b[i] = guide_f[i] * mask_f[i];
    }
    let ii_src = buf_a.clone();
    let ip_src = buf_b.clone();
    box_filter(&ii_src, w, h, radius, &mut integral, &mut buf_a); // mean_ii in buf_a
    box_filter(&ip_src, w, h, radius, &mut integral, &mut buf_b); // mean_ip in buf_b

    // Compute a and b in-place (reuse guide_f and mask_f which are no longer needed)
    for i in 0..n {
        let var_i = buf_a[i] - mean_i[i] * mean_i[i];
        let cov_ip = buf_b[i] - mean_i[i] * mean_p[i];
        guide_f[i] = cov_ip / (var_i + epsilon); // a
        mask_f[i] = mean_p[i] - guide_f[i] * mean_i[i]; // b
    }

    // Average a and b
    box_filter(&guide_f, w, h, radius, &mut integral, &mut buf_a); // mean_a
    box_filter(&mask_f, w, h, radius, &mut integral, &mut buf_b); // mean_b

    // Compute output: q = mean_a * I + mean_b
    // Recompute guide luminance from the original (guide_f was overwritten with 'a')
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let gp = guide.get_pixel(x, y);
            let lum = (0.299 * gp[0] as f32 + 0.587 * gp[1] as f32 + 0.114 * gp[2] as f32) / 255.0;
            let val = (buf_a[idx] * lum + buf_b[idx]).clamp(0.0, 1.0);
            out.put_pixel(x, y, image::Luma([(val * 255.0) as u8]));
        }
    }
    out
}

/// O(1) box filter using integral image. Writes result into `out`.
fn box_filter(src: &[f32], w: u32, h: u32, radius: u32, integral: &mut [f64], out: &mut [f32]) {
    let w = w as usize;
    let h = h as usize;
    let r = radius as i64;

    // Build integral image (reuses caller's buffer)
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
        for p in result.pixels() {
            assert!(p[0] >= 250, "Expected ~255, got {}", p[0]);
        }
    }

    #[test]
    fn test_box_filter_uniform_is_identity() {
        let data = vec![0.5f32; 16];
        let mut integral = vec![0.0f64; 16];
        let mut out = vec![0.0f32; 16];
        box_filter(&data, 4, 4, 1, &mut integral, &mut out);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.01, "Expected ~0.5, got {v}");
        }
    }
}
