//! Tier-2 postprocess operations applied to the cached `upscale_raw`
//! RGBA buffer. Four pure functions operate in-place; they do NOT
//! re-allocate the image (the caller's RAM accounting depends on
//! this).
//!
//! All four operate on STRAIGHT (un-premultiplied) sRGB u8 per
//! CLAUDE.md ## Numerical & color invariants. Internal math uses
//! f32 working buffers; clamp+cast at the u8 boundary.
//!
//! HSL (not HSV) for saturation.
//! Reinhard Lab (not LMS) for color match, with proper IEC 61966-2-1
//! linearisation and D65 XYZ→Lab pipeline.

use image::RgbaImage;

// ── sRGB ↔ linear helpers ──────────────────────────────────────────────────

/// sRGB encoded → linear-light (IEC 61966-2-1).
#[inline(always)]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-light → sRGB encoded (IEC 61966-2-1 inverse).
#[inline(always)]
fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ── CIE XYZ ↔ Lab (D65) ──────────────────────────────────────────────────

/// Linear sRGB → CIE XYZ D65.
/// Bradford-adapted D65 matrix from IEC 61966-2-1.
#[inline(always)]
fn linear_rgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;
    (x, y, z)
}

/// CIE XYZ D65 → linear sRGB.
#[inline(always)]
fn xyz_to_linear_rgb(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r =  3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266 * x + 1.876_010_8 * y + 0.041_556_0 * z;
    let b =  0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    (r, g, b)
}

const D65_X: f32 = 0.950_489;
const D65_Y: f32 = 1.000_000;
const D65_Z: f32 = 1.088_84;

#[inline(always)]
fn lab_f(t: f32) -> f32 {
    // CIE standard cube-root nonlinearity
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA;           // ≈ 0.008856
    const INV3DELTA2: f32 = 1.0 / (3.0 * DELTA * DELTA); // ≈ 7.787
    if t > DELTA3 {
        t.cbrt()
    } else {
        INV3DELTA2 * t + (4.0 / 29.0)
    }
}

#[inline(always)]
fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// sRGB u8 pixel → CIE Lab (L in [0,100], a/b unbounded typical ±128).
#[inline(always)]
fn srgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (lr, lg, lb) = (
        srgb_to_linear(r as f32 / 255.0),
        srgb_to_linear(g as f32 / 255.0),
        srgb_to_linear(b as f32 / 255.0),
    );
    let (x, y, z) = linear_rgb_to_xyz(lr, lg, lb);
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_lab = 200.0 * (fy - fz);
    (l, a, b_lab)
}

/// CIE Lab → sRGB u8, clamped.
#[inline(always)]
fn lab_to_srgb(l: f32, a: f32, b_lab: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_lab / 200.0;
    let x = lab_f_inv(fx) * D65_X;
    let y = lab_f_inv(fy) * D65_Y;
    let z = lab_f_inv(fz) * D65_Z;
    let (lr, lg, lb) = xyz_to_linear_rgb(x, y, z);
    let r = (linear_to_srgb(lr) * 255.0).clamp(0.0, 255.0) as u8;
    let g = (linear_to_srgb(lg) * 255.0).clamp(0.0, 255.0) as u8;
    let b = (linear_to_srgb(lb) * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ── HSL helpers ────────────────────────────────────────────────────────────

/// sRGB u8 → HSL (H in [0,360), S in [0,1], L in [0,1]).
/// Operates on gamma-encoded sRGB values directly — no linearisation.
#[inline(always)]
fn srgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;
    let l = (max + min) / 2.0;
    let s = if chroma < 1e-10 {
        0.0
    } else {
        chroma / (1.0 - (2.0 * l - 1.0).abs())
    };
    let h = if chroma < 1e-10 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / chroma).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / chroma + 2.0)
    } else {
        60.0 * ((r - g) / chroma + 4.0)
    };
    (h, s, l)
}

/// HSL → sRGB u8.
#[inline(always)]
fn hsl_to_srgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let chroma = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = chroma * (1.0 - (h_prime.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = if h_prime < 1.0 {
        (chroma, x, 0.0)
    } else if h_prime < 2.0 {
        (x, chroma, 0.0)
    } else if h_prime < 3.0 {
        (0.0, chroma, x)
    } else if h_prime < 4.0 {
        (0.0, x, chroma)
    } else if h_prime < 5.0 {
        (x, 0.0, chroma)
    } else {
        (chroma, 0.0, x)
    };
    let m = l - chroma / 2.0;
    let r = ((r1 + m) * 255.0).clamp(0.0, 255.0) as u8;
    let g = ((g1 + m) * 255.0).clamp(0.0, 255.0) as u8;
    let b = ((b1 + m) * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ── Gaussian blur (separable 5-tap, σ ≈ 1.0) ──────────────────────────────

/// Separable 5-tap Gaussian kernel weights (σ ≈ 1.0).
/// Computed from the standard Gaussian; pinned here so live-preview
/// feel is stable across refactors.
const GAUSS5: [f32; 5] = [0.061, 0.245, 0.388, 0.245, 0.061];

/// Horizontal 5-tap Gaussian pass on a single-channel f32 buffer.
/// Replicates border pixels for edge handling.
fn gauss_h(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0_f32;
            for (ki, &kw) in GAUSS5.iter().enumerate() {
                let c = (col as isize + ki as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += kw * src[row * w + c];
            }
            dst[row * w + col] = acc;
        }
    }
}

/// Vertical 5-tap Gaussian pass on a single-channel f32 buffer.
fn gauss_v(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0_f32;
            for (ki, &kw) in GAUSS5.iter().enumerate() {
                let r = (row as isize + ki as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += kw * src[r * w + col];
            }
            dst[row * w + col] = acc;
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Unsharp-mask sharpen. `strength` in [-1.0, 1.0]:
///   - 0.0 = no-op (early return).
///   - Positive → `out = in + strength * (in - blur(in))` (sharpens).
///   - Negative → `out = lerp(in, blur(in), |strength|)` (blurs).
///
/// Blur kernel: 5-tap separable Gaussian, σ ≈ 1.0.
/// Alpha is copied unchanged.
pub fn apply_sharpen(img: &mut RgbaImage, strength: f32) {
    if strength == 0.0 {
        return;
    }
    let w = img.width() as usize;
    let h = img.height() as usize;
    let n = w * h;

    // f32 working buffers: one channel at a time.
    let mut ch_f32 = vec![0.0_f32; n];
    let mut blur_tmp = vec![0.0_f32; n];
    let mut blur_out = vec![0.0_f32; n];

    for c in 0..3usize {
        // Extract channel (read-only borrow ends before mutable borrow below)
        {
            let raw_ro = img.as_raw();
            for i in 0..n {
                ch_f32[i] = raw_ro[i * 4 + c] as f32;
            }
        }

        // Two-pass separable Gaussian
        gauss_h(&ch_f32, &mut blur_tmp, w, h);
        gauss_v(&blur_tmp, &mut blur_out, w, h);

        // Unsharp or blur depending on sign
        let raw = img.as_mut();
        if strength > 0.0 {
            for i in 0..n {
                let v = ch_f32[i] + strength * (ch_f32[i] - blur_out[i]);
                raw[i * 4 + c] = v.clamp(0.0, 255.0) as u8;
            }
        } else {
            let t = -strength; // lerp weight toward blur
            for i in 0..n {
                let v = ch_f32[i] + t * (blur_out[i] - ch_f32[i]);
                raw[i * 4 + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    // Alpha: untouched (channel 3 stays as-is in all iterations above).
}

/// Per-pixel lerp from the AI upscale toward a bicubic-resized
/// version of the original source. `weight` is the AI weight:
///   - 1.0 = full AI (no-op early return).
///   - 0.0 = full bicubic.
///
/// Caller MUST ensure `bicubic_source.dimensions() == img.dimensions()`.
/// Enforced by `debug_assert_eq!`; release builds cap iteration to the
/// shorter pixel count rather than panicking.
/// Alpha is from the AI input — NOT lerped from bicubic.
pub fn apply_ai_blend(img: &mut RgbaImage, bicubic_source: &RgbaImage, weight: f32) {
    debug_assert_eq!(
        img.dimensions(),
        bicubic_source.dimensions(),
        "apply_ai_blend: dimension mismatch — caller must pre-scale bicubic_source"
    );
    let w = weight.clamp(0.0, 1.0);
    if (w - 1.0).abs() < f32::EPSILON {
        return;
    }
    let n = (img.width() * img.height()) as usize;
    let n = n.min((bicubic_source.width() * bicubic_source.height()) as usize);
    let src_raw = bicubic_source.as_raw();
    // Build per-pixel RGB result into a flat buffer, then write back.
    let mut results = vec![0u8; n * 3];
    {
        let ai_raw = img.as_raw();
        for i in 0..n {
            for c in 0..3usize {
                let ai = ai_raw[i * 4 + c] as f32;
                let src = src_raw[i * 4 + c] as f32;
                results[i * 3 + c] = (ai * w + src * (1.0 - w)).clamp(0.0, 255.0) as u8;
            }
        }
    }
    let ai_raw_mut = img.as_mut();
    for i in 0..n {
        ai_raw_mut[i * 4] = results[i * 3];
        ai_raw_mut[i * 4 + 1] = results[i * 3 + 1];
        ai_raw_mut[i * 4 + 2] = results[i * 3 + 2];
        // channel 3 (alpha): from the AI input unchanged
    }
}

/// HSL-space saturation adjustment. `amount` in [-1.0, 1.0]:
///   - 0.0 = no-op.
///   - Negative = desaturate toward grey (amount=-1.0 → S=0).
///   - Positive = saturate (amount=+1.0 → S=1).
///
/// HSL (NOT HSV) — preserves luminance better.
/// Alpha unchanged.
pub fn apply_saturation(img: &mut RgbaImage, amount: f32) {
    if amount == 0.0 {
        return;
    }
    let n = (img.width() * img.height()) as usize;
    let mut results = vec![[0u8; 3]; n];
    {
        let raw = img.as_raw();
        for i in 0..n {
            let r = raw[i * 4];
            let g = raw[i * 4 + 1];
            let b = raw[i * 4 + 2];
            let (h, s, l) = srgb_to_hsl(r, g, b);
            let s_new = if amount >= 0.0 {
                (s + amount * (1.0 - s)).clamp(0.0, 1.0)
            } else {
                (s * (1.0 + amount)).clamp(0.0, 1.0)
            };
            let (ro, go, bo) = hsl_to_srgb(h, s_new, l);
            results[i] = [ro, go, bo];
        }
    }
    let raw = img.as_mut();
    for i in 0..n {
        raw[i * 4] = results[i][0];
        raw[i * 4 + 1] = results[i][1];
        raw[i * 4 + 2] = results[i][2];
        // raw[i * 4 + 3]: alpha unchanged
    }
}

/// Reinhard Lab-space mean+stddev color transfer. Snaps `target`'s
/// per-channel Lab statistics to match `source`. Compensates for
/// the upscale model's color drift.
///
/// `source` and `target` may have different dimensions — the
/// transfer uses image-wide statistics, not per-pixel pairing.
///
/// Variance guard: `stddev.max(1e-6)` before division. Without
/// this, a uniform-color source (stddev = 0) produces NaN pixels.
///
/// Alpha unchanged.
pub fn apply_color_match(target: &mut RgbaImage, source: &RgbaImage) {
    // Collect Lab values for source
    let src_n = (source.width() * source.height()) as usize;
    let mut src_l = Vec::with_capacity(src_n);
    let mut src_a = Vec::with_capacity(src_n);
    let mut src_b = Vec::with_capacity(src_n);

    let src_raw = source.as_raw();
    for i in 0..src_n {
        let (l, a, b) = srgb_to_lab(src_raw[i * 4], src_raw[i * 4 + 1], src_raw[i * 4 + 2]);
        src_l.push(l);
        src_a.push(a);
        src_b.push(b);
    }

    let (s_mean_l, s_std_l) = mean_std(&src_l);
    let (s_mean_a, s_std_a) = mean_std(&src_a);
    let (s_mean_b, s_std_b) = mean_std(&src_b);

    // Collect Lab values for target
    let tgt_n = (target.width() * target.height()) as usize;
    let mut tgt_l = Vec::with_capacity(tgt_n);
    let mut tgt_a_ch = Vec::with_capacity(tgt_n);
    let mut tgt_b_ch = Vec::with_capacity(tgt_n);

    let tgt_raw_ro = target.as_raw();
    for i in 0..tgt_n {
        let (l, a, b) = srgb_to_lab(tgt_raw_ro[i * 4], tgt_raw_ro[i * 4 + 1], tgt_raw_ro[i * 4 + 2]);
        tgt_l.push(l);
        tgt_a_ch.push(a);
        tgt_b_ch.push(b);
    }

    let (t_mean_l, t_std_l) = mean_std(&tgt_l);
    let (t_mean_a, t_std_a) = mean_std(&tgt_a_ch);
    let (t_mean_b, t_std_b) = mean_std(&tgt_b_ch);

    // Transfer and write back
    let raw = target.as_mut();
    for i in 0..tgt_n {
        let l_new = (tgt_l[i] - t_mean_l) * (s_std_l / t_std_l.max(1e-6)) + s_mean_l;
        let a_new = (tgt_a_ch[i] - t_mean_a) * (s_std_a / t_std_a.max(1e-6)) + s_mean_a;
        let b_new = (tgt_b_ch[i] - t_mean_b) * (s_std_b / t_std_b.max(1e-6)) + s_mean_b;
        let (ro, go, bo) = lab_to_srgb(l_new, a_new, b_new);
        raw[i * 4] = ro;
        raw[i * 4 + 1] = go;
        raw[i * 4 + 2] = bo;
        // raw[i * 4 + 3]: alpha unchanged
    }
}

/// Compute mean and population standard deviation of a slice.
fn mean_std(vals: &[f32]) -> (f32, f32) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let n = vals.len() as f64;
    let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = vals.iter().map(|&v| {
        let d = v as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    (mean as f32, variance.sqrt() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn make_image(w: u32, h: u32, fill: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(w, h);
        for p in img.pixels_mut() {
            *p = Rgba(fill);
        }
        img
    }

    fn gradient_image(w: u32, h: u32) -> RgbaImage {
        let mut img = RgbaImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *p = Rgba([v, (255 - v), v / 2, 200]);
        }
        img
    }

    // ── apply_sharpen ─────────────────────────────────────────────────────

    #[test]
    fn apply_sharpen_zero_strength_is_identity() {
        let orig = gradient_image(32, 32);
        let mut img = orig.clone();
        apply_sharpen(&mut img, 0.0);
        assert_eq!(img.as_raw(), orig.as_raw());
    }

    #[test]
    fn apply_sharpen_preserves_alpha() {
        let mut img = gradient_image(16, 16);
        // Override all alpha to 192
        for p in img.pixels_mut() {
            p.0[3] = 192;
        }
        apply_sharpen(&mut img, 0.7);
        for p in img.pixels() {
            assert_eq!(p.0[3], 192, "alpha must be preserved after sharpen");
        }
    }

    #[test]
    fn apply_sharpen_negative_strength_blurs() {
        // Single bright pixel (200,200,200) on dark background (10,10,10).
        // After blur (strength < 0), bright pixel dims and neighbours brighten.
        let mut img = make_image(7, 7, [10, 10, 10, 255]);
        img.put_pixel(3, 3, Rgba([200, 200, 200, 255]));
        let center_before = img.get_pixel(3, 3).0[0];
        apply_sharpen(&mut img, -1.0);
        let center_after = img.get_pixel(3, 3).0[0];
        let neighbor_after = img.get_pixel(3, 2).0[0];
        assert!(
            center_after < center_before,
            "blur must dim the bright center pixel: before={center_before} after={center_after}"
        );
        assert!(
            neighbor_after > 10,
            "blur must brighten neighbours: got {neighbor_after}"
        );
    }

    #[test]
    fn apply_sharpen_positive_strength_sharpens() {
        // Single bright pixel. After sharpen (strength > 0), neighbours
        // should be dimmer than the original background.
        let mut img = make_image(7, 7, [10, 10, 10, 255]);
        img.put_pixel(3, 3, Rgba([200, 200, 200, 255]));
        let center_before = img.get_pixel(3, 3).0[0];
        apply_sharpen(&mut img, 1.0);
        let center_after = img.get_pixel(3, 3).0[0];
        // After sharpening the center is boosted (clamped at 255) and
        // neighbours are pulled down.
        assert!(
            center_after >= center_before,
            "sharpen must keep/boost bright center: before={center_before} after={center_after}"
        );
        let neighbor_after = img.get_pixel(3, 2).0[0];
        assert!(
            neighbor_after < 10 || neighbor_after == 0,
            "sharpen must dim dark neighbours: got {neighbor_after}"
        );
    }

    // ── apply_ai_blend ───────────────────────────────────────────────────

    #[test]
    fn apply_ai_blend_weight_one_is_identity() {
        let src = make_image(4, 4, [0, 0, 255, 255]);
        let orig_ai = make_image(4, 4, [255, 0, 0, 255]);
        let mut ai = orig_ai.clone();
        apply_ai_blend(&mut ai, &src, 1.0);
        assert_eq!(ai.as_raw(), orig_ai.as_raw());
    }

    #[test]
    fn apply_ai_blend_weight_zero_replaces_rgb() {
        let src = make_image(4, 4, [0, 0, 255, 128]); // blue bicubic
        let mut ai = make_image(4, 4, [255, 0, 0, 255]); // red AI
        apply_ai_blend(&mut ai, &src, 0.0);
        for p in ai.pixels() {
            assert_eq!(p.0[0], 0, "R should be 0 (full bicubic blue)");
            assert_eq!(p.0[1], 0, "G should be 0");
            assert_eq!(p.0[2], 255, "B should be 255");
            assert_eq!(p.0[3], 255, "alpha from AI input, not bicubic");
        }
    }

    #[test]
    fn apply_ai_blend_weight_half_mixes() {
        let src = make_image(4, 4, [0, 0, 255, 255]); // blue
        let mut ai = make_image(4, 4, [255, 0, 0, 255]); // red
        apply_ai_blend(&mut ai, &src, 0.5);
        for p in ai.pixels() {
            let r = p.0[0];
            let b = p.0[2];
            assert!((r as i32 - 127).abs() <= 1, "R should be ~127, got {r}");
            assert_eq!(p.0[1], 0);
            assert!((b as i32 - 127).abs() <= 1, "B should be ~127, got {b}");
            assert_eq!(p.0[3], 255);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn apply_ai_blend_dimension_mismatch_panics() {
        let src = make_image(8, 8, [0, 0, 0, 255]);
        let mut ai = make_image(4, 4, [255, 0, 0, 255]);
        apply_ai_blend(&mut ai, &src, 0.5);
    }

    // ── apply_saturation ─────────────────────────────────────────────────

    #[test]
    fn apply_saturation_zero_is_identity() {
        let orig = gradient_image(16, 16);
        let mut img = orig.clone();
        apply_saturation(&mut img, 0.0);
        assert_eq!(img.as_raw(), orig.as_raw());
    }

    #[test]
    fn apply_saturation_negative_desaturates() {
        // Pure red [255, 0, 0] fully desaturated → grey (R == G == B ≈ luminance).
        let mut img = make_image(4, 4, [255, 0, 0, 255]);
        apply_saturation(&mut img, -1.0);
        for p in img.pixels() {
            let [r, g, b, _] = p.0;
            assert!(
                (r as i32 - g as i32).abs() < 10,
                "desaturated: |R-G| should be < 10, got R={r} G={g}"
            );
            assert!(
                (r as i32 - b as i32).abs() < 10,
                "desaturated: |R-B| should be < 10, got R={r} B={b}"
            );
        }
    }

    #[test]
    fn apply_saturation_positive_saturates() {
        // Muted red [200, 100, 100]: after positive saturation, gap widens.
        let mut img = make_image(4, 4, [200, 100, 100, 255]);
        let gap_before = {
            let p = img.get_pixel(0, 0);
            p.0[0] as i32 - p.0[1].max(p.0[2]) as i32
        };
        apply_saturation(&mut img, 1.0);
        let gap_after = {
            let p = img.get_pixel(0, 0);
            p.0[0] as i32 - p.0[1].max(p.0[2]) as i32
        };
        assert!(
            gap_after > gap_before,
            "positive saturation must widen the dominant-channel gap: before={gap_before} after={gap_after}"
        );
    }

    #[test]
    fn apply_saturation_preserves_alpha() {
        let mut img = make_image(8, 8, [200, 100, 50, 192]);
        apply_saturation(&mut img, 0.5);
        for p in img.pixels() {
            assert_eq!(p.0[3], 192, "alpha must be preserved after saturation");
        }
    }

    #[test]
    fn apply_saturation_hsl_not_hsv() {
        // HSL at amount=-0.5 on pure red [255,0,0]:
        //   HSL S of pure red = 1.0 → new S = 1.0 * (1 - 0.5) = 0.5
        //   At S=0.5, L=0.5 → the pixel should be noticeably de-saturated
        //   (not full red). Output R must be < 240 to distinguish from HSV
        //   (which would keep R near 255 since HSV saturates differently).
        let mut img = make_image(4, 4, [255, 0, 0, 255]);
        apply_saturation(&mut img, -0.5);
        let r = img.get_pixel(0, 0).0[0];
        assert!(
            r < 240,
            "HSL desaturation at -0.5 should produce R < 240 (got {r}); \
             HSV would keep near 255 — this distinguishes the two"
        );
    }

    // ── apply_color_match ────────────────────────────────────────────────

    #[test]
    fn apply_color_match_uniform_source_no_nan() {
        // Uniform grey source → stddev = 0 in all Lab channels.
        // Without the variance guard (stddev.max(1e-6)), output is NaN.
        let source = make_image(16, 16, [128, 128, 128, 255]);
        let mut target = gradient_image(32, 32);
        apply_color_match(&mut target, &source);
        for p in target.pixels() {
            let [r, g, b, _] = p.0;
            // All values must be finite (no NaN/Inf propagated through u8).
            // Since they're u8, they're always finite — but we check they're
            // not all the same impossible sentinel.
            let _ = (r, g, b); // ensure the pixels are accessible
        }
        // The real check: the function must not panic or produce all-black output
        // due to NaN contamination. We verify the target has valid pixel values.
        let raw = target.as_raw();
        assert!(
            raw.iter().any(|&v| v != 0),
            "uniform source variance guard test: output should not be all-black (NaN→0)"
        );
    }

    #[test]
    fn apply_color_match_identity_source_unchanged() {
        // When source and target are identical, stat_source == stat_target
        // → scale = 1, offset = 0 → output is bit-identical to input.
        let base = gradient_image(32, 32);
        let source = base.clone();
        let mut target = base.clone();
        apply_color_match(&mut target, &source);
        // Allow ±1 per-pixel tolerance due to f32 rounding in Lab roundtrip.
        for (orig, out) in base.pixels().zip(target.pixels()) {
            for c in 0..3 {
                let diff = (orig.0[c] as i32 - out.0[c] as i32).abs();
                assert!(
                    diff <= 2,
                    "identity color match: channel {c} diff {diff} > 2 at pixel {:?}",
                    orig
                );
            }
            assert_eq!(orig.0[3], out.0[3], "alpha must be unchanged");
        }
    }

    #[test]
    fn apply_color_match_dimension_mismatch_allowed() {
        // Different dimensions are valid — transfer uses statistics only.
        let source = gradient_image(64, 64);
        let mut target = gradient_image(32, 32);
        // Must not panic
        apply_color_match(&mut target, &source);
    }

    #[test]
    fn apply_color_match_warms_cool_target() {
        // Cool target: shifted blue. Warm source: shifted red/yellow.
        // After transfer, target mean R should increase and mean B decrease.
        let mut source = RgbaImage::new(32, 32);
        let mut target_orig = RgbaImage::new(32, 32);
        for p in source.pixels_mut() {
            *p = Rgba([200, 160, 80, 255]); // warm
        }
        for p in target_orig.pixels_mut() {
            *p = Rgba([80, 120, 200, 255]); // cool
        }
        let mut target = target_orig.clone();
        apply_color_match(&mut target, &source);

        let mean_r_before: f32 = target_orig.pixels().map(|p| p.0[0] as f32).sum::<f32>()
            / (target_orig.width() * target_orig.height()) as f32;
        let mean_b_before: f32 = target_orig.pixels().map(|p| p.0[2] as f32).sum::<f32>()
            / (target_orig.width() * target_orig.height()) as f32;

        let mean_r_after: f32 = target.pixels().map(|p| p.0[0] as f32).sum::<f32>()
            / (target.width() * target.height()) as f32;
        let mean_b_after: f32 = target.pixels().map(|p| p.0[2] as f32).sum::<f32>()
            / (target.width() * target.height()) as f32;

        assert!(
            mean_r_after > mean_r_before,
            "color match must increase mean R (warm source): before={mean_r_before:.1} after={mean_r_after:.1}"
        );
        assert!(
            mean_b_after < mean_b_before,
            "color match must decrease mean B (warm source): before={mean_b_before:.1} after={mean_b_after:.1}"
        );
    }

    #[test]
    fn apply_color_match_preserves_alpha() {
        let source = gradient_image(16, 16);
        let mut target = make_image(16, 16, [100, 150, 200, 192]);
        apply_color_match(&mut target, &source);
        for p in target.pixels() {
            assert_eq!(p.0[3], 192, "alpha must be preserved after color match");
        }
    }
}
