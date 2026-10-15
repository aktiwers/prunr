//! Tier-2 postprocess operations applied to the cached `upscale_raw`
//! RGBA buffer. Pure functions operating in-place; they do NOT
//! re-allocate the image (the caller's RAM accounting depends on
//! this).
//!
//! All operate on STRAIGHT (un-premultiplied) sRGB u8 per
//! CLAUDE.md ## Numerical & color invariants. Internal math uses
//! f32 working buffers; clamp+cast at the u8 boundary.
//!
//! Reinhard Lab (not LMS) for color match, with proper IEC 61966-2-1
//! linearisation and D65 XYZ→Lab pipeline.

use image::RgbaImage;

use crate::color::{lab_to_srgb, srgb_to_lab};

// ── Public API ─────────────────────────────────────────────────────────────

/// `src` through an unsharp mask. `strength` in [-1.0, 1.0]: 0 changes
/// nothing, positive sharpens (`in + strength · (in − blur)`), negative
/// blends toward the blur. `sigma` is the blur radius in output pixels;
/// detail narrower than it is left alone. Alpha is unchanged.
///
/// Peak working set: `src` plus the result, which is the blur buffer
/// written over (2 × w·h·4 bytes).
pub fn sharpened(src: &RgbaImage, strength: f32, sigma: f32) -> RgbaImage {
    if strength == 0.0 {
        return src.clone();
    }
    // reused as the result, pixel by pixel after reading its blur value
    use rayon::prelude::*;
    let mut out = crate::blur::blur_rgba(src, sigma.max(0.3));
    out.as_mut().par_chunks_exact_mut(4).zip(src.as_raw().par_chunks_exact(4)).for_each(|(o, p)| {
        for c in 0..3 {
            let (v, bv) = (p[c] as f32, o[c] as f32);
            let mixed = if strength > 0.0 { v + strength * (v - bv) } else { v - strength * (bv - v) };
            o[c] = mixed.round().clamp(0.0, 255.0) as u8;
        }
        o[3] = p[3];
    });
    out
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
    use rayon::prelude::*;
    img.as_mut().par_chunks_exact_mut(4).zip(bicubic_source.as_raw().par_chunks_exact(4)).for_each(|(px, src)| {
        for c in 0..3usize {
            px[c] = (px[c] as f32 * w + src[c] as f32 * (1.0 - w)).clamp(0.0, 255.0) as u8;
        }
        // channel 3 (alpha): from the AI input unchanged
    });
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
/// Alpha unchanged. Rows run in parallel and the statistics are folded
/// as they go, so no Lab copy of either image is kept: no working set
/// beyond a few numbers per thread.
pub fn apply_color_match(target: &mut RgbaImage, source: &RgbaImage) {
    use rayon::prelude::*;
    let s = LabStats::of(source);
    let t = LabStats::of(target);
    let scale = |i: usize| s.std[i] / t.std[i].max(1e-6);
    let (kl, ka, kb) = (scale(0), scale(1), scale(2));
    target.as_mut().par_chunks_exact_mut(4).for_each(|px| {
        let (l, a, b) = srgb_to_lab(px[0], px[1], px[2]);
        let (ro, go, bo) = lab_to_srgb(
            (l - t.mean[0]) * kl + s.mean[0],
            (a - t.mean[1]) * ka + s.mean[1],
            (b - t.mean[2]) * kb + s.mean[2],
        );
        px[..3].copy_from_slice(&[ro, go, bo]);
    });
}

/// Mean and population standard deviation of an image's L, a and b.
struct LabStats {
    mean: [f32; 3],
    std: [f32; 3],
}

impl LabStats {
    fn of(img: &RgbaImage) -> Self {
        use rayon::prelude::*;
        let moments = img.as_raw()
            .par_chunks_exact(4)
            .fold(Moments::default, |mut m, px| {
                let (l, a, b) = srgb_to_lab(px[0], px[1], px[2]);
                m.push([l, a, b]);
                m
            })
            .reduce(Moments::default, Moments::merge);
        Self {
            mean: moments.mean.map(|v| v as f32),
            std: std::array::from_fn(|i| if moments.n == 0.0 { 0.0 } else { (moments.m2[i] / moments.n).sqrt() as f32 }),
        }
    }
}

/// Running count, mean and sum of squared deviations per channel
/// (Welford), mergeable across threads (Chan et al.): as exact as the
/// two-pass mean then variance, in one pass.
#[derive(Default, Clone, Copy)]
struct Moments {
    n: f64,
    mean: [f64; 3],
    m2: [f64; 3],
}

impl Moments {
    fn push(&mut self, v: [f32; 3]) {
        self.n += 1.0;
        for i in 0..3 {
            let d = f64::from(v[i]) - self.mean[i];
            self.mean[i] += d / self.n;
            self.m2[i] += d * (f64::from(v[i]) - self.mean[i]);
        }
    }

    fn merge(a: Self, b: Self) -> Self {
        let n = a.n + b.n;
        if n == 0.0 {
            return a;
        }
        let mut out = Self { n, ..a };
        for i in 0..3 {
            let d = b.mean[i] - a.mean[i];
            out.mean[i] = a.mean[i] + d * b.n / n;
            out.m2[i] = a.m2[i] + b.m2[i] + d * d * a.n * b.n / n;
        }
        out
    }
}

/// What Refine applies after the model: Sharpen, AI blend toward the
/// plain enlargement, Color match to the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refine {
    pub sharpen: f32,
    pub sharpen_sigma: f32,
    pub ai_blend: f32,
    pub color_match: bool,
}

impl Refine {
    pub fn of_recipe(r: &crate::UpscaleRecipe) -> Self {
        Self { sharpen: r.sharpen(), sharpen_sigma: r.output_scale.sharpen_sigma(), ai_blend: r.ai_blend(), color_match: r.color_match }
    }

    /// AI blend and Color match compare against the upscaled image's input.
    fn needs_reference(&self) -> bool {
        self.ai_blend < 1.0 - f32::EPSILON || self.color_match
    }

    fn change_nothing(&self) -> bool {
        self.sharpen == 0.0 && !self.needs_reference()
    }
}

/// The upscale result to show, from the model's raw output and `input`, the
/// image the model read: Refine's knobs applied after the model. AI blend
/// and Color match compare against `input` enlarged to the output size,
/// reused from `reference` when it already is that size and built (and
/// returned for caching) otherwise. With every knob at its default the raw
/// output is shown as is, not copied.
///
/// Order: AI blend, then Sharpen (it sharpens what the blend left), then
/// Color match (global statistics last).
///
/// Peak working set besides `raw`: the result, a blur buffer while AI
/// blend and Sharpen are both on, and the enlarged input for AI blend or
/// Color match (1-3 × w·h·4 bytes).
pub fn refine(
    raw: &std::sync::Arc<RgbaImage>,
    input: &RgbaImage,
    reference: Option<std::sync::Arc<RgbaImage>>,
    knobs: &Refine,
) -> (std::sync::Arc<RgbaImage>, Option<std::sync::Arc<RgbaImage>>) {
    use std::sync::Arc;
    if knobs.change_nothing() {
        return (Arc::clone(raw), reference);
    }
    let reference = knobs.needs_reference().then(|| {
        reference.filter(|r| r.dimensions() == raw.dimensions()).unwrap_or_else(|| {
            Arc::new(crate::formats::resize_rgba(input, raw.width(), raw.height(), crate::formats::ResizeFilter::CatmullRom))
        })
    });
    let blend = reference.as_deref().filter(|_| knobs.ai_blend < 1.0 - f32::EPSILON);
    let mut shown = match blend {
        Some(reference) => {
            let mut blended = (**raw).clone();
            apply_ai_blend(&mut blended, reference, knobs.ai_blend);
            if knobs.sharpen == 0.0 { blended } else { sharpened(&blended, knobs.sharpen, knobs.sharpen_sigma) }
        }
        None => sharpened(raw, knobs.sharpen, knobs.sharpen_sigma),
    };
    if let Some(reference) = reference.as_deref().filter(|_| knobs.color_match) {
        apply_color_match(&mut shown, reference);
    }
    (Arc::new(shown), reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With every knob at its default the model's output is shown as is,
    /// not copied; AI blend at 0 shows the plain enlargement of the input.
    #[test]
    fn refine_leaves_the_output_alone_at_defaults_and_blends_toward_the_input() {
        use std::sync::Arc;
        let raw = Arc::new(RgbaImage::from_pixel(4, 4, image::Rgba([200, 200, 200, 255])));
        let input = RgbaImage::from_pixel(2, 2, image::Rgba([40, 40, 40, 255]));
        let off = Refine { sharpen: 0.0, sharpen_sigma: 2.0, ai_blend: 1.0, color_match: false };
        let (shown, reference) = refine(&raw, &input, None, &off);
        assert!(Arc::ptr_eq(&shown, &raw) && reference.is_none());
        let (shown, reference) = refine(&raw, &input, None, &Refine { ai_blend: 0.0, ..off });
        assert_eq!(shown.get_pixel(1, 1).0, [40, 40, 40, 255]);
        assert!(reference.is_some_and(|r| r.dimensions() == (4, 4)), "the enlarged input comes back for reuse");
    }
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

    // ── sharpened ─────────────────────────────────────────────────────

    /// A wider radius sharpens detail as wide as itself: after a 4x
    /// upscale that is ~4 output pixels, not the per-pixel grain a 1 px
    /// radius picks out. The halo reaches farther from an edge.
    #[test]
    fn a_wider_sharpen_radius_reaches_farther_from_an_edge() {
        let step = || RgbaImage::from_fn(32, 4, |x, _| if x < 16 { Rgba([60, 60, 60, 255]) } else { Rgba([180, 180, 180, 255]) });
        let (mut narrow, mut wide) = (step(), step());
        narrow = sharpened(&narrow, 1.0, 0.5);
        wide = sharpened(&wide, 1.0, 2.0);
        let three_px_out = |img: &RgbaImage| img.get_pixel(12, 1).0[0];
        assert_eq!(three_px_out(&narrow), 60, "a narrow radius leaves it alone");
        assert!(three_px_out(&wide) < 60, "a wide radius darkens the dark side 3 px out: {}", three_px_out(&wide));
    }

    #[test]
    fn apply_sharpen_zero_strength_is_identity() {
        let orig = gradient_image(32, 32);
        let mut img = orig.clone();
        img = sharpened(&img, 0.0, 1.0);
        assert_eq!(img.as_raw(), orig.as_raw());
    }

    #[test]
    fn apply_sharpen_preserves_alpha() {
        let mut img = gradient_image(16, 16);
        // Override all alpha to 192
        for p in img.pixels_mut() {
            p.0[3] = 192;
        }
        img = sharpened(&img, 0.7, 1.0);
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
        img = sharpened(&img, -1.0, 1.0);
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
        img = sharpened(&img, 1.0, 1.0);
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

    // ── apply_color_match ────────────────────────────────────────────────

    /// The parallel transfer is the two-pass serial one it replaced, to
    /// within one level from summation order.
    #[test]
    fn apply_color_match_is_the_serial_two_pass_transfer() {
        fn serial(target: &mut RgbaImage, source: &RgbaImage) {
            let lab = |img: &RgbaImage| img.pixels().map(|p| srgb_to_lab(p.0[0], p.0[1], p.0[2])).collect::<Vec<_>>();
            let stats = |v: &[(f32, f32, f32)], f: fn(&(f32, f32, f32)) -> f32| {
                let n = v.len() as f64;
                let mean = v.iter().map(|p| f(p) as f64).sum::<f64>() / n;
                let var = v.iter().map(|p| (f(p) as f64 - mean).powi(2)).sum::<f64>() / n;
                (mean as f32, var.sqrt() as f32)
            };
            let (s, t) = (lab(source), lab(target));
            let ch: [fn(&(f32, f32, f32)) -> f32; 3] = [|p| p.0, |p| p.1, |p| p.2];
            let (ss, ts) = (ch.map(|f| stats(&s, f)), ch.map(|f| stats(&t, f)));
            for (px, l) in target.pixels_mut().zip(&t) {
                let v = [l.0, l.1, l.2];
                let m: [f32; 3] = std::array::from_fn(|i| (v[i] - ts[i].0) * (ss[i].1 / ts[i].1.max(1e-6)) + ss[i].0);
                let (r, g, b) = lab_to_srgb(m[0], m[1], m[2]);
                px.0[..3].copy_from_slice(&[r, g, b]);
            }
        }
        let source = RgbaImage::from_fn(300, 200, |x, y| image::Rgba([(x % 200) as u8 + 30, (y + 40) as u8, ((x * y) % 97) as u8 + 100, 255]));
        let target = RgbaImage::from_fn(640, 480, |x, y| image::Rgba([((x + y) % 256) as u8, (x % 180) as u8, (y % 220) as u8 + 20, (x % 256) as u8]));
        let (mut want, mut got) = (target.clone(), target.clone());
        serial(&mut want, &source);
        apply_color_match(&mut got, &source);
        let worst = want.as_raw().iter().zip(got.as_raw()).map(|(a, b)| a.abs_diff(*b)).max();
        assert!(worst <= Some(1), "worst channel difference {worst:?}");
        assert!(got.pixels().zip(target.pixels()).all(|(a, b)| a.0[3] == b.0[3]), "alpha unchanged");
    }


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
