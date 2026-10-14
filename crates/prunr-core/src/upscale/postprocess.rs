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
    let mut out = crate::blur::blur_rgba(src, sigma.max(0.3));
    for (o, p) in out.pixels_mut().zip(src.pixels()) {
        for c in 0..3 {
            let (v, bv) = (p.0[c] as f32, o.0[c] as f32);
            let mixed = if strength > 0.0 { v + strength * (v - bv) } else { v - strength * (bv - v) };
            o.0[c] = mixed.round().clamp(0.0, 255.0) as u8;
        }
        o.0[3] = p.0[3];
    }
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
    let n = (img.width() * img.height()) as usize;
    let n = n.min((bicubic_source.width() * bicubic_source.height()) as usize);
    let src_raw = bicubic_source.as_raw();
    let raw = img.as_mut();
    for i in 0..n {
        for c in 0..3usize {
            let ai = raw[i * 4 + c] as f32;
            let src = src_raw[i * 4 + c] as f32;
            raw[i * 4 + c] = (ai * w + src * (1.0 - w)).clamp(0.0, 255.0) as u8;
        }
        // channel 3 (alpha): from the AI input unchanged
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
    // free ~3 × src_n × 4 bytes before allocating the target Lab buffers
    drop(src_l);
    drop(src_a);
    drop(src_b);

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
