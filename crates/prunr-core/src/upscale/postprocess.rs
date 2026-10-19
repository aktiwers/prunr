//! Refine's steps after the model: sharpen, AI blend, Reinhard Lab
//! colour match. Straight sRGB u8 in and out, f32 math in between.
//!
//! Colour match works in Lab, not LMS, with IEC 61966-2-1
//! linearisation and a D65 XYZ→Lab pipeline.

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
            let mixed = v + strength * (v - bv);
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
            px[c] = crate::math::round_u8(px[c] as f32 * w + src[c] as f32 * (1.0 - w));
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
    color_match_to(target, &LabStats::of(source));
}

/// `apply_color_match` toward a source whose statistics are `s`.
fn color_match_to(target: &mut RgbaImage, s: &LabStats) {
    use rayon::prelude::*;
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

/// The plain enlargement of the input a Refine blends and colour-matches
/// toward, with its Lab statistics worked out once, at its first colour
/// match: a drag re-renders with the same reference every tick.
pub struct Reference {
    image: RgbaImage,
    stats: std::sync::OnceLock<LabStats>,
}

impl Reference {
    pub fn new(image: RgbaImage) -> Self {
        Self { image, stats: std::sync::OnceLock::new() }
    }

    fn stats(&self) -> &LabStats {
        self.stats.get_or_init(|| LabStats::of(&self.image))
    }
}

impl std::ops::Deref for Reference {
    type Target = RgbaImage;

    fn deref(&self) -> &RgbaImage {
        &self.image
    }
}

/// Pixels per partial sum in `LabStats::of`.
const STATS_CHUNK_PIXELS: usize = 64 * 1024;

/// Mean and population standard deviation of an image's L, a and b.
struct LabStats {
    mean: [f32; 3],
    std: [f32; 3],
}

impl LabStats {
    fn of(img: &RgbaImage) -> Self {
        use rayon::prelude::*;
        // Fixed chunks merged in order: rayon's own splits follow thread
        // scheduling, and a different merge order could move a pixel on a
        // rounding edge by a level between two runs of the same image.
        let moments = img.as_raw()
            .par_chunks(STATS_CHUNK_PIXELS * 4)
            .map(|chunk| chunk.chunks_exact(4).fold(Moments::default(), |mut m, px| {
                let (l, a, b) = srgb_to_lab(px[0], px[1], px[2]);
                m.push([l, a, b]);
                m
            }))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Moments::default(), Moments::merge);
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
        for ((value, mean), m2) in v.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let d = f64::from(*value) - *mean;
            *mean += d / self.n;
            *m2 += d * (f64::from(*value) - *mean);
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

    fn blends(&self) -> bool {
        self.ai_blend < 1.0 - f32::EPSILON
    }

    /// AI blend and Color match compare against the upscaled image's input.
    fn needs_reference(&self) -> bool {
        self.blends() || self.color_match
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
    reference: Option<std::sync::Arc<Reference>>,
    knobs: &Refine,
) -> (std::sync::Arc<RgbaImage>, Option<std::sync::Arc<Reference>>) {
    use std::sync::Arc;
    if knobs.change_nothing() {
        return (Arc::clone(raw), reference);
    }
    let reference = knobs.needs_reference().then(|| {
        reference.filter(|r| r.dimensions() == raw.dimensions()).unwrap_or_else(|| {
            Arc::new(Reference::new(crate::formats::resize_rgba(input, raw.width(), raw.height(), crate::formats::ResizeFilter::CatmullRom)))
        })
    });
    let blend = reference.as_deref().filter(|_| knobs.blends());
    let mut shown = match blend {
        Some(reference) => {
            let mut blended = (**raw).clone();
            apply_ai_blend(&mut blended, reference, knobs.ai_blend);
            if knobs.sharpen == 0.0 { blended } else { sharpened(&blended, knobs.sharpen, knobs.sharpen_sigma) }
        }
        None => sharpened(raw, knobs.sharpen, knobs.sharpen_sigma),
    };
    if let Some(reference) = reference.as_deref().filter(|_| knobs.color_match) {
        color_match_to(&mut shown, reference.stats());
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

    /// A bright dot on dark: a positive strength keeps or boosts the dot
    /// and pulls its neighbours down; a negative one blurs, dimming the
    /// dot and lifting them.
    #[test]
    fn apply_sharpen_sign_picks_sharpen_or_blur() {
        let mut dot = make_image(7, 7, [10, 10, 10, 255]);
        dot.put_pixel(3, 3, Rgba([200, 200, 200, 255]));
        let sharp = sharpened(&dot, 1.0, 1.0);
        assert!(sharp.get_pixel(3, 3).0[0] >= 200, "the dot keeps or gains: {:?}", sharp.get_pixel(3, 3));
        assert!(sharp.get_pixel(3, 2).0[0] < 10, "the neighbour darkens: {:?}", sharp.get_pixel(3, 2));
        let soft = sharpened(&dot, -1.0, 1.0);
        assert!(soft.get_pixel(3, 3).0[0] < 200, "the dot dims: {:?}", soft.get_pixel(3, 3));
        assert!(soft.get_pixel(3, 2).0[0] > 10, "the neighbour lifts: {:?}", soft.get_pixel(3, 2));
    }

    // ── apply_ai_blend ───────────────────────────────────────────────────

    /// The AI weight lerps from the bicubic source (0) to the model's
    /// RGB (1, unchanged); alpha always comes from the model.
    #[test]
    fn apply_ai_blend_lerps_rgb_toward_the_source() {
        let src = make_image(4, 4, [0, 0, 255, 128]); // blue bicubic
        let ai = make_image(4, 4, [255, 0, 0, 255]); // red AI
        for (weight, want) in [(1.0, [255, 0, 127]), (0.0, [0, 0, 255]), (0.5, [127, 0, 127])] {
            let mut out = ai.clone();
            apply_ai_blend(&mut out, &src, weight);
            if weight == 1.0 {
                assert_eq!(out, ai, "weight 1 is the model's output");
                continue;
            }
            for p in out.pixels() {
                assert!(p.0[..3].iter().zip(want).all(|(&got, w)| got.abs_diff(w) <= 1), "weight {weight}: {p:?}");
                assert_eq!(p.0[3], 255, "alpha from AI input, not bicubic");
            }
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

    /// The same images give the same pixels on every run, whatever
    /// threads the statistics were split over.
    #[test]
    fn apply_color_match_is_the_same_on_every_run() {
        let source = RgbaImage::from_fn(700, 500, |x, y| image::Rgba([(x % 200) as u8 + 30, (y % 180) as u8 + 40, ((x * y) % 97) as u8 + 100, 255]));
        let target = RgbaImage::from_fn(900, 700, |x, y| image::Rgba([((x + y) % 256) as u8, (x % 180) as u8, (y % 220) as u8 + 20, 255]));
        let run = || { let mut t = target.clone(); apply_color_match(&mut t, &source); t };
        let first = run();
        for _ in 0..8 {
            assert!(run() == first, "a run differs from the first");
        }
    }

    /// The parallel transfer is the two-pass serial one it replaced, to
    /// within one level from summation order.
    #[test]
    fn apply_color_match_is_the_serial_two_pass_transfer() {
        fn serial(target: &mut RgbaImage, source: &RgbaImage) {
            let lab = |img: &RgbaImage| img.pixels().map(|p| srgb_to_lab(p.0[0], p.0[1], p.0[2])).collect::<Vec<_>>();
            type Channel = fn(&(f32, f32, f32)) -> f32;
            let stats = |v: &[(f32, f32, f32)], f: Channel| {
                let n = v.len() as f64;
                let mean = v.iter().map(|p| f(p) as f64).sum::<f64>() / n;
                let var = v.iter().map(|p| (f(p) as f64 - mean).powi(2)).sum::<f64>() / n;
                (mean as f32, var.sqrt() as f32)
            };
            let (s, t) = (lab(source), lab(target));
            let ch: [Channel; 3] = [|p| p.0, |p| p.1, |p| p.2];
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


    /// A kept reference colour-matches as the image itself does, its
    /// statistics worked out on the first match and kept for the next.
    #[test]
    fn a_kept_reference_colour_matches_once_worked_out() {
        use std::sync::Arc;
        let raw = Arc::new(RgbaImage::from_fn(64, 48, |x, y| image::Rgba([(x * 3) as u8, (y * 5) as u8, 90, 255])));
        let input = RgbaImage::from_fn(32, 24, |x, y| image::Rgba([(x * 7) as u8, 120, (y * 9) as u8, 255]));
        let knobs = Refine { sharpen: 0.0, sharpen_sigma: 2.0, ai_blend: 1.0, color_match: true };
        let (shown, reference) = refine(&raw, &input, None, &knobs);
        let reference = reference.expect("built");
        assert!(reference.stats.get().is_some(), "worked out on the first match");
        let mut want = (*raw).clone();
        apply_color_match(&mut want, &reference);
        assert!(*shown == want);
        let (again, kept) = refine(&raw, &input, Some(Arc::clone(&reference)), &knobs);
        assert!(Arc::ptr_eq(&kept.expect("kept"), &reference) && *again == want);
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
    fn apply_color_match_warms_cool_target() {
        // Cool target: shifted blue. Warm source: shifted red/yellow.
        // After transfer, target mean R should increase and mean B decrease.
        // The target is flat, so its spread is 0: this also holds the
        // variance guard (unguarded, 0 · ∞ turns every pixel to NaN).
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
}
