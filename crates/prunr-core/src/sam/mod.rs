//! SAM 2 inference primitives — encoder preprocessing, decoder prompt
//! construction, and decoder output → MaskArtifact conversion.
//!
//! PURE module — no ORT dependency. The ort::Session.run plumbing
//! lives in `prunr-app/src/gui/processor.rs` so this module stays
//! unit-testable from prunr-core alone.

use rayon::prelude::*;

use crate::selection::{BrushMode, FULL};

pub mod preprocess;
pub mod prompt;

/// Encoder input edge length. SAM 2 Hiera Small is fixed at 1024×1024.
pub const SAM_ENCODER_INPUT: u32 = 1024;

/// Decoder mask output resolution. SAM 2 emits 256×256 logits;
/// caller upsamples to source resolution.
pub const SAM_MASK_RESOLUTION: u32 = 256;

/// Cached encoder output for one source image.
#[derive(Debug)]
pub struct SamEmbedding {
    /// f32[1, 256, 64, 64] — 1_048_576 f32 ≈ 4 MB
    pub image_embed: Vec<f32>,
    /// f32[1, 32, 256, 256] — 2_097_152 f32 ≈ 8 MB
    pub high_res_feats_0: Vec<f32>,
    /// f32[1, 64, 128, 128] — 1_048_576 f32 ≈ 4 MB
    pub high_res_feats_1: Vec<f32>,
}

impl SamEmbedding {
    /// Stable byte estimate for BatchItem::cache_size.
    pub const fn expected_bytes() -> usize {
        (1_048_576 + 2_097_152 + 1_048_576) * 4
    }

    /// Verify the three Vec<f32> lengths match SAM 2 Hiera Small
    /// expected output shapes.
    pub fn validate_shapes(&self) -> Result<(), String> {
        if self.image_embed.len() != 1_048_576 {
            return Err(format!(
                "image_embed len {} != 1_048_576",
                self.image_embed.len()
            ));
        }
        if self.high_res_feats_0.len() != 2_097_152 {
            return Err(format!(
                "high_res_feats_0 len {} != 2_097_152",
                self.high_res_feats_0.len()
            ));
        }
        if self.high_res_feats_1.len() != 1_048_576 {
            return Err(format!(
                "high_res_feats_1 len {} != 1_048_576",
                self.high_res_feats_1.len()
            ));
        }
        Ok(())
    }
}

/// Decoder output as returned by the ORT session. Consumed by
/// `decode_to_mask_artifact`.
pub struct SamDecoderOutput {
    /// 3 candidates × SAM_MASK_RESOLUTION² logits.
    pub masks: Vec<f32>,
    /// IoU prediction per candidate.
    pub iou_predictions: [f32; 3],
}

/// The logit a pixel must reach to count as selected at `confidence`,
/// the probability the knob exposes. 0.5 is the model's own decision
/// boundary (logit 0); the clamp keeps the extremes finite.
pub fn confidence_logit(confidence: f32) -> f32 {
    let c = confidence.clamp(0.02, 0.98);
    (c / (1.0 - c)).ln()
}

/// How decoder logits become a selection: the probability a pixel must
/// reach (`confidence_logit`), and whether specks are cleaned away
/// (`smooth3`, then `remove_small_regions`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskReading {
    pub confidence: f32,
    pub remove_specks: bool,
}

/// Smallest region, in cells of the decoder's 256² grid (about 0.05 % of
/// the image), a cleaned selection keeps: smaller islands go and smaller
/// holes fill, as Meta's SAM reference cleans its masks. Where the model
/// is unsure its logits hover at the threshold cell by cell, which reads
/// as a grid of specks.
pub const MIN_REGION_CELLS: usize = 32;

/// 3×3 mean of the 256² logits, edges clamped. Where SAM is unsure it
/// dithers around the threshold in a period-2 checkerboard; the mean
/// cancels the dither and settles each area by its local average, while a
/// real edge (a step of several logits) stays on the same cells.
fn smooth3(logits: &[f32]) -> Vec<f32> {
    const M: usize = SAM_MASK_RESOLUTION as usize;
    let at = |x: usize, y: usize| logits[y * M + x];
    let mut out = vec![0.0; M * M];
    for y in 0..M {
        let (y0, y1) = (y.saturating_sub(1), (y + 1).min(M - 1));
        for x in 0..M {
            let (x0, x1) = (x.saturating_sub(1), (x + 1).min(M - 1));
            let mut sum = 0.0;
            for yy in [y0, y, y1] {
                sum += at(x0, yy) + at(x, yy) + at(x1, yy);
            }
            out[y * M + x] = sum / 9.0;
        }
    }
    out
}

/// Push every region of the thresholded grid smaller than
/// `MIN_REGION_CELLS` (8-connected) to the other side of `threshold`,
/// by a margin that bilinear upsampling cannot cross: an island's
/// neighbours are all below the threshold, a hole's all above.
fn remove_small_regions(logits: &mut [f32], threshold: f32) {
    const M: usize = SAM_MASK_RESOLUTION as usize;
    let inside: Vec<bool> = logits.iter().map(|&v| v >= threshold).collect();
    let mut seen = vec![false; M * M];
    let (mut stack, mut region) = (Vec::new(), Vec::new());
    for start in 0..M * M {
        if seen[start] {
            continue;
        }
        let side = inside[start];
        seen[start] = true;
        stack.push(start);
        region.clear();
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = (i % M, i / M);
            for ny in y.saturating_sub(1)..=(y + 1).min(M - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(M - 1) {
                    let j = ny * M + nx;
                    if !seen[j] && inside[j] == side {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        if region.len() < MIN_REGION_CELLS {
            let flipped = if side { threshold - 1.0 } else { threshold + 1.0 };
            for &i in &region {
                logits[i] = flipped;
            }
        }
    }
}

/// Pick the candidate with the highest predicted IoU; bilinear-upsample
/// its 256×256 logits to source resolution; keep the pixels whose
/// probability reaches the reading's confidence (see `confidence_logit`),
/// signed by `mode`. Higher confidence keeps the sure core of the object,
/// lower grows into the uncertain rim.
///
/// Peak working set: source_w * source_h bytes output + 256*256*4 bytes
/// slice view (read-only). At 4K source: ~8 MB output.
pub fn decode_to_mask_artifact(
    output: &SamDecoderOutput,
    source_w: u32,
    source_h: u32,
    reading: MaskReading,
    mode: BrushMode,
) -> crate::selection::MaskArtifact {
    let threshold = confidence_logit(reading.confidence);
    let best_idx = output
        .iou_predictions
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    const M: usize = SAM_MASK_RESOLUTION as usize;
    let raw = &output.masks[best_idx * M * M..(best_idx + 1) * M * M];
    let cleaned;
    let mask_logits: &[f32] = if reading.remove_specks {
        cleaned = { let mut l = smooth3(raw); remove_small_regions(&mut l, threshold); l };
        &cleaned
    } else {
        raw
    };
    let logit_row = |i: u8| -> &[f32; M] {
        // M logits per row by construction of the slice above.
        mask_logits[i as usize * M..][..M].try_into().expect("row of M logits")
    };

    let selected = mode.sign() * FULL;
    // Bilinear taps are separable: the x taps repeat on every row and
    // the y taps on every column, so each is computed once.
    let taps = |i: u32, n: u32| -> Tap {
        let scale = SAM_MASK_RESOLUTION as f32 / n as f32;
        let s = (i as f32 + 0.5) * scale - 0.5; // half-pixel centre
        let i0 = s.floor().clamp(0.0, (M - 1) as f32) as u8;
        let f = (s - i0 as f32).clamp(0.0, 1.0);
        Tap { i0, i1: i0.saturating_add(1), f, g: 1.0 - f }
    };
    let xs: Vec<Tap> = (0..source_w).map(|x| taps(x, source_w)).collect();
    let mut data = vec![0i8; (source_w * source_h) as usize];
    data.par_chunks_mut(source_w as usize).enumerate().for_each(|(y, row)| {
        let Tap { i0: y0, i1: y1, f: fy, g: gy } = taps(y as u32, source_h);
        let (top, bottom) = (logit_row(y0), logit_row(y1));
        for (cell, &Tap { i0: x0, i1: x1, f: fx, g: gx }) in row.iter_mut().zip(&xs) {
            let v = top[x0 as usize] * gx * gy
                + top[x1 as usize] * fx * gy
                + bottom[x0 as usize] * gx * fy
                + bottom[x1 as usize] * fx * fy;
            *cell = if v >= threshold { selected } else { 0 };
        }
    });

    crate::selection::MaskArtifact::from_cells(source_w, source_h, data)
}

/// One axis of a bilinear sample: the two logit indices and the
/// weights of the far (`f`) and near (`g = 1 - f`) tap. `u8` indices
/// into `[f32; 256]` rows need no bounds check.
#[derive(Clone, Copy)]
struct Tap {
    i0: u8,
    i1: u8,
    f: f32,
    g: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(confidence: f32) -> MaskReading {
        MaskReading { confidence, remove_specks: false }
    }

    /// Measured on a real stroke: where SAM is unsure its logits dither
    /// ±0.2 around the threshold in a period-2 checkerboard. Every dot
    /// touches the next diagonally, so they join into big regions no area
    /// filter removes; the cleanup must read the local average instead.
    #[test]
    fn cleaning_settles_a_dithered_area_by_its_average() {
        let mut masks = vec![-2.0f32; 3 * 256 * 256];
        for y in 0..256 {
            for x in 0..256 {
                let dither = if (x + y) % 2 == 0 { 0.2 } else { -0.2 };
                let bias = if x < 128 { 0.1 } else { -0.1 };
                masks[y * 256 + x] = dither + bias;
            }
        }
        let output = SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] };
        let raw = decode_to_mask_artifact(&output, 256, 256, plain(0.5), BrushMode::Add);
        let stripes = |m: &crate::selection::MaskArtifact, x0: usize| (0..256).filter(|&y| m.cells()[y * 256 + x0] != m.cells()[y * 256 + x0 + 1]).count();
        assert!(stripes(&raw, 40) > 100, "the raw read is a checkerboard");
        let clean = decode_to_mask_artifact(&output, 256, 256, MaskReading { confidence: 0.5, remove_specks: true }, BrushMode::Add);
        assert!((10..118).all(|x| (10..246).all(|y| clean.cells()[y * 256 + x] == FULL)), "the leaning-in half is solid");
        assert!((138..246).all(|x| (10..246).all(|y| clean.cells()[y * 256 + x] == 0)), "the leaning-out half is empty");
    }

    #[test]
    fn cleaning_drops_specks_and_fills_pinholes_but_keeps_the_object() {
        let mut masks = vec![-2.0f32; 3 * 256 * 256];
        for y in 100..160 {
            for x in 100..160 {
                masks[y * 256 + x] = 2.0;
            }
        }
        masks[130 * 256 + 130] = -2.0; // a pinhole in the object
        masks[20 * 256 + 20] = 2.0; // a speck far from it
        let output = SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] };
        let cell = |m: &crate::selection::MaskArtifact, x: usize, y: usize| m.cells()[y * 256 + x];
        let raw = decode_to_mask_artifact(&output, 256, 256, plain(0.5), BrushMode::Add);
        assert_eq!((cell(&raw, 20, 20), cell(&raw, 130, 130)), (FULL, 0), "without cleaning both show");
        let clean = decode_to_mask_artifact(&output, 256, 256, MaskReading { confidence: 0.5, remove_specks: true }, BrushMode::Add);
        assert_eq!(cell(&clean, 20, 20), 0, "the speck is gone");
        assert_eq!(cell(&clean, 130, 130), FULL, "the pinhole is filled");
        assert_eq!(cell(&clean, 110, 110), FULL, "the object stays");
        assert_eq!(cell(&clean, 200, 200), 0, "the background stays");
    }

    #[test]
    fn expected_bytes_is_16_mb() {
        assert_eq!(SamEmbedding::expected_bytes(), 16_777_216);
    }

    #[test]
    fn validate_shapes_accepts_correct_lengths() {
        let emb = SamEmbedding {
            image_embed: vec![0.0; 1_048_576],
            high_res_feats_0: vec![0.0; 2_097_152],
            high_res_feats_1: vec![0.0; 1_048_576],
        };
        assert!(emb.validate_shapes().is_ok());
    }

    #[test]
    fn validate_shapes_rejects_wrong_image_embed_len() {
        let emb = SamEmbedding {
            image_embed: vec![0.0; 42],
            high_res_feats_0: vec![0.0; 2_097_152],
            high_res_feats_1: vec![0.0; 1_048_576],
        };
        let err = emb.validate_shapes().unwrap_err();
        assert!(err.contains("image_embed"), "error: {err}");
    }

    #[test]
    fn validate_shapes_rejects_wrong_feats_len() {
        let emb = SamEmbedding {
            image_embed: vec![0.0; 1_048_576],
            high_res_feats_0: vec![0.0; 1],
            high_res_feats_1: vec![0.0; 1_048_576],
        };
        let err = emb.validate_shapes().unwrap_err();
        assert!(err.contains("high_res_feats_0"), "error: {err}");
    }

    #[test]
    fn decode_picks_the_highest_iou_candidate() {
        // 3 candidates, IoUs 0.3 / 0.7 / 0.5 → candidate 1 wins
        let mut masks = vec![0.0f32; 3 * 256 * 256];
        // Candidate 1 (index 1): all +1.0 logits → all selected after upsample
        for v in &mut masks[256 * 256..2 * 256 * 256] {
            *v = 1.0;
        }
        let output = SamDecoderOutput {
            masks,
            iou_predictions: [0.3, 0.7, 0.5],
        };
        let result = decode_to_mask_artifact(&output, 64, 64, plain(0.5), BrushMode::Add);
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        // All upsampled pixels are fully selected (candidate 1, all +1 logits)
        assert!(result.cells().iter().all(|&v| v == FULL));
        let result = decode_to_mask_artifact(&output, 64, 64, plain(0.5), BrushMode::Subtract);
        assert!(result.cells().iter().all(|&v| v == -FULL), "Subtract mode signs the region negative");
    }

    /// The knob is a probability: 0.5 is the model's decision boundary,
    /// higher values select a subset, lower values a superset.
    #[test]
    fn confidence_moves_the_pixel_threshold_monotonically() {
        assert!(confidence_logit(0.5).abs() < 1e-6);
        assert!(confidence_logit(0.9) > 0.0 && confidence_logit(0.1) < 0.0);
        let m = 256usize;
        // Logits ramp from -4 (left) to +4 (right) on every row.
        let mut masks = vec![0.0f32; 3 * m * m];
        for y in 0..m {
            for x in 0..m {
                masks[y * m + x] = -4.0 + 8.0 * x as f32 / (m - 1) as f32;
            }
        }
        let output = SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] };
        let count = |c: f32| decode_to_mask_artifact(&output, 64, 64, plain(c), BrushMode::Add)
            .cells().iter().filter(|&&v| v == FULL).count();
        let (low, mid, high) = (count(0.1), count(0.5), count(0.9));
        assert!(low > mid && mid > high, "{low} > {mid} > {high}");
        assert_eq!(mid, 64 * 32, "0.5 splits the ramp at logit 0");
        assert_eq!(count(0.0), count(0.02), "extremes clamp instead of selecting everything");
    }

    /// Per-pixel bilinear sample written the long way: pins the hoisted
    /// taps and the row-parallel indexing against the plain formula.
    fn brute_force_decode(logits: &[f32], w: u32, h: u32, selected: i8) -> Vec<i8> {
        let m = SAM_MASK_RESOLUTION as usize;
        let m_max_f = (SAM_MASK_RESOLUTION - 1) as f32;
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * (SAM_MASK_RESOLUTION as f32 / w as f32) - 0.5;
                let sy = (y as f32 + 0.5) * (SAM_MASK_RESOLUTION as f32 / h as f32) - 0.5;
                let x0 = sx.floor().clamp(0.0, m_max_f) as usize;
                let y0 = sy.floor().clamp(0.0, m_max_f) as usize;
                let x1 = (x0 + 1).min(m - 1);
                let y1 = (y0 + 1).min(m - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let fy = (sy - y0 as f32).clamp(0.0, 1.0);
                let v = logits[y0 * m + x0] * (1.0 - fx) * (1.0 - fy)
                    + logits[y0 * m + x1] * fx * (1.0 - fy)
                    + logits[y1 * m + x0] * (1.0 - fx) * fy
                    + logits[y1 * m + x1] * fx * fy;
                out.push(if v >= 0.0 { selected } else { 0 });
            }
        }
        out
    }

    #[test]
    fn decode_matches_the_per_pixel_bilinear_sample() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let n = (SAM_MASK_RESOLUTION * SAM_MASK_RESOLUTION) as usize;
        let mut masks = vec![0.0f32; 3 * n];
        for v in &mut masks[n..2 * n] {
            *v = rng.random_range(-2.0f32..2.0);
        }
        let output = SamDecoderOutput { masks, iou_predictions: [0.1, 0.9, 0.2] };
        for (w, h) in [(1, 1), (7, 3), (300, 200), (641, 97), (1024, 1024)] {
            let result = decode_to_mask_artifact(&output, w, h, plain(0.5), BrushMode::Subtract);
            let expected = brute_force_decode(&output.masks[n..2 * n], w, h, -FULL);
            assert!(result.cells() == expected.as_slice(), "{w}x{h}");
        }
    }

    #[test]
    fn decode_negative_logits_produce_zero_mask() {
        // Candidate 0 has IoU 0.9 but all logits are -1.0 → all mask pixels 0.0
        let mut masks = vec![0.0f32; 3 * 256 * 256];
        for v in &mut masks[0..256 * 256] {
            *v = -1.0;
        }
        let output = SamDecoderOutput {
            masks,
            iou_predictions: [0.9, 0.1, 0.1],
        };
        let result = decode_to_mask_artifact(&output, 32, 32, plain(0.5), BrushMode::Subtract);
        assert!(result.cells().iter().all(|&v| v == 0));
    }
}
