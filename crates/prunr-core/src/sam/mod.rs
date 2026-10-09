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

/// Pick the highest-IoU candidate above `confidence_threshold`; bilinear-
/// upsample its 256×256 logits to source resolution; threshold at logit
/// 0.0 (sigmoid 0.5) into a binary MaskArtifact signed by `mode`.
///
/// Returns `None` if no candidate passes the confidence threshold.
///
/// Peak working set: source_w * source_h bytes output + 256*256*4 bytes
/// slice view (read-only). At 4K source: ~8 MB output.
pub fn decode_to_mask_artifact(
    output: &SamDecoderOutput,
    source_w: u32,
    source_h: u32,
    confidence_threshold: f32,
    mode: BrushMode,
) -> Option<crate::selection::MaskArtifact> {
    let best_idx = output
        .iou_predictions
        .iter()
        .enumerate()
        .filter(|(_, &iou)| iou >= confidence_threshold)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;

    let m = SAM_MASK_RESOLUTION as usize;
    let m_max = SAM_MASK_RESOLUTION - 1;
    let m_max_f = m_max as f32;
    let mask_logits: &[f32] = &output.masks[best_idx * m * m..(best_idx + 1) * m * m];

    let selected = mode.sign() * FULL;
    // Bilinear taps are separable: the x taps repeat on every row and
    // the y taps on every column, so each is computed once. Half-pixel
    // offset for correct bilinear alignment.
    let taps = |i: u32, n: u32| -> (usize, usize, f32) {
        let scale = SAM_MASK_RESOLUTION as f32 / n as f32;
        let s = (i as f32 + 0.5) * scale - 0.5;
        let i0 = s.floor().clamp(0.0, m_max_f) as u32;
        let i1 = (i0 + 1).min(m_max);
        (i0 as usize, i1 as usize, (s - i0 as f32).clamp(0.0, 1.0))
    };
    let xs: Vec<(usize, usize, f32)> = (0..source_w).map(|x| taps(x, source_w)).collect();
    let mut data = vec![0i8; (source_w * source_h) as usize];
    data.par_chunks_mut(source_w as usize).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = taps(y as u32, source_h);
        let (top, bottom) = (&mask_logits[y0 * m..(y0 + 1) * m], &mask_logits[y1 * m..(y1 + 1) * m]);
        for (cell, &(x0, x1, fx)) in row.iter_mut().zip(&xs) {
            let v = top[x0] * (1.0 - fx) * (1.0 - fy)
                + top[x1] * fx * (1.0 - fy)
                + bottom[x0] * (1.0 - fx) * fy
                + bottom[x1] * fx * fy;
            *cell = if v >= 0.0 { selected } else { 0 };
        }
    });

    Some(crate::selection::MaskArtifact::from_cells(source_w, source_h, data))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn decode_picks_highest_iou_above_threshold() {
        // 3 candidates, IoUs 0.3 / 0.7 / 0.5; threshold 0.4 → candidate 1 wins
        let mut masks = vec![0.0f32; 3 * 256 * 256];
        // Candidate 1 (index 1): all +1.0 logits → all selected after upsample
        for v in &mut masks[256 * 256..2 * 256 * 256] {
            *v = 1.0;
        }
        let output = SamDecoderOutput {
            masks,
            iou_predictions: [0.3, 0.7, 0.5],
        };
        let result = decode_to_mask_artifact(&output, 64, 64, 0.4, BrushMode::Add).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        // All upsampled pixels are fully selected (candidate 1, all +1 logits)
        assert!(result.cells().iter().all(|&v| v == FULL));
        let result = decode_to_mask_artifact(&output, 64, 64, 0.4, BrushMode::Subtract).unwrap();
        assert!(result.cells().iter().all(|&v| v == -FULL), "Subtract mode signs the region negative");
    }

    #[test]
    fn decode_returns_none_if_all_below_threshold() {
        let masks = vec![1.0f32; 3 * 256 * 256];
        let output = SamDecoderOutput {
            masks,
            iou_predictions: [0.1, 0.2, 0.3],
        };
        let result = decode_to_mask_artifact(&output, 64, 64, 0.5, BrushMode::Add);
        assert!(result.is_none());
    }

    /// Per-pixel bilinear sample written the long way, as the spec for
    /// the row-parallel decode.
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
            let result = decode_to_mask_artifact(&output, w, h, 0.5, BrushMode::Subtract).unwrap();
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
        let result = decode_to_mask_artifact(&output, 32, 32, 0.5, BrushMode::Subtract).unwrap();
        assert!(result.cells().iter().all(|&v| v == 0));
    }
}
