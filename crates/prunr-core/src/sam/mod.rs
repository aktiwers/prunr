//! SAM 2 inference primitives — encoder preprocessing, decoder prompt
//! construction, and decoder output → MaskArtifact conversion.
//!
//! PURE module — no ORT dependency. The ort::Session.run plumbing
//! lives in `prunr-app/src/gui/processor.rs` so this module stays
//! unit-testable from prunr-core alone.

use std::sync::Arc;

use crate::selection::FULL;

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
/// 0.0 (sigmoid 0.5) into a binary, unsigned MaskArtifact (`FULL` / 0).
/// The caller signs it with the active brush mode (`with_mode`).
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
    let m_max_f = (SAM_MASK_RESOLUTION - 1) as f32;
    let mask_logits: &[f32] = &output.masks[best_idx * m * m..(best_idx + 1) * m * m];

    let mut data = vec![0i8; (source_w * source_h) as usize];
    let scale_x = SAM_MASK_RESOLUTION as f32 / source_w as f32;
    let scale_y = SAM_MASK_RESOLUTION as f32 / source_h as f32;
    for y in 0..source_h {
        for x in 0..source_w {
            // Half-pixel offset for correct bilinear alignment.
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            let sy = (y as f32 + 0.5) * scale_y - 0.5;
            let x0 = sx.floor().clamp(0.0, m_max_f) as u32;
            let y0 = sy.floor().clamp(0.0, m_max_f) as u32;
            let x1 = (x0 + 1).min(m_max);
            let y1 = (y0 + 1).min(m_max);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            let i00 = (y0 as usize) * m + x0 as usize;
            let i01 = (y0 as usize) * m + x1 as usize;
            let i10 = (y1 as usize) * m + x0 as usize;
            let i11 = (y1 as usize) * m + x1 as usize;
            let v = mask_logits[i00] * (1.0 - fx) * (1.0 - fy)
                + mask_logits[i01] * fx * (1.0 - fy)
                + mask_logits[i10] * (1.0 - fx) * fy
                + mask_logits[i11] * fx * fy;
            data[(y * source_w + x) as usize] = if v >= 0.0 { FULL } else { 0 };
        }
    }

    Some(crate::selection::MaskArtifact {
        width: source_w,
        height: source_h,
        data: Arc::new(data),
    })
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
        let result = decode_to_mask_artifact(&output, 64, 64, 0.4).unwrap();
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        // All upsampled pixels are fully selected (candidate 1, all +1 logits)
        assert!(result.data.iter().all(|&v| v == FULL));
    }

    #[test]
    fn decode_returns_none_if_all_below_threshold() {
        let masks = vec![1.0f32; 3 * 256 * 256];
        let output = SamDecoderOutput {
            masks,
            iou_predictions: [0.1, 0.2, 0.3],
        };
        let result = decode_to_mask_artifact(&output, 64, 64, 0.5);
        assert!(result.is_none());
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
        let result = decode_to_mask_artifact(&output, 32, 32, 0.5).unwrap();
        assert!(result.data.iter().all(|&v| v == 0));
    }
}
