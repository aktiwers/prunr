//! A coarse look at where the subject is, taken the moment segmentation
//! inference ends, so the app can outline it while the edges are refined.

use serde::{Deserialize, Serialize};

use crate::types::ModelKind;

/// Longest side of a subject hint; plenty for a glow around the subject.
pub const SUBJECT_HINT_SIDE: usize = 128;

/// Subject coverage over the whole image (stretched to the model's square
/// input), `0` background to `255` subject, row-major.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubjectHint {
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

/// The hint for a raw `tensor_h × tensor_w` segmentation output, read the
/// way the mask is (`SegNormaliser`), over the sampled values.
pub fn subject_hint(tensor: &[f32], tensor_h: usize, tensor_w: usize, model: ModelKind) -> SubjectHint {
    let scale = (SUBJECT_HINT_SIDE as f32 / tensor_h.max(tensor_w).max(1) as f32).min(1.0);
    let (w, h) = (((tensor_w as f32 * scale) as usize).max(1), ((tensor_h as f32 * scale) as usize).max(1));
    let samples: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let ty = (y * tensor_h / h).min(tensor_h - 1);
            let tx = (x * tensor_w / w).min(tensor_w - 1);
            tensor.get(ty * tensor_w + tx).copied().unwrap_or(0.0)
        })
        .collect();
    let (lo, hi) = samples.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let norm = crate::postprocess::SegNormaliser::new(lo, hi, model);
    let coverage = samples.iter().map(|&v| (norm.apply(v) * 255.0) as u8).collect();
    SubjectHint { width: w as u32, height: h as u32, coverage }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hint_is_small_stretched_and_keeps_where_the_subject_is() {
        let (h, w) = (1024usize, 1024usize);
        let tensor: Vec<f32> = (0..h * w).map(|i| if (i % w) < w / 2 { 4.0 } else { -4.0 }).collect();
        let hint = subject_hint(&tensor, h, w, ModelKind::BiRefNetLite);
        assert_eq!((hint.width, hint.height), (128, 128));
        assert_eq!(hint.coverage[0], 255, "the left half is the subject");
        assert_eq!(hint.coverage[127], 0, "the right half is not");
    }

    #[test]
    fn a_flat_tensor_reads_as_all_or_nothing() {
        let hint = subject_hint(&[0.9; 16], 4, 4, ModelKind::Silueta);
        assert!(hint.coverage.iter().all(|&c| c == 255));
        assert_eq!((hint.width, hint.height), (4, 4), "a small tensor is not enlarged");
    }
}
