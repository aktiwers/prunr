//! SAM 2 decoder prompt construction. Click + Stroke interactions
//! with Shift (add) + Alt (subtract) modifiers. All coordinate
//! normalization runs through `normalize` — source-pixel coords are
//! mapped to 0..SAM_ENCODER_INPUT space; screen-pixel or raw click coords
//! must NOT reach the decoder, the resulting mask would be off-by-scale.

use super::{SAM_ENCODER_INPUT, SAM_MASK_RESOLUTION};

/// Decimation cap for stroke prompts. 8 points is the practical sweet
/// spot — more add noise to the decoder without improving mask quality.
pub const MAX_STROKE_POINTS: usize = 8;

/// Decoder prompt: the four tensors the SAM 2 decoder expects
/// for point-based interaction.
#[derive(Debug)]
pub struct SamPrompt {
    /// f32[N, 2] flattened — [x0, y0, x1, y1, ...] in 0–1024 space
    pub point_coords: Vec<f32>,
    /// f32[N] — 1.0=fg, 0.0=bg, -1.0=padding
    pub point_labels: Vec<f32>,
    /// f32[1, 1, 256, 256] zeros for first call (no prior mask)
    pub mask_input: Vec<f32>,
    /// 0.0 for first call (no prior mask)
    pub has_mask_input: f32,
}

/// Returned when a stroke has zero points.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("stroke must have at least one point")]
    EmptyStroke,
}

fn empty_mask_input() -> Vec<f32> {
    let m = SAM_MASK_RESOLUTION as usize;
    vec![0.0; m * m]
}

/// Normalize source-pixel coords to 0–1024 space, clamping out-of-bounds
/// inputs. SAM 2 always operates at SAM_ENCODER_INPUT×SAM_ENCODER_INPUT
/// regardless of the source image resolution.
fn normalize(x_pixel: f32, y_pixel: f32, source_w: u32, source_h: u32) -> (f32, f32) {
    let cx = x_pixel.clamp(0.0, source_w.saturating_sub(1) as f32);
    let cy = y_pixel.clamp(0.0, source_h.saturating_sub(1) as f32);
    let nx = (cx / source_w as f32) * SAM_ENCODER_INPUT as f32;
    let ny = (cy / source_h as f32) * SAM_ENCODER_INPUT as f32;
    (nx, ny)
}

/// Single-point click prompt. Foreground hint with one padding slot
/// to satisfy SAM 2's even-N convention. Caller applies the resulting
/// MaskArtifact directly (replaces current selection).
pub fn build_click_prompt(
    x_pixel: f32,
    y_pixel: f32,
    source_w: u32,
    source_h: u32,
) -> SamPrompt {
    let (nx, ny) = normalize(x_pixel, y_pixel, source_w, source_h);
    SamPrompt {
        point_coords: vec![nx, ny, 0.0, 0.0], // real point + padding at (0,0)
        point_labels: vec![1.0, -1.0],        // fg + padding
        mask_input: empty_mask_input(),
        has_mask_input: 0.0,
    }
}

/// Multi-point stroke prompt. Decimates to at most MAX_STROKE_POINTS evenly-
/// spaced samples, all foreground. Always appends one padding point at (0,0)
/// with label -1.0 for SAM 2 robustness. Caller applies the resulting
/// MaskArtifact directly (replaces current selection).
///
/// Returns `Err(PromptError::EmptyStroke)` for zero-length input.
pub fn build_stroke_prompt(
    points: &[(f32, f32)],
    source_w: u32,
    source_h: u32,
) -> Result<SamPrompt, PromptError> {
    if points.is_empty() {
        return Err(PromptError::EmptyStroke);
    }

    let decimated = decimate(points, MAX_STROKE_POINTS);
    let n = decimated.len();
    // +1 padding slot appended after all real points
    let mut point_coords = Vec::with_capacity((n + 1) * 2);
    let mut point_labels = Vec::with_capacity(n + 1);

    for (x, y) in &decimated {
        let (nx, ny) = normalize(*x, *y, source_w, source_h);
        point_coords.push(nx);
        point_coords.push(ny);
        point_labels.push(1.0f32);
    }
    // Padding slot
    point_coords.push(0.0);
    point_coords.push(0.0);
    point_labels.push(-1.0);

    Ok(SamPrompt {
        point_coords,
        point_labels,
        mask_input: empty_mask_input(),
        has_mask_input: 0.0,
    })
}

/// Alt modifier — background hint (label 0.0). Caller applies the resulting
/// MaskArtifact via `MaskArtifact::subtract_mask` to remove from selection.
pub fn build_alt_modifier_prompt(
    x_pixel: f32,
    y_pixel: f32,
    source_w: u32,
    source_h: u32,
) -> SamPrompt {
    let (nx, ny) = normalize(x_pixel, y_pixel, source_w, source_h);
    SamPrompt {
        point_coords: vec![nx, ny, 0.0, 0.0],
        point_labels: vec![0.0, -1.0], // background hint + padding
        mask_input: empty_mask_input(),
        has_mask_input: 0.0,
    }
}

/// Uniform stride decimation: pick `cap` evenly-spaced samples from input.
/// If input length ≤ cap, returns all points unchanged.
fn decimate(points: &[(f32, f32)], cap: usize) -> Vec<(f32, f32)> {
    if points.len() <= cap {
        return points.to_vec();
    }
    let step = points.len() as f32 / cap as f32;
    (0..cap)
        .map(|i| {
            let idx = (i as f32 * step) as usize;
            points[idx.min(points.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_prompt_has_real_point_and_padding() {
        let p = build_click_prompt(100.0, 200.0, 1000, 1000);
        assert_eq!(p.point_coords.len(), 4); // 2 points × 2 coords
        assert_eq!(p.point_labels.len(), 2);
        assert_eq!(p.point_labels, vec![1.0, -1.0]);
    }

    #[test]
    fn click_prompt_normalizes_to_1024_space() {
        // 500 / 1000 * 1024 = 512.0
        let p = build_click_prompt(500.0, 500.0, 1000, 1000);
        assert!(
            (p.point_coords[0] - 512.0).abs() < 0.1,
            "x expected 512, got {}",
            p.point_coords[0]
        );
        assert!(
            (p.point_coords[1] - 512.0).abs() < 0.1,
            "y expected 512, got {}",
            p.point_coords[1]
        );
    }

    #[test]
    fn click_prompt_clamps_out_of_bounds_input() {
        let p = build_click_prompt(99999.0, -5.0, 1000, 1000);
        // X clamped to source_w - 1 = 999; 999/1000 * 1024 = 1022.976
        assert!(p.point_coords[0] < SAM_ENCODER_INPUT as f32, "x must be < SAM_ENCODER_INPUT, got {}", p.point_coords[0]);
        assert!(p.point_coords[1] >= 0.0, "y must be >= 0, got {}", p.point_coords[1]);
    }

    #[test]
    fn stroke_prompt_decimates_to_max_8_points() {
        let points: Vec<(f32, f32)> = (0..100).map(|i| (i as f32, i as f32)).collect();
        let p = build_stroke_prompt(&points, 100, 100).unwrap();
        // 8 real points + 1 padding = 9 labels
        assert_eq!(p.point_labels.len(), 9, "expected 9 labels (8 real + 1 padding)");
        assert_eq!(p.point_coords.len(), 18, "expected 18 coords (9 points × 2)");
        // All real labels are 1.0; last is padding -1.0
        assert!(p.point_labels[..8].iter().all(|&v| v == 1.0));
        assert_eq!(p.point_labels[8], -1.0);
    }

    #[test]
    fn stroke_short_input_keeps_all_points() {
        let points = vec![(10.0f32, 20.0), (30.0, 40.0)];
        let p = build_stroke_prompt(&points, 100, 100).unwrap();
        // 2 real + 1 padding
        assert_eq!(p.point_labels.len(), 3);
        assert_eq!(p.point_labels[2], -1.0);
    }

    #[test]
    fn stroke_empty_returns_err() {
        let result = build_stroke_prompt(&[], 100, 100);
        assert!(matches!(result, Err(PromptError::EmptyStroke)));
    }

    #[test]
    fn alt_prompt_uses_label_zero_not_one() {
        let p = build_alt_modifier_prompt(50.0, 50.0, 100, 100);
        assert_eq!(p.point_labels[0], 0.0, "alt hint should be bg (0.0)");
        assert_eq!(p.point_labels[1], -1.0, "second slot should be padding");
    }

    #[test]
    fn mask_input_is_zero_filled_at_sam_mask_resolution() {
        let p = build_click_prompt(0.0, 0.0, 100, 100);
        let m = SAM_MASK_RESOLUTION as usize;
        assert_eq!(p.mask_input.len(), m * m);
        assert!(p.mask_input.iter().all(|&v| v == 0.0));
        assert_eq!(p.has_mask_input, 0.0);
    }
}
