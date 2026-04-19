use std::sync::Mutex;

use image::{DynamicImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::Session, value::Tensor};

use crate::types::{CoreError, EdgeScale};

const DEXINED_H: u32 = 480;
const DEXINED_W: u32 = 640;
// BGR mean for DexiNed (OpenCV Zoo variant)
const MEAN_BGR: [f32; 3] = [103.5, 116.2, 123.6];

/// Number of DexiNed outputs we surface as `EdgeScale` variants.
pub const EDGE_SCALE_COUNT: usize = 4;

/// All 4 outputs from one DexiNed inference, indexed by `EdgeScale as usize`.
pub struct EdgeInferenceResult {
    pub tensors: [Vec<f32>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

/// Opaque wrapper around the DexiNed ORT session.
/// Thread-safe via internal Mutex (same pattern as OrtEngine).
pub struct EdgeEngine {
    session: Mutex<Session>,
}

/// Compile-time lock on `EdgeScale` discriminants. `infer_all_tensors` builds
/// the result array as `[fine, balanced, bold, fused]` and callers index by
/// `scale as usize`; reordering the enum without updating the array would
/// silently point every scale at the wrong tensor. This assertion fails the
/// build before that can ship.
const _: () = {
    assert!(EdgeScale::Fine as usize == 0);
    assert!(EdgeScale::Balanced as usize == 1);
    assert!(EdgeScale::Bold as usize == 2);
    assert!(EdgeScale::Fused as usize == 3);
    assert!(EDGE_SCALE_COUNT == 4);
};

/// Layout assumption: `block0..block5` (fine → coarse), then fused `block_cat`.
/// Validated at `EdgeEngine::new`.
fn scale_to_output_index(scale: EdgeScale, last: usize) -> usize {
    match scale {
        EdgeScale::Fine => 0,
        EdgeScale::Balanced => 3,
        EdgeScale::Bold => 5,
        EdgeScale::Fused => last,
    }
}

impl EdgeEngine {
    /// Create a new DexiNed edge detection engine.
    pub fn new() -> Result<Self, CoreError> {
        let edge_bytes = prunr_models::dexined_bytes();
        let session = Session::builder()
            .map_err(|e| CoreError::Inference(format!("Edge builder init failed: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("Edge set opt level failed: {e}")))?
            .with_intra_threads(num_cpus::get().max(1))
            .map_err(|e| CoreError::Inference(format!("Edge set threads failed: {e}")))?
            .commit_from_memory(&edge_bytes)
            .map_err(|e| CoreError::Inference(format!("Edge model load failed: {e}")))?;

        // Validate the output layout. If a model re-export ever renames /
        // reorders the outputs, we want a clear init error instead of a
        // silent wrong-scale result downstream.
        let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        tracing::info!(?names, "DexiNed output layout");
        let last = names.last().map(String::as_str);
        if last != Some("block_cat") {
            return Err(CoreError::Inference(format!(
                "DexiNed export layout changed: expected last output 'block_cat', got {:?}. \
                 Scale selection would pick the wrong tensor.",
                last,
            )));
        }
        if names.len() < 6 {
            return Err(CoreError::Inference(format!(
                "DexiNed export layout changed: expected ≥6 side outputs + block_cat, got {} total.",
                names.len(),
            )));
        }

        Ok(Self { session: Mutex::new(session) })
    }

    /// One-shot: inference + finalize_edges for CLI / single-shot flows.
    pub fn detect(&self, original: &DynamicImage, edge: &crate::EdgeSettings) -> Result<RgbaImage, CoreError> {
        let (tensor, h, w) = self.infer_tensor(original, edge.edge_scale)?;
        Ok(finalize_edges(&tensor, h, w, original, edge))
    }

    /// Extract a single scale from one inference run. Use when the caller
    /// doesn't cache per-scale tensors (CLI).
    pub fn infer_tensor(&self, original: &DynamicImage, scale: EdgeScale) -> Result<(Vec<f32>, u32, u32), CoreError> {
        let mut session = self.session.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = run_inference(&mut session, original)?;
        let idx = scale_to_output_index(scale, outputs.len() - 1);
        let tensor = extract_output(&outputs, idx)?;
        Ok((tensor, DEXINED_H, DEXINED_W))
    }

    /// Extract all 4 scales from one inference run. Used by the GUI subprocess
    /// path so scale switching in live preview is a cached tensor lookup.
    pub fn infer_all_tensors(&self, original: &DynamicImage) -> Result<EdgeInferenceResult, CoreError> {
        let mut session = self.session.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = run_inference(&mut session, original)?;
        let last = outputs.len() - 1;
        // Order must match `EdgeScale as usize` so callers can index by it.
        let fine = extract_output(&outputs, scale_to_output_index(EdgeScale::Fine, last))?;
        let balanced = extract_output(&outputs, scale_to_output_index(EdgeScale::Balanced, last))?;
        let bold = extract_output(&outputs, scale_to_output_index(EdgeScale::Bold, last))?;
        let fused = extract_output(&outputs, scale_to_output_index(EdgeScale::Fused, last))?;
        Ok(EdgeInferenceResult {
            tensors: [fine, balanced, bold, fused],
            height: DEXINED_H,
            width: DEXINED_W,
        })
    }
}

/// Run the ONNX session once and return the raw outputs vec. Shared by the
/// single-scale and multi-scale extraction paths.
fn run_inference<'s>(
    session: &'s mut Session,
    original: &DynamicImage,
) -> Result<ort::session::SessionOutputs<'s>, CoreError> {
    let input_array = preprocess(original);
    let input_name = session.inputs()[0].name().to_string();
    let input_tensor = Tensor::from_array(input_array)
        .map_err(|e| CoreError::Inference(format!("Failed to create edge tensor: {e}")))?;
    session
        .run(inputs![input_name.as_str() => &input_tensor])
        .map_err(|e| CoreError::Inference(format!("Edge detection failed: {e}")))
}

/// Pull one output tensor from a session result by index, copying into a Vec.
fn extract_output(outputs: &ort::session::SessionOutputs, idx: usize) -> Result<Vec<f32>, CoreError> {
    let edge_map = outputs[idx]
        .try_extract_array::<f32>()
        .map_err(|e| CoreError::Inference(format!("Failed to extract edge output at index {idx}: {e}")))?;
    let slice = edge_map.as_slice()
        .ok_or_else(|| CoreError::Inference("Edge output tensor is not contiguous".to_string()))?;
    Ok(slice.to_vec())
}

/// Threshold + resize a DexiNed tensor into a full-resolution edge mask.
/// Depends only on `line_strength`; callers can cache the result and reuse it
/// across `edge_thickness` / `solid_line_color` tweaks.
pub fn tensor_to_edge_mask(
    edge_tensor: &[f32],
    tensor_h: u32,
    tensor_w: u32,
    out_w: u32,
    out_h: u32,
    line_strength: f32,
) -> image::GrayImage {
    let h = tensor_h as usize;
    let w = tensor_w as usize;

    // Sigmoid → edge probability, then apply strength as contrast/threshold control.
    // Exponential curve: slider 0.0→threshold 0.95, slider 0.5→0.3, slider 1.0→0.01
    let s = line_strength.clamp(0.0, 1.0);
    let threshold = (1.0 - s).powi(2) * 0.95 + 0.01;
    let mut mask_buf = vec![0u8; h * w];
    for i in 0..h * w {
        let prob = 1.0 / (1.0 + (-edge_tensor[i]).exp());
        // Smooth step: remap [threshold-0.1, threshold+0.1] to [0, 1] for anti-aliased edges
        let edge = ((prob - threshold + 0.1) / 0.2).clamp(0.0, 1.0);
        let val = edge * edge * (3.0 - 2.0 * edge); // smoothstep
        mask_buf[i] = (val * 255.0) as u8;
    }

    let mask = image::GrayImage::from_raw(w as u32, h as u32, mask_buf)
        .expect("edge mask buffer size matches dimensions");
    crate::formats::resize_gray_lanczos3(&mask, out_w, out_h)
}

/// Dilate + composite a pre-built edge mask into an RGBA. Cheap; safe to call
/// every live-preview tweak.
///
/// Semantics: the output alpha channel IS the dilated edge mask — any alpha
/// already on `original` is overwritten. That's what `LineMode::EdgesOnly`
/// wants (show only the lines, transparent everywhere else). For
/// `LineMode::SubjectOutline`, use `compose_edges_over_rgba` instead — it
/// merges the edge mask with the base's existing alpha so the masked subject
/// stays visible under the outline.
pub fn compose_edges(
    mask: &image::GrayImage,
    original: &DynamicImage,
    solid_line_color: Option<[u8; 3]>,
    edge_thickness: u32,
) -> RgbaImage {
    let (ow, oh) = (original.width(), original.height());
    let mut mask = mask.clone();
    crate::postprocess::dilate_mask(&mut mask, edge_thickness);
    let mask_raw = mask.as_raw();
    if let Some(c) = solid_line_color {
        let mut buf = vec![0u8; (ow * oh * 4) as usize];
        for i in 0..(ow * oh) as usize {
            buf[i * 4]     = c[0];
            buf[i * 4 + 1] = c[1];
            buf[i * 4 + 2] = c[2];
            buf[i * 4 + 3] = mask_raw[i];
        }
        RgbaImage::from_raw(ow, oh, buf).expect("edge output buffer size matches dimensions")
    } else {
        let mut rgba = original.to_rgba8();
        let out_raw = rgba.as_mut();
        for i in 0..(ow * oh) as usize {
            out_raw[i * 4 + 3] = mask_raw[i];
        }
        rgba
    }
}

/// Composite a pre-built edge mask INSIDE an already-masked RGBA — the edge
/// only shows where the base's alpha (subject mask) also contains content.
/// Used by `SubjectOutline` mode.
///
/// Alpha merge is multiplicative: `alpha = base_alpha * edge_mask / 255`.
/// That keeps the lines-only silhouette the user expects (transparent
/// background, no filled subject) while tying line visibility to the subject
/// mask — so gamma / threshold / refine tweaks visibly change where the
/// lines fade out along the subject silhouette. With a `solid_line_color`,
/// edge pixels are recoloured to that color; otherwise the base RGB shows
/// through at edges.
pub fn compose_edges_over_rgba(
    mask: &image::GrayImage,
    base: &RgbaImage,
    solid_line_color: Option<[u8; 3]>,
    edge_thickness: u32,
) -> RgbaImage {
    let (ow, oh) = (base.width(), base.height());
    let mut mask = mask.clone();
    crate::postprocess::dilate_mask(&mut mask, edge_thickness);
    let mask_raw = mask.as_raw();
    let mut rgba = base.clone();
    let out_raw = rgba.as_mut();
    if let Some(c) = solid_line_color {
        for i in 0..(ow * oh) as usize {
            let subject = out_raw[i * 4 + 3] as u16;
            let edge = mask_raw[i] as u16;
            let effective = (subject * edge / 255) as u8;
            if effective == 0 {
                out_raw[i * 4 + 3] = 0;
                continue;
            }
            out_raw[i * 4]     = c[0];
            out_raw[i * 4 + 1] = c[1];
            out_raw[i * 4 + 2] = c[2];
            out_raw[i * 4 + 3] = effective;
        }
    } else {
        for i in 0..(ow * oh) as usize {
            let subject = out_raw[i * 4 + 3] as u16;
            let edge = mask_raw[i] as u16;
            out_raw[i * 4 + 3] = (subject * edge / 255) as u8;
        }
    }
    rgba
}

/// Tier 2 edge convenience: tensor → mask → RGBA in one call. Prefer the two
/// split functions when you want to cache the mask between dispatches.
pub fn finalize_edges(
    edge_tensor: &[f32],
    tensor_h: u32,
    tensor_w: u32,
    original: &DynamicImage,
    edge: &crate::EdgeSettings,
) -> RgbaImage {
    let mask = tensor_to_edge_mask(
        edge_tensor,
        tensor_h,
        tensor_w,
        original.width(),
        original.height(),
        edge.line_strength,
    );
    compose_edges(&mask, original, edge.solid_line_color, edge.edge_thickness)
}

/// Preprocess an image for DexiNed: resize, BGR float32, subtract mean.
/// Flatten RGBA onto white: transparent pixels become white so edge detection
/// doesn't see ghost content behind removed backgrounds.
fn flatten_on_white(img: &DynamicImage) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let src = rgba.as_raw();
    let mut out = vec![255u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        let a = src[i * 4 + 3] as f32 / 255.0;
        if a > 0.0 {
            // Alpha-blend onto white: result = fg * alpha + 255 * (1 - alpha)
            let inv_a = 1.0 - a;
            out[i * 4]     = (src[i * 4]     as f32 * a + 255.0 * inv_a) as u8;
            out[i * 4 + 1] = (src[i * 4 + 1] as f32 * a + 255.0 * inv_a) as u8;
            out[i * 4 + 2] = (src[i * 4 + 2] as f32 * a + 255.0 * inv_a) as u8;
        }
        out[i * 4 + 3] = 255; // fully opaque for preprocessing
    }
    DynamicImage::ImageRgba8(RgbaImage::from_raw(w, h, out).expect("flatten buffer size matches dimensions"))
}

fn preprocess(img: &DynamicImage) -> Array4<f32> {
    let flattened;
    let source = if img.color().has_alpha() {
        flattened = flatten_on_white(img);
        &flattened
    } else {
        img
    };
    let resized = crate::formats::resize_rgb_lanczos3(source, DEXINED_W, DEXINED_H);
    let raw = resized.as_raw();
    let h = DEXINED_H as usize;
    let w = DEXINED_W as usize;

    let mut out = Array4::<f32>::zeros((1, 3, h, w));
    // DexiNed expects BGR with mean subtraction (no /255)
    for c in 0..3 {
        let bgr_c = 2 - c; // RGB→BGR: channel 0(R)→2(B), 1(G)→1(G), 2(B)→0(R)
        let mut plane = out.slice_mut(ndarray::s![0, bgr_c, .., ..]);
        // invariant: slice of a freshly-zeroed Array4 is contiguous.
        let plane_slice = plane.as_slice_mut().unwrap();
        for i in 0..h * w {
            plane_slice[i] = raw[i * 3 + c] as f32 - MEAN_BGR[bgr_c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, Rgb};

    fn solid_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([120, 120, 120])))
    }

    #[test]
    fn finalize_edges_solid_color_paints_edges() {
        // Synthetic edge tensor: half high logits, half zero.
        let w = DEXINED_W as usize;
        let h = DEXINED_H as usize;
        let mut tensor = vec![0.0_f32; h * w];
        for i in 0..h * w / 2 {
            tensor[i] = 10.0; // sigmoid → ~1 → edge
        }
        let original = solid_rgb(64, 48);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0, edge_scale: crate::EdgeScale::Fused };
        let out = finalize_edges(&tensor, h as u32, w as u32, &original, &edge);
        assert_eq!(out.width(), 64);
        assert_eq!(out.height(), 48);
        // With high-logit side → opaque red, zero-logit → transparent.
        let strong_red = out.get_pixel(0, 0);
        assert_eq!([strong_red[0], strong_red[1], strong_red[2]], [255, 0, 0]);
    }

    #[test]
    fn finalize_edges_preserves_original_rgb_when_no_line_color() {
        let w = DEXINED_W as usize;
        let h = DEXINED_H as usize;
        let tensor = vec![10.0_f32; h * w]; // all edges
        let original = solid_rgb(32, 32);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: None, edge_thickness: 0, edge_scale: crate::EdgeScale::Fused };
        let out = finalize_edges(&tensor, h as u32, w as u32, &original, &edge);
        // Original color preserved
        assert_eq!(out.get_pixel(0, 0)[0], 120);
        assert_eq!(out.get_pixel(0, 0)[1], 120);
        assert_eq!(out.get_pixel(0, 0)[2], 120);
    }
}
