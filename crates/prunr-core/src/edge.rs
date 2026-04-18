use std::sync::Mutex;

use image::{DynamicImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::Session, value::Tensor};

use crate::types::CoreError;

const DEXINED_H: u32 = 480;
const DEXINED_W: u32 = 640;
// BGR mean for DexiNed (OpenCV Zoo variant)
const MEAN_BGR: [f32; 3] = [103.5, 116.2, 123.6];

/// Opaque wrapper around the DexiNed ORT session.
/// Thread-safe via internal Mutex (same pattern as OrtEngine).
pub struct EdgeEngine {
    session: Mutex<Session>,
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
        Ok(Self { session: Mutex::new(session) })
    }

    /// Run edge detection on an image. Returns RGBA where edges are opaque.
    /// If `line_color` is Some, all edge pixels are painted that color.
    ///
    /// Convenience wrapper around `infer_tensor` + `finalize_edges` for callers
    /// that don't need the intermediate tensor (e.g. CLI or single-shot flows).
    pub fn detect(&self, original: &DynamicImage, edge: &crate::EdgeSettings) -> Result<RgbaImage, CoreError> {
        let (tensor, h, w) = self.infer_tensor(original)?;
        Ok(finalize_edges(&tensor, h, w, original, edge))
    }

    /// Run only the DexiNed inference stage. Returns the raw sigmoid-logits
    /// tensor at DexiNed's native resolution (480×640), plus (height, width).
    ///
    /// Split from `detect` so callers can cache this tensor and rerun
    /// `finalize_edges` with new line_strength / line_color without re-running the model.
    pub fn infer_tensor(&self, original: &DynamicImage) -> Result<(Vec<f32>, u32, u32), CoreError> {
        let mut session = self.session.lock()
            .map_err(|e| CoreError::Inference(format!("Edge session lock failed: {e}")))?;

        let input_array = preprocess(original);
        let input_name = session.inputs()[0].name().to_string();
        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| CoreError::Inference(format!("Failed to create edge tensor: {e}")))?;
        let outputs = session
            .run(inputs![input_name.as_str() => &input_tensor])
            .map_err(|e| CoreError::Inference(format!("Edge detection failed: {e}")))?;

        // Take the fused output (last one: "block_cat")
        let fused_idx = outputs.len() - 1;
        let edge_map = outputs[fused_idx]
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("Failed to extract edge output: {e}")))?;
        let edge_slice = edge_map.as_slice()
            .ok_or_else(|| CoreError::Inference("Edge output tensor is not contiguous".to_string()))?;
        Ok((edge_slice.to_vec(), DEXINED_H, DEXINED_W))
    }
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
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0 };
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
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: None, edge_thickness: 0 };
        let out = finalize_edges(&tensor, h as u32, w as u32, &original, &edge);
        // Original color preserved
        assert_eq!(out.get_pixel(0, 0)[0], 120);
        assert_eq!(out.get_pixel(0, 0)[1], 120);
        assert_eq!(out.get_pixel(0, 0)[2], 120);
    }
}
