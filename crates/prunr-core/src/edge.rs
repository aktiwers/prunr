use std::mem::MaybeUninit;
use std::sync::Mutex;

use image::{DynamicImage, RgbaImage};
use ndarray::Array4;
use rayon::prelude::*;
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
        let session = crate::ort_runtime::session_builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("Edge set opt level failed: {e}")))?
            .with_intra_threads(num_cpus::get().max(1))
            .map_err(|e| CoreError::Inference(format!("Edge set threads failed: {e}")))?
            .commit_from_memory(edge_bytes)
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
    // Clamp the transition window floor at 0: for line_strength > 0.75 the
    // threshold falls below 0.10, so `threshold - 0.1` goes negative. Without
    // the clamp, background pixels (prob ≈ 0) land in the negative-floored
    // window and produce ~50% alpha — a gray haze across the whole image.
    let lo = (threshold - 0.1_f32).max(0.0);
    let mut mask_buf = vec![0u8; h * w];
    for i in 0..h * w {
        let prob = 1.0 / (1.0 + (-edge_tensor[i]).exp());
        // Remap [lo, threshold+0.1] to [0, 1] for anti-aliased edges.
        let edge = ((prob - lo) / 0.2).clamp(0.0, 1.0);
        mask_buf[i] = (crate::math::smoothstep(edge) * 255.0) as u8;
    }

    let mask = image::GrayImage::from_raw(w as u32, h as u32, mask_buf)
        .expect("edge mask buffer size matches dimensions");
    crate::formats::resize_gray_lanczos3(&mask, out_w, out_h)
}

/// `base` dilated by `thickness` pixels; `None` at 0, where the base
/// already is the plane a composition draws.
pub fn thickened_edges(base: &image::GrayImage, thickness: u32) -> Option<image::GrayImage> {
    (thickness > 0).then(|| {
        let mut m = base.clone();
        crate::morphology::shift_mask(&mut m, -(thickness as f32));
        m
    })
}

/// The edge plane a composition draws: `tensor_to_edge_mask` dilated by
/// `thickness` pixels. Live preview caches the two steps separately so a
/// thickness drag redoes the dilation only.
pub fn edge_plane(
    edge_tensor: &[f32],
    tensor_h: u32,
    tensor_w: u32,
    out_w: u32,
    out_h: u32,
    line_strength: f32,
    thickness: u32,
) -> image::GrayImage {
    let base = tensor_to_edge_mask(edge_tensor, tensor_h, tensor_w, out_w, out_h, line_strength);
    thickened_edges(&base, thickness).unwrap_or(base)
}

/// Composite a dilated edge plane into an RGBA. Cheap; safe to call every
/// live-preview tweak.
///
/// Semantics: the output alpha channel IS the edge plane — any alpha
/// already on `original` is overwritten. That's what `LineMode::EdgesOnly`
/// wants (show only the lines, transparent everywhere else). For
/// `LineMode::SubjectOutline`, use `compose_edges_styled` instead — it
/// merges the edge plane with the base's existing alpha so the masked
/// subject stays visible under the outline.
pub fn compose_edges(
    mask: &image::GrayImage,
    original: &DynamicImage,
    solid_line_color: Option<[u8; 3]>,
) -> RgbaImage {
    let (ow, oh) = (original.width(), original.height());
    let (row_len, mask_raw) = (ow as usize * 4, mask.as_raw());
    let mut buf = vec![0u8; (ow * oh * 4) as usize];
    match (solid_line_color, original.as_rgba8()) {
        (Some(c), _) => {
            buf.par_chunks_mut(row_len).zip(mask_raw.par_chunks(ow as usize)).for_each(|(row, mrow)| {
                for (px, &a) in row.chunks_exact_mut(4).zip(mrow) {
                    px.copy_from_slice(&[c[0], c[1], c[2], a]);
                }
            });
        }
        (None, Some(rgba)) => {
            buf.par_chunks_mut(row_len)
                .zip(rgba.as_raw().par_chunks(row_len))
                .zip(mask_raw.par_chunks(ow as usize))
                .for_each(|((row, srow), mrow)| {
                    for ((px, src), &a) in row.chunks_exact_mut(4).zip(srow.chunks_exact(4)).zip(mrow) {
                        px.copy_from_slice(&[src[0], src[1], src[2], a]);
                    }
                });
        }
        (None, None) => {
            buf = original.to_rgba8().into_raw();
            buf.par_chunks_mut(row_len).zip(mask_raw.par_chunks(ow as usize)).for_each(|(row, mrow)| {
                for (px, &a) in row.chunks_exact_mut(4).zip(mrow) {
                    px[3] = a;
                }
            });
        }
    }
    RgbaImage::from_raw(ow, oh, buf).expect("edge output buffer size matches dimensions")
}

/// Composite a dilated edge plane and the subject mask using a named
/// `ComposeMode` formula. Used by `LineMode::SubjectOutline` — picks how
/// the two cached masks combine into the final alpha. All modes run at
/// compose time on already-cached tensors, so switching modes is instant
/// in live preview.
///
/// `base` carries the subject silhouette in its alpha channel (output of
/// `postprocess::postprocess_from_flat`). With a `solid_line_color`, line
/// pixels are blended toward that color weighted by edge strength;
/// otherwise base RGB shows through at line pixels.
pub fn compose_edges_styled(
    mask: &image::GrayImage,
    base: &RgbaImage,
    compose: crate::types::ComposeMode,
    line_style: crate::types::LineStyle,
    solid_line_color: Option<[u8; 3]>,
) -> RgbaImage {
    use crate::types::LineStyle;
    let (ow, oh) = (base.width(), base.height());
    let mask_raw = mask.as_raw();
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    let pixel_count = (ow * oh) as usize;

    // One monomorphised row loop per style, so the colour rule is
    // inlined and the per-pixel work carries no dispatch.
    match line_style {
        // LineStyle gradients supersede `solid_line_color` — they compute
        // the target colour per pixel from position. Solid style defers
        // to the user's colour chip (or passes source RGB through if None).
        LineStyle::Solid => styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |_, _, _| solid_line_color),
        // DualScale belongs to `compose_edges_dual_styled`; here it only
        // renders the active scale with the base RGB showing through.
        LineStyle::DualScale { .. } => styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |_, _, _| None),
        LineStyle::GradientY { top, bottom } => styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |_, y, _| {
            let t = (y as u32 * 255 / oh.max(1)) as u16;
            Some(lerp_rgb(top, bottom, t))
        }),
        LineStyle::GradientX { left, right } => styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |x, _, _| {
            let t = (x as u32 * 255 / ow.max(1)) as u16;
            Some(lerp_rgb(left, right, t))
        }),
        LineStyle::RadialGradient { center, inner, outer } => {
            let cx = (center[0] as u32 * ow / 255) as i32;
            let cy = (center[1] as u32 * oh / 255) as i32;
            let far_x = cx.max(ow as i32 - cx);
            let far_y = cy.max(oh as i32 - cy);
            let max_dist_sq = (far_x * far_x + far_y * far_y).max(1) as i64;
            styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |x, y, _| {
                let dx = (x as i32 - cx) as i64;
                let dy = (y as i32 - cy) as i64;
                // i64 to survive `dist_sq * 255` past ~1830² (i32 caps
                // at 2.147 G, dist_sq * 255 hits that threshold there).
                let dist_sq = dx * dx + dy * dy;
                let t = ((dist_sq * 255) / max_dist_sq).min(255) as u16;
                Some(lerp_rgb(inner, outer, t))
            })
        }
        LineStyle::Rainbow { cycles } => {
            // Hue cycles along pixel index so the colour changes smoothly
            // across the whole image. 360 hues, so the conversion is a table.
            let lut = hue_table(255, 255);
            styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |x, y, _| {
                let i = (y * ow as usize + x) as u64;
                let hue = ((i * 360 * cycles.max(1) as u64 / pixel_count.max(1) as u64) % 360) as usize;
                Some(lut[hue])
            })
        }
        LineStyle::Chromatic { offset } => styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |x, _, mrow| {
            // RGB-split ghosting: sample the edge mask at horizontal
            // offsets for R and B so line colour drifts between channels.
            // Green stays at the centre. Output clamps at image edges.
            let o = (offset.min(64)) as i32;
            let r_x = (x as i32 - o).clamp(0, ow as i32 - 1) as usize;
            let b_x = (x as i32 + o).clamp(0, ow as i32 - 1) as usize;
            Some([mrow[r_x], mrow[x], mrow[b_x]])
        }),
        LineStyle::Noise { amount } => {
            let lut = hue_table(200, 240);
            styled_rows(&mut out, base.as_raw(), mask_raw, ow, compose, |x, y, _| {
                // Deterministic hash → per-pixel hue jitter. Cheap integer
                // mixer (Wang-like) avoids RNG setup cost per dispatch.
                let i = y * ow as usize + x;
                let mut h = (i as u32).wrapping_mul(0x9e37_79b1);
                h ^= h >> 16;
                h = h.wrapping_mul(0x7feb_352d);
                h ^= h >> 15;
                let jitter = (h & 0xFF) as i32 - 128; // -128..=127
                let strength = amount as i32;
                let shift = (jitter * strength) / 128; // -amount..=amount
                // i64 to survive the multiply past ~5.96 M pixels
                // (2700×2200) — Rainbow above already does the same.
                let hue = (((i as i64 * 360) / pixel_count.max(1) as i64) + shift as i64).rem_euclid(360) as usize;
                Some(lut[hue])
            })
        }
    }
    RgbaImage::from_raw(ow, oh, out).expect("edge output buffer size matches dimensions")
}

/// `hsv_to_rgb` for every whole hue at one saturation and value.
fn hue_table(s: u8, v: u8) -> [[u8; 3]; 360] {
    std::array::from_fn(|h| {
        let (r, g, b) = crate::postprocess::hsv_to_rgb(h as u16, s, v);
        [r, g, b]
    })
}

/// Row-parallel compose into `out`: first every pixel's RGB from `base`
/// and its alpha from `compose` (branch-free, so it vectorises), then the
/// colour `target(x, y, mask_row)` blended in by edge strength where the
/// edge is non-zero.
fn styled_rows<T>(out: &mut [u8], base: &[u8], mask: &[u8], ow: u32, compose: crate::types::ComposeMode, target: T)
where
    T: Fn(usize, usize, &[u8]) -> Option<[u8; 3]> + Sync,
{
    let ow = ow as usize;
    out.par_chunks_mut(ow * 4)
        .zip(base.par_chunks(ow * 4))
        .zip(mask.par_chunks(ow))
        .enumerate()
        .for_each(|(y, ((row, brow), mrow))| {
            for ((px, src), &edge) in row.chunks_exact_mut(4).zip(brow.chunks_exact(4)).zip(mrow) {
                px.copy_from_slice(&[src[0], src[1], src[2], compose.alpha(src[3] as i32, edge as i32)]);
            }
            for (x, &edge) in mrow.iter().enumerate() {
                if edge == 0 {
                    continue;
                }
                if let Some(t) = target(x, y, mrow) {
                    blend_rgb(&mut row[x * 4..x * 4 + 3], t, edge as u16);
                }
            }
        });
}

/// Compose two dilated edge planes from different DexiNed scales with
/// independent colours. Used by `LineStyle::DualScale` — fine details at
/// `fine_color`, structural edges at `bold_color`. Alpha merges via the
/// same ComposeMode formulas, using the max of the two edges.
pub fn compose_edges_dual_styled(
    fine_mask: &image::GrayImage,
    bold_mask: &image::GrayImage,
    base: &RgbaImage,
    compose: crate::types::ComposeMode,
    fine_color: [u8; 3],
    bold_color: [u8; 3],
) -> RgbaImage {
    let (ow, oh) = (base.width(), base.height());
    let ow_us = ow as usize;
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    out.par_chunks_mut(ow_us * 4)
        .zip(base.as_raw().par_chunks(ow_us * 4))
        .zip(fine_mask.as_raw().par_chunks(ow_us))
        .zip(bold_mask.as_raw().par_chunks(ow_us))
        .for_each(|(((row, brow), frow), bold_row)| {
            for (((px, src), &fine), &bold) in row.chunks_exact_mut(4).zip(brow.chunks_exact(4)).zip(frow).zip(bold_row) {
                px.copy_from_slice(&[src[0], src[1], src[2], compose.alpha(src[3] as i32, fine.max(bold) as i32)]);
            }
            for (x, (&fine, &bold)) in frow.iter().zip(bold_row).enumerate() {
                if fine > 0 {
                    blend_rgb(&mut row[x * 4..x * 4 + 3], fine_color, fine as u16);
                }
                if bold > 0 {
                    blend_rgb(&mut row[x * 4..x * 4 + 3], bold_color, bold as u16);
                }
            }
        });
    RgbaImage::from_raw(ow, oh, out).expect("edge output buffer size matches dimensions")
}

/// Blend `rgb` toward `target` using `weight` (0..=255 as a /255 fraction).
/// Used where an edge pixel is painted with the user's `solid_line_color`.
#[inline]
fn blend_rgb(rgb: &mut [u8], target: [u8; 3], weight: u16) {
    if weight == 255 {
        rgb.copy_from_slice(&target);
        return;
    }
    let inv = 255 - weight;
    rgb[0] = ((rgb[0] as u16 * inv + target[0] as u16 * weight) / 255) as u8;
    rgb[1] = ((rgb[1] as u16 * inv + target[1] as u16 * weight) / 255) as u8;
    rgb[2] = ((rgb[2] as u16 * inv + target[2] as u16 * weight) / 255) as u8;
}

/// Lerp two RGB triples: `a` at t=0, `b` at t=255.
#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: u16) -> [u8; 3] {
    let inv = 255 - t;
    [
        ((a[0] as u16 * inv + b[0] as u16 * t) / 255) as u8,
        ((a[1] as u16 * inv + b[1] as u16 * t) / 255) as u8,
        ((a[2] as u16 * inv + b[2] as u16 * t) / 255) as u8,
    ]
}

/// Compose subject outline: builds the edge mask(s) from a DexiNed result
/// and calls `compose_edges_styled` (or `compose_edges_dual_styled` for
/// `LineStyle::DualScale`) onto the supplied masked subject RGBA. Used by
/// the subprocess worker and the animation-sweep path so they share the
/// same line-styling logic.
pub fn compose_subject_outline(
    edge_res: &EdgeInferenceResult,
    masked_rgba: &RgbaImage,
    edge: &crate::EdgeSettings,
) -> RgbaImage {
    use crate::{EdgeScale, LineStyle};
    // DualScale layers Fine + Bold by definition (matches the chip
    // tooltip + types.rs:336 doc-comment). The UI greys out the scale
    // chip in this mode but presets / CLI flags can still arrive with
    // `edge_scale = Bold`; without this guard, primary and secondary
    // would both come from the Bold tensor and the second blend would
    // overwrite the first — the user sees single-tone Bold output
    // labelled "Dual scale".
    let primary_scale = if matches!(edge.line_style, LineStyle::DualScale { .. }) {
        EdgeScale::Fine
    } else {
        edge.edge_scale
    };
    let active = &edge_res.tensors[primary_scale as usize];
    let primary_mask = edge_plane(
        active, edge_res.height, edge_res.width,
        masked_rgba.width(), masked_rgba.height(),
        edge.line_strength, edge.edge_thickness,
    );
    if let LineStyle::DualScale { fine_color, bold_color } = edge.line_style {
        let bold = &edge_res.tensors[EdgeScale::Bold as usize];
        let bold_mask = edge_plane(
            bold, edge_res.height, edge_res.width,
            masked_rgba.width(), masked_rgba.height(),
            edge.line_strength, edge.edge_thickness,
        );
        // Fine and Bold must come from distinct tensor slots — equal raw
        // pointers mean the caller collapsed to single-scale output.
        debug_assert_ne!(
            active.as_ptr(), bold.as_ptr(),
            "DualScale collapsed to single-scale: primary and bold tensors are the same slice",
        );
        compose_edges_dual_styled(
            &primary_mask, &bold_mask, masked_rgba,
            edge.compose_mode,
            fine_color, bold_color,
        )
    } else {
        compose_edges_styled(
            &primary_mask, masked_rgba,
            edge.compose_mode,
            edge.line_style,
            edge.solid_line_color,
        )
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
    let mask = edge_plane(
        edge_tensor,
        tensor_h,
        tensor_w,
        original.width(),
        original.height(),
        edge.line_strength,
        edge.edge_thickness,
    );
    compose_edges(&mask, original, edge.solid_line_color)
}

/// Pre-process an image according to the user's `InputTransform`. The
/// identity (`None`) case returns `Cow::Borrowed(img)` and skips the
/// `to_rgba8()` clone (~32 MB at 4K). Transformed arms must allocate
/// — `to_rgba8()` already takes a clone-only fast path when the input
/// is `ImageRgba8`, so there's no further alloc to skip without taking
/// ownership of the input.
pub fn apply_input_transform<'a>(
    img: &'a DynamicImage,
    transform: crate::types::InputTransform,
) -> std::borrow::Cow<'a, DynamicImage> {
    use crate::types::InputTransform;
    use std::borrow::Cow;
    if matches!(transform, InputTransform::None) {
        return Cow::Borrowed(img);
    }
    let mut rgba = img.to_rgba8();
    match transform {
        InputTransform::None => unreachable!(),
        InputTransform::Grayscale => {
            for p in rgba.pixels_mut() {
                let y = ((p.0[0] as u32 * 2126 + p.0[1] as u32 * 7152 + p.0[2] as u32 * 722) / 10000) as u8;
                p.0[0] = y; p.0[1] = y; p.0[2] = y;
            }
        }
        InputTransform::ContrastBoost { percent } => {
            let factor = percent.clamp(50, 300) as i32;
            for p in rgba.pixels_mut() {
                for i in 0..3 {
                    // Expand around 128: new = 128 + (v - 128) * factor/100
                    let v = p.0[i] as i32;
                    let shifted = 128 + ((v - 128) * factor / 100);
                    p.0[i] = shifted.clamp(0, 255) as u8;
                }
            }
        }
        InputTransform::Posterize { levels } => {
            let n = levels.max(2) as u16 - 1;
            for p in rgba.pixels_mut() {
                for i in 0..3 {
                    let v = p.0[i] as u16;
                    p.0[i] = ((v * n / 255) * 255 / n) as u8;
                }
            }
        }
    }
    Cow::Owned(DynamicImage::ImageRgba8(rgba))
}

/// Preprocess an image for DexiNed: resize, BGR float32, subtract mean.
/// Flatten RGBA onto white: transparent pixels become white so edge detection
/// doesn't see ghost content behind removed backgrounds.
///
/// Operates in-place on the `to_rgba8` clone — saves one full-image
/// allocation vs. allocating a separate output buffer (~50 MB at 4 K).
/// Rows are independent → parallel via `par_chunks_mut`. Equivalent
/// to the previous `result = src * alpha + 255 * (1-alpha)` formula.
fn flatten_on_white(img: &DynamicImage) -> DynamicImage {
    use rayon::prelude::*;
    let mut rgba = img.to_rgba8();
    let row_stride = (rgba.width() * 4) as usize;
    rgba.as_mut().par_chunks_mut(row_stride).for_each(|row| {
        let n = row.len() / 4;
        for i in 0..n {
            let p = i * 4;
            let a = row[p + 3] as f32 / 255.0;
            if a < 1.0 {
                let inv_a = 1.0 - a;
                row[p]     = (row[p]     as f32 * a + 255.0 * inv_a) as u8;
                row[p + 1] = (row[p + 1] as f32 * a + 255.0 * inv_a) as u8;
                row[p + 2] = (row[p + 2] as f32 * a + 255.0 * inv_a) as u8;
            }
            row[p + 3] = 255;
        }
    });
    DynamicImage::ImageRgba8(rgba)
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

    // Skip the zero-fill; the loop writes every element of all three planes.
    let mut out: Array4<MaybeUninit<f32>> = Array4::uninit((1, 3, h, w));
    // DexiNed expects BGR with mean subtraction (no /255)
    for c in 0..3 {
        let bgr_c = 2 - c; // RGB→BGR: channel 0(R)→2(B), 1(G)→1(G), 2(B)→0(R)
        let mut plane = out.slice_mut(ndarray::s![0, bgr_c, .., ..]);
        // invariant: slice of a freshly-allocated Array4 is contiguous.
        let plane_slice = plane.as_slice_mut().unwrap();
        for i in 0..h * w {
            plane_slice[i].write(raw[i * 3 + c] as f32 - MEAN_BGR[bgr_c]);
        }
    }
    // Safety: every element of all three planes was written above.
    unsafe { out.assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The compositions as shipped before the row-parallel rewrite: the
    // bit-exact oracle for every style and compose mode. They dilate the
    // mask themselves; production receives the dilated plane.
    fn reference_dilated(mask: &image::GrayImage, thickness: u32) -> Vec<u8> {
        let mut out = mask.clone();
        if thickness > 0 {
            crate::morphology::shift_mask(&mut out, -(thickness as f32));
        }
        out.into_raw()
    }
    fn reference_compose_edges(
        mask: &image::GrayImage,
        original: &DynamicImage,
        solid_line_color: Option<[u8; 3]>,
        edge_thickness: u32,
    ) -> RgbaImage {
        let (ow, oh) = (original.width(), original.height());
        let mask_raw = reference_dilated(mask, edge_thickness);
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

    fn reference_compose_edges_styled(
        mask: &image::GrayImage,
        base: &RgbaImage,
        compose: crate::types::ComposeMode,
        line_style: crate::types::LineStyle,
        solid_line_color: Option<[u8; 3]>,
        edge_thickness: u32,
    ) -> RgbaImage {
        use crate::types::{ComposeMode, LineStyle};
        let (ow, oh) = (base.width(), base.height());
        let mask_raw = reference_dilated(mask, edge_thickness);
        let mut rgba = base.clone();
        let out_raw = rgba.as_mut();
        let pixel_count = (ow * oh) as usize;

        // Single alpha-formula dispatch keeps the mode match out of the per-pixel
        // loop. LLVM will typically hoist it anyway, but spelling it out makes
        // the cost predictable regardless of optimization settings.
        let alpha_fn: fn(i32, i32) -> u8 = match compose {
            ComposeMode::LinesOnly => |s, e| (s * e / 255) as u8,
            ComposeMode::SubjectFilled => |s, e| s.max(e) as u8,
            ComposeMode::Engraving => |s, e| (s - e).max(0) as u8,
            // 0.3 * subject + 0.8 * edge, clamped. Sums to > 1.0 on purpose —
            // saturates to fully opaque where subject AND edge both contribute.
            ComposeMode::Ghost => |s, e| ((s * 77 + e * 204) / 255).clamp(0, 255) as u8,
            ComposeMode::InverseMask => |s, e| ((255 - s) * e / 255) as u8,
        };

        // LineStyle gradients supersede `solid_line_color` — they compute the
        // target colour per pixel from position. Solid style defers to the
        // user's colour chip (or passes source RGB through if None).
        let solid_tint = match line_style {
            LineStyle::Solid => solid_line_color,
            _ => None,
        };

        // Precompute geometry for radial gradient so the hot loop doesn't
        // redo the centre conversion per pixel.
        let (rg_cx, rg_cy, rg_max_dist_sq) = if let LineStyle::RadialGradient { center, .. } = line_style {
            let cx = (center[0] as u32 * ow / 255) as i32;
            let cy = (center[1] as u32 * oh / 255) as i32;
            let far_x = cx.max(ow as i32 - cx);
            let far_y = cy.max(oh as i32 - cy);
            (cx, cy, (far_x * far_x + far_y * far_y).max(1))
        } else {
            (0, 0, 1)
        };

        for i in 0..pixel_count {
            let subject = out_raw[i * 4 + 3] as i32;
            let edge = mask_raw[i] as i32;
            out_raw[i * 4 + 3] = alpha_fn(subject, edge);
            if edge == 0 { continue; }

            let gradient_target: Option<[u8; 3]> = match line_style {
                LineStyle::Solid => solid_tint,
                LineStyle::GradientY { top, bottom } => {
                    let y = (i as u32 / ow) as u16;
                    let t = (y as u32 * 255 / oh.max(1)) as u16;
                    Some(lerp_rgb(top, bottom, t))
                }
                LineStyle::GradientX { left, right } => {
                    let x = (i as u32 % ow) as u16;
                    let t = (x as u32 * 255 / ow.max(1)) as u16;
                    Some(lerp_rgb(left, right, t))
                }
                LineStyle::RadialGradient { inner, outer, .. } => {
                    let x = (i as u32 % ow) as i32;
                    let y = (i as u32 / ow) as i32;
                    let dx = (x - rg_cx) as i64;
                    let dy = (y - rg_cy) as i64;
                    // i64 to survive `dist_sq * 255` past ~1830² (i32 caps
                    // at 2.147 G, dist_sq * 255 hits that threshold there).
                    let dist_sq = dx * dx + dy * dy;
                    let t = ((dist_sq * 255) / (rg_max_dist_sq as i64)).min(255) as u16;
                    Some(lerp_rgb(inner, outer, t))
                }
                LineStyle::Rainbow { cycles } => {
                    // Hue cycles along pixel index so the colour changes smoothly
                    // across the whole image.
                    let hue = ((i as u64 * 360 * cycles.max(1) as u64 / pixel_count.max(1) as u64) % 360) as u16;
                    let (r, g, b) = crate::postprocess::hsv_to_rgb(hue, 255, 255);
                    Some([r, g, b])
                }
                LineStyle::Chromatic { offset } => {
                    // RGB-split ghosting: sample the edge mask at horizontal
                    // offsets for R and B so line colour drifts between channels.
                    // Green stays at the centre. Output clamps at image edges.
                    let x = (i as u32 % ow) as i32;
                    let y = (i as u32 / ow) as i32;
                    let o = (offset.min(64)) as i32;
                    let r_x = (x - o).clamp(0, ow as i32 - 1) as u32;
                    let b_x = (x + o).clamp(0, ow as i32 - 1) as u32;
                    let r_idx = (y as u32 * ow + r_x) as usize;
                    let b_idx = (y as u32 * ow + b_x) as usize;
                    let rv = mask_raw[r_idx];
                    let gv = edge as u8;
                    let bv = mask_raw[b_idx];
                    Some([rv, gv, bv])
                }
                LineStyle::Noise { amount } => {
                    // Deterministic hash → per-pixel hue jitter. Cheap integer
                    // mixer (Wang-like) avoids RNG setup cost per dispatch.
                    let mut h = (i as u32).wrapping_mul(0x9e37_79b1);
                    h ^= h >> 16;
                    h = h.wrapping_mul(0x7feb_352d);
                    h ^= h >> 15;
                    let jitter = (h & 0xFF) as i32 - 128; // -128..=127
                    let strength = amount as i32;
                    let shift = (jitter * strength) / 128; // -amount..=amount
                    // i64 to survive the multiply past ~5.96 M pixels
                    // (2700×2200) — Rainbow above already does the same.
                    let hue = (((i as i64 * 360) / pixel_count.max(1) as i64) + shift as i64)
                        .rem_euclid(360) as u16;
                    let (r, g, b) = crate::postprocess::hsv_to_rgb(hue, 200, 240);
                    Some([r, g, b])
                }
                // DualScale is handled by `compose_edges_dual_styled` — the
                // single-mask path here renders only the active scale. Callers
                // that select DualScale must dispatch to the dual function; the
                // fallthrough here prevents compile errors and degrades to the
                // user's solid_line_color for correctness.
                LineStyle::DualScale { .. } => solid_tint,
            };
            if let Some(target) = gradient_target {
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], target, edge as u16);
            }
        }
        rgba
    }


    fn reference_compose_edges_dual_styled(
        fine_mask: &image::GrayImage,
        bold_mask: &image::GrayImage,
        base: &RgbaImage,
        compose: crate::types::ComposeMode,
        fine_color: [u8; 3],
        bold_color: [u8; 3],
        edge_thickness: u32,
    ) -> RgbaImage {
        use crate::types::ComposeMode;
        let (ow, oh) = (base.width(), base.height());
        let fine_raw = reference_dilated(fine_mask, edge_thickness);
        let bold_raw = reference_dilated(bold_mask, edge_thickness);
        let mut rgba = base.clone();
        let out_raw = rgba.as_mut();
        let pixel_count = (ow * oh) as usize;

        let alpha_fn: fn(i32, i32) -> u8 = match compose {
            ComposeMode::LinesOnly => |s, e| (s * e / 255) as u8,
            ComposeMode::SubjectFilled => |s, e| s.max(e) as u8,
            ComposeMode::Engraving => |s, e| (s - e).max(0) as u8,
            ComposeMode::Ghost => |s, e| ((s * 77 + e * 204) / 255).clamp(0, 255) as u8,
            ComposeMode::InverseMask => |s, e| ((255 - s) * e / 255) as u8,
        };

        for i in 0..pixel_count {
            let subject = out_raw[i * 4 + 3] as i32;
            let fine = fine_raw[i] as i32;
            let bold = bold_raw[i] as i32;
            let edge = fine.max(bold);
            out_raw[i * 4 + 3] = alpha_fn(subject, edge);
            if fine > 0 {
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], fine_color, fine as u16);
            }
            if bold > 0 {
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], bold_color, bold as u16);
            }
        }
        rgba
    }


    fn noisy_gray(w: u32, h: u32, seed: u64) -> image::GrayImage {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        image::GrayImage::from_fn(w, h, |_, _| image::Luma([if rng.random::<u8>() < 40 { rng.random() } else { 0 }]))
    }

    fn noisy_rgba(w: u32, h: u32, seed: u64) -> RgbaImage {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        RgbaImage::from_fn(w, h, |_, _| image::Rgba(rng.random()))
    }

    #[test]
    fn compositions_match_the_reference_for_every_style_and_mode() {
        use crate::types::{ComposeMode, LineStyle};
        let (w, h) = (97, 61);
        let mask = noisy_gray(w, h, 1);
        let bold = noisy_gray(w, h, 2);
        let base = noisy_rgba(w, h, 3);
        let original = DynamicImage::ImageRgba8(base.clone());
        let styles = [
            LineStyle::Solid,
            LineStyle::GradientY { top: [255, 0, 0], bottom: [0, 0, 255] },
            LineStyle::GradientX { left: [0, 255, 0], right: [255, 0, 255] },
            LineStyle::RadialGradient { center: [40, 200], inner: [255, 255, 0], outer: [0, 255, 255] },
            LineStyle::Rainbow { cycles: 3 },
            LineStyle::Chromatic { offset: 5 },
            LineStyle::Noise { amount: 90 },
            LineStyle::DualScale { fine_color: [1, 2, 3], bold_color: [4, 5, 6] },
        ];
        let dilated = |m: &image::GrayImage, t: u32| image::GrayImage::from_raw(w, h, reference_dilated(m, t)).unwrap();
        for thickness in [0, 2] {
            let (dm, db) = (dilated(&mask, thickness), dilated(&bold, thickness));
            for color in [None, Some([10u8, 200, 30])] {
                assert!(compose_edges(&dm, &original, color) == reference_compose_edges(&mask, &original, color, thickness), "plain t={thickness} {color:?}");
            }
            for &compose in ComposeMode::ALL {
                for &style in &styles {
                    let fast = compose_edges_styled(&dm, &base, compose, style, Some([10, 200, 30]));
                    let slow = reference_compose_edges_styled(&mask, &base, compose, style, Some([10, 200, 30]), thickness);
                    assert!(fast == slow, "{compose:?} {style:?} t={thickness}");
                }
                let fast = compose_edges_dual_styled(&dm, &db, &base, compose, [255, 0, 0], [0, 0, 255]);
                let slow = reference_compose_edges_dual_styled(&mask, &bold, &base, compose, [255, 0, 0], [0, 0, 255], thickness);
                assert!(fast == slow, "dual {compose:?} t={thickness}");
            }
        }
    }
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
        for slot in tensor.iter_mut().take(h * w / 2) {
            *slot = 10.0; // sigmoid → ~1 → edge
        }
        let original = solid_rgb(64, 48);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0, edge_scale: crate::EdgeScale::Fused, compose_mode: crate::ComposeMode::default(), line_style: crate::LineStyle::default(), input_transform: crate::InputTransform::default() };
        let out = finalize_edges(&tensor, h as u32, w as u32, &original, &edge);
        assert_eq!(out.width(), 64);
        assert_eq!(out.height(), 48);
        // With high-logit side → opaque red, zero-logit → transparent.
        let strong_red = out.get_pixel(0, 0);
        assert_eq!([strong_red[0], strong_red[1], strong_red[2]], [255, 0, 0]);
    }

    /// `LineStyle::DualScale` must always layer Fine + Bold tensors —
    /// the chip docstring + tooltip say so. The UI greys out the scale
    /// chip in DualScale mode, but presets / CLI flags can still
    /// arrive with `edge_scale = Bold`. Without the guard in
    /// `compose_subject_outline`, primary and secondary masks would
    /// both come from the Bold tensor and the second blend would
    /// overwrite the first — silently degenerating to single-tone
    /// Bold. Build distinguishable per-scale tensors and assert Fine
    /// drives the primary mask regardless of `edge.edge_scale`.
    #[test]
    fn dual_scale_uses_fine_for_primary_even_when_edge_scale_is_bold() {
        let h = DEXINED_H as usize;
        let w = DEXINED_W as usize;
        let fine_tensor: Vec<f32> = vec![10.0; h * w];   // sigmoid → ~1
        let bold_tensor: Vec<f32> = vec![-10.0; h * w];  // sigmoid → ~0
        let edge_res = EdgeInferenceResult {
            tensors: [
                fine_tensor,
                vec![0.0; h * w],
                bold_tensor,
                vec![0.0; h * w],
            ],
            height: DEXINED_H,
            width: DEXINED_W,
        };
        let masked = RgbaImage::from_pixel(64, 48, image::Rgba([10, 10, 10, 255]));
        let edge = crate::EdgeSettings {
            line_strength: 0.5,
            solid_line_color: None,
            edge_thickness: 0,
            // The bypass: user-supplied edge_scale = Bold while the
            // pipeline is in DualScale mode. Pre-fix, primary would
            // also be Bold (≈0) and the test pixel would stay [10,10,10].
            edge_scale: crate::EdgeScale::Bold,
            compose_mode: crate::ComposeMode::SubjectFilled,
            line_style: crate::LineStyle::DualScale {
                fine_color: [200, 80, 40],
                bold_color: [0, 0, 200],
            },
            input_transform: crate::InputTransform::default(),
        };
        let out = compose_subject_outline(&edge_res, &masked, &edge);
        let p = out.get_pixel(32, 24);
        // Fine tensor saturates → primary mask ≈ 255 → blend pulls all
        // the way to fine_color. Pre-fix this pixel would be ≈[10,10,10]
        // (no Bold edges, so neither blend fires).
        assert_eq!(
            [p[0], p[1], p[2]], [200, 80, 40],
            "DualScale primary must come from Fine tensor; got {:?}", p,
        );
    }

    /// `RadialGradient`'s `dist_sq * 255` blew past `i32::MAX` for any
    /// image past ~1830² with the centre near a corner — the wrapping
    /// multiply produced a wildly wrong `t` and the colour ramp
    /// inverted in the second half of the image. Drive a 4096×1 strip
    /// with the centre at [0, 0] and assert the gradient is monotonic
    /// across pixels well past the overflow threshold. Pre-fix the i32
    /// multiply wraps to a negative value at pixel ~2900 and produces
    /// out-of-order RGB values; in debug builds the `255 - t` step in
    /// `lerp_rgb` panics outright when `t` wraps to a giant u16.
    #[test]
    fn radial_gradient_monotonic_past_overflow_threshold() {
        let w = 4096_u32;
        let h = 1_u32;
        let mask = image::GrayImage::from_pixel(w, h, image::Luma([255]));
        let base = RgbaImage::from_pixel(w, h, image::Rgba([10, 10, 10, 200]));
        let out = compose_edges_styled(
            &mask, &base,
            crate::ComposeMode::SubjectFilled,
            crate::LineStyle::RadialGradient {
                center: [0, 0],
                inner: [0, 0, 0],
                outer: [255, 255, 255],
            },
            None,
        );
        // `dist_sq * 255` first overflows i32 at x ≈ 2900 (255·x² > 2^31).
        // Sample on either side: monotonic ramp toward outer means
        // mid > early and far ≥ mid - small slack for u8 quantisation.
        let early = out.get_pixel(2000, 0)[0];
        let mid = out.get_pixel(3500, 0)[0];   // past overflow threshold
        let far = out.get_pixel(w - 1, 0)[0];  // near image edge
        assert!(
            early < mid && mid <= far,
            "radial gradient must stay monotonic across the i32 overflow \
             threshold — got early={early}, mid={mid}, far={far}",
        );
        assert!(
            far >= 250,
            "far-corner pixel should be near `outer` ([255,255,255]); got {far}",
        );
    }

    /// `LineStyle::Noise`'s `i as i32 * 360` overflowed i32 above
    /// ~5.96 M pixels — past that threshold the second half of the
    /// image got wrapped numerators and the noise hue became
    /// uncorrelated with pixel position. In debug builds the overflow
    /// check panics; in release the output is garbled. Drive a strip
    /// big enough to cross the overflow threshold and assert the call
    /// completes (post-fix uses i64).
    #[test]
    fn noise_line_style_does_not_overflow_above_6m_pixels() {
        let w = 4096_u32;
        let h = 1500_u32; // 6.144 M pixels — past the i32 threshold
        let mask = image::GrayImage::from_pixel(w, h, image::Luma([255]));
        let base = RgbaImage::from_pixel(w, h, image::Rgba([10, 10, 10, 200]));
        // Pre-fix this would panic in debug from the i32 overflow at
        // i = ~5.96 M; post-fix it produces a valid output across the
        // whole image. The contract worth pinning is "function doesn't
        // overflow at the project's image-size cap".
        let out = compose_edges_styled(
            &mask, &base,
            crate::ComposeMode::SubjectFilled,
            crate::LineStyle::Noise { amount: 80 },
            None,
        );
        assert_eq!(out.dimensions(), (w, h));
    }

    /// `apply_input_transform`'s `None` arm must return a borrowed
    /// `Cow` so the ~32 MB `to_rgba8()` clone is skipped on the
    /// identity case. CLAUDE.md `## Test expectations` requires unit
    /// tests for new pure functions in `edge.rs` — this contract was
    /// implicit via the `if matches!(...) { return Cow::Borrowed(...) }`
    /// fast path with no test pinning it.
    #[test]
    fn apply_input_transform_none_returns_borrowed() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([10, 20, 30])));
        let out = apply_input_transform(&img, crate::types::InputTransform::None);
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "InputTransform::None must skip the to_rgba8 alloc",
        );
    }

    /// Grayscale uses the BT.709 luma weights `(2126, 7152, 722) / 10000`.
    /// Pure red → 54, pure green → 71, pure blue → 7. Pin those exact
    /// outputs so a refactor of the weights gets caught.
    #[test]
    fn apply_input_transform_grayscale_collapses_channels() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([255, 0, 0])));
        let out = apply_input_transform(&img, crate::types::InputTransform::Grayscale);
        let rgba = out.to_rgba8();
        let p = rgba.get_pixel(0, 0);
        // (255 * 2126 + 0 + 0) / 10000 = 54
        assert_eq!([p[0], p[1], p[2]], [54, 54, 54], "pure-red luma");
    }

    /// `ContrastBoost` expands around 128: pivot stays put, extremes
    /// saturate. percent=200 doubles the deviation from 128.
    #[test]
    fn apply_input_transform_contrast_boost_around_pivot() {
        let boost = crate::types::InputTransform::ContrastBoost { percent: 200 };
        for (input, expected) in [(128_u8, 128_u8), (0, 0), (255, 255)] {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([input, input, input])));
            let out = apply_input_transform(&img, boost);
            let p = out.to_rgba8().get_pixel(0, 0).0;
            assert_eq!(
                p[0], expected,
                "ContrastBoost(200%) at v={input}: expected {expected}, got {}",
                p[0],
            );
        }
    }

    /// `Posterize { levels: 4 }` quantises each channel to one of
    /// `{0, 85, 170, 255}`. Pin the bucket boundaries so the
    /// `(v * n / 255) * 255 / n` formula doesn't drift on a refactor.
    #[test]
    fn apply_input_transform_posterize_steps() {
        let post = crate::types::InputTransform::Posterize { levels: 4 };
        for (input, expected) in [
            (0_u8, 0_u8),
            (84, 0),
            (85, 85),
            (170, 170),
            (255, 255),
        ] {
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([input, input, input])));
            let out = apply_input_transform(&img, post);
            let p = out.to_rgba8().get_pixel(0, 0).0;
            assert_eq!(
                p[0], expected,
                "Posterize(4) at v={input}: expected {expected}, got {}",
                p[0],
            );
        }
    }

    /// At `line_strength = 1.0` the threshold curve yields 0.01, so
    /// `threshold - 0.1 = -0.09`. Without the floor clamp, a background
    /// pixel with `prob = 0.0` computes `edge = (0.0 + 0.09) / 0.2 = 0.45`
    /// → smoothstep ≈ 0.5 → alpha ≈ 127: a 50%-gray haze across the whole
    /// image. With the fix, `lo = 0.0` and `prob = 0.0` → `edge = 0.0`
    /// → alpha = 0. Pin both ends of the range so no regression sneaks back.
    #[test]
    fn tensor_to_edge_mask_no_haze_at_max_line_strength() {
        // Use a tiny 2×2 tensor: 4 representative probability logits.
        // sigmoid(logit):
        //   -100  → prob ≈ 0.0  (pure background)
        //    -2.9 → prob ≈ 0.05 (very faint, below threshold even at max strength)
        //     0.0 → prob = 0.5  (mid-strength)
        //     3.0 → prob ≈ 0.95 (strong edge — must remain opaque)
        let logits = vec![-100.0_f32, -2.9, 0.0, 3.0];
        let mask = tensor_to_edge_mask(&logits, 2, 2, 2, 2, 1.0);
        let raw = mask.as_raw();
        assert!(
            raw[0] <= 5,
            "background pixel (prob≈0) must produce alpha≤5 at line_strength=1.0; got {}",
            raw[0],
        );
        assert!(
            raw[3] >= 250,
            "strong-edge pixel (prob≈0.95) must produce alpha≥250 at line_strength=1.0; got {}",
            raw[3],
        );
    }

    #[test]
    fn finalize_edges_preserves_original_rgb_when_no_line_color() {
        let w = DEXINED_W as usize;
        let h = DEXINED_H as usize;
        let tensor = vec![10.0_f32; h * w]; // all edges
        let original = solid_rgb(32, 32);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: None, edge_thickness: 0, edge_scale: crate::EdgeScale::Fused, compose_mode: crate::ComposeMode::default(), line_style: crate::LineStyle::default(), input_transform: crate::InputTransform::default() };
        let out = finalize_edges(&tensor, h as u32, w as u32, &original, &edge);
        // Original color preserved
        assert_eq!(out.get_pixel(0, 0)[0], 120);
        assert_eq!(out.get_pixel(0, 0)[1], 120);
        assert_eq!(out.get_pixel(0, 0)[2], 120);
    }
}
