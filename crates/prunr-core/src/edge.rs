use std::mem::MaybeUninit;
use std::sync::Mutex;

use image::{DynamicImage, RgbaImage};
use ndarray::Array4;
use rayon::prelude::*;
use ort::{inputs, session::Session, value::Tensor};

use crate::types::{CoreError, EdgeScale};

/// The size TEED reads a photo at, whatever its aspect: Bold and Fused
/// come from this run, Fine from one at twice the size.
const BASE_H: u32 = 480;
const BASE_W: u32 = 640;
// BGR mean of TEED's training set (BIPED).
const MEAN_BGR: [f32; 3] = [104.007, 116.669, 122.679];

const BOLD_OUTPUT: &str = "out_3";
const FUSED_OUTPUT: &str = "block_cat";

/// Number of `EdgeScale` variants, one plane each.
pub const EDGE_SCALE_COUNT: usize = EdgeScale::ALL.len();

/// The detector's planes as edge probabilities in `0..=1`, indexed by
/// `EdgeScale as usize`. `height` × `width` is the base run's size; a
/// scale's own plane size is `dims(scale)`. A scale nobody asked for
/// holds an empty plane.
pub struct EdgeInferenceResult {
    pub tensors: [Vec<f32>; EDGE_SCALE_COUNT],
    pub height: u32,
    pub width: u32,
}

impl EdgeInferenceResult {
    pub fn dims(&self, scale: EdgeScale) -> (u32, u32) {
        edge_scale_dims(scale, self.height, self.width)
    }
}

/// `scale`'s plane size, height first, for a base run of `base_h` × `base_w`.
pub fn edge_scale_dims(scale: EdgeScale, base_h: u32, base_w: u32) -> (u32, u32) {
    let k = if scale == EdgeScale::Fine { 2 } else { 1 };
    (base_h * k, base_w * k)
}

/// Opaque wrapper around the TEED ORT session.
/// Thread-safe via internal Mutex (same pattern as OrtEngine).
pub struct EdgeEngine {
    session: Mutex<Session>,
}

/// Compile-time lock on `EdgeScale` discriminants: callers index
/// `EdgeInferenceResult::tensors` by `scale as usize`.
const _: () = {
    let mut i = 0;
    while i < EDGE_SCALE_COUNT {
        assert!(EdgeScale::ALL[i] as usize == i);
        i += 1;
    }
};

impl EdgeEngine {
    /// Create a new TEED edge detection engine.
    pub fn new() -> Result<Self, CoreError> {
        let edge_bytes = prunr_models::teed_bytes();
        let session = crate::ort_runtime::session_builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| CoreError::Inference(format!("Edge set opt level failed: {e}")))?
            .with_intra_threads(num_cpus::get().max(1))
            .map_err(|e| CoreError::Inference(format!("Edge set threads failed: {e}")))?
            .commit_from_memory(edge_bytes)
            .map_err(|e| CoreError::Inference(format!("Edge model load failed: {e}")))?;

        // A re-export that renames the outputs would otherwise surface as
        // a wrong-scale result instead of an init error.
        let names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        tracing::info!(?names, "TEED output layout");
        if let Some(missing) = [BOLD_OUTPUT, FUSED_OUTPUT].into_iter().find(|n| !names.contains(n)) {
            return Err(CoreError::Inference(format!(
                "TEED export layout changed: no '{missing}' output in {names:?}",
            )));
        }
        Ok(Self { session: Mutex::new(session) })
    }

    /// One-shot: the lines over the whole photo. Runs only the scales
    /// `edge` draws.
    pub fn detect(&self, original: &DynamicImage, edge: &crate::EdgeSettings) -> Result<RgbaImage, CoreError> {
        Ok(compose_whole_photo(&self.infer_scales(original, edge.drawn_scales())?, original, edge, None))
    }

    /// Every scale, for a caller that keeps them so a scale switch needs
    /// no new run.
    pub fn infer_all_tensors(&self, original: &DynamicImage) -> Result<EdgeInferenceResult, CoreError> {
        self.infer_scales(original, &EdgeScale::ALL)
    }

    /// The `scales` asked for; the rest stay empty. Bold and Fused share
    /// one run at 640×480, Fine is its own run at 1280×960.
    ///
    /// Peak working set (f32, ORT activations measured separately):
    ///
    /// | run       | input  | outputs (4 planes) | kept planes |
    /// |-----------|--------|--------------------|-------------|
    /// | 640×480   | 3.7 MB | 4.9 MB             | 2.5 MB      |
    /// | 1280×960  | 15 MB  | 20 MB              | 4.9 MB      |
    ///
    /// Both runs' inputs are built before the session lock is taken, so
    /// with several images in flight one image's resize never holds up
    /// another's run. The runs go one after the other, each run's input
    /// and outputs dropped before the next, so the peak is the 1280×960
    /// run plus the 2.5 MB already kept and the 3.7 MB base input waiting
    /// for it. While the inputs are built, an input that is not 8-bit RGB
    /// costs one full-size RGB copy (W×H×3 bytes), flattened onto white
    /// when it has alpha. A whole Sketch on a 20 MP photo peaks at about
    /// 0.9 GB resident in the CLI (DexiNed: 1.4 GB).
    pub fn infer_scales(&self, original: &DynamicImage, scales: &[EdgeScale]) -> Result<EdgeInferenceResult, CoreError> {
        let wants = |s: EdgeScale| scales.contains(&s);
        let (fine_h, fine_w) = edge_scale_dims(EdgeScale::Fine, BASE_H, BASE_W);
        let (base_input, fine_input) = {
            let rgb = opaque_rgb(original);
            (
                (wants(EdgeScale::Bold) || wants(EdgeScale::Fused)).then(|| preprocess(&rgb, BASE_W, BASE_H)),
                wants(EdgeScale::Fine).then(|| preprocess(&rgb, fine_w, fine_h)),
            )
        };
        let mut session = self.session.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut tensors: [Vec<f32>; EDGE_SCALE_COUNT] = Default::default();
        if let Some(input) = base_input {
            let outputs = run_inference(&mut session, input)?;
            for (scale, name) in [(EdgeScale::Bold, BOLD_OUTPUT), (EdgeScale::Fused, FUSED_OUTPUT)] {
                if wants(scale) {
                    tensors[scale as usize] = extract_probabilities(&outputs, name, BASE_H, BASE_W)?;
                }
            }
        }
        if let Some(input) = fine_input {
            let outputs = run_inference(&mut session, input)?;
            tensors[EdgeScale::Fine as usize] = extract_probabilities(&outputs, FUSED_OUTPUT, fine_h, fine_w)?;
        }
        Ok(EdgeInferenceResult { tensors, height: BASE_H, width: BASE_W })
    }
}

/// Run the ONNX session once on a `preprocess`ed input.
fn run_inference(
    session: &mut Session,
    input_array: Array4<f32>,
) -> Result<ort::session::SessionOutputs<'_>, CoreError> {
    let input_name = session.inputs()[0].name().to_string();
    let input_tensor = Tensor::from_array(input_array)
        .map_err(|e| CoreError::Inference(format!("Failed to create edge tensor: {e}")))?;
    session
        .run(inputs![input_name.as_str() => &input_tensor])
        .map_err(|e| CoreError::Inference(format!("Edge detection failed: {e}")))
}

/// One output plane by name, copied out and calibrated into probabilities.
fn extract_probabilities(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    h: u32,
    w: u32,
) -> Result<Vec<f32>, CoreError> {
    let edge_map = outputs[name]
        .try_extract_array::<f32>()
        .map_err(|e| CoreError::Inference(format!("Failed to extract edge output '{name}': {e}")))?;
    let slice = edge_map.as_slice()
        .ok_or_else(|| CoreError::Inference("Edge output tensor is not contiguous".to_string()))?;
    if slice.len() != (h * w) as usize {
        return Err(CoreError::Inference(format!(
            "Edge output '{name}' has {} values, expected {h}×{w}", slice.len(),
        )));
    }
    let mut plane = slice.to_vec();
    calibrate_plane(&mut plane, h as usize, w as usize);
    Ok(plane)
}

/// Turn one plane of TEED logits into edge probabilities, in place.
///
/// TEED's sigmoid does not sit at 0 on plain background (`block_cat`'s
/// floor is ≈ 0.44 on every photo), so TEED's own inference stretches
/// each plane's min..max to 0..1; this does the same. The range is
/// floored at `MIN_RANGE`: a flat image spans under 0.02, and stretched
/// to 0..1 its noise would draw lines everywhere. A photo's planes span
/// over 0.5, where the floor never binds.
///
/// TEED also marks a small patch in the corners of any input, a flat
/// one too; those pixels are set to the floor before the stretch, so
/// they neither draw a dot nor set the plane's max.
pub(crate) fn calibrate_plane(plane: &mut [f32], h: usize, w: usize) {
    const MIN_RANGE: f32 = 0.25;
    const CORNER: usize = 8;
    plane.par_iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
    let min = plane.par_iter().copied().reduce(|| f32::INFINITY, f32::min);
    if h >= 2 * CORNER && w >= 2 * CORNER {
        for y in (0..CORNER).chain(h - CORNER..h) {
            for x in (0..CORNER).chain(w - CORNER..w) {
                plane[y * w + x] = min;
            }
        }
    }
    let max = plane.par_iter().copied().reduce(|| f32::NEG_INFINITY, f32::max);
    let inv_range = 1.0 / (max - min).max(MIN_RANGE);
    plane.par_iter_mut().for_each(|v| *v = (*v - min) * inv_range);
}

/// Threshold + resize a plane of edge probabilities into a
/// full-resolution edge mask. Depends only on `line_strength`; callers
/// can cache the result and reuse it across `edge_thickness` /
/// `solid_line_color` tweaks.
pub fn tensor_to_edge_mask(
    edge_probs: &[f32],
    tensor_h: u32,
    tensor_w: u32,
    out_w: u32,
    out_h: u32,
    line_strength: f32,
) -> image::GrayImage {
    // Strength as contrast/threshold control on an exponential curve:
    // slider 0.0→threshold 0.95, slider 0.5→0.3, slider 1.0→0.01
    let s = line_strength.clamp(0.0, 1.0);
    let threshold = (1.0 - s).powi(2) * 0.95 + 0.01;
    // Clamp the transition window floor at 0: for line_strength > 0.75 the
    // threshold falls below 0.10, so `threshold - 0.1` goes negative. Without
    // the clamp, background pixels (prob ≈ 0) land in the negative-floored
    // window and produce ~50% alpha — a gray haze across the whole image.
    let lo = (threshold - 0.1_f32).max(0.0);
    let mask_buf: Vec<u8> = edge_probs[..(tensor_h * tensor_w) as usize]
        .iter()
        .map(|&prob| {
            // Remap [lo, threshold+0.1] to [0, 1] for anti-aliased edges.
            let edge = ((prob - lo) / 0.2).clamp(0.0, 1.0);
            (crate::math::smoothstep(edge) * 255.0) as u8
        })
        .collect();
    let mask = image::GrayImage::from_raw(tensor_w, tensor_h, mask_buf)
        .expect("edge mask buffer size matches dimensions");
    crate::formats::resize_gray_lanczos3(&mask, out_w, out_h)
}

/// `base` grown by a round pen `thickness` pixels wide on each side;
/// `None` at 0, where the base already is the plane a composition draws.
pub fn thickened_edges(base: &image::GrayImage, thickness: u32) -> Option<image::GrayImage> {
    (thickness > 0).then(|| crate::morphology::dilate_round(base, thickness))
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
                    blend_rgb(&mut row[x * 4..x * 4 + 3], t, line_weight(compose, edge));
                }
            }
        });
}

/// Compose two dilated edge planes from different detector scales with
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
                    blend_rgb(&mut row[x * 4..x * 4 + 3], fine_color, line_weight(compose, fine));
                }
                if bold > 0 {
                    blend_rgb(&mut row[x * 4..x * 4 + 3], bold_color, line_weight(compose, bold));
                }
            }
        });
    RgbaImage::from_raw(ow, oh, out).expect("edge output buffer size matches dimensions")
}

/// How far a line pixel takes its colour. Where the line alone sets the
/// alpha (Lines only) it is the line's colour at that alpha, as straight
/// alpha should be; over a visible subject it blends in by line strength.
#[inline]
fn line_weight(compose: crate::types::ComposeMode, edge: u8) -> u16 {
    if compose == crate::types::ComposeMode::LinesOnly { 255 } else { edge as u16 }
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

/// Compose subject outline: builds the edge mask(s) from a detector result
/// and calls `compose_edges_styled` (or `compose_edges_dual_styled` for
/// `LineStyle::DualScale`) onto the supplied masked subject RGBA. Used by
/// the subprocess worker and the animation-sweep path so they share the
/// same line-styling logic.
pub fn compose_subject_outline(
    edge_res: &EdgeInferenceResult,
    masked_rgba: &RgbaImage,
    edge: &crate::EdgeSettings,
    fade: Option<crate::depth::LineFade<'_>>,
) -> RgbaImage {
    use crate::{EdgeScale, LineStyle};
    let primary_scale = edge.primary_scale();
    let plane = |scale: EdgeScale| {
        let (th, tw) = edge_res.dims(scale);
        let mut p = edge_plane(&edge_res.tensors[scale as usize], th, tw, masked_rgba.width(), masked_rgba.height(), edge.line_strength, edge.edge_thickness);
        if let Some(fade) = fade {
            fade.apply(&mut p);
        }
        p
    };
    let primary_mask = plane(primary_scale);
    if let LineStyle::DualScale { fine_color, bold_color } = edge.line_style {
        let active = &edge_res.tensors[primary_scale as usize];
        let bold = &edge_res.tensors[EdgeScale::Bold as usize];
        let bold_mask = plane(EdgeScale::Bold);
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

/// Lines over the subject (Sketch on Subject): `trace` (the line
/// detector) reads the plain cut-out after the pre-filter, so an effect
/// never moves the lines; then `fill` colours the subject and the lines
/// are drawn over it. Returns the result and the traced tensors.
pub fn outline_subject(
    subject: RgbaImage,
    fill: crate::FillStyle,
    edge: &crate::EdgeSettings,
    fade: Option<crate::depth::LineFade<'_>>,
    trace: impl FnOnce(&DynamicImage) -> Result<EdgeInferenceResult, CoreError>,
) -> Result<(RgbaImage, EdgeInferenceResult), CoreError> {
    let plain = DynamicImage::ImageRgba8(subject);
    let traced = trace(&apply_input_transform(&plain, edge.input_transform))?;
    let mut subject = plain.into_rgba8();
    crate::postprocess::apply_fill_style(&mut subject, fill);
    Ok((compose_subject_outline(&traced, &subject, edge, fade), traced))
}

/// The lines over the whole photo (Sketch on Full): every line colour
/// style, Dual scale included, on transparent. The same compose as the
/// subject's outline with Lines only.
pub fn compose_whole_photo(
    edge_res: &EdgeInferenceResult,
    original: &DynamicImage,
    edge: &crate::EdgeSettings,
    fade: Option<crate::depth::LineFade<'_>>,
) -> RgbaImage {
    let lines_only = crate::EdgeSettings { compose_mode: crate::types::ComposeMode::LinesOnly, ..*edge };
    compose_subject_outline(edge_res, &crate::formats::rgba_view(original), &lines_only, fade)
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
    const SMOOTH_SIGMA: f32 = 1.2;
    match transform {
        InputTransform::None => return Cow::Borrowed(img),
        // At the detector's largest size: blurred at full size, the
        // softening would shrink away with the photo before the detector
        // sees it; at the base size the Fine scale would get an upscaled
        // photo and lose the detail it exists for. The sigma scales with
        // the size, so the base run sees the same softening as before.
        InputTransform::Smooth => {
            let (h, w) = edge_scale_dims(EdgeScale::Fine, BASE_H, BASE_W);
            let k = w as f32 / BASE_W as f32;
            let seen = crate::formats::resize_rgb_image_lanczos3(&opaque_rgb(img), w, h);
            return Cow::Owned(DynamicImage::ImageRgb8(image::imageops::blur(&seen, SMOOTH_SIGMA * k)));
        }
        _ => {}
    }
    let mut rgba = img.to_rgba8();
    match transform {
        InputTransform::None | InputTransform::Smooth => unreachable!("returned above"),
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

/// The photo as opaque 8-bit RGB, borrowed when it already is. Alpha is
/// flattened onto white so edge detection doesn't see ghost content
/// behind removed backgrounds: `src * alpha + 255 * (1 - alpha)`.
fn opaque_rgb(img: &DynamicImage) -> std::borrow::Cow<'_, image::RgbImage> {
    use std::borrow::Cow;
    match img {
        DynamicImage::ImageRgb8(rgb) => Cow::Borrowed(rgb),
        DynamicImage::ImageRgba8(rgba) => Cow::Owned(flatten_on_white(rgba)),
        other if other.color().has_alpha() => Cow::Owned(flatten_on_white(&other.to_rgba8())),
        other => Cow::Owned(other.to_rgb8()),
    }
}

fn flatten_on_white(rgba: &RgbaImage) -> image::RgbImage {
    let (w, h) = rgba.dimensions();
    let mut rgb = image::RgbImage::new(w, h);
    let (dst_stride, src_stride) = (w as usize * 3, w as usize * 4);
    rgb.par_chunks_mut(dst_stride.max(1)).zip(rgba.par_chunks(src_stride.max(1))).for_each(|(dst, src)| {
        for (d, s) in dst.chunks_exact_mut(3).zip(src.chunks_exact(4)) {
            let a = s[3] as f32 / 255.0;
            if a < 1.0 {
                let inv_a = 1.0 - a;
                for (dc, &sc) in d.iter_mut().zip(&s[..3]) {
                    *dc = (sc as f32 * a + 255.0 * inv_a) as u8;
                }
            } else {
                d.copy_from_slice(&s[..3]);
            }
        }
    });
    rgb
}

/// Resize to `w` × `h`, BGR float32, subtract mean (no /255).
fn preprocess(source: &image::RgbImage, w: u32, h: u32) -> Array4<f32> {
    let resized = crate::formats::resize_rgb_image_lanczos3(source, w, h);
    let raw = resized.as_raw();
    let h = h as usize;
    let w = w as usize;

    // Skip the zero-fill; the loop writes every element of all three planes.
    let mut out: Array4<MaybeUninit<f32>> = Array4::uninit((1, 3, h, w));
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

    /// The detector sees a 640×480 copy, so Smooth must soften that copy
    /// alike on a large photo and a small one.
    /// Smooth is made at the Fine scale's size, so Fine still sees twice
    /// the base detail rather than an upscaled base-size photo.
    #[test]
    fn smooth_keeps_the_fine_scales_resolution() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(3000, 2000, Rgb([90, 90, 90])));
        let (h, w) = edge_scale_dims(EdgeScale::Fine, BASE_H, BASE_W);
        let smooth = apply_input_transform(&img, crate::types::InputTransform::Smooth);
        assert_eq!((smooth.width(), smooth.height()), (w, h));
    }

    #[test]
    fn smooth_softens_what_the_detector_sees_at_any_photo_size() {
        for (w, h) in [(3200u32, 2400u32), (320, 240)] {
            let step = |x: u32| if x < w / 2 { image::Rgba([0, 0, 0, 255]) } else { image::Rgba([255, 255, 255, 255]) };
            let img = DynamicImage::ImageRgba8(RgbaImage::from_fn(w, h, |x, _| step(x)));
            let seen = |i: &DynamicImage| crate::formats::resize_rgb_lanczos3(i, BASE_W, BASE_H);
            let plain = seen(&img);
            let smooth = seen(&apply_input_transform(&img, crate::types::InputTransform::Smooth));
            let x = BASE_W / 2 - 2;
            let (p, s) = (plain.get_pixel(x, 240)[0], smooth.get_pixel(x, 240)[0]);
            assert!(s > p.saturating_add(10), "{w}×{h}: Smooth lightens the dark side next to the step: {p} → {s}");
        }
    }

    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    /// TEED's background sits at sigmoid ≈ 0.44, not 0: calibrated, the
    /// background reads 0 and the strongest line 1, so a threshold means
    /// the same on every plane.
    #[test]
    fn calibration_maps_an_offset_background_to_zero() {
        let (h, w) = (BASE_H as usize, BASE_W as usize);
        let mut plane = vec![logit(0.44); h * w];
        plane[(h / 2) * w..(h / 2 + 1) * w].fill(logit(0.98));
        calibrate_plane(&mut plane, h, w);
        assert!(plane[0..w].iter().all(|&p| p.abs() < 1e-5), "background is 0");
        assert!((plane[(h / 2) * w + w / 2] - 1.0).abs() < 1e-5, "the line is 1");
        let mask = tensor_to_edge_mask(&plane, h as u32, w as u32, w as u32, h as u32, 0.5);
        assert_eq!(mask.get_pixel(w as u32 / 2, 10)[0], 0, "no line on the background");
        assert_eq!(mask.get_pixel(w as u32 / 2, h as u32 / 2)[0], 255, "the line is drawn");
    }

    /// A flat photo's plane spans almost nothing; stretched to 0..1 its
    /// noise would draw lines everywhere.
    #[test]
    fn calibration_keeps_a_flat_plane_flat() {
        let (h, w) = (BASE_H as usize, BASE_W as usize);
        let mut plane: Vec<f32> = (0..h * w).map(|i| logit(0.438 + 0.005 * ((i * 7919) % 101) as f32 / 100.0)).collect();
        calibrate_plane(&mut plane, h, w);
        let mask = tensor_to_edge_mask(&plane, h as u32, w as u32, w as u32, h as u32, 0.5);
        assert!(mask.as_raw().iter().all(|&v| v == 0), "no lines on a flat image");
    }

    /// The detector marks a small patch in each corner of any input, a
    /// flat one too; those are not lines, and must not set the plane's
    /// max either. A real line along the border stays.
    #[test]
    fn the_detectors_corner_marks_are_not_lines() {
        let (h, w) = (BASE_H as usize, BASE_W as usize);
        let mut plane = vec![logit(0.44); h * w];
        for (y, x) in [(0, 0), (3, 5), (0, w - 2), (6, w - 1), (h - 2, 0), (h - 1, 7), (h - 1, w - 1), (h - 8, w - 8)] {
            plane[y * w + x] = logit(0.99);
        }
        plane[(h / 2) * w + w / 2] = logit(0.7);
        calibrate_plane(&mut plane, h, w);
        assert!((plane[(h / 2) * w + w / 2] - 1.0).abs() < 1e-5, "the real edge, not a corner mark, is the max");
        let mask = tensor_to_edge_mask(&plane, h as u32, w as u32, 1600, 1200, 0.5);
        for (x, y) in [(0, 0), (1599, 0), (0, 1199), (1599, 1199)] {
            assert_eq!(mask.get_pixel(x, y)[0], 0, "no mark left at corner ({x}, {y})");
        }

        let mut border = vec![logit(0.44); h * w];
        border[..w].fill(logit(0.99));
        calibrate_plane(&mut border, h, w);
        let mask = tensor_to_edge_mask(&border, h as u32, w as u32, w as u32, h as u32, 0.5);
        assert_eq!(mask.get_pixel(w as u32 / 2, 0)[0], 255, "a line along the top edge stays");
    }

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
                // DualScale is handled by `compose_edges_dual_styled` — the
                // single-mask path here renders only the active scale. Callers
                // that select DualScale must dispatch to the dual function; the
                // fallthrough here prevents compile errors and degrades to the
                // user's solid_line_color for correctness.
                LineStyle::DualScale { .. } => solid_tint,
            };
            if let Some(target) = gradient_target {
                let w = if compose == crate::types::ComposeMode::LinesOnly { 255 } else { edge as u16 };
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], target, w);
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
        };

        for i in 0..pixel_count {
            let subject = out_raw[i * 4 + 3] as i32;
            let fine = fine_raw[i] as i32;
            let bold = bold_raw[i] as i32;
            let edge = fine.max(bold);
            out_raw[i * 4 + 3] = alpha_fn(subject, edge);
            if fine > 0 {
                let w = if compose == crate::types::ComposeMode::LinesOnly { 255 } else { fine as u16 };
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], fine_color, w);
            }
            if bold > 0 {
                let w = if compose == crate::types::ComposeMode::LinesOnly { 255 } else { bold as u16 };
                blend_rgb(&mut out_raw[i * 4..i * 4 + 3], bold_color, w);
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
        let styles = [
            LineStyle::Solid,
            LineStyle::GradientY { top: [255, 0, 0], bottom: [0, 0, 255] },
            LineStyle::GradientX { left: [0, 255, 0], right: [255, 0, 255] },
            LineStyle::RadialGradient { center: [40, 200], inner: [255, 255, 0], outer: [0, 255, 255] },
            LineStyle::Rainbow { cycles: 3 },
            LineStyle::DualScale { fine_color: [1, 2, 3], bold_color: [4, 5, 6] },
        ];
        let dilated = |m: &image::GrayImage, t: u32| image::GrayImage::from_raw(w, h, reference_dilated(m, t)).unwrap();
        for thickness in [0, 2] {
            let (dm, db) = (dilated(&mask, thickness), dilated(&bold, thickness));
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

    /// In Subject mode the lines are traced from the plain cut-out, so an
    /// effect never moves them; the effect then colours the subject the
    /// lines are drawn over.
    #[test]
    fn subject_lines_are_traced_before_the_effect() {
        let subject = RgbaImage::from_pixel(4, 4, image::Rgba([200, 40, 40, 255]));
        let mut traced_from = None;
        let edge = crate::EdgeSettings { compose_mode: crate::ComposeMode::SubjectFilled, ..crate::EdgeSettings::default() };
        let (out, _) = outline_subject(subject.clone(), crate::FillStyle::Invert, &edge, None, |img| {
            traced_from = Some(img.to_rgba8());
            Ok(EdgeInferenceResult { tensors: std::array::from_fn(|_| vec![0.0; 4]), height: 2, width: 2 })
        })
        .expect("traces");
        assert_eq!(traced_from.expect("traced").get_pixel(0, 0).0, [200, 40, 40, 255], "the plain subject");
        assert_eq!(out.get_pixel(0, 0).0, [55, 215, 215, 255], "drawn over the subject with its effect");
    }

    /// Wide lines end and turn in round caps, like a pen, not squares.
    #[test]
    fn thick_lines_have_round_ends() {
        let mut dot = image::GrayImage::new(21, 21);
        dot.put_pixel(10, 10, image::Luma([255]));
        let thick = thickened_edges(&dot, 6).expect("width above 0 thickens");
        assert_eq!(thick.get_pixel(16, 10)[0], 255, "reaches the full width along an axis");
        assert_eq!(thick.get_pixel(16, 16)[0], 0, "but not the square's corner");
    }

    fn solid_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([120, 120, 120])))
    }

    /// Every scale's plane at its own size, each pixel `prob(index, len)`.
    fn every_scale(prob: impl Fn(usize, usize) -> f32) -> EdgeInferenceResult {
        let tensors = EdgeScale::ALL.map(|scale| {
            let (h, w) = edge_scale_dims(scale, BASE_H, BASE_W);
            let len = (h * w) as usize;
            (0..len).map(|j| prob(j, len)).collect()
        });
        EdgeInferenceResult { tensors, height: BASE_H, width: BASE_W }
    }

    #[test]
    fn whole_photo_solid_color_paints_edges() {
        let original = solid_rgb(64, 48);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: Some([255, 0, 0]), edge_thickness: 0, edge_scale: crate::EdgeScale::Fused, compose_mode: crate::ComposeMode::default(), line_style: crate::LineStyle::default(), input_transform: crate::InputTransform::default() };
        let top_half_lines = every_scale(|i, len| if i < len / 2 { 1.0 } else { 0.0 });
        let out = compose_whole_photo(&top_half_lines, &original, &edge, None);
        assert_eq!(out.width(), 64);
        assert_eq!(out.height(), 48);
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
        let fine_len = (BASE_H * BASE_W * 4) as usize;
        let edge_res = every_scale(|_, len| if len == fine_len { 1.0 } else { 0.0 });
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
        let out = compose_subject_outline(&edge_res, &masked, &edge, None);
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
    fn rainbow_line_style_does_not_overflow_above_6m_pixels() {
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
            crate::LineStyle::Rainbow { cycles: 3 },
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
        // Background, very faint, mid-strength, strong edge.
        let probs = vec![0.0_f32, 0.05, 0.5, 0.95];
        let mask = tensor_to_edge_mask(&probs, 2, 2, 2, 2, 1.0);
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
    fn whole_photo_preserves_original_rgb_when_no_line_color() {
        let original = solid_rgb(32, 32);
        let edge = crate::EdgeSettings { line_strength: 0.5, solid_line_color: None, edge_thickness: 0, edge_scale: crate::EdgeScale::Fused, compose_mode: crate::ComposeMode::default(), line_style: crate::LineStyle::default(), input_transform: crate::InputTransform::default() };
        let out = compose_whole_photo(&every_scale(|_, _| 1.0), &original, &edge, None);
        // Original color preserved
        assert_eq!(out.get_pixel(0, 0)[0], 120);
        assert_eq!(out.get_pixel(0, 0)[1], 120);
        assert_eq!(out.get_pixel(0, 0)[2], 120);
    }

    /// Fade with distance thins the lines toward the far top of a ground
    /// plane and leaves the near bottom at full strength.
    #[test]
    fn faded_lines_lose_alpha_with_distance() {
        let original = solid_rgb(32, 32);
        let edge = crate::EdgeSettings::default();
        let ramp = crate::depth::DepthMap::from_raw(1, 32, (0..32).map(|y| y as f32).collect());
        let fade = crate::depth::LineFade { depth: &ramp, amount: 1.0 };
        let plain = compose_whole_photo(&every_scale(|_, _| 1.0), &original, &edge, None);
        let faded = compose_whole_photo(&every_scale(|_, _| 1.0), &original, &edge, Some(fade));
        assert!(faded.get_pixel(5, 0)[3] < plain.get_pixel(5, 0)[3] / 4, "far lines fade");
        assert!(u16::from(faded.get_pixel(5, 31)[3]) + 8 >= u16::from(plain.get_pixel(5, 31)[3]), "near lines keep their strength");
    }
}
