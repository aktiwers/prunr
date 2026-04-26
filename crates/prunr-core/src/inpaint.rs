//! LaMa-based inpainting. Public entry: `process_inpaint(image, mask)`.
//! Tiles 512-px with 64-px feathered overlap so large images stay
//! inside LaMa's fixed input size and seams stay invisible against
//! flat backgrounds.

use std::sync::{Mutex, OnceLock};

use image::{GrayImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::{Session, builder::GraphOptimizationLevel}, value::Tensor};

use crate::types::CoreError;

/// LaMa input/output side length in pixels.
pub const TILE: u32 = 512;
/// Overlap between adjacent tiles in pixels. Half the tile to stay safe
/// on the corners; the feather blend tapers within this region.
pub const OVERLAP: u32 = 64;

/// Eagerly initialise the inpaint session for `id` in the background so
/// the first stroke doesn't pay the ~5-10s session-build latency.
/// Idempotent — repeated calls for the same id are no-ops.
pub fn prewarm(id: prunr_models::ModelId) -> Result<(), CoreError> {
    if is_sd_inpaint(id) {
        return crate::inpaint_sd::prewarm(id);
    }
    LamaSession::get(id).map(|_| ())
}

fn is_sd_inpaint(id: prunr_models::ModelId) -> bool {
    matches!(id, prunr_models::ModelId::SdV15InpaintFp16)
}

/// Dilate (px > 0) or erode (px < 0) a binary mask. `|px|` iterations
/// of single-pixel 4-connected morphology — fine for the small values
/// the GUI exposes (±16 px max). px == 0 returns a clone unchanged.
pub fn grow_mask(mask: &GrayImage, px: i32) -> GrayImage {
    if px == 0 {
        return mask.clone();
    }
    let iters = px.unsigned_abs() as usize;
    let dilate = px > 0;
    let mut current = mask.clone();
    for _ in 0..iters {
        current = morphology_step(&current, dilate);
    }
    current
}

fn morphology_step(mask: &GrayImage, dilate: bool) -> GrayImage {
    let (w, h) = mask.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;
    let raw = mask.as_raw();
    let mut out = GrayImage::new(w, h);
    let buf = out.as_mut();
    for y in 0..h_us {
        for x in 0..w_us {
            // 4-connected: center + N, S, E, W. Edge pixels treat OOB
            // as zero (so erosion shrinks at the image boundary).
            let center = raw[y * w_us + x] > 127;
            let n = y > 0 && raw[(y - 1) * w_us + x] > 127;
            let s = y + 1 < h_us && raw[(y + 1) * w_us + x] > 127;
            let e = x + 1 < w_us && raw[y * w_us + x + 1] > 127;
            let we = x > 0 && raw[y * w_us + x - 1] > 127;
            let bit = if dilate {
                center || n || s || e || we
            } else {
                center
                    && (y > 0 && n)
                    && (y + 1 < h_us && s)
                    && (x + 1 < w_us && e)
                    && (x > 0 && we)
            };
            buf[y * w_us + x] = if bit { 255 } else { 0 };
        }
    }
    out
}

/// Soft-blend the inpainted output toward `source` within `feather_px`
/// of the mask boundary. Pixels deep inside the mask stay fully
/// inpainted; pixels right at the boundary fade toward source. Hides
/// the LaMa↔source seam.
///
/// Uses a 2-pass chamfer distance transform — accurate enough for the
/// small radii exposed (≤32 px) and O(n) on the image.
pub fn feather_inpainted(
    inpainted: &RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    feather_px: f32,
) -> RgbaImage {
    if feather_px <= 0.0 {
        return inpainted.clone();
    }
    if inpainted.dimensions() != mask.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        tracing::warn!("feather_inpainted: dim mismatch, skipping");
        return inpainted.clone();
    }
    let dist = chamfer_distance_inside(mask);
    let (w, h) = inpainted.dimensions();
    let mut out = inpainted.clone();
    let out_raw = out.as_mut();
    let src_raw = source.as_raw();
    let inp_raw = inpainted.as_raw();
    let n = (w * h) as usize;
    for i in 0..n {
        let pix = i * 4;
        let d = dist[i];
        if d <= 0.0 {
            // Outside the mask: source. (decode_tile already passes
            // these through, but be defensive in case feather is
            // chained after a different pipeline.)
            out_raw[pix] = src_raw[pix];
            out_raw[pix + 1] = src_raw[pix + 1];
            out_raw[pix + 2] = src_raw[pix + 2];
            continue;
        }
        if d >= feather_px {
            // Deep inside: full inpainted (already in `out` from clone).
            continue;
        }
        let t = d / feather_px;
        for c in 0..3 {
            let s = src_raw[pix + c] as f32;
            let p = inp_raw[pix + c] as f32;
            out_raw[pix + c] = (s + t * (p - s)).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Forward + backward chamfer pass returning, for each pixel, the
/// distance to the nearest mask==0 pixel (0.0 if outside). Diagonal
/// step uses √2 to keep the metric near-Euclidean.
fn chamfer_distance_inside(mask: &GrayImage) -> Vec<f32> {
    let (w, h) = mask.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;
    let large = (w + h) as f32;
    let mut d: Vec<f32> = mask.as_raw().iter()
        .map(|&v| if v > 127 { large } else { 0.0 })
        .collect();
    const DIAG: f32 = 1.4142135;
    // Forward pass: top-left → bottom-right.
    for y in 0..h_us {
        for x in 0..w_us {
            let i = y * w_us + x;
            if d[i] == 0.0 { continue; }
            let mut best = d[i];
            if x > 0 { best = best.min(d[i - 1] + 1.0); }
            if y > 0 { best = best.min(d[i - w_us] + 1.0); }
            if x > 0 && y > 0 { best = best.min(d[i - w_us - 1] + DIAG); }
            if x + 1 < w_us && y > 0 { best = best.min(d[i - w_us + 1] + DIAG); }
            d[i] = best;
        }
    }
    // Backward pass: bottom-right → top-left.
    for y in (0..h_us).rev() {
        for x in (0..w_us).rev() {
            let i = y * w_us + x;
            if d[i] == 0.0 { continue; }
            let mut best = d[i];
            if x + 1 < w_us { best = best.min(d[i + 1] + 1.0); }
            if y + 1 < h_us { best = best.min(d[i + w_us] + 1.0); }
            if x + 1 < w_us && y + 1 < h_us { best = best.min(d[i + w_us + 1] + DIAG); }
            if x > 0 && y + 1 < h_us { best = best.min(d[i + w_us - 1] + DIAG); }
            d[i] = best;
        }
    }
    d
}

/// Apply an unsharp-mask sharpen pass over the inpainted region. LaMa
/// output tends to be soft on high-frequency texture; this counters
/// the blur without disturbing source pixels outside the mask.
///
/// `amount` in [0.0, 2.0]. 0.0 leaves `image` unchanged. 0.3-0.7 is
/// the practical range; higher introduces visible halos. Operates
/// in-place by allocating a fresh RGBA and returning it.
///
/// Algorithm: `out = src + amount * (src - blur(src))`, applied only
/// to pixels where `mask > 127`. Blur is a separable 5-tap Gaussian
/// (sigma ≈ 1.0).
pub fn sharpen_inpainted(image: &RgbaImage, mask: &GrayImage, amount: f32) -> RgbaImage {
    if amount <= 0.0 {
        return image.clone();
    }
    let amount = amount.min(2.0);
    let (w, h) = image.dimensions();
    if image.dimensions() != mask.dimensions() {
        tracing::warn!(
            image_dims = ?image.dimensions(),
            mask_dims = ?mask.dimensions(),
            "sharpen_inpainted: dim mismatch, skipping",
        );
        return image.clone();
    }
    // 5-tap separable Gaussian, sigma ≈ 1.0 (kernel: 1, 4, 6, 4, 1 / 16).
    let kernel: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let src = image.as_raw();
    let w_us = w as usize;
    let h_us = h as usize;

    // Horizontal pass into a temp f32 buffer.
    let mut tmp: Vec<f32> = vec![0.0; w_us * h_us * 3];
    for y in 0..h_us {
        for x in 0..w_us {
            for c in 0..3 {
                let mut sum = 0.0;
                for k in 0..5 {
                    let xi = (x as isize + k as isize - 2).clamp(0, w_us as isize - 1) as usize;
                    sum += kernel[k] * src[(y * w_us + xi) * 4 + c] as f32;
                }
                tmp[(y * w_us + x) * 3 + c] = sum;
            }
        }
    }

    let mut out = RgbaImage::new(w, h);
    let out_raw = out.as_mut();
    let msk_raw = mask.as_raw();
    for y in 0..h_us {
        for x in 0..w_us {
            let pix = (y * w_us + x) * 4;
            // Always copy alpha; RGB depends on mask.
            out_raw[pix + 3] = src[pix + 3];
            if msk_raw[y * w_us + x] <= 127 {
                out_raw[pix] = src[pix];
                out_raw[pix + 1] = src[pix + 1];
                out_raw[pix + 2] = src[pix + 2];
                continue;
            }
            for c in 0..3 {
                let mut blur = 0.0;
                for k in 0..5 {
                    let yi = (y as isize + k as isize - 2).clamp(0, h_us as isize - 1) as usize;
                    blur += kernel[k] * tmp[(yi * w_us + x) * 3 + c];
                }
                let s = src[pix + c] as f32;
                let sharp = s + amount * (s - blur);
                out_raw[pix + c] = sharp.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Top-level inpaint entry. `id` selects the inpaint backend
/// (LaMaFp32, BigLaMa, MI-GAN, SD …). Returns the input unchanged when
/// the mask is all-zero (no work). SD-family ids dispatch to the
/// `inpaint_sd` module which has its own multi-model pipeline.
pub fn process_inpaint(image: &RgbaImage, mask: &GrayImage, id: prunr_models::ModelId) -> Result<RgbaImage, CoreError> {
    if image.dimensions() != mask.dimensions() {
        return Err(CoreError::Inference(format!(
            "inpaint: dim mismatch — image {:?} vs mask {:?}",
            image.dimensions(),
            mask.dimensions()
        )));
    }
    // Short-circuit before the 200 MB session load — empty masks are
    // common during live preview when the user hasn't started painting.
    if mask_is_empty(mask) {
        return Ok(image.clone());
    }
    if is_sd_inpaint(id) {
        // 20 inference steps — SD-1.5 sweet spot at default scheduler.
        return crate::inpaint_sd::process_inpaint(image, mask, id, 20);
    }
    let session = LamaSession::get(id)?;
    tile_compose(image, mask, |tile_rgba, tile_mask| {
        session.run_tile(tile_rgba, tile_mask).unwrap_or_else(|e| {
            tracing::error!(%e, "LaMa tile inference failed; leaving tile unchanged");
            tile_rgba.clone()
        })
    })
}

/// One LaMa session per process — see `LamaSession::get`. Tiles run
/// sequentially under one Mutex; do NOT parallelise across tiles
/// (ORT session inference is already multi-threaded internally).
struct LamaSession {
    session: Mutex<Session>,
    image_input_name: String,
    mask_input_name: String,
}

impl LamaSession {
    /// Per-id cache. Each LamaSession is built once on first use and
    /// reused across strokes. Failures are also cached as strings (the
    /// inner Mutex protects the map; CoreError isn't Clone so we store
    /// the message and rebuild the wrapper at each call).
    fn get(id: prunr_models::ModelId) -> Result<&'static LamaSession, CoreError> {
        static CACHE: OnceLock<Mutex<std::collections::HashMap<prunr_models::ModelId, Result<&'static LamaSession, String>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        {
            let guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(entry) = guard.get(&id) {
                return entry.clone().map_err(CoreError::Inference);
            }
        }
        // Build outside the lock — session creation is seconds-long.
        let entry: Result<&'static LamaSession, String> = match Self::new_inner(id) {
            Ok(s) => Ok(Box::leak(Box::new(s))),
            Err(e) => Err(e),
        };
        let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Concurrent caller may have inserted between our check and
        // build; keep their entry (drops our duplicate session, harmless).
        let stored = guard.entry(id).or_insert(entry);
        stored.clone().map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId) -> Result<LamaSession, String> {
        let bytes = prunr_models::resolve_bytes(id)
            .ok_or_else(|| prunr_models::not_installed_error(id))?;
        let bytes = bytes.as_ref();
        // Match physical cores so LaMa convolutions saturate the CPU.
        // Unlike the seg models (single-image dispatch with intra=1),
        // LaMa runs on the GUI hot path — the user is waiting per stroke.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);

        // Try GPU EPs first; fall back to CPU on any failure. Mirrors
        // `OrtEngine`'s one-EP-at-a-time ladder so a crashing EP can't
        // abort the whole session creation. Logs which EP won.
        let (session, provider) = build_lama_session(id, bytes, threads)?;

        let inputs = session.inputs().iter()
            .map(|i| i.name().to_string())
            .collect::<Vec<_>>();
        if inputs.len() < 2 {
            return Err(format!("LaMa: expected 2 inputs, got {}: {inputs:?}", inputs.len()));
        }
        let (img_name, mask_name) = pick_input_names(&inputs);
        tracing::info!(
            ?id,
            image_input = %img_name,
            mask_input = %mask_name,
            provider = %provider,
            threads,
            "Inpaint session ready",
        );
        Ok(LamaSession {
            session: Mutex::new(session),
            image_input_name: img_name,
            mask_input_name: mask_name,
        })
    }

    fn run_tile(&self, image: &RgbaImage, mask: &GrayImage) -> Result<RgbaImage, CoreError> {
        let (w, h) = image.dimensions();
        let (img_in, mask_in) = encode_tile(image, mask);

        let painted = {
            let mut session = self.session.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let img_t = Tensor::from_array(img_in)
                .map_err(|e| CoreError::Inference(format!("LaMa: image tensor: {e}")))?;
            let mask_t = Tensor::from_array(mask_in)
                .map_err(|e| CoreError::Inference(format!("LaMa: mask tensor: {e}")))?;
            let outputs = session.run(inputs![
                self.image_input_name.as_str() => &img_t,
                self.mask_input_name.as_str() => &mask_t,
            ])
                .map_err(|e| CoreError::Inference(format!("LaMa: inference failed: {e}")))?;
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| CoreError::Inference(format!("LaMa: output extract: {e}")))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| CoreError::Inference(format!("LaMa: output reshape: {e}")))?
                .to_owned()
        };

        Ok(decode_tile(&painted, image, mask, w, h))
    }
}

/// Build the LaMa ORT session, trying GPU EPs first and falling back
/// to CPU. Returns `(session, provider_name)` so the caller can log
/// which path won. CPU is the universal fallback; if even CPU fails
/// we propagate the error string.
///
/// Each GPU candidate is **smoke-tested** with a tiny inference before
/// being accepted — some EPs (notably CUDA on certain ORT versions)
/// can `commit_from_memory` successfully but fail at runtime on
/// specific ops, e.g. `"Reshape node ... GetElementType is not
/// implemented"` on this LaMa export. A silent runtime failure
/// would leave every tile unchanged; the smoke test catches it
/// at init so the CPU fallback engages and the user sees actual
/// inpainting.
fn build_lama_session(
    id: prunr_models::ModelId,
    bytes: &[u8],
    threads: usize,
) -> Result<(Session, String), String> {
    let gpu_eps = crate::engine::available_gpu_eps();

    for ep_name in gpu_eps {
        // Static catalog + dynamic cache, same as engine.rs.
        if !prunr_models::is_ep_compatible(id, ep_name) {
            tracing::debug!(?id, ep = %ep_name, "LaMa: EP statically incompatible; skipping");
            continue;
        }
        if crate::ep_compat::is_known_failure(ep_name, id) {
            tracing::debug!(?id, ep = %ep_name, "LaMa: EP cached as incompatible; skipping");
            continue;
        }
        let builder = match base_builder(threads) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(ep = %ep_name, %e, "LaMa: builder init failed");
                continue;
            }
        };
        let registered = match *ep_name {
            #[cfg(not(target_os = "macos"))]
            "CUDA" => builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ]),
            #[cfg(target_os = "macos")]
            "CoreML" => builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ]),
            #[cfg(windows)]
            "DirectML" => builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ]),
            // Default device "AUTO" lets OpenVINO pick the best target
            // (iGPU when present + driver works, NPU on newer Intel,
            // else CPU). Smoke test below catches op-incompat failures.
            #[cfg(not(target_os = "macos"))]
            "OpenVINO" => builder.with_execution_providers([
                ort::execution_providers::OpenVINOExecutionProvider::default().build(),
            ]),
            _ => continue,
        };
        let mut built = match registered {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(ep = %ep_name, %e, "LaMa: register EP failed");
                continue;
            }
        };
        let mut session = match built.commit_from_memory(bytes) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(ep = %ep_name, %e, "LaMa: GPU session commit failed — trying next");
                crate::ep_compat::record_failure(ep_name, id, &format!("{e}"));
                continue;
            }
        };
        match smoke_test_session(&mut session) {
            Ok(()) => {
                tracing::info!(ep = %ep_name, "LaMa: GPU session validated");
                return Ok((session, (*ep_name).to_string()));
            }
            Err(e) => {
                tracing::warn!(
                    ep = %ep_name, %e,
                    "LaMa: GPU session smoke test failed — falling back to next EP/CPU",
                );
                crate::ep_compat::record_failure(ep_name, id, &e);
            }
        }
    }

    // CPU fallback: no EP registration, ORT uses its default CPU provider.
    let session = base_builder(threads)?
        .commit_from_memory(bytes)
        .map_err(|e| format!("LaMa: CPU session commit failed: {e}"))?;
    Ok((session, "CPU".to_string()))
}

/// Run one inference with zero-padded inputs to validate that the
/// session can actually execute this graph. Catches EP/op
/// incompatibilities that don't surface until the first run.
fn smoke_test_session(session: &mut Session) -> Result<(), String> {
    let s = TILE as usize;
    let img = ndarray::Array4::<f32>::zeros((1, 3, s, s));
    let msk = ndarray::Array4::<f32>::zeros((1, 1, s, s));
    let inputs = session.inputs().iter()
        .map(|i| i.name().to_string())
        .collect::<Vec<_>>();
    if inputs.len() < 2 {
        return Err(format!("smoke test: need 2 inputs, got {inputs:?}"));
    }
    let (img_name, mask_name) = pick_input_names(&inputs);
    let img_t = Tensor::from_array(img)
        .map_err(|e| format!("smoke test: image tensor: {e}"))?;
    let msk_t = Tensor::from_array(msk)
        .map_err(|e| format!("smoke test: mask tensor: {e}"))?;
    session.run(inputs![
        img_name.as_str() => &img_t,
        mask_name.as_str() => &msk_t,
    ])
        .map_err(|e| format!("smoke test: inference failed: {e}"))?;
    Ok(())
}

fn base_builder(threads: usize) -> Result<ort::session::builder::SessionBuilder, String> {
    Session::builder()
        .map_err(|e| format!("LaMa: ORT builder init failed: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("LaMa: optimization level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| format!("LaMa: intra_threads: {e}"))
}

/// LaMa-ONNX exports use `image` + `mask` as input names (Carve/LaMa-ONNX
/// convention). Match case-insensitively; fall back to positional order.
fn pick_input_names(names: &[String]) -> (String, String) {
    let mut img = None;
    let mut mask = None;
    for n in names {
        let l = n.to_ascii_lowercase();
        if img.is_none() && l.contains("image") { img = Some(n.clone()); continue; }
        if mask.is_none() && l.contains("mask") { mask = Some(n.clone()); continue; }
    }
    (img.unwrap_or_else(|| names[0].clone()),
     mask.unwrap_or_else(|| names[1].clone()))
}

/// Encode an RGBA tile + binary mask into the two NCHW float tensors LaMa
/// expects. Pads to TILE×TILE with zeros when the tile is smaller (edge
/// tiles or sub-512 images).
///
/// LaMa-ONNX (Carve) input convention:
/// - `image` [1, 3, TILE, TILE] f32 in [0, 1]
/// - `mask`  [1, 1, TILE, TILE] f32 in {0, 1} (1 = inpaint here)
fn encode_tile(image: &RgbaImage, mask: &GrayImage) -> (Array4<f32>, Array4<f32>) {
    let s = TILE as usize;
    let plane = s * s;
    let mut img = Array4::<f32>::zeros((1, 3, s, s));
    let mut msk = Array4::<f32>::zeros((1, 1, s, s));
    // invariant: freshly allocated, standard layout — slice views are valid.
    let img_buf = img.as_slice_mut().unwrap();
    let msk_buf = msk.as_slice_mut().unwrap();
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let img_raw = image.as_raw();
    let msk_raw = mask.as_raw();
    for y in 0..h_us {
        let src_row = y * w_us * 4;
        let dst_row = y * s;
        let mask_row = y * w_us;
        for x in 0..w_us {
            let src = src_row + x * 4;
            let dst = dst_row + x;
            img_buf[dst] = img_raw[src] as f32 / 255.0;
            img_buf[plane + dst] = img_raw[src + 1] as f32 / 255.0;
            img_buf[plane * 2 + dst] = img_raw[src + 2] as f32 / 255.0;
            msk_buf[dst] = if msk_raw[mask_row + x] > 127 { 1.0 } else { 0.0 };
        }
    }
    (img, msk)
}

/// Decode LaMa output back to RGBA at the tile's actual size.
///
/// Outside the mask, copy from the source tile so the un-inpainted
/// region matches exactly — LaMa often perturbs unmasked pixels by
/// fractions of a unit, which would defeat the feather blend.
fn decode_tile(
    output: &Array4<f32>,
    source: &RgbaImage,
    mask: &GrayImage,
    w: u32,
    h: u32,
) -> RgbaImage {
    let s = TILE as usize;
    let plane = s * s;
    let buf = output.as_slice().unwrap_or(&[]);
    if buf.len() < plane * 3 {
        tracing::warn!(buf_len = buf.len(), "LaMa output smaller than expected; returning source");
        return source.clone();
    }
    // Heuristic: max > 1.5 ⇒ output is in [0, 255], else [0, 1].
    // Sample all three planes — a dim red channel could mis-detect.
    let max = buf[..plane * 3].iter().copied().fold(0.0_f32, f32::max);
    let scale = if max > 1.5 { 1.0 } else { 255.0 };

    let mut out = RgbaImage::new(w, h);
    let src_raw = source.as_raw();
    let msk_raw = mask.as_raw();
    let out_raw = out.as_mut();
    let w_us = w as usize;
    for y in 0..h as usize {
        let nchw_row = y * s;
        let rgba_row = y * w_us * 4;
        let mask_row = y * w_us;
        for x in 0..w_us {
            let i = nchw_row + x;
            let pix = rgba_row + x * 4;
            if msk_raw[mask_row + x] > 127 {
                out_raw[pix] = (buf[i] * scale).clamp(0.0, 255.0) as u8;
                out_raw[pix + 1] = (buf[plane + i] * scale).clamp(0.0, 255.0) as u8;
                out_raw[pix + 2] = (buf[plane * 2 + i] * scale).clamp(0.0, 255.0) as u8;
            } else {
                out_raw[pix] = src_raw[pix];
                out_raw[pix + 1] = src_raw[pix + 1];
                out_raw[pix + 2] = src_raw[pix + 2];
            }
            out_raw[pix + 3] = src_raw[pix + 3];
        }
    }
    out
}

/// True when every pixel of `mask` is zero. O(n) scan; cheap relative
/// to inference, and short-circuits the entire pipeline.
fn mask_is_empty(mask: &GrayImage) -> bool {
    mask.as_raw().iter().all(|&v| v == 0)
}

/// Source rect (in image pixels) for one inpaint tile + the same rect
/// in the destination buffer (always identical here, but the type
/// makes future repositioning explicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlacement {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Lay tiles across `(width, height)` with `OVERLAP` overlap. Tiles
/// near the right/bottom edges shift backwards to keep `TILE × TILE`
/// extent — never extending past the image bounds. For images <= TILE
/// in either axis, a single tile covers the whole image at its actual
/// size (LaMa accepts smaller inputs after pad-to-512 in the wrapper).
pub(crate) fn plan_tiles(width: u32, height: u32) -> Vec<TilePlacement> {
    if width <= TILE && height <= TILE {
        return vec![TilePlacement { x: 0, y: 0, w: width, h: height }];
    }
    let step = TILE - OVERLAP;
    let mut xs: Vec<u32> = (0..)
        .map(|i| i * step)
        .take_while(|&x| x + TILE <= width || x == 0)
        .collect();
    if let Some(&last_x) = xs.last() {
        if last_x + TILE < width {
            xs.push(width - TILE);
        }
    }
    if xs.is_empty() {
        xs.push(0);
    }
    let mut ys: Vec<u32> = (0..)
        .map(|i| i * step)
        .take_while(|&y| y + TILE <= height || y == 0)
        .collect();
    if let Some(&last_y) = ys.last() {
        if last_y + TILE < height {
            ys.push(height - TILE);
        }
    }
    if ys.is_empty() {
        ys.push(0);
    }

    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            let w = TILE.min(width - x);
            let h = TILE.min(height - y);
            out.push(TilePlacement { x, y, w, h });
        }
    }
    out
}

/// Smoothstep weight for blending tile contributions in the overlap
/// region. Returns 1.0 at the tile interior, 0.0 at the very edge,
/// with a smooth `t² (3 - 2t)` taper across `OVERLAP` pixels.
pub(crate) fn feather_weight(distance_from_edge: u32) -> f32 {
    if distance_from_edge >= OVERLAP {
        1.0
    } else {
        let t = distance_from_edge as f32 / OVERLAP as f32;
        crate::math::smoothstep(t)
    }
}

/// Compose the inpaint output by walking each tile, running the
/// inference closure, and feather-blending into the output buffer.
/// Skips tiles whose mask is all-zero (no work). The closure runs
/// once per non-empty tile.
pub(crate) fn tile_compose<F>(
    image: &RgbaImage,
    mask: &GrayImage,
    mut inpaint_tile: F,
) -> Result<RgbaImage, CoreError>
where
    F: FnMut(&RgbaImage, &GrayImage) -> RgbaImage,
{
    let (w, h) = image.dimensions();
    let mut out = image.clone();

    // Per-pixel weight accumulator for the feather blend. Tiles overlap
    // in the OVERLAP band; each contributes a smoothstep-weighted
    // sample, normalized at the end.
    let mut weight_acc: Vec<f32> = vec![0.0; (w * h) as usize];
    // Accumulator for the weighted RGBA in f32. We blend in linear
    // f32 space and quantize back to u8 at the end.
    let mut color_acc: Vec<[f32; 4]> = vec![[0.0; 4]; (w * h) as usize];

    for tile in plan_tiles(w, h) {
        let tile_mask = image::imageops::crop_imm(mask, tile.x, tile.y, tile.w, tile.h).to_image();
        if mask_is_empty(&tile_mask) {
            continue;
        }
        let tile_rgba = image::imageops::crop_imm(image, tile.x, tile.y, tile.w, tile.h).to_image();
        let painted = inpaint_tile(&tile_rgba, &tile_mask);
        accumulate_tile(&mut color_acc, &mut weight_acc, &painted, &tile, w);
    }

    // Resolve accumulated tiles. Unmasked pixels are skipped — `out`
    // already started as `image.clone()`, so leaving them untouched
    // gives byte-identical source preservation. Going through the
    // float blend would introduce 1-unit drift in u8 quantization
    // even when every contributing tile sampled identical source
    // pixels (255 * w / w ≠ 255 in float arithmetic).
    let mask_raw = mask.as_raw();
    for (i, pixel) in out.pixels_mut().enumerate() {
        if mask_raw[i] == 0 {
            continue;
        }
        let wsum = weight_acc[i];
        if wsum > 0.0 {
            let inv = 1.0 / wsum;
            let c = color_acc[i];
            // Alpha stays from the source clone — LaMa is RGB-only.
            pixel.0[0] = (c[0] * inv).clamp(0.0, 255.0) as u8;
            pixel.0[1] = (c[1] * inv).clamp(0.0, 255.0) as u8;
            pixel.0[2] = (c[2] * inv).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

fn accumulate_tile(
    color_acc: &mut [[f32; 4]],
    weight_acc: &mut [f32],
    tile: &RgbaImage,
    placement: &TilePlacement,
    image_w: u32,
) {
    for ty in 0..placement.h {
        let dist_top = ty;
        let dist_bottom = placement.h - 1 - ty;
        let edge_y = dist_top.min(dist_bottom);
        for tx in 0..placement.w {
            let dist_left = tx;
            let dist_right = placement.w - 1 - tx;
            let edge_x = dist_left.min(dist_right);
            let w = feather_weight(edge_x.min(edge_y));
            let dst_idx = ((placement.y + ty) * image_w + (placement.x + tx)) as usize;
            let p = tile.get_pixel(tx, ty).0;
            let acc = &mut color_acc[dst_idx];
            acc[0] += p[0] as f32 * w;
            acc[1] += p[1] as f32 * w;
            acc[2] += p[2] as f32 * w;
            acc[3] += p[3] as f32 * w;
            weight_acc[dst_idx] += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    #[test]
    fn plan_tiles_small_image_single_tile() {
        let tiles = plan_tiles(400, 300);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], TilePlacement { x: 0, y: 0, w: 400, h: 300 });
    }

    #[test]
    fn plan_tiles_exact_tile_size_single_tile() {
        let tiles = plan_tiles(TILE, TILE);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], TilePlacement { x: 0, y: 0, w: TILE, h: TILE });
    }

    #[test]
    fn plan_tiles_horizontal_strip_overlaps_correctly() {
        // 1024 wide → first tile [0..512], second tile must be at
        // 1024 - 512 = 512 to cover the right edge.
        let tiles = plan_tiles(1024, TILE);
        assert!(tiles.len() >= 2, "1024-wide must produce >=2 tiles");
        let xs: Vec<u32> = tiles.iter().map(|t| t.x).collect();
        assert!(xs.contains(&0));
        assert!(xs.iter().any(|&x| x + TILE == 1024), "must cover the right edge");
    }

    #[test]
    fn plan_tiles_grid_full_size() {
        // 1024×1024 with 64-px overlap: a clean 2×2 split would leave
        // tiles touching with NO overlap (no feather possible). Plan
        // therefore packs 3 tiles per axis with proper overlap and
        // an extra row/column hugging the right/bottom edge.
        let tiles = plan_tiles(1024, 1024);
        assert!(tiles.len() >= 4, "must have at least 2×2 coverage, got {}", tiles.len());
        for t in &tiles {
            assert_eq!(t.w, TILE);
            assert_eq!(t.h, TILE);
        }
        // Right + bottom edges must be covered.
        assert!(tiles.iter().any(|t| t.x + t.w == 1024), "right edge");
        assert!(tiles.iter().any(|t| t.y + t.h == 1024), "bottom edge");
    }

    #[test]
    fn plan_tiles_covers_full_image_pixel_by_pixel() {
        // Every pixel of a 1500×900 image must be covered by ≥1 tile.
        let (w, h) = (1500u32, 900u32);
        let tiles = plan_tiles(w, h);
        for y in 0..h {
            for x in 0..w {
                let covered = tiles.iter().any(|t| {
                    x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h
                });
                assert!(covered, "pixel ({x}, {y}) uncovered");
            }
        }
    }

    #[test]
    fn feather_weight_full_inside_overlap() {
        // Far from the tile edge: weight 1.0.
        assert!((feather_weight(OVERLAP) - 1.0).abs() < 1e-6);
        assert!((feather_weight(OVERLAP * 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn feather_weight_zero_at_edge() {
        assert!(feather_weight(0).abs() < 1e-6);
    }

    #[test]
    fn feather_weight_smoothstep_monotone() {
        let mut prev = 0.0;
        for d in 0..=OVERLAP {
            let w = feather_weight(d);
            assert!(w >= prev - 1e-6, "feather weight must be monotonic non-decreasing");
            prev = w;
        }
    }

    #[test]
    fn process_inpaint_dim_mismatch_errors() {
        let img = RgbaImage::new(100, 100);
        let mask = GrayImage::new(50, 50);
        let result = process_inpaint(&img, &mask, prunr_models::ModelId::LaMaFp32);
        assert!(result.is_err());
    }

    #[test]
    fn process_inpaint_empty_mask_returns_input_unchanged() {
        let mut img = RgbaImage::new(64, 64);
        for (_, _, p) in img.enumerate_pixels_mut() {
            *p = Rgba([10, 20, 30, 255]);
        }
        let mask = GrayImage::new(64, 64); // all zero
        // Empty mask short-circuits BEFORE the model-load check, so this
        // works even when LaMa isn't available.
        let result = process_inpaint(&img, &mask, prunr_models::ModelId::LaMaFp32)
            .expect("empty mask is no-op");
        assert_eq!(result.as_raw(), img.as_raw());
    }

    #[test]
    fn pick_input_names_matches_by_keyword() {
        let names = vec!["image".to_string(), "mask".to_string()];
        let (img, msk) = pick_input_names(&names);
        assert_eq!(img, "image");
        assert_eq!(msk, "mask");
    }

    #[test]
    fn pick_input_names_case_insensitive() {
        let names = vec!["IMAGE_INPUT".to_string(), "Mask_Input".to_string()];
        let (img, msk) = pick_input_names(&names);
        assert_eq!(img, "IMAGE_INPUT");
        assert_eq!(msk, "Mask_Input");
    }

    #[test]
    fn pick_input_names_falls_back_to_positional() {
        // No keyword match — first input becomes image, second becomes mask.
        let names = vec!["input_0".to_string(), "input_1".to_string()];
        let (img, msk) = pick_input_names(&names);
        assert_eq!(img, "input_0");
        assert_eq!(msk, "input_1");
    }

    #[test]
    fn pick_input_names_handles_swapped_order() {
        // Mask appears before image in the export — keyword match still
        // assigns them correctly.
        let names = vec!["the_mask".to_string(), "the_image".to_string()];
        let (img, msk) = pick_input_names(&names);
        assert_eq!(img, "the_image");
        assert_eq!(msk, "the_mask");
    }

    #[test]
    fn grow_mask_zero_is_identity() {
        let mut m = GrayImage::new(8, 8);
        m.put_pixel(4, 4, Luma([255]));
        let out = grow_mask(&m, 0);
        assert_eq!(out.as_raw(), m.as_raw());
    }

    #[test]
    fn grow_mask_dilates_one_pixel() {
        let mut m = GrayImage::new(5, 5);
        m.put_pixel(2, 2, Luma([255]));
        let out = grow_mask(&m, 1);
        // Center stays + 4-connected neighbours light up.
        assert_eq!(out.get_pixel(2, 2).0[0], 255);
        assert_eq!(out.get_pixel(1, 2).0[0], 255);
        assert_eq!(out.get_pixel(3, 2).0[0], 255);
        assert_eq!(out.get_pixel(2, 1).0[0], 255);
        assert_eq!(out.get_pixel(2, 3).0[0], 255);
        // Diagonal NOT dilated under 4-connectivity.
        assert_eq!(out.get_pixel(1, 1).0[0], 0);
    }

    #[test]
    fn grow_mask_erodes_negative() {
        // 3×3 filled block surrounded by zero. Erosion strips the
        // boundary down to a single pixel.
        let mut m = GrayImage::new(5, 5);
        for y in 1..=3 { for x in 1..=3 { m.put_pixel(x, y, Luma([255])); } }
        let out = grow_mask(&m, -1);
        // Only the centre survives — corners and edges of the 3×3 had
        // a zero neighbour.
        assert_eq!(out.get_pixel(2, 2).0[0], 255);
        assert_eq!(out.get_pixel(1, 1).0[0], 0);
        assert_eq!(out.get_pixel(1, 2).0[0], 0);
    }

    #[test]
    fn feather_inpainted_zero_is_identity() {
        let inp = RgbaImage::from_pixel(8, 8, Rgba([200, 100, 50, 255]));
        let src = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 255]));
        let mask = GrayImage::from_pixel(8, 8, Luma([255]));
        let out = feather_inpainted(&inp, &src, &mask, 0.0);
        assert_eq!(out.as_raw(), inp.as_raw(), "feather=0 must be identity");
    }

    #[test]
    fn feather_inpainted_blends_at_edge() {
        // Solid mask in the centre; pixel at the boundary should land
        // somewhere between source and inpainted.
        let inp = RgbaImage::from_pixel(8, 8, Rgba([200, 200, 200, 255]));
        let src = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 255]));
        let mut mask = GrayImage::new(8, 8);
        for y in 2..=5 { for x in 2..=5 { mask.put_pixel(x, y, Luma([255])); } }
        let out = feather_inpainted(&inp, &src, &mask, 4.0);
        // (2, 2) is at the mask edge — distance 1 from boundary.
        let edge = out.get_pixel(2, 2).0[0];
        assert!(edge > 0 && edge < 200,
            "edge pixel must be blended between source(0) and inpainted(200), got {edge}");
        // (3, 3) is one in from the edge — distance 2.
        let mid = out.get_pixel(3, 3).0[0];
        assert!(mid > edge,
            "deeper-in pixel must be closer to inpainted, got mid={mid} edge={edge}");
    }

    #[test]
    fn feather_inpainted_outside_mask_is_source() {
        let inp = RgbaImage::from_pixel(4, 4, Rgba([200, 200, 200, 255]));
        let src = RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]));
        let mask = GrayImage::new(4, 4); // all zero
        let out = feather_inpainted(&inp, &src, &mask, 4.0);
        for p in out.pixels() {
            assert_eq!(p.0, [10, 20, 30, 255]);
        }
    }

    #[test]
    fn decode_tile_unmasked_passes_through_source() {
        // Output buffer is irrelevant for unmasked pixels — they must
        // come from the source, byte-identical (alpha included).
        let mut src = RgbaImage::new(64, 64);
        for (i, p) in src.pixels_mut().enumerate() {
            *p = Rgba([i as u8, (i >> 1) as u8, (i >> 2) as u8, 200]);
        }
        let mask = GrayImage::new(64, 64); // all zero ⇒ all unmasked
        let s = TILE as usize;
        // Fill output with junk to prove we don't read it.
        let mut output = Array4::<f32>::zeros((1, 3, s, s));
        for v in output.iter_mut() { *v = 0.5; }
        let out = decode_tile(&output, &src, &mask, 64, 64);
        assert_eq!(out.as_raw(), src.as_raw(), "unmasked pixels must equal source");
    }

    #[test]
    fn decode_tile_masked_writes_inpainted_pixels() {
        let src = RgbaImage::from_pixel(8, 8, Rgba([10, 20, 30, 255]));
        let mut mask = GrayImage::new(8, 8);
        mask.put_pixel(4, 4, Luma([255]));
        let s = TILE as usize;
        let mut output = Array4::<f32>::zeros((1, 3, s, s));
        // Pretend LaMa returned R=255 G=0 B=128 for the masked pixel
        // (in [0, 1] range).
        output[[0, 0, 4, 4]] = 1.0;
        output[[0, 1, 4, 4]] = 0.0;
        output[[0, 2, 4, 4]] = 128.0 / 255.0;
        let out = decode_tile(&output, &src, &mask, 8, 8);
        let p = out.get_pixel(4, 4);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 128);
        assert_eq!(p[3], 255, "alpha always copied from source");
        // Surrounding pixel stays source.
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30, 255]);
    }

    #[test]
    fn decode_tile_detects_0_255_range() {
        // Output already in [0, 255] range — no rescaling.
        let src = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 255]));
        let mut mask = GrayImage::from_pixel(8, 8, Luma([255]));
        mask.put_pixel(0, 0, Luma([255]));
        let s = TILE as usize;
        let mut output = Array4::<f32>::zeros((1, 3, s, s));
        // Big values across the whole tile so the heuristic picks 1.0 scale.
        for v in output.iter_mut() { *v = 200.0; }
        let out = decode_tile(&output, &src, &mask, 8, 8);
        assert_eq!(out.get_pixel(4, 4).0[0], 200, "200.0 in [0,255] range stays 200");
    }

    #[test]
    fn encode_tile_pads_small_inputs_to_tile_size() {
        let img = RgbaImage::new(64, 64);
        let mut mask = GrayImage::new(64, 64);
        mask.put_pixel(32, 32, Luma([255]));
        let (img_t, msk_t) = encode_tile(&img, &mask);
        assert_eq!(img_t.shape(), &[1, 3, TILE as usize, TILE as usize]);
        assert_eq!(msk_t.shape(), &[1, 1, TILE as usize, TILE as usize]);
        // The painted mask cell maps to NCHW[0,0,32,32].
        assert_eq!(msk_t[[0, 0, 32, 32]], 1.0);
        // Outside the source region, mask stays 0 (zero-padded).
        assert_eq!(msk_t[[0, 0, 200, 200]], 0.0);
    }

}
