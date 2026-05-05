//! LaMa-based inpainting. Public entry: `process_inpaint(image, mask)`.
//! Tiles 512-px with 64-px feathered overlap so large images stay
//! inside LaMa's fixed input size and seams stay invisible against
//! flat backgrounds.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use image::{GrayImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::{Session, builder::GraphOptimizationLevel}, value::Tensor};

use crate::engine::{apply_ort_graph_cache, EpKind};
use crate::types::CoreError;

/// Cross-thread progress channel for an in-flight inpaint stroke.
///
/// The worker writes `current` between scheduler steps (SD UNet) or
/// at tile boundaries (LaMa); the GUI reads on its render thread to
/// show "Erasing — step N of M". Writes are `Release` and reads are
/// `Acquire` so a banner in the middle of a frame never sees a
/// partial update. `total == 0` means "indeterminate" — the banner
/// falls back to the spinner-only form.
#[derive(Debug, Default)]
pub struct InpaintProgress {
    pub current: AtomicU32,
    pub total: AtomicU32,
}

impl InpaintProgress {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_total(&self, total: u32) {
        self.total.store(total, Ordering::Release);
    }
    pub fn set_step(&self, step: u32) {
        self.current.store(step, Ordering::Release);
    }
    /// Returns `(current, total)`. `(0, 0)` means no progress yet.
    pub fn read(&self) -> (u32, u32) {
        (self.current.load(Ordering::Acquire), self.total.load(Ordering::Acquire))
    }
}

/// Cross-cutting hooks for an in-flight inpaint stroke.
/// Bundles `cancel` + `progress` so callers don't grow `process_inpaint_with`
/// past the 6-param alarm as more hooks land.
#[derive(Default)]
pub struct InpaintHooks {
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub progress: Option<std::sync::Arc<InpaintProgress>>,
}

/// LaMa input/output side length in pixels.
pub const TILE: u32 = 512;
/// Overlap between adjacent tiles in pixels. Half the tile to stay safe
/// on the corners; the feather blend tapers within this region.
pub const OVERLAP: u32 = 64;

/// 5-tap separable Gaussian for `sharpen_inpainted`'s unsharp-mask blur
/// (sigma ≈ 1.0). Constant + derived margin lock the kernel↔bbox
/// invariant: `mask_bbox` must expand by `kernel.len() / 2` so the
/// clamped reads at bbox edges land on the same source pixels the
/// full-image variant would have hit.
const SHARPEN_KERNEL: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
const SHARPEN_MARGIN: u32 = (SHARPEN_KERNEL.len() / 2) as u32;

/// Region of interest in pixel coordinates. Distinct from `TilePlacement`
/// (which is a planned-tile-of-work); a `Bbox` is just a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Bbox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Eagerly initialise the inpaint session for `id` in the background so
/// the first stroke doesn't pay the ~5-10s session-build latency.
/// Idempotent — repeated calls for the same id are no-ops.
pub fn prewarm(id: prunr_models::ModelId) -> Result<(), CoreError> {
    if id.is_sd_family() {
        return crate::inpaint_sd::prewarm(id);
    }
    LamaSession::get(id).map(|_| ())
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
/// small radii exposed (≤32 px) and O(n) on the bbox.
///
/// RAM: chamfer + per-pixel walk are bbox-restricted with a
/// `ceil(feather_px) + 1` margin. The `+1` guarantees a complete ring
/// of mask=0 chamfer seeds *inside* the cropped image even when
/// `ceil(feather_px) == 0` rounds to zero margin elsewhere. Without
/// it a mask=255 pixel at the cropped-image edge would have no seed
/// to chamfer from. The `tile_compose` / SD pipelines preserve source
/// byte-for-byte at mask=0, so the original "always copy src" branch
/// outside the bbox is redundant.
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
    let margin = feather_px.ceil() as u32 + 1;
    let Some(bbox) = mask_bbox(mask, 1, margin) else {
        return inpainted.clone();
    };
    let crop_mask =
        image::imageops::crop_imm(mask, bbox.x, bbox.y, bbox.w, bbox.h).to_image();
    let dist = chamfer_distance_inside(&crop_mask);

    let (img_w, _) = inpainted.dimensions();
    let img_w_us = img_w as usize;
    let bbox_w_us = bbox.w as usize;
    let mut out = inpainted.clone();
    let out_raw = out.as_mut();
    let src_raw = source.as_raw();
    let inp_raw = inpainted.as_raw();

    // Per-row prefix hoisted out of the inner `bx` loop. Rows are
    // write-disjoint across `by`, so we either parallelise via
    // `SendMutPtr` (mirrors `sharpen_inpainted` and
    // `inpaint_blend::composite_channel`) or fall back to sequential
    // for tiny bboxes where rayon overhead would dominate.
    use rayon::prelude::*;
    #[derive(Clone, Copy)]
    struct SendMutPtr(*mut u8);
    unsafe impl Send for SendMutPtr {}
    unsafe impl Sync for SendMutPtr {}

    let row_step = |by: u32, out_writer: &mut dyn FnMut(usize, [u8; 3])| {
        let global_y = bbox.y + by;
        let row_off_global = global_y as usize * img_w_us;
        let row_off_dist = (by as usize) * bbox_w_us;
        for bx in 0..bbox.w {
            let global_idx = row_off_global + (bbox.x + bx) as usize;
            let pix = global_idx * 4;
            let d = dist[row_off_dist + bx as usize];
            if d <= 0.0 {
                out_writer(pix, [src_raw[pix], src_raw[pix + 1], src_raw[pix + 2]]);
                continue;
            }
            if d >= feather_px {
                continue; // deep inside: keep inpainted
            }
            let t = d / feather_px;
            let s0 = src_raw[pix] as f32;
            let s1 = src_raw[pix + 1] as f32;
            let s2 = src_raw[pix + 2] as f32;
            let p0 = inp_raw[pix] as f32;
            let p1 = inp_raw[pix + 1] as f32;
            let p2 = inp_raw[pix + 2] as f32;
            out_writer(pix, [
                (s0 + t * (p0 - s0)).clamp(0.0, 255.0) as u8,
                (s1 + t * (p1 - s1)).clamp(0.0, 255.0) as u8,
                (s2 + t * (p2 - s2)).clamp(0.0, 255.0) as u8,
            ]);
        }
    };

    if bbox.h >= 64 {
        // Parallel rows. SAFETY: `by` partitions writes by row;
        // `[pix, pix+3)` for any (by, bx) lies in row `global_y`,
        // which is unique to this iteration.
        let dp = SendMutPtr(out_raw.as_mut_ptr());
        (0..bbox.h).into_par_iter().for_each(|by| {
            let mut writer = |pix: usize, rgb: [u8; 3]| unsafe {
                *dp.0.add(pix) = rgb[0];
                *dp.0.add(pix + 1) = rgb[1];
                *dp.0.add(pix + 2) = rgb[2];
            };
            row_step(by, &mut writer);
            // Force whole-struct capture (Rust 2021 disjoint-capture
            // would otherwise grab `dp.0: *mut u8` alone, not Sync).
            let _ = &dp;
        });
    } else {
        for by in 0..bbox.h {
            let mut writer = |pix: usize, rgb: [u8; 3]| {
                out_raw[pix] = rgb[0];
                out_raw[pix + 1] = rgb[1];
                out_raw[pix + 2] = rgb[2];
            };
            row_step(by, &mut writer);
        }
    }
    out
}

/// Forward + backward chamfer pass returning, for each pixel, the
/// distance to the nearest mask==0 pixel (0.0 if outside). Diagonal
/// step uses √2 to keep the metric near-Euclidean.
pub(crate) fn chamfer_distance_inside(mask: &GrayImage) -> Vec<f32> {
    let (w, h) = mask.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;
    let large = (w + h) as f32;
    let mut d: Vec<f32> = mask.as_raw().iter()
        .map(|&v| if v > 127 { large } else { 0.0 })
        .collect();
    const DIAG: f32 = std::f32::consts::SQRT_2;
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
///
/// RAM: working buffers are sized to the **mask bounding box** plus
/// the kernel half-width (2 px) so blur reads from cropped pixels
/// stay bit-identical to a full-image variant. A 200 × 200 stroke on
/// a 4096 × 3072 image holds ~480 KB of f32 instead of ~150 MB.
pub fn sharpen_inpainted(image: &RgbaImage, mask: &GrayImage, amount: f32) -> RgbaImage {
    if amount <= 0.0 {
        return image.clone();
    }
    let amount = amount.min(2.0);
    let (w, _h) = image.dimensions();
    if image.dimensions() != mask.dimensions() {
        tracing::warn!(
            image_dims = ?image.dimensions(),
            mask_dims = ?mask.dimensions(),
            "sharpen_inpainted: dim mismatch, skipping",
        );
        return image.clone();
    }
    // No mask>127 pixels → nothing to sharpen. Bypasses the temp f32
    // allocation entirely (was a ~150 MB no-op on full-image masks at 4K).
    let Some(bbox) = mask_bbox(mask, 128, SHARPEN_MARGIN) else {
        return image.clone();
    };
    let src = image.as_raw();
    let img_w = w as usize;
    let bbox_w = bbox.w as usize;
    let bbox_h = bbox.h as usize;
    let bbox_x = bbox.x as usize;
    let bbox_y = bbox.y as usize;

    // Horizontal pass into a bbox-sized temp f32 buffer (RGB only —
    // alpha never changes). Rows are independent → parallelise.
    let mut tmp: Vec<f32> = vec![0.0; bbox_w * bbox_h * 3];
    use rayon::prelude::*;
    tmp.par_chunks_mut(bbox_w * 3).enumerate().for_each(|(by, row)| {
        let global_y = bbox_y + by;
        for bx in 0..bbox_w {
            let global_x = bbox_x + bx;
            for c in 0..3 {
                let mut sum = 0.0;
                for (k, &kv) in SHARPEN_KERNEL.iter().enumerate() {
                    let xi = (global_x as isize + k as isize - 2)
                        .clamp(0, img_w as isize - 1) as usize;
                    sum += kv * src[(global_y * img_w + xi) * 4 + c] as f32;
                }
                row[bx * 3 + c] = sum;
            }
        }
    });

    // Output starts as image clone. Pixels outside the bbox are
    // mask <= 127 by definition (bbox covers all mask >= 128) — they
    // stay as source. Vertical blur + sharpen overrides bbox-masked
    // pixels. Rows are write-disjoint → parallelise via SendMutPtr,
    // same pattern as `inpaint_blend::composite_channel`.
    let mut out = image.clone();
    let out_raw = out.as_mut();
    let msk_raw = mask.as_raw();
    #[derive(Clone, Copy)]
    struct SendMutPtr(*mut u8);
    unsafe impl Send for SendMutPtr {}
    unsafe impl Sync for SendMutPtr {}
    let dp = SendMutPtr(out_raw.as_mut_ptr());
    (0..bbox_h).into_par_iter().for_each(|by| {
        let global_y = bbox_y + by;
        for bx in 0..bbox_w {
            let global_x = bbox_x + bx;
            let global_idx = global_y * img_w + global_x;
            if msk_raw[global_idx] <= 127 {
                continue;
            }
            let pix = global_idx * 4;
            for c in 0..3 {
                let mut blur = 0.0;
                for (k, &kv) in SHARPEN_KERNEL.iter().enumerate() {
                    // Clamp at bbox bounds — they extend the mask
                    // region by `SHARPEN_MARGIN`, so the clamped index
                    // points at the same source pixel a full-image
                    // variant's clamp would have hit.
                    let by_kernel = (by as isize + k as isize - 2)
                        .clamp(0, bbox_h as isize - 1) as usize;
                    blur += kv * tmp[(by_kernel * bbox_w + bx) * 3 + c];
                }
                let s = src[pix + c] as f32;
                let sharp = s + amount * (s - blur);
                // SAFETY: each parallel iteration `by` writes to bytes
                // at global rows {global_y} only — different `by`s are
                // row-disjoint. Channels (c=0,1,2) within one pixel
                // are also distinct bytes. No race.
                unsafe { *dp.0.add(pix + c) = sharp.clamp(0.0, 255.0) as u8; }
            }
        }
        // Force whole-struct capture of `dp` (Rust 2021 disjoint
        // capture would otherwise grab `dp.0: *mut u8` alone, which
        // isn't Sync — same pattern as `guided_filter::box_filter`).
        let _ = &dp;
    });
    out
}

/// Top-level inpaint entry. `id` selects the inpaint backend
/// (LaMaFp32, BigLaMa, MI-GAN, SD …). Returns the input unchanged when
/// the mask is all-zero (no work). SD-family ids dispatch to the
/// `inpaint_sd` module which has its own multi-model pipeline.
pub fn process_inpaint(image: &RgbaImage, mask: &GrayImage, id: prunr_models::ModelId) -> Result<RgbaImage, CoreError> {
    process_inpaint_with(image, mask, id, None, &InpaintHooks::default())
}

/// Same as `process_inpaint` but takes optional SD-specific tuning
/// (prompt / negative prompt / guidance_scale / steps) and a `hooks`
/// bundle (cancel flag + progress sink). Cancel is checked between
/// LaMa tiles and between SD UNet steps — ORT has no per-op cancel
/// hook so worst-case latency on cancel is one tile (LaMa) or one
/// UNet step (SD). Progress (when supplied) is updated between SD
/// UNet steps; LaMa keeps the spinner-only form because per-tile
/// updates are noisy and most strokes are a single tile.
pub fn process_inpaint_with(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    sd_req: Option<crate::inpaint_sd::SdInpaintRequest>,
    hooks: &InpaintHooks,
) -> Result<RgbaImage, CoreError> {
    use std::sync::atomic::Ordering;
    let cancel = hooks.cancel.as_ref();
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
    if id.is_sd_family() {
        // RAM pre-flight is SD-only: the check probes for the 4+ GB working
        // set the SD pipeline needs. LaMa / MI-GAN require < 1 GB and the
        // sysinfo refresh (~5 ms) costs more than it saves on those paths.
        crate::inpaint_sd::check_ram_for(id).map_err(CoreError::Inference)?;
        let req = sd_req.unwrap_or_else(|| crate::inpaint_sd::SdInpaintRequest {
            num_inference_steps: 20,
            ..Default::default()
        });
        return crate::inpaint_sd::process_inpaint_with(image, mask, id, req, hooks);
    }
    let session = LamaSession::get(id)?;
    let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Acquire));
    let composed = tile_compose(image, mask, |tile_rgba, tile_mask| {
        // When cancelled, short-circuit each remaining tile to a no-op.
        // The outer Cancelled error below discards the partial result.
        if is_cancelled() {
            return tile_rgba.clone();
        }
        session.run_tile(tile_rgba, tile_mask).unwrap_or_else(|e| {
            tracing::error!(%e, "LaMa tile inference failed; leaving tile unchanged");
            tile_rgba.clone()
        })
    })?;
    if is_cancelled() {
        return Err(CoreError::Cancelled);
    }
    Ok(composed)
}

/// One LaMa session per process — see `LamaSession::get`. Tiles run
/// sequentially under one Mutex; do NOT parallelise across tiles
/// (ORT session inference is already multi-threaded internally).
struct LamaSession {
    session: Mutex<Session>,
    image_input_name: String,
    mask_input_name: String,
}

/// Idle-release window for LaMa sessions. Mirrors `SD_IDLE_RELEASE_SECS`
/// in `inpaint_sd.rs`. A 5-minute window balances "user might paint
/// again any second" against "session is hundreds of MB to a few GB
/// after EP compile and the OS should get it back if they switched
/// away or moved on to bg removal."
const LAMA_IDLE_RELEASE_SECS: u64 = 300;

/// Background sweeper interval. 60 s gives reclaim within ~1 minute of
/// the idle threshold.
const LAMA_SWEEP_INTERVAL_SECS: u64 = 60;

/// Per-id deferred session — same shape as `SdBundleSlot`. Outer
/// `Arc<OnceLock>` closes the build race between two concurrent
/// `get()` callers (e.g. a prewarm racing the first stroke).
type LamaSlot = Arc<OnceLock<Result<Arc<LamaSession>, String>>>;
type LamaCache = HashMap<prunr_models::ModelId, crate::inpaint_sd::CacheEntry<LamaSlot>>;

fn lama_cache() -> &'static Mutex<LamaCache> {
    static CACHE: OnceLock<Mutex<LamaCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop every cached LaMa session immediately. In-flight callers keep
/// their own `Arc<LamaSession>` so the cache clear is safe; once the
/// dispatch finishes, the last Arc goes away and the ORT session
/// frees. Called by the GUI when the user switches inpaint models —
/// keeping the previous backend cached for 5 min after they've
/// committed to a different tool wastes ~700 MB–2 GB on the model
/// they're no longer using.
pub fn release_all_lama_sessions() {
    let cache = lama_cache();
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let dropped = guard.len();
    guard.clear();
    if dropped > 0 {
        tracing::info!(
            dropped,
            rss_mb = crate::inpaint_sd::process_rss_mb_pub(),
            "LaMa: cache cleared on model switch",
        );
    }
}

fn ensure_lama_sweeper_running() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        std::thread::Builder::new()
            .name("lama-idle-sweeper".to_string())
            .spawn(|| {
                let interval = Duration::from_secs(LAMA_SWEEP_INTERVAL_SECS);
                let idle = Duration::from_secs(LAMA_IDLE_RELEASE_SECS);
                loop {
                    std::thread::sleep(interval);
                    let cache = lama_cache();
                    let mut guard = cache.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let dropped = crate::inpaint_sd::sweep_idle(
                        &mut guard, Instant::now(), idle,
                    );
                    if dropped > 0 {
                        tracing::info!(
                            dropped,
                            idle_secs = LAMA_IDLE_RELEASE_SECS,
                            rss_mb = crate::inpaint_sd::process_rss_mb_pub(),
                            "LaMa: background sweeper released idle session(s)",
                        );
                    }
                }
            })
            .expect("spawn lama-idle-sweeper thread");
    });
}

impl LamaSession {
    /// Per-id cache. Each LamaSession is built once on first use, kept
    /// in an `Arc`, and released after `LAMA_IDLE_RELEASE_SECS` of no
    /// `get()` calls. Mirrors `SdSession::get` so both inpaint backends
    /// have the same lifecycle: in-flight callers hold their own Arc
    /// (no use-after-free), the cache drops its ref on idle, the OS
    /// reclaims the underlying ORT session.
    ///
    /// Pre-fix this was `Box::leak`'d to a `&'static`, which meant the
    /// session lived for the whole process lifetime — a single LaMa
    /// stroke pinned ~700 MB–2 GB of resident RAM until shutdown.
    fn get(id: prunr_models::ModelId) -> Result<Arc<LamaSession>, CoreError> {
        ensure_lama_sweeper_running();
        let cache = lama_cache();
        let now = Instant::now();
        let idle = Duration::from_secs(LAMA_IDLE_RELEASE_SECS);

        // Cache lock held only for the HashMap lookup and at most one
        // Arc<OnceLock> allocation — never for the seconds-long build.
        let slot: LamaSlot = {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dropped = crate::inpaint_sd::sweep_idle(&mut guard, now, idle);
            if dropped > 0 {
                tracing::info!(
                    dropped,
                    idle_secs = LAMA_IDLE_RELEASE_SECS,
                    rss_mb = crate::inpaint_sd::process_rss_mb_pub(),
                    "LaMa: released idle session(s)",
                );
            }
            let entry = guard.entry(id).or_insert_with(|| crate::inpaint_sd::CacheEntry {
                value: Arc::new(OnceLock::new()),
                last_used: now,
            });
            // Don't refresh last_used on Err entries — sticky failures
            // shouldn't keep refreshing their idle timer (mirrors SD).
            if !matches!(entry.value.get(), Some(Err(_))) {
                entry.last_used = now;
            }
            entry.value.clone()
        };

        // Build outside the cache lock. Concurrent callers block on
        // the OnceLock and receive the same Result — no duplicate
        // session gets built.
        slot.get_or_init(|| Self::new_inner(id).map(Arc::new))
            .clone()
            .map_err(CoreError::Inference)
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

        // Borrow the output view straight into `decode_tile` — no
        // `.to_owned()` between extract and decode. The session lock
        // is held across the closure so the ORT-allocated buffer the
        // view points into stays alive. Saves one 3 MB clone per tile.
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
        let view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("LaMa: output extract: {e}")))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("LaMa: output reshape: {e}")))?;
        Ok(decode_tile(view, image, mask, w, h))
    }
}

impl Drop for LamaSession {
    fn drop(&mut self) {
        tracing::info!(
            rss_mb = crate::inpaint_sd::process_rss_mb_pub(),
            "LaMa session dropped (idle release or process exit)",
        );
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

    for &ep in gpu_eps {
        // Static catalog + dynamic cache, same as engine.rs.
        if !prunr_models::is_ep_compatible(id, ep.as_str()) {
            tracing::debug!(?id, ep = %ep, "LaMa: EP statically incompatible; skipping");
            continue;
        }
        if crate::ep_compat::is_known_failure(ep, id) {
            tracing::debug!(?id, ep = %ep, "LaMa: EP cached as incompatible; skipping");
            continue;
        }
        let builder = match base_builder(threads) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(ep = %ep, %e, "LaMa: builder init failed");
                continue;
            }
        };
        #[allow(unused_mut)] // mut only used on non-macOS via the CUDA arm
        let mut bytes_owner: Cow<'_, [u8]> = Cow::Borrowed(bytes);
        let builder = match ep {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => {
                let (b, owned) = apply_ort_graph_cache(builder, bytes, id, ep.as_str());
                bytes_owner = owned;
                b
            }
            _ => builder,
        };
        let registered = match ep {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ]),
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => {
                let mut p = ort::execution_providers::CoreMLExecutionProvider::default();
                if let Some(dir) = crate::cache::cache_dir_for(id, ep.as_str()) {
                    p = p.with_model_cache_dir(dir.to_string_lossy().into_owned());
                }
                builder.with_execution_providers([p.build()])
            }
            #[cfg(windows)]
            EpKind::DirectMl => builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ]),
            // Default device "AUTO" lets OpenVINO pick the best target
            // (iGPU when present + driver works, NPU on newer Intel,
            // else CPU). Smoke test below catches op-incompat failures.
            // `with_num_threads` caps the EP-internal TBB pool to match
            // our outer rayon budget.
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => {
                let mut p = ort::execution_providers::OpenVINOExecutionProvider::default()
                    .with_num_threads(threads.max(1));
                if let Some(dir) = crate::cache::cache_dir_for(id, ep.as_str()) {
                    p = p.with_cache_dir(dir.to_string_lossy());
                }
                builder.with_execution_providers([p.build()])
            }
        };
        let mut built = match registered {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(ep = %ep, %e, "LaMa: register EP failed");
                continue;
            }
        };
        let started = Instant::now();
        let mut session = match built.commit_from_memory(&bytes_owner) {
            Ok(s) => {
                tracing::info!(?id, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "LaMa: session committed");
                s
            }
            Err(e) => {
                tracing::warn!(ep = %ep, %e, "LaMa: GPU session commit failed — trying next");
                crate::ep_compat::record_failure(ep, id, &format!("{e}"));
                continue;
            }
        };
        match smoke_test_session(&mut session) {
            Ok(()) => {
                tracing::info!(ep = %ep, "LaMa: GPU session validated");
                return Ok((session, ep.as_str().to_string()));
            }
            Err(e) => {
                tracing::warn!(
                    ep = %ep, %e,
                    "LaMa: GPU session smoke test failed — falling back to next EP/CPU",
                );
                crate::ep_compat::record_failure(ep, id, &e);
            }
        }
    }

    // CPU fallback: no EP registration, ORT uses its default CPU provider.
    let builder = base_builder(threads)?;
    let (mut builder, cpu_bytes) = apply_ort_graph_cache(builder, bytes, id, "CPU");
    let started = Instant::now();
    let session = builder
        .commit_from_memory(&cpu_bytes)
        .map_err(|e| format!("LaMa: CPU session commit failed: {e}"))?;
    tracing::info!(?id, ep = "CPU", elapsed_ms = started.elapsed().as_millis() as u64, "LaMa: session committed");
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
    output: ndarray::ArrayView4<'_, f32>,
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
///
/// RAM: accumulators are sized to the **mask bounding box**, not the
/// image. A 200 × 200 stroke on a 4096 × 3072 image holds ~640 KB of
/// f32 accumulators instead of ~250 MB. Pixels outside the bbox are
/// guaranteed mask==0 and so would be skipped by the composite loop
/// anyway — the bbox crop is bit-exact with the full-image variant.
// `pub` to be reachable from `benches/tile_compose.rs`; `#[doc(hidden)]`
// because it's still an internal kernel — callers should use
// `process_inpaint`.
#[doc(hidden)]
pub fn tile_compose<F>(
    image: &RgbaImage,
    mask: &GrayImage,
    mut inpaint_tile: F,
) -> Result<RgbaImage, CoreError>
where
    F: FnMut(&RgbaImage, &GrayImage) -> RgbaImage,
{
    let (w, h) = image.dimensions();
    let mut out = image.clone();

    // No mask → no work. Avoids allocating accumulators for the all-zero
    // case (was a 250 MB allocation on a 4K image even when nothing
    // needed to be inpainted).
    let Some(bbox) = mask_bbox(mask, 1, 0) else {
        return Ok(out);
    };

    let bbox_w = bbox.w as usize;
    let bbox_n = (bbox.w as usize) * (bbox.h as usize);
    // Per-pixel weight accumulator for the feather blend. Tiles overlap
    // in the OVERLAP band; each contributes a smoothstep-weighted
    // sample, normalized at the end. Bbox-sized — see fn docs.
    let mut weight_acc: Vec<f32> = vec![0.0; bbox_n];
    // Accumulator for the weighted RGBA in f32 — blended in linear
    // space, quantised back to u8 at the end.
    let mut color_acc: Vec<[f32; 4]> = vec![[0.0; 4]; bbox_n];

    for tile in plan_tiles(w, h) {
        let tile_mask = image::imageops::crop_imm(mask, tile.x, tile.y, tile.w, tile.h).to_image();
        if mask_is_empty(&tile_mask) {
            continue;
        }
        let tile_rgba = image::imageops::crop_imm(image, tile.x, tile.y, tile.w, tile.h).to_image();
        let painted = inpaint_tile(&tile_rgba, &tile_mask);
        accumulate_tile(&mut color_acc, &mut weight_acc, &painted, &tile, bbox);
    }

    // Resolve accumulated tiles into the bbox region of `out`. Pixels
    // outside the bbox stay byte-identical to source (already cloned
    // into `out`); pixels inside-bbox-but-mask=0 also stay byte-
    // identical (skipped here). Float-blend even on a uniform region
    // would introduce 1-unit u8 quantisation drift (255*w/w ≠ 255 in
    // float math), so the skip on mask==0 is load-bearing.
    let mask_raw = mask.as_raw();
    let img_w = w as usize;
    let out_raw = out.as_mut();
    for by in 0..bbox.h {
        let global_y = bbox.y + by;
        let row_off = global_y as usize * img_w;
        for bx in 0..bbox.w {
            let global_x = bbox.x + bx;
            let global_idx = row_off + global_x as usize;
            if mask_raw[global_idx] == 0 {
                continue;
            }
            let bbox_idx = (by as usize) * bbox_w + (bx as usize);
            let wsum = weight_acc[bbox_idx];
            if wsum > 0.0 {
                let inv = 1.0 / wsum;
                let c = color_acc[bbox_idx];
                // Direct raw-buffer write — saves the bounds-check +
                // Rgba<u8> reconstruction per pixel that
                // `get_pixel_mut` would do. Alpha (offset +3) is left
                // untouched: source clone already filled it, and LaMa
                // is RGB-only.
                let pix = global_idx * 4;
                out_raw[pix]     = (c[0] * inv).clamp(0.0, 255.0) as u8;
                out_raw[pix + 1] = (c[1] * inv).clamp(0.0, 255.0) as u8;
                out_raw[pix + 2] = (c[2] * inv).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// Bounding box of pixels where `mask >= min_value`, expanded by
/// `margin` pixels on each side and clamped to image bounds. `None`
/// for an empty result. Parallel row-reduction so a 4 K mask scans
/// in ~1 ms on multi-core.
///
/// Margin asymmetry at image edges is intentional: when the bbox
/// touches an image edge, that side's margin is clipped to 0. Callers
/// that subsequently `clamp(0, bbox_edge - 1)` their kernel reads
/// produce bit-identical output to a full-image variant — both clamp
/// at the same image edge, just framed differently.
///
/// Two callers in this module today:
/// - `tile_compose`: `min_value=1, margin=0` (any nonzero pixel
///   triggers inference; matches `mask_is_empty`'s convention).
/// - `sharpen_inpainted`: `min_value=128, margin=SHARPEN_MARGIN`
///   (binary mask threshold; margin = kernel half-width so clamped
///   blur reads stay bit-identical to the full-image variant).
pub(crate) fn mask_bbox(mask: &GrayImage, min_value: u8, margin: u32) -> Option<Bbox> {
    use rayon::prelude::*;
    let raw = mask.as_raw();
    let w = mask.width() as usize;
    let h = mask.height() as usize;
    let merged = (0..h)
        .into_par_iter()
        .filter_map(|y| {
            let row = &raw[y * w..(y + 1) * w];
            let mut min_x = usize::MAX;
            let mut max_x = 0usize;
            let mut any = false;
            for (x, &v) in row.iter().enumerate() {
                if v >= min_value {
                    if !any { min_x = x; }
                    max_x = x;
                    any = true;
                }
            }
            if any { Some((min_x, y, max_x, y)) } else { None }
        })
        .reduce_with(|a, b| {
            (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
        })?;
    let (min_x, min_y, max_x, max_y) = merged;
    let m = margin as usize;
    let bx = min_x.saturating_sub(m);
    let by = min_y.saturating_sub(m);
    let bx_end = (max_x + m + 1).min(w);
    let by_end = (max_y + m + 1).min(h);
    Some(Bbox {
        x: bx as u32,
        y: by as u32,
        w: (bx_end - bx) as u32,
        h: (by_end - by) as u32,
    })
}

fn accumulate_tile(
    color_acc: &mut [[f32; 4]],
    weight_acc: &mut [f32],
    tile: &RgbaImage,
    placement: &TilePlacement,
    bbox: Bbox,
) {
    // Hoist tile↔bbox intersection out of the per-pixel loop. For a
    // small stroke far from this tile's edge, this skips ~62 K
    // per-iteration `if outside bbox: continue` branches.
    let bbox_x_end = bbox.x + bbox.w;
    let bbox_y_end = bbox.y + bbox.h;
    let tile_x_end = placement.x + placement.w;
    let tile_y_end = placement.y + placement.h;
    if placement.x >= bbox_x_end || placement.y >= bbox_y_end
        || tile_x_end <= bbox.x || tile_y_end <= bbox.y
    {
        return; // tile and bbox don't overlap — nothing to accumulate
    }
    let tx_start = bbox.x.saturating_sub(placement.x);
    let ty_start = bbox.y.saturating_sub(placement.y);
    let tx_end = (bbox_x_end - placement.x).min(placement.w);
    let ty_end = (bbox_y_end - placement.y).min(placement.h);

    let bbox_w = bbox.w as usize;
    let tile_raw = tile.as_raw();
    let tile_w = tile.width() as usize;
    for ty in ty_start..ty_end {
        let by = ((placement.y + ty) - bbox.y) as usize;
        let dist_top = ty;
        let dist_bottom = placement.h - 1 - ty;
        let edge_y = dist_top.min(dist_bottom);
        let tile_row = ty as usize * tile_w;
        for tx in tx_start..tx_end {
            let bx = ((placement.x + tx) - bbox.x) as usize;
            let dist_left = tx;
            let dist_right = placement.w - 1 - tx;
            let edge_x = dist_left.min(dist_right);
            let w = feather_weight(edge_x.min(edge_y));
            let dst_idx = by * bbox_w + bx;
            // Direct raw indexing — saves the bounds-check + Rgba<u8>
            // reconstruction `tile.get_pixel(tx, ty).0` would do.
            let p_off = (tile_row + tx as usize) * 4;
            let acc = &mut color_acc[dst_idx];
            acc[0] += tile_raw[p_off]     as f32 * w;
            acc[1] += tile_raw[p_off + 1] as f32 * w;
            acc[2] += tile_raw[p_off + 2] as f32 * w;
            acc[3] += tile_raw[p_off + 3] as f32 * w;
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
    fn feather_inpainted_outside_bbox_keeps_inpainted() {
        // Bbox-feather only operates on the mask bbox + margin. For
        // mask=0 pixels OUTSIDE that region, output stays as inpainted
        // (the upstream pipelines — `tile_compose` / SD — preserve
        // source byte-for-byte at mask=0, so the original "always
        // copy src" branch was redundant for outside-bbox pixels).
        let inp = RgbaImage::from_pixel(4, 4, Rgba([200, 200, 200, 255]));
        let src = RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]));
        let mask = GrayImage::new(4, 4); // all zero ⇒ no bbox at all
        let out = feather_inpainted(&inp, &src, &mask, 4.0);
        // Empty mask short-circuits to `inpainted.clone()`.
        assert_eq!(out, inp);
    }

    /// Inside the bbox, mask=0 pixels still get the defensive copy
    /// (that branch is load-bearing for any pipeline that drifts
    /// inside the bbox region — e.g. a low-alpha brush stroke pixel
    /// whose neighbour is mask>=128).
    #[test]
    fn feather_inpainted_inside_bbox_mask_zero_pixels_get_source() {
        // Mask geometry: single mask=255 pixel at (8, 8). With
        // feather_px=4.0 → margin=ceil(4)+1=5 → bbox = (3, 3, 11, 11).
        // The probe pixel (7, 8) sits inside the bbox at offset (4, 5),
        // mask=0 → defensive copy fires.
        let mut inp = RgbaImage::from_pixel(16, 16, Rgba([200, 200, 200, 255]));
        let src = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 255]));
        let mut mask = GrayImage::new(16, 16);
        mask.put_pixel(8, 8, Luma([255]));
        inp.put_pixel(7, 8, Rgba([200, 200, 200, 255]));
        let out = feather_inpainted(&inp, &src, &mask, 4.0);
        assert_eq!(out.get_pixel(7, 8).0, [10, 20, 30, 255]);
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
        let out = decode_tile(output.view(), &src, &mask, 64, 64);
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
        let out = decode_tile(output.view(), &src, &mask, 8, 8);
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
        let out = decode_tile(output.view(), &src, &mask, 8, 8);
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

    #[test]
    fn mask_bbox_empty_returns_none() {
        let mask = GrayImage::new(64, 64);
        assert!(mask_bbox(&mask, 1, 0).is_none());
        assert!(mask_bbox(&mask, 128, 2).is_none());
    }

    #[test]
    fn mask_bbox_single_pixel_tight_box_no_margin() {
        let mut mask = GrayImage::new(64, 64);
        mask.put_pixel(10, 20, Luma([1]));
        let bbox = mask_bbox(&mask, 1, 0).expect("non-empty");
        assert_eq!(bbox, Bbox { x: 10, y: 20, w: 1, h: 1 });
    }

    #[test]
    fn mask_bbox_single_pixel_with_margin_clamps_to_image() {
        let mut mask = GrayImage::new(32, 32);
        // Hit the corner — margin 4 on a corner pixel must clamp.
        mask.put_pixel(0, 0, Luma([255]));
        let bbox = mask_bbox(&mask, 128, 4).expect("non-empty");
        assert_eq!(bbox, Bbox { x: 0, y: 0, w: 5, h: 5 });
    }

    #[test]
    fn mask_bbox_full_image_no_margin() {
        let mask = GrayImage::from_pixel(32, 32, Luma([255]));
        let bbox = mask_bbox(&mask, 1, 0).expect("non-empty");
        assert_eq!(bbox, Bbox { x: 0, y: 0, w: 32, h: 32 });
    }

    #[test]
    fn mask_bbox_threshold_separates_low_from_high() {
        // A pixel with value 100 counts at threshold=1 but NOT at 128.
        let mut mask = GrayImage::new(32, 32);
        mask.put_pixel(10, 10, Luma([100]));
        assert!(mask_bbox(&mask, 1, 0).is_some(), "low value present at threshold 1");
        assert!(mask_bbox(&mask, 128, 0).is_none(), "low value absent at threshold 128");
    }

    #[test]
    fn tile_compose_empty_mask_is_byte_identical_to_source() {
        // The bbox optimisation early-returns on empty masks. Output
        // must be byte-identical to source — no allocation, no float
        // round-trip.
        let mut img = RgbaImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                img.put_pixel(x, y, Rgba([x as u8, y as u8, 100, 255]));
            }
        }
        let mask = GrayImage::new(128, 128); // all-zero
        let out = tile_compose(&img, &mask, |t, _| t.clone()).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn sharpen_inpainted_zero_amount_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 255]));
        let mut mask = GrayImage::new(16, 16);
        mask.put_pixel(8, 8, Luma([255]));
        let out = sharpen_inpainted(&img, &mask, 0.0);
        assert_eq!(out, img);
    }

    #[test]
    fn sharpen_inpainted_empty_mask_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 255]));
        let mask = GrayImage::new(16, 16);
        let out = sharpen_inpainted(&img, &mask, 0.5);
        assert_eq!(out, img);
    }

    #[test]
    fn sharpen_inpainted_dim_mismatch_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 255]));
        let mask = GrayImage::new(8, 8);
        let out = sharpen_inpainted(&img, &mask, 0.5);
        assert_eq!(out, img);
    }

    #[test]
    fn sharpen_inpainted_unmasked_pixels_byte_identical_to_source() {
        // Bbox-restricted sharpen must leave every mask <= 127 pixel
        // byte-identical to source — Gemini's "silent seam" risk lives
        // here. We use a high-frequency checkerboard so any leakage of
        // the blur outside the mask would be obvious.
        let mut img = RgbaImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                img.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        let mut mask = GrayImage::new(64, 64);
        for y in 24..40 {
            for x in 24..40 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = sharpen_inpainted(&img, &mask, 0.7);
        for y in 0..64 {
            for x in 0..64 {
                if mask.get_pixel(x, y).0[0] > 127 {
                    continue;
                }
                assert_eq!(
                    out.get_pixel(x, y), img.get_pixel(x, y),
                    "unmasked pixel ({x}, {y}) must be byte-identical to source",
                );
            }
        }
    }

    /// Reference: re-implement the original full-image algorithm
    /// inline so we can compare bbox output to it pixel-for-pixel.
    /// Intentional duplication — this is the contract that bbox
    /// preserves; copy-paste is the right pattern for a regression
    /// test that locks the math.
    #[cfg(test)]
    fn reference_sharpen(image: &RgbaImage, mask: &GrayImage, amount: f32) -> RgbaImage {
        let kernel: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
        let (w, h) = image.dimensions();
        let src = image.as_raw();
        let w_us = w as usize;
        let h_us = h as usize;
        let mut tmp: Vec<f32> = vec![0.0; w_us * h_us * 3];
        for y in 0..h_us {
            for x in 0..w_us {
                for c in 0..3 {
                    let mut sum = 0.0;
                    for (k, &kv) in kernel.iter().enumerate() {
                        let xi = (x as isize + k as isize - 2)
                            .clamp(0, w_us as isize - 1) as usize;
                        sum += kv * src[(y * w_us + xi) * 4 + c] as f32;
                    }
                    tmp[(y * w_us + x) * 3 + c] = sum;
                }
            }
        }
        let mut out = RgbaImage::new(w, h);
        let out_raw = out.as_mut();
        let msk = mask.as_raw();
        for y in 0..h_us {
            for x in 0..w_us {
                let pix = (y * w_us + x) * 4;
                out_raw[pix + 3] = src[pix + 3];
                if msk[y * w_us + x] <= 127 {
                    out_raw[pix] = src[pix];
                    out_raw[pix + 1] = src[pix + 1];
                    out_raw[pix + 2] = src[pix + 2];
                    continue;
                }
                for c in 0..3 {
                    let mut blur = 0.0;
                    for (k, &kv) in kernel.iter().enumerate() {
                        let yi = (y as isize + k as isize - 2)
                            .clamp(0, h_us as isize - 1) as usize;
                        blur += kv * tmp[(yi * w_us + x) * 3 + c];
                    }
                    let s = src[pix + c] as f32;
                    let sharp = s + amount * (s - blur);
                    out_raw[pix + c] = sharp.clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }

    #[cfg(test)]
    fn assert_bbox_sharpen_matches_reference(img: &RgbaImage, mask: &GrayImage, amount: f32) {
        let bbox_out = sharpen_inpainted(img, mask, amount);
        let ref_out = reference_sharpen(img, mask, amount);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let a = bbox_out.get_pixel(x, y);
                let b = ref_out.get_pixel(x, y);
                assert_eq!(a, b, "({x}, {y}) diverged: bbox={a:?} ref={b:?}");
            }
        }
    }

    /// Reference: the original full-image feather algorithm, inlined
    /// for regression testing. Same intentional-duplication pattern
    /// as `reference_sharpen`.
    #[cfg(test)]
    fn reference_feather(
        inpainted: &RgbaImage,
        source: &RgbaImage,
        mask: &GrayImage,
        feather_px: f32,
    ) -> RgbaImage {
        if feather_px <= 0.0 { return inpainted.clone(); }
        let dist = chamfer_distance_inside(mask);
        let mut out = inpainted.clone();
        let out_raw = out.as_mut();
        let src = source.as_raw();
        let inp = inpainted.as_raw();
        for (i, &d) in dist.iter().enumerate() {
            let pix = i * 4;
            if d <= 0.0 {
                out_raw[pix] = src[pix];
                out_raw[pix + 1] = src[pix + 1];
                out_raw[pix + 2] = src[pix + 2];
                continue;
            }
            if d >= feather_px { continue; }
            let t = d / feather_px;
            for c in 0..3 {
                let s = src[pix + c] as f32;
                let p = inp[pix + c] as f32;
                out_raw[pix + c] = (s + t * (p - s)).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Build a (source, inpainted) pair where `inpainted == source`
    /// at every mask=0 pixel — matches what the real pipelines
    /// produce (`tile_compose`'s composite loop guarantees this for
    /// LaMa). With this invariant, bbox-feather and full-image
    /// reference produce bit-identical output.
    #[cfg(test)]
    fn make_realistic_inpaint_pair(w: u32, h: u32, mask: &GrayImage) -> (RgbaImage, RgbaImage) {
        let source = gradient_image(w, h);
        let mut inpainted = source.clone();
        // Only alter pixels inside the mask region — mimics LaMa output.
        for y in 0..h {
            for x in 0..w {
                if mask.get_pixel(x, y).0[0] > 0 {
                    let p = source.get_pixel(x, y).0;
                    let inv = Rgba([255 - p[0], 255 - p[1], 255 - p[2], p[3]]);
                    inpainted.put_pixel(x, y, inv);
                }
            }
        }
        (source, inpainted)
    }

    #[cfg(test)]
    fn assert_bbox_feather_matches_reference(
        source: &RgbaImage, inpainted: &RgbaImage,
        mask: &GrayImage, feather_px: f32,
    ) {
        let bbox_out = feather_inpainted(inpainted, source, mask, feather_px);
        let ref_out = reference_feather(inpainted, source, mask, feather_px);
        for y in 0..source.height() {
            for x in 0..source.width() {
                let a = bbox_out.get_pixel(x, y);
                let b = ref_out.get_pixel(x, y);
                assert_eq!(a, b, "({x}, {y}) diverged: bbox={a:?} ref={b:?}");
            }
        }
    }

    #[test]
    fn feather_inpainted_zero_px_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 200]));
        let mut mask = GrayImage::new(16, 16);
        mask.put_pixel(8, 8, Luma([255]));
        let out = feather_inpainted(&img, &img, &mask, 0.0);
        assert_eq!(out, img);
    }

    #[test]
    fn feather_inpainted_empty_mask_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 200]));
        let mask = GrayImage::new(16, 16);
        let out = feather_inpainted(&img, &img, &mask, 4.0);
        assert_eq!(out, img);
    }

    #[test]
    fn feather_inpainted_dim_mismatch_is_clone() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([10, 20, 30, 200]));
        let mask = GrayImage::new(8, 8);
        let out = feather_inpainted(&img, &img, &mask, 4.0);
        assert_eq!(out, img);
    }

    /// Bit-exactness — corner mask exercises the boundary clamp.
    #[test]
    fn feather_inpainted_matches_reference_corner_mask() {
        let mut mask = GrayImage::new(48, 48);
        for y in 0..16 { for x in 0..16 { mask.put_pixel(x, y, Luma([255])); } }
        let (src, inp) = make_realistic_inpaint_pair(48, 48, &mask);
        assert_bbox_feather_matches_reference(&src, &inp, &mask, 6.0);
    }

    /// Bit-exactness — interior mask exercises the pure-bbox path.
    #[test]
    fn feather_inpainted_matches_reference_interior_mask() {
        let mut mask = GrayImage::new(64, 64);
        for y in 24..40 { for x in 24..40 { mask.put_pixel(x, y, Luma([255])); } }
        let (src, inp) = make_realistic_inpaint_pair(64, 64, &mask);
        assert_bbox_feather_matches_reference(&src, &inp, &mask, 8.0);
    }

    fn gradient_image(w: u32, h: u32) -> RgbaImage {
        let mut img = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 5 + y * 3) as u8;
                let g = (x * 7) as u8;
                let b = (y * 11) as u8;
                img.put_pixel(x, y, Rgba([r, g, b, 200]));
            }
        }
        img
    }

    /// Bit-exactness — corner-touching mask (exercises asymmetric
    /// boundary clamp where margin runs into the image edge).
    #[test]
    fn sharpen_inpainted_matches_reference_full_image_computation() {
        let img = gradient_image(48, 48);
        let mut mask = GrayImage::new(48, 48);
        for y in 0..16 {
            for x in 0..16 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        assert_bbox_sharpen_matches_reference(&img, &mask, 0.5);
    }

    /// Bit-exactness — interior mask with margin on every side
    /// (exercises the pure-bbox path with no edge clamping).
    #[test]
    fn sharpen_inpainted_matches_reference_interior_mask() {
        let img = gradient_image(64, 64);
        let mut mask = GrayImage::new(64, 64);
        // 16×16 mask centred at (32, 32), 16 px from any image edge —
        // bbox + margin (2) is still well inside, no boundary clamp.
        for y in 24..40 {
            for x in 24..40 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        assert_bbox_sharpen_matches_reference(&img, &mask, 0.7);
    }

    /// Bit-exactness — mask spanning the right + bottom edges
    /// simultaneously (exercises clamps on two adjacent edges).
    #[test]
    fn sharpen_inpainted_matches_reference_two_edge_mask() {
        let img = gradient_image(48, 48);
        let mut mask = GrayImage::new(48, 48);
        for y in 36..48 {
            for x in 36..48 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        assert_bbox_sharpen_matches_reference(&img, &mask, 0.5);
    }

    #[test]
    fn tile_compose_small_mask_preserves_unmasked_pixels_exactly() {
        // Stroke is a 16×16 region in the centre. Every pixel outside
        // the stroke must equal the source byte-for-byte (unchanged
        // by the bbox crop or by tile feathering).
        let mut img = RgbaImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                img.put_pixel(x, y, Rgba([x as u8, y as u8, 100, 255]));
            }
        }
        let mut mask = GrayImage::new(128, 128);
        for y in 56..72 {
            for x in 56..72 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        // Inpaint closure paints the masked region pure red.
        let out = tile_compose(&img, &mask, |tile, tile_mask| {
            let mut t = tile.clone();
            for y in 0..tile.height() {
                for x in 0..tile.width() {
                    if tile_mask.get_pixel(x, y).0[0] > 0 {
                        t.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                    }
                }
            }
            t
        }).unwrap();
        // Spot-check: every pixel outside the masked region is identical.
        for y in 0..128 {
            for x in 0..128 {
                if mask.get_pixel(x, y).0[0] != 0 {
                    continue;
                }
                assert_eq!(
                    out.get_pixel(x, y), img.get_pixel(x, y),
                    "unmasked pixel ({x}, {y}) must be unchanged",
                );
            }
        }
        // Masked centre pixel is roughly red (feathered tiles average).
        let centre = out.get_pixel(64, 64).0;
        assert!(centre[0] > 200, "centre red channel high, got {centre:?}");
        assert!(centre[1] < 50, "centre green channel low, got {centre:?}");
    }
}
