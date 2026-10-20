//! LaMa-based inpainting. Tiles 512 px with a 64 px feathered overlap so
//! large images fit LaMa's fixed input size without visible seams.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use image::{GrayImage, RgbaImage};
use ndarray::Array4;
use ort::{inputs, session::{Session, builder::GraphOptimizationLevel}, value::Tensor};

use crate::engine::{apply_ort_graph_cache, cached_graph, EpKind, EpSite, Miss, SessionPick, Weights};
use crate::progress::{Progress, Step, TileRect, Unit};
use crate::types::CoreError;

/// LaMa input/output side length in pixels.
pub const TILE: u32 = 512;
/// Overlap between adjacent tiles in pixels: context both tiles share,
/// which the feather blend tapers across.
pub const OVERLAP: u32 = 64;

/// 5-tap separable Gaussian (sigma ≈ 1.0). The bbox must expand by
/// `kernel.len() / 2` so clamped reads at its edges hit the same source
/// pixels a full-image pass would.
const SHARPEN_KERNEL: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
const SHARPEN_MARGIN: u32 = (SHARPEN_KERNEL.len() / 2) as u32;

/// Region of interest in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Bbox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Bbox {
    pub(crate) fn area(&self) -> usize {
        (self.w as usize) * (self.h as usize)
    }

    /// This box clipped to `rect` (x, y, w, h), which it must overlap.
    pub(crate) fn within(self, (x, y, w, h): (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
        let (x0, y0) = (self.x.max(x), self.y.max(y));
        let (x1, y1) = ((self.x + self.w).min(x + w), (self.y + self.h).min(y + h));
        debug_assert!(x1 > x0 && y1 > y0, "{self:?} does not overlap {:?}", (x, y, w, h));
        (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }
}

/// Eagerly initialise the inpaint session for `id` in the background so
/// the first stroke doesn't pay its build: ~5-10 s for LaMa, 10-30 s for
/// the four sessions of a Stable Diffusion bundle (built at the default
/// tuning). Idempotent.
pub fn prewarm(id: prunr_models::ModelId, cpu_only: bool) -> Result<(), CoreError> {
    if id.is_sd_family() {
        crate::inpaint_sd::load_bundle(id, &crate::inpaint_sd::SdTuning::default()).map(drop)
    } else {
        load_lama(id, cpu_only).map(drop)
    }
}

/// Dilate (px > 0) or erode (px < 0) a binary mask by `|px|` iterations
/// of 4-connected morphology — fine for the small values the GUI exposes.
///
/// Works on the mask's bbox (plus `px` when dilating): nothing farther
/// than `px` from the mask can change, so the result is bit-identical
/// to a full-image pass. Peak RAM: the output plus two bbox buffers.
pub fn grow_mask(mask: &GrayImage, px: i32) -> GrayImage {
    if px == 0 {
        return mask.clone();
    }
    let (w, h) = mask.dimensions();
    let iters = px.unsigned_abs();
    let dilate = px > 0;
    let mut out = GrayImage::new(w, h);
    let Some(b) = mask_bbox(mask, 128, if dilate { iters } else { 0 }) else {
        return out;
    };
    let mut current = image::imageops::crop_imm(mask, b.x, b.y, b.w, b.h).to_image();
    let mut next = GrayImage::new(b.w, b.h);
    for _ in 0..iters {
        morphology_step_into(&current, &mut next, dilate);
        std::mem::swap(&mut current, &mut next);
    }
    image::imageops::replace(&mut out, &current, b.x as i64, b.y as i64);
    out
}

fn morphology_step_into(mask: &GrayImage, out: &mut GrayImage, dilate: bool) {
    let (w, h) = mask.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;
    let raw = mask.as_raw();
    let buf = out.as_mut();
    for y in 0..h_us {
        for x in 0..w_us {
            // Out of bounds counts as zero, so erosion shrinks at the
            // image boundary.
            let center = raw[y * w_us + x] > 127;
            let n = y > 0 && raw[(y - 1) * w_us + x] > 127;
            let s = y + 1 < h_us && raw[(y + 1) * w_us + x] > 127;
            let e = x + 1 < w_us && raw[y * w_us + x + 1] > 127;
            let we = x > 0 && raw[y * w_us + x - 1] > 127;
            let bit = if dilate { center || n || s || e || we } else { center && n && s && e && we };
            buf[y * w_us + x] = if bit { 255 } else { 0 };
        }
    }
}

/// For each pixel, the distance to the nearest mask==0 pixel (0.0 if
/// outside).
pub(crate) fn chamfer_distance_inside(mask: &GrayImage) -> Vec<f32> {
    let outside: Vec<bool> = mask.as_raw().iter().map(|&v| v <= 127).collect();
    crate::math::distance_to_nearest(&outside, mask.width() as usize, mask.height() as usize)
}

/// Unsharp-mask the inpainted region (`mask > 127`) to counter LaMa's
/// softness on high-frequency texture.
///
/// `amount` in [0.0, 2.0]; 0.0 leaves `image` unchanged. 0.3-0.7 is the
/// practical range; higher introduces visible halos.
///
/// RAM: buffers are sized to the mask bbox plus the 2 px kernel
/// half-width. A 200 × 200 stroke on a 4096 × 3072 image holds ~480 KB
/// of f32 instead of ~150 MB.
pub fn sharpen_inpainted(image: &mut RgbaImage, mask: &GrayImage, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let amount = amount.min(2.0);
    let (w, _h) = image.dimensions();
    if image.dimensions() != mask.dimensions() {
        tracing::warn!(
            image_dims = ?image.dimensions(),
            mask_dims = ?mask.dimensions(),
            "sharpen_inpainted: dim mismatch, skipping",
        );
        return;
    }
    let Some(bbox) = mask_bbox(mask, 128, SHARPEN_MARGIN) else {
        return;
    };
    let src = image.as_raw();
    let img_w = w as usize;
    let bbox_w = bbox.w as usize;
    let bbox_h = bbox.h as usize;
    let bbox_x = bbox.x as usize;
    let bbox_y = bbox.y as usize;

    // Horizontal pass, RGB only — alpha never changes.
    let mut tmp: Vec<f32> = vec![0.0; bbox_w * bbox_h * 3];
    use rayon::prelude::*;
    tmp.par_chunks_mut(bbox_w * 3).enumerate().for_each(|(by, row)| {
        let global_y = bbox_y + by;
        for bx in 0..bbox_w {
            let global_x = bbox_x + bx;
            for c in 0..3 {
                let mut sum = 0.0;
                for (k, &kv) in SHARPEN_KERNEL.iter().enumerate() {
                    let xi = (global_x as isize + k as isize - SHARPEN_MARGIN as isize)
                        .clamp(0, img_w as isize - 1) as usize;
                    sum += kv * src[(global_y * img_w + xi) * 4 + c] as f32;
                }
                row[bx * 3 + c] = sum;
            }
        }
    });

    // Sharpened in place: the blur reads only `tmp`, and each pixel's
    // own value is read just before it is written.
    let msk_raw = mask.as_raw();
    let dp = crate::math::SendMutPtr(image.as_mut_ptr());
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
                    let by_kernel = (by as isize + k as isize - SHARPEN_MARGIN as isize)
                        .clamp(0, bbox_h as isize - 1) as usize;
                    blur += kv * tmp[(by_kernel * bbox_w + bx) * 3 + c];
                }
                // SAFETY: each parallel iteration `by` reads and writes
                // bytes at global row {global_y} only — different `by`s
                // are row-disjoint. Channels (c=0,1,2) within one pixel
                // are also distinct bytes. No race.
                unsafe {
                    let p = dp.0.add(pix + c);
                    let s = f32::from(*p);
                    *p = crate::math::round_u8(s + amount * (s - blur));
                }
            }
        }
        // Capture `dp` whole: its raw-pointer field alone is not Sync.
        let _ = &dp;
    });
}

/// Fill `mask` with the backend `id`. Returns the input unchanged when
/// the mask is all-zero.
pub fn process_inpaint(image: &RgbaImage, mask: &GrayImage, id: prunr_models::ModelId) -> Result<RgbaImage, CoreError> {
    process_inpaint_with(image, mask, id, None, false, &Progress::none()).map(|f| f.rgba)
}

/// `process_inpaint` with optional SD tuning and the run's progress
/// handle. `cpu_only` keeps every session off the GPU providers (for SD
/// it overrides the request's tuning). ORT has no per-op cancel hook, so
/// cancel takes up to one LaMa tile or one SD UNet step.
pub fn process_inpaint_with(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    sd_req: Option<crate::inpaint_sd::SdInpaintRequest>,
    cpu_only: bool,
    progress: &Progress,
) -> Result<Filled, CoreError> {
    if image.dimensions() != mask.dimensions() {
        return Err(CoreError::Inference(format!(
            "inpaint: dim mismatch — image {:?} vs mask {:?}",
            image.dimensions(),
            mask.dimensions()
        )));
    }
    // Before the 200 MB session load: empty masks are common in live
    // preview before the user paints.
    if mask_is_empty(mask) {
        return Ok(Filled::full(image.clone()));
    }
    if id.is_sd_family() {
        // The SD pipeline gates RAM itself; LaMa / MI-GAN need < 1 GB.
        let mut req = sd_req.unwrap_or_else(|| crate::inpaint_sd::SdInpaintRequest {
            tuning: crate::inpaint_sd::plan_tuning(false, crate::inpaint_sd::SD_DEFAULT_MARGIN_MB),
            ..Default::default()
        });
        req.tuning.cpu_only = cpu_only;
        return crate::inpaint_sd::process_inpaint_with(image, mask, id, req, progress);
    }
    process_lama(image, mask, id, cpu_only, FillScale::Auto, progress)
}

/// A fill, and the scale its region was filled at: 1 is full size; with
/// several regions, the smallest of their scales.
pub struct Filled {
    pub rgba: RgbaImage,
    pub scale: f32,
}

impl Filled {
    pub fn full(rgba: RgbaImage) -> Self {
        Self { rgba, scale: 1.0 }
    }
}

/// How large an eraser fills a hole. LaMa works in 512 px tiles and SD
/// in one crop of up to 768 px: a hole much wider than that leaves parts
/// of the model's input with no photo in them (LaMa fills those flat
/// grey; SD fills each tile on its own, a patchwork). Filling a
/// scaled-down copy keeps the whole hole in view of its surroundings, a
/// little softer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FillScale {
    /// Full size while a region fits the model (`Fit`), else shrunk to fit.
    #[default]
    Auto,
    Full,
    Half,
    Quarter,
}

/// The largest region, in working pixels, a model fills at full size
/// with photo around it to go on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fit {
    pub long: u32,
    pub short: u32,
}

/// LaMa: room for 64 px of photo on each side within one tile.
pub const LAMA_FIT: Fit = Fit { long: TILE - 2 * OVERLAP, short: TILE - 2 * OVERLAP };

impl FillScale {
    pub const ALL: [Self; 4] = [Self::Auto, Self::Full, Self::Half, Self::Quarter];

    /// The factor a `w` × `h` region is filled at.
    pub fn factor(self, (w, h): (u32, u32), fit: Fit) -> f32 {
        match self {
            Self::Auto => {
                let (long, short) = (w.max(h).max(1), w.min(h).max(1));
                (fit.long as f32 / long as f32).min(fit.short as f32 / short as f32).min(1.0)
            }
            Self::Full => 1.0,
            Self::Half => 0.5,
            Self::Quarter => 0.25,
        }
    }
}

impl std::fmt::Display for FillScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::Half => "half",
            Self::Quarter => "quarter",
        })
    }
}

impl std::str::FromStr for FillScale {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter()
            .find(|v| v.to_string().eq_ignore_ascii_case(s))
            .ok_or_else(|| format!("unknown fill scale '{s}' (expected: auto, full, half, quarter)"))
    }
}

/// LaMa, Big-LaMa or MI-GAN fill of `mask` (255 = fill) at `scale`.
pub fn process_lama(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    cpu_only: bool,
    scale: FillScale,
    progress: &Progress,
) -> Result<Filled, CoreError> {
    if mask_is_empty(mask) {
        return Ok(Filled::full(image.clone()));
    }
    progress.step(Step::LoadingModel);
    let session = LamaSession::get(id, cpu_only)?;
    fill_at_scale(image, mask, scale, LAMA_FIT, progress, |rgba, m, _, progress| tile_compose(rgba, m, progress, |tile_rgba, tile_mask| {
        session.run_tile(tile_rgba, tile_mask).unwrap_or_else(|e| {
            tracing::error!(%e, "LaMa tile inference failed; leaving tile unchanged");
            tile_rgba.clone()
        })
    }))
}

/// Fill `mask` through `fill` at `scale`, region by region: each region
/// (nearby pieces grouped) `scale` shrinks is filled on its own, scaled down;
/// the rest in one full-size pass. A small mark far from a large one
/// stays sharp. `fill` gets the scale it fills at (1 at full size), and
/// reports through the handle it is given, which places a scaled-down
/// crop's tiles where the crop sits in the photo.
pub(crate) fn fill_at_scale(
    image: &RgbaImage,
    mask: &GrayImage,
    scale: FillScale,
    fit: Fit,
    progress: &Progress,
    mut fill: impl FnMut(&RgbaImage, &GrayImage, f32, &Progress) -> Result<RgbaImage, CoreError>,
) -> Result<Filled, CoreError> {
    if scale == FillScale::Full {
        return Ok(Filled::full(fill(image, mask, 1.0, progress)?));
    }
    let shrunk: Vec<(Bbox, f32)> = crate::inpaint_sd::mask_regions(mask).into_iter()
        .map(|r| {
            let r = r.to_bbox();
            (r, scale.factor((r.w, r.h), fit))
        })
        .filter(|&(_, f)| f < 1.0)
        .collect();
    if shrunk.is_empty() {
        return Ok(Filled::full(fill(image, mask, 1.0, progress)?));
    }
    let smallest = shrunk.iter().map(|&(_, f)| f).fold(1.0f32, f32::min);
    let mut out = image.clone();
    let mut rest = mask.clone();
    let width = mask.width() as usize;
    for (region, factor) in shrunk {
        fill_scaled_down(&mut out, mask, region, factor, progress, &mut fill)?;
        for y in region.y as usize..(region.y + region.h) as usize {
            let row = y * width + region.x as usize;
            rest.as_mut()[row..row + region.w as usize].fill(0);
        }
    }
    if !mask_is_empty(&rest) {
        out = fill(&out, &rest, 1.0, progress)?;
    }
    Ok(Filled { rgba: out, scale: smallest })
}

/// Fill the part of `mask` inside `hole` through `fill` at `factor` < 1,
/// in place: the hole and `OVERLAP` working pixels of photo around it
/// (on Auto, one LaMa tile in all), scaled down, filled, scaled back up
/// and pasted inside the hole only. Other holes in the crop stay holes,
/// so the object being erased next door is never read as photo.
///
/// Peak working set for a hole of side L: the crop around it and its
/// fill scaled back up, each about (L + 2·64/factor)² RGBA; the scaled
/// copies are one tile.
fn fill_scaled_down(
    out: &mut RgbaImage,
    mask: &GrayImage,
    hole: Bbox,
    factor: f32,
    progress: &Progress,
    fill: &mut impl FnMut(&RgbaImage, &GrayImage, f32, &Progress) -> Result<RgbaImage, CoreError>,
) -> Result<(), CoreError> {
    let margin = (OVERLAP as f32 / factor).ceil() as u32;
    let (x0, y0) = (hole.x.saturating_sub(margin), hole.y.saturating_sub(margin));
    let (x1, y1) = ((hole.x + hole.w + margin).min(mask.width()), (hole.y + hole.h + margin).min(mask.height()));
    let crop = Bbox { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    let small = |side: u32| ((side as f32 * factor).round() as u32).max(1);
    let (sw, sh) = (small(crop.w), small(crop.h));
    let crop_rgba = image::imageops::crop_imm(out, crop.x, crop.y, crop.w, crop.h).to_image();
    let small_rgba = crate::formats::resize_rgba(&crop_rgba, sw, sh, crate::formats::ResizeFilter::Lanczos3);
    drop(crop_rgba);
    let crop_mask = image::imageops::crop_imm(mask, crop.x, crop.y, crop.w, crop.h).to_image();
    // Any coverage stays a hole, so nothing of the object survives the shrink.
    let mut small_mask = crate::formats::resize_gray_lanczos3(&crop_mask, sw, sh);
    small_mask.pixels_mut().for_each(|p| p.0[0] = if p.0[0] > 0 { 255 } else { 0 });
    let small_fill = fill(&small_rgba, &small_mask, factor, &progress.within((crop.x, crop.y, crop.w, crop.h), mask.dimensions()))?;
    let filled = crate::formats::resize_rgba(&small_fill, crop.w, crop.h, crate::formats::ResizeFilter::Lanczos3);
    // Only this hole: another one's pixels in the crop get their own fill.
    let in_hole = |x: u32, y: u32| (hole.x..hole.x + hole.w).contains(&x) && (hole.y..hole.y + hole.h).contains(&y);
    for (x, y, m) in crop_mask.enumerate_pixels() {
        let (px, py) = (crop.x + x, crop.y + y);
        if m.0[0] >= 128 && in_hole(px, py) {
            out.put_pixel(px, py, *filled.get_pixel(x, y));
        }
    }
    Ok(())
}

/// Do NOT parallelise across tiles: ORT inference is already
/// multi-threaded internally.
struct LamaSession {
    session: Mutex<Session>,
    image_input_name: String,
    mask_input_name: String,
    pick: SessionPick,
}

/// Load `id`'s LaMa-family session (or find it loaded) and say where it
/// runs. `process_lama` with the same `cpu_only` uses this session.
pub fn load_lama(id: prunr_models::ModelId, cpu_only: bool) -> Result<SessionPick, CoreError> {
    LamaSession::get(id, cpu_only).map(|s| s.pick.clone())
}



/// The `Arc<OnceLock>` closes the build race between concurrent `get()`
/// callers (e.g. a prewarm racing the first stroke).
type LamaSlot = Arc<OnceLock<Result<Arc<LamaSession>, String>>>;
/// Keyed by model and "CPU only", so a session built under one choice is
/// never handed out under the other.
type LamaCache = HashMap<(prunr_models::ModelId, bool), crate::inpaint_sd::CacheEntry<LamaSlot>>;

fn lama_cache() -> &'static Mutex<LamaCache> {
    static CACHE: OnceLock<Mutex<LamaCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop every cached eraser session, LaMa-class and Stable Diffusion,
/// kept ones too; in-flight callers keep their own `Arc`. Used on a model
/// switch, where idling the old backend would hold ~700 MB–2 GB (LaMa) or
/// ~16 GB (SD).
///
/// The caches are empty when this returns; the sessions free where the
/// result is dropped, so a caller that must not wait hundreds of
/// milliseconds on the free drops it on another thread.
pub fn release_all() -> ReleasedSessions {
    let lama = std::mem::take(&mut *lama_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    if !lama.is_empty() {
        tracing::info!(dropped = lama.len(), "LaMa: cache cleared on model switch");
    }
    ReleasedSessions { _lama: lama, _sd: crate::inpaint_sd::release_all() }
}

/// Eraser sessions taken out of their caches; dropping it frees them.
#[must_use = "dropping it frees the sessions; drop it where the wait does not matter"]
pub struct ReleasedSessions {
    _lama: LamaCache,
    _sd: crate::inpaint_sd::SdCache,
}

fn ensure_lama_sweeper_running() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        crate::inpaint_sd::spawn_idle_sweeper(
            "lama-idle-sweeper",
            lama_cache,
            "LaMa: background sweeper released idle session(s)",
        );
    });
}

impl LamaSession {
    /// Built once on first use and released after `crate::IDLE_RELEASE`
    /// without a `get()`; in-flight callers hold their own `Arc`. A build
    /// that fails is not kept, so the next stroke tries again.
    fn get(id: prunr_models::ModelId, cpu_only: bool) -> Result<Arc<LamaSession>, CoreError> {
        ensure_lama_sweeper_running();
        let cache = lama_cache();
        let now = Instant::now();

        // Never hold the cache lock across the seconds-long build.
        let (slot, idle): (LamaSlot, _) = {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let idle = crate::inpaint_sd::sweep_and_log(&mut guard, now, "LaMa: released idle session(s)");
            let entry = guard.entry((id, cpu_only))
                .or_insert_with(|| crate::inpaint_sd::CacheEntry::new(Arc::new(OnceLock::new()), now));
            entry.touch(now);
            (entry.value.clone(), idle)
        };
        drop(idle);

        // Concurrent callers block on the OnceLock and share one build.
        let built = slot.get_or_init(|| Self::new_inner(id, cpu_only).map(Arc::new)).clone();
        if built.is_err() {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            crate::inpaint_sd::forget_failed(&mut guard, &(id, cpu_only), &slot);
        }
        built.map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId, cpu_only: bool) -> Result<LamaSession, String> {
        let bytes = prunr_models::resolve_bytes(id)
            .ok_or_else(|| prunr_models::not_installed_error(id))?;
        let bytes = bytes.as_ref();
        // All cores, unlike the seg models: the user waits on every stroke.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);

        let (session, pick) = build_lama_session(id, bytes, threads, cpu_only)?;

        let (img_name, mask_name) = lama_input_names(&session)
            .map_err(|inputs| format!("LaMa: expected 2 inputs, got {}: {inputs:?}", inputs.len()))?;
        tracing::info!(
            ?id,
            image_input = %img_name,
            mask_input = %mask_name,
            provider = %pick.provider,
            threads,
            "Inpaint session ready",
        );
        Ok(LamaSession {
            session: Mutex::new(session),
            image_input_name: img_name,
            mask_input_name: mask_name,
            pick,
        })
    }

    fn run_tile(&self, image: &RgbaImage, mask: &GrayImage) -> Result<RgbaImage, CoreError> {
        let (img_in, mask_in) = encode_tile(image, mask);

        // The lock stays held until decode so the ORT buffer the output
        // view borrows stays alive — saves a 3 MB clone per tile.
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
        Ok(decode_tile(view, image, mask))
    }
}

impl Drop for LamaSession {
    fn drop(&mut self) {
        tracing::info!(
            rss_mb = crate::sysmem::process_rss_mb(),
            "LaMa session dropped (idle release or process exit)",
        );
    }
}

/// Build the LaMa session, GPU EPs first, then CPU. Returns where it
/// runs, with each GPU provider that failed on the way.
///
/// Each GPU candidate is smoke-tested with one inference: some EPs (CUDA
/// on certain ORT versions) commit fine but fail at runtime on specific
/// ops ("Reshape node ... GetElementType is not implemented"), which
/// would silently leave every tile unchanged.
fn build_lama_session(
    id: prunr_models::ModelId,
    bytes: &[u8],
    threads: usize,
    cpu_only: bool,
) -> Result<(Session, SessionPick), String> {
    lama_ladder(id, bytes, crate::engine::gpu_eps_for(cpu_only), threads, || base_builder(threads))
}

/// `build_lama_session` over `eps`, each session started from `builder`.
fn lama_ladder(
    id: prunr_models::ModelId,
    bytes: &[u8],
    eps: &[EpKind],
    threads: usize,
    builder: impl Fn() -> Result<ort::session::builder::SessionBuilder, String>,
) -> Result<(Session, SessionPick), String> {
    let landed = crate::engine::walk_ladder("LaMa: ", id, Weights::Fp32, eps, |ep| {
        crate::cache::gc_stale_for_model(id, ep.as_str());
        let builder = builder().map_err(Miss::Unusable)?;
        let (builder, bytes_owner, draft) = if ep.keeps_optimized_graph() {
            apply_ort_graph_cache(builder, bytes, id, ep.as_str(), Weights::Fp32)
        } else {
            (builder, Cow::Borrowed(bytes), None)
        };
        let (builder, coverage) = crate::ep_coverage::watch(ep, builder);
        let mut built = builder
            .with_execution_providers([ep.provider(id, EpSite::Fill { threads })])
            .map_err(|e| Miss::Unusable(format!("ORT register {ep} EP failed: {e}")))?;
        let failed = |error| Miss::Failed { error, suspect_cache: cached_graph(matches!(bytes_owner, Cow::Owned(_)), id, ep, Weights::Fp32) };
        let started = Instant::now();
        let mut session = built
            .commit_from_memory(&bytes_owner)
            .map_err(|e| failed(format!("ORT session creation failed ({ep}): {e}")))?;
        tracing::info!(?id, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "LaMa: session committed");
        coverage.report(id, None);
        smoke_test_session(&mut session).map_err(|e| failed(format!("{ep} test run failed: {e}")))?;
        if let Some(draft) = draft { draft.publish(); }
        tracing::info!(ep = %ep, "LaMa: GPU session validated");
        Ok(session)
    });
    let failed = match landed {
        Ok((session, ep, failed)) => return Ok((session, SessionPick { failed, ..SessionPick::on(ep.as_str()) })),
        Err(walk) => walk.failed,
    };

    crate::cache::gc_stale_for_model(id, crate::engine::CPU_NAME);
    let (mut builder, cpu_bytes, draft) = apply_ort_graph_cache(builder()?, bytes, id, crate::engine::CPU_NAME, Weights::Fp32);
    let started = Instant::now();
    let session = builder
        .commit_from_memory(&cpu_bytes)
        .map_err(|e| format!("LaMa: CPU session commit failed: {e}"))?;
    if let Some(draft) = draft { draft.publish(); }
    tracing::info!(?id, ep = crate::engine::CPU_NAME, elapsed_ms = started.elapsed().as_millis() as u64, "LaMa: session committed");
    Ok((session, SessionPick { failed, ..SessionPick::on(crate::engine::CPU_NAME) }))
}

/// One inference on zero inputs: catches EP/op incompatibilities that
/// don't surface until the first run.
fn smoke_test_session(session: &mut Session) -> Result<(), String> {
    let s = TILE as usize;
    let img = ndarray::Array4::<f32>::zeros((1, 3, s, s));
    let msk = ndarray::Array4::<f32>::zeros((1, 1, s, s));
    let (img_name, mask_name) = lama_input_names(session)
        .map_err(|inputs| format!("smoke test: need 2 inputs, got {inputs:?}"))?;
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
    crate::ort_runtime::session_builder()
        .map_err(|e| format!("LaMa: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("LaMa: optimization level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| format!("LaMa: intra_threads: {e}"))
}

/// The session's (image, mask) input names, or every input name when
/// there are fewer than two.
fn lama_input_names(session: &Session) -> Result<(String, String), Vec<String>> {
    let inputs = session.inputs().iter()
        .map(|i| i.name().to_string())
        .collect::<Vec<_>>();
    if inputs.len() < 2 {
        return Err(inputs);
    }
    Ok(pick_input_names(&inputs))
}

/// Inputs named like `image` / `mask` (Carve/LaMa-ONNX convention),
/// case-insensitively; else positional order.
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
/// expects. A tile smaller than TILE×TILE (a photo under 512 px) is
/// mirrored out to it, mask included: the reflection of a painted object
/// is filled too, not handed to the model as photo.
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
    let (image, mask) = (crate::formats::pad_mirrored(image, TILE, TILE), crate::formats::pad_mirrored(mask, TILE, TILE));
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

/// Decode LaMa output back to RGBA at the tile's actual size. Outside
/// the mask the source is copied: LaMa perturbs unmasked pixels by
/// fractions of a unit, which would defeat the feather blend.
fn decode_tile(
    output: ndarray::ArrayView4<'_, f32>,
    source: &RgbaImage,
    mask: &GrayImage,
) -> RgbaImage {
    let (w, h) = source.dimensions();
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
                out_raw[pix] = crate::math::round_u8(buf[i] * scale);
                out_raw[pix + 1] = crate::math::round_u8(buf[plane + i] * scale);
                out_raw[pix + 2] = crate::math::round_u8(buf[plane * 2 + i] * scale);
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

/// No masked pixel inside `tile`, read in place.
fn rect_is_empty(mask: &GrayImage, tile: &Bbox) -> bool {
    let w = mask.width() as usize;
    let raw = mask.as_raw();
    (tile.y..tile.y + tile.h).all(|y| {
        let start = y as usize * w + tile.x as usize;
        raw[start..start + tile.w as usize].iter().all(|&v| v == 0)
    })
}

fn mask_is_empty(mask: &GrayImage) -> bool {
    mask.as_raw().iter().all(|&v| v == 0)
}

/// Lay tiles across `(width, height)` with `OVERLAP` overlap. Tiles at
/// the right/bottom edges shift back to keep `TILE × TILE` inside the
/// image. An image within TILE on both axes is one tile at its own size.
pub(crate) fn plan_tiles(width: u32, height: u32) -> Vec<Bbox> {
    if width <= TILE && height <= TILE {
        return vec![Bbox { x: 0, y: 0, w: width, h: height }];
    }
    let step = TILE - OVERLAP;
    let axis = |len: u32| {
        let mut starts: Vec<u32> = (0..).map(|i| i * step).take_while(|&v| v + TILE <= len || v == 0).collect();
        if starts.last().is_some_and(|&last| last + TILE < len) {
            starts.push(len - TILE);
        }
        starts
    };
    let (xs, ys) = (axis(width), axis(height));
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            let w = TILE.min(width - x);
            let h = TILE.min(height - y);
            out.push(Bbox { x, y, w, h });
        }
    }
    out
}

/// One tile centred on the painted `bbox` when it fits with `OVERLAP` of
/// photo on every side (or the photo is no larger than a tile there):
/// one model run, and the most photo around the hole. On the grid a
/// small stroke over a seam would run up to four tiles, each seeing it
/// at its edge.
fn tile_around(bbox: Bbox, width: u32, height: u32) -> Option<Bbox> {
    let axis = |start: u32, len: u32, full: u32| {
        let size = TILE.min(full);
        (size == full || len + 2 * OVERLAP <= size).then(|| ((start + len / 2).saturating_sub(size / 2).min(full - size), size))
    };
    let ((x, w), (y, h)) = (axis(bbox.x, bbox.w, width)?, axis(bbox.y, bbox.h, height)?);
    Some(Bbox { x, y, w, h })
}

/// Tile blend weight: 0.0 at the edge, smoothstep up to 1.0 at
/// `OVERLAP` pixels in.
pub(crate) fn feather_weight(distance_from_edge: u32) -> f32 {
    if distance_from_edge >= OVERLAP {
        1.0
    } else {
        let t = distance_from_edge as f32 / OVERLAP as f32;
        crate::math::smoothstep(t)
    }
}

/// Run `inpaint_tile` on every tile that touches the mask and
/// feather-blend the results into a copy of `image`.
///
/// RAM: accumulators are sized to the mask bbox. A 200 × 200 stroke on a
/// 4096 × 3072 image holds ~640 KB of f32 instead of ~250 MB.
///
/// Asked for previews, each finished tile goes out as its fill over the
/// photo (`tile_fill_preview`), made in the tile's photo crop, which the loop
/// holds anyway, for the part of the tile the mask's box reaches; the
/// photo the listener shows under the previews holds the rest. The result
/// is the same either way. Extra per tile (the resample, then the part
/// sent, at most the tile at preview size) and for a listener that paints
/// them all into one preview of the whole photo:
///
/// | Photo, preview edge | Extra per tile         | Whole preview |
/// |---------------------|------------------------|---------------|
/// | 1024 × 768, 1536    | the part, ≤ 1 MB (1:1) | 3 MB          |
/// | 4096 × 3072, 1536   | 150 KB + the part      | 7 MB          |
/// | 8192 × 6144, 1536   | 40 KB + the part       | 7 MB          |
// `pub` only for the benches; callers use `process_inpaint`.
#[doc(hidden)]
pub fn tile_compose<F>(
    image: &RgbaImage,
    mask: &GrayImage,
    progress: &Progress,
    mut inpaint_tile: F,
) -> Result<RgbaImage, CoreError>
where
    F: FnMut(&RgbaImage, &GrayImage) -> RgbaImage,
{
    let (w, h) = image.dimensions();
    let mut out = image.clone();

    let Some(bbox) = mask_bbox(mask, 1, 0) else {
        return Ok(out);
    };

    let bbox_w = bbox.w as usize;
    let bbox_n = (bbox.w as usize) * (bbox.h as usize);
    let mut weight_acc: Vec<f32> = vec![0.0; bbox_n];
    // No alpha: the resolve keeps the photo's.
    let mut color_acc: Vec<[f32; 3]> = vec![[0.0; 3]; bbox_n];

    // Only tiles that touch the mask run; they are the tile map's layout.
    let work: Vec<Bbox> = match tile_around(bbox, w, h) {
        Some(tile) => vec![tile],
        None => plan_tiles(w, h).into_iter().filter(|t| !rect_is_empty(mask, t)).collect(),
    };
    let total = work.len() as u32;
    progress.step(Step::Erasing);
    progress.tiles(work.iter().map(|t| TileRect::of_pixels(t.x, t.y, t.w, t.h, w, h)).collect());
    if total > 0 {
        progress.inner(0, total, Unit::Tile);
    }
    for (i, tile) in work.iter().enumerate() {
        if progress.is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        progress.tile(i as u32, false);
        let tile_mask = image::imageops::crop_imm(mask, tile.x, tile.y, tile.w, tile.h).to_image();
        let tile_rgba = image::imageops::crop_imm(image, tile.x, tile.y, tile.w, tile.h).to_image();
        let painted = inpaint_tile(&tile_rgba, &tile_mask);
        accumulate_tile(&mut color_acc, &mut weight_acc, &painted, tile, bbox, (w, h));
        let rect = (tile.x, tile.y, tile.w, tile.h);
        progress.tile_preview_part(rect, (w, h), bbox.within(rect), |size, part| tile_fill_preview(tile_rgba, &painted, &tile_mask, size, part));
        progress.inner(i as u32 + 1, total, Unit::Tile);
        progress.tile(i as u32, true);
    }

    // The mask==0 skip is load-bearing: the float blend drifts a u8 by
    // one even on a uniform region (255*w/w ≠ 255 in f32).
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
                let pix = global_idx * 4;
                out_raw[pix]     = crate::math::round_u8(c[0] * inv);
                out_raw[pix + 1] = crate::math::round_u8(c[1] * inv);
                out_raw[pix + 2] = crate::math::round_u8(c[2] * inv);
            }
        }
    }
    Ok(out)
}

/// `part` of a finished tile as the erase shows it, previewed at `size`:
/// the fill where the mask is set (the rule the final composite follows),
/// the photo elsewhere, the photo's alpha throughout. Fills `photo` in
/// place.
fn tile_fill_preview(mut photo: RgbaImage, painted: &RgbaImage, mask: &GrayImage, size: (u32, u32), part: (u32, u32, u32, u32)) -> Vec<u8> {
    for ((p, f), m) in photo.pixels_mut().zip(painted.pixels()).zip(mask.pixels()) {
        if m.0[0] != 0 {
            p.0[..3].copy_from_slice(&f.0[..3]);
        }
    }
    crate::formats::preview_rgba(Cow::Owned(photo), size, part)
}

/// Bounding box of pixels where `mask >= min_value`, expanded by
/// `margin` pixels on each side and clamped to image bounds. `None`
/// for an empty result.
///
/// A side touching the image edge gets no margin, so kernel reads
/// clamped to the bbox match a full-image pass clamped to the image.
///
/// `min_value = 1` finds any nonzero pixel; `128` is the binary-mask
/// threshold.
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
    color_acc: &mut [[f32; 3]],
    weight_acc: &mut [f32],
    tile: &RgbaImage,
    placement: &Bbox,
    bbox: Bbox,
    (img_w, img_h): (u32, u32),
) {
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

    // The feather fades a side only where a neighbouring tile takes
    // over; on the photo's edge a zero weight would leave pixels unfilled.
    let faded = |dist: u32, neighbour: bool| if neighbour { dist } else { u32::MAX };
    let bbox_w = bbox.w as usize;
    let tile_raw = tile.as_raw();
    let tile_w = tile.width() as usize;
    for ty in ty_start..ty_end {
        let by = ((placement.y + ty) - bbox.y) as usize;
        let dist_top = faded(ty, placement.y > 0);
        let dist_bottom = faded(placement.h - 1 - ty, tile_y_end < img_h);
        let edge_y = dist_top.min(dist_bottom);
        let tile_row = ty as usize * tile_w;
        for tx in tx_start..tx_end {
            let bx = ((placement.x + tx) - bbox.x) as usize;
            let dist_left = faded(tx, placement.x > 0);
            let dist_right = faded(placement.w - 1 - tx, tile_x_end < img_w);
            let edge_x = dist_left.min(dist_right);
            let w = feather_weight(edge_x.min(edge_y));
            let dst_idx = by * bbox_w + bx;
            let p_off = (tile_row + tx as usize) * 4;
            let acc = &mut color_acc[dst_idx];
            acc[0] += tile_raw[p_off]     as f32 * w;
            acc[1] += tile_raw[p_off + 1] as f32 * w;
            acc[2] += tile_raw[p_off + 2] as f32 * w;
            weight_acc[dst_idx] += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w`×`h` mask at 255 over `x0..x1`, `y0..y1`, 0 elsewhere.
    fn rect_mask(w: u32, h: u32, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> GrayImage {
        GrayImage::from_fn(w, h, |x, y| Luma([if (x0..x1).contains(&x) && (y0..y1).contains(&y) { 255 } else { 0 }]))
    }

    /// `img` with `f` applied to a copy.
    fn applied(img: &RgbaImage, f: impl FnOnce(&mut RgbaImage)) -> RgbaImage {
        let mut out = img.clone();
        f(&mut out);
        out
    }
    use image::{Luma, Rgba};

    /// LaMa's tiler reports a contract-clean run over only the tiles
    /// the mask touches, and a cancel stops it before the next tile.
    #[test]
    fn tile_compose_reports_the_masked_tiles() {
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        let img = RgbaImage::from_pixel(1200, 600, Rgba([10, 20, 30, 255]));
        let mut mask = GrayImage::new(1200, 600);
        mask.put_pixel(100, 100, Luma([255]));
        mask.put_pixel(1100, 500, Luma([255]));
        let sink = std::sync::Arc::new(RecordingSink::default());
        tile_compose(&img, &mask, &Progress::new(sink.clone()), |t, _| t.clone()).unwrap();
        let updates = sink.updates();
        assert_eq!(check_contract(&updates), Ok(()));
        let layout = updates.iter().find_map(|u| match u { ProgressUpdate::Tiles(t) => Some(t.len()), _ => None });
        assert!(layout.is_some_and(|n| n >= 2 && n < plan_tiles(1200, 600).len()), "only the masked tiles: {layout:?}");

        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let ran = std::cell::Cell::new(false);
        let result = tile_compose(&img, &mask, &Progress::none().with_cancel(flag), |t, _| { ran.set(true); t.clone() });
        assert!(matches!(result, Err(CoreError::Cancelled)) && !ran.get());
    }

    /// Asked for previews, each finished tile is sent once, while it
    /// runs, at its place: the fill inside the mask, the photo untouched
    /// outside it. A photo of one tile goes the same way. The result is
    /// the same with or without previews, and none is sent unless asked.
    #[test]
    fn tile_compose_previews_each_finished_tile_over_the_photo() {
        use crate::progress::{check_contract, ProgressUpdate, RecordingSink};
        let fill = |t: &RgbaImage, _: &GrayImage| RgbaImage::from_pixel(t.width(), t.height(), Rgba([250, 0, 0, 7]));
        for (w, h) in [(1200, 600), (300, 200)] {
            let img = RgbaImage::from_fn(w, h, |x, y| Rgba([(x % 251) as u8, (y % 241) as u8, 40, 200]));
            let mut mask = GrayImage::new(w, h);
            for (x, y) in [(100, 100), (101, 100), (w - 20, h - 30)] {
                mask.put_pixel(x, y, Luma([255]));
            }
            mask.put_pixel(102, 100, Luma([60]));
            let mask_box = mask_bbox(&mask, 1, 0).expect("a mask");
            let work: Vec<Bbox> = plan_tiles(w, h).into_iter().filter(|t| !rect_is_empty(&mask, t)).collect();
            let sink = std::sync::Arc::new(RecordingSink::default());
            let with = tile_compose(&img, &mask, &Progress::new(sink.clone()).with_tile_previews(w), fill).unwrap();
            let updates = sink.updates();
            assert_eq!(check_contract(&updates), Ok(()));

            let mut running = None;
            let mut previews = 0;
            for u in &updates {
                match u {
                    ProgressUpdate::Tile { index, done: false } => running = Some(*index as usize),
                    ProgressUpdate::Tile { done: true, .. } => running = None,
                    ProgressUpdate::TilePreview(p) => {
                        let t = work[running.expect("a preview is sent while its tile runs")];
                        let part = mask_box.within((t.x, t.y, t.w, t.h));
                        assert_eq!((p.x, p.y, p.width, p.height, p.full_width, p.full_height), (part.0, part.1, part.2, part.3, w, h), "the tile's part in the mask's box, at its place");
                        for (i, px) in p.rgba.chunks(4).enumerate() {
                            let (x, y) = (p.x + i as u32 % p.width, p.y + i as u32 / p.width);
                            let photo = img.get_pixel(x, y).0;
                            let want = if mask.get_pixel(x, y).0[0] != 0 { [250, 0, 0, photo[3]] } else { photo };
                            assert_eq!(px, want, "({x}, {y})");
                        }
                        previews += 1;
                    }
                    _ => {}
                }
            }
            assert_eq!(previews, work.len(), "one preview per tile, {w} × {h}");
            assert!(w > TILE || work.len() == 1, "a small photo is one tile");

            let quiet = std::sync::Arc::new(RecordingSink::default());
            let without = tile_compose(&img, &mask, &Progress::new(quiet.clone()), fill).unwrap();
            assert!(!quiet.updates().iter().any(|u| matches!(u, ProgressUpdate::TilePreview(_))), "no preview unless asked");
            assert_eq!(with.as_raw(), without.as_raw(), "previews leave the result as it is");
            assert_eq!(tile_compose(&img, &mask, &Progress::none(), fill).unwrap().as_raw(), with.as_raw());
        }
    }

    /// Shrunk for the preview, each tile is sent for the part its mask can
    /// reach: the same pixels the whole tile's preview has there, and
    /// every pixel where that preview differs from the photo's is inside.
    #[test]
    fn a_shrunk_tile_preview_part_holds_what_the_fill_changes() {
        use crate::progress::{ProgressUpdate, RecordingSink};
        let (w, h) = (700, 500);
        let img = RgbaImage::from_fn(w, h, |x, y| Rgba([(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 239) as u8, 255]));
        let mut mask = GrayImage::new(w, h);
        // Wider than one tile less its margins, so the grid runs.
        for (x, y) in (150..560).flat_map(|x| (200..230).map(move |y| (x, y))) {
            mask.put_pixel(x, y, Luma([255]));
        }
        let fill = |t: &RgbaImage, _: &GrayImage| RgbaImage::from_pixel(t.width(), t.height(), Rgba([250, 0, 0, 7]));
        let previewed = |progress: &Progress| progress.clone().with_tile_previews(300);
        let sink = std::sync::Arc::new(RecordingSink::default());
        tile_compose(&img, &mask, &previewed(&Progress::new(sink.clone())), fill).unwrap();
        let parts: Vec<_> = sink.updates().into_iter().filter_map(|u| match u {
            ProgressUpdate::TilePreview(p) => Some(p),
            _ => None,
        }).collect();
        let work: Vec<Bbox> = plan_tiles(w, h).into_iter().filter(|t| !rect_is_empty(&mask, t)).collect();
        assert!(work.len() > 1 && parts.len() == work.len(), "one part per tile the mask touches");
        for (t, part) in work.iter().zip(&parts) {
            let rect = (t.x, t.y, t.w, t.h);
            let tile_rgba = image::imageops::crop_imm(&img, t.x, t.y, t.w, t.h).to_image();
            let tile_mask = image::imageops::crop_imm(&mask, t.x, t.y, t.w, t.h).to_image();
            let whole_sink = std::sync::Arc::new(RecordingSink::default());
            let whole_progress = previewed(&Progress::new(whole_sink.clone()));
            whole_progress.tile_preview(rect, (w, h), |pw, ph| tile_fill_preview(tile_rgba.clone(), &fill(&tile_rgba, &tile_mask), &tile_mask, (pw, ph), (0, 0, pw, ph)));
            whole_progress.tile_preview(rect, (w, h), |pw, ph| crate::formats::preview_rgba(Cow::Borrowed(&tile_rgba), (pw, ph), (0, 0, pw, ph)));
            let updates = whole_sink.updates();
            let [ProgressUpdate::TilePreview(whole), ProgressUpdate::TilePreview(photo)] = updates.as_slice() else { panic!("{updates:?}") };
            assert!(part.width < whole.width, "less than the whole tile");
            for (i, (a, b)) in whole.rgba.chunks(4).zip(photo.rgba.chunks(4)).enumerate() {
                let (x, y) = (whole.x + i as u32 % whole.width, whole.y + i as u32 / whole.width);
                if (part.x..part.x + part.width).contains(&x) && (part.y..part.y + part.height).contains(&y) {
                    let j = (((y - part.y) * part.width + x - part.x) * 4) as usize;
                    assert_eq!(part.rgba[j..j + 4], *a, "({x}, {y}) as the whole tile's preview has it");
                } else {
                    assert_eq!(a, b, "({x}, {y}) changes outside the part");
                }
            }
        }
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

    /// A provider whose session cannot even be started is named in the
    /// pick, as the engine's ladder names it, and is not held against the
    /// model; the CPU then takes the session.
    #[test]
    fn a_builder_that_cannot_start_on_a_provider_is_in_the_pick() {
        use prunr_models::ModelId;
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let Some(bytes) = prunr_models::resolve_bytes(ModelId::Silueta) else { return };
        #[cfg(not(target_os = "macos"))]
        let ep = EpKind::Cuda;
        #[cfg(target_os = "macos")]
        let ep = EpKind::CoreMl;
        let id = ModelId::Migan;
        let calls = std::cell::Cell::new(0);
        let builder = || {
            calls.set(calls.get() + 1);
            if calls.get() == 1 { Err("LaMa: no builder here".to_string()) } else { base_builder(1) }
        };
        let (_, pick) = lama_ladder(id, &bytes, &[ep], 1, builder).expect("the CPU takes the session");
        assert_eq!((pick.provider.as_str(), pick.failed.as_slice()), ("CPU", ["LaMa: no builder here".to_string()].as_slice()));
        assert!(!crate::ep_compat::is_known_failure(ep, id, Weights::Fp32), "the machine's lack, not the model's");
    }

    /// The bbox-cropped grow matches full-image morphology, for masks in
    /// the middle and touching each image edge, growing and shrinking.
    #[test]
    fn grow_mask_matches_full_image_morphology() {
        let full = |m: &GrayImage, px: i32| {
            let mut cur = m.clone();
            let mut next = GrayImage::new(m.width(), m.height());
            for _ in 0..px.unsigned_abs() {
                morphology_step_into(&cur, &mut next, px > 0);
                std::mem::swap(&mut cur, &mut next);
            }
            cur
        };
        for (x0, y0, x1, y1) in [(10, 12, 20, 25), (0, 0, 6, 9), (30, 33, 40, 40), (0, 15, 40, 18)] {
            let mut m = rect_mask(40, 40, (x0, y0, x1, y1));
            m.put_pixel(35, 3, Luma([90]));
            for px in [-3, -1, 1, 4, 7] {
                assert_eq!(grow_mask(&m, px), full(&m, px), "rect {:?} px {px}", (x0, y0, x1, y1));
            }
        }
        assert_eq!(grow_mask(&GrayImage::new(8, 8), 3), GrayImage::new(8, 8));
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
        let out = decode_tile(output.view(), &src, &mask);
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
        let out = decode_tile(output.view(), &src, &mask);
        let p = out.get_pixel(4, 4);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 128);
        assert_eq!(p[3], 255, "alpha always copied from source");
        // Surrounding pixel stays source.
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30, 255]);
    }

    /// A model answer a hair under v/255 is still the byte v, for every
    /// v: truncating it would land one low.
    #[test]
    fn decode_tile_gives_back_every_byte() {
        let src = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 255]));
        let mask = GrayImage::from_pixel(16, 16, Luma([255]));
        let s = TILE as usize;
        let mut output = Array4::<f32>::zeros((1, 3, s, s));
        for v in 0..256 {
            for c in 0..3 {
                output[[0, c, v / 16, v % 16]] = (v as f32 - 0.2).max(0.0) / 255.0;
            }
        }
        let out = decode_tile(output.view(), &src, &mask);
        let got: Vec<u8> = (0..256u32).map(|v| out.get_pixel(v % 16, v / 16)[0]).collect();
        assert_eq!(got, (0..=255).collect::<Vec<u8>>());
    }

    /// A small stroke runs one tile centred on it, wherever it falls on
    /// the grid: one model run, with photo all around it. A stroke too
    /// large for that still gets the grid.
    #[test]
    fn a_small_stroke_runs_one_tile_centred_on_it() {
        let (w, h) = (1100, 700);
        let img = RgbaImage::from_pixel(w, h, Rgba([10, 10, 10, 255]));
        let run = |x0: u32, y0: u32, side: u32| {
            let mask = rect_mask(w, h, (x0, y0, x0 + side, y0 + side));
            let mut tiles = Vec::new();
            tile_compose(&img, &mask, &Progress::none(), |t, m| {
                tiles.push((t.dimensions(), mask_bbox(m, 1, 0).unwrap()));
                t.clone()
            }).unwrap();
            tiles
        };
        // Across the first grid seam (tiles start every 448 px).
        let tiles = run(430, 420, 40);
        assert_eq!(tiles.len(), 1, "{tiles:?}");
        let (size, painted) = tiles[0];
        assert_eq!(size, (TILE, TILE));
        assert_eq!((painted.x, painted.y), ((TILE - 40) / 2, (TILE - 40) / 2), "centred");
        // Near a corner the tile stays inside the photo.
        let (_, painted) = run(5, 650, 30)[0];
        assert_eq!((painted.x, painted.y), (5, 650 - (h - TILE)));
        assert!(run(100, 100, 400).len() > 1, "a stroke wider than a tile less its margins");
    }

    /// A painted pixel on the photo's edge is filled like any other: a
    /// tile's feather fades to nothing at its border, but only where a
    /// neighbouring tile takes over.
    #[test]
    fn a_stroke_on_the_photos_edge_is_filled() {
        for (w, h) in [(300, 200), (1100, 700)] {
            let img = RgbaImage::from_pixel(w, h, Rgba([10, 10, 10, 255]));
            let mut mask = GrayImage::new(w, h);
            for (x, y) in [(0, 0), (w - 1, h - 1), (0, h / 2), (w / 2, 0)] {
                mask.put_pixel(x, y, Luma([255]));
            }
            let out = tile_compose(&img, &mask, &Progress::none(), |t, _| RgbaImage::from_pixel(t.width(), t.height(), Rgba([200, 0, 0, 255]))).unwrap();
            for (x, y) in [(0, 0), (w - 1, h - 1), (0, h / 2), (w / 2, 0)] {
                assert_eq!(out.get_pixel(x, y).0, [200, 0, 0, 255], "{w}x{h} at ({x}, {y})");
            }
        }
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
        let out = decode_tile(output.view(), &src, &mask);
        assert_eq!(out.get_pixel(4, 4).0[0], 200, "200.0 in [0,255] range stays 200");
    }

    #[test]
    fn encode_tile_pads_small_inputs_to_tile_size() {
        let img = RgbaImage::from_fn(64, 64, |x, _| Rgba([x as u8 * 4, 0, 0, 255]));
        let mut mask = GrayImage::new(64, 64);
        mask.put_pixel(32, 32, Luma([255]));
        let (img_t, msk_t) = encode_tile(&img, &mask);
        assert_eq!(img_t.shape(), &[1, 3, TILE as usize, TILE as usize]);
        assert_eq!(msk_t.shape(), &[1, 1, TILE as usize, TILE as usize]);
        // The painted mask cell maps to NCHW[0,0,32,32].
        assert_eq!(msk_t[[0, 0, 32, 32]], 1.0);
        // Past the photo both are mirrored: column 94 reflects column 32.
        assert_eq!(msk_t[[0, 0, 32, 94]], 1.0);
        assert_eq!(img_t[[0, 0, 0, 94]], img_t[[0, 0, 0, 32]]);
        assert_eq!(msk_t[[0, 0, 200, 200]], 0.0);
    }

    /// Auto fills a hole up to one tile's worth at full size and shrinks
    /// a larger one to that; the fixed scales ignore the hole.
    #[test]
    fn fill_scale_factor_follows_the_hole_on_auto() {
        let fit = Fit { long: 700, short: 400 };
        assert_eq!(FillScale::Auto.factor((700, 400), fit), 1.0, "fits as it is");
        assert!((FillScale::Auto.factor((2800, 100), fit) - 0.25).abs() < 1e-6, "too long");
        assert!((FillScale::Auto.factor((800, 650), fit) - 400.0 / 650.0).abs() < 1e-6, "too wide on the short side");
        assert_eq!(FillScale::Full.factor((5000, 5000), fit), 1.0);
        assert_eq!(FillScale::Half.factor((10, 10), fit), 0.5);
        assert_eq!(FillScale::Quarter.factor((10, 10), fit), 0.25);
        for v in FillScale::ALL {
            assert_eq!(v.to_string().parse::<FillScale>(), Ok(v));
        }
    }

    /// On Auto a large hole and its context come to one LaMa tile, and a
    /// neighbouring hole inside the crop stays a hole there but is left
    /// for its own fill.
    #[test]
    fn auto_fills_a_large_hole_in_one_tile_and_keeps_other_holes_out() {
        let image = RgbaImage::from_pixel(3000, 2000, Rgba([10, 20, 30, 255]));
        let mut mask = GrayImage::new(3000, 2000);
        let mut paint = |x0: u32, y0: u32, x1: u32, y1: u32| for y in y0..y1 { for x in x0..x1 { mask.put_pixel(x, y, Luma([255])); } };
        paint(500, 500, 1700, 1300);
        paint(1800, 800, 1830, 830);
        let hole = Bbox { x: 500, y: 500, w: 1200, h: 800 };
        let factor = FillScale::Auto.factor((hole.w, hole.h), LAMA_FIT);
        let mut seen = None;
        let mut out = image.clone();
        fill_scaled_down(&mut out, &mask, hole, factor, &Progress::none(), &mut |rgba: &RgbaImage, m: &GrayImage, _: f32, _: &Progress| {
            // The mark at (1800-1830, 800-830): crop starts 200 px out, scaled by 0.32.
            let neighbour = m.get_pixel(485, 165).0[0];
            seen = Some((rgba.dimensions(), neighbour));
            let mut filled = rgba.clone();
            for (x, y, p) in m.enumerate_pixels() {
                if p.0[0] >= 128 { filled.put_pixel(x, y, Rgba([200, 0, 0, 255])); }
            }
            Ok(filled)
        }).unwrap();
        let ((w, h), neighbour) = seen.expect("filled");
        assert!(w <= TILE && h <= TILE, "one tile, got {w}x{h}");
        assert_eq!(neighbour, 255, "the neighbouring hole is a hole to the model, not photo");
        assert_eq!(out.get_pixel(1100, 900).0, [200, 0, 0, 255], "the hole is filled");
        assert_eq!(out.get_pixel(1815, 815).0, [10, 20, 30, 255], "the neighbour is left for its own fill");
    }

    /// Auto judges each region on its own: a small mark far from a large
    /// one fills at full size in one pass, the large one scaled down.
    #[test]
    fn auto_shrinks_only_the_regions_that_need_it() {
        let image = RgbaImage::from_pixel(1200, 400, Rgba([10, 20, 30, 255]));
        let mut mask = rect_mask(1200, 400, (100, 50, 700, 350));
        for y in 190..210 {
            for x in 1100..1120 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let mut calls = Vec::new();
        let Filled { rgba: out, scale } = fill_at_scale(&image, &mask, FillScale::Auto, Fit { long: 300, short: 300 }, &Progress::none(), |rgba, m, at, _| {
            calls.push((rgba.dimensions(), m.pixels().filter(|p| p.0[0] == 255).count(), at));
            let mut filled = rgba.clone();
            for (x, y, p) in m.enumerate_pixels() {
                if p.0[0] >= 128 {
                    filled.put_pixel(x, y, Rgba([200, 0, 0, 255]));
                }
            }
            Ok(filled)
        }).unwrap();
        assert!(scale > 0.0 && scale < 1.0, "reports the large region's scale: {scale}");
        let full = fill_at_scale(&image, &mask, FillScale::Full, Fit { long: 300, short: 300 }, &Progress::none(), |rgba, _, _, _| Ok(rgba.clone())).unwrap();
        assert_eq!(full.scale, 1.0, "Full fills at full size");
        assert_eq!(calls.len(), 2, "the large region scaled down, then the rest");
        assert!(calls[0].0.0 < 1200, "the large region's crop is scaled down");
        assert_eq!(calls[0].2, scale, "the fill is told the scale it fills at");
        assert_eq!(calls[1], ((1200, 400), 20 * 20, 1.0), "the small mark alone, at full size");
        assert_eq!(out.get_pixel(1110, 200).0, [200, 0, 0, 255]);
        assert_eq!(out.get_pixel(400, 200).0, [200, 0, 0, 255]);
        assert_eq!(out.get_pixel(900, 200).0, [10, 20, 30, 255]);
    }

    /// A region filled scaled down reports its tiles and previews where
    /// its crop sits in the photo, not over the whole photo; the rest,
    /// filled at full size, reports in the photo's own terms.
    #[test]
    fn a_shrunk_region_reports_at_its_place_in_the_photo() {
        use crate::progress::{ProgressUpdate, RecordingSink};
        let image = RgbaImage::from_pixel(1200, 400, Rgba([10, 20, 30, 255]));
        let mut mask = GrayImage::new(1200, 400);
        for (x0, x1, y0, y1) in [(300, 700, 100, 300), (1100, 1120, 190, 210)] {
            for y in y0..y1 {
                for x in x0..x1 {
                    mask.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let sink = std::sync::Arc::new(RecordingSink::default());
        let progress = Progress::new(sink.clone()).with_tile_previews(600);
        fill_at_scale(&image, &mask, FillScale::Auto, Fit { long: 300, short: 300 }, &progress, |rgba, _, _, p| {
            let (w, h) = rgba.dimensions();
            p.tiles(vec![TileRect::of_pixels(0, 0, w, h, w, h)]);
            p.tile_preview((0, 0, w, h), (w, h), |pw, ph| vec![0; (pw * ph * 4) as usize]);
            Ok(rgba.clone())
        }).unwrap();
        // The hole is 400 × 200 against a 300 px fit: filled at 3/4, with
        // ceil(64 / 0.75) = 86 px of photo around it.
        let crop = (214u32, 14u32, 572u32, 372u32);
        let updates = sink.updates();
        let layouts: Vec<&TileRect> = updates.iter().filter_map(|u| match u { ProgressUpdate::Tiles(t) => Some(&t[0]), _ => None }).collect();
        let near = |a: f32, b: u32, of: u32| (a * of as f32 - b as f32).abs() < 0.01;
        let r = layouts[0];
        assert!(near(r.x, crop.0, 1200) && near(r.y, crop.1, 400) && near(r.w, crop.2, 1200) && near(r.h, crop.3, 400), "the crop's place: {r:?}");
        assert_eq!(*layouts[1], TileRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 }, "the rest fills the photo itself");
        let previews: Vec<_> = updates.iter().filter_map(|u| match u { ProgressUpdate::TilePreview(p) => Some((p.x, p.y, p.width, p.height, p.full_width, p.full_height)), _ => None }).collect();
        assert_eq!(previews, vec![(107, 7, 286, 186, 600, 200), (0, 0, 600, 200, 600, 200)], "at half the photo's size");
    }

    /// Below full size the fill sees the hole and photo around it, scaled
    /// down, and only the hole changes in the result.
    #[test]
    fn a_scaled_fill_changes_only_the_hole() {
        let image = RgbaImage::from_pixel(400, 300, Rgba([10, 20, 30, 255]));
        let mask = rect_mask(400, 300, (150, 100, 270, 180));
        let hole = mask_bbox(&mask, 128, 0).unwrap();
        let mut seen = None;
        let mut out = image.clone();
        fill_scaled_down(&mut out, &mask, hole, 0.25, &Progress::none(), &mut |rgba: &RgbaImage, m: &GrayImage, _: f32, _: &Progress| {
            seen = Some((rgba.dimensions(), m.pixels().filter(|p| p.0[0] == 255).count()));
            Ok(RgbaImage::from_pixel(rgba.width(), rgba.height(), Rgba([200, 0, 0, 255])))
        }).unwrap();
        let ((w, h), holes) = seen.expect("filled");
        assert_eq!((w, h), (100, 75), "64 working px each side reach the image's edges here: all of it, at a quarter");
        assert!(holes >= 30 * 20, "the whole hole is a hole at the small size");
        assert_eq!(out.get_pixel(200, 140).0, [200, 0, 0, 255], "inside: the fill");
        assert_eq!(out.get_pixel(140, 140).0, [10, 20, 30, 255], "outside: the photo");
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30, 255]);
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
        let out = tile_compose(&img, &mask, &Progress::none(), |t, _| t.clone()).unwrap();
        assert_eq!(out, img);
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
                    out_raw[pix + c] = crate::math::round_u8(sharp);
                }
            }
        }
        out
    }

    #[cfg(test)]
    fn assert_bbox_sharpen_matches_reference(img: &RgbaImage, mask: &GrayImage, amount: f32) {
        let mut bbox_out = img.clone();
        sharpen_inpainted(&mut bbox_out, mask, amount);
        let ref_out = reference_sharpen(img, mask, amount);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let a = bbox_out.get_pixel(x, y);
                let b = ref_out.get_pixel(x, y);
                assert_eq!(a, b, "({x}, {y}) diverged: bbox={a:?} ref={b:?}");
            }
        }
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

    #[test]
    fn plan_tiles_keeps_a_photo_up_to_one_tile_whole() {
        for (w, h) in [(400, 300), (TILE, TILE)] {
            assert_eq!(plan_tiles(w, h), [Bbox { x: 0, y: 0, w, h }], "{w}×{h}");
        }
    }

    /// The feather is 0 at a tile's edge, 1 from the overlap's width in,
    /// and never falls on the way.
    #[test]
    fn feather_weight_ramps_from_the_edge_to_the_overlap() {
        assert!(feather_weight(0).abs() < 1e-6);
        assert!((feather_weight(OVERLAP) - 1.0).abs() < 1e-6);
        assert!((feather_weight(OVERLAP * 2) - 1.0).abs() < 1e-6);
        for d in 1..=OVERLAP {
            assert!(feather_weight(d) >= feather_weight(d - 1) - 1e-6, "falls at {d}");
        }
    }

    /// Image and mask inputs are told apart by name in any case and
    /// order; with no name to go by, the first is the image.
    #[test]
    fn pick_input_names_matches_by_keyword_or_position() {
        for (names, image, mask) in [
            (["image", "mask"], "image", "mask"),
            (["IMAGE_INPUT", "Mask_Input"], "IMAGE_INPUT", "Mask_Input"),
            (["the_mask", "the_image"], "the_image", "the_mask"),
            (["input_0", "input_1"], "input_0", "input_1"),
        ] {
            let names = names.map(String::from).to_vec();
            let (img, msk) = pick_input_names(&names);
            assert_eq!((img.as_str(), msk.as_str()), (image, mask), "{names:?}");
        }
    }

    /// Grow 0 is the mask itself; +1 adds the 4-connected neighbours (not
    /// the diagonals); −1 strips a pixel off every edge.
    #[test]
    fn grow_mask_dilates_and_erodes_by_4_connectivity() {
        let mut dot = GrayImage::new(5, 5);
        dot.put_pixel(2, 2, Luma([255]));
        assert_eq!(grow_mask(&dot, 0).as_raw(), dot.as_raw());
        let grown = grow_mask(&dot, 1);
        for (x, y) in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)] {
            assert_eq!(grown.get_pixel(x, y).0[0], 255, "({x}, {y})");
        }
        assert_eq!(grown.get_pixel(1, 1).0[0], 0, "a diagonal");
        let block = rect_mask(5, 5, (1, 1, 4, 4));
        let shrunk = grow_mask(&block, -1);
        assert_eq!(shrunk.get_pixel(2, 2).0[0], 255, "the centre survives");
        assert_eq!((shrunk.get_pixel(1, 1).0[0], shrunk.get_pixel(1, 2).0[0]), (0, 0), "every pixel with a clear neighbour goes");
    }

    /// The box of the pixels at or above the threshold, grown by the
    /// margin and clipped to the image; none when nothing reaches it.
    #[test]
    fn mask_bbox_bounds_the_mask_with_its_margin() {
        let mask = |w, h, on: &[(u32, u32, u8)]| {
            let mut m = GrayImage::new(w, h);
            for &(x, y, v) in on { m.put_pixel(x, y, Luma([v])); }
            m
        };
        let bbox = |x, y, w, h| Some(Bbox { x, y, w, h });
        assert_eq!(mask_bbox(&GrayImage::new(64, 64), 1, 0), None);
        assert_eq!(mask_bbox(&GrayImage::new(64, 64), 128, 2), None);
        assert_eq!(mask_bbox(&mask(64, 64, &[(10, 20, 1)]), 1, 0), bbox(10, 20, 1, 1), "a tight box");
        assert_eq!(mask_bbox(&mask(32, 32, &[(0, 0, 255)]), 128, 4), bbox(0, 0, 5, 5), "the margin clips at the corner");
        let square: Vec<_> = (20..30).flat_map(|y| (25..35).map(move |x| (x, y, 255))).collect();
        assert_eq!(mask_bbox(&mask(64, 64, &square), 128, 4), bbox(21, 16, 18, 18), "padded on every side");
        assert_eq!(mask_bbox(&GrayImage::from_pixel(32, 32, Luma([255])), 1, 0), bbox(0, 0, 32, 32));
        let low = mask(32, 32, &[(10, 10, 100)]);
        assert!(mask_bbox(&low, 1, 0).is_some() && mask_bbox(&low, 128, 0).is_none(), "100 counts at 1, not at 128");
    }

    /// No amount, no mask, or a mask of another size: the image as it was.
    #[test]
    fn sharpen_inpainted_with_nothing_to_do_leaves_the_image() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 255]));
        let mut dot = GrayImage::new(16, 16);
        dot.put_pixel(8, 8, Luma([255]));
        for (mask, amount) in [(dot, 0.0), (GrayImage::new(16, 16), 0.5), (GrayImage::new(8, 8), 0.5)] {
            assert_eq!(applied(&img, |o| sharpen_inpainted(o, &mask, amount)), img, "{:?} at {amount}", mask.dimensions());
        }
    }

    /// Bit-exact against the full-image sharpen, for a mask in a corner
    /// (the margin clamps at two edges), in the middle (no clamp), and
    /// across the right and bottom edges.
    #[test]
    fn sharpen_inpainted_matches_reference_full_image_computation() {
        for (side, x0, y0, x1, y1, amount) in [(48, 0, 0, 16, 16, 0.5), (64, 24, 24, 40, 40, 0.7), (48, 36, 36, 48, 48, 0.5)] {
            let img = gradient_image(side, side);
            let mask = rect_mask(side, side, (x0, y0, x1, y1));
            assert_bbox_sharpen_matches_reference(&img, &mask, amount);
        }
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
        let mask = rect_mask(128, 128, (56, 56, 72, 72));
        // Inpaint closure paints the masked region pure red.
        let out = tile_compose(&img, &mask, &Progress::none(), |tile, tile_mask| {
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
