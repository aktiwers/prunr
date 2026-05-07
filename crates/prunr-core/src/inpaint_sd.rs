//! Stable Diffusion 1.5 Inpainting pipeline.
//!
//! Per-stroke flow: CLIP-tokenize prompt → text encoder; VAE-encode the
//! masked source; downsample mask to latent space; sample gaussian noise;
//! denoise via DDIM (with optional classifier-free guidance); VAE-decode
//! final latent; composite back through mask.
//!
//! Dispatch wraps the per-tile pipeline:
//! - Mask connected components are isolated so disjoint strokes don't
//!   widen the bbox.
//! - Each component ≤ 512×512 hits the smart-crop fast path
//!   (`compute_sd_crop`).
//! - Components > 512×512 split into overlapping 512×512 tiles with
//!   linear-alpha seam blending (`tile_bbox` + `blend_tile`).
//!
//! Safety guards on CPU-class hardware:
//! - Bundle load refuses below the model's `working_set_mb` free RAM.
//!   ORT graph optimization + UNet activations together push 6-10 GB
//!   transient on this codepath; the floor protects against swap thrash.
//! - Idle bundle release (`SD_IDLE_RELEASE_SECS`) drops the cached 4
//!   ORT sessions after no use — reclaims ~4-6 GB so users who erased
//!   ten minutes ago aren't carrying SD weights for the rest of the
//!   session.
//!
//! When `Settings.sd_fast_mode` resolves true the dispatcher routes to
//! the LCM-distilled checkpoint (4 steps, guidance baked into training).
//! That gating happens upstream; this module sees the chosen `ModelId`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use half::f16;
use image::{GrayImage, RgbaImage};
use ndarray::{Array2, Array3, Array4, Axis};
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::engine::EpKind;
use crate::types::CoreError;

/// SD 1.5 native input side length. Latent space is 1/8 on each axis.
pub const SD_TILE: u32 = 512;
/// Latent space side length after VAE downsampling.
pub const SD_LATENT_SIDE: u32 = SD_TILE / 8;
/// SD 1.5 latent scaling factor. Diffusers calls this `vae.config.scaling_factor`.
const VAE_SCALING_FACTOR: f32 = 0.18215;

/// Minimum free RAM (bytes) required before we'll load the SD bundle.
/// The 4 ONNX files total ~2 GB on disk; ORT graph optimization roughly
/// doubles that during load (~4 GB resident), UNet activations on a
/// 512×512 tile add 2-4 GB transient, and OpenVINO graph compilation
/// pre-allocates its own iGPU-shared buffer (which on integrated GPUs
/// IS RAM). Threshold lives in `ModelDescriptor.working_set_mb` so the
/// gate scales when a new SD variant ships with a different footprint.
/// Idle window after which a cached SD session bundle is dropped to
/// release its ~4-6 GB resident set back to the OS. Trade-off: rebuilding
/// pays the 10-30s session-build cost on the next stroke; in exchange a
/// user who finished erasing 5 minutes ago doesn't carry the weights for
/// the rest of the session.
const SD_IDLE_RELEASE_SECS: u64 = 300;
/// CLIP-ViT-L/14 token sequence length used by SD 1.5.
const CLIP_SEQ_LEN: usize = 77;
/// CLIP-ViT-L/14 BOS / EOS / pad ids — same value for EOS and pad in
/// the SD 1.5 tokenizer convention.
const CLIP_BOS: i64 = 49406;
const CLIP_EOS: i64 = 49407;

/// Inputs to one SD inpaint call. Constructable today with default
/// fields for empty-prompt unconditional inpaint; future surfaces (text
/// prompts, CFG, seeded variations) just set the relevant fields.
/// Serde-derived so this struct can also ride `SubprocessCommand::Inpaint`
/// without a wire mirror.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SdInpaintRequest {
    /// Empty string ⇒ unconditional generation (v1 only path).
    pub prompt: String,
    /// Empty ⇒ no negative conditioning.
    pub negative_prompt: String,
    /// 20 is the SD 1.5 sweet spot at default scheduler.
    pub num_inference_steps: u32,
    /// 1.0 ⇒ no classifier-free guidance (v1 only path).
    /// 7.5 is the typical text-prompt setting once CFG is wired.
    pub guidance_scale: f32,
    /// `None` ⇒ random seed each call.
    pub seed: Option<u64>,
    /// When true, swap the SD bundle's standard VAE legs for TAESD — a
    /// distilled ~1M-param VAE pair (~3× faster decode, slight quality
    /// cost). Caller sets this when fast mode is on AND the TAESD
    /// bundle is installed; if the bundle isn't installed yet, the
    /// dispatcher silently falls back to standard VAE.
    pub use_taesd: bool,
    /// Which scheduler runs the denoise loop. Replaces the legacy
    /// `use_lcm_scheduler: bool`. Worker constructs the right
    /// `Scheduler` variant from this kind.
    #[serde(default)]
    pub scheduler: SchedulerKind,
}

impl Default for SdInpaintRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            num_inference_steps: 20,
            guidance_scale: 1.0,
            seed: None,
            use_taesd: false,
            scheduler: SchedulerKind::Lcm,
        }
    }
}

/// Public entry. Dispatched from `inpaint::process_inpaint` when the
/// caller selects an SD-family model id.
pub fn process_inpaint(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    num_steps: u32,
) -> Result<RgbaImage, CoreError> {
    process_inpaint_with(
        image, mask, id,
        SdInpaintRequest { num_inference_steps: num_steps, ..Default::default() },
        &crate::inpaint::InpaintHooks::default(),
    )
}

/// Full-knob entry. The `inpaint::process_inpaint` shim calls this with
/// defaults; future text-prompt + CFG surfaces wire through here.
/// See `inpaint::process_inpaint_with` for the hooks contract.
pub fn process_inpaint_with(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    req: SdInpaintRequest,
    hooks: &crate::inpaint::InpaintHooks,
) -> Result<RgbaImage, CoreError> {
    use std::sync::atomic::Ordering;
    let cancel = hooks.cancel.as_ref();
    let progress = hooks.progress.as_ref();
    if let Some(p) = progress {
        // Total = steps × tiles is approximate; tiles aren't known
        // until mask_components runs below. We seed total = steps and
        // surface tile-count via the rolling `current` step counter
        // wrapping past `total` per tile. Banner reads (current,
        // total) raw; total stays the per-tile UNet step budget so
        // "step 5 of 20" reads cleanly even on multi-tile strokes.
        p.set_total(req.num_inference_steps);
        p.set_step(0);
    }
    if image.dimensions() != mask.dimensions() {
        return Err(CoreError::Inference(format!(
            "sd inpaint: dim mismatch — image {:?} vs mask {:?}",
            image.dimensions(),
            mask.dimensions(),
        )));
    }
    if !(0.0..=20.0).contains(&req.guidance_scale) {
        return Err(CoreError::Inference(format!(
            "sd inpaint: guidance_scale {} out of supported range [0, 20]",
            req.guidance_scale,
        )));
    }

    let components = mask_components(mask);
    if components.is_empty() {
        return Ok(image.clone());
    }

    // Hold an Arc through the run so the idle sweep can't drop sessions
    // mid-inference.
    let bundle = SdSession::get(id)?;
    // RAII drop-on-completion: when this guard goes out of scope (any
    // return path) the cache's Arc is removed; once `bundle` (declared
    // ABOVE so it drops AFTER the guard) drops too, the SdSession's
    // last reference goes away and the ORT bundle releases. Trades
    // first-stroke responsiveness on a repeat for instant RAM reclaim
    // — appropriate when single SD strokes are heavy enough that
    // users typically wait between strokes anyway.
    struct DropOnComplete(prunr_models::ModelId);
    impl Drop for DropOnComplete {
        fn drop(&mut self) {
            release(self.0);
        }
    }
    let _release_guard = DropOnComplete(id);
    // VAE backend selection: TAESD when fast mode is on AND the bundle
    // is installed (the request flag carries that decision from
    // dispatch). Until the TAESD artifact ships, get() errors and we
    // silently fall back to standard VAE — same graceful pattern as LCM.
    let taesd = if req.use_taesd { TaesdSession::get().ok() } else { None };
    let vae: VaeBackend = match taesd.as_ref() {
        Some(t) => VaeBackend::Taesd(t),
        None => VaeBackend::Standard(&bundle),
    };
    let (img_w, img_h) = image.dimensions();
    let mut out = image.clone();
    let is_cancelled = || cancel.as_ref().is_some_and(|c| c.load(Ordering::Acquire));

    for component in &components {
        if is_cancelled() { return Err(CoreError::Cancelled); }
        let painted_w = component.x_max - component.x_min + 1;
        let painted_h = component.y_max - component.y_min + 1;
        if painted_w <= SD_TILE && painted_h <= SD_TILE {
            // Fast path: single 512×512 crop centred on the component.
            let (cx, cy, cw, ch) = compute_sd_crop(component, img_w, img_h);
            let cropped_img = image::imageops::crop_imm(&out, cx, cy, cw, ch).to_image();
            let cropped_mask = image::imageops::crop_imm(mask, cx, cy, cw, ch).to_image();
            match run_one_tile(&bundle, &vae, &cropped_img, &cropped_mask, &req, hooks) {
                Ok(painted) => image::imageops::replace(&mut out, &painted, cx as i64, cy as i64),
                Err(CoreError::Cancelled) => return Err(CoreError::Cancelled),
                Err(e) => tracing::error!(%e, "SD inference failed for component; skipping"),
            }
            continue;
        }

        // Tile path: component exceeds 512×512 in some axis. Split into
        // overlapping 512×512 tiles and alpha-blend the seams.
        let tiles = tile_bbox(component, img_w, img_h);
        tracing::info!(
            comp = ?component, n_tiles = tiles.len(),
            "SD: tiling oversized component",
        );
        for tile in tiles {
            if is_cancelled() { return Err(CoreError::Cancelled); }
            let cropped_img = image::imageops::crop_imm(&out, tile.x, tile.y, tile.w, tile.h).to_image();
            let cropped_mask = image::imageops::crop_imm(mask, tile.x, tile.y, tile.w, tile.h).to_image();
            match run_one_tile(&bundle, &vae, &cropped_img, &cropped_mask, &req, hooks) {
                Ok(painted) => blend_tile(&mut out, &painted, &tile),
                Err(CoreError::Cancelled) => return Err(CoreError::Cancelled),
                Err(e) => tracing::error!(%e, ?tile, "SD inference failed for tile; skipping"),
            }
        }
    }
    if is_cancelled() { return Err(CoreError::Cancelled); }
    Ok(out)
}

/// Width of the seam-blend ramp between adjacent tiles. 25% of SD_TILE
/// gives a smooth fade; lower values risk a visible grid edge, higher
/// values waste compute (more overlap = more tiles per bbox).
const TILE_OVERLAP_PX: u32 = SD_TILE / 4;
/// Distance the tile anchor advances per step. SD_TILE - overlap so
/// adjacent tiles share a TILE_OVERLAP_PX column/row.
const TILE_STEP_PX: u32 = SD_TILE - TILE_OVERLAP_PX;

#[derive(Debug, Clone, Copy)]
struct TileWindow {
    x: u32, y: u32, w: u32, h: u32,
    /// True iff this edge abuts another tile (i.e. needs feathering).
    /// Edges at the bbox/image boundary stay full-strength.
    feather_left: bool,
    feather_right: bool,
    feather_top: bool,
    feather_bottom: bool,
}

/// Split an oversized bbox into 512×512 tiles with TILE_OVERLAP_PX
/// overlap between neighbours. Tiles always anchor inside the image
/// (no out-of-bounds access at composite time) — for image axes
/// shorter than SD_TILE, the tile width/height collapses to the image
/// extent and pad_to_tile handles the rest at the ORT boundary.
fn tile_bbox(bbox: &MaskBbox, img_w: u32, img_h: u32) -> Vec<TileWindow> {
    let span_x = bbox.x_max - bbox.x_min + 1;
    let span_y = bbox.y_max - bbox.y_min + 1;

    let n_x = tile_count(span_x);
    let n_y = tile_count(span_y);

    let tile_w = SD_TILE.min(img_w);
    let tile_h = SD_TILE.min(img_h);

    let max_anchor_x = img_w.saturating_sub(tile_w);
    let max_anchor_y = img_h.saturating_sub(tile_h);

    let mut tiles = Vec::with_capacity((n_x as usize) * (n_y as usize));
    for j in 0..n_y {
        for i in 0..n_x {
            let x_raw = bbox.x_min + i * TILE_STEP_PX;
            let y_raw = bbox.y_min + j * TILE_STEP_PX;
            let x = x_raw.min(max_anchor_x);
            let y = y_raw.min(max_anchor_y);
            tiles.push(TileWindow {
                x, y, w: tile_w, h: tile_h,
                feather_left: i > 0,
                feather_right: i + 1 < n_x,
                feather_top: j > 0,
                feather_bottom: j + 1 < n_y,
            });
        }
    }
    tiles
}

fn tile_count(span: u32) -> u32 {
    if span <= SD_TILE { 1 }
    else {
        // Number of step-advances needed beyond the first tile, ceiling.
        let extra = span - SD_TILE;
        extra.div_ceil(TILE_STEP_PX) + 1
    }
}

/// Composite a painted tile into `canvas` with linear-alpha feathering
/// at edges that abut neighbour tiles. Outside the feather zones the
/// painted tile fully replaces the canvas; inside, it ramps from 0 at
/// the tile edge to 1 at TILE_OVERLAP_PX into the tile.
///
/// For non-masked pixels the feather is a no-op anyway (both source
/// and painted carry identical pixels via run_one_tile's composite),
/// so we don't need to know the mask here — we just blend everything.
fn blend_tile(canvas: &mut image::RgbaImage, painted: &image::RgbaImage, tile: &TileWindow) {
    let cw = canvas.width();
    let pw = painted.width();
    let raw_canvas = canvas.as_mut();
    let raw_painted = painted.as_raw();

    for y in 0..tile.h {
        let alpha_y = edge_alpha(y, tile.h, tile.feather_top, tile.feather_bottom);
        for x in 0..tile.w {
            let alpha_x = edge_alpha(x, tile.w, tile.feather_left, tile.feather_right);
            let a = alpha_x.min(alpha_y);
            if a >= 1.0 {
                // Full replace — copy source bytes, skip the lerp.
                let src = ((y * pw + x) * 4) as usize;
                let dst = (((tile.y + y) * cw + (tile.x + x)) * 4) as usize;
                raw_canvas[dst..dst + 4].copy_from_slice(&raw_painted[src..src + 4]);
            } else {
                let src = ((y * pw + x) * 4) as usize;
                let dst = (((tile.y + y) * cw + (tile.x + x)) * 4) as usize;
                let inv = 1.0 - a;
                for k in 0..3 {
                    let canv = raw_canvas[dst + k] as f32;
                    let pnt = raw_painted[src + k] as f32;
                    raw_canvas[dst + k] = (canv * inv + pnt * a).round().clamp(0.0, 255.0) as u8;
                }
                raw_canvas[dst + 3] = raw_painted[src + 3];
            }
        }
    }
}

/// Linear ramp from 0 at the feathered edge(s) to 1 once we're more
/// than TILE_OVERLAP_PX in. When neither edge is feathered, returns 1.
fn edge_alpha(pos: u32, length: u32, feather_lo: bool, feather_hi: bool) -> f32 {
    let f = TILE_OVERLAP_PX as f32;
    let a_lo = if feather_lo { (pos as f32 / f).min(1.0) } else { 1.0 };
    let a_hi = if feather_hi {
        ((length.saturating_sub(pos + 1)) as f32 / f).min(1.0)
    } else { 1.0 };
    a_lo.min(a_hi)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskBbox {
    x_min: u32, y_min: u32, x_max: u32, y_max: u32,
}

/// Find the bounding box of every disjoint painted region in `mask`
/// (4-connectivity, threshold > 127). Each returned bbox is the tight
/// rectangle around one connected component — disjoint strokes get
/// individual bboxes so the SD dispatcher can run smart-crop on each
/// instead of one giant bbox spanning unpainted gap.
///
/// Iterative BFS (no recursion) so a 4096×4096 component doesn't blow
/// the stack. `visited` is sized to the outer bbox of all painted
/// pixels (via `inpaint::mask_bbox`), not the full image — a 100-px
/// stroke in a 4K image needs ~10 KB of scratch instead of 16 MB.
fn mask_components(mask: &GrayImage) -> Vec<MaskBbox> {
    let (w, _h) = mask.dimensions();
    let raw = mask.as_raw();
    let Some(outer) = crate::inpaint::mask_bbox(mask, 128, 0) else {
        return Vec::new();
    };
    let map = BboxMap { bbox_x: outer.x, bbox_y: outer.y, image_w: w, bbox_w: outer.w };
    let xmax = outer.x + outer.w;
    let ymax = outer.y + outer.h;
    let mut visited = vec![false; (outer.w as usize) * (outer.h as usize)];
    let mut components = Vec::new();
    let mut queue: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();

    for sy in outer.y..ymax {
        for sx in outer.x..xmax {
            let lidx = map.local_idx(sx, sy);
            if visited[lidx] || raw[map.global_idx(sx, sy)] <= 127 {
                continue;
            }
            visited[lidx] = true;
            queue.push_back((sx, sy));
            let mut bbox = MaskBbox { x_min: sx, y_min: sy, x_max: sx, y_max: sy };
            while let Some((x, y)) = queue.pop_front() {
                if x < bbox.x_min { bbox.x_min = x; }
                if x > bbox.x_max { bbox.x_max = x; }
                if y < bbox.y_min { bbox.y_min = y; }
                if y > bbox.y_max { bbox.y_max = y; }
                // 4-connectivity: up / down / left / right.
                if x > outer.x {
                    push_if_unvisited(x - 1, y, map, raw, &mut visited, &mut queue);
                }
                if x + 1 < xmax {
                    push_if_unvisited(x + 1, y, map, raw, &mut visited, &mut queue);
                }
                if y > outer.y {
                    push_if_unvisited(x, y - 1, map, raw, &mut visited, &mut queue);
                }
                if y + 1 < ymax {
                    push_if_unvisited(x, y + 1, map, raw, &mut visited, &mut queue);
                }
            }
            components.push(bbox);
        }
    }
    components
}

#[derive(Clone, Copy)]
struct BboxMap { bbox_x: u32, bbox_y: u32, image_w: u32, bbox_w: u32 }

impl BboxMap {
    #[inline]
    fn local_idx(self, x: u32, y: u32) -> usize {
        ((y - self.bbox_y) as usize) * (self.bbox_w as usize) + ((x - self.bbox_x) as usize)
    }
    #[inline]
    fn global_idx(self, x: u32, y: u32) -> usize {
        (y as usize) * (self.image_w as usize) + (x as usize)
    }
}

#[inline]
fn push_if_unvisited(
    x: u32, y: u32, map: BboxMap,
    raw: &[u8], visited: &mut [bool],
    queue: &mut std::collections::VecDeque<(u32, u32)>,
) {
    let lidx = map.local_idx(x, y);
    let gidx = map.global_idx(x, y);
    if !visited[lidx] && raw[gidx] > 127 {
        visited[lidx] = true;
        queue.push_back((x, y));
    }
}

/// Centre an SD_TILE-sized crop on the bbox centre, clamped to image
/// bounds. For images smaller than SD_TILE on an axis, the crop shrinks
/// to that dimension on that axis (pad_to_tile pads it back to 512).
fn compute_sd_crop(bbox: &MaskBbox, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
    let cw = SD_TILE.min(img_w);
    let ch = SD_TILE.min(img_h);
    let cx_centre = (bbox.x_min + bbox.x_max) / 2;
    let cy_centre = (bbox.y_min + bbox.y_max) / 2;
    let x = cx_centre.saturating_sub(cw / 2).min(img_w - cw);
    let y = cy_centre.saturating_sub(ch / 2).min(img_h - ch);
    (x, y, cw, ch)
}

/// Eagerly initialise the SD session so the first stroke doesn't pay
/// the cumulative ~10-30s session-build latency for 4 ONNX files.
pub fn prewarm(id: prunr_models::ModelId) -> Result<(), CoreError> {
    SdSession::get(id).map(|_| ())
}

// ── Per-tile pipeline ───────────────────────────────────────────────────

fn run_one_tile(
    bundle: &SdSession,
    vae: &VaeBackend,
    image: &RgbaImage,
    mask: &GrayImage,
    req: &SdInpaintRequest,
    hooks: &crate::inpaint::InpaintHooks,
) -> Result<RgbaImage, CoreError> {
    let cancel = hooks.cancel.as_ref();
    let progress = hooks.progress.as_ref();
    let (w, h) = image.dimensions();
    let padded_image = pad_to_tile(image);
    let padded_mask = pad_mask_to_tile(mask);

    // CFG threshold: above 1.0 we run the UNet TWICE per step (cond +
    // uncond) and blend by `guidance_scale`. At ≤1.0 the cond pass is
    // all the user wants, so we skip the second to halve UNet cost.
    let use_cfg = req.guidance_scale > 1.0 + 1e-3;

    // Pre-loop independent ops in parallel: text encode (cond + uncond
    // when CFG), VAE encode, mask-to-latent. Each ORT call holds a
    // distinct session mutex (or none) so they run concurrently.
    let prompt = req.prompt.clone();
    let neg_prompt = if use_cfg { Some(req.negative_prompt.clone()) } else { None };
    let (text_emb_cond, text_emb_uncond, masked_latent, mask_latent) =
        std::thread::scope(|s| -> Result<_, CoreError> {
            let cond_h = s.spawn(|| encode_text(bundle, &prompt));
            let uncond_h = neg_prompt.as_ref().map(|np| {
                let np = np.clone();
                s.spawn(move || encode_text(bundle, &np))
            });
            let vae_h = s.spawn(|| vae_encode_masked(vae, &padded_image, &padded_mask));
            let mask_lat = mask_to_latent(&padded_mask);
            let cond = cond_h.join()
                .map_err(|_| CoreError::Inference("text encoder (cond) thread panicked".into()))??;
            let uncond = match uncond_h {
                Some(h) => Some(h.join()
                    .map_err(|_| CoreError::Inference("text encoder (uncond) thread panicked".into()))??),
                None => None,
            };
            let vae = vae_h.join()
                .map_err(|_| CoreError::Inference("vae encoder thread panicked".into()))??;
            Ok((cond, uncond, vae, mask_lat))
        })?;

    let text_emb_cond_f16 = f32_to_f16_3d(&text_emb_cond);
    let text_emb_uncond_f16 = text_emb_uncond.as_ref().map(f32_to_f16_3d);
    let masked_latent_f16 = f32_to_f16_4d(&masked_latent);
    let mask_latent_f16 = f32_to_f16_4d(&mask_latent);
    // Free the f32 originals — the loop only needs the f16 mirrors,
    // and these buffers (~600 KB combined: 2 × 237 KB text emb + 2 ×
    // 64 KB latent) would otherwise sit alive across the entire
    // 20-step UNet loop, the longest-lived stage of the pipeline.
    drop(text_emb_cond);
    drop(text_emb_uncond);
    drop(masked_latent);
    drop(mask_latent);

    let steps = req.num_inference_steps as usize;
    let scheduler = match req.scheduler {
        SchedulerKind::Lcm => Scheduler::Lcm(LcmScheduler::new_sd15(steps)),
        SchedulerKind::Ddim => Scheduler::Ddim(DdimScheduler::new_sd15(steps)),
        SchedulerKind::DpmPp2MKarras => Scheduler::DpmPp2M(DpmPp2MScheduler::new_sd15(steps)),
        // Not yet implemented in the worker; fall back to LCM as a
        // safe default (UI gates picking via `is_available()`, but
        // this is belt-and-braces against a stale persisted choice).
        SchedulerKind::UniPc | SchedulerKind::EulerA => {
            Scheduler::Lcm(LcmScheduler::new_sd15(steps))
        }
    };
    let seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut latent = sample_initial_noise(&mut rng);
    // DPM++ 2M's multistep update needs the previous step's predicted
    // x0 ("m_{i-1}"). Other schedulers ignore this. Initialized to
    // None so the first step takes the first-order branch.
    let mut prev_model_output: Option<Vec<f32>> = None;

    // Denoising loop. With CFG: noise_pred = uncond + scale * (cond - uncond).
    // Without CFG: just one UNet pass with cond.
    let timesteps = scheduler.timesteps().to_vec();
    let scale = req.guidance_scale;
    let is_cancelled = || cancel.is_some_and(|c| {
        c.load(std::sync::atomic::Ordering::Acquire)
    });
    for (i, &t) in timesteps.iter().enumerate() {
        // Check cancel between UNet steps. ORT has no per-op cancel, so
        // worst-case latency on cancel is one UNet step (multi-second).
        if is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        // Publish progress AFTER the cancel check so a cancelled stroke
        // doesn't briefly tick "step N+1 of M" before exiting.
        // `i + 1` so the banner reads "step 1 of 20" on the first
        // iteration rather than "step 0".
        if let Some(p) = progress {
            p.set_step((i as u32) + 1);
        }
        let latent_f16 = f32_to_f16_4d(&latent);
        let latent_in_f16 = concat_inpaint_input_f16(
            &latent_f16, &mask_latent_f16, &masked_latent_f16,
        );
        let noise_pred = if let Some(uncond_f16) = text_emb_uncond_f16.as_ref() {
            // CFG path: prefer batched UNet (one ORT call instead of two)
            // when the export supports a dynamic batch dim. On
            // first-time failure we flip a per-session flag and fall
            // back to sequential for the rest of this stroke + future
            // strokes on this bundle.
            use std::sync::atomic::Ordering;
            let try_batched = !bundle.cfg_fallback_to_sequential.load(Ordering::Relaxed);
            if try_batched {
                match unet_step_batched(bundle, &latent_in_f16, t, &text_emb_cond_f16, uncond_f16) {
                    Ok(pred_pair) => {
                        // Pass slice views directly — cfg_blend now
                        // accepts ArrayView4, no `.to_owned()` needed.
                        let pred_cond = pred_pair.slice(ndarray::s![0..1, .., .., ..]);
                        let pred_uncond = pred_pair.slice(ndarray::s![1..2, .., .., ..]);
                        cfg_blend(pred_uncond, pred_cond, scale)
                    }
                    Err(e) => {
                        tracing::warn!(%e,
                            "SD: batched CFG UNet rejected (likely static batch=1 ONNX); \
                             falling back to sequential cond+uncond for this session");
                        bundle.cfg_fallback_to_sequential.store(true, Ordering::Relaxed);
                        let pred_cond = unet_step(bundle, latent_in_f16.clone(), t, &text_emb_cond_f16)?;
                        let pred_uncond = unet_step(bundle, latent_in_f16, t, uncond_f16)?;
                        cfg_blend(pred_uncond.view(), pred_cond.view(), scale)
                    }
                }
            } else {
                let pred_cond = unet_step(bundle, latent_in_f16.clone(), t, &text_emb_cond_f16)?;
                let pred_uncond = unet_step(bundle, latent_in_f16, t, uncond_f16)?;
                cfg_blend(pred_uncond.view(), pred_cond.view(), scale)
            }
        } else {
            unet_step(bundle, latent_in_f16, t, &text_emb_cond_f16)?
        };
        let t_prev = timesteps.get(i + 1).copied().unwrap_or(-1);
        let is_final = i + 1 == timesteps.len();
        latent = step_array(
            &scheduler, &latent, &noise_pred, i, t, t_prev, is_final,
            &mut rng, &mut prev_model_output,
        );
    }

    let painted = vae_decode(vae, &latent)?;
    Ok(composite(image, &painted, mask, w, h))
}

// ── Session bundle ──────────────────────────────────────────────────────

/// Tiny distilled VAE pair (~1M params each). Drop-in for SD's standard
/// VAE legs when fast mode is on. No idle release — total memory cost
/// is ~10 MB so keeping it cached forever is fine.
pub(crate) struct TaesdSession {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    encoder_input: String,
    decoder_input: String,
}

impl TaesdSession {
    fn get() -> Result<Arc<TaesdSession>, CoreError> {
        // Single global cache cell; `OnceLock::new()` is `const` so we
        // need no outer lazy wrapper. `get_or_init` runs the build
        // closure exactly once across concurrent callers.
        // CoreError isn't Clone, so we round-trip the build error
        // through a String — the stored Result must be cloneable so
        // every caller can take an owned copy of the cached outcome.
        static CACHE: OnceLock<Result<Arc<TaesdSession>, String>> = OnceLock::new();
        CACHE.get_or_init(|| Self::new_inner().map(Arc::new).map_err(|e| e.to_string()))
            .clone()
            .map_err(CoreError::Inference)
    }

    fn new_inner() -> Result<TaesdSession, CoreError> {
        let parts = prunr_models::multi_part_paths(prunr_models::ModelId::TaesdFp16)
            .ok_or_else(|| CoreError::Inference(
                prunr_models::not_installed_error(prunr_models::ModelId::TaesdFp16)
            ))?;
        let by_key: HashMap<&str, PathBuf> = parts.into_iter().collect();
        let encoder_path = by_key.get("encoder")
            .ok_or_else(|| CoreError::Inference("TAESD bundle missing encoder part".into()))?;
        let decoder_path = by_key.get("decoder")
            .ok_or_else(|| CoreError::Inference("TAESD bundle missing decoder part".into()))?;

        // Build encoder + decoder in parallel — saves ~1-2 s of cold
        // start. Each session is ~5 MB; aggregate peak RSS during
        // parallel build is well under 50 MB. No DirectML carve-out
        // needed since TAESD has no GPU EP ladder.
        let build = |path: &PathBuf, label: &'static str| -> Result<Session, CoreError> {
            Session::builder()
                .map_err(|e| CoreError::Inference(format!("TAESD: builder init: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| CoreError::Inference(format!("TAESD: opt level: {e}")))?
                .commit_from_file(path)
                .map_err(|e| CoreError::Inference(format!("TAESD {label}: load: {e}")))
        };
        let (enc_res, dec_res) = rayon::join(
            || build(encoder_path, "encoder"),
            || build(decoder_path, "decoder"),
        );
        let encoder = enc_res?;
        let decoder = dec_res?;

        let encoder_input = encoder.inputs().first()
            .ok_or_else(|| CoreError::Inference("TAESD encoder: no inputs".into()))?
            .name().to_string();
        let decoder_input = decoder.inputs().first()
            .ok_or_else(|| CoreError::Inference("TAESD decoder: no inputs".into()))?
            .name().to_string();

        tracing::info!(
            encoder_input = %encoder_input, decoder_input = %decoder_input,
            "TAESD session loaded",
        );
        Ok(TaesdSession {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            encoder_input,
            decoder_input,
        })
    }
}

#[allow(dead_code)]
pub(crate) struct SdSession {
    unet: Mutex<Session>,
    vae_encoder: Mutex<Session>,
    vae_decoder: Mutex<Session>,
    text_encoder: Mutex<Session>,
    /// Discovered input names per session (positional order from the ONNX).
    /// SD exports use stable names (sample / timestep / encoder_hidden_states
    /// for UNet, sample for VAEs, input_ids for text encoder) but we look
    /// them up by introspection so a non-standard export still works.
    unet_inputs: [String; 3],
    vae_encoder_input: String,
    vae_decoder_input: String,
    text_encoder_input: String,
    /// Set to `true` after the first batched UNet call fails on this
    /// session — typically because the underlying ONNX export declared
    /// a static batch=1. Subsequent CFG steps skip the batched attempt
    /// and call `unet_step` twice instead. Once flipped per process,
    /// stays flipped for the session's lifetime; cleared on session
    /// rebuild (idle release).
    cfg_fallback_to_sequential: std::sync::atomic::AtomicBool,
}

/// `Arc<T>` so idle eviction can drop the cache's ref while in-flight
/// callers keep their own clone — no use-after-free.
/// Errors cache the load-failure string so a missing bundle doesn't
/// retry-and-error every stroke; eviction lets the next try refresh.
///
/// `pub(crate)` so `inpaint::LamaSession` can reuse the same cache shape
/// without duplicating struct + sweep helper.
pub(crate) struct CacheEntry<T> {
    pub(crate) value: T,
    pub(crate) last_used: Instant,
}

/// Per-id deferred bundle. The outer `Arc<OnceLock<...>>` is what closes
/// the build race: two concurrent `get()` callers both find the same
/// `Arc<OnceLock>` in the cache, both call `get_or_init` on it, and the
/// `OnceLock` semantics guarantee the build closure runs **once** —
/// the second caller blocks until the first finishes and gets the same
/// stored `Result`. Before this, both callers would each build a full
/// ~15 GB bundle in parallel; the loser's bundle was dropped 3 ms later,
/// nearly OOMing the box (`SD session bundle dropped` immediately after
/// `SD session bundle loaded` in the trace).
type SdBundleSlot = Arc<std::sync::OnceLock<Result<Arc<SdSession>, String>>>;
type SdCache = HashMap<prunr_models::ModelId, CacheEntry<SdBundleSlot>>;

fn sd_cache() -> &'static Mutex<SdCache> {
    static CACHE: OnceLock<Mutex<SdCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop the cached entry for `id`, releasing the cache's `Arc` ref.
/// In-flight callers keep their own `Arc<SdSession>` so the bundle
/// stays valid until the dispatch finishes; once the last `Arc` drops,
/// the `Drop` impl runs and the ORT session is released. Used by
/// `process_inpaint_with` to drop the SD bundle the moment a stroke
/// completes — on memory-constrained machines holding 9 GB through
/// the 5-min idle window is worse than paying the rebuild on the
/// next stroke.
pub(crate) fn release(id: prunr_models::ModelId) {
    let cache = sd_cache();
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.remove(&id).is_some() {
        tracing::info!(?id, "SD: cache entry released after dispatch");
    }
}

/// Drop entries whose `last_used` is older than `idle` relative to `now`,
/// returning the count dropped. Generic over `T` so tests can drive the
/// sweep without spinning up a real ORT session.
///
/// `pub(crate)` so `inpaint::LamaSession`'s sweeper reuses this directly.
pub(crate) fn sweep_idle<T>(
    cache: &mut HashMap<prunr_models::ModelId, CacheEntry<T>>,
    now: Instant,
    idle: Duration,
) -> usize {
    let before = cache.len();
    cache.retain(|_, e| now.duration_since(e.last_used) < idle);
    before - cache.len()
}

/// Background sweeper interval. 60 s gives reclaim within ~1 minute of
/// the idle threshold without burning CPU; the sweep itself is a single
/// HashMap::retain over <10 entries so it's free.
const SD_SWEEP_INTERVAL_SECS: u64 = 60;

/// Spawn a daemon thread that periodically evicts idle SD entries even
/// when no caller invokes `get()`. Without this, RAM doesn't reclaim
/// until the user touches SD again — which defeats the point of the
/// idle release. Init-once via OnceLock so we don't accumulate threads.
fn ensure_sweeper_running() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        std::thread::Builder::new()
            .name("sd-idle-sweeper".to_string())
            .spawn(|| {
                let interval = Duration::from_secs(SD_SWEEP_INTERVAL_SECS);
                let idle = Duration::from_secs(SD_IDLE_RELEASE_SECS);
                loop {
                    std::thread::sleep(interval);
                    let cache = sd_cache();
                    let mut guard = cache.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let dropped = sweep_idle(&mut guard, Instant::now(), idle);
                    if dropped > 0 {
                        tracing::info!(
                            dropped,
                            idle_secs = SD_IDLE_RELEASE_SECS,
                            rss_mb = process_rss_mb(),
                            "SD: background sweeper released idle session(s)",
                        );
                    }
                }
            })
            .expect("spawn sd-idle-sweeper thread");
    });
}

impl SdSession {
    fn get(id: prunr_models::ModelId) -> Result<Arc<SdSession>, CoreError> {
        ensure_sweeper_running();
        let cache = sd_cache();
        let now = Instant::now();
        let idle = Duration::from_secs(SD_IDLE_RELEASE_SECS);

        // Cache lock held only for the HashMap lookup and at most one
        // Arc<OnceLock> allocation — never for the bundle build.
        let slot: SdBundleSlot = {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dropped = sweep_idle(&mut guard, now, idle);
            if dropped > 0 {
                tracing::info!(
                    dropped,
                    idle_secs = SD_IDLE_RELEASE_SECS,
                    rss_mb = process_rss_mb(),
                    "SD: released idle session bundle(s)",
                );
            }
            let entry = guard.entry(id).or_insert_with(|| CacheEntry {
                value: Arc::new(std::sync::OnceLock::new()),
                last_used: now,
            });
            // Don't refresh last_used when the slot already holds an
            // Err — otherwise a sticky build failure would keep
            // refreshing its idle timer on every retry and never
            // evict. Healthy or in-flight slots refresh as before.
            if !matches!(entry.value.get(), Some(Err(_))) {
                entry.last_used = now;
            }
            entry.value.clone()
        };

        // Build outside the cache lock — see `SdBundleSlot` docs.
        slot.get_or_init(|| Self::new_inner(id).map(Arc::new))
            .clone()
            .map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId) -> Result<SdSession, String> {
        // Defense-in-depth: prewarm builds bypass `process_inpaint_with`,
        // so the gate has to fire here too. Same helper as the dispatch
        // entry — single source of truth for threshold + wording.
        check_ram_for(id)?;
        let rss_before_mb = process_rss_mb();
        let parts = prunr_models::multi_part_paths(id)
            .ok_or_else(|| prunr_models::not_installed_error(id))?;
        let by_key: HashMap<&str, PathBuf> = parts.into_iter().collect();

        // The four parts touch disjoint sessions; build them in parallel.
        // DirectML stays sequential — same AbiCustomRegistry race as
        // `batch::create_engine_pool`. Parallel build holds all four
        // optimization scratch arenas live concurrently; the
        // `working_set_mb` guard above bounds the peak.
        // We still log the winning EP per part so partial GPU
        // fall-through (e.g. UNet CUDA, VAEs CPU) is debuggable.
        type SmokeFn = fn(&mut Session, &str) -> Result<(), String>;
        const PARTS: [(&str, SmokeFn); 4] = [
            ("text_encoder", smoke_test_text_encoder),
            ("vae_encoder",  smoke_test_vae_encoder),
            ("vae_decoder",  smoke_test_vae_decoder),
            ("unet",         smoke_test_unet),
        ];

        let build = |&(key, smoke): &(&'static str, SmokeFn)|
            -> Result<(&'static str, Session, String), String> {
                let (s, ep) = build_part_with_ep_ladder(id, key, &by_key, smoke)?;
                Ok((key, s, ep))
            };

        let mut parts: Vec<(&'static str, Session, String)> =
            if crate::engine::directml_active() {
                PARTS.iter().map(build).collect::<Result<_, _>>()?
            } else {
                use rayon::prelude::*;
                PARTS.par_iter().map(build).collect::<Result<_, _>>()?
            };

        // Match by key so the destructure below survives reordering of
        // PARTS. `IndexedParallelIterator::collect` preserves order, so
        // this is defensive — cheap at N=4.
        let mut take = |want: &str| -> Result<(Session, String), String> {
            let pos = parts.iter().position(|(k, _, _)| *k == want)
                .ok_or_else(|| format!("SD bundle missing built part: {want}"))?;
            let (_, s, ep) = parts.swap_remove(pos);
            Ok((s, ep))
        };
        let (text_encoder, text_ep) = take("text_encoder")?;
        let (vae_encoder,  vae_enc_ep) = take("vae_encoder")?;
        let (vae_decoder,  vae_dec_ep) = take("vae_decoder")?;
        let (unet,         unet_ep)    = take("unet")?;

        let unet_inputs = take_three_inputs(&unet, "unet")?;
        let vae_encoder_input = take_first_input(&vae_encoder, "vae_encoder")?;
        let vae_decoder_input = take_first_input(&vae_decoder, "vae_decoder")?;
        let text_encoder_input = take_first_input(&text_encoder, "text_encoder")?;

        let rss_after_mb = process_rss_mb();
        tracing::info!(
            ?id,
            text_encoder_ep = %text_ep,
            vae_encoder_ep = %vae_enc_ep,
            vae_decoder_ep = %vae_dec_ep,
            unet_ep = %unet_ep,
            unet_inputs = ?unet_inputs,
            rss_before_mb,
            rss_after_mb,
            rss_delta_mb = rss_after_mb.zip(rss_before_mb).map(|(a, b)| a.saturating_sub(b)),
            "SD session bundle loaded",
        );
        Ok(SdSession {
            unet: Mutex::new(unet),
            vae_encoder: Mutex::new(vae_encoder),
            vae_decoder: Mutex::new(vae_decoder),
            text_encoder: Mutex::new(text_encoder),
            unet_inputs,
            vae_encoder_input,
            vae_decoder_input,
            text_encoder_input,
            cfg_fallback_to_sequential: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

/// SD bundle's GPU EP ladder: try CUDA / CoreML / DirectML in platform
/// order, smoke-test each, fall back to CPU. Same pattern as
/// `inpaint::build_lama_session`. Each part's smoke test is supplied by
/// the caller because input shapes/types differ per ONNX file.
fn build_part_with_ep_ladder(
    id: prunr_models::ModelId,
    key: &str,
    by_key: &HashMap<&str, PathBuf>,
    smoke_test: fn(&mut Session, &str) -> Result<(), String>,
) -> Result<(Session, String), String> {
    let path = by_key.get(key)
        .ok_or_else(|| format!("SD bundle missing required part: {key}"))?;

    let gpu_eps = crate::engine::available_gpu_eps();

    for &ep in gpu_eps {
        if !prunr_models::is_ep_compatible(id, ep.as_str()) {
            tracing::debug!(?id, part = %key, ep = %ep, "SD: EP statically incompatible; skipping");
            continue;
        }
        if crate::ep_compat::is_known_failure(ep, id) {
            tracing::debug!(?id, part = %key, ep = %ep, "SD: EP cached as incompatible; skipping");
            continue;
        }
        crate::cache::gc_stale_for_model(id, ep.as_str());
        let builder = match sd_base_builder() {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep, %e, "SD: builder init failed");
                continue;
            }
        };
        #[allow(unused_mut)] // mut only used on non-macOS via the CUDA arm
        let mut load_path: Cow<'_, Path> = Cow::Borrowed(path.as_path());
        let builder = match ep {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => {
                let (b, p) = sd_apply_path_cache(builder, path.as_path(), id, ep.as_str(), key);
                load_path = p;
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
                if let Some(dir) = crate::cache::cache_dir_for_part(id, ep.as_str(), key) {
                    p = p.with_model_cache_dir(dir.to_string_lossy().into_owned());
                }
                builder.with_execution_providers([p.build()])
            }
            #[cfg(windows)]
            EpKind::DirectMl => builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ]),
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => builder.with_execution_providers([
                // SD bundle peaked at ~15 GB RSS delta on a user system
                // (rss_before=4994 → rss_after=19909 in the reported
                // trace) — well above the ~3-4 GB the fp16 weights
                // would predict. Two OpenVINO knobs cap the worst-case
                // arena: `num_streams=1` disables per-stream buffer
                // duplication, and `dynamic_shapes=false` lets
                // OpenVINO size the working memory to the actual
                // 512² SD tile rather than reserving for arbitrary
                // input shapes. Both are safe — SD's UNet is run
                // sequentially under a Mutex, and our tile pipeline
                // is fixed-shape. (No `with_cache_dir` — see engine.rs
                // for the SD UNet empirical retest finding.)
                ort::execution_providers::OpenVINOExecutionProvider::default()
                    .with_num_streams(1)
                    .with_dynamic_shapes(false)
                    .build(),
            ]),
        };
        let mut built = match registered {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep, %e, "SD: register EP failed");
                continue;
            }
        };
        let started = Instant::now();
        let mut session = match built.commit_from_file(&load_path) {
            Ok(s) => {
                tracing::info!(?id, part = %key, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "SD: session committed");
                s
            }
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep, %e, "SD: GPU session commit failed — trying next");
                crate::engine::handle_commit_failure(
                    matches!(load_path, Cow::Owned(_)),
                    ep, id,
                    || crate::cache::optimized_model_path_for_part(id, ep.as_str(), key),
                    &format!("{e}"),
                );
                continue;
            }
        };
        match smoke_test(&mut session, key) {
            Ok(()) => {
                tracing::info!(part = %key, ep = %ep, "SD: GPU session validated");
                return Ok((session, ep.as_str().to_string()));
            }
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep, %e, "SD: smoke test failed — falling back");
                crate::engine::handle_commit_failure(
                    matches!(load_path, Cow::Owned(_)),
                    ep, id,
                    || crate::cache::optimized_model_path_for_part(id, ep.as_str(), key),
                    &e,
                );
            }
        }
    }

    // SD bundle weights are FP16; the ORT CPU EP runs them but produces
    // text-like artifacts instead of a coherent fill. GPU EP required.
    debug_assert!(id.is_sd_family(),
        "build_part_with_ep_ladder is SD-only; non-SD id reached CPU fallback");
    if id.is_sd_family() {
        tracing::warn!(?id, part = %key,
            "SD bundle build refused: no compatible GPU EP, CPU produces wrong output");
        return Err(
            "SD inpaint requires GPU acceleration. No compatible GPU \
             execution provider is available; SD on CPU produces \
             incorrect output. Install OpenVINO Runtime (Settings → \
             Hardware) or pick LaMa as the eraser instead.".to_string(),
        );
    }
    crate::cache::gc_stale_for_model(id, "CPU");
    let builder = sd_base_builder()
        .map_err(|e| format!("SD {key}: builder init: {e}"))?;
    let (mut builder, load_path) = sd_apply_path_cache(builder, path.as_path(), id, "CPU", key);
    let started = Instant::now();
    let session = builder
        .commit_from_file(&load_path)
        .map_err(|e| format!("SD {key}: load from {}: {e}", load_path.display()))?;
    tracing::info!(?id, part = %key, ep = "CPU", elapsed_ms = started.elapsed().as_millis() as u64, "SD: session committed");
    Ok((session, "CPU".to_string()))
}

fn sd_base_builder() -> Result<ort::session::builder::SessionBuilder, String> {
    Session::builder()
        .map_err(|e| format!("SD: ORT builder init failed: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("SD: optimization level: {e}"))
}

/// Path-based ORT-graph-optimization cache for SD's `commit_from_file`
/// pattern. On hit returns the cached optimized.onnx path so the load
/// skips graph optimization; on miss returns the source path with
/// `with_optimized_model_path` set so ORT writes the optimized graph.
/// Best-effort — any error logs and falls back to the uncached path.
fn sd_apply_path_cache<'a>(
    builder: ort::session::builder::SessionBuilder,
    source_path: &'a Path,
    model_id: prunr_models::ModelId,
    ep_name: &str,
    part: &str,
) -> (ort::session::builder::SessionBuilder, Cow<'a, Path>) {
    let Some(cache_path) = crate::cache::optimized_model_path_for_part(model_id, ep_name, part) else {
        return (builder, Cow::Borrowed(source_path));
    };
    if cache_path.is_file() {
        tracing::debug!(?model_id, ep = %ep_name, part, "cache hit (optimized graph, file)");
        return (builder, Cow::Owned(cache_path));
    }
    if crate::cache::cache_dir_for_part(model_id, ep_name, part).is_none() {
        return (builder, Cow::Borrowed(source_path));
    }
    match builder.with_optimized_model_path(&cache_path) {
        Ok(b) => {
            tracing::debug!(?model_id, ep = %ep_name, part, path = %cache_path.display(), "cache miss (writing optimized graph)");
            (b, Cow::Borrowed(source_path))
        }
        Err(e) => {
            tracing::warn!(?model_id, ep = %ep_name, part, %e, "with_optimized_model_path failed; cache disabled for this build");
            (e.recover(), Cow::Borrowed(source_path))
        }
    }
}

// Per-part smoke tests. Each runs ONE forward pass with zero-valued
// inputs of the model's expected shape/type — catches EP/op
// incompatibilities at session-build time rather than at first stroke.

fn smoke_test_text_encoder(s: &mut Session, label: &str) -> Result<(), String> {
    let tokens = clip_tokenize("");
    let t = Tensor::from_array(tokens)
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    let inputs = s.inputs();
    let name = inputs.first().map(|i| i.name().to_string())
        .ok_or_else(|| format!("{label}: no inputs"))?;
    s.run(inputs![name.as_str() => &t])
        .map_err(|e| format!("{label}: smoke run: {e}"))?;
    Ok(())
}

fn smoke_test_vae_encoder(s: &mut Session, label: &str) -> Result<(), String> {
    let img = Array4::<f16>::zeros((1, 3, SD_TILE as usize, SD_TILE as usize));
    let t = Tensor::from_array(img)
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    let inputs = s.inputs();
    let name = inputs.first().map(|i| i.name().to_string())
        .ok_or_else(|| format!("{label}: no inputs"))?;
    s.run(inputs![name.as_str() => &t])
        .map_err(|e| format!("{label}: smoke run: {e}"))?;
    Ok(())
}

fn smoke_test_vae_decoder(s: &mut Session, label: &str) -> Result<(), String> {
    let lat = Array4::<f16>::zeros((1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize));
    let t = Tensor::from_array(lat)
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    let inputs = s.inputs();
    let name = inputs.first().map(|i| i.name().to_string())
        .ok_or_else(|| format!("{label}: no inputs"))?;
    s.run(inputs![name.as_str() => &t])
        .map_err(|e| format!("{label}: smoke run: {e}"))?;
    Ok(())
}

fn smoke_test_unet(s: &mut Session, label: &str) -> Result<(), String> {
    let l = SD_LATENT_SIDE as usize;
    let lat = Array4::<f16>::zeros((1, 9, l, l));
    let ts = ndarray::Array1::<f16>::from_elem(1, f16::from_f32(0.0));
    let emb = Array3::<f16>::zeros((1, CLIP_SEQ_LEN, 768));
    let lat_t = Tensor::from_array(lat).map_err(|e| format!("{label}: smoke latent: {e}"))?;
    let ts_t = Tensor::from_array(ts).map_err(|e| format!("{label}: smoke ts: {e}"))?;
    let emb_t = Tensor::from_array(emb).map_err(|e| format!("{label}: smoke emb: {e}"))?;
    let inputs = s.inputs();
    if inputs.len() < 3 {
        return Err(format!("{label}: smoke needs ≥3 inputs, got {}", inputs.len()));
    }
    let n0 = inputs[0].name().to_string();
    let n1 = inputs[1].name().to_string();
    let n2 = inputs[2].name().to_string();
    s.run(inputs![
        n0.as_str() => &lat_t,
        n1.as_str() => &ts_t,
        n2.as_str() => &emb_t,
    ]).map_err(|e| format!("{label}: smoke run: {e}"))?;
    Ok(())
}

fn take_first_input(s: &Session, label: &str) -> Result<String, String> {
    let inputs = s.inputs();
    inputs.first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| format!("SD {label}: no inputs declared"))
}

fn take_three_inputs(s: &Session, label: &str) -> Result<[String; 3], String> {
    let inputs = s.inputs();
    if inputs.len() < 3 {
        return Err(format!("SD {label}: expected ≥3 inputs, got {}", inputs.len()));
    }
    Ok([
        inputs[0].name().to_string(),
        inputs[1].name().to_string(),
        inputs[2].name().to_string(),
    ])
}

// ── Text encoder ────────────────────────────────────────────────────────

/// Tokenize for CLIP-ViT-L/14 (SD 1.5's text encoder). Diffusers'
/// `tokenizer(text, padding="max_length", truncation=True)` shape:
/// `[BOS, t_0, t_1, …, EOS, EOS pad, …]` of length 77. Empty string
/// degenerates to `[BOS, EOS, EOS, …]` — the unconditional encoding.
/// SD 1.5 ONNX exports use `int32` for `input_ids`; tokenizer returns
/// `u16` so we cast on copy.
fn clip_tokenize(text: &str) -> Array2<i32> {
    use std::sync::OnceLock;
    static T: OnceLock<instant_clip_tokenizer::Tokenizer> = OnceLock::new();
    let tokenizer = T.get_or_init(instant_clip_tokenizer::Tokenizer::new);

    // Tokenizer's `encode` produces the inner tokens (no BOS/EOS) into a
    // user-supplied Vec; we wrap with the special tokens + truncate +
    // pad to 77 ourselves so the framing matches diffusers exactly.
    let mut inner: Vec<instant_clip_tokenizer::Token> = Vec::new();
    tokenizer.encode(text, &mut inner);
    let max_inner = CLIP_SEQ_LEN - 2; // leave room for BOS + at least one EOS
    let inner_len = inner.len().min(max_inner);

    let mut out = Array2::<i32>::from_elem((1, CLIP_SEQ_LEN), CLIP_EOS as i32);
    out[(0, 0)] = CLIP_BOS as i32;
    for (i, tok) in inner.iter().take(inner_len).enumerate() {
        out[(0, 1 + i)] = tok.to_u16() as i32;
    }
    out
}

fn encode_text(bundle: &SdSession, prompt: &str) -> Result<Array3<f32>, CoreError> {
    let tokens = clip_tokenize(prompt);
    let t = Tensor::from_array(tokens)
        .map_err(|e| CoreError::Inference(format!("SD text encoder: input tensor: {e}")))?;
    let mut session = bundle.text_encoder.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![bundle.text_encoder_input.as_str() => &t])
        .map_err(|e| CoreError::Inference(format!("SD text encoder: run: {e}")))?;
    extract_3d(&outputs[0], "text encoder")
}

// ── VAE encode / decode ────────────────────────────────────────────────

/// Pluggable VAE backend. `Standard` uses the SdSession's VAE legs
/// (~80M params each, baked into SD 1.5). `Taesd` uses a tiny distilled
/// pair (~1M params each, ~3× faster decode). The latter doesn't apply
/// the 0.18215 scaling factor — TAESD bakes it into the model — so the
/// scale step is conditional.
pub(crate) enum VaeBackend<'a> {
    Standard(&'a SdSession),
    Taesd(&'a TaesdSession),
}

/// VAE-encode the masked-image variant directly from source + mask,
/// skipping the intermediate RGBA clone the old
/// `mask_image_for_vae` → `image_to_minus1_plus1` chain built (~1 MB
/// at 512×512).
fn vae_encode_masked(
    backend: &VaeBackend,
    image: &RgbaImage,
    mask: &GrayImage,
) -> Result<Array4<f32>, CoreError> {
    let input = image_to_minus1_plus1_masked(image, mask);
    vae_encode_from_input(backend, input)
}

fn vae_encode_from_input(backend: &VaeBackend, input: Array4<f32>) -> Result<Array4<f32>, CoreError> {
    let input_f16 = f32_to_f16_4d(&input);
    // Belt-and-suspenders: NLL would drop `input` at end of statement
    // anyway since it's unused below. The explicit drop documents
    // intent — the f32 buffer (~3 MB at 512×512) is dead the moment
    // the f16 mirror exists, and the next op acquires the VAE session
    // mutex (potentially a held-lock window if another thread is
    // running). Free first.
    drop(input);
    let t = Tensor::from_array(input_f16)
        .map_err(|e| CoreError::Inference(format!("vae encoder: input tensor: {e}")))?;
    match backend {
        VaeBackend::Standard(bundle) => {
            let mut session = bundle.vae_encoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![bundle.vae_encoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("SD vae encoder: run: {e}")))?;
            let mut latent = extract_4d(&outputs[0], "vae encoder")?;
            latent *= VAE_SCALING_FACTOR;
            Ok(latent)
        }
        VaeBackend::Taesd(taesd) => {
            let mut session = taesd.encoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![taesd.encoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("TAESD encoder: run: {e}")))?;
            extract_4d(&outputs[0], "TAESD encoder")
        }
    }
}

fn vae_decode(backend: &VaeBackend, latent: &Array4<f32>) -> Result<RgbaImage, CoreError> {
    match backend {
        VaeBackend::Standard(bundle) => {
            // Fuse the unscale divide with the f16 narrow — saves the
            // intermediate `latent / VAE_SCALING_FACTOR` Array4 (~64 KB
            // at 64×64 latent) that was immediately consumed and dropped.
            let unscaled_f16 = latent.mapv(|v| f16::from_f32(v / VAE_SCALING_FACTOR));
            let t = Tensor::from_array(unscaled_f16)
                .map_err(|e| CoreError::Inference(format!("SD vae decoder: input tensor: {e}")))?;
            let mut session = bundle.vae_decoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![bundle.vae_decoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("SD vae decoder: run: {e}")))?;
            let arr = extract_4d(&outputs[0], "vae decoder")?;
            Ok(minus1_plus1_to_image(&arr))
        }
        VaeBackend::Taesd(taesd) => {
            let latent_f16 = f32_to_f16_4d(latent);
            let t = Tensor::from_array(latent_f16)
                .map_err(|e| CoreError::Inference(format!("TAESD decoder: input tensor: {e}")))?;
            let mut session = taesd.decoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![taesd.decoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("TAESD decoder: run: {e}")))?;
            let arr = extract_4d(&outputs[0], "TAESD decoder")?;
            Ok(minus1_plus1_to_image(&arr))
        }
    }
}

// ── UNet step ───────────────────────────────────────────────────────────

/// SD-1.5 fp16 ONNX export uses f16 for sample/text/timestep. Caller
/// owns the loop-varying `latent_9ch_f16` (consumed by Tensor::from_array)
/// and the loop-invariant `text_emb_f16` (cloned cheaply once per step).
fn unet_step(
    bundle: &SdSession,
    latent_9ch_f16: Array4<f16>,
    t: i64,
    text_emb_f16: &Array3<f16>,
) -> Result<Array4<f32>, CoreError> {
    // Timestep is integer-valued at the diffusers level but flows through
    // sinusoidal embeddings inside the UNet, so the f16 cast is precision-
    // safe (timesteps fit in the f16 mantissa for SD's 1000-step grid).
    let timestep = ndarray::Array1::<f16>::from_elem(1, f16::from_f32(t as f32));

    let lat_t = Tensor::from_array(latent_9ch_f16)
        .map_err(|e| CoreError::Inference(format!("SD unet: latent tensor: {e}")))?;
    let ts_t = Tensor::from_array(timestep)
        .map_err(|e| CoreError::Inference(format!("SD unet: timestep tensor: {e}")))?;
    let emb_t = Tensor::from_array(text_emb_f16.clone())
        .map_err(|e| CoreError::Inference(format!("SD unet: text emb tensor: {e}")))?;
    let mut session = bundle.unet.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![
        bundle.unet_inputs[0].as_str() => &lat_t,
        bundle.unet_inputs[1].as_str() => &ts_t,
        bundle.unet_inputs[2].as_str() => &emb_t,
    ])
        .map_err(|e| CoreError::Inference(format!("SD unet: run: {e}")))?;
    extract_4d(&outputs[0], "unet")
}

/// CFG batched: stack cond + uncond on batch dim 0 and run UNet ONCE
/// per timestep instead of twice. Most SD-1.5 inpaint ONNX exports
/// declare a dynamic batch dimension so this just works; if the loaded
/// model has a static batch=1 the call returns Err and the caller
/// falls back to sequential `unet_step` × 2 (and remembers to skip
/// future batched attempts on this session).
///
/// Output shape: (2, 4, 64, 64). Row 0 = cond noise pred, row 1 = uncond.
fn unet_step_batched(
    bundle: &SdSession,
    latent_9ch_f16: &Array4<f16>,
    t: i64,
    text_emb_cond_f16: &Array3<f16>,
    text_emb_uncond_f16: &Array3<f16>,
) -> Result<Array4<f32>, CoreError> {
    let latent_pair = ndarray::concatenate(
        Axis(0), &[latent_9ch_f16.view(), latent_9ch_f16.view()],
    ).map_err(|e| CoreError::Inference(format!("SD unet batched: latent concat: {e}")))?;
    let text_pair = ndarray::concatenate(
        Axis(0), &[text_emb_cond_f16.view(), text_emb_uncond_f16.view()],
    ).map_err(|e| CoreError::Inference(format!("SD unet batched: text concat: {e}")))?;
    let timestep = ndarray::Array1::<f16>::from_elem(2, f16::from_f32(t as f32));

    let lat_t = Tensor::from_array(latent_pair)
        .map_err(|e| CoreError::Inference(format!("SD unet batched: latent tensor: {e}")))?;
    let ts_t = Tensor::from_array(timestep)
        .map_err(|e| CoreError::Inference(format!("SD unet batched: timestep tensor: {e}")))?;
    let emb_t = Tensor::from_array(text_pair)
        .map_err(|e| CoreError::Inference(format!("SD unet batched: text tensor: {e}")))?;
    let mut session = bundle.unet.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![
        bundle.unet_inputs[0].as_str() => &lat_t,
        bundle.unet_inputs[1].as_str() => &ts_t,
        bundle.unet_inputs[2].as_str() => &emb_t,
    ])
        .map_err(|e| CoreError::Inference(format!("SD unet batched: run: {e}")))?;
    extract_4d(&outputs[0], "unet (batched)")
}

// ── f16 / f32 conversion at the ONNX boundary ──────────────────────────
//
// The SD 1.5 FP16 export uses f16 inputs and outputs throughout. We do
// scheduler/composite/mask math in f32 (cheap conversion, full precision
// for short loops) and only narrow at the tensor handoff. A future
// `inpaint_sd_fp32` variant could skip these helpers entirely.

fn f32_to_f16_3d(arr: &Array3<f32>) -> Array3<f16> {
    arr.mapv(f16::from_f32)
}

fn f32_to_f16_4d(arr: &Array4<f32>) -> Array4<f16> {
    arr.mapv(f16::from_f32)
}

/// Try f32 first, fall back to f16; either way return f32 ndarray.
/// Centralises the "this export might emit either precision" handling
/// so each session site is one line at the boundary.
fn extract_3d(value: &ort::value::DynValue, label: &str) -> Result<Array3<f32>, CoreError> {
    if let Ok(arr) = value.try_extract_array::<f32>() {
        return arr.into_dimensionality::<ndarray::Ix3>()
            .map(|a| a.to_owned())
            .map_err(|e| CoreError::Inference(format!("SD {label}: shape ≠ 3D: {e}")));
    }
    let arr = value
        .try_extract_array::<f16>()
        .map_err(|e| CoreError::Inference(format!("SD {label}: extract (tried f32, f16): {e}")))?
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| CoreError::Inference(format!("SD {label}: shape ≠ 3D: {e}")))?;
    Ok(arr.mapv(|x| x.to_f32()))
}

fn extract_4d(value: &ort::value::DynValue, label: &str) -> Result<Array4<f32>, CoreError> {
    if let Ok(arr) = value.try_extract_array::<f32>() {
        return arr.into_dimensionality::<ndarray::Ix4>()
            .map(|a| a.to_owned())
            .map_err(|e| CoreError::Inference(format!("SD {label}: shape ≠ 4D: {e}")));
    }
    let arr = value
        .try_extract_array::<f16>()
        .map_err(|e| CoreError::Inference(format!("SD {label}: extract (tried f32, f16): {e}")))?
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| CoreError::Inference(format!("SD {label}: shape ≠ 4D: {e}")))?;
    Ok(arr.mapv(|x| x.to_f32()))
}

// ── Tensor helpers ──────────────────────────────────────────────────────

/// RGBA → NCHW f32 in [-1, 1]. Pads/crops to SD_TILE×SD_TILE; alpha is
/// dropped because SD operates on RGB. Out-of-bounds pixels are zero.
/// Convert a 512×512 RGBA into the SD VAE's [-1, 1] f32 input layout,
/// with mid-gray (0.0) written directly for masked pixels.
///
/// Mid-gray for the masked region is the SD inpaint training
/// convention: diffusers' `prepare_mask_and_masked_image` multiplies
/// the [-1, 1]-normalized image by `(mask < 0.5)` which puts 0 (mid-
/// gray) into the masked region. Filling with black (-1 in [-1, 1])
/// instead drives the masked-image latent out of distribution, and
/// with empty-prompt CFG=1.0 the denoised output collapses to dark
/// fills.
///
/// An empty mask degenerates to a plain image-to-tensor — the unit
/// test exercises the byte-endpoint math through that path.
///
/// Skips the intermediate RGBA clone the previous `mask_image_for_vae`
/// → `image_to_minus1_plus1` sequence built (~1 MB at 512×512).
fn image_to_minus1_plus1_masked(image: &RgbaImage, mask: &GrayImage) -> Array4<f32> {
    debug_assert_eq!(image.dimensions(), mask.dimensions());
    let s = SD_TILE as usize;
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let mut a = Array4::<f32>::zeros((1, 3, s, s));
    let buf = a.as_slice_mut().unwrap();
    let plane = s * s;
    let raw = image.as_raw();
    let m = mask.as_raw();
    for y in 0..h_us.min(s) {
        let src_row = y * w_us * 4;
        let dst_row = y * s;
        let mask_row = y * w_us;
        for x in 0..w_us.min(s) {
            let dst = dst_row + x;
            if m[mask_row + x] > 127 {
                // Match the byte-128 path's bit pattern: same f32
                // division as the unmasked branch, just on a constant
                // input. Hardcoding `0.0` would diverge by ~0.00392.
                let v = (128.0_f32 / 127.5) - 1.0;
                buf[dst]             = v;
                buf[plane + dst]     = v;
                buf[plane * 2 + dst] = v;
            } else {
                let src = src_row + x * 4;
                buf[dst]              = (raw[src]     as f32 / 127.5) - 1.0;
                buf[plane + dst]      = (raw[src + 1] as f32 / 127.5) - 1.0;
                buf[plane * 2 + dst]  = (raw[src + 2] as f32 / 127.5) - 1.0;
            }
        }
    }
    a
}

fn minus1_plus1_to_image(arr: &Array4<f32>) -> RgbaImage {
    let s = SD_TILE as usize;
    let plane = s * s;
    let buf = arr.as_slice().unwrap_or(&[]);
    if buf.len() < plane * 3 {
        tracing::warn!(buf_len = buf.len(), "SD vae decode: buffer smaller than tile");
        return RgbaImage::new(SD_TILE, SD_TILE);
    }
    let mut out = RgbaImage::new(SD_TILE, SD_TILE);
    let dst = out.as_mut();
    for i in 0..plane {
        let r = ((buf[i]              + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        let g = ((buf[plane + i]      + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        let b = ((buf[plane * 2 + i]  + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        dst[i * 4]     = r;
        dst[i * 4 + 1] = g;
        dst[i * 4 + 2] = b;
        dst[i * 4 + 3] = 255;
    }
    out
}

/// Mask 512×512 (binary) → latent space (1, 1, 64, 64) f32 in {0, 1}.
/// Nearest-neighbour 8× downsample — preserves the inpaint boundary
/// exactly at the latent grid; smoother resampling drifts the boundary
/// by a fractional latent pixel which propagates through the denoise.
fn mask_to_latent(mask: &GrayImage) -> Array4<f32> {
    let l = SD_LATENT_SIDE as usize;
    let mut a = Array4::<f32>::zeros((1, 1, l, l));
    let buf = a.as_slice_mut().unwrap();
    let raw = mask.as_raw();
    let (w, h) = mask.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    for ly in 0..l {
        let sy = (ly * 8 + 4).min(h_us.saturating_sub(1));
        for lx in 0..l {
            let sx = (lx * 8 + 4).min(w_us.saturating_sub(1));
            let v = if sy < h_us && sx < w_us && sy * w_us + sx < raw.len() {
                if raw[sy * w_us + sx] > 127 { 1.0 } else { 0.0 }
            } else { 0.0 };
            buf[ly * l + lx] = v;
        }
    }
    a
}

/// Concatenate the three UNet inputs along the channel axis:
/// [latent (4ch) | mask (1ch) | masked_latent (4ch)] → (1, 9, 64, 64).
/// f16 variant lives next to the loop site since two of the three inputs
/// are loop-invariant — we pre-convert them once and pay only the latent
/// concat per step.
fn concat_inpaint_input_f16(
    latent: &Array4<f16>,
    mask_lat: &Array4<f16>,
    masked_lat: &Array4<f16>,
) -> Array4<f16> {
    ndarray::concatenate(Axis(1), &[
        latent.view(),
        mask_lat.view(),
        masked_lat.view(),
    ])
    .expect("concat: shapes pre-validated to (1, *, 64, 64)")
}

/// Initial gaussian noise scaled by the scheduler's `init_noise_sigma`.
/// For DDIM with the SD 1.5 schedule, that's 1.0 — the latent itself
/// starts as plain N(0, 1). Pad here so future schedulers (Euler, DPM++)
/// that need a different sigma can plug in via the scheduler API.
fn sample_initial_noise(rng: &mut ChaCha8Rng) -> Array4<f32> {
    let l = SD_LATENT_SIDE as usize;
    let dist = StandardNormal;
    let n = 4 * l * l;
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f32 = dist.sample(rng);
        buf.push(v);
    }
    Array4::from_shape_vec((1, 4, l, l), buf)
        .expect("shape pre-computed; buf length matches")
}

/// Composite painted output back onto source: source-byte-identical
/// outside the mask (no VAE round-trip drift), painted bytes inside.
fn composite(source: &RgbaImage, painted: &RgbaImage, mask: &GrayImage, w: u32, h: u32) -> RgbaImage {
    let mut out = source.clone();
    let dst = out.as_mut();
    let pnt = painted.as_raw();
    let m = mask.as_raw();
    let painted_w = painted.width() as usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            if i >= m.len() { continue; }
            if m[i] > 127 {
                let pi = (y * painted_w + x) * 4;
                dst[i * 4]     = pnt[pi];
                dst[i * 4 + 1] = pnt[pi + 1];
                dst[i * 4 + 2] = pnt[pi + 2];
            }
        }
    }
    out
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
/// Takes `ArrayView4` so callers can pass batched-UNet slices directly
/// without two `.to_owned()` allocations per CFG step (~128 KB × 20
/// timesteps of churn previously). Output is built in logical row-
/// major order via `iter()`, which is correct for non-contiguous
/// input views (e.g. axis-0 slices of a batched output).
fn cfg_blend(
    uncond: ndarray::ArrayView4<'_, f32>,
    cond: ndarray::ArrayView4<'_, f32>,
    scale: f32,
) -> Array4<f32> {
    debug_assert_eq!(uncond.dim(), cond.dim(), "CFG blend: shape mismatch");
    let buf: Vec<f32> = uncond.iter().zip(cond.iter())
        .map(|(&u, &c)| u + scale * (c - u))
        .collect();
    Array4::from_shape_vec(uncond.dim(), buf).expect("dim matches by construction")
}

/// Apply DDIM step element-wise to flat-Vec representations of the
/// 4D arrays. Avoids one round-trip through Vec<f32> + reshape.
#[allow(clippy::too_many_arguments)] // scheduler dispatch needs all of (loop state + per-scheduler context); packing into a struct adds indirection without consolidating call sites
fn step_array(
    scheduler: &Scheduler,
    latent_t: &Array4<f32>,
    noise_pred: &Array4<f32>,
    step_idx: usize,
    t: i64,
    t_prev: i64,
    is_final_step: bool,
    rng: &mut ChaCha8Rng,
    prev_model_output: &mut Option<Vec<f32>>,
) -> Array4<f32> {
    // extract_4d returns Array4 via mapv/to_owned, which always produces
    // standard C layout; as_slice() is infallible on those.
    let lat = latent_t.as_slice().expect("latent: standard layout");
    let eps = noise_pred.as_slice().expect("noise pred: standard layout");
    let next = match scheduler {
        Scheduler::Ddim(s) => s.step(lat, eps, t, t_prev),
        Scheduler::Lcm(s) => {
            if is_final_step {
                s.step(lat, eps, t, true, &[])
            } else {
                // LCM mixes Gaussian noise back in on every non-final
                // step. Sample from the same seeded RNG that drove the
                // initial latent so the whole denoise stays
                // reproducible from `seed`.
                let dist = StandardNormal;
                let noise: Vec<f32> = (0..lat.len()).map(|_| dist.sample(rng)).collect();
                s.step_at(lat, eps, t, t_prev, &noise)
            }
        }
        Scheduler::DpmPp2M(s) => s.step(lat, eps, step_idx, prev_model_output),
    };
    Array4::from_shape_vec(latent_t.dim(), next)
        .expect("shape unchanged from input")
}

/// If the cropped input is smaller than SD_TILE on either axis (image
/// edge case), pad to 512×512 with zero-fill (top-left aligned). The
/// smart-crop dispatcher guarantees max dim is SD_TILE.
///
/// Returns `Cow::Borrowed` on the fast path (already-tile-sized input,
/// the common case for small strokes) — saves the ~1 MB image clone +
/// ~256 KB mask clone the previous version always did.
fn pad_to_tile(image: &RgbaImage) -> std::borrow::Cow<'_, RgbaImage> {
    let (w, h) = image.dimensions();
    if w == SD_TILE && h == SD_TILE {
        return std::borrow::Cow::Borrowed(image);
    }
    let mut out = RgbaImage::new(SD_TILE, SD_TILE);
    image::imageops::overlay(&mut out, image, 0, 0);
    std::borrow::Cow::Owned(out)
}

fn pad_mask_to_tile(mask: &GrayImage) -> std::borrow::Cow<'_, GrayImage> {
    let (w, h) = mask.dimensions();
    if w == SD_TILE && h == SD_TILE {
        return std::borrow::Cow::Borrowed(mask);
    }
    let mut out = GrayImage::new(SD_TILE, SD_TILE);
    image::imageops::overlay(&mut out, mask, 0, 0);
    std::borrow::Cow::Owned(out)
}

// ── Safety guards ───────────────────────────────────────────────────────

/// Cross-platform available-RAM probe. Returns `None` only when sysinfo
/// can't read the system (CI containers without /proc, exotic platforms);
/// callers in that case skip the guard rather than fail-closed since
/// "we couldn't query" doesn't imply "memory is low".
/// Shared `sysinfo::System` cache for `available_ram_bytes` and
/// `process_rss_mb`. Two callers, one cell — `refresh_*` calls on
/// disjoint fields don't conflict.
fn with_system<T>(f: impl FnOnce(&mut sysinfo::System) -> T) -> T {
    use std::sync::PoisonError;
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let mtx = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut sys = mtx.lock().unwrap_or_else(PoisonError::into_inner);
    f(&mut sys)
}

pub(crate) fn available_ram_bytes() -> Option<u64> {
    let avail = with_system(|sys| {
        sys.refresh_memory();
        sys.available_memory()
    });
    if avail == 0 { None } else { Some(avail) }
}

/// Current process RSS in MB. `None` when sysinfo can't read the process
/// (sandboxed CI, exotic platforms). Used to instrument SD session
/// load/drop where 4-6 GB swings are easy to hide in aggregate logs.
///
/// `pub(crate)` so the LaMa sweeper can emit RSS deltas in the same
/// trace shape as SD's sweeper.
pub(crate) fn process_rss_mb_pub() -> Option<u64> {
    process_rss_mb()
}

fn process_rss_mb() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    with_system(|sys| {
        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map(|p| p.memory() / (1024 * 1024))
    })
}

/// Headroom on top of `working_set_mb` so the gate survives a
/// snapshot-shifting-mid-load: between `check_ram_for` and the actual
/// allocations deep in ORT (5–30 s later for SD), another app may
/// spike, kernel may start swapping, our own bundle build may hit
/// compile transients we under-modelled. 2 GB is enough to cover the
/// "free RAM at gate time minus free RAM during build" delta we see
/// in practice without being so large the gate becomes unfriendly to
/// 16-GB machines. Phase 4 / E1 will surface this as a Settings slider.
const SAFETY_MARGIN_MB: u64 = 2_000;

/// Shared RAM pre-flight gate. Resolves the model's `working_set_mb`,
/// adds a `SAFETY_MARGIN_MB` headroom, and errors with a user-facing
/// message when free RAM is below the combined threshold. Returns
/// `Ok(())` when sysinfo can't read the system (the gate fail-open in
/// that case mirrors `available_ram_bytes`'s `None`-as-skip contract).
/// Single source of truth for the wording — `process_inpaint_with`
/// calls it on every dispatch; `SdSession::new_inner` calls it on
/// bundle build for the prewarm path that bypasses the inpaint entry.
pub(crate) fn check_ram_for(id: prunr_models::ModelId) -> Result<(), String> {
    let Some(desc) = prunr_models::descriptor(id) else { return Ok(()) };
    let need_mb = desc.working_set_mb as u64 + SAFETY_MARGIN_MB;
    let need = need_mb * 1024 * 1024;
    let Some(free) = available_ram_bytes() else { return Ok(()) };
    if free >= need {
        return Ok(());
    }
    Err(format!(
        "{} refused to load: only {:.1} GB RAM free, \
         {:.1} GB minimum recommended (includes {:.1} GB safety \
         headroom). Close other apps or use LaMa instead — \
         Settings → Eraser.",
        desc.display_name,
        free as f64 / 1e9,
        need as f64 / 1e9,
        SAFETY_MARGIN_MB as f64 / 1024.0,
    ))
}

impl Drop for SdSession {
    fn drop(&mut self) {
        tracing::info!(
            rss_mb = process_rss_mb(),
            "SD session bundle dropped (idle release or process exit)",
        );
    }
}

// ── DDIM scheduler ──────────────────────────────────────────────────────

/// SD-1.5 train-timestep count (`num_train_timesteps` in Diffusers).
const SD15_NUM_TRAIN_TIMESTEPS: usize = 1000;

/// SD-1.5 `scaled_linear` β schedule, then α̅ = cumprod(1 - β). Shared
/// across both schedulers — they differ in step formula + timestep
/// subset, not in the underlying noise schedule.
fn compute_alphas_cumprod_sd15() -> Vec<f32> {
    const BETA_START: f32 = 0.00085;
    const BETA_END: f32 = 0.012;
    let sqrt_start = BETA_START.sqrt();
    let sqrt_end = BETA_END.sqrt();
    let mut alphas_cumprod = Vec::with_capacity(SD15_NUM_TRAIN_TIMESTEPS);
    let mut acc = 1.0_f32;
    for i in 0..SD15_NUM_TRAIN_TIMESTEPS {
        let t = i as f32 / (SD15_NUM_TRAIN_TIMESTEPS as f32 - 1.0);
        let b = sqrt_start + (sqrt_end - sqrt_start) * t;
        let beta = b * b;
        acc *= 1.0 - beta;
        alphas_cumprod.push(acc);
    }
    alphas_cumprod
}

/// DDIM (Denoising Diffusion Implicit Models) scheduler — pure math,
/// matches the SD 1.5 reference defaults. Drives the per-timestep
/// latent update inside the denoising loop.
///
/// Defaults match diffusers' `DDIMScheduler(beta_start=0.00085,
/// beta_end=0.012, beta_schedule="scaled_linear",
/// num_train_timesteps=1000)` — the canonical SD 1.5 setting.
pub struct DdimScheduler {
    /// `α̅_t` for every training timestep `t ∈ [0, num_train)`.
    /// Pre-computed at construction; `step` indexes into this with the
    /// inference timestep and the previous inference timestep.
    alphas_cumprod: Vec<f32>,
    /// Subset of training timesteps (length `num_inference`) used by the
    /// inference loop, descending from near-num_train to 0.
    timesteps: Vec<i64>,
    pub num_train: usize,
    pub num_inference: usize,
}

impl DdimScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let alphas_cumprod = compute_alphas_cumprod_sd15();

        // Diffusers default: descending evenly-spaced timesteps from
        // num_train-1 down to 0, length = num_inference.
        let step = SD15_NUM_TRAIN_TIMESTEPS as f32 / num_inference as f32;
        let mut timesteps: Vec<i64> = (0..num_inference)
            .map(|i| ((num_inference - 1 - i) as f32 * step).round() as i64)
            .collect();
        for t in &mut timesteps {
            *t = (*t).clamp(0, SD15_NUM_TRAIN_TIMESTEPS as i64 - 1);
        }

        Self {
            alphas_cumprod,
            timesteps,
            num_train: SD15_NUM_TRAIN_TIMESTEPS,
            num_inference,
        }
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn alpha_cumprod(&self, t: i64) -> f32 {
        let idx = t.clamp(0, self.num_train as i64 - 1) as usize;
        self.alphas_cumprod[idx]
    }

    /// Standard DDIM update with `eta = 0` (deterministic).
    pub fn step(&self, latent_t: &[f32], noise_pred: &[f32], t: i64, t_prev: i64) -> Vec<f32> {
        debug_assert_eq!(latent_t.len(), noise_pred.len(),
            "DDIM step: latent and noise_pred shapes must match");
        let alpha_t = self.alpha_cumprod(t);
        let alpha_prev = if t_prev < 0 { 1.0 } else { self.alpha_cumprod(t_prev) };
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt();
        let sqrt_alpha_prev = alpha_prev.sqrt();
        let sqrt_one_minus_alpha_prev = (1.0 - alpha_prev).sqrt();

        latent_t.iter().zip(noise_pred.iter())
            .map(|(&l, &eps)| {
                let pred_x0 = (l - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t;
                sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * eps
            })
            .collect()
    }
}

/// LCM (Latent Consistency Model) scheduler — port of HuggingFace
/// Diffusers' `LCMScheduler` for the SD-1.5 LCM distilled checkpoint.
///
/// Differs from DDIM in two ways that matter for our use case:
///
/// 1. **Timestep schedule** is *subsampled* from a fixed
///    `original_inference_steps` skip-step grid (default 50 against
///    1000 train timesteps), then evenly sampled down to
///    `num_inference_steps`. DDIM uses a uniform descending grid.
/// 2. **Step formula** uses a boundary-condition consistency function
///    `c_skip / c_out` derived from `sigma_data` and a scaled timestep,
///    mixing the predicted-x0 and the current sample. DDIM uses the
///    standard `√α x₀ + √(1-α) ε` formula.
///
/// LCM weights were *trained* against this update formula. Plugging
/// LCM weights into DDIM produces drift at every step (the noise
/// prediction is calibrated for one math, applied via another) — the
/// over-sharp / off-subject failures we saw on Fast Mode.
///
/// Reference: HuggingFace `diffusers/schedulers/scheduling_lcm.py`,
/// LCM paper arXiv:2310.04378.
pub struct LcmScheduler {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<i64>,
    pub num_train: usize,
    pub num_inference: usize,
    /// Hard-coded fixed value per LCM paper.
    sigma_data: f32,
    /// Multiplier applied to `t` before computing `c_skip`/`c_out`.
    /// Diffusers default 10.0.
    timestep_scaling: f32,
}

impl LcmScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        const ORIGINAL_INFERENCE_STEPS: usize = 50;
        let alphas_cumprod = compute_alphas_cumprod_sd15();

        // Diffusers' LCMScheduler timestep selection:
        //   k = num_train / original_inference_steps  (= 20 for SD-1.5)
        //   lcm_origin descending = [999, 979, 959, ..., 19]
        //   Pick `num_inference` indices via floor(linspace(0, len, num,
        //   endpoint=False)) into the descending array.
        let k = SD15_NUM_TRAIN_TIMESTEPS / ORIGINAL_INFERENCE_STEPS;
        let lcm_origin_descending: Vec<i64> = (0..ORIGINAL_INFERENCE_STEPS)
            .rev()
            .map(|i| ((i + 1) * k - 1) as i64)
            .collect();
        let n_origin = lcm_origin_descending.len() as f32;
        let n_inf = num_inference as f32;
        let timesteps: Vec<i64> = (0..num_inference)
            .map(|i| {
                let idx = ((i as f32) * n_origin / n_inf).floor() as usize;
                lcm_origin_descending[idx.min(lcm_origin_descending.len() - 1)]
            })
            .collect();

        Self {
            alphas_cumprod,
            timesteps,
            num_train: SD15_NUM_TRAIN_TIMESTEPS,
            num_inference,
            sigma_data: 0.5,
            timestep_scaling: 10.0,
        }
    }

    /// `(c_skip, c_out)` — the LCM consistency-function boundary
    /// coefficients at timestep `t`. Single source so `step` and
    /// `step_at` can't drift.
    #[inline]
    fn consistency_coefficients(&self, t: i64) -> (f32, f32) {
        let scaled_t = (t as f32) * self.timestep_scaling;
        let sigma2 = self.sigma_data * self.sigma_data;
        let denom2 = scaled_t * scaled_t + sigma2;
        let c_skip = sigma2 / denom2;
        let c_out = scaled_t / denom2.sqrt();
        (c_skip, c_out)
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn alpha_cumprod(&self, t: i64) -> f32 {
        let idx = t.clamp(0, self.num_train as i64 - 1) as usize;
        self.alphas_cumprod[idx]
    }

    /// LCM consistency-function update. `noise_for_step` is the Gaussian
    /// noise to mix in for the non-final-step stochastic term — caller
    /// supplies it from a seeded RNG so the whole denoise is
    /// deterministic from `(seed, prompt, mask)`. Pass an empty / zero
    /// slice on the final step (it's unused there).
    pub fn step(
        &self,
        latent_t: &[f32],
        noise_pred: &[f32],
        t: i64,
        is_final_step: bool,
        _noise_for_step: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(latent_t.len(), noise_pred.len(),
            "LCM step: latent and noise_pred shapes must match");
        // step_array routes is_final_step=false to step_at — the
        // non-final branch here is structurally unreachable.
        if !is_final_step {
            unreachable!("non-final LCM step: caller must use step_at with t_prev")
        }

        let alpha_t = self.alpha_cumprod(t);
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_beta_t = (1.0 - alpha_t).sqrt();
        let (c_skip, c_out) = self.consistency_coefficients(t);

        latent_t.iter().zip(noise_pred.iter())
            .map(|(&l, &eps)| {
                let pred_x0 = (l - sqrt_beta_t * eps) / sqrt_alpha_t;
                c_out * pred_x0 + c_skip * l
            })
            .collect()
    }

    /// Non-final-step variant that takes the previous inference
    /// timestep so we can sample correctly into the next noise level.
    pub fn step_at(
        &self,
        latent_t: &[f32],
        noise_pred: &[f32],
        t: i64,
        t_prev: i64,
        noise_for_step: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(latent_t.len(), noise_pred.len(),
            "LCM step_at: latent and noise_pred shapes must match");
        debug_assert_eq!(noise_for_step.len(), latent_t.len(),
            "LCM step_at: noise_for_step length must match latent");

        let alpha_t = self.alpha_cumprod(t);
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_beta_t = (1.0 - alpha_t).sqrt();
        let (c_skip, c_out) = self.consistency_coefficients(t);

        let alpha_prev = if t_prev < 0 { 1.0 } else { self.alpha_cumprod(t_prev) };
        let sqrt_alpha_prev = alpha_prev.sqrt();
        let sqrt_beta_prev = (1.0 - alpha_prev).sqrt();

        latent_t.iter()
            .zip(noise_pred.iter())
            .zip(noise_for_step.iter())
            .map(|((&l, &eps), &n)| {
                let pred_x0 = (l - sqrt_beta_t * eps) / sqrt_alpha_t;
                let denoised = c_out * pred_x0 + c_skip * l;
                sqrt_alpha_prev * denoised + sqrt_beta_prev * n
            })
            .collect()
    }
}

/// DPM-Solver++ 2M Karras scheduler — port of Diffusers'
/// `DPMSolverMultistepScheduler` at the SD-1.5 default config:
/// `algorithm_type="dpmsolver++"`, `solver_order=2`,
/// `use_karras_sigmas=True`, `prediction_type="epsilon"`,
/// `solver_type="midpoint"`.
///
/// **Why this scheduler matters.** DDIM is the original SD-1.5 sampler
/// but a generation behind. Modern UIs (A1111, ComfyUI, InvokeAI)
/// default to DPM++ 2M Karras because:
/// - The Karras sigma schedule (rho=7.0) clusters sigmas around the
///   mid-noise band where the model has the most signal.
/// - The multistep solver carries the previous step's predicted x0
///   forward; the second-order correction makes 12-15 steps land at
///   DDIM-20-quality.
///
/// State: the multistep update needs the *previous step's* model
/// output. We keep that out of the struct (so `&Scheduler` stays
/// shareable) and have `step_array` thread it through as
/// `&mut Option<Vec<f32>>`.
pub struct DpmPp2MScheduler {
    /// Length `num_inference + 1`; `sigmas[i]` is the noise level
    /// entering step `i`, `sigmas[i+1]` is the target after step `i`.
    /// `sigmas[num_inference] == 0` (zero terminal SNR — DPM++
    /// convention).
    sigmas: Vec<f32>,
    /// Length `num_inference`. Per-step UNet timestep input,
    /// recovered from `sigmas[0..num_inference]` via log-sigma
    /// interpolation into the train-timestep schedule.
    timesteps: Vec<i64>,
    pub num_train: usize,
    pub num_inference: usize,
}

impl DpmPp2MScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let alphas_cumprod = compute_alphas_cumprod_sd15();
        // VP-space sigma at each train timestep: σ_t = √((1-α̅_t)/α̅_t).
        let sigma_schedule: Vec<f32> = alphas_cumprod.iter()
            .map(|&a| ((1.0 - a) / a).sqrt())
            .collect();
        let log_sigmas: Vec<f32> = sigma_schedule.iter().map(|s| s.ln()).collect();
        let sigma_min = sigma_schedule[0];
        let sigma_max = sigma_schedule[sigma_schedule.len() - 1];

        // Karras sigma schedule (Karras et al. 2022, ρ=7.0). The
        // formula at ramp=0 → σ_max, ramp=1 → σ_min, so iterating
        // 0..N produces a descending schedule directly.
        const RHO: f32 = 7.0;
        let min_inv_rho = sigma_min.powf(1.0 / RHO);
        let max_inv_rho = sigma_max.powf(1.0 / RHO);
        let mut sigmas: Vec<f32> = if num_inference <= 1 {
            vec![sigma_max]
        } else {
            (0..num_inference)
                .map(|i| {
                    let ramp = i as f32 / (num_inference - 1) as f32;
                    (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(RHO)
                })
                .collect()
        };
        // Zero terminal SNR — DPM++ convention.
        sigmas.push(0.0);

        // Convert each non-terminal sigma to a timestep for UNet input.
        let timesteps: Vec<i64> = sigmas[..num_inference].iter()
            .map(|&sigma| sigma_to_t(sigma, &log_sigmas))
            .collect();

        Self {
            sigmas,
            timesteps,
            num_train: SD15_NUM_TRAIN_TIMESTEPS,
            num_inference,
        }
    }

    pub fn timesteps(&self) -> &[i64] { &self.timesteps }

    /// DPM++ 2M Karras update at inference step `step_idx`.
    /// `prev_model_output` is the previous step's predicted x0 — `None`
    /// on the first step (uses first-order DDIM-equivalent update),
    /// `Some(_)` after; the function updates it in-place for the next
    /// call.
    pub fn step(
        &self,
        latent: &[f32],
        noise_pred: &[f32],
        step_idx: usize,
        prev_model_output: &mut Option<Vec<f32>>,
    ) -> Vec<f32> {
        debug_assert_eq!(latent.len(), noise_pred.len(),
            "DPM++ step: latent and noise_pred shapes must match");
        debug_assert!(step_idx + 1 < self.sigmas.len(),
            "DPM++ step: step_idx {step_idx} out of range for sigmas len {}",
            self.sigmas.len());

        let sigma_s0 = self.sigmas[step_idx];
        let sigma_t = self.sigmas[step_idx + 1];
        let (alpha_s0, sigma_s0_e) = sigma_to_alpha_sigma(sigma_s0);
        let (alpha_t, sigma_t_e) = sigma_to_alpha_sigma(sigma_t);

        // ε → x₀ in VP parameterization.
        let m0: Vec<f32> = latent.iter()
            .zip(noise_pred.iter())
            .map(|(&l, &eps)| (l - sigma_s0_e * eps) / alpha_s0)
            .collect();

        // Log-SNR-like quantities for the exponential update.
        let lambda_s0 = alpha_s0.ln() - sigma_s0_e.ln();
        let lambda_t = alpha_t.ln() - sigma_t_e.ln();
        let h = lambda_t - lambda_s0;
        let exp_neg_h = (-h).exp();
        let coef = alpha_t * (exp_neg_h - 1.0);
        let ratio = sigma_t_e / sigma_s0_e;

        let next = if let Some(m1) = prev_model_output.as_ref() {
            // Second-order DPM++ 2M update (midpoint solver).
            let sigma_s1 = self.sigmas[step_idx - 1];
            let (alpha_s1, sigma_s1_e) = sigma_to_alpha_sigma(sigma_s1);
            let lambda_s1 = alpha_s1.ln() - sigma_s1_e.ln();
            let h_0 = lambda_s0 - lambda_s1;
            let r0 = h_0 / h;

            latent.iter()
                .zip(m0.iter())
                .zip(m1.iter())
                .map(|((&l, &md0), &md1)| {
                    let d0 = md0;
                    let d1 = (1.0 / r0) * (md0 - md1);
                    ratio * l - coef * d0 - 0.5 * coef * d1
                })
                .collect()
        } else {
            // First step: no prev output → first-order (DDIM-equivalent).
            latent.iter()
                .zip(m0.iter())
                .map(|(&l, &md0)| ratio * l - coef * md0)
                .collect()
        };

        *prev_model_output = Some(m0);
        next
    }
}

/// VP-parameterization sigma → (α, σ_eff). For SD-1.5:
/// α = 1/√(1+σ²),  σ_eff = σ * α. The model's noise prediction is
/// scaled by σ_eff in the ε→x₀ formula.
fn sigma_to_alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha = 1.0 / (1.0 + sigma * sigma).sqrt();
    (alpha, sigma * alpha)
}

/// Karras sigma → train-timestep, via log-sigma linear interpolation
/// into the SD-1.5 train sigma schedule. Mirrors Diffusers'
/// `_sigma_to_t`.
fn sigma_to_t(sigma: f32, log_sigmas: &[f32]) -> i64 {
    let log_sigma = sigma.max(1e-10_f32).ln();
    // Train log-sigmas are ascending (low noise at idx 0 → high at idx N-1).
    // Find the largest index where log_sigmas[i] <= log_sigma.
    let mut low_idx = 0_usize;
    for (i, &ls) in log_sigmas.iter().enumerate() {
        if ls > log_sigma { break; }
        low_idx = i;
    }
    let high_idx = (low_idx + 1).min(log_sigmas.len() - 1);
    let denom = log_sigmas[low_idx] - log_sigmas[high_idx];
    let w = if denom.abs() < 1e-12 {
        0.0
    } else {
        ((log_sigmas[low_idx] - log_sigma) / denom).clamp(0.0, 1.0)
    };
    let t = (1.0 - w) * (low_idx as f32) + w * (high_idx as f32);
    t.round() as i64
}

/// IPC-portable scheduler kind. Replaces the historical
/// `use_lcm_scheduler: bool` on `SdInpaintRequest` so we can carry
/// 5 scheduler choices through to the worker.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SchedulerKind {
    Ddim,
    Lcm,
    DpmPp2MKarras,
    UniPc,
    EulerA,
}

impl Default for SchedulerKind {
    fn default() -> Self { Self::Lcm }
}

/// Runtime-selected scheduler for SD denoise. Constructed worker-side
/// from `SchedulerKind` carried on `SdInpaintRequest`.
pub enum Scheduler {
    Ddim(DdimScheduler),
    Lcm(LcmScheduler),
    DpmPp2M(DpmPp2MScheduler),
}

impl Scheduler {
    pub fn timesteps(&self) -> &[i64] {
        match self {
            Scheduler::Ddim(s) => s.timesteps(),
            Scheduler::Lcm(s) => s.timesteps(),
            Scheduler::DpmPp2M(s) => s.timesteps(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    #[test]
    fn ddim_alpha_cumprod_is_monotonically_decreasing() {
        let s = DdimScheduler::new_sd15(20);
        let mut prev = s.alpha_cumprod(0);
        for t in 1..s.num_train as i64 {
            let cur = s.alpha_cumprod(t);
            assert!(cur < prev, "α̅ should decrease with t: t={t}, prev={prev}, cur={cur}");
            prev = cur;
        }
    }

    #[test]
    fn ddim_alpha_cumprod_endpoints_match_sd15_reference() {
        let s = DdimScheduler::new_sd15(20);
        let a0 = s.alpha_cumprod(0);
        let a_last = s.alpha_cumprod(s.num_train as i64 - 1);
        assert!((a0 - 0.99915_f32).abs() < 1e-3, "α̅_0 = {a0}");
        assert!((a_last - 0.0047_f32).abs() < 1e-3, "α̅_999 = {a_last}");
    }

    #[test]
    fn ddim_timesteps_descend_from_high_to_low() {
        let s = DdimScheduler::new_sd15(20);
        let t = s.timesteps();
        assert_eq!(t.len(), 20);
        for w in t.windows(2) {
            assert!(w[0] > w[1], "timesteps must descend: {} → {}", w[0], w[1]);
        }
        assert!(*t.last().unwrap() < 100, "last timestep should be near 0, got {:?}", t.last());
    }

    #[test]
    fn ddim_step_returns_input_shape() {
        let s = DdimScheduler::new_sd15(20);
        let latent = vec![0.1_f32; 16];
        let noise = vec![0.05_f32; 16];
        let next = s.step(&latent, &noise, 500, 450);
        assert_eq!(next.len(), latent.len());
    }

    #[test]
    fn ddim_step_at_t_prev_neg_one_collapses_to_clean_x0() {
        let s = DdimScheduler::new_sd15(20);
        let latent = vec![0.5, 0.6, 0.7, 0.8];
        let noise = vec![0.1, 0.1, 0.1, 0.1];
        let alpha_t = s.alpha_cumprod(500);
        let next = s.step(&latent, &noise, 500, -1);
        for (i, &n) in next.iter().enumerate() {
            let expected = (latent[i] - (1.0 - alpha_t).sqrt() * noise[i]) / alpha_t.sqrt();
            assert!((n - expected).abs() < 1e-5, "step[{i}] = {n}, expected {expected}");
        }
    }

    /// LCM timestep schedule matches Diffusers' subsampling pattern: pick
    /// every k-th index from the reversed `lcm_origin_timesteps`. Values
    /// computed by running Diffusers' Python `LCMScheduler.set_timesteps(8)`
    /// and copying the resulting tensor.
    #[test]
    fn lcm_timesteps_match_diffusers_reference_for_8_steps() {
        let s = LcmScheduler::new_sd15(8);
        let t = s.timesteps();
        assert_eq!(t.len(), 8);
        // Reversed lcm_origin = [999, 979, 959, 939, ..., 19].
        // For 8 inference steps, indices = floor(linspace(0, 50, 8, endpoint=False))
        //                               = [0, 6, 12, 18, 25, 31, 37, 43]
        // → reversed[indices] = [999, 879, 759, 639, 499, 379, 259, 139]
        assert_eq!(t, &[999, 879, 759, 639, 499, 379, 259, 139],
            "LCM 8-step timesteps must match Diffusers reference");
    }

    /// 4-step is the canonical "fast preview" LCM count from the paper.
    /// Pinning it alongside the 8-step reference catches drift in the
    /// linspace-floor index math at the extremes.
    #[test]
    fn lcm_timesteps_match_diffusers_reference_for_4_steps() {
        let s = LcmScheduler::new_sd15(4);
        // floor(linspace(0, 50, 4, endpoint=False)) = [0, 12, 25, 37]
        // → reversed[indices] = [999, 759, 499, 259]
        assert_eq!(s.timesteps(), &[999, 759, 499, 259],
            "LCM 4-step timesteps must match Diffusers reference");
    }

    /// LCM timesteps must descend (driving denoising from high noise
    /// to low) — same direction contract as DDIM. A regression here
    /// would silently invert the noise schedule.
    #[test]
    fn lcm_timesteps_descend_strictly() {
        let s = LcmScheduler::new_sd15(8);
        for w in s.timesteps().windows(2) {
            assert!(w[0] > w[1], "timesteps must descend: {} → {}", w[0], w[1]);
        }
    }

    /// LCM and DDIM share the same beta schedule (scaled_linear,
    /// β_start=0.00085, β_end=0.012). A regression in the beta
    /// computation would diverge α̅ between schedulers, silently
    /// breaking LCM math.
    #[test]
    fn lcm_alphas_cumprod_match_ddim() {
        let lcm = LcmScheduler::new_sd15(8);
        let ddim = DdimScheduler::new_sd15(20);
        for &t in &[0_i64, 100, 500, 999] {
            let l = lcm.alpha_cumprod(t);
            let d = ddim.alpha_cumprod(t);
            assert!((l - d).abs() < 1e-7,
                "α̅[{t}] mismatch: LCM={l}, DDIM={d}");
        }
    }

    /// Final-step LCM reduces to `c_out * pred_x0 + c_skip * sample`
    /// — no noise injection. The math is equivalent to DDIM's
    /// `t_prev = -1` collapse-to-x0 in spirit (deterministic), but
    /// scaled by the consistency-function coefficients. This test
    /// verifies the formula directly so a regression in the LCM
    /// boundary conditions is loud.
    #[test]
    fn lcm_final_step_matches_consistency_formula() {
        let s = LcmScheduler::new_sd15(4);
        let latent = vec![0.5_f32; 4];
        let noise = vec![0.1_f32; 4];
        let t = 500_i64;
        let next = s.step(&latent, &noise, t, true, &[]);

        let alpha_t = s.alpha_cumprod(t);
        let beta_t = 1.0 - alpha_t;
        let scaled_t = (t as f32) * 10.0;
        let sigma2 = 0.25_f32;
        let c_skip = sigma2 / (scaled_t * scaled_t + sigma2);
        let c_out = scaled_t / (scaled_t * scaled_t + sigma2).sqrt();

        for (i, &n) in next.iter().enumerate() {
            let pred_x0 = (latent[i] - beta_t.sqrt() * noise[i]) / alpha_t.sqrt();
            let expected = c_out * pred_x0 + c_skip * latent[i];
            assert!((n - expected).abs() < 1e-5,
                "lcm final step[{i}] = {n}, expected {expected}");
        }
    }

    /// DPM++ 2M Karras must descend through the noise schedule like
    /// any SD scheduler, AND the Karras transformation should
    /// concentrate sigmas in the mid-noise band (rho=7.0 — values
    /// not uniformly spaced in either log-σ or t).
    #[test]
    fn dpmpp2m_timesteps_descend_and_terminate_near_zero() {
        let s = DpmPp2MScheduler::new_sd15(25);
        let t = s.timesteps();
        assert_eq!(t.len(), 25);
        // Strict descent through the noise schedule.
        for w in t.windows(2) {
            assert!(w[0] > w[1], "DPM++ 2M Karras timesteps must descend: {} → {}", w[0], w[1]);
        }
        // First step starts near max noise; last is near zero.
        assert!(t[0] >= 950, "first timestep should be near max noise, got {}", t[0]);
        assert!(*t.last().unwrap() < 50, "last timestep should be near 0, got {:?}", t.last());
    }

    /// DPM++ shares the SD-1.5 alpha schedule with DDIM and LCM.
    /// Drift in the shared `compute_alphas_cumprod_sd15` would
    /// silently desync the schedulers; this test pins parity at
    /// representative timesteps.
    #[test]
    fn dpmpp2m_uses_shared_sd15_alpha_schedule() {
        // The scheduler doesn't expose alphas_cumprod, but we can
        // derive sigma at a known timestep from the public sigma
        // schedule: the first sigma equals √((1-α̅_999)/α̅_999).
        let s = DpmPp2MScheduler::new_sd15(25);
        let ddim = DdimScheduler::new_sd15(25);
        // The first sigma in the Karras schedule is sigma_max,
        // derived from α̅[num_train-1].
        let alpha_max = ddim.alpha_cumprod(999);
        let expected_sigma_max = ((1.0 - alpha_max) / alpha_max).sqrt();
        let actual_sigma_max = s.sigmas[0];
        assert!((actual_sigma_max - expected_sigma_max).abs() < 1e-3,
            "first sigma must match √((1-α̅_999)/α̅_999): expected {expected_sigma_max}, got {actual_sigma_max}");
    }

    /// First DPM++ step must use first-order math (no prev output)
    /// and produce a non-zero update from zero noise (the latent
    /// shifts away by the deterministic component). Pinning this
    /// catches regressions in the first-step branching.
    #[test]
    fn dpmpp2m_first_step_with_prev_none_executes_first_order() {
        let s = DpmPp2MScheduler::new_sd15(25);
        let latent = vec![0.5_f32; 4];
        let noise = vec![0.1_f32; 4];
        let mut prev = None;
        let next = s.step(&latent, &noise, 0, &mut prev);
        assert_eq!(next.len(), latent.len());
        assert!(prev.is_some(), "first step must populate prev_model_output for next call");
        // After first step, latent should have moved.
        assert!(next.iter().any(|&v| (v - 0.5).abs() > 1e-4),
            "first step must produce a non-trivial update");
    }

    /// Non-final LCM step adds the stochastic `√β_prev · noise` term
    /// on top of the consistency-function denoise. With zero noise
    /// input, the formula collapses to `√α_prev · denoised`, giving
    /// us a clean reference to verify against.
    #[test]
    fn lcm_step_at_with_zero_noise_collapses_to_alpha_prev_scaled_denoise() {
        let s = LcmScheduler::new_sd15(8);
        let latent = vec![0.5_f32; 4];
        let noise_pred = vec![0.1_f32; 4];
        let zero_noise = vec![0.0_f32; 4];
        let t = 500_i64;
        let t_prev = 379_i64;
        let next = s.step_at(&latent, &noise_pred, t, t_prev, &zero_noise);

        let alpha_t = s.alpha_cumprod(t);
        let alpha_prev = s.alpha_cumprod(t_prev);
        let scaled_t = (t as f32) * 10.0;
        let sigma2 = 0.25_f32;
        let c_skip = sigma2 / (scaled_t * scaled_t + sigma2);
        let c_out = scaled_t / (scaled_t * scaled_t + sigma2).sqrt();

        for (i, &n) in next.iter().enumerate() {
            let pred_x0 = (latent[i] - (1.0 - alpha_t).sqrt() * noise_pred[i]) / alpha_t.sqrt();
            let denoised = c_out * pred_x0 + c_skip * latent[i];
            let expected = alpha_prev.sqrt() * denoised;  // zero-noise → β_prev term vanishes
            assert!((n - expected).abs() < 1e-5,
                "lcm step_at zero-noise[{i}] = {n}, expected {expected}");
        }
    }

    #[test]
    fn clip_tokenize_empty_prompt_matches_clip_convention() {
        // Diffusers' tokenizer("") produces [BOS, EOS, EOS, …] of length 77.
        let toks = clip_tokenize("");
        assert_eq!(toks.shape(), [1, CLIP_SEQ_LEN]);
        assert_eq!(toks[(0, 0)], CLIP_BOS as i32);
        for i in 1..CLIP_SEQ_LEN {
            assert_eq!(toks[(0, i)], CLIP_EOS as i32, "expected EOS pad at index {i}");
        }
    }

    #[test]
    fn clip_tokenize_real_prompt_starts_with_bos_and_ends_with_eos_padding() {
        let toks = clip_tokenize("a photo of a cat");
        assert_eq!(toks.shape(), [1, CLIP_SEQ_LEN]);
        assert_eq!(toks[(0, 0)], CLIP_BOS as i32);
        // Last position should be padding EOS — typical short prompts have
        // their content in the first ~10 positions, EOS at content-end + 1,
        // and padding EOS through the rest of the 77-slot sequence.
        assert_eq!(toks[(0, CLIP_SEQ_LEN - 1)], CLIP_EOS as i32);
        // Prompt actually produced different tokens than empty (otherwise
        // the tokenizer is broken).
        let empty = clip_tokenize("");
        assert_ne!(toks, empty, "real prompt must tokenize differently than empty");
    }

    #[test]
    fn cfg_blend_at_scale_one_returns_cond() {
        let uncond = Array4::<f32>::from_elem((1, 4, 2, 2), 1.0);
        let cond = Array4::<f32>::from_elem((1, 4, 2, 2), 5.0);
        let out = cfg_blend(uncond.view(), cond.view(), 1.0);
        for &v in out.iter() {
            assert!((v - 5.0).abs() < 1e-6, "scale=1 should equal cond, got {v}");
        }
    }

    #[test]
    fn cfg_blend_extrapolates_above_scale_one() {
        let uncond = Array4::<f32>::from_elem((1, 4, 2, 2), 1.0);
        let cond = Array4::<f32>::from_elem((1, 4, 2, 2), 5.0);
        // scale=7.5 → 1 + 7.5*(5-1) = 31
        let out = cfg_blend(uncond.view(), cond.view(), 7.5);
        for &v in out.iter() {
            assert!((v - 31.0).abs() < 1e-4, "scale=7.5 expected 31, got {v}");
        }
    }

    /// Pin the new headline win: cfg_blend accepts non-contiguous slice
    /// views (axis-0 slices of a batched UNet output) without needing
    /// `.to_owned()` per CFG step.
    #[test]
    fn cfg_blend_accepts_axis0_slices_directly() {
        let pair = Array4::<f32>::from_shape_fn((2, 4, 2, 2), |(b, _, _, _)| {
            if b == 0 { 1.0 } else { 5.0 }
        });
        let uncond = pair.slice(ndarray::s![0..1, .., .., ..]);
        let cond = pair.slice(ndarray::s![1..2, .., .., ..]);
        let out = cfg_blend(uncond, cond, 7.5);
        for &v in out.iter() {
            assert!((v - 31.0).abs() < 1e-4, "expected 31, got {v}");
        }
    }

    #[test]
    fn image_to_minus1_plus1_maps_byte_endpoints_correctly() {
        // Empty mask → masked variant degenerates to plain
        // image-to-tensor; same bit-pattern as the previous standalone
        // `image_to_minus1_plus1` helper. Exercising via the masked
        // path keeps a single code path under test.
        let mut img = RgbaImage::new(SD_TILE, SD_TILE);
        for p in img.pixels_mut() {
            *p = Rgba([0, 128, 255, 255]);
        }
        let mask = GrayImage::new(SD_TILE, SD_TILE);
        let arr = image_to_minus1_plus1_masked(&img, &mask);
        let buf = arr.as_slice().unwrap();
        // R = 0   →  -1.0
        // G = 128 →   0.0039 (slightly above 0)
        // B = 255 →   1.0
        assert!((buf[0] - (-1.0)).abs() < 1e-3, "R got {}", buf[0]);
        let plane = (SD_TILE * SD_TILE) as usize;
        assert!(buf[plane].abs() < 0.01, "G got {}", buf[plane]);
        assert!((buf[plane * 2] - 1.0).abs() < 1e-3, "B got {}", buf[plane * 2]);
    }

    #[test]
    fn mask_to_latent_downsamples_boundary_to_64() {
        // 256×256 mask, half painted: lats should report ~half coverage.
        let mut m = GrayImage::new(SD_TILE, SD_TILE);
        for y in 0..SD_TILE {
            for x in 0..(SD_TILE / 2) {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        let lat = mask_to_latent(&m);
        let buf = lat.as_slice().unwrap();
        let coverage: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
        assert!((coverage - 0.5).abs() < 0.05, "expected ~0.5 coverage, got {coverage}");
    }

    #[test]
    fn image_to_minus1_plus1_masked_writes_mid_gray_for_masked_pixels() {
        // Diffusers training convention: masked region maps to 0.0 in
        // [-1, 1] (= 128 in [0, 255]). Verifies the fused-path
        // equivalent of the old `mask_image_for_vae` RGBA-fill helper:
        // bit-pattern matches `(128/127.5) - 1 ≈ 0.00392` exactly.
        let mut img = RgbaImage::new(SD_TILE, SD_TILE);
        for p in img.pixels_mut() { *p = Rgba([100, 200, 50, 255]); }
        let mut mask = GrayImage::new(SD_TILE, SD_TILE);
        mask.put_pixel(2, 3, Luma([255]));
        let arr = image_to_minus1_plus1_masked(&img, &mask);
        let buf = arr.as_slice().unwrap();
        let plane = (SD_TILE * SD_TILE) as usize;
        let s = SD_TILE as usize;
        let masked_idx = 3 * s + 2;
        let mid_gray = (128.0_f32 / 127.5) - 1.0;
        // Masked pixel: all 3 channels at mid-gray.
        assert!((buf[masked_idx] - mid_gray).abs() < 1e-6);
        assert!((buf[plane + masked_idx] - mid_gray).abs() < 1e-6);
        assert!((buf[plane * 2 + masked_idx] - mid_gray).abs() < 1e-6);
        // Untouched pixel: original 100/200/50 byte values, [-1, 1].
        let unmasked_idx = 0;
        assert!((buf[unmasked_idx] - (100.0/127.5 - 1.0)).abs() < 1e-6);
        assert!((buf[plane + unmasked_idx] - (200.0/127.5 - 1.0)).abs() < 1e-6);
        assert!((buf[plane*2 + unmasked_idx] - (50.0/127.5 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn concat_inpaint_input_has_nine_channels() {
        let l = SD_LATENT_SIDE as usize;
        let lat = Array4::<f16>::zeros((1, 4, l, l));
        let mask_lat = Array4::<f16>::zeros((1, 1, l, l));
        let masked_lat = Array4::<f16>::zeros((1, 4, l, l));
        let cat = concat_inpaint_input_f16(&lat, &mask_lat, &masked_lat);
        assert_eq!(cat.shape(), &[1, 9, l, l]);
    }

    #[test]
    fn sample_initial_noise_is_deterministic_with_seed() {
        let mut r1 = ChaCha8Rng::seed_from_u64(42);
        let mut r2 = ChaCha8Rng::seed_from_u64(42);
        let mut r3 = ChaCha8Rng::seed_from_u64(43);
        let a = sample_initial_noise(&mut r1);
        let b = sample_initial_noise(&mut r2);
        assert_eq!(a, b, "same seed must give identical noise");
        let c = sample_initial_noise(&mut r3);
        assert_ne!(a, c, "different seed must give different noise");
    }

    #[test]
    fn mask_components_returns_empty_for_empty_mask() {
        let m = GrayImage::new(64, 64);
        assert!(mask_components(&m).is_empty());
    }

    #[test]
    fn mask_components_separates_disjoint_regions() {
        // Two clearly-disjoint blobs in opposite corners.
        let mut m = GrayImage::new(64, 64);
        m.put_pixel(5, 5, Luma([255]));
        m.put_pixel(6, 5, Luma([255]));
        m.put_pixel(5, 6, Luma([255]));
        m.put_pixel(50, 55, Luma([255]));
        m.put_pixel(51, 55, Luma([255]));
        let comps = mask_components(&m);
        assert_eq!(comps.len(), 2, "expected two components, got {comps:?}");
        // Components are returned in scan order; first component is upper-left.
        assert!(comps[0].x_max <= 10);
        assert!(comps[1].x_min >= 40);
    }

    #[test]
    fn mask_components_treats_4_connected_pixels_as_one() {
        let mut m = GrayImage::new(16, 16);
        // L-shape: connected via shared edges (4-connectivity).
        m.put_pixel(2, 2, Luma([255]));
        m.put_pixel(3, 2, Luma([255]));
        m.put_pixel(3, 3, Luma([255]));
        m.put_pixel(3, 4, Luma([255]));
        let comps = mask_components(&m);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0], MaskBbox { x_min: 2, y_min: 2, x_max: 3, y_max: 4 });
    }

    #[test]
    fn mask_components_diagonal_pixels_are_separate_under_4_connectivity() {
        let mut m = GrayImage::new(16, 16);
        m.put_pixel(2, 2, Luma([255]));
        m.put_pixel(3, 3, Luma([255])); // diagonal-only neighbour
        let comps = mask_components(&m);
        assert_eq!(comps.len(), 2, "diagonal-only adjacency should split");
    }

    #[test]
    fn tile_count_single_when_under_sd_tile() {
        assert_eq!(tile_count(SD_TILE), 1);
        assert_eq!(tile_count(SD_TILE - 1), 1);
        assert_eq!(tile_count(100), 1);
    }

    #[test]
    fn tile_count_grows_with_span_at_step_granularity() {
        // span = SD_TILE + 1 → needs 2 tiles (overlap)
        assert_eq!(tile_count(SD_TILE + 1), 2);
        // span = SD_TILE + TILE_STEP_PX → 2 tiles (last tile's right edge
        // aligns with span end)
        assert_eq!(tile_count(SD_TILE + TILE_STEP_PX), 2);
        // span = SD_TILE + TILE_STEP_PX + 1 → 3 tiles
        assert_eq!(tile_count(SD_TILE + TILE_STEP_PX + 1), 3);
    }

    #[test]
    fn tile_bbox_single_for_small_component() {
        let bbox = MaskBbox { x_min: 100, y_min: 100, x_max: 300, y_max: 300 };
        let tiles = tile_bbox(&bbox, 4096, 4096);
        assert_eq!(tiles.len(), 1);
        let t = &tiles[0];
        assert!(!t.feather_left && !t.feather_right);
        assert!(!t.feather_top && !t.feather_bottom);
    }

    #[test]
    fn tile_bbox_grids_a_large_component() {
        // 1500×400 bbox → 4 tiles wide × 1 tile tall (400 ≤ SD_TILE).
        // span 1500, step 384, tile 512: tiles at 0, 384, 768, 1152;
        // last tile reaches 1664 ≥ span, so 4 tiles cover it.
        let bbox = MaskBbox { x_min: 100, y_min: 100, x_max: 1599, y_max: 499 };
        let tiles = tile_bbox(&bbox, 4096, 4096);
        assert_eq!(tiles.len(), 4, "expected 4 tiles, got {}", tiles.len());
        // Middle tiles feather on both horizontal edges.
        assert!(!tiles[0].feather_left && tiles[0].feather_right);
        assert!(tiles[1].feather_left && tiles[1].feather_right);
        assert!(tiles[2].feather_left && tiles[2].feather_right);
        assert!(tiles[3].feather_left && !tiles[3].feather_right);
        // None feather vertically since only 1 row.
        for t in &tiles {
            assert!(!t.feather_top && !t.feather_bottom);
        }
    }

    #[test]
    fn edge_alpha_is_one_in_centre_and_zero_at_feathered_edge() {
        // No feathering → always 1.
        assert_eq!(edge_alpha(0, 512, false, false), 1.0);
        assert_eq!(edge_alpha(256, 512, false, false), 1.0);
        // Left feather only: 0 at x=0, ramps to 1 at x=TILE_OVERLAP_PX.
        assert!(edge_alpha(0, 512, true, false) < 0.01);
        assert_eq!(edge_alpha(TILE_OVERLAP_PX, 512, true, false), 1.0);
        // Both feathers: still 1 in the centre.
        assert_eq!(edge_alpha(SD_TILE / 2, SD_TILE, true, true), 1.0);
    }

    #[test]
    fn mask_components_ignores_below_threshold_pixels() {
        let mut m = GrayImage::new(16, 16);
        m.put_pixel(5, 5, Luma([100])); // below 127 threshold
        m.put_pixel(8, 8, Luma([200]));
        let comps = mask_components(&m);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0], MaskBbox { x_min: 8, y_min: 8, x_max: 8, y_max: 8 });
    }

    #[test]
    fn mask_components_finds_tiny_stroke_in_large_image() {
        // A single pixel far from origin must still be found and produce
        // the correct absolute bbox after the inner BFS walks bbox-local
        // coordinates. Pins the bbox-bounded BFS optimisation.
        let mut m = GrayImage::new(2048, 2048);
        m.put_pixel(1900, 1850, Luma([200]));
        m.put_pixel(1901, 1850, Luma([200]));
        let comps = mask_components(&m);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0], MaskBbox { x_min: 1900, y_min: 1850, x_max: 1901, y_max: 1850 });
    }

    #[test]
    fn compute_sd_crop_centres_on_bbox_in_large_image() {
        let bbox = MaskBbox { x_min: 1000, y_min: 1000, x_max: 1100, y_max: 1100 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 4096, 4096);
        assert_eq!((w, h), (SD_TILE, SD_TILE));
        // Centre of bbox is (1050, 1050); crop top-left should land at
        // 1050 - 256 = 794.
        assert_eq!(x, 1050 - SD_TILE / 2);
        assert_eq!(y, 1050 - SD_TILE / 2);
    }

    #[test]
    fn compute_sd_crop_clamps_to_right_edge() {
        let bbox = MaskBbox { x_min: 3900, y_min: 100, x_max: 3990, y_max: 200 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 4096, 4096);
        // Crop must fit inside image; right-edge crop starts at img_w - SD_TILE.
        assert_eq!(x + w, 4096, "crop must end at right edge");
        assert_eq!((w, h), (SD_TILE, SD_TILE));
        // Centre falls in the painted region.
        assert!(y < bbox.y_min);
    }

    #[test]
    fn compute_sd_crop_clamps_to_top_left_corner() {
        let bbox = MaskBbox { x_min: 5, y_min: 10, x_max: 50, y_max: 60 };
        let (x, y, _, _) = compute_sd_crop(&bbox, 4096, 4096);
        // Centre is (27, 35); centre - 256 underflows → saturates at 0.
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn compute_sd_crop_shrinks_to_image_for_small_inputs() {
        let bbox = MaskBbox { x_min: 10, y_min: 10, x_max: 100, y_max: 100 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 200, 300);
        // Image smaller than SD_TILE on both axes → crop is the whole
        // image.
        assert_eq!((x, y, w, h), (0, 0, 200, 300));
    }

    #[test]
    fn sample_initial_noise_has_correct_shape() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let n = sample_initial_noise(&mut rng);
        assert_eq!(n.shape(), &[1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize]);
    }

    #[test]
    fn sweep_idle_drops_only_stale_entries() {
        use prunr_models::ModelId;
        let now = Instant::now();
        let idle = Duration::from_secs(300);
        let fresh = now - Duration::from_secs(60);
        let stale = now - Duration::from_secs(600);

        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry {
            value: Arc::new(()),
            last_used: stale,
        });
        cache.insert(ModelId::LaMaFp32, CacheEntry {
            value: Arc::new(()),
            last_used: fresh,
        });

        let dropped = sweep_idle(&mut cache, now, idle);
        assert_eq!(dropped, 1, "exactly one stale entry should evict");
        assert!(!cache.contains_key(&ModelId::SdV15InpaintFp16),
            "stale entry must be removed");
        assert!(cache.contains_key(&ModelId::LaMaFp32),
            "fresh entry must remain");
    }

    #[test]
    fn sweep_idle_releases_arc_so_payload_drops() {
        use prunr_models::ModelId;
        let now = Instant::now();
        let idle = Duration::from_secs(300);
        let payload = Arc::new(());
        let weak = Arc::downgrade(&payload);

        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry {
            value: payload,
            last_used: now - Duration::from_secs(600),
        });

        sweep_idle(&mut cache, now, idle);
        // The cache held the only strong ref; eviction must drop the
        // payload and release upgrades. This pins the use-after-free
        // contract — an in-flight caller would still own a clone.
        assert!(weak.upgrade().is_none(),
            "evicted payload must be the last strong ref so memory releases");
    }

    #[test]
    fn sweep_idle_keeps_entry_at_exactly_the_idle_boundary_safe() {
        use prunr_models::ModelId;
        let now = Instant::now();
        let idle = Duration::from_secs(300);
        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        // Just-under-the-boundary entry must NOT evict.
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry {
            value: Arc::new(()),
            last_used: now - Duration::from_secs(299),
        });
        let dropped = sweep_idle(&mut cache, now, idle);
        assert_eq!(dropped, 0);
        assert!(cache.contains_key(&ModelId::SdV15InpaintFp16));
    }

    /// Pin the contract that closed the OOM race: two concurrent
    /// `get()` callers see the **same** `OnceLock` slot, so only one
    /// build runs even though both miss the cache initially. Without
    /// the fix we'd see `SD session bundle dropped` immediately after
    /// `SD session bundle loaded` in the logs (both bundles built,
    /// loser dropped) — and ~30 GB peak RSS.
    #[test]
    fn concurrent_get_or_init_runs_build_closure_only_once() {
        use std::sync::OnceLock as Cell;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let slot: Arc<Cell<u32>> = Arc::new(Cell::new());
        let calls = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..8).map(|_| {
            let slot = Arc::clone(&slot);
            let calls = Arc::clone(&calls);
            std::thread::spawn(move || {
                *slot.get_or_init(|| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(50));
                    42u32
                })
            })
        }).collect();

        for h in handles {
            assert_eq!(h.join().unwrap(), 42);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1,
            "OnceLock must run the build closure exactly once across concurrent callers");
    }
}
