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
    /// Inpaint strength in [0, 1]. 1.0 = pure noise init, fully
    /// creative rewrite (default, original behavior). 0.7-0.85 =
    /// VAE-encode the source pixels and add proportional noise →
    /// preserves structure / lighting, makes targeted edits. 0.0 =
    /// preserve the original (no work). The dispatcher skips
    /// `(1-strength) * num_inference_steps` early denoise steps and
    /// initializes from `add_noise(image_latents, noise, t_start)`.
    #[serde(default = "default_strength")]
    pub strength: f32,
    /// LCM-only: Karras sigma schedule vs linear (the default). Off
    /// by default — LCM was distilled against the linear schedule, so
    /// Karras shifts the inference timestep distribution away from
    /// training. Surface as a user toggle for A/B comparison rather
    /// than a silent default.
    #[serde(default)]
    pub use_karras_sigmas: bool,
}

fn default_strength() -> f32 { 1.0 }

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
            strength: default_strength(),
            use_karras_sigmas: false,
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
    let mut scheduler = match req.scheduler {
        SchedulerKind::Lcm => Scheduler::Lcm(LcmScheduler::new_sd15(steps, req.use_karras_sigmas)),
        SchedulerKind::Ddim => Scheduler::Ddim(DdimScheduler::new_sd15(steps)),
        SchedulerKind::DpmPp2MKarras => Scheduler::DpmPp2M(DpmPp2MScheduler::new_sd15(steps)),
        SchedulerKind::EulerA => Scheduler::EulerA(EulerAScheduler::new_sd15(steps)),
        SchedulerKind::UniPc => Scheduler::UniPc(UniPcScheduler::new_sd15(steps)),
    };
    let seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Strength < 1.0 → skip early denoise steps and initialize from a
    // VAE-encoded source image mixed with proportional noise. At
    // strength = 1.0 the path is the original "pure noise init,
    // run all steps".
    /// Treat strength near 1.0 as full-noise to avoid skipping a step
    /// on a 4-step run due to f32 rounding.
    const STRENGTH_FULL_THRESHOLD: f32 = 0.999;
    /// init_noise_sigma identity guard — schedulers returning ~1.0 skip
    /// the in-place latent rescale.
    const INIT_NOISE_SIGMA_IDENTITY_EPS: f32 = 1e-6;

    let strength = req.strength.clamp(0.0, 1.0);
    let total_steps = scheduler.timesteps().len();
    let t_start: usize = if strength >= STRENGTH_FULL_THRESHOLD {
        0
    } else {
        // Floor so a 0.99 doesn't skip a step on a 4-step LCM run.
        let skipped = ((1.0 - strength) * total_steps as f32).floor() as usize;
        skipped.min(total_steps - 1)
    };
    // Override the entry-level total so the user sees the actual
    // step count being run, not the pre-strength budget.
    if let Some(p) = progress {
        p.set_total((total_steps - t_start) as u32);
    }

    // Hold denoising state as Vec<f32> + captured dim. f16 conversion
    // uses ArrayView4::from_shape (zero-copy on the f32 side); the
    // scheduler writes via step_array_into into a reused scratch buffer.
    let latent_dim = (1_usize, 4_usize,
        SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize);
    let mut latent_buf: Vec<f32> = if t_start == 0 {
        let mut l = sample_initial_noise(&mut rng);
        // σ-space schedulers (Euler-A) lift the initial sample onto
        // their σ_max scale; α-space schedulers leave unit variance.
        let init_scale = scheduler.init_noise_sigma();
        if (init_scale - 1.0).abs() > INIT_NOISE_SIGMA_IDENTITY_EPS {
            l.mapv_inplace(|v| v * init_scale);
        }
        l.into_raw_vec_and_offset().0
    } else {
        // Strength < 1: VAE-encode the full source pixels (no mask
        // gate), then have the scheduler mix in noise at sigmas[t_start].
        let image_latents = vae_encode_from_input(vae, image_to_minus1_plus1(&padded_image))?;
        let n = image_latents.len();
        let dist = StandardNormal;
        let noise: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        scheduler.add_noise(
            image_latents.as_slice().expect("image latent: standard layout"),
            &noise,
            t_start,
        )
    };

    // Denoising loop. With CFG: noise_pred = uncond + scale * (cond - uncond).
    // Without CFG: just one UNet pass with cond.
    let timesteps = scheduler.timesteps().to_vec();
    let scale = req.guidance_scale;
    let is_cancelled = || cancel.is_some_and(|c| {
        c.load(std::sync::atomic::Ordering::Acquire)
    });
    let needs_precondition = scheduler.requires_preconditioning();
    let mut precond_buf: Vec<f32> = Vec::new();
    // Scratch buffer for step output — hoisted outside the loop so
    // mem::swap reuses the allocation across steps (no per-step alloc).
    let mut step_out_buf: Vec<f32> = Vec::with_capacity(latent_buf.len());
    // Skip the first `t_start` entries when strength<1: those are
    // the high-noise steps we're bypassing.
    for (i, &t) in timesteps.iter().enumerate().skip(t_start) {
        // Check cancel between UNet steps. ORT has no per-op cancel, so
        // worst-case latency on cancel is one UNet step (multi-second).
        if is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        // Progress is reported relative to the steps actually run
        // (total_steps - t_start) so the user sees "step 1 of 5" on
        // a 50%-strength 10-step run, not "step 6 of 10".
        if let Some(p) = progress {
            p.set_step(((i - t_start) as u32) + 1);
        }
        let latent_f16 = if needs_precondition {
            scheduler.scale_model_input_into(&latent_buf, i, &mut precond_buf);
            // Zero-copy view of precond_buf — only the Array4<f16> output stays.
            let view = ndarray::ArrayView4::from_shape(latent_dim, &precond_buf)
                .expect("latent_dim matches latent_buf length by construction");
            f32_to_f16_view(view)
        } else {
            // Zero-copy view of latent_buf — only the Array4<f16> output stays.
            let view = ndarray::ArrayView4::from_shape(latent_dim, &latent_buf)
                .expect("latent_dim matches latent_buf length by construction");
            f32_to_f16_view(view)
        };
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
        let np_slice = noise_pred.as_slice().expect("noise_pred: standard layout");
        step_array_into(
            &mut scheduler, &latent_buf, np_slice, i, t, t_prev, is_final,
            &mut rng, &mut step_out_buf,
        );
        // step_out_buf now holds the new denoised state; swap so
        // latent_buf becomes current and step_out_buf is the stale
        // scratch (overwritten next iteration).
        std::mem::swap(&mut latent_buf, &mut step_out_buf);
    }

    // VAE decode — wrap final buf into Array4 (no extra allocation).
    let final_array = ndarray::Array4::from_shape_vec(latent_dim, latent_buf)
        .expect("latent_dim matches latent_buf length by construction");
    let painted = vae_decode(vae, &final_array)?;
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

fn f32_to_f16_view(view: ndarray::ArrayView4<'_, f32>) -> Array4<f16> {
    view.mapv(f16::from_f32)
}

fn f32_to_f16_4d(arr: &Array4<f32>) -> Array4<f16> {
    f32_to_f16_view(arr.view())
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

/// RGBA → NCHW f32 in [-1, 1] for the SD VAE encoder. Alpha dropped;
/// masked pixels (mask > 127) write mid-gray (0.0) per the SD inpaint
/// training convention — diffusers' `prepare_mask_and_masked_image`
/// zeroes the masked region so filling with -1 (black) drives the
/// VAE latent out of distribution. `None` mask encodes the full image.
fn image_to_minus1_plus1_inner(image: &RgbaImage, mask: Option<&GrayImage>) -> Array4<f32> {
    let s = SD_TILE as usize;
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let mut a = Array4::<f32>::zeros((1, 3, s, s));
    let buf = a.as_slice_mut().unwrap();
    let plane = s * s;
    let raw = image.as_raw();
    let m = mask.map(|m| m.as_raw());
    for y in 0..h_us.min(s) {
        let src_row = y * w_us * 4;
        let dst_row = y * s;
        let mask_row = y * w_us;
        for x in 0..w_us.min(s) {
            let dst = dst_row + x;
            if m.is_some_and(|m| m[mask_row + x] > 127) {
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

/// Full-image encode — used by the strength<1 init path so the mixed
/// init latent retains the original masked-region pixels.
fn image_to_minus1_plus1(image: &RgbaImage) -> Array4<f32> {
    image_to_minus1_plus1_inner(image, None)
}

fn image_to_minus1_plus1_masked(image: &RgbaImage, mask: &GrayImage) -> Array4<f32> {
    debug_assert_eq!(image.dimensions(), mask.dimensions());
    image_to_minus1_plus1_inner(image, Some(mask))
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

/// Scheduler step dispatched from `run_one_tile`. Writes the new latent
/// state into `out` (clear + extend on the EulerA arm; assign-from-Vec on
/// the others — net-zero on those arms but consistent shape for the caller).
// `step_idx` is only meaningful to the DPM++ multistep arm, but the
// dispatch is shared. Threading it through a `StepCtx` struct would
// force the DDIM/LCM arms to carry a field they never read.
#[allow(clippy::too_many_arguments)]
fn step_array_into(
    scheduler: &mut Scheduler,
    latent: &[f32],
    noise_pred: &[f32],
    step_idx: usize,
    t: i64,
    t_prev: i64,
    is_final_step: bool,
    rng: &mut ChaCha8Rng,
    out: &mut Vec<f32>,
) {
    match scheduler {
        Scheduler::Ddim(s) => {
            *out = s.step(latent, noise_pred, t, t_prev);
        }
        Scheduler::Lcm(s) => {
            if is_final_step {
                *out = s.step(latent, noise_pred, t, true, &[]);
            } else {
                // LCM mixes Gaussian noise back in on every non-final
                // step. Sample from the same seeded RNG that drove the
                // initial latent so the whole denoise stays
                // reproducible from `seed`.
                let dist = StandardNormal;
                let noise: Vec<f32> = (0..latent.len()).map(|_| dist.sample(rng)).collect();
                *out = s.step_at(latent, noise_pred, t, t_prev, &noise);
            }
        }
        Scheduler::DpmPp2M(s) => {
            *out = s.step(latent, noise_pred, step_idx);
        }
        Scheduler::EulerA(s) => {
            // Ancestral noise sampled per step from the same seeded
            // RNG so the whole denoise stays reproducible from
            // `seed`. Final step has σ_to=0 → sigma_up=0, so the
            // noise contribution vanishes regardless of value.
            let dist = StandardNormal;
            let noise: Vec<f32> = (0..latent.len()).map(|_| dist.sample(rng)).collect();
            s.step_into(latent, noise_pred, step_idx, &noise, out);
        }
        Scheduler::UniPc(s) => {
            *out = s.step(noise_pred, latent);
        }
    }
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

/// Per-train-timestep `ln σ_t` where σ_t = √((1-α̅_t)/α̅_t). Shared
/// schedule input for any Karras-family scheduler that needs to map
/// a sampling sigma back to a train timestep.
fn compute_log_sigmas_sd15() -> &'static [f32] {
    static CACHE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        compute_alphas_cumprod_sd15()
            .iter()
            .map(|&a| (((1.0 - a) / a).sqrt()).ln())
            .collect()
    })
}

/// Karras sigma schedule (Karras et al. 2022, ρ=7.0) descending from
/// `sigma_max` to `sigma_min` over `num_inference` points. The terminal
/// sigma is the caller's responsibility — DPM++ appends `sigma_min`,
/// Euler-A appends 0.
fn compute_karras_sigmas(num_inference: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
    const RHO: f32 = 7.0;
    if num_inference <= 1 {
        return vec![sigma_max];
    }
    let min_inv_rho = sigma_min.powf(1.0 / RHO);
    let max_inv_rho = sigma_max.powf(1.0 / RHO);
    (0..num_inference)
        .map(|i| {
            let ramp = i as f32 / (num_inference - 1) as f32;
            (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(RHO)
        })
        .collect()
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

        // Diffusers DDIMScheduler default: timestep_spacing="leading" +
        // steps_offset=1. step_ratio = floor(num_train / num_inference)
        // (integer division, matching Python's // operator). Then:
        //   arange(0, N) * step_ratio + 1, reversed.
        // For 20 steps: [951, 901, 851, ..., 51, 1].
        let step_ratio = SD15_NUM_TRAIN_TIMESTEPS / num_inference;
        let mut timesteps: Vec<i64> = (0..num_inference)
            .rev()
            .map(|i| (i * step_ratio + 1) as i64)
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
        // SD-1.5 ships set_alpha_to_one=false; final-step alpha_prev =
        // alphas_cumprod[0] (≈0.99915), not 1.0.
        let alpha_prev = if t_prev < 0 { self.alpha_cumprod(0) } else { self.alpha_cumprod(t_prev) };
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
    pub fn new_sd15(num_inference: usize, use_karras: bool) -> Self {
        const ORIGINAL_INFERENCE_STEPS: usize = 50;
        let alphas_cumprod = compute_alphas_cumprod_sd15();

        let timesteps: Vec<i64> = if use_karras {
            // Karras sigmas mapped back to train timesteps via sigma_to_t.
            // LCM's sigma range matches the SD-1.5 training noise schedule.
            let log_sigmas = compute_log_sigmas_sd15();
            let sigma_min = (1.0_f32 - alphas_cumprod[0]).sqrt() / alphas_cumprod[0].sqrt();
            let sigma_max = (1.0_f32 - alphas_cumprod[SD15_NUM_TRAIN_TIMESTEPS - 1]).sqrt()
                / alphas_cumprod[SD15_NUM_TRAIN_TIMESTEPS - 1].sqrt();
            let sigmas = compute_karras_sigmas(num_inference, sigma_min, sigma_max);
            sigmas.iter().map(|&s| sigma_to_t(s, log_sigmas)).collect()
        } else {
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
            (0..num_inference)
                .map(|i| {
                    let idx = ((i as f32) * n_origin / n_inf).floor() as usize;
                    lcm_origin_descending[idx.min(lcm_origin_descending.len() - 1)]
                })
                .collect()
        };

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

        // SD-1.5 ships set_alpha_to_one=false; final-step alpha_prev =
        // alphas_cumprod[0] (≈0.99915), not 1.0.
        let alpha_prev = if t_prev < 0 { self.alpha_cumprod(0) } else { self.alpha_cumprod(t_prev) };
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
pub struct DpmPp2MScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    pub num_train: usize,
    pub num_inference: usize,
    /// Previous step's predicted x0 — `None` on first step (first-order
    /// branch), `Some` thereafter (second-order multistep correction).
    /// Buffer is reused across steps via `clone_from` to avoid a
    /// per-step allocation.
    prev_model_output: Option<Vec<f32>>,
}

impl DpmPp2MScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let log_sigmas = compute_log_sigmas_sd15();
        let sigma_min = log_sigmas[0].exp();
        let sigma_max = log_sigmas[log_sigmas.len() - 1].exp();
        let mut sigmas = compute_karras_sigmas(num_inference, sigma_min, sigma_max);
        // Terminal sigma = sigma_min. Diffusers' default is 0 with a
        // `lower_order_final` guard that suppresses the second-order D1
        // term at the terminal step (which would otherwise propagate
        // lambda_t = +∞ → d1 = NaN). We force first-order at terminal
        // AND use sigma_min as belt-and-braces — either guard alone is
        // sufficient, but both keeps the math finite under any future
        // refactor of the step dispatch.
        sigmas.push(sigma_min);

        // Convert each non-terminal sigma to a timestep for UNet input.
        let timesteps: Vec<i64> = sigmas[..num_inference].iter()
            .map(|&sigma| sigma_to_t(sigma, &log_sigmas))
            .collect();

        Self {
            sigmas,
            timesteps,
            num_train: SD15_NUM_TRAIN_TIMESTEPS,
            num_inference,
            prev_model_output: None,
        }
    }

    pub fn timesteps(&self) -> &[i64] { &self.timesteps }

    pub fn step(
        &mut self,
        latent: &[f32],
        noise_pred: &[f32],
        step_idx: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(latent.len(), noise_pred.len(),
            "DPM++ step: latent and noise_pred shapes must match");
        debug_assert!(step_idx + 1 < self.sigmas.len(),
            "DPM++ step: step_idx {step_idx} out of range for sigmas len {}",
            self.sigmas.len());

        let sigma_s0 = self.sigmas[step_idx];
        let sigma_t = self.sigmas[step_idx + 1];
        let (_, sigma_s0_e) = sigma_to_alpha_sigma(sigma_s0);
        let (alpha_t, sigma_t_e) = sigma_to_alpha_sigma(sigma_t);

        let m0 = convert_epsilon_to_x0(latent, noise_pred, sigma_s0);

        let lambda_s0 = sigma_to_lambda(sigma_s0);
        let lambda_t = sigma_to_lambda(sigma_t);
        let h = lambda_t - lambda_s0;
        let coef = alpha_t * ((-h).exp() - 1.0);
        let ratio = sigma_t_e / sigma_s0_e;

        // Second-order only when NOT at the terminal step.
        // Diffusers' `lower_order_final=True` (default) forces first-order on
        // the last step because the per-step sigma drop is largest there and
        // the D1 correction can overshoot. We mirror that guard here.
        let next: Vec<f32> = if step_idx + 1 < self.num_inference {
            if let Some(m1) = self.prev_model_output.as_ref() {
                let sigma_s1 = self.sigmas[step_idx - 1];
                let lambda_s1 = sigma_to_lambda(sigma_s1);
                let r0 = (lambda_s0 - lambda_s1) / h;
                let inv_r0 = 1.0 / r0;

                latent.iter()
                    .zip(m0.iter())
                    .zip(m1.iter())
                    .map(|((&l, &md0), &md1)| {
                        let d1 = inv_r0 * (md0 - md1);
                        ratio * l - coef * md0 - 0.5 * coef * d1
                    })
                    .collect()
            } else {
                latent.iter()
                    .zip(m0.iter())
                    .map(|(&l, &md0)| ratio * l - coef * md0)
                    .collect()
            }
        } else {
            // Terminal step: first-order regardless of prev_model_output.
            latent.iter()
                .zip(m0.iter())
                .map(|(&l, &md0)| ratio * l - coef * md0)
                .collect()
        };

        // Reuse the prev-buffer's allocation across steps.
        match &mut self.prev_model_output {
            Some(buf) => buf.clone_from(&m0),
            slot @ None => *slot = Some(m0),
        }
        next
    }
}

/// Euler-Ancestral scheduler — port of Diffusers'
/// `EulerAncestralDiscreteScheduler` at SD-1.5 epsilon prediction.
///
/// σ-space parameterization: the latent state is held in σ-space
/// (initial sample = noise * √(σ_max²+1) for "leading" timestep
/// spacing). Before each UNet call, `scale_model_input` divides by
/// √(σ²+1) to map back to the α-space input the UNet was trained
/// on. DDIM/LCM/DPM++ hold the latent in α-space and don't need
/// preconditioning.
pub struct EulerAScheduler {
    /// Length `num_inference + 1`. Karras schedule descending from
    /// sigma_max to 0 (terminal sigma=0 makes `sigma_up` vanish on
    /// the final step → no leftover noise contamination).
    sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    pub num_train: usize,
    pub num_inference: usize,
}

impl EulerAScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let log_sigmas = compute_log_sigmas_sd15();
        let sigma_min = log_sigmas[0].exp();
        let sigma_max = log_sigmas[log_sigmas.len() - 1].exp();
        let mut sigmas = compute_karras_sigmas(num_inference, sigma_min, sigma_max);
        // Euler-A terminates at 0 — sigma_up² = σ_to²·(σ²-σ_to²)/σ²
        // vanishes when σ_to=0, giving a deterministic final step
        // and zero residual noise. Different terminal convention
        // from DPM++ (which uses sigma_min to avoid lambda=+∞ in
        // the multistep formula).
        sigmas.push(0.0);

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

    /// Multiplier applied to the initial standard-normal noise before
    /// the denoise loop. Diffusers' "leading" timestep spacing:
    /// `init_noise_sigma = √(σ_max² + 1)`.
    pub fn init_noise_sigma(&self) -> f32 {
        let sigma_max = self.sigmas[0];
        (sigma_max * sigma_max + 1.0).sqrt()
    }

    /// Pre-UNet preconditioning: σ-space sample → α-space UNet input.
    /// `α(σ) = 1/√(σ²+1)`, so `α-input = sample · α(σ) = sample / √(σ²+1)`.
    pub fn scale_model_input_into(&self, latent: &[f32], step_idx: usize, out: &mut Vec<f32>) {
        let sigma = self.sigmas[step_idx];
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        out.clear();
        out.extend(latent.iter().map(|&v| v * scale));
    }

    /// Euler-Ancestral update at inference step `step_idx`, writing the
    /// new latent into a caller-owned `out` (clear + extend — no per-step
    /// allocation when `out` already has the right capacity).
    pub fn step_into(
        &self,
        latent: &[f32],
        noise_pred: &[f32],
        step_idx: usize,
        noise: &[f32],
        out: &mut Vec<f32>,
    ) {
        debug_assert_eq!(latent.len(), noise_pred.len(),
            "Euler-A step: latent and noise_pred shapes must match");
        debug_assert_eq!(latent.len(), noise.len(),
            "Euler-A step: latent and noise shapes must match");
        debug_assert!(step_idx + 1 < self.sigmas.len());

        let sigma = self.sigmas[step_idx];
        let sigma_to = self.sigmas[step_idx + 1];

        // Ancestral noise split:
        //   sigma_up² = σ_to² · (σ² - σ_to²) / σ²
        //   sigma_down = √(σ_to² - sigma_up²)
        // At terminal (σ_to=0), sigma_up = sigma_down = 0 → next = sample + ε * (-σ).
        let sigma_up = if sigma > 0.0 {
            let raw = sigma_to * sigma_to * (sigma * sigma - sigma_to * sigma_to)
                / (sigma * sigma);
            raw.max(0.0).sqrt()
        } else {
            0.0
        };
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).max(0.0).sqrt();

        // For epsilon prediction: derivative = (sample - pred_x0) / σ = ε.
        // Step: prev = sample + ε · (σ_down - σ) + noise · σ_up.
        let dt = sigma_down - sigma;
        out.clear();
        out.extend(
            latent.iter().zip(noise_pred.iter()).zip(noise.iter())
                .map(|((&l, &eps), &n)| l + eps * dt + n * sigma_up),
        );
    }

    /// Thin wrapper around `step_into` for backwards compatibility with
    /// existing tests and any future callers that don't own a scratch buffer.
    pub fn step(
        &self,
        latent: &[f32],
        noise_pred: &[f32],
        step_idx: usize,
        noise: &[f32],
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(latent.len());
        self.step_into(latent, noise_pred, step_idx, noise, &mut out);
        out
    }
}

/// UniPC (Unified Predictor-Corrector) multistep scheduler.
///
/// Port of Diffusers' `UniPCMultistepScheduler` at SD-1.5 defaults:
/// `solver_order=2`, `predict_x0=True`, `solver_type="bh2"`,
/// `use_karras_sigmas=True`, `lower_order_final=True`,
/// `final_sigmas_type="zero"`.
///
/// UniPC's corrector solves a 2×2 linear system (Cramer's rule) to
/// compensate for the predictor's hardcoded `rhos_p=0.5`
/// approximation, yielding better convergence per step than DPM++ 2M
/// at the same step budget — best quality in the 8-12 step range.
pub struct UniPcScheduler {
    pub(super) sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    /// Ring buffer of converted x0 predictions (solver_order=2).
    /// [0]=m_{k-2}, [1]=m_{k-1} after ring shift; new output goes to [1].
    pub(super) model_outputs: [Option<Vec<f32>>; 2],
    timestep_list: [Option<i64>; 2],
    /// Warmup counter — saturates at solver_order (2).
    pub(super) lower_order_nums: usize,
    /// Corrected sample from the previous step; input for the corrector.
    pub(super) last_sample: Option<Vec<f32>>,
    /// Order used by the predictor at step k; read by corrector at step k+1.
    pub(super) this_order: usize,
    pub(super) step_index: usize,
    pub num_train: usize,
    pub num_inference: usize,
}

/// Floor for Cramer's-rule denominators in the UniPC corrector.
/// Prevents ±∞/NaN when `rk → 1` (det = 1−rk → 0) or `rk → 0` at very
/// low step counts or unusual sigma schedules.
const CRAMER_DET_FLOOR: f32 = 1e-6;

impl UniPcScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let log_sigmas = compute_log_sigmas_sd15();
        let sigma_min = log_sigmas[0].exp();
        let sigma_max = log_sigmas[log_sigmas.len() - 1].exp();
        let mut sigmas = compute_karras_sigmas(num_inference, sigma_min, sigma_max);
        // Terminal 0: lower_order_final reduces the last predictor to
        // order=1, which doesn't have the lambda=+∞ NaN issue, so 0
        // is safe here (unlike DPM++ which needs sigma_min).
        sigmas.push(0.0);

        let timesteps: Vec<i64> = sigmas[..num_inference].iter()
            .map(|&s| sigma_to_t(s, log_sigmas))
            .collect();

        Self {
            sigmas,
            timesteps,
            model_outputs: [None, None],
            timestep_list: [None, None],
            lower_order_nums: 0,
            last_sample: None,
            this_order: 1,
            step_index: 0,
            num_train: SD15_NUM_TRAIN_TIMESTEPS,
            num_inference,
        }
    }

    pub fn timesteps(&self) -> &[i64] { &self.timesteps }

    /// Full UniPC step (predictor-corrector). Returns the denoised
    /// latent estimate for the next timestep.
    ///
    /// # Arguments
    /// - `epsilon`: UNet noise prediction (ε) at the current timestep.
    /// - `sample`: Current latent x_k in α-space.
    pub fn step(&mut self, epsilon: &[f32], sample: &[f32]) -> Vec<f32> {
        debug_assert!(self.step_index + 1 < self.sigmas.len(),
            "UniPC step: step_index {} out of range (sigmas len {})",
            self.step_index, self.sigmas.len());

        // A. Convert ε → x0_pred.
        let m_new = self.convert_model_output(epsilon, sample);

        // B. Corrector — reads self.this_order set by the PREVIOUS step.
        // take() avoids a simultaneous borrow of self through last_sample and
        // through the method receiver; we restore it immediately after.
        let working_sample = if self.step_index > 0 {
            if let Some(ls) = self.last_sample.take() {
                let out = self.uni_c_bh_update(&m_new, &ls, sample, self.this_order);
                self.last_sample = Some(ls); // restore allocation
                out
            } else {
                sample.to_vec()
            }
        } else {
            sample.to_vec()
        };

        // C. Ring shift: [0] ← [1], then [1] ← m_new.
        self.model_outputs[0] = self.model_outputs[1].take();
        self.timestep_list[0] = self.timestep_list[1];
        self.model_outputs[1] = Some(m_new);
        self.timestep_list[1] = Some(self.timesteps[self.step_index]);

        // D. Compute this_order for THIS step's predictor (and NEXT step's corrector).
        // Must come BEFORE E: the predictor uses this_order, and the corrector at
        // step k+1 reads the value set here — wrong ordering silently feeds stale order.
        let mut order = 2_usize;
        if order > self.num_inference - self.step_index {
            order = self.num_inference - self.step_index;
        }
        order = order.min(self.lower_order_nums + 1);
        self.this_order = order;

        // E. Move working_sample into last_sample BEFORE the predictor call so
        // the predictor can borrow from last_sample via &. No clone needed —
        // one Vec<f32> persists across steps, saving ~64 KB per step.
        self.last_sample = Some(working_sample);
        // just stored above, non-empty
        let last_ref = self.last_sample.as_deref().unwrap();

        // F. Predictor.
        let prev_sample = self.uni_p_bh_update(last_ref, order);

        // G. Warmup increment.
        if self.lower_order_nums < 2 {
            self.lower_order_nums += 1;
        }

        // H. Advance.
        self.step_index += 1;

        prev_sample
    }

    /// ε → x0_pred conversion for epsilon-prediction mode.
    fn convert_model_output(&self, epsilon: &[f32], sample: &[f32]) -> Vec<f32> {
        convert_epsilon_to_x0(sample, epsilon, self.sigmas[self.step_index])
    }

    /// Predictor update: `multistep_uni_p_bh_update`.
    ///
    /// `k = step_index`. Source sigma = sigmas[k], target = sigmas[k+1].
    fn uni_p_bh_update(&self, sample: &[f32], order: usize) -> Vec<f32> {
        let k = self.step_index;
        let (_, sigma_s0) = sigma_to_alpha_sigma(self.sigmas[k]);
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma(self.sigmas[k + 1]);

        let lambda_s0 = sigma_to_lambda(self.sigmas[k]);
        let lambda_t = sigma_to_lambda(self.sigmas[k + 1]);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        // model_outputs[1] = m_k (just inserted during ring shift).
        let m0 = self.model_outputs[1].as_deref().expect("ring[1] populated before predictor");

        if order == 1 {
            sample.iter().zip(m0.iter())
                .map(|(&x, &m)| (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m)
                .collect()
        } else {
            // order == 2: model_outputs[0] = m_{k-1}.
            let m1 = self.model_outputs[0].as_deref().expect("ring[0] populated for order-2 predictor");
            let lambda_si = sigma_to_lambda(self.sigmas[k - 1]);
            let rk = (lambda_si - lambda_s0) / h;
            let d1: Vec<f32> = m1.iter().zip(m0.iter())
                .map(|(&a, &b)| (a - b) / rk)
                .collect();

            // rhos_p = 0.5 HARDCODED for predictor (Pitfall #6).
            sample.iter().zip(m0.iter()).zip(d1.iter())
                .map(|((&x, &m), &d)| {
                    (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m - alpha_t * b_h * (0.5 * d)
                })
                .collect()
        }
    }

    /// Corrector update: `multistep_uni_c_bh_update`.
    ///
    /// Asymmetric sigma indices vs predictor:
    /// source = sigmas[k-1], target = sigmas[k].
    fn uni_c_bh_update(
        &self,
        model_t: &[f32],
        last_sample: &[f32],
        _this_sample: &[f32],
        order: usize,
    ) -> Vec<f32> {
        let k = self.step_index;
        // Corrector uses [k-1] → [k], not [k] → [k+1].
        let (_, sigma_s0) = sigma_to_alpha_sigma(self.sigmas[k - 1]);
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma(self.sigmas[k]);

        let lambda_s0 = sigma_to_lambda(self.sigmas[k - 1]);
        let lambda_t = sigma_to_lambda(self.sigmas[k]);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        // model_outputs[1] = m_{k-1} (BEFORE ring shift — pitfall #4).
        let m0 = self.model_outputs[1].as_deref().expect("ring[1] populated before corrector");

        if order == 1 {
            // Corrector order-1: rho_c = 0.5 hardcoded.
            let d1_t: Vec<f32> = model_t.iter().zip(m0.iter())
                .map(|(&mt, &m)| mt - m)
                .collect();
            last_sample.iter().zip(m0.iter()).zip(d1_t.iter())
                .map(|((&ls, &m), &d)| {
                    (sigma_t / sigma_s0) * ls - alpha_t * h_phi_1 * m - alpha_t * b_h * (0.5 * d)
                })
                .collect()
        } else {
            // order == 2: solve full 2×2 via Cramer's rule.
            let lambda_si = sigma_to_lambda(self.sigmas[k - 2]);
            let rk = (lambda_si - lambda_s0) / h;

            // phi recurrence (pitfall #7: factorial_i updates BEFORE h_phi_k).
            let mut factorial_i: f32 = 1.0;
            let mut h_phi_k = h_phi_1 / hh - 1.0;
            let b0 = h_phi_k * factorial_i / b_h;
            factorial_i = 2.0;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
            let b1 = h_phi_k * factorial_i / b_h;

            // Cramer: R = [[1,1],[rk,1]], det = 1 - rk.
            // Guard rk → 1 (det → 0) and rk → 0 (d1 blows up).
            let det = 1.0 - rk;
            debug_assert!(det.abs() > CRAMER_DET_FLOOR,
                "UniPC corrector: rk too close to 1.0 ({rk}); Cramer det = {det}");
            debug_assert!(rk.abs() > CRAMER_DET_FLOOR,
                "UniPC corrector: rk too close to 0 ({rk})");
            let det_guarded = if det.abs() < CRAMER_DET_FLOOR {
                CRAMER_DET_FLOOR.copysign(if det == 0.0 { 1.0 } else { det })
            } else {
                det
            };
            let rk_guarded = rk.abs().max(CRAMER_DET_FLOOR).copysign(rk);
            let rho_c0 = (b0 - b1) / det_guarded;
            let rho_c1 = b0 - rho_c0;

            let m1 = self.model_outputs[0].as_deref().expect("ring[0] populated for order-2 corrector");
            let d1: Vec<f32> = m1.iter().zip(m0.iter())
                .map(|(&a, &b)| (a - b) / rk_guarded)
                .collect();
            let d1_t: Vec<f32> = model_t.iter().zip(m0.iter())
                .map(|(&mt, &m)| mt - m)
                .collect();

            last_sample.iter().zip(m0.iter()).zip(d1.iter()).zip(d1_t.iter())
                .map(|(((&ls, &m), &d), &dt)| {
                    (sigma_t / sigma_s0) * ls
                        - alpha_t * h_phi_1 * m
                        - alpha_t * b_h * (rho_c0 * d + rho_c1 * dt)
                })
                .collect()
        }
    }
}

/// VP-parameterization sigma → (α, σ_eff). For SD-1.5:
/// α = 1/√(1+σ²),  σ_eff = σ * α. The model's noise prediction is
/// scaled by σ_eff in the ε→x₀ formula.
fn sigma_to_alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha = 1.0 / (1.0 + sigma * sigma).sqrt();
    (alpha, sigma * alpha)
}

/// Log-SNR λ(σ) = ln(α) − ln(σ_eff). Used by all multistep
/// schedulers in lambda-space update formulas.
fn sigma_to_lambda(sigma: f32) -> f32 {
    let (alpha, sigma_eff) = sigma_to_alpha_sigma(sigma);
    alpha.ln() - sigma_eff.ln()
}

/// VP-parameterization ε prediction → x₀ estimate.
/// `(sample − σ_eff · ε) / α` where (α, σ_eff) come from
/// `sigma_to_alpha_sigma(σ_edm)`.
fn convert_epsilon_to_x0(sample: &[f32], eps: &[f32], sigma_edm: f32) -> Vec<f32> {
    let (alpha, sigma_eff) = sigma_to_alpha_sigma(sigma_edm);
    sample.iter()
        .zip(eps.iter())
        .map(|(&x, &e)| (x - sigma_eff * e) / alpha)
        .collect()
}

/// Karras sigma → train-timestep, via log-sigma linear interpolation
/// into the SD-1.5 train sigma schedule.
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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SchedulerKind {
    Ddim,
    #[default]
    Lcm,
    DpmPp2MKarras,
    UniPc,
    EulerA,
}

/// Runtime-selected scheduler for SD denoise. Constructed worker-side
/// from `SchedulerKind` carried on `SdInpaintRequest`.
pub enum Scheduler {
    Ddim(DdimScheduler),
    Lcm(LcmScheduler),
    DpmPp2M(DpmPp2MScheduler),
    EulerA(EulerAScheduler),
    UniPc(UniPcScheduler),
}

impl Scheduler {
    pub fn timesteps(&self) -> &[i64] {
        match self {
            Scheduler::Ddim(s) => s.timesteps(),
            Scheduler::Lcm(s) => s.timesteps(),
            Scheduler::DpmPp2M(s) => s.timesteps(),
            Scheduler::EulerA(s) => s.timesteps(),
            Scheduler::UniPc(s) => s.timesteps(),
        }
    }

    /// Multiplier on the initial standard-normal noise. 1.0 for
    /// α-space schedulers (DDIM/LCM/DPM++/UniPC); √(σ_max²+1) for
    /// σ-space schedulers (Euler-A).
    pub fn init_noise_sigma(&self) -> f32 {
        match self {
            Scheduler::Ddim(_)
            | Scheduler::Lcm(_)
            | Scheduler::DpmPp2M(_)
            | Scheduler::UniPc(_) => 1.0,
            Scheduler::EulerA(s) => s.init_noise_sigma(),
        }
    }

    /// Returns true if this scheduler needs `scale_model_input_into`
    /// applied to the latent before each UNet call. Lets the dispatch
    /// loop skip the per-step copy on α-space schedulers.
    pub fn requires_preconditioning(&self) -> bool {
        matches!(self, Scheduler::EulerA(_))
    }

    /// Pre-UNet preconditioning into a caller-owned buffer (so the
    /// allocation can be reused across denoise steps). Caller only
    /// invokes this when `requires_preconditioning()` is true.
    pub fn scale_model_input_into(&self, latent: &[f32], step_idx: usize, out: &mut Vec<f32>) {
        match self {
            Scheduler::EulerA(s) => s.scale_model_input_into(latent, step_idx, out),
            _ => {
                out.clear();
                out.extend_from_slice(latent);
            }
        }
    }

    /// Mix a clean signal `sample` (e.g. VAE-encoded source image)
    /// with `noise` at the noise level corresponding to inference
    /// step `step_idx`. Used for strength<1 inpaint init.
    /// α-space schedulers: `√α̅ · sample + √(1-α̅) · noise`.
    /// σ-space schedulers (Euler-A): `sample + σ · noise`.
    pub fn add_noise(&self, sample: &[f32], noise: &[f32], step_idx: usize) -> Vec<f32> {
        debug_assert_eq!(sample.len(), noise.len());
        match self {
            Scheduler::Ddim(s) => {
                let t = s.timesteps()[step_idx];
                add_noise_alpha_space(sample, noise, s.alpha_cumprod(t))
            }
            Scheduler::Lcm(s) => {
                let t = s.timesteps()[step_idx];
                add_noise_alpha_space(sample, noise, s.alpha_cumprod(t))
            }
            Scheduler::DpmPp2M(s) => {
                let sigma = s.sigmas[step_idx];
                let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
                add_noise_alpha_space(sample, noise, alpha * alpha)
            }
            Scheduler::UniPc(s) => {
                let sigma = s.sigmas[step_idx];
                let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
                add_noise_alpha_space(sample, noise, alpha * alpha)
            }
            Scheduler::EulerA(s) => {
                let sigma = s.sigmas[step_idx];
                sample.iter().zip(noise.iter())
                    .map(|(&x, &n)| x + sigma * n)
                    .collect()
            }
        }
    }
}

/// SD-1.5 forward-noise convention shared by DDIM, LCM, and DPM++ at inference.
fn add_noise_alpha_space(sample: &[f32], noise: &[f32], alpha_cumprod: f32) -> Vec<f32> {
    let sqrt_alpha = alpha_cumprod.sqrt();
    let sqrt_one_minus = (1.0 - alpha_cumprod).max(0.0).sqrt();
    sample.iter().zip(noise.iter())
        .map(|(&s, &n)| sqrt_alpha * s + sqrt_one_minus * n)
        .collect()
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

    /// Diffusers DDIMScheduler default: timestep_spacing="leading" + steps_offset=1.
    /// step_ratio = 1000 // 20 = 50; timesteps = arange(0, 20) * 50 + 1, reversed.
    /// Values verified against Diffusers' Python DDIMScheduler for SD-1.5 at 20 steps.
    #[test]
    fn ddim_timesteps_match_diffusers_leading_with_offset_1() {
        let s = DdimScheduler::new_sd15(20);
        let t = s.timesteps();
        assert_eq!(t[0], 951, "first timestep must be 951 (leading+offset=1)");
        assert_eq!(t[10], 451, "mid timestep (index 10) must be 451");
        assert_eq!(t[19], 1, "last timestep must be 1 (offset=1, not 0)");
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
    fn ddim_step_at_t_prev_neg_one_uses_alphas_cumprod_zero() {
        // set_alpha_to_one=false: final step uses alphas_cumprod[0], not 1.0.
        let s = DdimScheduler::new_sd15(20);
        let latent = vec![0.5, 0.6, 0.7, 0.8];
        let noise = vec![0.1, 0.1, 0.1, 0.1];
        let alpha_t = s.alpha_cumprod(500);
        let alpha_prev = s.alpha_cumprod(0);
        let next = s.step(&latent, &noise, 500, -1);
        for (i, &n) in next.iter().enumerate() {
            let pred_x0 = (latent[i] - (1.0 - alpha_t).sqrt() * noise[i]) / alpha_t.sqrt();
            let expected = alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * noise[i];
            assert!((n - expected).abs() < 1e-5, "step[{i}] = {n}, expected {expected}");
        }
    }

    /// Pins the invariant: DDIM with t_prev=-1 uses alphas_cumprod[0] (≈0.99915),
    /// not 1.0. Verifies both that the result differs from the 1.0 baseline AND
    /// matches the expected per-element computation, so any regression is caught.
    #[test]
    fn ddim_final_step_uses_alphas_cumprod_zero_not_one() {
        let s = DdimScheduler::new_sd15(20);
        let latent = vec![0.3_f32, -0.4, 0.8, 0.1];
        let noise_pred = vec![0.05_f32, 0.05, 0.05, 0.05];
        let t = 500_i64;
        let alpha_t = s.alpha_cumprod(t);
        let alpha_prev = s.alpha_cumprod(0);

        let next = s.step(&latent, &noise_pred, t, -1);

        for (i, &n) in next.iter().enumerate() {
            let pred_x0 = (latent[i] - (1.0 - alpha_t).sqrt() * noise_pred[i]) / alpha_t.sqrt();
            let expected_correct = alpha_prev.sqrt() * pred_x0
                + (1.0 - alpha_prev).sqrt() * noise_pred[i];
            // If the old 1.0 path were taken the result would be just pred_x0.
            let expected_wrong = pred_x0;
            assert!((n - expected_correct).abs() < 1e-5,
                "step[{i}] = {n}, expected alphas_cumprod[0]-based {expected_correct}");
            assert!((n - expected_wrong).abs() > 1e-6,
                "step[{i}] = {n} matches the wrong 1.0-based result — regression?");
        }
    }

    /// LCM timestep schedule matches Diffusers' subsampling pattern: pick
    /// every k-th index from the reversed `lcm_origin_timesteps`. Values
    /// computed by running Diffusers' Python `LCMScheduler.set_timesteps(8)`
    /// and copying the resulting tensor.
    #[test]
    fn lcm_timesteps_match_diffusers_reference_for_8_steps() {
        let s = LcmScheduler::new_sd15(8, false);
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
        let s = LcmScheduler::new_sd15(4, false);
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
        let s = LcmScheduler::new_sd15(8, false);
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
        let lcm = LcmScheduler::new_sd15(8, false);
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
        let s = LcmScheduler::new_sd15(4, false);
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
        // len == 25 just asserted, .last() is non-empty.
        let last = *t.last().unwrap();
        assert!(last < 50, "last timestep should be near 0, got {last}");
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

    /// Terminal sigma must be `sigma_min` not 0, otherwise the
    /// multistep update produces ±∞ / NaN at the last step.
    #[test]
    fn dpmpp2m_terminal_step_produces_finite_output() {
        let mut s = DpmPp2MScheduler::new_sd15(8);
        let latent = vec![0.5_f32; 4];
        let noise = vec![0.1_f32; 4];
        // Prime prev_model_output, then verify the terminal step is finite.
        // The lower_order_final guard ensures first-order runs at terminal;
        // sigma_min (not 0) is belt-and-braces so both guards are in place.
        let _ = s.step(&latent, &noise, 6);
        let next = s.step(&latent, &noise, 7);
        assert!(next.iter().all(|v| v.is_finite()),
            "terminal step produced non-finite values: {next:?}");
    }

    /// Terminal step must take the first-order path even when
    /// `prev_model_output` is populated (lower_order_final guard).
    /// Verified by comparing against a scheduler whose prev is None —
    /// both must produce identical output.
    #[test]
    fn dpmpp2m_terminal_step_uses_first_order_branch() {
        let latent = vec![0.5_f32; 4];
        let noise = vec![0.1_f32; 4];

        // Primed scheduler: populate prev_model_output at step 6, then
        // call terminal step 7. The guard must suppress the D1 correction.
        let mut primed = DpmPp2MScheduler::new_sd15(8);
        let _ = primed.step(&latent, &noise, 6);
        let primed_terminal = primed.step(&latent, &noise, 7);

        // Fresh scheduler: prev_model_output is None, so the first-order
        // branch runs unconditionally at step 7.
        let mut fresh = DpmPp2MScheduler::new_sd15(8);
        let fresh_terminal = fresh.step(&latent, &noise, 7);

        assert_eq!(primed_terminal, fresh_terminal,
            "terminal step must be first-order regardless of prev_model_output");
    }

    /// First DPM++ step must use first-order math (no prev output)
    /// and produce a non-zero update from zero noise (the latent
    /// shifts away by the deterministic component). Pinning this
    /// catches regressions in the first-step branching.
    #[test]
    fn dpmpp2m_first_step_with_prev_none_executes_first_order() {
        let mut s = DpmPp2MScheduler::new_sd15(25);
        let latent = vec![0.5_f32; 4];
        let noise = vec![0.1_f32; 4];
        let next = s.step(&latent, &noise, 0);
        assert_eq!(next.len(), latent.len());
        assert!(s.prev_model_output.is_some(),
            "first step must populate prev_model_output for next call");
        // After first step, latent should have moved.
        assert!(next.iter().any(|&v| (v - 0.5).abs() > 1e-4),
            "first step must produce a non-trivial update");
    }

    /// Euler-A `init_noise_sigma` for "leading" timestep spacing
    /// must be √(σ_max² + 1) — the σ-space sample needs to start on
    /// the scheduler's max sigma scale, not at unit variance.
    #[test]
    fn eulera_init_noise_sigma_matches_leading_convention() {
        let s = EulerAScheduler::new_sd15(25);
        let sigma_max = s.sigmas[0];
        let expected = (sigma_max * sigma_max + 1.0).sqrt();
        assert!((s.init_noise_sigma() - expected).abs() < 1e-5,
            "init_noise_sigma {} != √(σ_max² + 1) = {}", s.init_noise_sigma(), expected);
    }

    /// Pre-UNet preconditioning must divide the σ-space sample by
    /// √(σ²+1) so the α-space UNet sees the right scale.
    #[test]
    fn eulera_scale_model_input_divides_by_sqrt_sigma_squared_plus_one() {
        let s = EulerAScheduler::new_sd15(8);
        let latent = vec![1.0_f32; 4];
        let mut out = Vec::new();
        s.scale_model_input_into(&latent, 0, &mut out);
        let sigma = s.sigmas[0];
        let expected = 1.0 / (sigma * sigma + 1.0).sqrt();
        for v in &out {
            assert!((v - expected).abs() < 1e-6,
                "scaled latent {v} != 1.0 / √(σ²+1) = {expected}");
        }
    }

    /// Terminal step (sigma_to=0) must zero out `sigma_up` so noise
    /// doesn't leak into the final sample. Predicted output collapses
    /// to `latent + eps · (-σ)` regardless of the noise vector.
    #[test]
    fn eulera_terminal_step_zeroes_ancestral_noise() {
        let s = EulerAScheduler::new_sd15(8);
        let latent = vec![0.5_f32; 4];
        let eps = vec![0.1_f32; 4];
        // Non-zero noise that should be erased by sigma_up=0.
        let noise = vec![100.0_f32; 4];
        let terminal = s.num_inference - 1;
        let next = s.step(&latent, &eps, terminal, &noise);
        let sigma = s.sigmas[terminal];
        let expected: Vec<f32> = latent.iter().zip(eps.iter())
            .map(|(&l, &e)| l + e * (0.0 - sigma))
            .collect();
        for (a, b) in next.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5,
                "terminal step leaked ancestral noise: got {a}, expected {b}");
        }
    }

    /// Timesteps must descend strictly through the Karras schedule;
    /// last timestep near zero (sigma_to=0 by construction).
    #[test]
    fn eulera_timesteps_descend_and_terminate_near_zero() {
        let s = EulerAScheduler::new_sd15(25);
        let t = s.timesteps();
        assert_eq!(t.len(), 25);
        for w in t.windows(2) {
            assert!(w[0] > w[1],
                "Euler-A Karras timesteps must descend: {} → {}", w[0], w[1]);
        }
        assert!(t[0] >= 950, "first timestep should be near max noise, got {}", t[0]);
    }

    /// `add_noise` for α-space schedulers must implement the VP
    /// forward process: noisy = √α̅·clean + √(1-α̅)·noise. At step 0
    /// (high noise, α̅ near 0) the result should be dominated by the
    /// noise vector; at the final step (low noise, α̅ near 1) it
    /// should be dominated by the clean signal.
    #[test]
    fn ddim_add_noise_at_high_t_is_noise_dominated_at_low_t_is_sample_dominated() {
        let s = Scheduler::Ddim(DdimScheduler::new_sd15(20));
        let clean = vec![1.0_f32; 4];
        let noise = vec![0.1_f32; 4]; // small noise vs clean=1.0
        // step_idx = 0 → high noise → output dominated by 0.1 (noise scaled by √(1-α̅) ≈ 1)
        let mixed_high = s.add_noise(&clean, &noise, 0);
        // Each entry should be much closer to 0.1 (noise · ~1) than to 1.0 (clean · ~0).
        for v in &mixed_high {
            assert!(*v < 0.5,
                "high-t add_noise should be noise-dominated, got {v}");
        }
        // step_idx = num_inference - 1 → low noise → output close to clean signal
        let mixed_low = s.add_noise(&clean, &noise, 19);
        for v in &mixed_low {
            assert!(*v > 0.85,
                "low-t add_noise should be sample-dominated, got {v}");
        }
    }

    /// Euler-A `add_noise` is σ-space: noisy = sample + σ·noise.
    /// No square-root mixing — the σ-scale is applied to noise directly.
    #[test]
    fn eulera_add_noise_is_sample_plus_sigma_times_noise() {
        let euler = EulerAScheduler::new_sd15(8);
        let s = Scheduler::EulerA(EulerAScheduler::new_sd15(8));
        let clean = vec![0.5_f32; 4];
        let noise = vec![1.0_f32; 4];
        let step_idx = 3;
        let sigma = euler.sigmas[step_idx];
        let mixed = s.add_noise(&clean, &noise, step_idx);
        for v in &mixed {
            let expected = 0.5 + sigma * 1.0;
            assert!((v - expected).abs() < 1e-5,
                "Euler-A add_noise: got {v}, expected {expected}");
        }
    }

    /// Non-final LCM step adds the stochastic `√β_prev · noise` term
    /// on top of the consistency-function denoise. With zero noise
    /// input, the formula collapses to `√α_prev · denoised`, giving
    /// us a clean reference to verify against.
    #[test]
    fn lcm_step_at_with_zero_noise_collapses_to_alpha_prev_scaled_denoise() {
        let s = LcmScheduler::new_sd15(8, false);
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

    /// LCM with `use_karras=true` must produce a non-linear timestep
    /// schedule. With Karras sigmas mapped through `sigma_to_t`, the
    /// timesteps are clustered at the schedule endpoints (not uniform).
    #[test]
    fn lcm_karras_timesteps_differ_from_linear_when_enabled() {
        let linear = LcmScheduler::new_sd15(8, false);
        let karras = LcmScheduler::new_sd15(8, true);
        let lin = linear.timesteps();
        let kar = karras.timesteps();
        assert_eq!(lin.len(), 8);
        assert_eq!(kar.len(), 8);
        // Both descend.
        for w in lin.windows(2) { assert!(w[0] > w[1], "linear not descending: {:?}", lin); }
        for w in kar.windows(2) { assert!(w[0] > w[1], "karras not descending: {:?}", kar); }
        // Schedules MUST differ — at least one timestep should not match.
        assert_ne!(lin, kar,
            "Karras and linear LCM schedules should produce different timesteps");
    }

    // ── UniPC tests ─────────────────────────────────────────────────────────

    /// At step 0, no history is available, so UniPC falls back to
    /// first-order (= DDIM-equivalent formula). Compute the predictor
    /// output manually and compare with the scheduler's output.
    #[test]
    fn unipc_order_1_step_zero_matches_first_order_predictor() {
        let mut s = UniPcScheduler::new_sd15(8);
        let dim = 4_usize;
        let sample: Vec<f32> = (0..dim).map(|i| 0.3 + 0.1 * i as f32).collect();
        let eps: Vec<f32> = (0..dim).map(|i| 0.05 * i as f32).collect();

        let result = s.step(&eps, &sample);

        // Manually compute first-order predictor.
        let sigma_s0_edm = s.sigmas[0]; // before step() advanced step_index
        let sigma_t_edm  = s.sigmas[1];
        let (_, sigma_s0_ddpm) = sigma_to_alpha_sigma(sigma_s0_edm);
        let (alpha_t,  sigma_t_ddpm)  = sigma_to_alpha_sigma(sigma_t_edm);
        let lambda_s0 = sigma_to_lambda(sigma_s0_edm);
        let lambda_t  = sigma_to_lambda(sigma_t_edm);
        let h = lambda_t - lambda_s0;
        let h_phi_1 = (-h).exp_m1();

        let m_ref = convert_epsilon_to_x0(&sample, &eps, sigma_s0_edm);
        let expected: Vec<f32> = sample.iter().zip(m_ref.iter())
            .map(|(&x, &m0)| {
                (sigma_t_ddpm / sigma_s0_ddpm) * x - alpha_t * h_phi_1 * m0
            })
            .collect();

        for (i, (&r, &ex)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - ex).abs() < 1e-5,
                "unipc step-0 output[{i}]: got {r}, expected {ex}");
        }
    }

    /// At the terminal step (N-1), `lower_order_final` must cap the
    /// predictor at order=1 — no D1 correction term. Verified by
    /// constructing an isolated first-order predictor with the same
    /// inputs (bypassing the corrector) and asserting the scheduler
    /// output equals the pure first-order formula to within f32 tol.
    #[test]
    fn unipc_lower_order_final_caps_predictor_at_terminal() {
        let n = 4_usize;
        let dim = 4_usize;
        let eps  = vec![0.1_f32; dim];
        let samp = vec![0.5_f32; dim];

        // Run N-1 steps to get to the state just before the terminal step.
        let mut s = UniPcScheduler::new_sd15(n);
        let mut state = samp.clone();
        for _ in 0..(n - 1) {
            state = s.step(&eps, &state);
        }
        // step_index is now n-1. D in step() computes:
        //   order = min(2, n - (n-1)) = 1.
        // That order is set as this_order during the terminal call.

        // Build a pure first-order predictor reference WITHOUT the corrector.
        // We use `state` directly as the working_sample (i.e., we bypass the
        // corrector path that would modify it) to isolate the predictor-order
        // guard.
        let k = s.step_index; // n-1
        let sigma_s0_edm = s.sigmas[k];
        let sigma_t_edm  = s.sigmas[k + 1];
        let (_, sigma_s0_ddpm) = sigma_to_alpha_sigma(sigma_s0_edm);
        let (alpha_t,  sigma_t_ddpm)  = sigma_to_alpha_sigma(sigma_t_edm);
        let lambda_s0 = sigma_to_lambda(sigma_s0_edm);
        let lambda_t  = sigma_to_lambda(sigma_t_edm);
        let h = lambda_t - lambda_s0;
        let h_phi_1 = (-h).exp_m1();

        // ε → x0_pred conversion on `state`:
        let m_new = convert_epsilon_to_x0(&state, &eps, sigma_s0_edm);
        // Pure first-order predictor (no D1 correction term):
        let first_order_pred: Vec<f32> = state.iter().zip(m_new.iter())
            .map(|(&x, &m)| (sigma_t_ddpm / sigma_s0_ddpm) * x - alpha_t * h_phi_1 * m)
            .collect();

        // Run the actual terminal step through the scheduler.
        let terminal_out = s.step(&eps, &state);

        // Must be finite.
        assert!(terminal_out.iter().all(|v| v.is_finite()),
            "terminal step produced non-finite output: {terminal_out:?}");

        // step_index must be n after all steps.
        assert_eq!(s.step_index, n,
            "step_index must equal num_inference after all steps");

        // this_order must be 1 — the lower_order_final guard fired.
        assert_eq!(s.this_order, 1,
            "this_order after terminal step must be 1 (lower_order_final guard)");

        // The terminal predictor (order=1, no D1 term) must match the
        // reference formula element-wise. The corrector also runs at the
        // terminal step and may adjust `working_sample`, so we allow a
        // small tolerance (1e-3) to accommodate that correction pass.
        for (i, (&sched, &ref_val)) in terminal_out.iter().zip(first_order_pred.iter()).enumerate() {
            assert!((sched - ref_val).abs() < 1e-3,
                "terminal output[{i}]: scheduler={sched}, first-order-ref={ref_val} (diff > 1e-3)");
        }
    }

    /// Verify the Cramer's-rule 2×2 corrector coefficients analytically
    /// for a known (rk, hh) pair and compare with the scheduler output.
    #[test]
    fn unipc_corrector_2x2_cramer_matches_analytical() {
        // Chosen values: hh = -0.5, rk = -0.3 (typical mid-schedule ratios).
        let hh: f32 = -0.5;
        let rk: f32 = -0.3;
        let b_h = hh.exp_m1();
        let h_phi_1 = b_h;

        // Phi recurrence (pitfall #7: factorial_i updates BEFORE h_phi_k).
        let mut factorial_i: f32 = 1.0;
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let b0 = h_phi_k * factorial_i / b_h;
        factorial_i = 2.0;
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        let b1 = h_phi_k * factorial_i / b_h;

        // Cramer: det = 1 - rk.
        let rho_c0 = (b0 - b1) / (1.0 - rk);
        let rho_c1 = b0 - rho_c0;

        // Sanity: rho_c0 + rho_c1 = b0 (from derivation).
        assert!((rho_c0 + rho_c1 - b0).abs() < 1e-6,
            "rho_c0 + rho_c1 must equal b0: {rho_c0} + {rho_c1} != {b0}");

        // Alternatively: rho_c1 = (b1 - rk*b0)/(1-rk).
        let rho_c1_alt = (b1 - rk * b0) / (1.0 - rk);
        assert!((rho_c1 - rho_c1_alt).abs() < 1e-6,
            "two Cramer forms must agree: {rho_c1} vs {rho_c1_alt}");

        // Verify b[0] and b[1] via the phi definitions directly.
        // phi_1(-0.5) = (e^-0.5 - 1) / (-0.5) ≈ 0.787
        // phi_2(-0.5) = (phi_1 - 1) / (-0.5) ≈ 0.426 → b[0] = phi_2/phi_1 ≈ 0.541
        // phi_3(-0.5) = (phi_2 - 0.5) / (-0.5) ≈ 0.148 → b[1] = 2*phi_3/phi_1 ≈ 0.376
        let phi1 = (hh.exp() - 1.0) / hh;
        let phi2 = (phi1 - 1.0) / hh;
        let phi3 = (phi2 - 0.5) / hh;
        let b0_ref = phi2 / phi1;
        let b1_ref = 2.0 * phi3 / phi1;
        assert!((b0 - b0_ref).abs() < 1e-5,
            "b[0] mismatch: recurrence {b0} vs direct {b0_ref}");
        assert!((b1 - b1_ref).abs() < 1e-5,
            "b[1] mismatch: recurrence {b1} vs direct {b1_ref}");
    }

    /// Cramer guard regression: sigmas packed close together collapse the
    /// lambda spacing, driving `det = 1 − rk` toward 0 or `rk` toward 0.
    /// The corrector must return all-finite output instead of ±∞/NaN.
    #[test]
    fn unipc_cramer_guard_prevents_nan_on_collapsed_lambda_spacing() {
        // A 4-step schedule with close sigmas at the start so that by step 2
        // the corrector sees near-degenerate rk values.
        let mut s = UniPcScheduler::new_sd15(4);
        let dim = 8_usize;
        let eps    = vec![0.1_f32; dim];
        let sample = vec![0.5_f32; dim];

        // Force two adjacent sigmas very close so lambda spacing collapses.
        // Replace sigmas[1] and sigmas[2] with nearly identical values to
        // stress both the det=1-rk and rk denominators in the corrector.
        s.sigmas[1] = s.sigmas[0] * 0.9999;
        s.sigmas[2] = s.sigmas[0] * 0.9998;

        let mut out = sample.clone();
        for _ in 0..4 {
            out = s.step(&eps, &out);
        }
        assert!(out.iter().all(|v| v.is_finite()),
            "Cramer guard failed: output contains NaN/Inf when lambda spacing collapses: {out:?}");
    }

    /// After step_index advances to 3, the ring must hold:
    /// model_outputs[1] = m_2 (most recent), model_outputs[0] = m_1.
    #[test]
    fn unipc_ring_state_after_three_steps() {
        let mut s = UniPcScheduler::new_sd15(8);
        let sample: Vec<f32> = vec![0.5_f32; 4];
        let eps:    Vec<f32> = vec![0.1_f32; 4];

        // Track the converted x0 at each step by pre-computing.
        // We call step() 3 times and verify ring contents.
        let mut st = sample.clone();
        // Capture m1 = convert_model_output at step 1 (after first step
        // incremented step_index to 1).
        for i in 0..3_usize {
            // Record what m_new will be before calling step.
            let expected_m: Vec<f32> = {
                let (alpha, sigma_ddpm) = sigma_to_alpha_sigma(s.sigmas[i]);
                st.iter().zip(eps.iter())
                    .map(|(&x, &e)| (x - sigma_ddpm * e) / alpha)
                    .collect()
            };
            st = s.step(&eps, &st);
            // After step 0: ring[1]=m0, ring[0]=None; step_index=1.
            // After step 1: ring[1]=m1, ring[0]=m0; step_index=2.
            // After step 2: ring[1]=m2, ring[0]=m1; step_index=3.
            if i == 0 {
                assert!(s.model_outputs[1].is_some(), "ring[1] must be set after step 0");
                assert!(s.model_outputs[0].is_none(), "ring[0] must still be None after step 0");
            }
            if i == 2 {
                // ring[1] must be m2 — the x0 conversion at step 2.
                let ring1 = s.model_outputs[1].as_ref().unwrap();
                for (j, (&r, &ex)) in ring1.iter().zip(expected_m.iter()).enumerate() {
                    assert!((r - ex).abs() < 1e-5,
                        "ring[1] at step 2 mismatch at [{j}]: {r} vs {ex}");
                }
                assert_eq!(s.step_index, 3, "step_index must be 3 after 3 calls");
            }
        }
    }

    /// Running UniPC to the terminal step must produce all-finite values.
    #[test]
    fn unipc_terminal_step_produces_finite_output() {
        let mut s = UniPcScheduler::new_sd15(8);
        let mut sample = vec![0.5_f32; 4];
        let eps = vec![0.1_f32; 4];
        for _ in 0..8 {
            sample = s.step(&eps, &sample);
        }
        assert!(sample.iter().all(|v| v.is_finite()),
            "UniPC terminal step produced non-finite output: {sample:?}");
    }

    /// Karras schedule must be strictly descending and terminate at 0.
    #[test]
    fn unipc_karras_sigmas_descend_and_terminate_at_zero() {
        let s = UniPcScheduler::new_sd15(20);
        assert_eq!(s.sigmas.len(), 21, "sigmas len must be num_inference + 1");
        for w in s.sigmas.windows(2) {
            assert!(w[0] >= w[1],
                "sigmas must be non-increasing: {} → {}", w[0], w[1]);
        }
        assert!(s.sigmas[0] > 1.0, "first sigma should be well above 1.0");
        assert_eq!(s.sigmas[20], 0.0, "terminal sigma must be exactly 0.0");
    }

    /// Dispatch-level integration: `Scheduler::UniPc` dispatched through
    /// `step_array_into` must not panic and must produce output of the
    /// expected length. Pins the wiring from SchedulerKind → Scheduler →
    /// step_array_into without requiring a live ORT session.
    #[test]
    fn unipc_step_array_into_dispatch_roundtrip() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n = 8_usize;
        let dim = 16_usize;
        let mut sched = Scheduler::UniPc(UniPcScheduler::new_sd15(n));
        let latent:    Vec<f32> = (0..dim).map(|i| 0.1 * i as f32).collect();
        let noise_pred:Vec<f32> = (0..dim).map(|i| 0.05 * i as f32).collect();
        let mut out = Vec::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let ts = sched.timesteps().to_vec();
        step_array_into(&mut sched, &latent, &noise_pred, 0, ts[0], ts[0], false, &mut rng, &mut out);

        assert_eq!(out.len(), dim, "step_array_into output length must match input");
        assert!(out.iter().all(|v| v.is_finite()),
            "step_array_into must produce finite output: {out:?}");
    }
}
