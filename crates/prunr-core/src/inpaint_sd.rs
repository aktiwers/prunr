//! Stable Diffusion 1.5 Inpainting pipeline.
//!
//! Per-stroke flow: CLIP-tokenize prompt → text encoder; VAE-encode the
//! masked source; downsample mask to latent space; sample gaussian noise;
//! denoise via LCM or DPM++ 2M Karras (with optional classifier-free
//! guidance); VAE-decode
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
//! - Idle bundle release (`crate::IDLE_RELEASE`) drops the cached 4
//!   ORT sessions after no use — reclaims ~4-6 GB so users who erased
//!   ten minutes ago aren't carrying SD weights for the rest of the
//!   session.
//!
//! When the user picks the LCM scheduler on the SD chip (and the LCM
//! bundle is installed), the dispatcher routes to the LCM-distilled
//! checkpoint — 4 steps, guidance baked into training. That gating
//! happens upstream; this module sees the chosen `ModelId`.

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
/// Latent side of a 512² tile; the pipeline derives its own from the
/// crop, this only shapes the session smoke tests.
const SD_LATENT_SIDE: u32 = SD_TILE / 8;
/// Longest crop side with `SdTuning::tall_crop`: a region up to this
/// long runs as one 512×768 / 768×512 / 768² crop instead of tiles.
pub const SD_CROP_MAX: u32 = 768;
/// The largest region SD fills at full size in one crop with 32 px of
/// photo around it: up to 768 on its long side with the tall crop.
pub fn fit(req: &SdInpaintRequest) -> crate::inpaint::Fit {
    let side = |crop: u32| crop - 64;
    crate::inpaint::Fit {
        long: side(if req.tall_crop() { SD_CROP_MAX } else { SD_TILE }),
        short: side(SD_TILE),
    }
}

/// The UNet downsamples three times, so a crop side is a multiple of 64.
const SD_SIDE_STEP: u32 = 64;
/// The steps RORem's LCM LoRA was distilled for.
pub const ROREM_STEPS: u32 = 4;

/// The SD pipeline's facts about `id`; every id it is given is SD-family.
fn sd_spec(id: prunr_models::ModelId) -> prunr_models::SdSpec {
    // sd_spec is Some for every id routed here (is_sd_family is defined by it)
    id.sd_spec().expect("the SD pipeline runs SD-family models only")
}

/// Minimum free RAM (bytes) required before we'll load the SD bundle.
/// The 4 ONNX files total ~2 GB on disk; ORT graph optimization roughly
/// doubles that during load (~4 GB resident), UNet activations on a
/// 512×512 tile add 2-4 GB transient, and OpenVINO graph compilation
/// pre-allocates its own iGPU-shared buffer (which on integrated GPUs
/// IS RAM). Threshold lives in `ModelDescriptor.working_set_mb` so the
/// gate scales when a new SD variant ships with a different footprint.
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
    /// How the pipeline runs for this stroke; see `plan_tuning`.
    #[serde(default)]
    pub tuning: SdTuning,
    /// How large a region is filled; Auto shrinks one that would need
    /// tiles to fit one crop.
    #[serde(default)]
    pub fill_scale: crate::inpaint::FillScale,
}

/// How the SD pipeline runs, built at dispatch (`plan_tuning`) from the
/// settings and carried on the request, so the worker needs no settings
/// of its own; the RAM gate trims it where the stroke runs. Measured on an i7-6700: the
/// tall crop takes 21 % off a 328×607 erase and removes the seam for
/// 2.8 GB more peak RSS; the CPU plugin beats the HD 530 iGPU by half.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SdTuning {
    /// Leave the bundle resident after the stroke (about 16 GB) so the
    /// next one skips the ~24 s session build.
    pub keep_loaded: bool,
    /// OpenVINO device for the SD parts ("CPU", "GPU"); `None` is the
    /// EP's default, the CPU. A bench-only override for machines with a
    /// strong Intel GPU: a session built for it never records EP
    /// failures, so a broken override cannot poison the default path.
    pub ov_device: Option<String>,
    /// Run a region up to `SD_CROP_MAX` on its long side as one crop
    /// instead of 512² tiles, if the RAM gate finds room for it.
    pub tall_crop: bool,
    /// Headroom the RAM gate adds to the model's working set.
    #[serde(default = "default_margin_mb")]
    pub margin_mb: u64,
    /// The Processor setting's "CPU only": no GPU provider for the parts.
    #[serde(default)]
    pub cpu_only: bool,
    /// How the sessions use memory; the default is ORT's own.
    #[serde(default)]
    pub mem: SdMemPlan,
}

/// Session memory knobs for experiments (`prunr-export/rorem_mem`); the
/// default leaves each to the model (`SdSpec::lean_sessions`) or ORT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SdMemPlan {
    /// ORT's memory-pattern planning, which pre-sizes every activation
    /// for the first shape it sees; `None` is the model's own choice.
    pub memory_pattern: Option<bool>,
    /// The weights outside the arena, so it does not grow around them.
    pub device_initializers: bool,
    /// Threads for the UNet's matmuls; 0 is every hardware thread.
    pub intra_threads: u8,
}

/// The headroom when the caller names none. It covers the free-RAM drop
/// between the gate and the allocations deep in ORT (5-30 s later for
/// SD): another app spiking, swapping, or compile transients the working
/// set under-models. 2 GB covers what we see without making the gate
/// unfriendly to 16-GB machines.
pub const SD_DEFAULT_MARGIN_MB: u64 = 2_000;

fn default_margin_mb() -> u64 { SD_DEFAULT_MARGIN_MB }

impl Default for SdTuning {
    fn default() -> Self {
        Self { keep_loaded: false, ov_device: None, tall_crop: false, margin_mb: SD_DEFAULT_MARGIN_MB, cpu_only: false, mem: SdMemPlan::default() }
    }
}

/// RAM the tall crop needs beyond the model's gate (measured peak RSS
/// 18.8 GB against 16.1 GB).
pub const SD_TALL_CROP_EXTRA_MB: u64 = 3_072;

/// The tuning for a stroke: the tall crop wherever it fits. Whether it
/// fits is decided by `check_ram_for` in the process that runs the
/// stroke, because only there is a kept bundle visible as loaded; the
/// GUI's own view would count a kept 16 GB bundle twice.
pub fn plan_tuning(keep_loaded: bool, margin_mb: u64) -> SdTuning {
    SdTuning { keep_loaded, ov_device: None, tall_crop: true, margin_mb, cpu_only: false, mem: SdMemPlan::default() }
}

impl SdTuning {
    /// `self` with the bench overrides from the environment, for
    /// experiments the app and the bench run the same way:
    /// `PRUNR_SD_KEEP_LOADED`, `PRUNR_SD_TALL_CROP`, `PRUNR_SD_OV_DEVICE`,
    /// `PRUNR_SD_MEM_PATTERN`, `PRUNR_SD_DEV_INIT` and `PRUNR_SD_THREADS`.
    pub fn with_env_overrides(self) -> Self {
        Self {
            keep_loaded: env_flag("PRUNR_SD_KEEP_LOADED").unwrap_or(self.keep_loaded),
            tall_crop: env_flag("PRUNR_SD_TALL_CROP").unwrap_or(self.tall_crop),
            ov_device: std::env::var("PRUNR_SD_OV_DEVICE").ok().filter(|s| !s.is_empty()).or(self.ov_device),
            mem: SdMemPlan {
                memory_pattern: env_flag("PRUNR_SD_MEM_PATTERN").or(self.mem.memory_pattern),
                device_initializers: env_flag("PRUNR_SD_DEV_INIT").unwrap_or(self.mem.device_initializers),
                intra_threads: std::env::var("PRUNR_SD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(self.mem.intra_threads),
            },
            ..self
        }
    }
}

fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Free RAM a stroke needs: the model's working set unless its bundle
/// is already resident, the headroom, and the tall crop's extra.
fn ram_need_mb(id: prunr_models::ModelId, margin_mb: u64, resident: bool, tall: bool) -> u64 {
    let working_set = if resident { 0 } else { prunr_models::descriptor(id).map_or(0, |d| d.working_set_mb as u64) };
    working_set + margin_mb + if tall { SD_TALL_CROP_EXTRA_MB } else { 0 }
}

/// How a Stable Diffusion erase fits in `free_mb` with no bundle loaded,
/// by the same rule as the run-time gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdRamFit {
    /// Tall regions run as one crop.
    Full,
    /// It runs, but tall regions fall back to 512 px tiles.
    TilesOnly,
    /// The gate refuses the erase.
    Refused,
}

pub fn sd_ram_fit(free_mb: u64, margin_mb: u64) -> SdRamFit {
    let id = prunr_models::ModelId::SdV15InpaintFp16;
    if free_mb < ram_need_mb(id, margin_mb, false, false) {
        SdRamFit::Refused
    } else if free_mb < ram_need_mb(id, margin_mb, false, true) {
        SdRamFit::TilesOnly
    } else {
        SdRamFit::Full
    }
}

fn available_ram_mb() -> Option<u64> {
    available_ram_bytes().map(|b| b / (1024 * 1024))
}

/// A built bundle for `id` is in the cache, so its working set is
/// already counted in the RSS rather than in the free RAM.
fn resident(id: prunr_models::ModelId) -> bool {
    sd_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .any(|((cached, _), e)| *cached == id && matches!(e.value.get(), Some(Ok(_))))
}

impl SdTuning {
    fn session_key(&self) -> SdSessionKey {
        SdSessionKey { ov_device: self.ov_device.clone(), cpu_only: self.cpu_only, mem: self.mem }
    }
}

/// What a built session bundle depends on besides the model. Shapes are
/// not part of it: the OpenVINO sessions are built dynamic, so one
/// bundle serves tiles and tall crops alike.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct SdSessionKey {
    ov_device: Option<String>,
    cpu_only: bool,
    mem: SdMemPlan,
}

impl SdSessionKey {
    /// The GPU providers this bundle's parts may try.
    fn gpu_eps(&self) -> &'static [crate::engine::EpKind] {
        crate::engine::gpu_eps_for(self.cpu_only)
    }
}

impl SdSessionKey {
    /// A device override is bench-only (see `SdTuning::ov_device`).
    fn records_ep_failures(&self) -> bool {
        self.ov_device.is_none()
    }
}


/// A seed for a fill nobody pinned.
pub fn fresh_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// Where a refused SD erase sends the user instead.
pub const LIGHTER_ERASER_HINT: &str = "pick Eraser (Big-LaMa) under Model instead";

impl SdInpaintRequest {
    /// Whether this run may use the tall crop: TAESD's graph is fixed at
    /// 512², so with it a 768 crop failed outright.
    pub fn tall_crop(&self) -> bool {
        self.tuning.tall_crop && !self.use_taesd
    }
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
            tuning: SdTuning::default(),
            fill_scale: Default::default(),
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
        &crate::progress::Progress::none(),
    ).map(|f| f.rgba)
}

/// Full-knob entry. The `inpaint::process_inpaint` shim calls this with
/// defaults; future text-prompt + CFG surfaces wire through here.
/// See `inpaint::process_inpaint_with` for the progress contract.
pub fn process_inpaint_with(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    mut req: SdInpaintRequest,
    progress: &crate::progress::Progress,
) -> Result<crate::inpaint::Filled, CoreError> {
    use crate::progress::Step;
    // The other quality's bundle goes before the RAM gate counts it: it
    // would be dropped on load anyway, and kept-loaded it filled RAM.
    drop_other_bundles(
        &mut sd_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        &(id, req.tuning.session_key()),
    );
    check_ram_for(id, &mut req.tuning).map_err(CoreError::Inference)?;
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

    if mask_regions(mask).is_empty() {
        return Ok(crate::inpaint::Filled::full(image.clone()));
    }
    progress.step(Step::LoadingModel);

    // Hold an Arc through the run so the idle sweep can't drop sessions
    // mid-inference.
    let get_started = Instant::now();
    let bundle = SdSession::get(id, &req.tuning.session_key(), req.tuning.keep_loaded)?;
    let session_ms = get_started.elapsed().as_millis() as u64;
    // `bundle` is declared above the guard so it drops after it: the
    // guard removes the cache's Arc (unless the entry is pinned), then
    // the last reference goes and the ORT bundle is freed.
    let _release_guard = ReleaseOnDrop(id);
    let run_started = Instant::now();
    // VAE backend selection: TAESD when fast mode is on AND the bundle
    // is installed (the request flag carries that decision from
    // dispatch). Until the TAESD artifact ships, get() errors and we
    // silently fall back to standard VAE — same graceful pattern as LCM.
    let taesd = if req.use_taesd && sd_spec(id).takes_taesd { TaesdSession::get().ok() } else { None };
    let vae: VaeBackend = match taesd.as_ref() {
        Some(t) => VaeBackend::Taesd(t),
        None => VaeBackend::Standard(&bundle),
    };
    // One session for every region the fill scale fills on its own. The
    // fit reads the tall crop as the RAM gate above left it.
    let out = crate::inpaint::fill_at_scale(image, mask, req.fill_scale, fit(&req), progress, |img, m, progress| {
        paint_regions(&bundle, &vae, img, m, &req, progress)
    })?;
    if progress.is_cancelled() { return Err(CoreError::Cancelled); }
    tracing::info!(
        session_ms, run_ms = run_started.elapsed().as_millis() as u64,
        tuning = ?req.tuning,
        "SD: inpaint done",
    );
    Ok(out)
}

/// Fill every region of `mask` in `image` with the loaded bundle: one
/// crop per region that fits, overlapping tiles for one that doesn't.
fn paint_regions(
    bundle: &SdSession,
    vae: &VaeBackend,
    image: &RgbaImage,
    mask: &GrayImage,
    req: &SdInpaintRequest,
    progress: &crate::progress::Progress,
) -> Result<RgbaImage, CoreError> {
    use crate::progress::{TileRect, Unit};
    let components = mask_regions(mask);
    // One plan per component, made once: the outer progress total and
    // the loop read the same value, so they cannot disagree about how
    // many tiles a region near an image edge really gets.
    let (img_w, img_h) = image.dimensions();
    let max_side = if req.tall_crop() { SD_CROP_MAX } else { SD_TILE };
    let plans: Vec<RegionPlan> = components.iter().map(|c| plan_region(c, img_w, img_h, max_side)).collect();
    let outer_total: u32 = plans.iter().map(|p| p.tile_count()).sum();
    // The crops and tiles, in the order they run: the tile map's layout.
    progress.tiles(plans.iter().flat_map(|plan| match plan {
        RegionPlan::Crop { x, y, w, h } => vec![TileRect::of_pixels(*x, *y, *w, *h, img_w, img_h)],
        RegionPlan::Tiles(tiles) => tiles.iter().map(|t| TileRect::of_pixels(t.x, t.y, t.w, t.h, img_w, img_h)).collect(),
    }).collect());
    let mut tile_idx: u32 = 0;
    // One crop or tile starts: the outer count, and the tile map's cell.
    let start = |idx: u32| {
        progress.outer(idx, outer_total, Unit::Crop);
        progress.tile(idx, false);
    };
    let mut out = image.clone();
    for (component, plan) in components.iter().zip(plans) {
        if progress.is_cancelled() { return Err(CoreError::Cancelled); }
        let tiles = match plan {
            RegionPlan::Crop { x: cx, y: cy, w: cw, h: ch } => {
                start(tile_idx);
                let cropped_img = image::imageops::crop_imm(&out, cx, cy, cw, ch).to_image();
                let cropped_mask = image::imageops::crop_imm(mask, cx, cy, cw, ch).to_image();
                // A failed region fails the run: left as it was, it would read
                // as a finished erase.
                let painted = run_one_tile(bundle, vae, &cropped_img, &cropped_mask, req, progress)?;
                image::imageops::replace(&mut out, &painted, cx as i64, cy as i64);
                progress.tile(tile_idx, true);
                tile_idx += 1;
                continue;
            }
            RegionPlan::Tiles(tiles) => tiles,
        };
        // Longer than `max_side` on an axis: overlapping 512² tiles with
        // alpha-blended seams.
        tracing::info!(
            comp = ?component, n_tiles = tiles.len(),
            "SD: tiling oversized component",
        );
        for tile in tiles {
            if progress.is_cancelled() { return Err(CoreError::Cancelled); }
            start(tile_idx);
            let cropped_img = image::imageops::crop_imm(&out, tile.x, tile.y, tile.w, tile.h).to_image();
            let cropped_mask = image::imageops::crop_imm(mask, tile.x, tile.y, tile.w, tile.h).to_image();
            let painted = run_one_tile(bundle, vae, &cropped_img, &cropped_mask, req, progress)?;
            blend_tile(&mut out, &painted, &tile);
            progress.tile(tile_idx, true);
            tile_idx += 1;
        }
    }
    progress.outer(outer_total, outer_total, Unit::Crop);
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
/// extent and `pad_to` handles the rest at the ORT boundary.
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
pub(crate) struct MaskBbox {
    x_min: u32, y_min: u32, x_max: u32, y_max: u32,
}

impl MaskBbox {
    pub(crate) fn to_bbox(self) -> crate::inpaint::Bbox {
        crate::inpaint::Bbox { x: self.x_min, y: self.y_min, w: self.x_max - self.x_min + 1, h: self.y_max - self.y_min + 1 }
    }
}

/// Pieces of a mask closer than this are one region: a slatted bench or
/// a fence selected by Magic Brush is hundreds of pieces split by the
/// gaps between slats, and filling each on its own ran hundreds of crops.
pub(crate) const REGION_GAP: u32 = 64;

/// The regions a fill plans for: the mask's pieces (`mask_components`),
/// those within `REGION_GAP` of each other grouped under one bbox.
pub(crate) fn mask_regions(mask: &GrayImage) -> Vec<MaskBbox> {
    group_near(mask_components(mask), REGION_GAP)
}

fn group_near(mut regions: Vec<MaskBbox>, gap: u32) -> Vec<MaskBbox> {
    let near = |a: &MaskBbox, b: &MaskBbox| {
        a.x_min <= b.x_max.saturating_add(gap) && b.x_min <= a.x_max.saturating_add(gap)
            && a.y_min <= b.y_max.saturating_add(gap) && b.y_min <= a.y_max.saturating_add(gap)
    };
    // Grouping can bring a bbox near one already passed, so repeat until
    // a pass groups nothing.
    loop {
        let mut grouped = false;
        let mut i = 0;
        while i < regions.len() {
            let mut j = i + 1;
            while j < regions.len() {
                if near(&regions[i], &regions[j]) {
                    let b = regions.swap_remove(j);
                    let a = &mut regions[i];
                    *a = MaskBbox { x_min: a.x_min.min(b.x_min), y_min: a.y_min.min(b.y_min), x_max: a.x_max.max(b.x_max), y_max: a.y_max.max(b.y_max) };
                    grouped = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !grouped {
            return regions;
        }
    }
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
pub(crate) fn mask_components(mask: &GrayImage) -> Vec<MaskBbox> {
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

/// Releases the stroke's cache entry when the stroke ends, however it
/// ends; a pinned entry (`SdTuning::keep_loaded`) stays.
struct ReleaseOnDrop(prunr_models::ModelId);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        release(self.0);
    }
}

/// How one painted region runs: as a single crop or as tiles.
enum RegionPlan {
    Crop { x: u32, y: u32, w: u32, h: u32 },
    Tiles(Vec<TileWindow>),
}

impl RegionPlan {
    fn tile_count(&self) -> u32 {
        match self {
            Self::Crop { .. } => 1,
            Self::Tiles(t) => t.len() as u32,
        }
    }
}

fn plan_region(bbox: &MaskBbox, img_w: u32, img_h: u32, max_side: u32) -> RegionPlan {
    match compute_sd_crop(bbox, img_w, img_h, max_side) {
        Some((x, y, w, h)) => RegionPlan::Crop { x, y, w, h },
        None => RegionPlan::Tiles(tile_bbox(bbox, img_w, img_h)),
    }
}

/// The smallest crop side from `SD_TILE` up to `max_side` in `SD_TILE / 2`
/// steps that covers `span`; `None` when the region needs tiles.
fn crop_side(span: u32, max_side: u32) -> Option<u32> {
    (SD_TILE..=max_side).step_by((SD_TILE / 2) as usize).find(|&side| side >= span)
}

/// One crop centred on the bbox, clamped to image bounds, each side the
/// smallest that covers the region (see `crop_side`); `None` when the
/// region exceeds `max_side` on an axis or `SD_TILE` on both (only one
/// long side was measured; a 768² crop goes to tiles). For images
/// smaller than the crop on an axis, the crop shrinks to that dimension
/// (`pad_to` pads it back).
fn compute_sd_crop(bbox: &MaskBbox, img_w: u32, img_h: u32, max_side: u32) -> Option<(u32, u32, u32, u32)> {
    let cw = crop_side(bbox.x_max - bbox.x_min + 1, max_side)?;
    let ch = crop_side(bbox.y_max - bbox.y_min + 1, max_side)?;
    if cw > SD_TILE && ch > SD_TILE {
        return None;
    }
    let (cw, ch) = (cw.min(img_w), ch.min(img_h));
    let cx_centre = (bbox.x_min + bbox.x_max) / 2;
    let cy_centre = (bbox.y_min + bbox.y_max) / 2;
    let x = cx_centre.saturating_sub(cw / 2).min(img_w - cw);
    let y = cy_centre.saturating_sub(ch / 2).min(img_h - ch);
    Some((x, y, cw, ch))
}

/// The model-side size of a crop: at least `SD_TILE`, rounded up to a
/// multiple of `SD_SIDE_STEP` on each axis.
fn padded_dims(w: u32, h: u32) -> (u32, u32) {
    let up = |v: u32| v.max(SD_TILE).div_ceil(SD_SIDE_STEP) * SD_SIDE_STEP;
    (up(w), up(h))
}

/// Eagerly initialise the SD session so the first stroke doesn't pay
/// the cumulative ~10-30s session-build latency for 4 ONNX files.
pub fn prewarm(id: prunr_models::ModelId) -> Result<(), CoreError> {
    SdSession::get(id, &SdSessionKey::default(), false).map(|_| ())
}

// ── Per-tile pipeline ───────────────────────────────────────────────────

fn run_one_tile(
    bundle: &SdSession,
    vae: &VaeBackend,
    image: &RgbaImage,
    mask: &GrayImage,
    req: &SdInpaintRequest,
    progress: &crate::progress::Progress,
) -> Result<RgbaImage, CoreError> {
    use crate::progress::{Step, Unit};
    let (w, h) = image.dimensions();
    let (pw, ph) = padded_dims(w, h);
    let padded_image = pad_to(image, pw, ph);
    let padded_mask = pad_to(mask, pw, ph);
    let (lw, lh) = ((pw / 8) as usize, (ph / 8) as usize);
    let mut timer = TileTimer::start();


    // CFG threshold: above 1.0 we run the UNet TWICE per step (cond +
    // uncond) and blend by `guidance_scale`. At ≤1.0 the cond pass is
    // all the user wants, so we skip the second to halve UNet cost.
    // A baked-in prompt runs unguided: there is no prompt to steer by.
    let prompted = bundle.text_encoder.is_some();
    let use_cfg = prompted && req.guidance_scale > 1.0 + 1e-3;

    // Pre-loop independent ops in parallel: text encode (cond + uncond
    // when CFG), VAE encode, mask-to-latent. Each ORT call holds a
    // distinct session mutex (or none) so they run concurrently.
    if prompted {
        progress.step(Step::ReadingPrompt);
    }
    let prompt = req.prompt.clone();
    let neg_prompt = if use_cfg { Some(req.negative_prompt.clone()) } else { None };
    let (text_emb_cond_f16, text_emb_uncond_f16, masked_latent, mask_latent) =
        std::thread::scope(|s| -> Result<_, CoreError> {
            let cond_h = prompted.then(|| s.spawn(|| text_embedding_f16(bundle, &prompt)));
            let uncond_h = neg_prompt.as_ref().map(|np| {
                let np = np.clone();
                s.spawn(move || text_embedding_f16(bundle, &np))
            });
            let vae_h = s.spawn(|| vae_encode_masked(vae, &padded_image, &padded_mask));
            let mask_lat = mask_to_latent(&padded_mask);
            let cond = cond_h.map(|h| h.join()
                .map_err(|_| CoreError::Inference("text encoder (cond) thread panicked".into()))?)
                .transpose()?;
            let uncond = match uncond_h {
                Some(h) => Some(h.join()
                    .map_err(|_| CoreError::Inference("text encoder (uncond) thread panicked".into()))??),
                None => None,
            };
            let vae = vae_h.join()
                .map_err(|_| CoreError::Inference("vae encoder thread panicked".into()))??;
            Ok((cond, uncond, vae, mask_lat))
        })?;

    let masked_latent_f16 = f32_to_f16_4d(&masked_latent);
    let mask_latent_f16 = f32_to_f16_4d(&mask_latent);
    // Free the f32 originals — the loop only needs the f16 mirrors,
    // and these buffers would otherwise sit alive across the entire
    // 20-step UNet loop, the longest-lived stage of the pipeline.
    drop(masked_latent);
    drop(mask_latent);
    let prelude_ms = timer.lap();

    let steps = req.num_inference_steps as usize;
    let mut scheduler = match req.scheduler {
        SchedulerKind::Lcm => Scheduler::Lcm(LcmScheduler::new_sd15(steps)),
        SchedulerKind::DpmPp2MKarras => Scheduler::DpmPp2M(DpmPp2MScheduler::new_sd15(steps)),
    };
    let seed = req.seed.unwrap_or_else(fresh_seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let total_steps = scheduler.timesteps().len() as u32;
    progress.step(Step::Denoising);
    progress.inner(0, total_steps, Unit::Step);

    // Hold denoising state as Vec<f32> + captured dim. f16 conversion
    // uses ArrayView4::from_shape (zero-copy on the f32 side); the
    // scheduler writes via step_array_into into a reused scratch buffer.
    let latent_dim = (1_usize, 4_usize, lh, lw);
    let mut latent_buf: Vec<f32> = sample_initial_noise(&mut rng, lw, lh).into_raw_vec_and_offset().0;

    // Denoising loop. With CFG: noise_pred = uncond + scale * (cond - uncond).
    // Without CFG: just one UNet pass with cond.
    let timesteps = scheduler.timesteps().to_vec();
    let scale = req.guidance_scale;
    // Scratch buffer for step output — hoisted outside the loop so
    // mem::swap reuses the allocation across steps (no per-step alloc).
    let mut step_out_buf: Vec<f32> = Vec::with_capacity(latent_buf.len());
    for (i, &t) in timesteps.iter().enumerate() {
        // Check cancel between UNet steps. ORT has no per-op cancel, so
        // worst-case latency on cancel is one UNet step (multi-second).
        if progress.is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        // Zero-copy view of latent_buf — only the Array4<f16> output stays.
        let latent_f16 = f32_to_f16_view(
            ndarray::ArrayView4::from_shape(latent_dim, &latent_buf)
                .expect("latent_dim matches latent_buf length by construction"),
        );
        let latent_in_f16 = concat_inpaint_input_f16(
            &latent_f16, &mask_latent_f16, &masked_latent_f16,
        );
        let noise_pred = match (text_emb_cond_f16.as_ref(), text_emb_uncond_f16.as_ref()) {
            (Some(cond), Some(uncond)) => cfg_noise_pred(bundle, latent_in_f16, t, cond, uncond, scale)?,
            (Some(cond), None) => unet_step(bundle, latent_in_f16, t, Cond::Text(cond))?,
            (None, _) => unet_step(bundle, latent_in_f16, t, Cond::Size(ph, pw))?,
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
        progress.inner(i as u32 + 1, total_steps, Unit::Step);
    }

    let unet_ms = timer.lap();

    // VAE decode — wrap final buf into Array4 (no extra allocation).
    let final_array = ndarray::Array4::from_shape_vec(latent_dim, latent_buf)
        .expect("latent_dim matches latent_buf length by construction");
    progress.step(Step::DecodingResult);
    let painted = vae_decode(vae, &final_array)?;
    let decode_ms = timer.lap();
    let out = composite(image, &painted, mask, w, h);
    let steps_run = timesteps.len() as u64;
    tracing::info!(
        w, h, steps = steps_run, prelude_ms,
        unet_ms_per_step = unet_ms / steps_run.max(1),
        decode_ms, total_ms = timer.total(),
        rss_mb = process_rss_mb(),
        cfg = use_cfg,
        "SD: tile done",
    );
    Ok(out)
}

/// Stage clock for one tile's log line.
struct TileTimer {
    start: Instant,
    mark: Instant,
}

impl TileTimer {
    fn start() -> Self {
        let now = Instant::now();
        Self { start: now, mark: now }
    }

    /// Milliseconds since the previous lap (or the start).
    fn lap(&mut self) -> u64 {
        let now = Instant::now();
        let ms = now.duration_since(self.mark).as_millis() as u64;
        self.mark = now;
        ms
    }

    fn total(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

/// One guided step: the batched UNet call (cond and uncond together)
/// when the export allows it, else two calls. The first batched
/// failure (a static batch-1 export) switches the session to two calls
/// for good.
fn cfg_noise_pred(
    bundle: &SdSession,
    latent_in: Array4<f16>,
    t: i64,
    cond: &Array3<f16>,
    uncond: &Array3<f16>,
    scale: f32,
) -> Result<Array4<f32>, CoreError> {
    use std::sync::atomic::Ordering;
    if !bundle.cfg_fallback_to_sequential.load(Ordering::Relaxed) {
        match unet_step_batched(bundle, &latent_in, t, cond, uncond) {
            Ok(pair) => {
                let pred_cond = pair.slice(ndarray::s![0..1, .., .., ..]);
                let pred_uncond = pair.slice(ndarray::s![1..2, .., .., ..]);
                return Ok(cfg_blend(pred_uncond, pred_cond, scale));
            }
            Err(e) => {
                tracing::warn!(%e, "SD: batched CFG UNet rejected (static batch=1 export); \
                                   running cond and uncond separately for this session");
                bundle.cfg_fallback_to_sequential.store(true, Ordering::Relaxed);
            }
        }
    }
    let pred_cond = unet_step(bundle, latent_in.clone(), t, Cond::Text(cond))?;
    let pred_uncond = unet_step(bundle, latent_in, t, Cond::Text(uncond))?;
    Ok(cfg_blend(pred_uncond.view(), pred_cond.view(), scale))
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
            crate::ort_runtime::session_builder()?
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

pub(crate) struct SdSession {
    unet: Mutex<Session>,
    vae_encoder: Mutex<Session>,
    vae_decoder: Mutex<Session>,
    /// SD 1.5's CLIP encoder. `None` for a bundle with its prompt baked
    /// into the UNet (RORem), whose third UNet input is SDXL's size
    /// conditioning instead of a text embedding.
    text_encoder: Option<TextEncoder>,
    /// Discovered input names per session (positional order from the ONNX).
    /// SD exports use stable names (sample / timestep / encoder_hidden_states
    /// or time_ids for UNet, sample for VAEs, input_ids for text encoder) but
    /// we look them up by introspection so a non-standard export still works.
    unet_inputs: [String; 3],
    vae_encoder_input: String,
    vae_decoder_input: String,
    vae_scale: f32,
    /// The VAE takes f32: SDXL's overflows in f16.
    vae_f32: bool,
    /// Set once a batched UNet call fails (a static batch-1 export, the
    /// LCM bundle); guided steps then run cond and uncond one after the
    /// other for this session's lifetime.
    cfg_fallback_to_sequential: std::sync::atomic::AtomicBool,
    /// CLIP embeddings by prompt, in f16. The encoder is deterministic, so
    /// a repeated prompt (every tile, every stroke) skips it. Bounded by
    /// `TEXT_CACHE_ENTRIES`; ~118 KB each.
    text_cache: Mutex<Vec<(String, Arc<Array3<f16>>)>>,
}

const TEXT_CACHE_ENTRIES: usize = 8;

struct TextEncoder {
    session: Mutex<Session>,
    input: String,
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
    /// Kept through the idle sweep and the end-of-stroke release
    /// (`SdTuning::keep_loaded`).
    pub(crate) pinned: bool,
}

impl<T> CacheEntry<T> {
    pub(crate) fn new(value: T, now: Instant) -> Self {
        Self { value, last_used: now, pinned: false }
    }
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
type SdCache = HashMap<(prunr_models::ModelId, SdSessionKey), CacheEntry<SdBundleSlot>>;

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
    let before = guard.len();
    guard.retain(|(cached, _), e| *cached != id || e.pinned);
    if guard.len() < before {
        tracing::info!(?id, "SD: cache entry released after dispatch");
    }
}

/// Drop entries whose `last_used` is older than `idle` relative to `now`,
/// returning the count dropped. Generic over `T` so tests can drive the
/// sweep without spinning up a real ORT session.
///
/// `pub(crate)` so `inpaint::LamaSession`'s sweeper reuses this directly.
pub(crate) fn sweep_idle<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    now: Instant,
    idle: Duration,
) -> usize {
    let before = cache.len();
    cache.retain(|_, e| e.pinned || now.duration_since(e.last_used) < idle);
    before - cache.len()
}

/// Release every cached bundle but `keep`'s, pinned or not.
fn drop_other_bundles<K: std::hash::Hash + Eq, T>(cache: &mut HashMap<K, CacheEntry<T>>, keep: &K) {
    cache.retain(|k, _| k == keep);
}

/// The cache entry for `key`, made with `make` when missing, after
/// dropping every other entry: one SD bundle is resident at a time, so
/// "keep loaded" holds the eraser in use, not each one used before.
/// A dropped bundle a run still holds lives until that run ends.
fn admit<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    key: K,
    make: impl FnOnce() -> T,
    now: Instant,
    pin: bool,
) -> &mut CacheEntry<T> {
    drop_other_bundles(cache, &key);
    let entry = cache.entry(key).or_insert_with(|| CacheEntry::new(make(), now));
    entry.pinned = pin;
    entry
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
                let idle = crate::IDLE_RELEASE;
                loop {
                    std::thread::sleep(interval);
                    let cache = sd_cache();
                    let mut guard = cache.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let dropped = sweep_idle(&mut guard, Instant::now(), idle);
                    if dropped > 0 {
                        tracing::info!(
                            dropped,
                            idle_secs = crate::IDLE_RELEASE.as_secs(),
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
    fn get(id: prunr_models::ModelId, key: &SdSessionKey, pin: bool) -> Result<Arc<SdSession>, CoreError> {
        ensure_sweeper_running();
        let cache = sd_cache();
        let now = Instant::now();
        let idle = crate::IDLE_RELEASE;

        // Cache lock held only for the HashMap lookup and at most one
        // Arc<OnceLock> allocation — never for the bundle build.
        let slot: SdBundleSlot = {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dropped = sweep_idle(&mut guard, now, idle);
            if dropped > 0 {
                tracing::info!(
                    dropped,
                    idle_secs = crate::IDLE_RELEASE.as_secs(),
                    rss_mb = process_rss_mb(),
                    "SD: released idle session bundle(s)",
                );
            }
            let entry = admit(&mut guard, (id, key.clone()), || Arc::new(std::sync::OnceLock::new()), now, pin);
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
        slot.get_or_init(|| Self::new_inner(id, key).map(Arc::new))
            .clone()
            .map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId, session_key: &SdSessionKey) -> Result<SdSession, String> {
        // Defense-in-depth: prewarm builds bypass `process_inpaint_with`,
        // so the gate has to fire here too. Same helper as the dispatch
        // entry — single source of truth for threshold + wording.
        check_ram_for(id, &mut SdTuning::default())?;
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
                let (s, ep) = build_part_with_ep_ladder(id, key, &by_key, smoke, session_key)?;
                Ok((key, s, ep))
            };

        // A bundle with its prompt baked in ships no text encoder.
        let has_text = by_key.contains_key("text_encoder");
        let wanted: Vec<(&'static str, SmokeFn)> = PARTS.iter().copied()
            .filter(|(key, _)| *key != "text_encoder" || has_text)
            .collect();
        let mut parts: Vec<(&'static str, Session, String)> =
            if crate::engine::directml_active() {
                wanted.iter().map(build).collect::<Result<_, _>>()?
            } else {
                use rayon::prelude::*;
                wanted.par_iter().map(build).collect::<Result<_, _>>()?
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
        let text = has_text.then(|| take("text_encoder")).transpose()?;
        let (vae_encoder,  vae_enc_ep) = take("vae_encoder")?;
        let (vae_decoder,  vae_dec_ep) = take("vae_decoder")?;
        let (unet,         unet_ep)    = take("unet")?;

        let unet_inputs = take_three_inputs(&unet, "unet")?;
        let vae_encoder_input = take_first_input(&vae_encoder, "vae_encoder")?;
        let vae_decoder_input = take_first_input(&vae_decoder, "vae_decoder")?;
        let text_encoder = text.map(|(session, ep)| -> Result<(TextEncoder, String), String> {
            let input = take_first_input(&session, "text_encoder")?;
            Ok((TextEncoder { session: Mutex::new(session), input }, ep))
        }).transpose()?;
        let vae_f32 = input_is_f32(&vae_encoder, 0);

        let rss_after_mb = process_rss_mb();
        tracing::info!(
            ?id,
            text_encoder_ep = ?text_encoder.as_ref().map(|(_, ep)| ep),
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
            text_encoder: text_encoder.map(|(t, _)| t),
            unet_inputs,
            vae_encoder_input,
            vae_decoder_input,
            vae_scale: sd_spec(id).vae_scale,
            vae_f32,
            cfg_fallback_to_sequential: std::sync::atomic::AtomicBool::new(false),
            text_cache: Mutex::new(Vec::new()),
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
    session_key: &SdSessionKey,
) -> Result<(Session, String), String> {
    let path = by_key.get(key)
        .ok_or_else(|| format!("SD bundle missing required part: {key}"))?;

    let gpu_eps = session_key.gpu_eps();

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
        let builder = match sd_base_builder(session_key.mem, id.sd_spec().is_some_and(|s| s.lean_sessions)) {
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
        let record_failure = |msg: &str| {
            if session_key.records_ep_failures() {
                crate::engine::handle_commit_failure(
                    matches!(load_path, Cow::Owned(_)),
                    ep, id,
                    || crate::cache::optimized_model_path_for_part(id, ep.as_str(), key),
                    msg,
                );
            }
        };
        let registered = match ep {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default()
                    .with_device_id(0)
                    .build(),
            ]),
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => {
                let mut p = ort::ep::CoreML::default();
                if let Some(dir) = crate::cache::cache_dir_for_part(id, ep.as_str(), key) {
                    p = p.with_model_cache_dir(dir.to_string_lossy().into_owned());
                }
                builder.with_execution_providers([p.build()])
            }
            #[cfg(windows)]
            EpKind::DirectMl => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
            ]),
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => builder.with_execution_providers([
                // `num_streams=1` disables per-stream buffer duplication
                // (the bundle sits at ~16 GB RSS on an i7-6700 either
                // way; measured 2026-10). Dynamic shapes cost nothing
                // per step, so one bundle serves 512² tiles and the
                // 512×768 crop. (No `with_cache_dir` — see engine.rs
                // for the SD UNet empirical retest.)
                {
                    let mut p = ort::ep::OpenVINO::default()
                        .with_num_streams(1)
                        .with_dynamic_shapes(true);
                    if let Some(dev) = &session_key.ov_device {
                        p = p.with_device_type(dev);
                    }
                    p.build()
                },
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
                record_failure(&format!("{e}"));
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
                record_failure(&e);
            }
        }
    }

    if !sd_spec(id).cpu_output_correct {
        tracing::warn!(?id, part = %key,
            "SD bundle build refused: no compatible GPU EP, CPU produces wrong output");
        if session_key.cpu_only {
            return Err(format!(
                "The Stable Diffusion eraser needs a GPU provider, and Processor \
                 is set to CPU only (Settings). Set it to Auto, or {LIGHTER_ERASER_HINT}.",
            ));
        }
        return Err(format!(
            "SD inpaint requires GPU acceleration. No compatible GPU \
             execution provider is available; SD on CPU produces \
             incorrect output. Install OpenVINO Runtime (Settings → \
             Hardware) or {LIGHTER_ERASER_HINT}.",
        ));
    }
    crate::cache::gc_stale_for_model(id, "CPU");
    let builder = sd_base_builder(session_key.mem, id.sd_spec().is_some_and(|s| s.lean_sessions))
        .map_err(|e| format!("SD {key}: builder init: {e}"))?;
    let (mut builder, load_path) = sd_apply_path_cache(builder, path.as_path(), id, "CPU", key);
    let started = Instant::now();
    let session = builder
        .commit_from_file(&load_path)
        .map_err(|e| format!("SD {key}: load from {}: {e}", load_path.display()))?;
    tracing::info!(?id, part = %key, ep = "CPU", elapsed_ms = started.elapsed().as_millis() as u64, "SD: session committed");
    Ok((session, "CPU".to_string()))
}

/// `lean`: the model's sessions skip ORT's memory-pattern planning unless
/// the plan says otherwise.
fn sd_base_builder(mem: SdMemPlan, lean: bool) -> Result<ort::session::builder::SessionBuilder, String> {
    let mut builder = crate::ort_runtime::session_builder()
        .map_err(|e| format!("SD: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("SD: optimization level: {e}"))?;
    if !mem.memory_pattern.unwrap_or(!lean) {
        builder = builder.with_memory_pattern(false).map_err(|e| format!("SD: memory pattern: {e}"))?;
    }
    if mem.device_initializers {
        builder = builder.with_device_allocated_initializers().map_err(|e| format!("SD: initializers: {e}"))?;
    }
    let threads = if mem.intra_threads > 0 { mem.intra_threads as usize } else { num_cpus::get().max(1) };
    builder
        // Both settings act on ORT's own CPU EP, the only one RORem runs
        // on; SD 1.5 runs on a GPU EP or OpenVINO, which keep their own.
        // Prepacking keeps a second, repacked copy of the weights: RORem
        // peaked at 15.2 GB with it, 12.9 GB without, loaded in 45 s
        // instead of 96, and stepped as fast.
        .with_prepacking(false)
        .map_err(|e| format!("SD: prepacking: {e}"))?
        // ORT's default is the physical cores; the UNet's fp16 matmuls
        // also use the hyper-threads (RORem: 26 s a step on 4 of the
        // i7-6700's threads, 22.5 s on 8).
        .with_intra_threads(threads)
        .map_err(|e| format!("SD: threads: {e}"))
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
    // An optimized copy elsewhere would not find the weights' data files.
    if prunr_models::keeps_weights_apart(model_id) {
        return (builder, Cow::Borrowed(source_path));
    }
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
    smoke_test_vae(s, label, (1, 3, SD_TILE as usize, SD_TILE as usize))
}

fn smoke_test_vae_decoder(s: &mut Session, label: &str) -> Result<(), String> {
    smoke_test_vae(s, label, (1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize))
}

fn smoke_test_vae(s: &mut Session, label: &str, shape: (usize, usize, usize, usize)) -> Result<(), String> {
    let t = tensor_in(Array4::<f32>::zeros(shape), input_is_f32(s, 0))
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    let inputs = s.inputs();
    let name = inputs.first().map(|i| i.name().to_string())
        .ok_or_else(|| format!("{label}: no inputs"))?;
    s.run(inputs![name.as_str() => &t])
        .map_err(|e| format!("{label}: smoke run: {e}"))?;
    Ok(())
}

/// Whether input `idx` of `s` takes f32 (else f16).
fn input_is_f32(s: &Session, idx: usize) -> bool {
    s.inputs().get(idx).and_then(|i| i.dtype().tensor_type()) == Some(ort::value::TensorElementType::Float32)
}

/// `arr` as the tensor a session input takes: as it is, or narrowed to f16.
fn tensor_in(arr: Array4<f32>, f32_input: bool) -> ort::Result<ort::value::DynValue> {
    if f32_input {
        Ok(Tensor::from_array(arr)?.into_dyn())
    } else {
        Ok(Tensor::from_array(f32_to_f16_4d(&arr))?.into_dyn())
    }
}

/// The UNet's third input: SDXL's size conditioning, else a text embedding.
fn takes_size_ids(s: &Session) -> bool {
    s.inputs().get(2).is_some_and(|i| i.name() == "time_ids")
}

/// SDXL's `time_ids` for an `h × w` crop: original size, crop origin,
/// target size.
fn size_ids(h: u32, w: u32) -> ndarray::Array2<f16> {
    let (h, w) = (f16::from_f32(h as f32), f16::from_f32(w as f32));
    let zero = f16::from_f32(0.0);
    ndarray::Array2::from_shape_vec((1, 6), vec![h, w, zero, zero, h, w]).expect("6 values for (1, 6)")
}

fn smoke_test_unet(s: &mut Session, label: &str) -> Result<(), String> {
    let l = SD_LATENT_SIDE as usize;
    let lat = Array4::<f16>::zeros((1, 9, l, l));
    let ts = ndarray::Array1::<f16>::from_elem(1, f16::from_f32(0.0));
    let lat_t = Tensor::from_array(lat).map_err(|e| format!("{label}: smoke latent: {e}"))?;
    let ts_t = Tensor::from_array(ts).map_err(|e| format!("{label}: smoke ts: {e}"))?;
    let emb_t = if takes_size_ids(s) {
        Tensor::from_array(size_ids(SD_TILE, SD_TILE)).map(|t| t.into_dyn())
    } else {
        Tensor::from_array(Array3::<f16>::zeros((1, CLIP_SEQ_LEN, 768))).map(|t| t.into_dyn())
    }.map_err(|e| format!("{label}: smoke conditioning: {e}"))?;
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

/// The prompt's CLIP embedding in f16, from the bundle's cache when the
/// prompt was encoded before.
fn text_embedding_f16(bundle: &SdSession, prompt: &str) -> Result<Arc<Array3<f16>>, CoreError> {
    let hit = bundle.text_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter().find(|(p, _)| p == prompt).map(|(_, e)| Arc::clone(e));
    if let Some(e) = hit {
        return Ok(e);
    }
    let emb = Arc::new(f32_to_f16_3d(&encode_text(bundle, prompt)?));
    let mut cache = bundle.text_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    // Cond and uncond encode in parallel; the same prompt twice keeps one.
    if let Some((_, e)) = cache.iter().find(|(p, _)| p == prompt) {
        return Ok(Arc::clone(e));
    }
    if cache.len() >= TEXT_CACHE_ENTRIES {
        cache.remove(0);
    }
    cache.push((prompt.to_string(), Arc::clone(&emb)));
    Ok(emb)
}

fn encode_text(bundle: &SdSession, prompt: &str) -> Result<Array3<f32>, CoreError> {
    let tokens = clip_tokenize(prompt);
    let t = Tensor::from_array(tokens)
        .map_err(|e| CoreError::Inference(format!("SD text encoder: input tensor: {e}")))?;
    let encoder = bundle.text_encoder.as_ref()
        .ok_or_else(|| CoreError::Inference("SD: this bundle has no text encoder".into()))?;
    let mut session = encoder.session.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![encoder.input.as_str() => &t])
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
    let f32_input = matches!(backend, VaeBackend::Standard(b) if b.vae_f32);
    let t = tensor_in(input, f32_input)
        .map_err(|e| CoreError::Inference(format!("vae encoder: input tensor: {e}")))?;
    match backend {
        VaeBackend::Standard(bundle) => {
            let mut session = bundle.vae_encoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![bundle.vae_encoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("SD vae encoder: run: {e}")))?;
            let mut latent = extract_4d(&outputs[0], "vae encoder")?;
            latent *= bundle.vae_scale;
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
            let t = tensor_in(latent / bundle.vae_scale, bundle.vae_f32)
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

/// What a UNet step is conditioned on: a prompt's CLIP embedding (SD
/// 1.5), or SDXL's size conditioning for a crop of `(h, w)` when the
/// prompt is baked in (RORem).
enum Cond<'a> {
    Text(&'a Array3<f16>),
    Size(u32, u32),
}

fn unet_step(
    bundle: &SdSession,
    latent_9ch_f16: Array4<f16>,
    t: i64,
    cond: Cond<'_>,
) -> Result<Array4<f32>, CoreError> {
    // Timestep is integer-valued at the diffusers level but flows through
    // sinusoidal embeddings inside the UNet, so the f16 cast is precision-
    // safe (timesteps fit in the f16 mantissa for SD's 1000-step grid).
    let timestep = ndarray::Array1::<f16>::from_elem(1, f16::from_f32(t as f32));
    let lat_t = Tensor::from_array(latent_9ch_f16)
        .map_err(|e| CoreError::Inference(format!("SD unet: latent tensor: {e}")))?;
    let ts_t = Tensor::from_array(timestep)
        .map_err(|e| CoreError::Inference(format!("SD unet: timestep tensor: {e}")))?;
    let cond_t = match cond {
        Cond::Text(emb) => Tensor::from_array(emb.clone()).map(|t| t.into_dyn()),
        Cond::Size(h, w) => Tensor::from_array(size_ids(h, w)).map(|t| t.into_dyn()),
    }.map_err(|e| CoreError::Inference(format!("SD unet: conditioning tensor: {e}")))?;
    let mut session = bundle.unet.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![
        bundle.unet_inputs[0].as_str() => &lat_t,
        bundle.unet_inputs[1].as_str() => &ts_t,
        bundle.unet_inputs[2].as_str() => &cond_t,
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
/// masked pixels (mask > 127) are filled with the byte-128 mid-gray
/// value (≈ 0.0039 in [-1, 1]). Binary cliff is intentional — the
/// SD-1.5 inpaint UNet was trained on binary-zeroed masked-image
/// latents; partial-alpha blends toward grey at the boundary band
/// produce a visible grey ghost at inference. `None` mask encodes
/// the full image without any masking.
fn image_to_minus1_plus1_inner(image: &RgbaImage, mask: Option<&GrayImage>) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let mut a = Array4::<f32>::zeros((1, 3, h_us, w_us));
    // invariant: freshly allocated, standard layout — the slice view is valid.
    let buf = a.as_slice_mut().unwrap();
    let plane = w_us * h_us;
    let raw = image.as_raw();
    let m = mask.map(|m| m.as_raw());
    // Match the byte-128 path's bit pattern: same f32 division as the
    // unmasked branch, just on a constant input. Hardcoding `0.0`
    // would diverge by ~0.00392.
    let v_fill = (128.0_f32 / 127.5) - 1.0;
    for y in 0..h_us {
        let src_row = y * w_us * 4;
        let dst_row = y * w_us;
        let mask_row = y * w_us;
        for x in 0..w_us {
            let dst = dst_row + x;
            let src = src_row + x * 4;
            let r = (raw[src]     as f32 / 127.5) - 1.0;
            let g = (raw[src + 1] as f32 / 127.5) - 1.0;
            let b = (raw[src + 2] as f32 / 127.5) - 1.0;
            if let Some(m) = m {
                // Binary cliff: mask > 127 → fully grey (inpaint
                // region), else original. The SD-1.5 inpaint UNet was
                // trained on binary-zeroed masked-image latents — a
                // partial-alpha blend would put grey-tinted pixels at
                // the boundary, which the model treats literally and
                // outputs as a grey ghost. Composite-time softness is
                // handled by inpaint_feather, not here.
                if m[mask_row + x] > 127 {
                    buf[dst]             = v_fill;
                    buf[plane + dst]     = v_fill;
                    buf[plane * 2 + dst] = v_fill;
                } else {
                    buf[dst]             = r;
                    buf[plane + dst]     = g;
                    buf[plane * 2 + dst] = b;
                }
            } else {
                buf[dst]             = r;
                buf[plane + dst]     = g;
                buf[plane * 2 + dst] = b;
            }
        }
    }
    a
}

fn image_to_minus1_plus1_masked(image: &RgbaImage, mask: &GrayImage) -> Array4<f32> {
    debug_assert_eq!(image.dimensions(), mask.dimensions());
    image_to_minus1_plus1_inner(image, Some(mask))
}

/// The unmasked conversion, so a test can round-trip the pixel mapping.
#[cfg(test)]
fn image_to_minus1_plus1(image: &RgbaImage) -> Array4<f32> {
    image_to_minus1_plus1_inner(image, None)
}

fn minus1_plus1_to_image(arr: &Array4<f32>) -> RgbaImage {
    let (_, _, h, w) = arr.dim();
    let plane = w * h;
    let buf = arr.as_slice().unwrap_or(&[]);
    if buf.len() < plane * 3 {
        tracing::warn!(buf_len = buf.len(), "SD vae decode: buffer smaller than tile");
        return RgbaImage::new(w as u32, h as u32);
    }
    let mut out = RgbaImage::new(w as u32, h as u32);
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

/// Mask → latent space (1, 1, h/8, w/8) f32 in [0, 1].
/// Average-pool 8×8 → 1 (equivalent to `F.interpolate(mode="area")`
/// in PyTorch, which is what Diffusers uses for mask preprocessing).
///
/// Why average-pool, not nearest-neighbour center-sample: even with
/// a binary mask, cells that straddle the mask boundary average to
/// ~0.5, preserving sub-block boundary precision in the latent mask.
/// Center-sampling throws away that information, producing a jaggy
/// latent boundary that doesn't faithfully represent the pixel mask.
fn mask_to_latent(mask: &GrayImage) -> Array4<f32> {
    let (w, h) = mask.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let (lw, lh) = (w_us.div_ceil(8), h_us.div_ceil(8));
    let mut a = Array4::<f32>::zeros((1, 1, lh, lw));
    // invariant: freshly allocated, standard layout — the slice view is valid.
    let buf = a.as_slice_mut().unwrap();
    let raw = mask.as_raw();
    let inv = 1.0_f32 / (64.0 * 255.0);
    for ly in 0..lh {
        for lx in 0..lw {
            let mut sum = 0_u32;
            for dy in 0..8 {
                let sy = ly * 8 + dy;
                if sy >= h_us { continue; }
                let row = sy * w_us;
                for dx in 0..8 {
                    let sx = lx * 8 + dx;
                    if sx >= w_us { continue; }
                    sum += raw[row + sx] as u32;
                }
            }
            buf[ly * lw + lx] = sum as f32 * inv;
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

/// Initial gaussian noise: LCM and DPM++ hold the latent in α-space, so
/// it starts as plain N(0, 1).
fn sample_initial_noise(rng: &mut ChaCha8Rng, lw: usize, lh: usize) -> Array4<f32> {
    let dist = StandardNormal;
    let n = 4 * lw * lh;
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f32 = dist.sample(rng);
        buf.push(v);
    }
    Array4::from_shape_vec((1, 4, lh, lw), buf)
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

/// Scheduler step dispatched from `run_one_tile`, writing the new latent
/// state into `out`.
// `step_idx` is only meaningful to the DPM++ multistep arm, but the
// dispatch is shared. Threading it through a `StepCtx` struct would
// force the LCM arm to carry a field it never reads.
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
    }
}

/// If the cropped input is smaller than SD_TILE on either axis (image
/// edge case), pad to 512×512 with zero-fill (top-left aligned). The
/// smart-crop dispatcher guarantees max dim is SD_TILE.
///
/// Returns `Cow::Borrowed` on the fast path (already-tile-sized input,
/// the common case for small strokes) — saves the ~1 MB image clone +
/// ~256 KB mask clone the previous version always did.
/// `image` zero-padded to `w × h` (borrowed when it already is).
fn pad_to<P>(image: &image::ImageBuffer<P, Vec<P::Subpixel>>, w: u32, h: u32) -> std::borrow::Cow<'_, image::ImageBuffer<P, Vec<P::Subpixel>>>
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
{
    if image.dimensions() == (w, h) {
        return std::borrow::Cow::Borrowed(image);
    }
    let mut out = image::ImageBuffer::new(w, h);
    image::imageops::overlay(&mut out, image, 0, 0);
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

/// The run-time RAM gate for a stroke. Refuses when the model's base
/// need is not free; when only the tall crop's extra is missing, the
/// plan falls back to tiles instead. Unknown free RAM passes.
pub(crate) fn check_ram_for(id: prunr_models::ModelId, tuning: &mut SdTuning) -> Result<(), String> {
    let Some(free_mb) = available_ram_mb() else { return Ok(()) };
    fit_ram(id, tuning, free_mb, resident(id))
}

/// `check_ram_for` with the RAM facts passed in.
fn fit_ram(id: prunr_models::ModelId, tuning: &mut SdTuning, free_mb: u64, resident: bool) -> Result<(), String> {
    let Some(desc) = prunr_models::descriptor(id) else { return Ok(()) };
    if tuning.tall_crop && free_mb < ram_need_mb(id, tuning.margin_mb, resident, true) {
        tracing::info!(free_mb, "SD: no room for the tall crop; tiling instead");
        tuning.tall_crop = false;
    }
    let need_mb = ram_need_mb(id, tuning.margin_mb, resident, false);
    if free_mb >= need_mb {
        return Ok(());
    }
    tracing::info!(id = ?desc.id, free_mb, need_mb, "SD: refused to load");
    Err(format!(
        "This eraser needs {:.1} GB of free memory to load, {:.1} GB of it kept spare \
         for other apps; {:.1} GB is free. Close other apps, or {LIGHTER_ERASER_HINT}.",
        need_mb as f64 / 1024.0,
        tuning.margin_mb as f64 / 1024.0,
        free_mb as f64 / 1024.0,
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

// ── SD-1.5 noise schedule and samplers ──────────────────────────────────

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
/// sigma is the caller's responsibility — DPM++ appends `sigma_min`.
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
            .map(|&sigma| sigma_to_t(sigma, log_sigmas))
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
    #[default]
    Lcm,
    DpmPp2MKarras,
}

/// Runtime-selected scheduler for SD denoise. Constructed worker-side
/// from `SchedulerKind` carried on `SdInpaintRequest`.
pub enum Scheduler {
    Lcm(LcmScheduler),
    DpmPp2M(DpmPp2MScheduler),
}

impl Scheduler {
    pub fn timesteps(&self) -> &[i64] {
        match self {
            Scheduler::Lcm(s) => s.timesteps(),
            Scheduler::DpmPp2M(s) => s.timesteps(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

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
    /// to low). A regression here
    /// would silently invert the noise schedule.
    #[test]
    fn lcm_timesteps_descend_strictly() {
        let s = LcmScheduler::new_sd15(8);
        for w in s.timesteps().windows(2) {
            assert!(w[0] > w[1], "timesteps must descend: {} → {}", w[0], w[1]);
        }
    }


    /// Final-step LCM reduces to `c_out * pred_x0 + c_skip * sample`
    /// — no noise injection, scaled by the consistency-function
    /// coefficients. This test
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

    /// DPM++ shares the SD-1.5 alpha schedule with LCM.
    /// Drift in the shared `compute_alphas_cumprod_sd15` would
    /// silently desync the schedulers; this test pins parity at
    /// representative timesteps.
    #[test]
    fn dpmpp2m_uses_shared_sd15_alpha_schedule() {
        // The scheduler doesn't expose alphas_cumprod, but we can
        // derive sigma at a known timestep from the public sigma
        // schedule: the first sigma equals √((1-α̅_999)/α̅_999).
        let s = DpmPp2MScheduler::new_sd15(25);
        // The first sigma in the Karras schedule is sigma_max,
        // derived from α̅[num_train-1].
        let alpha_max = compute_alphas_cumprod_sd15()[999];
        let expected_sigma_max = ((1.0 - alpha_max) / alpha_max).sqrt();
        let actual_sigma_max = s.sigmas[0];
        assert!((actual_sigma_max - expected_sigma_max).abs() < 1e-3,
            "first sigma must match √((1-α̅_999)/α̅_999): expected {expected_sigma_max}, got {actual_sigma_max}");
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
    fn non_square_crops_round_trip_through_the_conversions() {
        let (w, h) = (128u32, 192u32);
        let img = RgbaImage::from_fn(w, h, |x, y| Rgba([(x * 2) as u8, (y) as u8, 7, 255]));
        let arr = image_to_minus1_plus1(&img);
        assert_eq!(arr.dim(), (1, 3, h as usize, w as usize));
        let back = minus1_plus1_to_image(&arr);
        assert_eq!(back.dimensions(), (w, h));
        // The u8 → [-1, 1] → u8 trip truncates, so odd values land one low.
        let off = back.as_raw().iter().zip(img.as_raw()).filter(|(a, b)| a.abs_diff(**b) > 1).count();
        assert_eq!(off, 0);
        let mut mask = GrayImage::new(w, h);
        for y in 0..8 { for x in 0..8 { mask.put_pixel(x, y + 184, Luma([255])); } }
        let lat = mask_to_latent(&mask);
        assert_eq!(lat.dim(), (1, 1, 24, 16));
        assert!((lat[(0, 0, 23, 0)] - 1.0).abs() < 1e-6);
        assert_eq!(lat[(0, 0, 0, 0)], 0.0);
        let noise = sample_initial_noise(&mut ChaCha8Rng::seed_from_u64(1), 16, 24);
        assert_eq!(noise.dim(), (1, 4, 24, 16));
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
        let a = sample_initial_noise(&mut r1, 64, 64);
        let b = sample_initial_noise(&mut r2, 64, 64);
        assert_eq!(a, b, "same seed must give identical noise");
        let c = sample_initial_noise(&mut r3, 64, 64);
        assert_ne!(a, c, "different seed must give different noise");
    }

    #[test]
    fn mask_components_returns_empty_for_empty_mask() {
        let m = GrayImage::new(64, 64);
        assert!(mask_components(&m).is_empty());
    }

    /// TAESD's graph is fixed at 512², so a run with it never plans the
    /// 768 tall crop (it failed and left the region as it was).
    #[test]
    fn the_fast_decoder_keeps_crops_at_512() {
        let tuning = SdTuning { tall_crop: true, ..SdTuning::default() };
        let req = SdInpaintRequest { tuning, use_taesd: true, ..Default::default() };
        assert!(!req.tall_crop());
        assert_eq!(fit(&req).long, SD_TILE - 64);
        let req = SdInpaintRequest { use_taesd: false, ..req };
        assert!(req.tall_crop());
        assert_eq!(fit(&req).long, SD_CROP_MAX - 64);
    }

    /// Slats a few pixels apart are one region; a mark across the image
    /// stays its own.
    #[test]
    fn nearby_pieces_are_one_region() {
        let mut m = GrayImage::new(1000, 300);
        for slat in 0..20 {
            for y in 50..250 {
                for x in 0..12 {
                    m.put_pixel(100 + slat * 20 + x, y, Luma([255]));
                }
            }
        }
        for y in 140..160 {
            for x in 900..920 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        assert_eq!(mask_components(&m).len(), 21);
        let regions = mask_regions(&m);
        assert_eq!(regions.len(), 2, "{regions:?}");
        assert!(regions.iter().any(|r| r.x_min == 100 && r.x_max == 100 + 19 * 20 + 11));
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
    fn the_gate_takes_the_tall_crop_only_with_the_extra_ram() {
        let id = prunr_models::ModelId::SdV15InpaintFp16;
        let m = SD_DEFAULT_MARGIN_MB;
        let base = ram_need_mb(id, m, false, false);
        assert!(base > m, "the working set counts when the bundle is not resident");
        assert_eq!(ram_need_mb(id, m, true, false), m, "a resident bundle needs only the headroom");
        assert_eq!(ram_need_mb(id, m, true, true), m + SD_TALL_CROP_EXTRA_MB);
        let fit = |free: u64, resident: bool| {
            let mut t = plan_tuning(false, m);
            fit_ram(id, &mut t, free, resident).map(|()| t.tall_crop)
        };
        assert_eq!(fit(base + SD_TALL_CROP_EXTRA_MB, false), Ok(true));
        assert_eq!(fit(base + SD_TALL_CROP_EXTRA_MB - 1, false), Ok(false), "short of the extra: tiles");
        assert!(fit(base - 1, false).is_err(), "short of the base: refused");
        // A kept bundle lowers the free RAM by its own size; the gate in
        // the process holding it must not count it again.
        assert_eq!(fit(m + SD_TALL_CROP_EXTRA_MB, true), Ok(true), "resident: the extra alone decides");
        assert_eq!(fit(m, true), Ok(false));
        let t = plan_tuning(true, m);
        assert!(t.keep_loaded && t.tall_crop && t.ov_device.is_none() && t.margin_mb == m);
        assert_eq!(t.session_key(), SdSessionKey::default());
        assert!(t.session_key().records_ep_failures());
        assert!(!SdTuning { ov_device: Some("GPU".into()), ..Default::default() }.session_key().records_ep_failures());
    }

    #[test]
    fn tall_regions_get_one_crop_up_to_the_max_side() {
        // 328×607: one 512-wide, 768-tall crop when tall crops are on.
        let bbox = MaskBbox { x_min: 1564, y_min: 470, x_max: 1891, y_max: 1076 };
        assert_eq!(compute_sd_crop(&bbox, 4096, 4096, SD_TILE), None, "tiles without tall crops");
        let (x, y, w, h) = compute_sd_crop(&bbox, 4096, 4096, SD_CROP_MAX).unwrap();
        assert_eq!((w, h), (512, 768));
        assert_eq!((x, y), (1727 - 256, 773 - 384));
        let wide = MaskBbox { x_min: 0, y_min: 0, x_max: 800, y_max: 100 };
        assert_eq!(compute_sd_crop(&wide, 4096, 4096, SD_CROP_MAX), None, "beyond the max side");
        let square = MaskBbox { x_min: 0, y_min: 0, x_max: 600, y_max: 600 };
        assert_eq!(compute_sd_crop(&square, 4096, 4096, SD_CROP_MAX), None, "one long side only");
        assert!(matches!(plan_region(&square, 4096, 4096, SD_CROP_MAX), RegionPlan::Tiles(t) if t.len() == 4));
        assert_eq!(plan_region(&bbox, 4096, 4096, SD_CROP_MAX).tile_count(), 1);
        assert_eq!(crop_side(512, SD_CROP_MAX), Some(512));
        assert_eq!(crop_side(513, SD_CROP_MAX), Some(768));
        assert_eq!(padded_dims(700, 300), (704, 512));
    }

    #[test]
    fn compute_sd_crop_centres_on_bbox_in_large_image() {
        let bbox = MaskBbox { x_min: 1000, y_min: 1000, x_max: 1100, y_max: 1100 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 4096, 4096, SD_TILE).unwrap();
        assert_eq!((w, h), (SD_TILE, SD_TILE));
        // Centre of bbox is (1050, 1050); crop top-left should land at
        // 1050 - 256 = 794.
        assert_eq!(x, 1050 - SD_TILE / 2);
        assert_eq!(y, 1050 - SD_TILE / 2);
    }

    #[test]
    fn compute_sd_crop_clamps_to_right_edge() {
        let bbox = MaskBbox { x_min: 3900, y_min: 100, x_max: 3990, y_max: 200 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 4096, 4096, SD_TILE).unwrap();
        // Crop must fit inside image; right-edge crop starts at img_w - SD_TILE.
        assert_eq!(x + w, 4096, "crop must end at right edge");
        assert_eq!((w, h), (SD_TILE, SD_TILE));
        // Centre falls in the painted region.
        assert!(y < bbox.y_min);
    }

    #[test]
    fn compute_sd_crop_clamps_to_top_left_corner() {
        let bbox = MaskBbox { x_min: 5, y_min: 10, x_max: 50, y_max: 60 };
        let (x, y, _, _) = compute_sd_crop(&bbox, 4096, 4096, SD_TILE).unwrap();
        // Centre is (27, 35); centre - 256 underflows → saturates at 0.
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn compute_sd_crop_shrinks_to_image_for_small_inputs() {
        let bbox = MaskBbox { x_min: 10, y_min: 10, x_max: 100, y_max: 100 };
        let (x, y, w, h) = compute_sd_crop(&bbox, 200, 300, SD_TILE).unwrap();
        // Image smaller than SD_TILE on both axes → crop is the whole
        // image.
        assert_eq!((x, y, w, h), (0, 0, 200, 300));
    }

    #[test]
    fn sample_initial_noise_has_correct_shape() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let n = sample_initial_noise(&mut rng, 64, 64);
        assert_eq!(n.shape(), &[1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize]);
    }

    /// A "now" far enough from the clock's origin to subtract minutes from.
    /// On Windows `Instant` counts from boot, and a fresh CI runner can be
    /// seconds old — `Instant::now() - 600 s` then panics with an overflow.
    fn sweep_now() -> Instant {
        Instant::now() + Duration::from_secs(3600)
    }

    /// The Settings memory note and the run-time gate agree: refused
    /// exactly where `fit_ram` refuses, tiles exactly where it drops the
    /// tall crop.
    #[test]
    fn sd_ram_fit_matches_the_run_time_gate() {
        let id = prunr_models::ModelId::SdV15InpaintFp16;
        let margin = 2048;
        let base = ram_need_mb(id, margin, false, false);
        for free in [base - 1, base, base + SD_TALL_CROP_EXTRA_MB - 1, base + SD_TALL_CROP_EXTRA_MB] {
            let mut t = plan_tuning(false, margin);
            let gate = fit_ram(id, &mut t, free, false);
            let expected = match (gate.is_ok(), t.tall_crop) {
                (false, _) => SdRamFit::Refused,
                (true, false) => SdRamFit::TilesOnly,
                (true, true) => SdRamFit::Full,
            };
            assert_eq!(sd_ram_fit(free, margin), expected, "free {free} MB");
        }
    }

    /// The Fast bundle peaks at about 14 GB while it loads: with 12 GB
    /// free and the default headroom the erase is refused before it
    /// starts, not killed by the memory watchdog halfway through.
    #[test]
    fn an_erase_that_cannot_fit_its_load_is_refused_up_front() {
        for id in [prunr_models::ModelId::SdV15LcmInpaintFp16, prunr_models::ModelId::SdV15InpaintFp16] {
            let mut t = plan_tuning(false, SD_DEFAULT_MARGIN_MB);
            assert!(fit_ram(id, &mut t, 12_147, false).is_err(), "{id:?} refused with 12 GB free");
            let mut t = plan_tuning(false, SD_DEFAULT_MARGIN_MB);
            assert!(fit_ram(id, &mut t, 16_500, false).is_ok(), "{id:?} runs with 16.5 GB free");
        }
        assert_eq!(sd_ram_fit(12_147, SD_DEFAULT_MARGIN_MB), SdRamFit::Refused);
    }

    /// "CPU only" reaches the SD parts through the bundle key, so a
    /// worker process needs no extra state, and a bundle built under the
    /// other policy is never reused.
    #[test]
    fn a_cpu_only_bundle_tries_no_gpu_provider_and_has_its_own_key() {
        let auto = plan_tuning(false, 0).session_key();
        let cpu = SdTuning { cpu_only: true, ..plan_tuning(false, 0) }.session_key();
        assert!(cpu.gpu_eps().is_empty());
        assert_ne!(auto, cpu);
    }

    /// One SD bundle is resident at a time: "keep loaded" pins the
    /// eraser, not every bundle it has used. Switching Fast to Quality
    /// with both pinned held two 2 GB bundles and swapped the machine.
    #[test]
    fn admitting_a_bundle_drops_the_other_ones_pinned_or_not() {
        let now = Instant::now();
        let mut cache: HashMap<&str, CacheEntry<u8>> = HashMap::new();
        admit(&mut cache, "fast", || 1, now, true);
        admit(&mut cache, "quality", || 2, now, true);
        assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["quality"]);
        assert!(cache["quality"].pinned);
        admit(&mut cache, "quality", || 9, now, false);
        assert_eq!(cache["quality"].value, 2, "the resident bundle is reused, not rebuilt");
    }

    #[test]
    fn sweep_idle_keeps_pinned_entries() {
        let now = Instant::now() + Duration::from_secs(3600);
        let mut cache: HashMap<prunr_models::ModelId, CacheEntry<u8>> = HashMap::new();
        let mut pinned = CacheEntry::new(1, now - Duration::from_secs(600));
        pinned.pinned = true;
        cache.insert(prunr_models::ModelId::SdV15InpaintFp16, pinned);
        cache.insert(prunr_models::ModelId::LaMaFp32, CacheEntry::new(2, now - Duration::from_secs(600)));
        assert_eq!(sweep_idle(&mut cache, now, Duration::from_secs(300)), 1);
        assert!(cache.contains_key(&prunr_models::ModelId::SdV15InpaintFp16));
    }

    #[test]
    fn sweep_idle_drops_only_stale_entries() {
        use prunr_models::ModelId;
        let now = sweep_now();
        let idle = Duration::from_secs(300);
        let fresh = now - Duration::from_secs(60);
        let stale = now - Duration::from_secs(600);

        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry::new(Arc::new(()), stale));
        cache.insert(ModelId::LaMaFp32, CacheEntry::new(Arc::new(()), fresh));

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
        let now = sweep_now();
        let idle = Duration::from_secs(300);
        let payload = Arc::new(());
        let weak = Arc::downgrade(&payload);

        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry {
            value: payload,
            pinned: false,
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
        let now = sweep_now();
        let idle = Duration::from_secs(300);
        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        // Just-under-the-boundary entry must NOT evict.
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry {
            value: Arc::new(()),
            pinned: false,
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


    /// `image_to_minus1_plus1_masked` must use a binary cliff at byte
    /// 127, NOT a float-alpha blend. A blend toward mid-grey at
    /// partial alpha values produces visible grey-ghost artefacts at
    /// inference — the SD-1.5 inpaint UNet treats grey-tinted inputs
    /// literally. Both the masked-image latent and mask_to_latent
    /// receive the binary mask; composite-time softness goes through
    /// inpaint_feather (Gaussian blur on the final alpha).
    #[test]
    fn image_to_minus1_plus1_masked_uses_binary_cliff_not_float_blend() {
        let img = RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 255]));
        let mut mask = GrayImage::new(2, 1);
        // Boundary band: byte 127 → unmasked (original), byte 128 → masked (mid-grey).
        mask.put_pixel(0, 0, image::Luma([127]));
        mask.put_pixel(1, 0, image::Luma([128]));
        let arr = image_to_minus1_plus1_masked(&img, &mask);
        // Pixel 0: mask=127 (≤127) → original red = (255/127.5) - 1 = 1.0.
        let r0 = arr[(0, 0, 0, 0)];
        assert!((r0 - 1.0).abs() < 1e-5, "byte 127 must be original, got {r0}");
        // Pixel 1: mask=128 (>127) → mid-grey fill = (128/127.5) - 1 ≈ 0.00392.
        let r1 = arr[(0, 0, 0, 1)];
        let v_fill = (128.0_f32 / 127.5) - 1.0;
        assert!((r1 - v_fill).abs() < 1e-5, "byte 128 must be mid-grey {v_fill}, got {r1}");
        // CRITICAL: at byte 192 (75% alpha if blending) the output
        // must STILL be mid-grey (binary cliff), not a partial blend.
        let mut mask2 = GrayImage::new(1, 1);
        mask2.put_pixel(0, 0, image::Luma([192]));
        let arr2 = image_to_minus1_plus1_masked(
            &RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255])),
            &mask2,
        );
        let r2 = arr2[(0, 0, 0, 0)];
        assert!((r2 - v_fill).abs() < 1e-5,
            "byte 192 must be binary-cliff mid-grey {v_fill} (NOT float blend ≈ 0.25), got {r2}");
    }

    #[test]
    fn mask_to_latent_passes_through_gradient_values() {
        // Float pass-through contract: a uniform-128 mask must average-
        // pool to ~0.502 in every cell. Pins both the mid-gradient
        // value AND the average-pool semantic.
        let mask = GrayImage::from_pixel(SD_TILE, SD_TILE, image::Luma([128]));
        let latent = mask_to_latent(&mask);
        let expected = 128.0_f32 / 255.0;
        for v in latent.iter() {
            assert!(
                (v - expected).abs() < 0.01,
                "mask_to_latent: average-pool of uniform 128 mask expected ≈{expected}, got {v}"
            );
        }
    }

    /// Average-pool downsample contract: a half-binary mask (left half
    /// = 255, right half = 0) must produce latent cells ≈ 1.0 in the
    /// fully-masked half, ≈ 0.0 in the unmasked half, AND a single
    /// transition column at the midpoint reflecting the partial 8×8
    /// block coverage. Center-sample (the previous behaviour) would
    /// produce a hard 0/1 step with no transition cell — losing
    /// sub-block boundary precision in the latent mask.
    #[test]
    fn mask_to_latent_average_pools_at_boundary_not_center_samples() {
        // Half/half binary mask: pixels 0..255 = 255, pixels 256..511 = 0.
        // The 8×8 block at latent x=32 (source x=256..263) straddles
        // nothing — entirely in the right half. The transition is at
        // a clean 8-pixel boundary, so each block is fully one side
        // or the other — average-pool gives clean 1.0 / 0.0 cells.
        let mut mask = GrayImage::new(SD_TILE, SD_TILE);
        let s = SD_TILE as usize;
        for y in 0..s {
            for x in 0..s {
                let v = if x < s / 2 { 255 } else { 0 };
                mask.put_pixel(x as u32, y as u32, image::Luma([v]));
            }
        }
        let latent = mask_to_latent(&mask);
        let l = SD_LATENT_SIDE as usize;
        // Left half latent cells (lx = 0..32): ~1.0
        // Right half latent cells (lx = 32..64): ~0.0
        let left_cell = latent[(0, 0, 0, 0)];
        assert!((left_cell - 1.0).abs() < 1e-3, "left half should be 1.0, got {left_cell}");
        let right_cell = latent[(0, 0, 0, l - 1)];
        assert!(right_cell.abs() < 1e-3, "right half should be 0.0, got {right_cell}");

        // Critical case: shift the boundary by 4 pixels so it lands
        // mid-block. The block at lx=32 now spans source x=256..263,
        // where x=256..259 are in the left half (255) and x=260..263
        // are in the right half (0). Average-pool: 4/8 columns × 8
        // rows = 32 of 64 pixels at 255 → average = 0.5.
        let mut mask2 = GrayImage::new(SD_TILE, SD_TILE);
        for y in 0..s {
            for x in 0..s {
                let v = if x < s / 2 + 4 { 255 } else { 0 };
                mask2.put_pixel(x as u32, y as u32, image::Luma([v]));
            }
        }
        let latent2 = mask_to_latent(&mask2);
        // The block at lx=32 (source x=256..263) is half-and-half →
        // average-pool ≈ 0.5. Center-sample (old behaviour) would
        // pick source[ly*8+4][lx*8+4] = source[..][260] = 0, missing
        // the gradient entirely.
        let mid_cell = latent2[(0, 0, 0, 32)];
        assert!(
            (mid_cell - 0.5).abs() < 1e-3,
            "boundary cell straddling the 4-pixel shift should average to ~0.5 \
             (proves average-pool, not center-sample), got {mid_cell}"
        );
    }
}

#[cfg(test)]
mod mem_plan_tests {
    use super::{SdMemPlan, SdTuning};

    /// A request saved before the memory knobs loads with ORT's defaults,
    /// and a different plan is a different session bundle.
    #[test]
    fn the_memory_plan_defaults_to_orts_and_keys_the_bundle() {
        let old = r#"{"keep_loaded":false,"ov_device":null,"tall_crop":true,"margin_mb":2000,"cpu_only":false}"#;
        let tuning: SdTuning = serde_json::from_str(old).expect("loads without mem");
        assert_eq!(tuning.mem, SdMemPlan::default());
        let lean = SdTuning { mem: SdMemPlan { memory_pattern: Some(false), device_initializers: true, intra_threads: 4 }, ..tuning.clone() };
        assert_ne!(lean.session_key(), tuning.session_key(), "the plan is part of what a bundle was built with");
        assert_eq!(SdTuning { mem: tuning.mem, ..lean.clone() }.session_key(), tuning.session_key());
    }

    /// The plan leaves the memory pattern to the model: RORem's sessions
    /// run without it (where its extra 6 GB went), SD 1.5's keep ORT's
    /// default, and the bench flag overrides either way.
    #[test]
    fn the_memory_pattern_is_the_models_own_unless_overridden() {
        use prunr_models::ModelId;
        assert_eq!(super::plan_tuning(false, 0).mem, SdMemPlan::default());
        let lean = |id: ModelId| id.sd_spec().expect("an SD model").lean_sessions;
        assert!(lean(ModelId::RoremFp16) && !lean(ModelId::SdV15InpaintFp16) && !lean(ModelId::SdV15LcmInpaintFp16));
        let plan = |memory_pattern| SdMemPlan { memory_pattern, ..SdMemPlan::default() };
        assert!(!plan(None).memory_pattern.unwrap_or(!lean(ModelId::RoremFp16)), "RORem: off");
        assert!(plan(None).memory_pattern.unwrap_or(!lean(ModelId::SdV15InpaintFp16)), "SD 1.5: ORT's");
        assert!(plan(Some(true)).memory_pattern.unwrap_or(!lean(ModelId::RoremFp16)), "the flag turns it back on");
    }
}
