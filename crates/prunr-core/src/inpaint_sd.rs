//! Stable Diffusion 1.5 Inpainting pipeline.
//!
//! Per-stroke flow: CLIP-tokenize prompt → text encoder; VAE-encode the
//! masked source; downsample mask to latent space; sample gaussian noise;
//! denoise via LCM or DPM++ 2M Karras (with optional classifier-free
//! guidance); VAE-decode
//! final latent; composite back through mask.
//!
//! Dispatch wraps the per-tile pipeline:
//! - Mask pieces within `REGION_GAP` of each other group into one
//!   region (`mask_regions`); distant strokes stay separate so they
//!   don't widen the bbox.
//! - A region that fits one crop (512², or up to `SD_CROP_MAX` on its
//!   long side with the tall crop) hits the smart-crop fast path
//!   (`compute_sd_crop`).
//! - Larger regions split into overlapping 512×512 tiles with
//!   linear-alpha seam blending (`tile_bbox` + `blend_tile`).
//!
//! Safety guards on CPU-class hardware:
//! - Bundle load refuses below the model's `needs` in free RAM.
//!   ORT graph optimization + UNet activations together push 6-10 GB
//!   transient on this codepath; the floor protects against swap thrash.
//! - Idle bundle release (`crate::IDLE_RELEASE`) drops the cached 4
//!   ORT sessions after no use, reclaiming ~4-6 GB.
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

use crate::engine::{EpKind, EpSite, Miss, SessionPick, Weights};
use crate::sysmem::process_rss_mb;
use crate::types::CoreError;

/// SD 1.5 native input side length.
pub const SD_TILE: u32 = 512;
/// Pixels per latent cell on each axis: the VAE downsamples 8×.
const VAE_FACTOR: u32 = 8;
/// Pixels in one latent cell.
const VAE_CELL_PIXELS: f32 = (VAE_FACTOR * VAE_FACTOR) as f32;
/// Latent side of a 512² tile; the pipeline derives its own from the
/// crop, this only shapes the session smoke tests and TAESD's input.
const SD_LATENT_SIDE: u32 = SD_TILE / VAE_FACTOR;
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

/// CLIP-ViT-L/14 token sequence length used by SD 1.5.
const CLIP_SEQ_LEN: usize = 77;
/// CLIP-ViT-L/14 BOS / EOS / pad ids — same value for EOS and pad in
/// the SD 1.5 tokenizer convention.
const CLIP_BOS: i64 = 49406;
const CLIP_EOS: i64 = 49407;

/// Inputs to one SD inpaint call; the defaults are an empty-prompt
/// unconditional inpaint. Serde-derived so it crosses the worker IPC
/// without a wire mirror.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SdInpaintRequest {
    /// Empty string ⇒ unconditional generation.
    pub prompt: String,
    /// Empty ⇒ no negative conditioning.
    pub negative_prompt: String,
    /// 20 is the SD 1.5 sweet spot at default scheduler.
    pub num_inference_steps: u32,
    /// ≤ 1.0 ⇒ no classifier-free guidance; 7.5 is the typical
    /// text-prompt setting.
    pub guidance_scale: f32,
    /// `None` ⇒ random seed each call.
    pub seed: Option<u64>,
    /// When true, swap the SD bundle's standard VAE legs for TAESD — a
    /// distilled ~1M-param VAE pair (~3× faster decode, slight quality
    /// cost). Falls back to the standard VAE when TAESD isn't installed.
    pub use_taesd: bool,
    #[serde(default)]
    pub scheduler: SchedulerKind,
    /// How the pipeline runs for this stroke; see `plan_tuning`.
    #[serde(default)]
    pub tuning: SdTuning,
    /// How large a region is filled; Auto shrinks one that would need
    /// tiles to fit one crop.
    #[serde(default)]
    pub fill_scale: crate::inpaint::FillScale,
    /// Width in pixels of the band at the region's edge that is generated
    /// last, held to the photo until then, and pasted back fading into it
    /// ("differential diffusion"); 0 pastes the fill with a hard edge.
    #[serde(default)]
    pub soft_edge_px: f32,
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
    SdTuning { keep_loaded, tall_crop: true, margin_mb, ..Default::default() }
}

impl SdTuning {
    /// `self` with the bench overrides from the environment, for
    /// experiments the app and the bench run the same way:
    /// `PRUNR_SD_KEEP_LOADED`, `PRUNR_SD_TALL_CROP`, `PRUNR_SD_OV_DEVICE`,
    /// `PRUNR_SD_MEM_PATTERN`, `PRUNR_SD_DEV_INIT` and `PRUNR_SD_THREADS`.
    pub fn with_env_overrides(self) -> Self {
        Self {
            keep_loaded: crate::env::flag("PRUNR_SD_KEEP_LOADED").unwrap_or(self.keep_loaded),
            tall_crop: crate::env::flag("PRUNR_SD_TALL_CROP").unwrap_or(self.tall_crop),
            ov_device: std::env::var("PRUNR_SD_OV_DEVICE").ok().filter(|s| !s.is_empty()).or(self.ov_device),
            mem: SdMemPlan {
                memory_pattern: crate::env::flag("PRUNR_SD_MEM_PATTERN").or(self.mem.memory_pattern),
                device_initializers: crate::env::flag("PRUNR_SD_DEV_INIT").unwrap_or(self.mem.device_initializers),
                intra_threads: std::env::var("PRUNR_SD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(self.mem.intra_threads),
            },
            ..self
        }
    }

    fn session_key(&self) -> SdSessionKey {
        SdSessionKey { ov_device: self.ov_device.clone(), cpu_only: self.cpu_only, mem: self.mem }
    }
}

/// What a stroke needs on top of what is free: the bundle where it runs
/// (`bundle`) unless already resident, the headroom in system RAM, and the
/// tall crop's extra, on the card when the bundle is on one.
fn stroke_need(id: prunr_models::ModelId, margin_mb: u64, resident: bool, tall: bool, bundle: Charge) -> Charge {
    let (ram_mb, card_mb) = if resident { (0, 0) } else { (bundle.ram_mb, bundle.card_mb) };
    match (tall, bundle.on_card()) {
        (true, true) => {
            // Not measured on a card: the host's extra, erring high.
            let extra = sd_spec(id).tall_crop_card_extra_mb.map_or(SD_TALL_CROP_EXTRA_MB, u64::from);
            Charge { ram_mb: ram_mb + margin_mb, card_mb: card_mb + extra }
        }
        (true, false) => Charge { ram_mb: ram_mb + margin_mb + SD_TALL_CROP_EXTRA_MB, card_mb },
        (false, _) => Charge { ram_mb: ram_mb + margin_mb, card_mb },
    }
}

/// What `id`'s bundle takes on the provider its parts start on.
fn bundle_charge(id: prunr_models::ModelId, cpu_only: bool) -> Option<(&'static str, Charge)> {
    let provider = crate::memory::provider_for(id, cpu_only);
    crate::memory::charge_on(id, provider).map(|c| (provider, c))
}

/// How a Stable Diffusion erase fits in `room` with no bundle loaded:
/// the run-time gate's own answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdRamFit {
    /// Tall regions run as one crop.
    Full,
    /// It runs, but tall regions fall back to 512 px tiles.
    TilesOnly,
    /// The gate refuses the erase: system RAM is short.
    Refused,
    /// The gate refuses the erase: the graphics card's memory is short.
    CardFull,
}

pub fn sd_ram_fit(id: prunr_models::ModelId, room: Room, margin_mb: u64, bundle: Charge) -> SdRamFit {
    let mut tuning = plan_tuning(false, margin_mb);
    match fit_ram(id, &mut tuning, room, false, bundle) {
        Ok(()) if tuning.tall_crop => SdRamFit::Full,
        Ok(()) => SdRamFit::TilesOnly,
        Err(Short::Ram { .. }) => SdRamFit::Refused,
        Err(Short::Card { .. }) => SdRamFit::CardFull,
    }
}

/// A built bundle for `id` is in the cache, so its working set is
/// already counted in the RSS rather than in the free RAM.
fn resident(id: prunr_models::ModelId) -> bool {
    sd_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .any(|((cached, _), e)| *cached == id && matches!(e.value.get(), Some(Ok(_))))
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
    /// The GPU providers bundle `id`'s parts may try, fastest first.
    fn gpu_eps(&self, id: prunr_models::ModelId) -> Vec<EpKind> {
        crate::engine::gpu_eps_for(id, self.cpu_only)
    }

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
            soft_edge_px: 0.0,
        }
    }
}

/// The SD fill of `mask` in `image`; a cancel lands within one UNet step.
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
    let others = take_other_bundles(
        &mut sd_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        &(id, req.tuning.session_key()),
    );
    drop(others);
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
    let bundle = SdSession::get(id, &req.tuning)?;
    let session_ms = get_started.elapsed().as_millis() as u64;
    // `bundle` is declared above the guard so it drops after it: the
    // guard removes the cache's Arc (unless the entry is pinned), then
    // the last reference goes and the ORT bundle is freed.
    let _release_guard = ReleaseOnDrop(id);
    let run_started = Instant::now();
    let taesd = if req.use_taesd && sd_spec(id).latent.takes_taesd() { TaesdSession::get().ok() } else { None };
    let vae: VaeBackend = match taesd.as_ref() {
        Some(t) => VaeBackend::Taesd(t),
        None => VaeBackend::Standard(&bundle),
    };
    // One session for every region the fill scale fills on its own. The
    // fit reads the tall crop as the RAM gate above left it.
    // A scaled-down region's Feather band is that much narrower.
    let out = crate::inpaint::fill_at_scale(image, mask, req.fill_scale, fit(&req), progress, |img, m, at, progress| {
        let scaled = SdInpaintRequest { soft_edge_px: req.soft_edge_px * at, ..req.clone() };
        paint_regions(&bundle, &vae, img, m, &scaled, progress)
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
    let max_side = if req.tall_crop() { SD_CROP_MAX } else { SD_TILE };
    paint_regions_with(image, mask, max_side, progress, |crop, crop_mask, crop_progress| {
        run_one_tile(bundle, vae, crop, crop_mask, req, crop_progress)
    })
}

/// `paint_regions` with the filler passed in: `fill` gets each crop or
/// tile, its mask, and a handle framed on where it sits, so what it
/// reports of its own pixels lands there.
fn paint_regions_with(
    image: &RgbaImage,
    mask: &GrayImage,
    max_side: u32,
    progress: &crate::progress::Progress,
    mut fill: impl FnMut(&RgbaImage, &GrayImage, &crate::progress::Progress) -> Result<RgbaImage, CoreError>,
) -> Result<RgbaImage, CoreError> {
    use crate::progress::{TileRect, Unit};
    let components = mask_regions(mask);
    // One plan per component, made once: the outer progress total and
    // the loop read the same value, so they cannot disagree about how
    // many tiles a region near an image edge really gets.
    let (img_w, img_h) = image.dimensions();
    let plans: Vec<RegionPlan> = components.iter().map(|c| plan_region(c, img_w, img_h, max_side)).collect();
    let outer_total: u32 = plans.iter().map(|p| p.tile_count()).sum();
    // The crops and tiles, in the order they run: the tile map's layout.
    progress.tiles(plans.iter()
        .flat_map(RegionPlan::windows)
        .map(|((x, y, w, h), _)| TileRect::of_pixels(x, y, w, h, img_w, img_h))
        .collect());
    let mut tile_idx: u32 = 0;
    let start = |idx: u32| {
        progress.outer(idx, outer_total, Unit::Crop);
        progress.tile(idx, false);
    };
    let mut out = image.clone();
    for (component, plan) in components.iter().zip(&plans) {
        if progress.is_cancelled() { return Err(CoreError::Cancelled); }
        if let RegionPlan::Tiles(tiles) = plan {
            tracing::info!(
                comp = ?component, n_tiles = tiles.len(),
                "SD: tiling oversized component",
            );
        }
        for ((x, y, w, h), tile) in plan.windows() {
            if progress.is_cancelled() { return Err(CoreError::Cancelled); }
            start(tile_idx);
            let cropped_img = image::imageops::crop_imm(&out, x, y, w, h).to_image();
            let cropped_mask = image::imageops::crop_imm(mask, x, y, w, h).to_image();
            // A failed region fails the run: left as it was, it would read
            // as a finished erase.
            let painted = fill(&cropped_img, &cropped_mask, &progress.within((x, y, w, h), (img_w, img_h)))?;
            match tile {
                Some(tile) => blend_tile(&mut out, &painted, tile),
                None => image::imageops::replace(&mut out, &painted, x as i64, y as i64),
            }
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
/// extent and `pad_mirrored` handles the rest at the ORT boundary.
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
        let extra = span - SD_TILE;
        extra.div_ceil(TILE_STEP_PX) + 1
    }
}

/// Composite a painted tile into `canvas`, ramping alpha from 0 at an
/// edge that abuts a neighbour tile to 1 at TILE_OVERLAP_PX in. No mask
/// needed: outside it the painted tile already carries the source pixels.
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
            let src = ((y * pw + x) * 4) as usize;
            let dst = (((tile.y + y) * cw + (tile.x + x)) * 4) as usize;
            if a >= 1.0 {
                raw_canvas[dst..dst + 4].copy_from_slice(&raw_painted[src..src + 4]);
            } else {
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
/// (4-connectivity, values above `IN_HOLE`). Each returned bbox is the tight
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
    let Some(outer) = crate::inpaint::mask_bbox(mask, IN_HOLE + 1, 0) else {
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
            if visited[lidx] || raw[map.global_idx(sx, sy)] <= IN_HOLE {
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
    if !visited[lidx] && raw[gidx] > IN_HOLE {
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

/// A window's (x, y, w, h) in the image.
type Window = (u32, u32, u32, u32);

impl RegionPlan {
    fn tile_count(&self) -> u32 {
        match self {
            Self::Crop { .. } => 1,
            Self::Tiles(t) => t.len() as u32,
        }
    }

    /// Each window the region runs as, in order: its (x, y, w, h), and
    /// the tile it blends back as (`None` for a crop, pasted as is).
    fn windows(&self) -> Vec<(Window, Option<&TileWindow>)> {
        match self {
            Self::Crop { x, y, w, h } => vec![((*x, *y, *w, *h), None)],
            Self::Tiles(tiles) => tiles.iter().map(|t| ((t.x, t.y, t.w, t.h), Some(t))).collect(),
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
/// (`pad_mirrored` pads it back).
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

// ── Per-tile pipeline ───────────────────────────────────────────────────

fn run_one_tile(
    bundle: &SdSession,
    vae: &VaeBackend,
    image: &RgbaImage,
    mask: &GrayImage,
    req: &SdInpaintRequest,
    progress: &crate::progress::Progress,
) -> Result<RgbaImage, CoreError> {
    use crate::progress::Step;
    let (w, h) = image.dimensions();
    let (pw, ph) = padded_dims(w, h);
    let padded_image = crate::formats::pad_mirrored(image, pw, ph);
    let padded_mask = crate::formats::pad_mirrored(mask, pw, ph);
    let (lw, lh) = ((pw / VAE_FACTOR) as usize, (ph / VAE_FACTOR) as usize);
    let mut timer = TileTimer::start();


    // Guidance above 1.0 runs the UNet twice per step (cond + uncond).
    // A baked-in prompt runs unguided: there is no prompt to steer by.
    let prompted = bundle.text_encoder.is_some();
    let use_cfg = prompted && req.guidance_scale > 1.0 + 1e-3;

    // These run concurrently: each ORT call holds a distinct session mutex.
    if prompted {
        progress.step(Step::ReadingPrompt);
    }
    let prompt = req.prompt.clone();
    let neg_prompt = if use_cfg { Some(req.negative_prompt.clone()) } else { None };
    let soft = (req.soft_edge_px >= 1.0).then(|| edge_ramp(&padded_mask, req.soft_edge_px));
    let (text_emb_cond_f16, text_emb_uncond_f16, masked_latent, mask_latent, photo_latent) =
        std::thread::scope(|s| -> Result<_, CoreError> {
            let cond_h = prompted.then(|| s.spawn(|| text_embedding_f16(bundle, &prompt)));
            let uncond_h = neg_prompt.as_ref().map(|np| {
                let np = np.clone();
                s.spawn(move || text_embedding_f16(bundle, &np))
            });
            let vae_h = s.spawn(|| vae_encode(vae, image_to_minus1_plus1_masked(&padded_image, &padded_mask)));
            // The soft edge holds its band to the photo itself, unmasked.
            let photo_h = soft.is_some().then(|| s.spawn(|| vae_encode(vae, image_to_minus1_plus1_inner(&padded_image, None))));
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
            let photo = photo_h.map(|h| h.join()
                .map_err(|_| CoreError::Inference("vae encoder (photo) thread panicked".into()))?)
                .transpose()?;
            Ok((cond, uncond, vae, mask_lat, photo))
        })?;

    let masked_latent_f16 = masked_latent.mapv(f16::from_f32);
    let mask_latent_f16 = mask_latent.mapv(f16::from_f32);
    // The loop needs only the f16 copies; the f32 ones would otherwise
    // live through the whole UNet loop.
    drop(masked_latent);
    drop(mask_latent);
    let prelude_ms = timer.lap();

    let mut scheduler = Scheduler::new_sd15(req.scheduler, req.num_inference_steps as usize);
    let seed = req.seed.unwrap_or_else(fresh_seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let latent_dim = (1_usize, 4_usize, lh, lw);
    let noise: Vec<f32> = sample_initial_noise(&mut rng, lw, lh).into_raw_vec_and_offset().0;

    let scale = req.guidance_scale;
    let predict = |latent: &[f32], step: UnetStep| {
        let latent_f16 = ndarray::ArrayView4::from_shape(latent_dim, latent)
            .expect("latent_dim matches the latent's length by construction")
            .mapv(f16::from_f32);
        let latent_in_f16 = concat_inpaint_input_f16(
            &latent_f16, &mask_latent_f16, &masked_latent_f16,
        );
        match (text_emb_cond_f16.as_ref(), text_emb_uncond_f16.as_ref()) {
            (Some(cond), Some(uncond)) => cfg_noise_pred(bundle, latent_in_f16, step, cond, uncond, scale),
            (Some(cond), None) => unet_step(bundle, latent_in_f16, step, Cond::Text(cond)),
            (None, _) => unet_step(bundle, latent_in_f16, step, Cond::Size(ph, pw)),
        }
    };
    let taesd = progress.preview_edge().and_then(|_| TaesdSession::get().ok());
    let mut previews = StepPreviews::new(image, mask, (lw, lh), bundle.latent, taesd, progress);
    let show = |x0: &[f32]| {
        if let Some(p) = previews.as_mut() {
            p.show(x0, progress);
        }
    };
    let held = soft.as_ref().zip(photo_latent).map(|(ramp, photo)| HeldEdge {
        photo: photo.into_raw_vec_and_offset().0,
        noise: noise.clone(),
        ramp: pool_cells(pw as usize, ph as usize, |i| ramp[i]),
    });
    let latent_buf = denoise(&mut scheduler, noise, &mut rng, progress, predict, show, held.as_ref())?;

    let unet_ms = timer.lap();

    let final_array = ndarray::Array4::from_shape_vec(latent_dim, latent_buf)
        .expect("latent_dim matches latent_buf length by construction");
    progress.step(Step::DecodingResult);
    let painted = vae_decode(vae, &final_array)?;
    let decode_ms = timer.lap();
    let out = match &soft {
        Some(ramp) => composite_soft(image, &painted, ramp, pw as usize),
        None => composite(image, &painted, mask, w, h),
    };
    let steps_run = scheduler.timesteps().len() as u64;
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

/// The denoising loop from `latent` (the starting noise): per step, the
/// UNet's noise prediction for the current latent (`predict`), the
/// scheduler's step and the count. When previews are asked for, `show`
/// gets the step's clean-latent estimate first; it only reads, so the
/// result is the same bits with previews on or off. Cancel is checked
/// between steps: ORT has no per-op cancel, so a cancel waits out at
/// most one UNet step.
fn denoise(
    scheduler: &mut Scheduler,
    mut latent: Vec<f32>,
    rng: &mut ChaCha8Rng,
    progress: &crate::progress::Progress,
    mut predict: impl FnMut(&[f32], UnetStep) -> Result<Array4<f32>, CoreError>,
    mut show: impl FnMut(&[f32]),
    held: Option<&HeldEdge>,
) -> Result<Vec<f32>, CoreError> {
    use crate::progress::{Step, Unit};
    let timesteps = scheduler.timesteps().to_vec();
    let total = timesteps.len() as u32;
    progress.step(Step::Denoising);
    progress.inner(0, total, Unit::Step);
    if let Some(h) = held {
        h.hold(&mut latent, scheduler.level(0), 0.0);
    }
    for (i, &t) in timesteps.iter().enumerate() {
        if progress.is_cancelled() {
            return Err(CoreError::Cancelled);
        }
        let is_final = i + 1 == timesteps.len();
        let noise_pred = predict(&latent, UnetStep { t, last: is_final })?;
        let np_slice = noise_pred.as_slice().expect("noise_pred: standard layout");
        if progress.preview_edge().is_some() {
            show(&scheduler.x0(&latent, np_slice, i));
        }
        let t_prev = timesteps.get(i + 1).copied().unwrap_or(-1);
        latent = step_array(scheduler, &latent, np_slice, i, t, t_prev, is_final, rng);
        if let Some(h) = held {
            h.hold(&mut latent, scheduler.level(i + 1), (i + 1) as f32 / total as f32);
        }
        progress.inner(i as u32 + 1, total, Unit::Step);
    }
    Ok(latent)
}

/// ComfyUI's latent-preview factors (comfy/latent_formats.py, `SD15` and
/// `SDXL`): a linear fit from a scaled latent's four channels (rows) to
/// the RGB, in [-1, 1], its VAE decodes it to; then the bias.
fn preview_factors(space: prunr_models::SdLatent) -> ([[f32; 3]; 4], [f32; 3]) {
    match space {
        prunr_models::SdLatent::Sd15 => (
            [[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350], [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]],
            [0.0; 3],
        ),
        prunr_models::SdLatent::Sdxl => (
            [[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068], [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]],
            [0.1084, -0.0175, -0.0011],
        ),
    }
}

/// The latent `x0` (planar 4 × `lh` × `lw`) as RGB in [-1, 1] through
/// `space`'s preview factors, one value per latent cell, row-major.
fn latent_to_rgb(x0: &[f32], (lw, lh): (usize, usize), space: prunr_models::SdLatent) -> Vec<[f32; 3]> {
    let (factors, bias) = preview_factors(space);
    let plane = lw * lh;
    (0..plane)
        .map(|i| std::array::from_fn(|ch| bias[ch] + (0..4).map(|k| x0[k * plane + i] * factors[k][ch]).sum::<f32>()))
        .collect()
}

/// The crop's own colour per latent cell, in [-1, 1]: the mean of the
/// 8×8 pixels of each cell wholly inside the crop and outside the mask;
/// `None` for the rest. What a step preview's colours are matched to.
fn known_cells(image: &RgbaImage, mask: &GrayImage, (lw, lh): (usize, usize)) -> Vec<Option<[f32; 3]>> {
    let (w, h) = image.dimensions();
    const F: u32 = VAE_FACTOR;
    (0..lh as u32).flat_map(|cy| (0..lw as u32).map(move |cx| (cx * F, cy * F))).map(|(x0, y0)| {
        if x0 + F > w || y0 + F > h {
            return None;
        }
        let mut sum = [0.0_f32; 3];
        for (x, y) in (y0..y0 + F).flat_map(|y| (x0..x0 + F).map(move |x| (x, y))) {
            if mask.get_pixel(x, y)[0] > 0 {
                return None;
            }
            let p = image.get_pixel(x, y);
            sum.iter_mut().zip(p.0).for_each(|(s, v)| *s += f32::from(v));
        }
        Some(sum.map(|s| byte_to_unit(s / VAE_CELL_PIXELS)))
    }).collect()
}

/// What a step's clean-latent estimate `x0` looks like at the size of
/// the crop's padded input (`8·lw × 8·lh`): through the latent space's
/// colour factors, each channel shifted so the cells the crop's own
/// pixels fill (`known`, from `known_cells`) average to those pixels,
/// then enlarged. The factors are a rough fit: against TAESD's encoding
/// of a photo they put blue ~30 levels high, and the shift brings a flat
/// colour's hole within 3 levels. Not TAESD even when it is the run's
/// VAE: its decode of a 512² crop took 1.06 s on an i7-6700, no small
/// part of a step; this and the composite and resample of a step's
/// preview took 45-100 ms in a debug build.
///
/// Peak per step with previews on, all at latent or crop scale (`x0`
/// and the cells, this picture, the crop the tile keeps composited
/// (`StepPreviews`); the resample and the part sent are each at most
/// the crop):
///
/// | Crop      | x0, cells | Picture | Composite | Total   |
/// |-----------|-----------|---------|-----------|---------|
/// | 512 × 512 | 112 KB    | 1 MB    | 1 MB      | ~3.1 MB |
/// | 768 × 512 | 168 KB    | 1.5 MB  | 1.5 MB    | ~4.7 MB |
fn estimate_picture(x0: &[f32], (lw, lh): (usize, usize), space: prunr_models::SdLatent, known: &[Option<[f32; 3]>]) -> RgbaImage {
    let cells = latent_to_rgb(x0, (lw, lh), space);
    let (mut shift, mut n) = ([0.0_f32; 3], 0.0_f32);
    for (cell, own) in cells.iter().zip(known) {
        let Some(own) = own else { continue };
        (0..3).for_each(|c| shift[c] += own[c] - cell[c]);
        n += 1.0;
    }
    let shift = shift.map(|s| s / n.max(1.0));
    let small = RgbaImage::from_fn(lw as u32, lh as u32, |x, y| {
        let cell = cells[y as usize * lw + x as usize];
        image::Rgba([unit_to_byte(cell[0] + shift[0]), unit_to_byte(cell[1] + shift[1]), unit_to_byte(cell[2] + shift[2]), 255])
    });
    crate::formats::resize_rgba(&small, lw as u32 * VAE_FACTOR, lh as u32 * VAE_FACTOR, crate::formats::ResizeFilter::Bilinear)
}

/// The one latent size the installed TAESD decoder takes: a 512 px tile.
const TAESD_LATENT: (usize, usize) = (SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize);

/// A tile's step previews: the crop with each step's estimate inside its
/// mask, as the final paste puts it, written into one copy of the crop
/// the tile keeps, and sent for the mask's box and the preview pixels
/// the resample reaches from it; the photo the listener shows under the
/// previews holds the rest.
struct StepPreviews<'a> {
    mask: &'a GrayImage,
    filled: RgbaImage,
    /// The box round the pixels the paste replaces (x, y, w, h).
    hole: (u32, u32, u32, u32),
    known: Vec<Option<[f32; 3]>>,
    latent: (usize, usize),
    space: prunr_models::SdLatent,
    /// Decodes each step's estimate as the finished picture will look
    /// (~0.5 s a step on a CPU against a 7-11 s step) when installed, the
    /// latents are SD 1.5's and the crop is the decoder's one square
    /// tile; the instant colour estimate otherwise (a tall crop, RORem).
    taesd: Option<Arc<TaesdSession>>,
}

impl<'a> StepPreviews<'a> {
    /// `None` when nobody asked for previews, or the paste replaces no
    /// pixel of the crop.
    fn new(image: &RgbaImage, mask: &'a GrayImage, latent: (usize, usize), space: prunr_models::SdLatent, taesd: Option<Arc<TaesdSession>>, progress: &crate::progress::Progress) -> Option<Self> {
        progress.preview_edge()?;
        let hole = crate::inpaint::mask_bbox(mask, IN_HOLE + 1, 0)?;
        Some(Self {
            mask,
            filled: image.clone(),
            hole: (hole.x, hole.y, hole.w, hole.h),
            known: known_cells(image, mask, latent),
            latent,
            space,
            taesd: taesd.filter(|_| space.takes_taesd() && latent == TAESD_LATENT),
        })
    }

    /// The picture `x0` makes: decoded by TAESD when there is one (a
    /// failed decode falls back to the estimate), else estimated.
    fn picture(&self, x0: &[f32]) -> RgbaImage {
        let (lw, lh) = self.latent;
        let decoded = self.taesd.as_ref().and_then(|t| {
            let latent = ndarray::Array4::from_shape_vec((1, 4, lh, lw), x0.to_vec()).ok()?;
            vae_decode(&VaeBackend::Taesd(t), &latent)
                .map_err(|err| tracing::debug!(%err, "TAESD step preview failed; estimating"))
                .ok()
        });
        decoded.unwrap_or_else(|| estimate_picture(x0, self.latent, self.space, &self.known))
    }

    fn show(&mut self, x0: &[f32], progress: &crate::progress::Progress) {
        let (w, h) = self.filled.dimensions();
        progress.tile_preview_part((0, 0, w, h), (w, h), self.hole, |size, part| {
            let estimate = self.picture(x0);
            paste(&mut self.filled, &estimate, self.mask, self.hole);
            crate::formats::preview_rgba(Cow::Borrowed(&self.filled), size, part)
        });
    }
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
/// when the export allows it, else two calls. A batch the export refuses
/// (a static batch-1 export) switches the session to two calls for good;
/// one the card had no memory for takes two calls for this step only.
fn cfg_noise_pred(
    bundle: &SdSession,
    latent_in: Array4<f16>,
    step: UnetStep,
    cond: &Array3<f16>,
    uncond: &Array3<f16>,
    scale: f32,
) -> Result<Array4<f32>, CoreError> {
    use std::sync::atomic::Ordering;
    if !bundle.cfg_fallback_to_sequential.load(Ordering::Relaxed) {
        match unet_step_batched(bundle, &latent_in, step, cond, uncond) {
            Ok(pair) => {
                let pred_cond = pair.slice(ndarray::s![0..1, .., .., ..]);
                let pred_uncond = pair.slice(ndarray::s![1..2, .., .., ..]);
                return Ok(cfg_blend(pred_uncond, pred_cond, scale));
            }
            Err(e) if batch_refused_for_good(&e.to_string()) => {
                tracing::warn!(%e, "SD: batched CFG UNet rejected (static batch=1 export); \
                                   running cond and uncond separately for this session");
                bundle.cfg_fallback_to_sequential.store(true, Ordering::Relaxed);
            }
            Err(e) => tracing::warn!(%e, "SD: no memory for the batched CFG UNet; this step runs cond and uncond separately"),
        }
    }
    let pred_cond = unet_step(bundle, latent_in.clone(), UnetStep { last: false, ..step }, Cond::Text(cond))?;
    let pred_uncond = unet_step(bundle, latent_in, step, Cond::Text(uncond))?;
    Ok(cfg_blend(pred_uncond.view(), pred_cond.view(), scale))
}

/// Whether a failed batched UNet call means the export takes one image
/// at a time, rather than a device that ran out of memory this once.
fn batch_refused_for_good(error: &str) -> bool {
    !crate::ep_compat::is_transient(error)
}

// ── Session bundle ──────────────────────────────────────────────────────

/// Tiny distilled VAE pair (~1M params each). Never idle-released: it
/// costs ~10 MB.
pub(crate) struct TaesdSession {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    encoder_input: String,
    decoder_input: String,
}

impl TaesdSession {
    fn get() -> Result<Arc<TaesdSession>, CoreError> {
        // CoreError isn't Clone, so the cached build error is a String.
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

        // In parallel: saves ~1-2 s of cold start; each session is ~5 MB.
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
    /// Input names in the ONNX's positional order, read from the session
    /// so a non-standard export still works.
    unet_inputs: [String; 3],
    vae_encoder_input: String,
    vae_decoder_input: String,
    latent: prunr_models::SdLatent,
    /// The VAE takes f32: SDXL's overflows in f16.
    vae_f32: bool,
    /// Set once the export refuses a batched UNet call (a static batch-1
    /// export, the LCM bundle); guided steps then run cond and uncond one
    /// after the other for this session's lifetime.
    cfg_fallback_to_sequential: std::sync::atomic::AtomicBool,
    /// CLIP embeddings by prompt, in f16. The encoder is deterministic, so
    /// a repeated prompt (every tile, every stroke) skips it. Bounded by
    /// `TEXT_CACHE_ENTRIES`; ~118 KB each.
    text_cache: Mutex<Vec<(String, Arc<Array3<f16>>)>>,
    /// Where each part runs, in build order.
    picks: Vec<(&'static str, SessionPick)>,
    /// The UNet runs on CUDA, where each part keeps its own memory pool:
    /// the UNet's still holds its loop's peak when the VAE decoder runs
    /// after it, and a tall crop's decode then ran out of a 12 GB card.
    /// It hands the spare back after each crop's last step.
    unet_on_cuda: bool,
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

/// Per-id deferred bundle. The shared `OnceLock` makes concurrent
/// `get()` callers wait on one build: two parallel ~15 GB builds nearly
/// OOMed the box.
type SdBundleSlot = Arc<std::sync::OnceLock<Result<Arc<SdSession>, String>>>;
pub(crate) type SdCache = HashMap<(prunr_models::ModelId, SdSessionKey), CacheEntry<SdBundleSlot>>;

fn sd_cache() -> &'static Mutex<SdCache> {
    static CACHE: OnceLock<Mutex<SdCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Take a failed build's entry out of `cache`, if it is still `slot`'s
/// (a later build may have replaced it), so the next caller builds again.
pub(crate) fn forget_failed<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<Arc<OnceLock<T>>>>,
    key: &K,
    slot: &Arc<OnceLock<T>>,
) {
    if cache.get(key).is_some_and(|e| Arc::ptr_eq(&e.value, slot)) {
        cache.remove(key);
    }
}

/// Drop the cache's `Arc` for `id` unless pinned; in-flight callers keep
/// theirs. Runs when a stroke ends: holding ~9 GB through the idle
/// window is worse than a rebuild on the next stroke.
pub(crate) fn release(id: prunr_models::ModelId) {
    let cache = sd_cache();
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let before = guard.len();
    guard.retain(|(cached, _), e| *cached != id || e.pinned);
    if guard.len() < before {
        tracing::info!(?id, "SD: cache entry released after dispatch");
    }
}

/// Load `id`'s bundle for `tuning` (or find it loaded) and say where each
/// part runs. A run with the same tuning uses this bundle; it stays
/// loaded between runs when `tuning.keep_loaded`.
pub fn load_bundle(id: prunr_models::ModelId, tuning: &SdTuning) -> Result<Vec<(&'static str, SessionPick)>, CoreError> {
    SdSession::get(id, tuning).map(|bundle| bundle.picks.clone())
}

/// Take every loaded bundle out of the cache, kept ones too; they free
/// where the result drops.
pub(crate) fn release_all() -> SdCache {
    std::mem::take(&mut *sd_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// The entries `gone` picks, taken out of `cache`. They free where the
/// result drops: after the caller lets go of the cache's lock, so freeing
/// a session (hundreds of ms) holds no other caller up.
fn take_where<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    mut gone: impl FnMut(&K, &CacheEntry<T>) -> bool,
) -> Vec<CacheEntry<T>> {
    let mut out = Vec::new();
    for (k, e) in std::mem::take(cache) {
        if gone(&k, &e) {
            out.push(e);
        } else {
            cache.insert(k, e);
        }
    }
    out
}

/// Entries whose `last_used` is older than `idle` relative to `now`,
/// taken out. Generic over `T` so tests can drive the sweep without
/// spinning up a real ORT session.
fn take_idle<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    now: Instant,
    idle: Duration,
) -> Vec<CacheEntry<T>> {
    take_where(cache, |_, e| !e.pinned && now.duration_since(e.last_used) >= idle)
}

#[cfg(test)]
fn sweep_idle<K: std::hash::Hash + Eq, T>(cache: &mut HashMap<K, CacheEntry<T>>, now: Instant, idle: Duration) -> usize {
    take_idle(cache, now, idle).len()
}

/// Every cached bundle but `keep`'s, pinned or not, taken out.
fn take_other_bundles<K: std::hash::Hash + Eq, T>(cache: &mut HashMap<K, CacheEntry<T>>, keep: &K) -> Vec<CacheEntry<T>> {
    take_where(cache, |k, _| k != keep)
}

/// The cache entry for `key`, made with `make` when missing, and every
/// other entry taken out: one SD bundle is resident at a time, so "keep
/// loaded" holds the eraser in use, not each one used before. A bundle
/// taken out that a run still holds lives until that run ends.
fn admit<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    key: K,
    make: impl FnOnce() -> T,
    now: Instant,
    pin: bool,
) -> (&mut CacheEntry<T>, Vec<CacheEntry<T>>) {
    let others = take_other_bundles(cache, &key);
    let entry = cache.entry(key).or_insert_with(|| CacheEntry::new(make(), now));
    entry.pinned = pin;
    (entry, others)
}

/// The entries idle past `crate::IDLE_RELEASE`, taken out, logging
/// `what` when there are any. Drop the result after the lock.
#[must_use = "the idle sessions free where this drops: drop it after the cache's lock"]
pub(crate) fn sweep_and_log<K: std::hash::Hash + Eq, T>(
    cache: &mut HashMap<K, CacheEntry<T>>,
    now: Instant,
    what: &'static str,
) -> Vec<CacheEntry<T>> {
    let idle = take_idle(cache, now, crate::IDLE_RELEASE);
    if !idle.is_empty() {
        tracing::info!(
            dropped = idle.len(),
            idle_secs = crate::IDLE_RELEASE.as_secs(),
            "{what}",
        );
    }
    idle
}

impl<V, E> CacheEntry<Arc<OnceLock<Result<V, E>>>> {
    /// Refresh `last_used` unless the slot holds an Err: a sticky build
    /// failure refreshed on every retry would never evict.
    pub(crate) fn touch(&mut self, now: Instant) {
        if !matches!(self.value.get(), Some(Err(_))) {
            self.last_used = now;
        }
    }
}

/// The cached value for `key`, built with `build` when there is none
/// (a failed build is returned, not cached, so a model installed since
/// works on the next try). Every other key is taken out: one warm at a
/// time. Two callers racing a build both build; the first stored wins.
pub(crate) fn warm_entry<K: std::hash::Hash + Eq + Clone, V, E>(
    cache: &Mutex<HashMap<K, CacheEntry<Arc<V>>>>,
    key: K,
    build: impl FnOnce() -> Result<V, E>,
) -> Result<Arc<V>, E> {
    let lock = || cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let taken = {
        let mut guard = lock();
        let taken = take_where(&mut guard, |k, _| *k != key);
        if let Some(entry) = guard.get_mut(&key) {
            entry.last_used = Instant::now();
            return Ok(Arc::clone(&entry.value));
        }
        taken
    };
    drop(taken);
    let built = Arc::new(build()?);
    Ok(Arc::clone(&lock().entry(key).or_insert_with(|| CacheEntry::new(built, Instant::now())).value))
}

/// Mark `key`'s entry used now: a run that outlasts the idle release
/// would otherwise be swept the moment it ends.
pub(crate) fn warm_touch<K: std::hash::Hash + Eq, V>(cache: &Mutex<HashMap<K, CacheEntry<Arc<V>>>>, key: &K) {
    if let Some(entry) = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_mut(key) {
        entry.last_used = Instant::now();
    }
}

/// Reclaim lands within ~1 minute of the idle threshold; the sweep
/// itself is near free (<10 entries).
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Spawn a daemon thread named `thread` that evicts idle entries of
/// `cache` every `SWEEP_INTERVAL`, so RAM reclaims even when nobody calls
/// `get()`. Start it once per cache (behind a `OnceLock`).
pub(crate) fn spawn_idle_sweeper<K, T>(
    thread: &'static str,
    cache: fn() -> &'static Mutex<HashMap<K, CacheEntry<T>>>,
    what: &'static str,
) where
    K: std::hash::Hash + Eq + Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .name(thread.to_string())
        .spawn(move || loop {
            std::thread::sleep(SWEEP_INTERVAL);
            let idle = sweep_and_log(&mut cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner), Instant::now(), what);
            drop(idle);
        })
        .unwrap_or_else(|e| panic!("spawn {thread} thread: {e}"));
}

fn ensure_sweeper_running() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        spawn_idle_sweeper("sd-idle-sweeper", sd_cache, "SD: background sweeper released idle session(s)");
    });
}

impl SdSession {
    /// The bundle for `id` under `tuning`, built if no other run has it.
    /// A build that fails is not kept: the next stroke builds again, so
    /// freeing memory or fixing an install takes effect at once.
    fn get(id: prunr_models::ModelId, tuning: &SdTuning) -> Result<Arc<SdSession>, CoreError> {
        let (key, pin) = (&tuning.session_key(), tuning.keep_loaded);
        ensure_sweeper_running();
        let cache = sd_cache();
        let now = Instant::now();

        // Never hold the cache lock through the bundle build.
        let (slot, released): (SdBundleSlot, _) = {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let idle = sweep_and_log(&mut guard, now, "SD: released idle session bundle(s)");
            let (entry, others) = admit(&mut guard, (id, key.clone()), || Arc::new(std::sync::OnceLock::new()), now, pin);
            entry.touch(now);
            (entry.value.clone(), (idle, others))
        };
        drop(released);

        let built = slot.get_or_init(|| Self::new_inner(id, key, tuning.margin_mb).map(Arc::new)).clone();
        if built.is_err() {
            let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            forget_failed(&mut guard, &(id, key.clone()), &slot);
        }
        built.map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId, session_key: &SdSessionKey, margin_mb: u64) -> Result<SdSession, String> {
        // Prewarm builds bypass `process_inpaint_with`, so gate here too,
        // with the margin the user chose.
        check_ram_for(id, &mut SdTuning { margin_mb, cpu_only: session_key.cpu_only, ..SdTuning::default() })?;
        let rss_before_mb = process_rss_mb();
        let parts = prunr_models::multi_part_paths(id)
            .ok_or_else(|| prunr_models::not_installed_error(id))?;
        let by_key: HashMap<&str, PathBuf> = parts.into_iter().collect();

        // Parts build in parallel (the `needs` gate above bounds
        // the four scratch arenas); DirectML stays sequential, its session
        // build races in AbiCustomRegistry.
        type SmokeFn = fn(&mut Session, &str) -> Result<(), String>;
        const PARTS: [(&str, SmokeFn); 4] = [
            ("text_encoder", smoke_test_text_encoder),
            ("vae_encoder",  |s, label| smoke_test_vae(s, label, (1, 3, SD_TILE as usize, SD_TILE as usize))),
            ("vae_decoder",  |s, label| smoke_test_vae(s, label, (1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize))),
            ("unet",         smoke_test_unet),
        ];

        let build = |&(key, smoke): &(&'static str, SmokeFn)|
            -> Result<(&'static str, Session, SessionPick), String> {
                let (s, pick) = build_part_with_ep_ladder(id, key, &by_key, smoke, session_key)?;
                Ok((key, s, pick))
            };

        // A bundle with its prompt baked in ships no text encoder.
        let has_text = by_key.contains_key("text_encoder");
        let wanted: Vec<(&'static str, SmokeFn)> = PARTS.iter().copied()
            .filter(|(key, _)| *key != "text_encoder" || has_text)
            .collect();
        let mut parts: Vec<(&'static str, Session, SessionPick)> =
            if crate::engine::directml_active() {
                wanted.iter().map(build).collect::<Result<_, _>>()?
            } else {
                use rayon::prelude::*;
                wanted.par_iter().map(build).collect::<Result<_, _>>()?
            };

        let picks: Vec<(&'static str, SessionPick)> = parts.iter().map(|(k, _, pick)| (*k, pick.clone())).collect();
        let unet_on_cuda = picks.iter().any(|(k, pick)| *k == "unet" && pick.provider == crate::engine::CUDA_NAME);
        let mut take = |want: &str| -> Result<Session, String> {
            let pos = parts.iter().position(|(k, _, _)| *k == want)
                .ok_or_else(|| format!("SD bundle missing built part: {want}"))?;
            Ok(parts.swap_remove(pos).1)
        };
        let text = has_text.then(|| take("text_encoder")).transpose()?;
        let vae_encoder = take("vae_encoder")?;
        let vae_decoder = take("vae_decoder")?;
        let unet = take("unet")?;

        let unet_inputs = take_three_inputs(&unet, "unet")?;
        let vae_encoder_input = take_first_input(&vae_encoder, "vae_encoder")?;
        let vae_decoder_input = take_first_input(&vae_decoder, "vae_decoder")?;
        let text_encoder = text.map(|session| -> Result<TextEncoder, String> {
            let input = take_first_input(&session, "text_encoder")?;
            Ok(TextEncoder { session: Mutex::new(session), input })
        }).transpose()?;
        let vae_f32 = input_is_f32(&vae_encoder, 0);

        let rss_after_mb = process_rss_mb();
        tracing::info!(
            ?id,
            providers = ?picks.iter().map(|(part, pick)| (*part, pick.provider.as_str())).collect::<Vec<_>>(),
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
            text_encoder,
            unet_inputs,
            vae_encoder_input,
            vae_decoder_input,
            latent: sd_spec(id).latent,
            vae_f32,
            cfg_fallback_to_sequential: std::sync::atomic::AtomicBool::new(false),
            text_cache: Mutex::new(Vec::new()),
            picks,
            unet_on_cuda,
        })
    }
}

/// Try each GPU EP in platform order with the part's smoke test (input
/// shapes differ per part), then fall back to CPU.
fn build_part_with_ep_ladder(
    id: prunr_models::ModelId,
    key: &str,
    by_key: &HashMap<&str, PathBuf>,
    smoke_test: fn(&mut Session, &str) -> Result<(), String>,
    session_key: &SdSessionKey,
) -> Result<(Session, SessionPick), String> {
    let path = by_key.get(key)
        .ok_or_else(|| format!("SD bundle missing required part: {key}"))?;
    let lean = id.sd_spec().is_some_and(|s| s.lean_sessions);

    let landed = tracing::info_span!("sd_part", part = key).in_scope(|| {
        crate::engine::walk_ladder("SD: ", id, Weights::Fp32, &session_key.gpu_eps(id), |ep| {
            crate::cache::gc_stale_for_model(id, ep.as_str());
            let builder = sd_base_builder(session_key.mem, lean).map_err(Miss::Unusable)?;
            let (builder, load_path, draft) = if ep.keeps_optimized_graph() {
                sd_apply_path_cache(builder, path.as_path(), id, ep.as_str(), key)
            } else {
                (builder, Cow::Borrowed(path.as_path()), None)
            };
            // A device override is the bench's; its failures say nothing
            // about the provider's default device.
            let failed = |error| if session_key.records_ep_failures() {
                let suspect_cache = match &load_path {
                    Cow::Owned(cached) => Some(cached.clone()),
                    Cow::Borrowed(_) => None,
                };
                Miss::Failed { error, suspect_cache }
            } else {
                Miss::Unusable(error)
            };
            let (builder, coverage) = crate::ep_coverage::watch(ep, builder);
            let mut built = ep.attach(builder, id, EpSite::SdPart { part: key, ov_device: session_key.ov_device.as_deref() })?;
            let started = Instant::now();
            let mut session = built
                .commit_from_file(&load_path)
                .map_err(|e| failed(format!("ORT session creation failed ({ep}): {e}")))?;
            tracing::info!(?id, part = %key, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "SD: session committed");
            coverage.report(id, Some(key));
            smoke_test(&mut session, key).map_err(|e| failed(format!("{ep} test run failed: {e}")))?;
            if let Some(draft) = draft { draft.publish(); }
            tracing::info!(part = %key, ep = %ep, "SD: GPU session validated");
            Ok(session)
        })
    });
    let failed = match landed {
        Ok((session, ep, failed)) => return Ok((session, SessionPick { failed, ..SessionPick::on(ep.as_str()) })),
        Err(walk) => walk.failed,
    };

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
             incorrect output. Install OpenVINO Runtime (Settings › \
             Hardware) or {LIGHTER_ERASER_HINT}.",
        ));
    }
    crate::cache::gc_stale_for_model(id, crate::engine::CPU_NAME);
    let session = crate::engine::commit_past_bad_cache(|| {
        let builder = sd_base_builder(session_key.mem, lean)
            .map_err(|e| (format!("SD {key}: builder init: {e}"), None))?;
        let (mut builder, load_path, draft) = sd_apply_path_cache(builder, path.as_path(), id, crate::engine::CPU_NAME, key);
        let suspect = match &load_path {
            Cow::Owned(cached) => Some(cached.clone()),
            Cow::Borrowed(_) => None,
        };
        let started = Instant::now();
        let session = builder
            .commit_from_file(&load_path)
            .map_err(|e| (format!("SD {key}: load from {}: {e}", load_path.display()), suspect))?;
        if let Some(draft) = draft { draft.publish(); }
        tracing::info!(?id, part = %key, ep = crate::engine::CPU_NAME, elapsed_ms = started.elapsed().as_millis() as u64, "SD: session committed");
        Ok(session)
    })?;
    Ok((session, SessionPick { failed, ..SessionPick::on(crate::engine::CPU_NAME) }))
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

/// The optimized-graph cache for a `commit_from_file` load: on hit the
/// cached graph's path, on miss the source path with a draft that writes
/// the optimized graph. Best-effort: any error falls back to the source.
fn sd_apply_path_cache<'a>(
    builder: ort::session::builder::SessionBuilder,
    source_path: &'a Path,
    model_id: prunr_models::ModelId,
    ep_name: &str,
    part: &str,
) -> (ort::session::builder::SessionBuilder, Cow<'a, Path>, Option<crate::cache::GraphDraft>) {
    // An optimized copy elsewhere would not find the weights' data files.
    if prunr_models::keeps_weights_apart(model_id) {
        return (builder, Cow::Borrowed(source_path), None);
    }
    let Some(cache_path) = crate::cache::optimized_model_path_for_part(model_id, ep_name, part) else {
        return (builder, Cow::Borrowed(source_path), None);
    };
    if cache_path.is_file() && !crate::cache::graph_not_read(&cache_path) {
        tracing::debug!(?model_id, ep = %ep_name, part, "cache hit (optimized graph, file)");
        return (builder, Cow::Owned(cache_path), None);
    }
    if crate::cache::cache_dir_for_part(model_id, ep_name, part).is_none() {
        return (builder, Cow::Borrowed(source_path), None);
    }
    let graph_bytes = std::fs::metadata(source_path).map_or(0, |m| m.len());
    let (builder, draft) = crate::cache::attach_draft(builder, &cache_path, graph_bytes, model_id, ep_name, Some(part));
    (builder, Cow::Borrowed(source_path), draft)
}

// Per-part smoke tests: one zero-input forward pass catches EP/op
// incompatibilities at build time rather than at the first stroke.

fn smoke_test_text_encoder(s: &mut Session, label: &str) -> Result<(), String> {
    let tokens = clip_tokenize("");
    let t = Tensor::from_array(tokens)
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    smoke_run_single(s, label, &t)
}

fn smoke_test_vae(s: &mut Session, label: &str, shape: (usize, usize, usize, usize)) -> Result<(), String> {
    let t = tensor_in(Array4::<f32>::zeros(shape), input_is_f32(s, 0))
        .map_err(|e| format!("{label}: smoke input: {e}"))?;
    smoke_run_single(s, label, &t)
}

fn smoke_run_single<T>(s: &mut Session, label: &str, input: &ort::value::Value<T>) -> Result<(), String>
where
    T: ort::value::ValueTypeMarker,
{
    let name = take_first_input(s, label)?;
    s.run(inputs![name.as_str() => input])
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
        Ok(Tensor::from_array(arr.mapv(f16::from_f32))?.into_dyn())
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
    let emb_t = if takes_size_ids(s) {
        Tensor::from_array(size_ids(SD_TILE, SD_TILE)).map(|t| t.into_dyn())
    } else {
        Tensor::from_array(Array3::<f16>::zeros((1, CLIP_SEQ_LEN, 768))).map(|t| t.into_dyn())
    }.map_err(|e| format!("{label}: smoke conditioning: {e}"))?;
    let names = take_three_inputs(s, label)?;
    run_unet(s, &names, Array4::<f16>::zeros((1, 9, l, l)), 0, emb_t, false)
        .map(drop)
        .map_err(|e| format!("{label}: smoke {e}"))
}

/// One UNet run of each batch row of `latent` at timestep `t`.
/// One UNet call of a crop's loop: its timestep, and whether it is the
/// loop's last.
#[derive(Clone, Copy, Debug)]
struct UnetStep {
    t: i64,
    last: bool,
}

/// `shrink`: the run hands its CUDA pool's spare memory back as it ends.
fn run_unet<'s>(
    session: &'s mut Session,
    [n0, n1, n2]: &[String; 3],
    latent: Array4<f16>,
    t: i64,
    cond: ort::value::DynValue,
    shrink: bool,
) -> Result<ort::session::SessionOutputs<'s>, String> {
    // Exact in f16: every timestep of SD's 1000-step grid fits the mantissa.
    let ts = ndarray::Array1::<f16>::from_elem(latent.dim().0, f16::from_f32(t as f32));
    let lat_t = Tensor::from_array(latent).map_err(|e| format!("latent tensor: {e}"))?;
    let ts_t = Tensor::from_array(ts).map_err(|e| format!("timestep tensor: {e}"))?;
    let inputs = inputs![
        n0.as_str() => &lat_t,
        n1.as_str() => &ts_t,
        n2.as_str() => &cond,
    ];
    match shrink.then(crate::engine::shrink_cuda_pool).flatten() {
        Some(options) => session.run_with_options(inputs, options),
        None => session.run(inputs),
    }.map_err(|e| format!("run: {e}"))
}

/// `run_unet` on the bundle's UNet, its noise prediction as f32.
fn unet_run(bundle: &SdSession, latent: Array4<f16>, step: UnetStep, cond: ort::value::DynValue, label: &str) -> Result<Array4<f32>, CoreError> {
    let mut session = bundle.unet.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let shrink = step.last && bundle.unet_on_cuda;
    let outputs = run_unet(&mut session, &bundle.unet_inputs, latent, step.t, cond, shrink)
        .map_err(|e| CoreError::Inference(format!("SD {label}: {e}")))?;
    extract::<ndarray::Ix4>(&outputs[0], label)
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
/// SD 1.5 ONNX exports take `int32` `input_ids`.
fn clip_tokenize(text: &str) -> Array2<i32> {
    use std::sync::OnceLock;
    static T: OnceLock<instant_clip_tokenizer::Tokenizer> = OnceLock::new();
    let tokenizer = T.get_or_init(instant_clip_tokenizer::Tokenizer::new);

    // `encode` yields no BOS/EOS; the framing is added here to match
    // diffusers exactly.
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
    let emb = Arc::new(encode_text(bundle, prompt)?.mapv(f16::from_f32));
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
    extract::<ndarray::Ix3>(&outputs[0], "text encoder")
}

// ── VAE encode / decode ────────────────────────────────────────────────

/// `Standard` is the bundle's own VAE (~80M params a leg); `Taesd` a
/// distilled pair (~1M params, ~3× faster decode) with the latent scale
/// baked into the model, so only `Standard` scales.
pub(crate) enum VaeBackend<'a> {
    Standard(&'a SdSession),
    Taesd(&'a TaesdSession),
}

fn vae_encode(backend: &VaeBackend, input: Array4<f32>) -> Result<Array4<f32>, CoreError> {
    let f32_input = matches!(backend, VaeBackend::Standard(b) if b.vae_f32);
    let t = tensor_in(input, f32_input)
        .map_err(|e| CoreError::Inference(format!("vae encoder: input tensor: {e}")))?;
    match backend {
        VaeBackend::Standard(bundle) => {
            let mut session = bundle.vae_encoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![bundle.vae_encoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("SD vae encoder: run: {e}")))?;
            let mut latent = extract::<ndarray::Ix4>(&outputs[0], "vae encoder")?;
            latent *= bundle.latent.vae_scale();
            Ok(latent)
        }
        VaeBackend::Taesd(taesd) => {
            let mut session = taesd.encoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![taesd.encoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("TAESD encoder: run: {e}")))?;
            extract::<ndarray::Ix4>(&outputs[0], "TAESD encoder")
        }
    }
}

fn vae_decode(backend: &VaeBackend, latent: &Array4<f32>) -> Result<RgbaImage, CoreError> {
    match backend {
        VaeBackend::Standard(bundle) => {
            let t = tensor_in(latent / bundle.latent.vae_scale(), bundle.vae_f32)
                .map_err(|e| CoreError::Inference(format!("SD vae decoder: input tensor: {e}")))?;
            let mut session = bundle.vae_decoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![bundle.vae_decoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("SD vae decoder: run: {e}")))?;
            let arr = extract::<ndarray::Ix4>(&outputs[0], "vae decoder")?;
            Ok(minus1_plus1_to_image(&arr))
        }
        VaeBackend::Taesd(taesd) => {
            let latent_f16 = latent.mapv(f16::from_f32);
            let t = Tensor::from_array(latent_f16)
                .map_err(|e| CoreError::Inference(format!("TAESD decoder: input tensor: {e}")))?;
            let mut session = taesd.decoder.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = session.run(inputs![taesd.decoder_input.as_str() => &t])
                .map_err(|e| CoreError::Inference(format!("TAESD decoder: run: {e}")))?;
            let arr = extract::<ndarray::Ix4>(&outputs[0], "TAESD decoder")?;
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
    step: UnetStep,
    cond: Cond<'_>,
) -> Result<Array4<f32>, CoreError> {
    let cond_t = match cond {
        Cond::Text(emb) => Tensor::from_array(emb.clone()).map(|t| t.into_dyn()),
        Cond::Size(h, w) => Tensor::from_array(size_ids(h, w)).map(|t| t.into_dyn()),
    }.map_err(|e| CoreError::Inference(format!("SD unet: conditioning tensor: {e}")))?;
    unet_run(bundle, latent_9ch_f16, step, cond_t, "unet")
}

/// Cond and uncond stacked on the batch axis, one UNet run per step.
/// Errs on a static batch-1 export. Output (2, 4, lh, lw): row 0 cond,
/// row 1 uncond.
fn unet_step_batched(
    bundle: &SdSession,
    latent_9ch_f16: &Array4<f16>,
    step: UnetStep,
    text_emb_cond_f16: &Array3<f16>,
    text_emb_uncond_f16: &Array3<f16>,
) -> Result<Array4<f32>, CoreError> {
    let latent_pair = ndarray::concatenate(
        Axis(0), &[latent_9ch_f16.view(), latent_9ch_f16.view()],
    ).map_err(|e| CoreError::Inference(format!("SD unet batched: latent concat: {e}")))?;
    let text_pair = ndarray::concatenate(
        Axis(0), &[text_emb_cond_f16.view(), text_emb_uncond_f16.view()],
    ).map_err(|e| CoreError::Inference(format!("SD unet batched: text concat: {e}")))?;
    let emb_t = Tensor::from_array(text_pair)
        .map_err(|e| CoreError::Inference(format!("SD unet batched: text tensor: {e}")))?;
    unet_run(bundle, latent_pair, step, emb_t.into_dyn(), "unet (batched)")
}

// ── f16 / f32 conversion at the ONNX boundary ──────────────────────────
//
// The FP16 exports take f16 (an f32 VAE is flagged by
// `SdSession::vae_f32`); all other math stays f32, narrowed only here.

/// An output as f32, whichever precision the export emits.
fn extract<D: ndarray::Dimension>(
    value: &ort::value::DynValue,
    label: &str,
) -> Result<ndarray::Array<f32, D>, CoreError> {
    let shape_err = |e: ndarray::ShapeError| {
        let ndim = D::NDIM.unwrap_or(0);
        CoreError::Inference(format!("SD {label}: shape ≠ {ndim}D: {e}"))
    };
    if let Ok(arr) = value.try_extract_array::<f32>() {
        return arr.into_dimensionality::<D>().map(|a| a.to_owned()).map_err(shape_err);
    }
    let arr = value
        .try_extract_array::<f16>()
        .map_err(|e| CoreError::Inference(format!("SD {label}: extract (tried f32, f16): {e}")))?
        .into_dimensionality::<D>()
        .map_err(shape_err)?;
    Ok(arr.mapv(|x| x.to_f32()))
}

// ── Tensor helpers ──────────────────────────────────────────────────────

/// RGBA → NCHW f32 in [-1, 1] for the SD VAE encoder, alpha dropped.
/// Pixels in the hole (`> IN_HOLE`) become byte-128 grey with a binary
/// cliff: the SD-1.5 inpaint UNet was trained on binary-zeroed masked
/// latents, and a soft edge comes out as a grey ghost. `None` encodes
/// the full image.
fn image_to_minus1_plus1_inner(image: &RgbaImage, mask: Option<&GrayImage>) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let mut a = Array4::<f32>::zeros((1, 3, h_us, w_us));
    // invariant: freshly allocated, standard layout — the slice view is valid.
    let buf = a.as_slice_mut().unwrap();
    let plane = w_us * h_us;
    let raw = image.as_raw();
    let m = mask.map(|m| m.as_raw());
    // The byte-128 grey through the same conversion as the unmasked
    // branch. Hardcoding `0.0` would diverge by ~0.00392.
    let v_fill = byte_to_unit(128.0);
    for y in 0..h_us {
        let src_row = y * w_us * 4;
        let dst_row = y * w_us;
        let mask_row = y * w_us;
        for x in 0..w_us {
            let dst = dst_row + x;
            let src = src_row + x * 4;
            let r = byte_to_unit(f32::from(raw[src]));
            let g = byte_to_unit(f32::from(raw[src + 1]));
            let b = byte_to_unit(f32::from(raw[src + 2]));
            if m.is_some_and(|m| m[mask_row + x] > IN_HOLE) {
                buf[dst]             = v_fill;
                buf[plane + dst]     = v_fill;
                buf[plane * 2 + dst] = v_fill;
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
        dst[i * 4]     = unit_to_byte(buf[i]);
        dst[i * 4 + 1] = unit_to_byte(buf[plane + i]);
        dst[i * 4 + 2] = unit_to_byte(buf[plane * 2 + i]);
        dst[i * 4 + 3] = 255;
    }
    out
}

/// A channel in [-1, 1] as a byte, as the VAE's output is read.
#[inline]
fn unit_to_byte(v: f32) -> u8 {
    crate::math::round_u8((v + 1.0) * 127.5)
}

/// A byte (or a mean of bytes) as the [-1, 1] the VAE reads.
#[inline]
fn byte_to_unit(v: f32) -> f32 {
    v / 127.5 - 1.0
}

/// Mask → latent space (1, 1, h/8, w/8) f32 in [0, 1].
/// Average-pool 8×8 → 1 (equivalent to `F.interpolate(mode="area")`
/// in PyTorch, which is what Diffusers uses for mask preprocessing).
///
/// Not centre-sampled: a cell straddling the boundary averages to ~0.5,
/// keeping sub-cell edge precision that sampling turns jaggy.
fn mask_to_latent(mask: &GrayImage) -> Array4<f32> {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let f = VAE_FACTOR as usize;
    let raw = mask.as_raw();
    let cells = pool_cells(w, h, |i| f32::from(raw[i]) / 255.0);
    Array4::from_shape_vec((1, 1, h.div_ceil(f), w.div_ceil(f)), cells).expect("one value per cell")
}

/// Concatenate the three UNet inputs along the channel axis:
/// [latent (4ch) | mask (1ch) | masked_latent (4ch)] → (1, 9, lh, lw).
/// In f16: the two loop-invariant inputs are converted once, before the loop.
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
    .expect("concat: shapes pre-validated to (1, *, lh, lw)")
}

/// Initial gaussian noise: LCM and DPM++ hold the latent in α-space, so
/// it starts as plain N(0, 1).
fn sample_initial_noise(rng: &mut ChaCha8Rng, lw: usize, lh: usize) -> Array4<f32> {
    Array4::from_shape_vec((1, 4, lh, lw), gaussian(rng, 4 * lw * lh))
        .expect("shape pre-computed; buf length matches")
}

/// `n` N(0, 1) draws from `rng`, in order: every draw of a denoise goes
/// through here so a seed reproduces the same fill.
fn gaussian(rng: &mut ChaCha8Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| StandardNormal.sample(rng)).collect()
}

/// How far in from the region's edge each pixel lies, over `width_px`:
/// 0 outside, rising from the edge to 1 at `width_px` in. `fill_region`
/// grows by 4-neighbour steps, so the distance is counted the same way
/// (city-block): everything painted lies more than the Feather in and is
/// filled in full, whatever its shape; only the Feather band fades.
fn edge_ramp(mask: &GrayImage, width_px: f32) -> Vec<f32> {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let far = (w + h) as u32;
    let mut d: Vec<u32> = mask.as_raw().iter().map(|&v| if v > IN_HOLE { far } else { 0 }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x > 0 { d[i] = d[i].min(d[i - 1] + 1); }
            if y > 0 { d[i] = d[i].min(d[i - w] + 1); }
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if x + 1 < w { d[i] = d[i].min(d[i + 1] + 1); }
            if y + 1 < h { d[i] = d[i].min(d[i + w] + 1); }
        }
    }
    // The region's outermost pixels lie 1 in; the band's innermost `width_px`.
    let span = (width_px - 1.0).max(1.0);
    d.iter().map(|&d| if d == 0 { 0.0 } else { ((d as f32 - 1.0) / span).clamp(0.0, 1.0) }).collect()
}

/// One value per latent cell: the mean of `value` over its 8×8 block of a
/// `w × h` plane; a block cut off at the right or bottom counts the
/// missing pixels as 0.
fn pool_cells(w: usize, h: usize, value: impl Fn(usize) -> f32) -> Vec<f32> {
    let f = VAE_FACTOR as usize;
    let lw = w.div_ceil(f);
    let mut cells = vec![0.0f32; lw * h.div_ceil(f)];
    for y in 0..h {
        let row = (y / f) * lw;
        for x in 0..w {
            cells[row + x / f] += value(y * w + x);
        }
    }
    cells.iter_mut().for_each(|c| *c /= VAE_CELL_PIXELS);
    cells
}

/// The edge band held to the photo while the fill denoises: a latent
/// cell whose ramp is below the share of steps run is set back to the
/// photo noised to the current level, so the band is generated last,
/// against surroundings already in place.
///
/// Peak working set: the photo's latent and the noise, 4 planes each, and
/// the ramp, one plane (a 768×512 crop: 96×64 cells, ~220 KB).
struct HeldEdge {
    photo: Vec<f32>,
    noise: Vec<f32>,
    /// One per latent cell, so its length is a latent plane's.
    ramp: Vec<f32>,
}

impl HeldEdge {
    fn hold(&self, latent: &mut [f32], (signal, noise): (f32, f32), released: f32) {
        let plane = self.ramp.len();
        for (cell, &r) in self.ramp.iter().enumerate() {
            // Released once the share of steps run reaches it.
            if r >= 1.0 - released {
                continue;
            }
            for c in 0..4 {
                let i = c * plane + cell;
                latent[i] = signal * self.photo[i] + noise * self.noise[i];
            }
        }
    }
}

/// The fill pasted onto the photo along `ramp`: the photo at the region's
/// edge, the fill from `ramp` = 1 in. `painted` is the padded crop, read
/// from its top-left; `pw` its width.
fn composite_soft(source: &RgbaImage, painted: &RgbaImage, ramp: &[f32], pw: usize) -> RgbaImage {
    let mut out = source.clone();
    let w = out.width() as usize;
    let pnt = painted.as_raw();
    for (y, row) in out.as_mut().chunks_exact_mut(w * 4).enumerate() {
        for (x, px) in row.chunks_exact_mut(4).enumerate() {
            let a = ramp[y * pw + x];
            if a <= 0.0 {
                continue;
            }
            let p = (y * pw + x) * 4;
            for c in 0..3 {
                px[c] = crate::math::round_u8(px[c] as f32 + (pnt[p + c] as f32 - px[c] as f32) * a);
            }
        }
    }
    out
}

/// Composite painted output back onto source: source-byte-identical
/// outside the mask (no VAE round-trip drift), painted bytes inside.
fn composite(source: &RgbaImage, painted: &RgbaImage, mask: &GrayImage, w: u32, h: u32) -> RgbaImage {
    let mut out = source.clone();
    paste(&mut out, painted, mask, (0, 0, w, h));
    out
}

/// A mask value above this is part of the hole SD fills: the regions it
/// crops round, what it encodes as grey, what the paste replaces.
const IN_HOLE: u8 = 127;

/// `painted`'s RGB over `out` in the hole, within `area` (x, y, w, h);
/// `painted` may be larger (the padded crop), read from its top-left.
fn paste(out: &mut RgbaImage, painted: &RgbaImage, mask: &GrayImage, (ax, ay, aw, ah): (u32, u32, u32, u32)) {
    let w = out.width() as usize;
    let dst = out.as_mut();
    let pnt = painted.as_raw();
    let m = mask.as_raw();
    let painted_w = painted.width() as usize;
    for y in ay as usize..(ay + ah) as usize {
        for x in ax as usize..(ax + aw) as usize {
            let i = y * w + x;
            if i >= m.len() { continue; }
            if m[i] > IN_HOLE {
                let pi = (y * painted_w + x) * 4;
                dst[i * 4]     = pnt[pi];
                dst[i * 4 + 1] = pnt[pi + 1];
                dst[i * 4 + 2] = pnt[pi + 2];
            }
        }
    }
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
/// Takes views so batched-UNet slices pass without copies; `iter()`
/// order is correct for non-contiguous views.
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

/// The scheduler's next latent state.
// `step_idx` is only meaningful to the DPM++ multistep arm, but the
// dispatch is shared. Threading it through a `StepCtx` struct would
// force the LCM arm to carry a field it never reads.
#[allow(clippy::too_many_arguments)]
fn step_array(
    scheduler: &mut Scheduler,
    latent: &[f32],
    noise_pred: &[f32],
    step_idx: usize,
    t: i64,
    t_prev: i64,
    is_final_step: bool,
    rng: &mut ChaCha8Rng,
) -> Vec<f32> {
    match scheduler {
        Scheduler::Lcm(s) => {
            if is_final_step {
                s.denoised(latent, noise_pred, t)
            } else {
                // Re-noise from the seeded RNG so the whole denoise
                // reproduces from `seed`.
                s.step_at(latent, noise_pred, t, t_prev, &gaussian(rng, latent.len()))
            }
        }
        Scheduler::DpmPp2M(s) => s.step(latent, noise_pred, step_idx),
    }
}

// ── Safety guards ───────────────────────────────────────────────────────

use crate::memory::{Charge, Room, Short};

/// The run-time memory gate for a stroke. Refuses when the bundle's base
/// need is not free, in system RAM or on the card it goes to; when only
/// the tall crop's extra is missing, the plan falls back to tiles instead.
/// Unknown free RAM passes.
pub(crate) fn check_ram_for(id: prunr_models::ModelId, tuning: &mut SdTuning) -> Result<(), String> {
    let Some((provider, bundle)) = bundle_charge(id, tuning.cpu_only) else { return Ok(()) };
    let Some(room) = crate::memory::room_now(bundle, provider) else { return Ok(()) };
    let wanted_tall_crop = tuning.tall_crop;
    fit_ram(id, tuning, room, resident(id), bundle).map_err(|short| {
        tracing::info!(?id, provider, ?room, ?short, "SD: refused to load");
        refusal(short, tuning.margin_mb)
    })?;
    if wanted_tall_crop && !tuning.tall_crop {
        tracing::info!(?id, provider, ?room, "SD: no room for the tall crop; tiling instead");
    }
    Ok(())
}

fn refusal(short: Short, margin_mb: u64) -> String {
    let gb = |mb: u64| mb as f64 / 1024.0;
    match short {
        Short::Ram { need_mb, free_mb } => format!(
            "This eraser needs {:.1} GB of free memory to load, {:.1} GB of it kept spare \
             for other apps; {:.1} GB is free. Close other apps, or {LIGHTER_ERASER_HINT}.",
            gb(need_mb), gb(margin_mb), gb(free_mb),
        ),
        Short::Card { need_mb, free_mb } => format!(
            "This eraser needs {:.1} GB free on the graphics card; {:.1} GB is free. \
             Close apps that use the graphics card, or {LIGHTER_ERASER_HINT}.",
            gb(need_mb), gb(free_mb),
        ),
    }
}

/// `check_ram_for` with the memory facts passed in. A model that also
/// runs right on the CPU takes the CPU's rule when its card is short, as
/// its parts then land there.
fn fit_ram(id: prunr_models::ModelId, tuning: &mut SdTuning, room: Room, resident: bool, bundle: Charge) -> Result<(), Short> {
    let margin_mb = tuning.margin_mb;
    let need = |tall| stroke_need(id, margin_mb, resident, tall, bundle);
    if let Err(short @ Short::Card { .. }) = crate::memory::fit(need(false), room) {
        let on_cpu = prunr_models::descriptor(id).map(|d| Charge::of(&d.needs, crate::engine::CPU_NAME));
        return match on_cpu {
            Some(cpu) if sd_spec(id).cpu_output_correct => fit_ram(id, tuning, room, resident, cpu),
            _ => Err(short),
        };
    }
    if tuning.tall_crop && crate::memory::fit(need(true), room).is_err() {
        tuning.tall_crop = false;
    }
    crate::memory::fit(need(false), room)
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

/// SD-1.5 `scaled_linear` β schedule, then α̅ = cumprod(1 - β).
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

/// Per-train-timestep `ln σ_t` where σ_t = √((1-α̅_t)/α̅_t).
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
/// Its timesteps subsample a 50-step skip grid and its step is the
/// `c_skip / c_out` consistency function; LCM weights were trained on
/// that formula, and through DDIM's they drift every step (over-sharp,
/// off-subject). Reference: diffusers `scheduling_lcm.py`, arXiv:2310.04378.
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
    /// coefficients at timestep `t`.
    #[inline]
    fn consistency_coefficients(&self, t: i64) -> (f32, f32) {
        let scaled_t = (t as f32) * self.timestep_scaling;
        let sigma2 = self.sigma_data * self.sigma_data;
        let denom2 = scaled_t * scaled_t + sigma2;
        let c_skip = sigma2 / denom2;
        let c_out = scaled_t / denom2.sqrt();
        (c_skip, c_out)
    }

    /// The consistency function at `t`, per latent value: the clean
    /// latent LCM predicts from a value and its noise prediction. The
    /// steps and the step previews share it, so a preview shows the
    /// estimate the step itself uses.
    fn denoiser(&self, t: i64) -> impl Fn(f32, f32) -> f32 {
        let alpha_t = self.alpha_cumprod(t);
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_beta_t = (1.0 - alpha_t).sqrt();
        let (c_skip, c_out) = self.consistency_coefficients(t);
        move |l, eps| {
            let pred_x0 = (l - sqrt_beta_t * eps) / sqrt_alpha_t;
            c_out * pred_x0 + c_skip * l
        }
    }

    /// The clean latent LCM predicts at `t`; the final step's output.
    pub fn denoised(&self, latent_t: &[f32], noise_pred: &[f32], t: i64) -> Vec<f32> {
        let denoise = self.denoiser(t);
        latent_t.iter().zip(noise_pred).map(|(&l, &eps)| denoise(l, eps)).collect()
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn alpha_cumprod(&self, t: i64) -> f32 {
        let idx = t.clamp(0, self.num_train as i64 - 1) as usize;
        self.alphas_cumprod[idx]
    }

    /// A non-final LCM step: the clean latent re-noised to `t_prev`'s
    /// level with `noise_for_step`, which the caller draws from the
    /// seeded RNG so the whole denoise is deterministic from
    /// `(seed, prompt, mask)`. The final step is `denoised`.
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

        let denoise = self.denoiser(t);

        // SD-1.5 ships set_alpha_to_one=false; final-step alpha_prev =
        // alphas_cumprod[0] (≈0.99915), not 1.0.
        let alpha_prev = if t_prev < 0 { self.alpha_cumprod(0) } else { self.alpha_cumprod(t_prev) };
        let sqrt_alpha_prev = alpha_prev.sqrt();
        let sqrt_beta_prev = (1.0 - alpha_prev).sqrt();

        latent_t.iter()
            .zip(noise_pred.iter())
            .zip(noise_for_step.iter())
            .map(|((&l, &eps), &n)| sqrt_alpha_prev * denoise(l, eps) + sqrt_beta_prev * n)
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
    prev_model_output: Option<Vec<f32>>,
}

impl DpmPp2MScheduler {
    pub fn new_sd15(num_inference: usize) -> Self {
        let log_sigmas = compute_log_sigmas_sd15();
        let sigma_min = log_sigmas[0].exp();
        let sigma_max = log_sigmas[log_sigmas.len() - 1].exp();
        let mut sigmas = compute_karras_sigmas(num_inference, sigma_min, sigma_max);
        // Ends at 0, as Diffusers does: the last step lands on the clean
        // latent. The schedule already ends at sigma_min, so ending there
        // again made the last UNet call change nothing.
        sigmas.push(0.0);

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

    /// The clean latent the noise prediction implies at step `step_idx`
    /// (the step's `m0`).
    pub fn x0(&self, latent: &[f32], noise_pred: &[f32], step_idx: usize) -> Vec<f32> {
        convert_epsilon_to_x0(latent, noise_pred, self.sigmas[step_idx])
    }

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

        let m0 = self.x0(latent, noise_pred, step_idx);
        // At sigma 0 the first-order update is the clean prediction
        // itself; its formula would take λ = +∞ to get there.
        if sigma_t == 0.0 {
            return m0;
        }

        let lambda_s0 = sigma_to_lambda(sigma_s0);
        let lambda_t = sigma_to_lambda(sigma_t);
        let h = lambda_t - lambda_s0;
        let coef = alpha_t * ((-h).exp() - 1.0);
        let ratio = sigma_t_e / sigma_s0_e;

        // First-order on the last step (Diffusers' `lower_order_final`):
        // the sigma drop is largest there and D1 can overshoot.
        let second_order_prev = self.prev_model_output.as_ref().filter(|_| step_idx + 1 < self.num_inference);
        let next: Vec<f32> = if let Some(m1) = second_order_prev {
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
        };

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

/// Log-SNR λ(σ) = ln(α) − ln(σ_eff).
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

/// IPC-portable scheduler kind; the worker builds the `Scheduler` from it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SchedulerKind {
    #[default]
    Lcm,
    DpmPp2MKarras,
}

/// Runtime-selected scheduler for SD denoise.
pub enum Scheduler {
    Lcm(LcmScheduler),
    DpmPp2M(DpmPp2MScheduler),
}

impl Scheduler {
    pub fn new_sd15(kind: SchedulerKind, steps: usize) -> Self {
        match kind {
            SchedulerKind::Lcm => Scheduler::Lcm(LcmScheduler::new_sd15(steps)),
            SchedulerKind::DpmPp2MKarras => Scheduler::DpmPp2M(DpmPp2MScheduler::new_sd15(steps)),
        }
    }

    pub fn timesteps(&self) -> &[i64] {
        match self {
            Scheduler::Lcm(s) => s.timesteps(),
            Scheduler::DpmPp2M(s) => s.timesteps(),
        }
    }

    /// The (signal, noise) scale of the latent before step `k` (`k` = the
    /// steps run so far): `signal · clean + noise · ε`. After the last
    /// step, either scheduler's latent is clean.
    pub fn level(&self, k: usize) -> (f32, f32) {
        match self {
            Scheduler::Lcm(s) => match s.timesteps.get(k) {
                Some(&t) => {
                    let a = s.alpha_cumprod(t);
                    (a.sqrt(), (1.0 - a).sqrt())
                }
                None => (1.0, 0.0),
            },
            Scheduler::DpmPp2M(s) => sigma_to_alpha_sigma(s.sigmas[k.min(s.sigmas.len() - 1)]),
        }
    }

    /// What the picture would be if denoising stopped at step
    /// `step_idx`: the clean latent this scheduler's step predicts from
    /// `latent` and the UNet's `noise_pred`.
    pub fn x0(&self, latent: &[f32], noise_pred: &[f32], step_idx: usize) -> Vec<f32> {
        match self {
            Scheduler::Lcm(s) => s.denoised(latent, noise_pred, s.timesteps[step_idx]),
            Scheduler::DpmPp2M(s) => s.x0(latent, noise_pred, step_idx),
        }
    }
}

#[cfg(test)]
mod tests {
    /// What a TAESD-decoded step preview would cost per step: one
    /// decode of a 512 px tile's latent.
    /// `cargo test --profile profiling -p prunr-core --lib taesd_step_preview_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn taesd_step_preview_timing() {
        let Ok(taesd) = TaesdSession::get() else { return };
        let backend = VaeBackend::Taesd(&taesd);
        let latent = Array4::<f32>::from_shape_fn((1, 4, 64, 64), |(_, c, y, x)| ((c * 7 + y * 3 + x) % 13) as f32 / 13.0 - 0.5);
        for _ in 0..6 {
            let t = std::time::Instant::now();
            let img = vae_decode(&backend, &latent).expect("decoded");
            println!("TAESD decode 64x64 latent -> {:?}: {:?}", img.dimensions(), t.elapsed());
        }
    }

    use super::*;
    use image::{Luma, Rgba};


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
        let next = s.denoised(&latent, &noise, t);

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

    /// A clean latent noised to each step's level with known noise, and
    /// that noise given as the prediction: every scheduler's x0 is the
    /// clean latent again, from pure noise down to the last step.
    #[test]
    fn every_schedulers_x0_recovers_the_clean_latent_from_the_true_noise() {
        let clean = [0.8_f32, -1.2, 0.05, 2.0];
        let eps = [0.3_f32, -0.7, 1.1, -0.2];
        let noised = |(a, s): (f32, f32)| -> Vec<f32> { clean.iter().zip(eps).map(|(&x, e)| a * x + s * e).collect() };
        let lcm = LcmScheduler::new_sd15(4);
        let lcm_levels: Vec<_> = lcm.timesteps().iter().map(|&t| (lcm.alpha_cumprod(t).sqrt(), (1.0 - lcm.alpha_cumprod(t)).sqrt())).collect();
        let dpm = DpmPp2MScheduler::new_sd15(8);
        let dpm_levels: Vec<_> = dpm.sigmas[..dpm.num_inference].iter().map(|&s| sigma_to_alpha_sigma(s)).collect();
        for (scheduler, levels) in [(Scheduler::Lcm(lcm), lcm_levels), (Scheduler::DpmPp2M(dpm), dpm_levels)] {
            for (i, &level) in levels.iter().enumerate() {
                let x0 = scheduler.x0(&noised(level), &eps, i);
                for (got, want) in x0.iter().zip(clean) {
                    assert!((got - want).abs() < 1e-3, "step {i}: x0 {got}, clean {want}");
                }
            }
        }
        // By hand at σ = 1: α = σ_eff = 1/√2, so x0 = √2·x − ε.
        let x0 = convert_epsilon_to_x0(&[1.0], &[0.5], 1.0);
        assert!((x0[0] - (std::f32::consts::SQRT_2 - 0.5)).abs() < 1e-6, "{x0:?}");
    }

    /// The estimate a preview shows is the one the step itself uses:
    /// LCM's final step returns it, DPM++ keeps it as its history.
    #[test]
    fn the_x0_estimate_is_the_one_the_step_uses() {
        let latent = [0.5_f32, -0.25, 1.5, 0.0];
        let noise_pred = [0.1_f32, 0.4, -0.3, 0.9];
        let lcm = LcmScheduler::new_sd15(4);
        let final_step = lcm.denoised(&latent, &noise_pred, lcm.timesteps()[3]);
        assert_eq!(Scheduler::Lcm(lcm).x0(&latent, &noise_pred, 3), final_step);
        let mut dpm = DpmPp2MScheduler::new_sd15(8);
        for i in 0..2 {
            let x0 = dpm.x0(&latent, &noise_pred, i);
            dpm.step(&latent, &noise_pred, i);
            assert_eq!(dpm.prev_model_output.as_deref(), Some(&x0[..]), "step {i}");
        }
    }

    /// ComfyUI's factors on known latents: zero is each space's bias, a
    /// single channel picks out its row, and the cells are planar.
    #[test]
    fn the_latent_preview_maps_known_latents_through_comfyuis_factors() {
        use prunr_models::SdLatent;
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        // Two cells: channel 0 at 1 in the first, channel 3 at 2 in the second.
        let mut x0 = vec![0.0_f32; 8];
        x0[0] = 1.0;
        x0[3 * 2 + 1] = 2.0;
        let sd15 = latent_to_rgb(&x0, (2, 1), SdLatent::Sd15);
        assert!(close(sd15[0], [0.3512, 0.2297, 0.3227]), "{sd15:?}");
        assert!(close(sd15[1], [-0.424, -0.5232, -1.4354]), "{sd15:?}");
        assert!(close(latent_to_rgb(&[0.0; 4], (1, 1), SdLatent::Sdxl)[0], [0.1084, -0.0175, -0.0011]), "SDXL's bias alone");
        assert!(close(latent_to_rgb(&[1.0, 0.0, 0.0, 0.0], (1, 1), SdLatent::Sdxl)[0], [0.4735, 0.4057, 0.4330]), "row 0 plus the bias");

        // With nothing to match, the cells read as the VAE's output does:
        // (v + 1) · 127.5, rounded, each cell 8 px.
        let picture = estimate_picture(&x0, (2, 1), SdLatent::Sd15, &[None, None]);
        assert_eq!(picture.dimensions(), (16, 8));
        assert_eq!(picture.get_pixel(0, 0).0, [172, 157, 169, 255]);
        assert_eq!(picture.get_pixel(15, 7).0, [73, 61, 0, 255], "below black clamps to 0");
    }

    /// The projection is shifted per channel so the cells the crop's own
    /// pixels fill match them, and the cells being filled move with them.
    #[test]
    fn the_preview_matches_its_colours_to_the_crops_own_pixels() {
        use prunr_models::SdLatent;
        // Zero latent: SD 1.5's projection is mid-grey in both cells. The
        // first cell's own pixels are (60, 120, 200).
        let own = [60.0_f32, 120.0, 200.0].map(|v| v / 127.5 - 1.0);
        let picture = estimate_picture(&[0.0; 8], (2, 1), SdLatent::Sd15, &[Some(own), None]);
        // Within one level: the bytes go through f32 and truncate.
        let near = |p: [u8; 4], want: [u8; 3]| p[..3].iter().zip(want).all(|(a, b)| a.abs_diff(b) <= 1);
        assert!(near(picture.get_pixel(0, 0).0, [60, 120, 200]), "{:?}", picture.get_pixel(0, 0));
        assert!(near(picture.get_pixel(12, 4).0, [60, 120, 200]), "the hole's cell moves by the same shift: {:?}", picture.get_pixel(12, 4));

        let image = RgbaImage::from_fn(20, 8, |x, _| Rgba([if x < 8 { 30 } else { 90 }, 60, 90, 255]));
        let mut mask = GrayImage::new(20, 8);
        mask.put_pixel(9, 3, Luma([1]));
        let cells = known_cells(&image, &mask, (3, 1));
        assert_eq!(cells[0], Some([30.0_f32, 60.0, 90.0].map(|v| v / 127.5 - 1.0)), "the mean of its 8×8 pixels");
        assert_eq!(cells[1], None, "any mask in the cell");
        assert_eq!(cells[2], None, "past the crop's edge");
    }

    /// The paste replaces the crop's pixels only inside its mask, reading
    /// the top-left of the padded picture; pasting within the hole's box
    /// alone gives the same crop.
    #[test]
    fn the_paste_changes_only_the_masked_pixels() {
        let image = RgbaImage::from_fn(60, 44, |x, y| Rgba([x as u8, y as u8, 200, 255 - x as u8]));
        let mask = GrayImage::from_fn(60, 44, |x, y| Luma([if (10..30).contains(&x) && (5..20).contains(&y) { 255 } else { 0 }]));
        let painted = RgbaImage::from_pixel(64, 48, Rgba([1, 2, 3, 255]));
        let out = composite(&image, &painted, &mask, 60, 44);
        for ((src, got), m) in image.pixels().zip(out.pixels()).zip(mask.pixels()) {
            let want = if m[0] > IN_HOLE { [1, 2, 3, src[3]] } else { src.0 };
            assert_eq!(got.0, want);
        }
        let hole = crate::inpaint::mask_bbox(&mask, IN_HOLE + 1, 0).expect("a hole");
        let mut boxed = image.clone();
        paste(&mut boxed, &painted, &mask, (hole.x, hole.y, hole.w, hole.h));
        assert_eq!(boxed, out);
    }

    /// The whole crop's step preview, as it was sent before only the
    /// hole's part went: what a part is held to.
    fn whole_crop_preview(image: &RgbaImage, mask: &GrayImage, painted: &RgbaImage, size: (u32, u32)) -> Vec<u8> {
        let (w, h) = image.dimensions();
        crate::formats::preview_rgba(Cow::Owned(composite(image, painted, mask, w, h)), size, (0, 0, size.0, size.1))
    }

    /// With TAESD installed an SD 1.5 step shows the decoded picture, at
    /// the crop's size; RORem's latents keep the estimate.
    #[test]
    fn a_step_preview_is_decoded_by_taesd_when_installed() {
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let Ok(taesd) = TaesdSession::get() else { return };
        let image = RgbaImage::from_pixel(512, 512, Rgba([90, 120, 150, 255]));
        let mask = GrayImage::from_fn(512, 512, |x, _| Luma([if x > 40 { 255 } else { 0 }]));
        let x0: Vec<f32> = (0..4 * 64 * 64).map(|i| (i % 23) as f32 / 23.0 - 0.5).collect();
        let progress = crate::progress::Progress::new(Arc::new(crate::progress::RecordingSink::default())).with_tile_previews(128);
        let estimate = |space| estimate_picture(&x0, (64, 64), space, &known_cells(&image, &mask, (64, 64)));
        let sd15 = StepPreviews::new(&image, &mask, (64, 64), prunr_models::SdLatent::Sd15, Some(Arc::clone(&taesd)), &progress).expect("a hole");
        let decoded = sd15.picture(&x0);
        assert_eq!(decoded.dimensions(), (512, 512), "the crop's size");
        assert_ne!(decoded, estimate(prunr_models::SdLatent::Sd15), "decoded, not estimated");
        let tall = StepPreviews::new(&image, &mask, (64, 96), prunr_models::SdLatent::Sd15, Some(Arc::clone(&taesd)), &progress).expect("a hole");
        assert!(tall.taesd.is_none(), "a tall crop is not the decoder's tile");
        let sdxl = StepPreviews::new(&image, &mask, (64, 64), prunr_models::SdLatent::Sdxl, Some(taesd), &progress).expect("a hole");
        assert_eq!(sdxl.picture(&x0), estimate(prunr_models::SdLatent::Sdxl), "SD 1.5's decoder never reads SDXL latents");
    }

    /// A step's preview is sent for the hole and the preview pixels the
    /// resample reaches from it: the same pixels the whole crop's preview
    /// has there, and every pixel where that preview differs from the
    /// crop's own is inside, unscaled or shrunk. A crop the paste leaves
    /// alone, or nobody listening, makes none.
    #[test]
    fn a_step_preview_sends_the_part_of_the_crop_the_hole_reaches() {
        use crate::progress::{Progress, ProgressUpdate, RecordingSink};
        let image = RgbaImage::from_fn(120, 80, |x, y| Rgba([x as u8, (y * 3) as u8, 200, 255 - x as u8]));
        let mask = GrayImage::from_fn(120, 80, |x, y| {
            Luma([if (40..70).contains(&x) && (20..45).contains(&y) { 255 } else if x == 10 { 100 } else { 0 }])
        });
        let (lw, lh) = (16, 16);
        let x0: Vec<f32> = (0..4 * lw * lh).map(|i| (i % 23) as f32 / 23.0 - 0.5).collect();
        let space = prunr_models::SdLatent::Sd15;
        for edge in [120, 50] {
            let sink = Arc::new(RecordingSink::default());
            let progress = Progress::new(sink.clone()).with_tile_previews(edge);
            let mut previews = StepPreviews::new(&image, &mask, (lw, lh), space, None, &progress).expect("a hole to preview");
            previews.show(&x0, &progress);
            let updates = sink.updates();
            let [ProgressUpdate::TilePreview(part)] = updates.as_slice() else { panic!("{updates:?}") };
            let size = (part.full_width, part.full_height);
            let estimate = estimate_picture(&x0, (lw, lh), space, &known_cells(&image, &mask, (lw, lh)));
            let whole = whole_crop_preview(&image, &mask, &estimate, size);
            let own = crate::formats::preview_rgba(Cow::Borrowed(&image), size, (0, 0, size.0, size.1));
            for (y, x) in (0..size.1).flat_map(|y| (0..size.0).map(move |x| (y, x))) {
                let i = ((y * size.0 + x) * 4) as usize;
                if (part.x..part.x + part.width).contains(&x) && (part.y..part.y + part.height).contains(&y) {
                    let j = (((y - part.y) * part.width + x - part.x) * 4) as usize;
                    assert_eq!(part.rgba[j..j + 4], whole[i..i + 4], "edge {edge}: ({x}, {y}) as the whole crop's preview has it");
                } else {
                    assert_eq!(whole[i..i + 4], own[i..i + 4], "edge {edge}: ({x}, {y}) changes outside the part");
                }
            }
            assert!(part.width < size.0 && part.height < size.1, "edge {edge}: less than the whole crop");
        }
        let listening = Progress::new(Arc::new(RecordingSink::default())).with_tile_previews(64);
        assert!(StepPreviews::new(&image, &GrayImage::from_pixel(120, 80, Luma([100])), (lw, lh), space, None, &listening).is_none());
        assert!(StepPreviews::new(&image, &mask, (lw, lh), space, None, &crate::progress::Progress::none()).is_none());
    }

    /// Driven with a stand-in UNet, the real step loop shows each step's
    /// estimate once, before counting the step, with the source kept
    /// outside the mask; and ends on the same latent bits with previews
    /// on, off, or not listened to.
    #[test]
    fn the_step_loop_previews_each_step_without_changing_the_result() {
        use crate::progress::{check_contract, Progress, ProgressUpdate, RecordingSink, Step as ProgressStep};
        let image = RgbaImage::from_fn(64, 64, |x, y| Rgba([(x * 4) as u8, (y * 4) as u8, 90, 255]));
        let mask = GrayImage::from_fn(64, 64, |x, y| Luma([if (16..48).contains(&x) && (16..48).contains(&y) { 255 } else { 0 }]));
        for (kind, steps) in [(SchedulerKind::Lcm, 4), (SchedulerKind::DpmPp2MKarras, 6)] {
            let run = |progress: &Progress| {
                let mut scheduler = Scheduler::new_sd15(kind, steps);
                let mut rng = ChaCha8Rng::seed_from_u64(7);
                let noise = sample_initial_noise(&mut rng, 8, 8).into_raw_vec_and_offset().0;
                let predict = |l: &[f32], _: UnetStep| Ok(Array4::from_shape_vec((1, 4, 8, 8), l.iter().map(|v| v * 0.5).collect()).expect("an 8×8 latent"));
                let mut previews = StepPreviews::new(&image, &mask, (8, 8), prunr_models::SdLatent::Sd15, None, progress);
                let show = |x0: &[f32]| {
                    if let Some(p) = previews.as_mut() {
                        p.show(x0, progress);
                    }
                };
                denoise(&mut scheduler, noise, &mut rng, progress, predict, show, None).expect("denoise")
            };
            let sink = Arc::new(RecordingSink::default());
            let previewed = run(&Progress::new(sink.clone()).with_tile_previews(64));
            let quiet = Arc::new(RecordingSink::default());
            assert_eq!(previewed, run(&Progress::new(quiet.clone())), "{kind:?}: previews change nothing");
            assert_eq!(previewed, run(&Progress::none()), "{kind:?}");
            assert!(!quiet.updates().iter().any(|u| matches!(u, ProgressUpdate::TilePreview(_))), "no preview unless asked");

            let updates = sink.updates();
            assert_eq!(check_contract(&updates), Ok(()));
            let order: Vec<String> = updates.iter().filter_map(|u| match u {
                ProgressUpdate::TilePreview(_) => Some("preview".to_string()),
                ProgressUpdate::Inner { done, .. } => Some(done.to_string()),
                _ => None,
            }).collect();
            let want: Vec<String> = std::iter::once("0".to_string())
                .chain((1..=steps).flat_map(|i| ["preview".to_string(), i.to_string()]))
                .collect();
            assert_eq!(order, want, "{kind:?}: one preview per step, before its count");
            assert_eq!(updates[0], ProgressUpdate::Step(ProgressStep::Denoising));
            for u in &updates {
                let ProgressUpdate::TilePreview(p) = u else { continue };
                assert_eq!((p.x, p.y, p.width, p.height, p.full_width, p.full_height), (16, 16, 32, 32, 64, 64), "the hole, at its place");
                let changed = p.rgba.chunks(4).enumerate().filter(|(i, got)| {
                    let (x, y) = (p.x + *i as u32 % p.width, p.y + *i as u32 / p.width);
                    *got != image.get_pixel(x, y).0.as_slice()
                }).count();
                assert!(changed > 0, "{kind:?}: the estimate shows inside the mask");
            }
        }
    }

    /// A batched call the card had no memory for runs as two halves for
    /// that step only; one the export refuses (a static batch of one)
    /// switches the session to halves for good.
    #[test]
    fn only_a_refused_batch_switches_to_halves_for_good() {
        let out_of_memory = "SD unet (batched): run: Non-zero status code returned while running Softmax node. \
            Status Message: bfc_arena.cc:358 Failed to allocate memory for requested buffer of size 1207959552";
        let static_batch = "SD unet (batched): run: Got invalid dimensions for input: sample for the following indices \
            index: 0 Got: 2 Expected: 1";
        assert!(!batch_refused_for_good(out_of_memory));
        assert!(batch_refused_for_good(static_batch));
    }

    /// The UNet hands back its pool's spare memory once per crop, after
    /// the loop's last call, so the VAE decoder after it has the room.
    #[test]
    fn only_the_last_unet_call_of_a_crop_is_marked_last() {
        for (kind, steps) in [(SchedulerKind::Lcm, 4), (SchedulerKind::DpmPp2MKarras, 6)] {
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            let lasts = std::cell::RefCell::new(Vec::new());
            let predict = |_: &[f32], step: UnetStep| { lasts.borrow_mut().push(step.last); Ok(Array4::zeros((1, 4, 1, 1))) };
            denoise(&mut Scheduler::new_sd15(kind, steps), vec![0.0; 4], &mut rng, &crate::progress::Progress::none(), predict, |_| {}, None).expect("denoise");
            let lasts = lasts.into_inner();
            assert_eq!(lasts.len(), steps, "{kind:?}");
            assert_eq!(lasts.iter().filter(|&&l| l).count(), 1, "{kind:?}");
            assert!(lasts.last().copied().unwrap_or(false), "{kind:?}: the final call");
        }
    }

    #[test]
    fn a_cancelled_step_loop_stops_before_the_next_unet_call() {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let progress = crate::progress::Progress::none().with_cancel(flag);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let called = std::cell::Cell::new(false);
        let predict = |_: &[f32], _: UnetStep| { called.set(true); Ok(Array4::zeros((1, 4, 1, 1))) };
        let result = denoise(&mut Scheduler::new_sd15(SchedulerKind::Lcm, 4), vec![0.0; 4], &mut rng, &progress, predict, |_| {}, None);
        assert!(matches!(result, Err(CoreError::Cancelled)) && !called.get());
    }

    /// Each crop or tile reports its preview where it sits: a region
    /// that fits one crop, one that needs three tiles, and a region
    /// filled on a scaled-down copy, all on the photo's preview.
    #[test]
    fn each_crop_previews_where_it_sits_in_the_photo() {
        use crate::progress::{check_contract, Progress, ProgressUpdate, RecordingSink, Step as ProgressStep};
        let fake_fill = |crop: &RgbaImage, _: &GrayImage, p: &Progress| {
            let (w, h) = crop.dimensions();
            p.tile_preview((0, 0, w, h), (w, h), |pw, ph| vec![9; (pw * ph * 4) as usize]);
            Ok(crop.clone())
        };
        let places = |updates: &[ProgressUpdate]| -> Vec<(u32, u32, u32, u32, u32, u32)> {
            updates.iter().filter_map(|u| match u {
                ProgressUpdate::TilePreview(p) => Some((p.x, p.y, p.width, p.height, p.full_width, p.full_height)),
                _ => None,
            }).collect()
        };

        let image = RgbaImage::from_pixel(1200, 900, Rgba([50, 60, 70, 255]));
        let mut mask = GrayImage::new(1200, 900);
        for (x, y) in (0..1200).flat_map(|x| (0..100).map(move |y| (x, y))).chain((1000..1100).flat_map(|x| (700..800).map(move |y| (x, y)))) {
            mask.put_pixel(x, y, Luma([255]));
        }
        let sink = Arc::new(RecordingSink::default());
        paint_regions_with(&image, &mask, SD_TILE, &Progress::new(sink.clone()).with_tile_previews(600), fake_fill).expect("paint");
        // At half size: the strip's three tiles at x 0, 384, 688, then the
        // square's crop at (688, 388).
        assert_eq!(places(&sink.updates()), vec![
            (0, 0, 256, 256, 600, 450), (192, 0, 256, 256, 600, 450),
            (344, 0, 256, 256, 600, 450), (344, 194, 256, 256, 600, 450),
        ]);
        let mut run = vec![ProgressUpdate::Step(ProgressStep::Erasing)];
        run.extend(sink.updates());
        assert_eq!(check_contract(&run), Ok(()));

        let photo = RgbaImage::from_pixel(1600, 1200, Rgba([50, 60, 70, 255]));
        let mut hole = GrayImage::new(1600, 1200);
        for (x, y) in (600..900).flat_map(|x| (400..700).map(move |y| (x, y))) {
            hole.put_pixel(x, y, Luma([255]));
        }
        let sink = Arc::new(RecordingSink::default());
        let fit = crate::inpaint::Fit { long: 448, short: 448 };
        crate::inpaint::fill_at_scale(&photo, &hole, crate::inpaint::FillScale::Half, fit, &Progress::new(sink.clone()).with_tile_previews(800), |img, m, _, p| {
            paint_regions_with(img, m, SD_TILE, p, fake_fill)
        }).expect("fill");
        let found = places(&sink.updates());
        assert_eq!(found.len(), 1, "{found:?}");
        let (x, y, w, h, full_w, full_h) = found[0];
        assert_eq!((full_w, full_h), (800, 600), "the photo's preview, not the shrunk copy's");
        assert!(x <= 300 && y <= 200 && x + w >= 450 && y + h >= 350, "covers the hole at half size: {:?}", found[0]);
        assert!(w < full_w && h < full_h, "only the crop around the hole: {:?}", found[0]);
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
        assert!(primed_terminal.iter().all(|v| v.is_finite()), "{primed_terminal:?}");
    }

    /// The last step lands on the clean latent its prediction implies, at
    /// sigma 0 as in Diffusers (`final_sigmas_type="zero"`). A last step
    /// from sigma_min to sigma_min ran the UNet and left that noise in.
    #[test]
    fn dpmpp2m_last_step_lands_on_the_clean_latent() {
        let latent = vec![0.5_f32, -0.2, 0.9, 0.0];
        let noise = vec![0.1_f32, 0.3, -0.4, 0.2];
        let mut s = DpmPp2MScheduler::new_sd15(8);
        let _ = s.step(&latent, &noise, 6);
        let clean = s.x0(&latent, &noise, 7);
        assert_eq!(s.step(&latent, &noise, 7), clean);
        assert_eq!(Scheduler::DpmPp2M(DpmPp2MScheduler::new_sd15(8)).level(8), (1.0, 0.0), "clean after the last step, as LCM");
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

    /// The soft edge's ramp is 0 outside and at the region's edge and 1
    /// from its width in.
    #[test]
    fn the_edge_ramp_rises_from_the_edge_to_the_core() {
        let mask = GrayImage::from_fn(64, 8, |x, _| Luma([if x >= 16 { 255 } else { 0 }]));
        let ramp = edge_ramp(&mask, 8.0);
        let at = |x: usize| ramp[4 * 64 + x];
        assert_eq!(at(10), 0.0, "outside");
        assert!(at(16) < 0.2, "at the edge: {}", at(16));
        assert!(at(20) > at(17) && at(20) < 1.0, "rising");
        assert_eq!(at(40), 1.0, "the core");
    }

    /// Whatever its shape, everything painted is filled in full: only the
    /// Feather band `fill_region` added fades into the photo. A speck and
    /// a disc, each grown by Expand and Feather.
    #[test]
    fn the_soft_edge_fills_everything_painted_in_full() {
        for (grow, feather) in [(0.0, 24.0), (2.0, 24.0), (2.0, 4.0), (16.0, 22.0)] {
            for radius in [1i32, 10] {
                let painted = GrayImage::from_fn(200, 200, |x, y| {
                    let (dx, dy) = (x as i32 - 100, y as i32 - 100);
                    Luma([if dx * dx + dy * dy <= radius * radius { 255 } else { 0 }])
                });
                let region = crate::inpaint_blend::fill_region(&painted, grow, feather);
                let ramp = edge_ramp(&region, feather);
                let least = painted.as_raw().iter().zip(&ramp).filter(|(&p, _)| p > 127).map(|(_, &r)| r).fold(1.0f32, f32::min);
                assert_eq!(least, 1.0, "grow {grow}, feather {feather}, radius {radius}");
                let outer = region.as_raw().iter().zip(&ramp).filter(|(&m, _)| m > 127).map(|(_, &r)| r).fold(1.0f32, f32::min);
                assert!(outer < 0.2, "the band fades to the photo at the edge: {outer}");
            }
        }
    }

    /// Each scheduler's latent gets cleaner step by step, ending (almost)
    /// clean.
    #[test]
    fn the_noise_level_falls_step_by_step() {
        for kind in [SchedulerKind::Lcm, SchedulerKind::DpmPp2MKarras] {
            let s = Scheduler::new_sd15(kind, 8);
            let levels: Vec<(f32, f32)> = (0..=8).map(|k| s.level(k)).collect();
            assert!(levels.windows(2).all(|p| p[1].0 >= p[0].0 && p[1].1 <= p[0].1), "{kind:?}: {levels:?}");
            assert!(levels[0].1 > 0.9 && levels[8].0 > 0.99, "{kind:?}: starts noisy, ends clean: {levels:?}");
        }
    }

    /// A band cell is held to the noised photo until its share of the
    /// steps has run, then left to the fill; the core is never held.
    #[test]
    fn the_edge_band_is_held_until_its_release() {
        let held = HeldEdge { photo: vec![1.0; 4 * 3], noise: vec![0.0; 4 * 3], ramp: vec![0.0, 0.5, 1.0] };
        let run = |released| {
            let mut latent = vec![9.0; 4 * 3];
            held.hold(&mut latent, (0.5, 0.5), released);
            [latent[0], latent[1], latent[2]]
        };
        assert_eq!(run(0.25), [0.5, 0.5, 9.0], "early: the outside and the band held, the core free");
        assert_eq!(run(0.5), [0.5, 9.0, 9.0], "halfway: the band at ½ released");
        assert_eq!(run(1.0), [9.0, 9.0, 9.0], "at the end nothing is held");
    }

    /// The soft paste keeps the photo at the region's edge and the fill in
    /// its core, and mixes them along the ramp between.
    #[test]
    fn the_soft_paste_fades_from_the_photo_to_the_fill() {
        let photo = RgbaImage::from_pixel(3, 1, Rgba([0, 0, 0, 255]));
        let fill = RgbaImage::from_pixel(3, 1, Rgba([200, 200, 200, 255]));
        let out = composite_soft(&photo, &fill, &[0.0, 0.5, 1.0], 3);
        assert_eq!([out.get_pixel(0, 0).0[0], out.get_pixel(1, 0).0[0], out.get_pixel(2, 0).0[0]], [0, 100, 200]);
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
    fn the_gate_takes_the_tall_crop_only_with_the_extra_ram() {
        let id = prunr_models::ModelId::SdV15InpaintFp16;
        let m = SD_DEFAULT_MARGIN_MB;
        let host = on_cpu(id);
        let base = stroke_need(id, m, false, false, host).ram_mb;
        assert!(base > m, "the working set counts when the bundle is not resident");
        assert_eq!(stroke_need(id, m, true, false, host).ram_mb, m, "a resident bundle needs only the headroom");
        assert_eq!(stroke_need(id, m, true, true, host).ram_mb, m + SD_TALL_CROP_EXTRA_MB);
        let fit = |free: u64, resident: bool| {
            let mut t = plan_tuning(false, m);
            fit_ram(id, &mut t, Room { ram_mb: free, card_mb: None }, resident, host).map(|()| t.tall_crop)
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
        use prunr_models::ModelId;
        let margin = 2048;
        for id in [ModelId::SdV15InpaintFp16, ModelId::SdV15LcmInpaintFp16, ModelId::RoremFp16] {
            for bundle in [on_cpu(id), TEST_CARD] {
                let base = stroke_need(id, margin, false, false, bundle).ram_mb;
                let tall = stroke_need(id, margin, false, true, bundle).ram_mb;
                for card_mb in [None, Some(0), Some(TEST_CARD.card_mb + crate::memory::CARD_RESERVE_MB), Some(u64::MAX)] {
                    for free in [base - 1, base, tall.max(base + 1) - 1, tall] {
                        let room = Room { ram_mb: free, card_mb };
                        let mut t = plan_tuning(false, margin);
                        let expected = match fit_ram(id, &mut t, room, false, bundle) {
                            Err(Short::Card { .. }) => SdRamFit::CardFull,
                            Err(Short::Ram { .. }) => SdRamFit::Refused,
                            Ok(()) if t.tall_crop => SdRamFit::Full,
                            Ok(()) => SdRamFit::TilesOnly,
                        };
                        assert_eq!(sd_ram_fit(id, room, margin, bundle), expected, "{id:?} {bundle:?} {room:?}");
                    }
                }
            }
        }
    }

    /// The bundle's needs on the CPU, for the rule's tests.
    fn on_cpu(id: prunr_models::ModelId) -> Charge {
        Charge::of(&prunr_models::descriptor(id).unwrap().needs, crate::engine::CPU_NAME)
    }

    /// A bundle on a card, for the rule's own tests apart from any measured one.
    const TEST_CARD: Charge = Charge { ram_mb: 3_000, card_mb: 5_000 };

    /// On a card the bundle counts the card's memory (with its reserve)
    /// and only its share of system RAM; the tall crop's extra is the
    /// card's. A short card refuses SD 1.5, which is wrong on the CPU,
    /// and sends RORem to the CPU's rule; an unread card is trusted.
    #[test]
    fn the_gate_on_a_card_counts_the_cards_memory() {
        use prunr_models::ModelId::{RoremFp16, SdV15InpaintFp16};
        let m = SD_DEFAULT_MARGIN_MB;
        let reserve = crate::memory::CARD_RESERVE_MB;
        let extra = sd_spec(SdV15InpaintFp16).tall_crop_card_extra_mb.map(u64::from).expect("measured on CUDA");
        assert!(extra > 0);
        let fit = |id, ram_mb, card_mb, resident| {
            let mut t = plan_tuning(false, m);
            fit_ram(id, &mut t, Room { ram_mb, card_mb }, resident, TEST_CARD).map(|()| t.tall_crop)
        };
        let (ram, card) = (TEST_CARD.ram_mb + m, TEST_CARD.card_mb + reserve);
        assert_eq!(fit(SdV15InpaintFp16, ram, Some(card + extra), false), Ok(true), "room for the tall crop on the card");
        assert_eq!(fit(SdV15InpaintFp16, ram, Some(card + extra - 1), false), Ok(false), "tiles when the card lacks the crop's extra");
        assert_eq!(fit(SdV15InpaintFp16, ram, Some(card - 1), false), Err(Short::Card { need_mb: card, free_mb: card - 1 }));
        assert_eq!(fit(SdV15InpaintFp16, ram - 1, Some(card + extra), false), Err(Short::Ram { need_mb: ram, free_mb: ram - 1 }), "the host share and the headroom");
        assert_eq!(fit(SdV15InpaintFp16, ram, None, false), Ok(true), "an unread card is trusted");
        assert_eq!(fit(SdV15InpaintFp16, m, Some(extra + reserve), true), Ok(true), "resident: the crop's extra and the headroom");
        let rorem_on_cpu = stroke_need(RoremFp16, m, false, false, on_cpu(RoremFp16)).ram_mb;
        assert_eq!(fit(RoremFp16, rorem_on_cpu - 1, Some(0), false), Err(Short::Ram { need_mb: rorem_on_cpu, free_mb: rorem_on_cpu - 1 }), "RORem off the card takes the CPU's need");
        assert!(fit(RoremFp16, rorem_on_cpu, Some(0), false).is_ok());
    }

    /// A bundle's weights sit in the card's memory on CUDA: a 16 GB
    /// machine with 11.6 GB free and a 12 GB card runs every SD eraser,
    /// tall crop included (each was refused for wanting 14-16 GB of system
    /// RAM, the CPU's need).
    #[test]
    fn every_sd_eraser_on_an_nvidia_card_fits_a_12_gb_card_and_a_16_gb_machine() {
        use prunr_models::ModelId;
        for id in [ModelId::SdV15InpaintFp16, ModelId::SdV15LcmInpaintFp16, ModelId::RoremFp16] {
            let bundle = Charge::of(&prunr_models::descriptor(id).unwrap().needs, "CUDA");
            assert!(bundle.on_card(), "{id:?} measured on CUDA");
            let mut t = plan_tuning(false, SD_DEFAULT_MARGIN_MB);
            assert_eq!(fit_ram(id, &mut t, Room { ram_mb: 11_600, card_mb: Some(11_600) }, false, bundle), Ok(()), "{id:?}");
            assert!(t.tall_crop, "{id:?}: the tall crop fits too");
        }
    }

    /// The Fast bundle peaks at about 14 GB while it loads: with 12 GB
    /// free and the default headroom the erase is refused before it
    /// starts, not killed by the memory watchdog halfway through.
    #[test]
    fn an_erase_that_cannot_fit_its_load_is_refused_up_front() {
        let ram = |ram_mb| Room { ram_mb, card_mb: None };
        for id in [prunr_models::ModelId::SdV15LcmInpaintFp16, prunr_models::ModelId::SdV15InpaintFp16] {
            let mut t = plan_tuning(false, SD_DEFAULT_MARGIN_MB);
            assert!(fit_ram(id, &mut t, ram(12_147), false, on_cpu(id)).is_err(), "{id:?} refused with 12 GB free");
            let mut t = plan_tuning(false, SD_DEFAULT_MARGIN_MB);
            assert!(fit_ram(id, &mut t, ram(16_500), false, on_cpu(id)).is_ok(), "{id:?} runs with 16.5 GB free");
        }
        let id = prunr_models::ModelId::SdV15InpaintFp16;
        assert_eq!(sd_ram_fit(id, ram(12_147), SD_DEFAULT_MARGIN_MB, on_cpu(id)), SdRamFit::Refused);
    }

    /// "CPU only" reaches the SD parts through the bundle key, so a
    /// worker process needs no extra state, and a bundle built under the
    /// other policy is never reused.
    #[test]
    fn a_cpu_only_bundle_tries_no_gpu_provider_and_has_its_own_key() {
        let auto = plan_tuning(false, 0).session_key();
        let cpu = SdTuning { cpu_only: true, ..plan_tuning(false, 0) }.session_key();
        assert!(cpu.gpu_eps(prunr_models::ModelId::SdV15InpaintFp16).is_empty());
        assert_ne!(auto, cpu);
    }

    /// One SD bundle is resident at a time: "keep loaded" pins the
    /// eraser, not every bundle it has used. Switching Fast to Quality
    /// with both pinned held two 2 GB bundles and swapped the machine.
    #[test]
    fn a_failed_bundle_build_is_forgotten() {
        let now = Instant::now();
        let key = (prunr_models::ModelId::SdV15InpaintFp16, SdTuning::default().session_key());
        let failed: SdBundleSlot = Arc::new(std::sync::OnceLock::new());
        let _ = failed.set(Err("needs 15.6 GB".into()));
        let mut cache: SdCache = HashMap::new();
        cache.insert(key.clone(), CacheEntry::new(Arc::clone(&failed), now));
        let other: SdBundleSlot = Arc::new(std::sync::OnceLock::new());
        forget_failed(&mut cache, &key, &other);
        assert!(cache.contains_key(&key), "an entry another build replaced it with stays");
        forget_failed(&mut cache, &key, &failed);
        assert!(!cache.contains_key(&key), "the failed build is gone, so the next stroke builds again");
    }

    #[test]
    fn admitting_a_bundle_drops_the_other_ones_pinned_or_not() {
        let now = Instant::now();
        let mut cache: HashMap<&str, CacheEntry<u8>> = HashMap::new();
        admit(&mut cache, "fast", || 1, now, true);
        let (_, others) = admit(&mut cache, "quality", || 2, now, true);
        assert_eq!(others.iter().map(|e| e.value).collect::<Vec<_>>(), vec![1], "taken out, to free after the lock");
        assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["quality"]);
        assert!(cache["quality"].pinned);
        admit(&mut cache, "quality", || 9, now, false);
        assert_eq!(cache["quality"].value, 2, "the resident bundle is reused, not rebuilt");
    }

    /// A warm model is built once and handed out again; asking for
    /// another frees the first; a failed build is not kept, so the next
    /// ask builds again.
    #[test]
    fn a_warm_entry_builds_once_and_keeps_one_model() {
        let cache: Mutex<HashMap<u8, CacheEntry<Arc<u32>>>> = Mutex::new(HashMap::new());
        let builds = std::cell::Cell::new(0);
        let build = |v: u32| { builds.set(builds.get() + 1); Ok::<_, String>(v) };
        let first = warm_entry(&cache, 1, || build(10)).unwrap();
        assert!(Arc::ptr_eq(&first, &warm_entry(&cache, 1, || build(11)).unwrap()));
        assert_eq!(builds.get(), 1);
        assert_eq!(*warm_entry(&cache, 2, || build(20)).unwrap(), 20);
        assert_eq!(cache.lock().unwrap().keys().copied().collect::<Vec<_>>(), vec![2], "one warm at a time");
        assert!(warm_entry(&cache, 3, || Err::<u32, _>("not installed".to_string())).is_err());
        assert_eq!(*warm_entry(&cache, 3, || build(30)).unwrap(), 30, "the failure was not kept");
    }


    /// LCM's timesteps are Diffusers' `LCMScheduler.set_timesteps(n)`,
    /// copied from its Python: every k-th index of the reversed origin
    /// schedule [999, 979, …, 19], k from floor(linspace(0, 50, n,
    /// endpoint=False)). 4 steps is the paper's fast preview; the two
    /// counts catch drift in the index math at the extremes.
    #[test]
    fn lcm_timesteps_match_diffusers_reference() {
        for (steps, want) in [(8, &[999, 879, 759, 639, 499, 379, 259, 139][..]), (4, &[999, 759, 499, 259][..])] {
            assert_eq!(LcmScheduler::new_sd15(steps).timesteps(), want, "{steps} steps");
        }
    }

    /// CFG is `uncond + scale · (cond − uncond)`, read straight from
    /// axis-0 slices of one batched UNet output: scale 1 is cond itself.
    #[test]
    fn cfg_blend_accepts_axis0_slices_directly() {
        let pair = Array4::<f32>::from_shape_fn((2, 4, 2, 2), |(b, _, _, _)| if b == 0 { 1.0 } else { 5.0 });
        let uncond = pair.slice(ndarray::s![0..1, .., .., ..]);
        let cond = pair.slice(ndarray::s![1..2, .., .., ..]);
        for (scale, want) in [(1.0, 5.0), (7.5, 31.0)] {
            let out = cfg_blend(uncond, cond, scale);
            assert!(out.iter().all(|&v| (v - want).abs() < 1e-4), "scale {scale}: {out:?}");
        }
    }

    /// Diffusers' tokenizer: BOS, the prompt, then EOS padding to 77; the
    /// empty prompt is BOS then EOS all the way.
    #[test]
    fn clip_tokenize_matches_clip_convention() {
        let empty = clip_tokenize("");
        assert_eq!(empty.shape(), [1, CLIP_SEQ_LEN]);
        assert_eq!(empty[(0, 0)], CLIP_BOS as i32);
        assert!((1..CLIP_SEQ_LEN).all(|i| empty[(0, i)] == CLIP_EOS as i32), "EOS padding");
        let toks = clip_tokenize("a photo of a cat");
        assert_eq!(toks.shape(), [1, CLIP_SEQ_LEN]);
        assert_eq!((toks[(0, 0)], toks[(0, CLIP_SEQ_LEN - 1)]), (CLIP_BOS as i32, CLIP_EOS as i32));
        assert_ne!(toks, empty, "a prompt tokenizes differently from none");
    }

    /// Components are 4-connected runs of pixels above 127, each with its
    /// absolute bounding box, in scan order.
    #[test]
    fn mask_components_finds_4_connected_regions() {
        let comps = |(w, h): (u32, u32), on: &[(u32, u32, u8)]| {
            let mut m = GrayImage::new(w, h);
            for &(x, y, v) in on {
                m.put_pixel(x, y, Luma([v]));
            }
            mask_components(&m)
        };
        let bbox = |x_min, y_min, x_max, y_max| MaskBbox { x_min, y_min, x_max, y_max };
        assert!(comps((64, 64), &[]).is_empty(), "an empty mask");
        let two = comps((64, 64), &[(5, 5, 255), (6, 5, 255), (5, 6, 255), (50, 55, 255), (51, 55, 255)]);
        assert_eq!(two, [bbox(5, 5, 6, 6), bbox(50, 55, 51, 55)], "disjoint, in scan order");
        assert_eq!(comps((16, 16), &[(2, 2, 255), (3, 2, 255), (3, 3, 255), (3, 4, 255)]), [bbox(2, 2, 3, 4)], "an L joined by shared edges");
        assert_eq!(comps((16, 16), &[(2, 2, 255), (3, 3, 255)]).len(), 2, "diagonal neighbours stay apart");
        assert_eq!(comps((16, 16), &[(5, 5, 100), (8, 8, 200)]), [bbox(8, 8, 8, 8)], "below the threshold is no mask");
        // The search walks box-local coordinates; the box comes back absolute.
        let far = comps((2048, 2048), &[(1900, 1850, 200), (1901, 1850, 200)]);
        assert_eq!(far, [bbox(1900, 1850, 1901, 1850)], "a tiny stroke far from the origin");
    }

    #[test]
    fn tile_count_grows_with_span_at_step_granularity() {
        // Past one tile the last tile's right edge meets the span's end.
        for (span, tiles) in [(100, 1), (SD_TILE - 1, 1), (SD_TILE, 1), (SD_TILE + 1, 2), (SD_TILE + TILE_STEP_PX, 2), (SD_TILE + TILE_STEP_PX + 1, 3)] {
            assert_eq!(tile_count(span), tiles, "span {span}");
        }
    }

    /// A crop is centred on its region and slides to stay inside the
    /// photo; a photo smaller than a tile is the whole crop.
    #[test]
    fn compute_sd_crop_centres_on_the_region_inside_the_photo() {
        let crop = |(x_min, y_min, x_max, y_max): (u32, u32, u32, u32), w: u32, h: u32| {
            compute_sd_crop(&MaskBbox { x_min, y_min, x_max, y_max }, w, h, SD_TILE).expect("a crop")
        };
        let c = 1050 - SD_TILE / 2;
        assert_eq!(crop((1000, 1000, 1100, 1100), 4096, 4096), (c, c, SD_TILE, SD_TILE), "centred on (1050, 1050)");
        let (x, y, w, h) = crop((3900, 100, 3990, 200), 4096, 4096);
        assert_eq!((x + w, w, h), (4096, SD_TILE, SD_TILE), "ends at the right edge");
        assert!(y < 100);
        assert_eq!(crop((5, 10, 50, 60), 4096, 4096).0, 0, "the centre less half a tile saturates at 0");
        assert_eq!(crop((5, 10, 50, 60), 4096, 4096).1, 0);
        assert_eq!(crop((10, 10, 100, 100), 200, 300), (0, 0, 200, 300), "a small photo is the crop");
    }

    /// The sweep drops entries idle past the limit, keeps pinned ones and
    /// ones just under it, and lets go of the payload it held.
    #[test]
    fn sweep_idle_drops_only_stale_unpinned_entries() {
        use prunr_models::ModelId;
        let now = sweep_now();
        let ago = |s| now - Duration::from_secs(s);
        let payload = Arc::new(());
        let freed = Arc::downgrade(&payload);
        let mut cache: HashMap<ModelId, CacheEntry<Arc<()>>> = HashMap::new();
        cache.insert(ModelId::SdV15InpaintFp16, CacheEntry::new(payload, ago(600)));
        cache.insert(ModelId::LaMaFp32, CacheEntry::new(Arc::new(()), ago(299)));
        let mut pinned = CacheEntry::new(Arc::new(()), ago(600));
        pinned.pinned = true;
        cache.insert(ModelId::Migan, pinned);
        assert_eq!(sweep_idle(&mut cache, now, Duration::from_secs(300)), 1);
        assert!(!cache.contains_key(&ModelId::SdV15InpaintFp16), "stale");
        assert!(cache.contains_key(&ModelId::LaMaFp32), "just under the limit");
        assert!(cache.contains_key(&ModelId::Migan), "pinned");
        // The cache held the only strong ref; a caller mid-run would own a clone.
        assert!(freed.upgrade().is_none(), "the evicted payload is freed");
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
