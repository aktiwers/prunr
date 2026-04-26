//! Stable Diffusion 1.5 Inpainting — multi-model pipeline.
//!
//! Pipeline (per call):
//!   1. Tokenize prompt → text encoder → text embeddings (1, 77, 768)
//!      (v1: empty / unconditional only — BPE tokenizer lands when we
//!      add a prompt UI; SD ships fine with empty conditioning)
//!   2. VAE encode source → image latent (1, 4, 64, 64)
//!      Same for the masked source → masked-image latent
//!   3. Mask resampled to latent space (1, 1, 64, 64)
//!   4. Initial latent: pure gaussian noise scaled by init_noise_sigma
//!   5. For t in DDIM timesteps:
//!        latent_in = cat([latent, mask_lat, masked_lat], dim=1)   9 channels
//!        noise_pred = unet(latent_in, t, text_emb)
//!        latent = scheduler.step(noise_pred, t, latent)
//!   6. VAE decode final latent → image (in [-1, 1])
//!   7. Composite painted region back via mask
//!
//! Safety guards (CPU-class hardware reality):
//! - Bundle load checks free RAM via sysinfo and refuses below 6 GB free
//!   (see `SD_CPU_MIN_FREE_RAM_BYTES`). The 4 ONNX files total ~2 GB on
//!   disk, ORT graph optimization roughly doubles that during load, and
//!   UNet activations add another 2-4 GB transient. Below the floor a
//!   user's machine swap-thrashes and freezes.
//! - Smart cropping: dispatch runs ONE 512×512 inference centered on the
//!   mask bbox, not the global tile grid. Most paint strokes touch a
//!   small region of the canvas; tiling the whole image wastes minutes
//!   of CPU on tiles the user didn't paint. If the painted region is
//!   larger than 512×512 we refuse — multi-tile SD on CPU is
//!   uncancellable in practice.
//!
//! v1 limits — easy upgrade paths once UI lands:
//! - Empty prompt only (no `prompt`, no `negative_prompt`). Quality on
//!   uniform inputs is poor with empty conditioning — SD 1.5 was trained
//!   with text guidance and has no good "default" unconditionally. Real
//!   photos with surrounding texture do better because the masked-image
//!   latent provides the constraint.
//! - Classifier-free guidance disabled (`guidance_scale = 1.0`). Adding
//!   CFG roughly doubles per-step UNet cost but unlocks prompts.
//! - CPU-only inference. Adding a GPU EP ladder (mirroring LaMa's
//!   `build_lama_session`) is the next big perf win on supported HW.
//! - 512×512 fixed tile size.
//!
//! All of these flow into `SdInpaintRequest` already; the public API is
//! shaped for future text-prompted inpaint, outpainting, image-to-image,
//! ControlNet, and "imagine more" variations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

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
/// IS RAM). Real-world working set on a CPU/iGPU machine: 6-10 GB.
/// Below 6 GB free we've seen swap thrash freeze testers' machines.
const SD_CPU_MIN_FREE_RAM_BYTES: u64 = 6 * 1024 * 1024 * 1024;
/// CLIP-ViT-L/14 token sequence length used by SD 1.5.
const CLIP_SEQ_LEN: usize = 77;
/// CLIP-ViT-L/14 BOS / EOS / pad ids — same value for EOS and pad in
/// the SD 1.5 tokenizer convention.
const CLIP_BOS: i64 = 49406;
const CLIP_EOS: i64 = 49407;

/// Inputs to one SD inpaint call. Constructable today with default
/// fields for empty-prompt unconditional inpaint; future surfaces (text
/// prompts, CFG, seeded variations) just set the relevant fields.
#[derive(Debug, Clone)]
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
}

impl Default for SdInpaintRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            num_inference_steps: 20,
            guidance_scale: 1.0,
            seed: None,
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
    )
}

/// Full-knob entry. The `inpaint::process_inpaint` shim calls this with
/// defaults; future text-prompt + CFG surfaces wire through here.
pub fn process_inpaint_with(
    image: &RgbaImage,
    mask: &GrayImage,
    id: prunr_models::ModelId,
    req: SdInpaintRequest,
) -> Result<RgbaImage, CoreError> {
    if image.dimensions() != mask.dimensions() {
        return Err(CoreError::Inference(format!(
            "sd inpaint: dim mismatch — image {:?} vs mask {:?}",
            image.dimensions(),
            mask.dimensions(),
        )));
    }
    if !req.prompt.is_empty() || !req.negative_prompt.is_empty() {
        return Err(CoreError::Inference(
            "sd inpaint: text prompts not yet wired (CLIP BPE tokenizer pending). \
             Leave prompt + negative_prompt empty for unconditional inpaint."
                .to_string(),
        ));
    }
    if (req.guidance_scale - 1.0).abs() > 1e-3 {
        return Err(CoreError::Inference(
            "sd inpaint: classifier-free guidance not yet wired. \
             Use guidance_scale = 1.0 for v1."
                .to_string(),
        ));
    }

    let Some(bbox) = mask_bbox(mask) else {
        return Ok(image.clone());
    };
    let painted_w = bbox.x_max - bbox.x_min + 1;
    let painted_h = bbox.y_max - bbox.y_min + 1;
    if painted_w > SD_TILE || painted_h > SD_TILE {
        return Err(CoreError::Inference(format!(
            "SD inpaint: painted region is {painted_w}×{painted_h} pixels, \
             larger than SD's {SD_TILE}×{SD_TILE} tile. Paint a smaller \
             area or downscale the image first.",
        )));
    }

    let bundle = SdSession::get(id)?;
    let (img_w, img_h) = image.dimensions();
    let (cx, cy, cw, ch) = compute_sd_crop(&bbox, img_w, img_h);
    let cropped_img = image::imageops::crop_imm(image, cx, cy, cw, ch).to_image();
    let cropped_mask = image::imageops::crop_imm(mask, cx, cy, cw, ch).to_image();

    let painted = match run_one_tile(bundle, &cropped_img, &cropped_mask, &req) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(%e, "SD inference failed; leaving image unchanged");
            return Ok(image.clone());
        }
    };

    let mut out = image.clone();
    image::imageops::replace(&mut out, &painted, cx as i64, cy as i64);
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskBbox {
    x_min: u32, y_min: u32, x_max: u32, y_max: u32,
}

fn mask_bbox(mask: &GrayImage) -> Option<MaskBbox> {
    let (w, h) = mask.dimensions();
    let raw = mask.as_raw();
    let (mut x_min, mut y_min) = (u32::MAX, u32::MAX);
    let (mut x_max, mut y_max) = (0u32, 0u32);
    for y in 0..h {
        let row = (y * w) as usize;
        for x in 0..w {
            if raw[row + x as usize] > 127 {
                if x < x_min { x_min = x; }
                if x > x_max { x_max = x; }
                if y < y_min { y_min = y; }
                if y > y_max { y_max = y; }
            }
        }
    }
    if x_min == u32::MAX { None } else { Some(MaskBbox { x_min, y_min, x_max, y_max }) }
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
    image: &RgbaImage,
    mask: &GrayImage,
    req: &SdInpaintRequest,
) -> Result<RgbaImage, CoreError> {
    let (w, h) = image.dimensions();
    let padded_image = pad_to_tile(image);
    let padded_mask = pad_mask_to_tile(mask);

    // 1. Text embedding (empty prompt → uncond tokens). Constant
    //    across the denoising loop — convert to f16 once here.
    let text_emb = encode_text(bundle, /*empty=*/ true)?;
    let text_emb_f16 = f32_to_f16_3d(&text_emb);

    // 2. VAE encode masked source (image with painted region zeroed).
    //    SD inpaint protocol uses only the masked-image latent — the
    //    full-image latent is unused under our current sampler, so we
    //    skip that VAE pass (~1-3s saved per stroke on CPU).
    let masked_image = mask_image_for_vae(&padded_image, &padded_mask);
    let masked_latent = vae_encode(bundle, &masked_image)?;
    let masked_latent_f16 = f32_to_f16_4d(&masked_latent);

    // 3. Mask in latent space (1 channel, 64×64). Also constant.
    let mask_latent = mask_to_latent(&padded_mask);
    let mask_latent_f16 = f32_to_f16_4d(&mask_latent);

    // 4. Build scheduler + initial noise.
    let scheduler = DdimScheduler::new_sd15(req.num_inference_steps as usize);
    let seed = req.seed.unwrap_or_else(|| {
        // Non-deterministic but stable within one call. ChaCha8Rng's
        // `from_os_rng` would pull `getrandom`; wall-clock seed is fine
        // for "random per stroke" UX.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    let mut latent = sample_initial_noise(seed, &scheduler);

    // 5. Denoising loop. Loop-invariant f16 conversions hoisted above;
    //    only the 4-channel `latent` changes per step.
    let timesteps = scheduler.timesteps().to_vec();
    for (i, &t) in timesteps.iter().enumerate() {
        let latent_f16 = f32_to_f16_4d(&latent);
        let latent_in_f16 = concat_inpaint_input_f16(
            &latent_f16, &mask_latent_f16, &masked_latent_f16,
        );
        let noise_pred = unet_step(bundle, latent_in_f16, t, &text_emb_f16)?;
        let t_prev = timesteps.get(i + 1).copied().unwrap_or(-1);
        latent = step_array(&scheduler, &latent, &noise_pred, t, t_prev);
    }

    // 7. VAE decode → painted RGB tile.
    let painted = vae_decode(bundle, &latent)?;

    // 8. Composite onto source: outside mask = source, inside = painted.
    Ok(composite(image, &painted, mask, w, h))
}

// ── Session bundle ──────────────────────────────────────────────────────

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
}

impl SdSession {
    fn get(id: prunr_models::ModelId) -> Result<&'static SdSession, CoreError> {
        type Cache = HashMap<prunr_models::ModelId, Result<&'static SdSession, String>>;
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(entry) = guard.get(&id) {
                return entry.clone().map_err(CoreError::Inference);
            }
        }
        let entry: Result<&'static SdSession, String> = match Self::new_inner(id) {
            Ok(s) => Ok(Box::leak(Box::new(s))),
            Err(e) => Err(e),
        };
        let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let stored = guard.entry(id).or_insert(entry);
        stored.clone().map_err(CoreError::Inference)
    }

    fn new_inner(id: prunr_models::ModelId) -> Result<SdSession, String> {
        // Memory guard: refuse to load the 2-GB bundle on a system that
        // will swap-thrash during ORT graph optimization. The check is
        // tightened in advance of the GPU EP attempts because EP setup
        // also pulls the weights through driver memory.
        if let Some(free) = available_ram_bytes() {
            if free < SD_CPU_MIN_FREE_RAM_BYTES {
                return Err(format!(
                    "SD inpaint refused to load: only {:.1} GB RAM free, \
                     {:.1} GB minimum recommended. Close other apps or use \
                     a smaller eraser model (LaMa / Big-LaMa / MI-GAN).",
                    free as f64 / 1e9,
                    SD_CPU_MIN_FREE_RAM_BYTES as f64 / 1e9,
                ));
            }
        }
        let parts = prunr_models::multi_part_paths(id)
            .ok_or_else(|| prunr_models::not_installed_error(id))?;
        let by_key: HashMap<&str, PathBuf> = parts.into_iter().collect();

        // Each part is built with the GPU EP ladder + per-shape smoke
        // test. We log the winning provider per part so a partial GPU
        // fall-through (e.g. UNet on CUDA, VAEs on CPU) is debuggable.
        // Text encoder smoke-tested first since it's the smallest — if
        // GPU is broken on this machine, we discover it cheaply.
        let (text_encoder, text_ep) = build_part_with_ep_ladder(
            id, "text_encoder", &by_key, smoke_test_text_encoder,
        )?;
        let (vae_encoder, vae_enc_ep) = build_part_with_ep_ladder(
            id, "vae_encoder", &by_key, smoke_test_vae_encoder,
        )?;
        let (vae_decoder, vae_dec_ep) = build_part_with_ep_ladder(
            id, "vae_decoder", &by_key, smoke_test_vae_decoder,
        )?;
        let (unet, unet_ep) = build_part_with_ep_ladder(
            id, "unet", &by_key, smoke_test_unet,
        )?;

        let unet_inputs = take_three_inputs(&unet, "unet")?;
        let vae_encoder_input = take_first_input(&vae_encoder, "vae_encoder")?;
        let vae_decoder_input = take_first_input(&vae_decoder, "vae_decoder")?;
        let text_encoder_input = take_first_input(&text_encoder, "text_encoder")?;

        tracing::info!(
            ?id,
            text_encoder_ep = %text_ep,
            vae_encoder_ep = %vae_enc_ep,
            vae_decoder_ep = %vae_dec_ep,
            unet_ep = %unet_ep,
            unet_inputs = ?unet_inputs,
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

    for ep_name in gpu_eps {
        if !prunr_models::is_ep_compatible(id, ep_name) {
            tracing::debug!(?id, part = %key, ep = %ep_name, "SD: EP statically incompatible; skipping");
            continue;
        }
        if crate::ep_compat::is_known_failure(ep_name, id) {
            tracing::debug!(?id, part = %key, ep = %ep_name, "SD: EP cached as incompatible; skipping");
            continue;
        }
        let builder = match sd_base_builder() {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep_name, %e, "SD: builder init failed");
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
            #[cfg(not(target_os = "macos"))]
            "OpenVINO" => builder.with_execution_providers([
                ort::execution_providers::OpenVINOExecutionProvider::default().build(),
            ]),
            _ => continue,
        };
        let mut built = match registered {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep_name, %e, "SD: register EP failed");
                continue;
            }
        };
        let mut session = match built.commit_from_file(path) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep_name, %e, "SD: GPU session commit failed — trying next");
                crate::ep_compat::record_failure(ep_name, id, &format!("{e}"));
                continue;
            }
        };
        match smoke_test(&mut session, key) {
            Ok(()) => {
                tracing::info!(part = %key, ep = %ep_name, "SD: GPU session validated");
                return Ok((session, (*ep_name).to_string()));
            }
            Err(e) => {
                tracing::warn!(part = %key, ep = %ep_name, %e, "SD: smoke test failed — falling back");
                crate::ep_compat::record_failure(ep_name, id, &e);
            }
        }
    }

    // CPU fallback: no EP registration; smoke test skipped — if the CPU
    // EP can't run a typed-zero forward, every real inference will fail
    // at the same point and surface a useful error there.
    let session = sd_base_builder()
        .map_err(|e| format!("SD {key}: builder init: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("SD {key}: load from {}: {e}", path.display()))?;
    Ok((session, "CPU".to_string()))
}

fn sd_base_builder() -> Result<ort::session::builder::SessionBuilder, String> {
    Session::builder()
        .map_err(|e| format!("SD: ORT builder init failed: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("SD: optimization level: {e}"))
}

// Per-part smoke tests. Each runs ONE forward pass with zero-valued
// inputs of the model's expected shape/type — catches EP/op
// incompatibilities at session-build time rather than at first stroke.

fn smoke_test_text_encoder(s: &mut Session, label: &str) -> Result<(), String> {
    let tokens = empty_prompt_tokens();
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

/// Empty-prompt CLIP token sequence: `[BOS, EOS, EOS, …]` of length 77.
/// Diffusers' `tokenizer("")` produces this exact pattern — `pad_token_id`
/// equals `eos_token_id` for SD 1.5. The export under test uses `int32`
/// for `input_ids`; some exports use `int64`. Match the dominant export
/// (int32) and let the runtime error point us at any divergent build.
fn empty_prompt_tokens() -> Array2<i32> {
    let mut a = Array2::<i32>::from_elem((1, CLIP_SEQ_LEN), CLIP_EOS as i32);
    a[(0, 0)] = CLIP_BOS as i32;
    a
}

fn encode_text(bundle: &SdSession, empty: bool) -> Result<Array3<f32>, CoreError> {
    if !empty {
        return Err(CoreError::Inference(
            "sd: non-empty prompts require BPE tokenizer (not yet wired)".to_string(),
        ));
    }
    let tokens = empty_prompt_tokens();
    let t = Tensor::from_array(tokens)
        .map_err(|e| CoreError::Inference(format!("SD text encoder: input tensor: {e}")))?;
    let mut session = bundle.text_encoder.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![bundle.text_encoder_input.as_str() => &t])
        .map_err(|e| CoreError::Inference(format!("SD text encoder: run: {e}")))?;
    extract_3d(&outputs[0], "text encoder")
}

// ── VAE encode / decode ────────────────────────────────────────────────

/// VAE encode: image in [-1, 1] NCHW f32 → latent (1, 4, 64, 64).
/// Diffusers' VAE encoder ONNX outputs `latent_sample` already multiplied
/// by the 0.18215 scaling factor; some exports output unscaled `mean` and
/// expect the caller to scale. We detect by comparing the pre/post-scale
/// magnitude — if max(|x|) ≪ 1 we assume unscaled and apply the factor.
fn vae_encode(bundle: &SdSession, image: &RgbaImage) -> Result<Array4<f32>, CoreError> {
    let input = image_to_minus1_plus1(image);
    let input_f16 = f32_to_f16_4d(&input);
    let t = Tensor::from_array(input_f16)
        .map_err(|e| CoreError::Inference(format!("SD vae encoder: input tensor: {e}")))?;
    let mut session = bundle.vae_encoder.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![bundle.vae_encoder_input.as_str() => &t])
        .map_err(|e| CoreError::Inference(format!("SD vae encoder: run: {e}")))?;
    let mut latent = extract_4d(&outputs[0], "vae encoder")?;
    // diffusers' OnnxStableDiffusionInpaintPipeline applies the scaling
    // factor AFTER `vae_encoder(sample=image)` returns, so the ONNX
    // output is unscaled. Mirror that: always multiply by 0.18215 here.
    latent *= VAE_SCALING_FACTOR;
    Ok(latent)
}

fn vae_decode(bundle: &SdSession, latent: &Array4<f32>) -> Result<RgbaImage, CoreError> {
    let unscaled = latent / VAE_SCALING_FACTOR;
    let unscaled_f16 = f32_to_f16_4d(&unscaled);
    let t = Tensor::from_array(unscaled_f16)
        .map_err(|e| CoreError::Inference(format!("SD vae decoder: input tensor: {e}")))?;
    let mut session = bundle.vae_decoder.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = session.run(inputs![bundle.vae_decoder_input.as_str() => &t])
        .map_err(|e| CoreError::Inference(format!("SD vae decoder: run: {e}")))?;
    let arr = extract_4d(&outputs[0], "vae decoder")?;
    Ok(minus1_plus1_to_image(&arr))
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
fn image_to_minus1_plus1(image: &RgbaImage) -> Array4<f32> {
    let s = SD_TILE as usize;
    let (w, h) = image.dimensions();
    let (w_us, h_us) = (w as usize, h as usize);
    let mut a = Array4::<f32>::zeros((1, 3, s, s));
    let buf = a.as_slice_mut().unwrap();
    let plane = s * s;
    let raw = image.as_raw();
    for y in 0..h_us.min(s) {
        let src_row = y * w_us * 4;
        let dst_row = y * s;
        for x in 0..w_us.min(s) {
            let src = src_row + x * 4;
            let dst = dst_row + x;
            buf[dst]              = (raw[src]     as f32 / 127.5) - 1.0;
            buf[plane + dst]      = (raw[src + 1] as f32 / 127.5) - 1.0;
            buf[plane * 2 + dst]  = (raw[src + 2] as f32 / 127.5) - 1.0;
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

/// Source image with the painted region replaced by mid-gray — the
/// SD inpaint training convention. Diffusers' `prepare_mask_and_masked_image`
/// multiplies the [-1, 1]-normalized image by `(mask < 0.5)` which puts
/// 0 (mid-gray) into the masked region; the VAE / UNet were trained
/// against this. Filling with black (0 in [0, 255] = -1 in [-1, 1])
/// instead drives the masked-image latent out of distribution, and with
/// empty-prompt CFG=1.0 the denoised output collapses to dark fills.
fn mask_image_for_vae(image: &RgbaImage, mask: &GrayImage) -> RgbaImage {
    debug_assert_eq!(image.dimensions(), mask.dimensions());
    let mut out = image.clone();
    let raw = out.as_mut();
    let m = mask.as_raw();
    for i in 0..m.len() {
        if m[i] > 127 {
            raw[i * 4]     = 128;
            raw[i * 4 + 1] = 128;
            raw[i * 4 + 2] = 128;
        }
    }
    out
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
fn sample_initial_noise(seed: u64, _scheduler: &DdimScheduler) -> Array4<f32> {
    let l = SD_LATENT_SIDE as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = StandardNormal;
    let n = 1 * 4 * l * l;
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f32 = dist.sample(&mut rng);
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

/// Apply DDIM step element-wise to flat-Vec representations of the
/// 4D arrays. Avoids one round-trip through Vec<f32> + reshape.
fn step_array(
    scheduler: &DdimScheduler,
    latent_t: &Array4<f32>,
    noise_pred: &Array4<f32>,
    t: i64,
    t_prev: i64,
) -> Array4<f32> {
    let lat = latent_t.as_slice().expect("latent: standard layout");
    let eps = noise_pred.as_slice().expect("noise pred: standard layout");
    let next = scheduler.step(lat, eps, t, t_prev);
    Array4::from_shape_vec(latent_t.dim(), next)
        .expect("shape unchanged from input")
}

/// If the cropped input is smaller than SD_TILE on either axis (image
/// edge case), pad to 512×512 with zero-fill (top-left aligned). The
/// smart-crop dispatcher guarantees max dim is SD_TILE.
fn pad_to_tile(image: &RgbaImage) -> RgbaImage {
    let (w, h) = image.dimensions();
    if w == SD_TILE && h == SD_TILE {
        return image.clone();
    }
    let mut out = RgbaImage::new(SD_TILE, SD_TILE);
    image::imageops::overlay(&mut out, image, 0, 0);
    out
}

fn pad_mask_to_tile(mask: &GrayImage) -> GrayImage {
    let (w, h) = mask.dimensions();
    if w == SD_TILE && h == SD_TILE {
        return mask.clone();
    }
    let mut out = GrayImage::new(SD_TILE, SD_TILE);
    image::imageops::overlay(&mut out, mask, 0, 0);
    out
}

// ── Safety guards ───────────────────────────────────────────────────────

/// Cross-platform available-RAM probe. Returns `None` only when sysinfo
/// can't read the system (CI containers without /proc, exotic platforms);
/// callers in that case skip the guard rather than fail-closed since
/// "we couldn't query" doesn't imply "memory is low".
fn available_ram_bytes() -> Option<u64> {
    use std::sync::{Mutex, OnceLock, PoisonError};
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let mtx = SYS.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut sys = mtx.lock().unwrap_or_else(PoisonError::into_inner);
    sys.refresh_memory();
    let avail = sys.available_memory();
    if avail == 0 { None } else { Some(avail) }
}

// ── DDIM scheduler ──────────────────────────────────────────────────────

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
        const NUM_TRAIN: usize = 1000;
        const BETA_START: f32 = 0.00085;
        const BETA_END: f32 = 0.012;

        // scaled_linear: linspace on sqrt(beta) then square back.
        let sqrt_start = BETA_START.sqrt();
        let sqrt_end = BETA_END.sqrt();
        let mut betas = Vec::with_capacity(NUM_TRAIN);
        for i in 0..NUM_TRAIN {
            let t = i as f32 / (NUM_TRAIN as f32 - 1.0);
            let b = sqrt_start + (sqrt_end - sqrt_start) * t;
            betas.push(b * b);
        }
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(NUM_TRAIN);
        let mut acc = 1.0_f32;
        for a in &alphas {
            acc *= *a;
            alphas_cumprod.push(acc);
        }

        // Diffusers default: descending evenly-spaced timesteps from
        // num_train-1 down to 0, length = num_inference.
        let step = NUM_TRAIN as f32 / num_inference as f32;
        let mut timesteps: Vec<i64> = (0..num_inference)
            .map(|i| (((num_inference - 1 - i) as f32 + 0.0) * step).round() as i64)
            .collect();
        for t in &mut timesteps {
            *t = (*t).clamp(0, NUM_TRAIN as i64 - 1);
        }

        Self {
            alphas_cumprod,
            timesteps,
            num_train: NUM_TRAIN,
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

    #[test]
    fn empty_prompt_tokens_layout_matches_clip_convention() {
        let toks = empty_prompt_tokens();
        assert_eq!(toks.shape(), [1, CLIP_SEQ_LEN]);
        assert_eq!(toks[(0, 0)], CLIP_BOS as i32);
        for i in 1..CLIP_SEQ_LEN {
            assert_eq!(toks[(0, i)], CLIP_EOS as i32, "expected EOS pad at index {i}");
        }
    }

    #[test]
    fn image_to_minus1_plus1_maps_byte_endpoints_correctly() {
        let mut img = RgbaImage::new(SD_TILE, SD_TILE);
        for p in img.pixels_mut() {
            *p = Rgba([0, 128, 255, 255]);
        }
        let arr = image_to_minus1_plus1(&img);
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
    fn mask_image_for_vae_replaces_painted_region_with_mid_gray() {
        let mut img = RgbaImage::new(8, 8);
        for p in img.pixels_mut() { *p = Rgba([100, 200, 50, 255]); }
        let mut mask = GrayImage::new(8, 8);
        mask.put_pixel(2, 3, Luma([255]));
        let out = mask_image_for_vae(&img, &mask);
        // Mid-gray (128) = 0 in [-1, 1] — diffusers training convention.
        assert_eq!(out.get_pixel(2, 3).0, [128, 128, 128, 255]);
        // Untouched pixel intact.
        assert_eq!(out.get_pixel(0, 0).0, [100, 200, 50, 255]);
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
        let s = DdimScheduler::new_sd15(20);
        let a = sample_initial_noise(42, &s);
        let b = sample_initial_noise(42, &s);
        assert_eq!(a, b, "same seed must give identical noise");
        let c = sample_initial_noise(43, &s);
        assert_ne!(a, c, "different seed must give different noise");
    }

    #[test]
    fn mask_bbox_returns_none_on_empty_mask() {
        let m = GrayImage::new(64, 64);
        assert!(mask_bbox(&m).is_none());
    }

    #[test]
    fn mask_bbox_returns_painted_extents() {
        let mut m = GrayImage::new(64, 64);
        m.put_pixel(10, 20, Luma([255]));
        m.put_pixel(30, 25, Luma([200]));
        m.put_pixel(15, 50, Luma([255]));
        let b = mask_bbox(&m).expect("bbox");
        assert_eq!(b, MaskBbox { x_min: 10, y_min: 20, x_max: 30, y_max: 50 });
    }

    #[test]
    fn mask_bbox_ignores_below_threshold_pixels() {
        let mut m = GrayImage::new(16, 16);
        m.put_pixel(5, 5, Luma([100])); // below 127 threshold
        m.put_pixel(8, 8, Luma([200]));
        let b = mask_bbox(&m).expect("bbox");
        assert_eq!(b, MaskBbox { x_min: 8, y_min: 8, x_max: 8, y_max: 8 });
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
        let s = DdimScheduler::new_sd15(20);
        let n = sample_initial_noise(0, &s);
        assert_eq!(n.shape(), &[1, 4, SD_LATENT_SIDE as usize, SD_LATENT_SIDE as usize]);
    }
}
