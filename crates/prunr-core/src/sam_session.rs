//! The Magic Brush's ORT sessions (SAM 2.1) and their runs; the tensors
//! in and out are `crate::sam`'s. The encoder goes to the GPU: on an RTX 3060 it encodes
//! a photo in 0.12 s against the CPU's 1.71 s, picking the same mask
//! candidate on 20 of 20 clicks (IoU 0.9998, summation order). The decoder
//! stays on the CPU, where a click decodes in under 0.1 s.

use std::borrow::Cow;
use std::sync::Mutex;

use image::RgbaImage;
use ort::session::Session;
use prunr_models::ModelId;

use crate::engine::{EpSite, Miss, SessionPick, Weights, CPU_NAME};
use crate::sam::{prompt::SamPrompt, SamDecoderOutput, SamEmbedding};

const ID: ModelId = ModelId::Sam21HieraSmall;
const ENCODER: &str = "encoder";
const DECODER: &str = "decoder";

/// The encoder's session on the first GPU provider that builds it and
/// passes a test run, else on the CPU; with where it landed.
pub fn encoder(cpu_only: bool) -> Result<(Session, SessionPick), String> {
    let bytes = part_bytes(ENCODER)?;
    let eps = crate::engine::gpu_eps_for(ID, cpu_only);
    let landed = crate::engine::walk_ladder("Magic Brush: ", ID, Weights::Fp32, &eps, |ep| {
        crate::cache::gc_stale_for_model(ID, ep.as_str());
        let builder = crate::ort_runtime::session_builder().map_err(|e| Miss::Unusable(e.to_string()))?;
        let (builder, graph, draft) = if ep.keeps_optimized_graph() {
            crate::engine::apply_ort_graph_cache(builder, &bytes, ID, ep.as_str(), Weights::Fp32, Some(ENCODER))
        } else {
            (builder, Cow::Borrowed(&bytes[..]), None)
        };
        let failed = |error| Miss::Failed {
            error,
            suspect_cache: crate::engine::cached_graph(matches!(graph, Cow::Owned(_)), ID, ep.as_str(), Weights::Fp32, Some(ENCODER)),
        };
        let mut built = ep.attach(builder, ID, EpSite::Selection { part: ENCODER })?;
        let mut session = built.commit_from_memory(&graph).map_err(|e| failed(format!("ORT session creation failed ({ep}): {e}")))?;
        crate::engine::kept_after_a_run(ID, ep, &mut session).map_err(failed)?;
        if let Some(draft) = draft {
            draft.publish();
        }
        tracing::info!(ep = %ep, "Magic Brush: encoder on the GPU");
        Ok(session)
    });
    let failed = match landed {
        Ok((session, ep, failed)) => return Ok((session, SessionPick { failed, ..SessionPick::on(ep.as_str()) })),
        Err(walk) => walk.failed,
    };
    Ok((on_cpu(&bytes, ENCODER)?, SessionPick { failed, ..SessionPick::on(CPU_NAME) }))
}

/// The decoder's session, on the CPU.
pub fn decoder() -> Result<Session, String> {
    on_cpu(&part_bytes(DECODER)?, DECODER)
}

fn on_cpu(bytes: &[u8], part: &str) -> Result<Session, String> {
    crate::cache::gc_stale_for_model(ID, CPU_NAME);
    crate::engine::commit_past_bad_cache(|| {
        let builder = crate::ort_runtime::session_builder().map_err(|e| (e.to_string(), None))?;
        let (mut builder, graph, draft) = crate::engine::apply_ort_graph_cache(builder, bytes, ID, CPU_NAME, Weights::Fp32, Some(part));
        let suspect = crate::engine::cached_graph(matches!(graph, Cow::Owned(_)), ID, CPU_NAME, Weights::Fp32, Some(part));
        let session = builder.commit_from_memory(&graph).map_err(|e| (format!("Could not load the Magic Brush model: {e}"), suspect))?;
        if let Some(draft) = draft {
            draft.publish();
        }
        Ok(session)
    })
}

fn part_bytes(part: &str) -> Result<std::sync::Arc<Vec<u8>>, String> {
    prunr_models::resolve_part_bytes(ID, part).map_err(|e| format!("Could not load the Magic Brush model ({part}): {e:?}"))
}

/// SAM 2 encoder + decoder sessions, built once and reused for every
/// click: the encoder on a GPU provider when there is one, the decoder
/// on the CPU.
pub struct Sessions {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    /// Where the encoder landed. On CUDA it hands its pool back after each
    /// encode: it held 2.1 GB of the card between photos, room an SD
    /// erase beside it needs.
    encoder_provider: String,
}

impl Sessions {
    pub fn new(encoder: Session, decoder: Session, encoder_provider: String) -> Self {
        Self { encoder: Mutex::new(encoder), decoder: Mutex::new(decoder), encoder_provider }
    }

    pub fn encoder_provider(&self) -> &str {
        &self.encoder_provider
    }

    pub fn encode(&self, source: &RgbaImage) -> Result<SamEmbedding, String> {
        use ort::{inputs, value::Tensor};

        // Preprocess before taking the lock: it runs on the rayon pool, and a
        // rayon wait while holding the lock can pick up a queued encode that
        // then blocks on this same lock. Do NOT move rayon work under it.
        let input_vec = crate::sam::preprocess::preprocess_for_sam(source);
        let mut session = self.encoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Shape: [1, 3, 1024, 1024] — NCHW, ImageNet-normalized f32.
        let arr = ndarray::Array4::from_shape_vec([1, 3, 1024, 1024], input_vec)
            .map_err(|e| format!("encoder input shape: {e}"))?;
        let tensor = Tensor::from_array(arr)
            .map_err(|e| format!("encoder input tensor: {e}"))?;

        let input_name = session.inputs()[0].name().to_string();
        let inputs = inputs![input_name => &tensor];
        let on_cuda = self.encoder_provider == crate::engine::CUDA_NAME;
        let outputs = match on_cuda.then(crate::engine::shrink_cuda_pool).flatten() {
            Some(options) => session.run_with_options(inputs, options),
            None => session.run(inputs),
        }
        .map_err(|e| format!("encoder session.run: {e}"))?;

        // By NAME: the export emits outputs in reverse of the documented
        // order. Shapes: image_embed [1,256,64,64], high_res_feats_0
        // [1,32,256,256], high_res_feats_1 [1,64,128,128].
        let image_embed = extract_f32(&outputs, "image_embed", "encoder")?;
        let high_res_feats_0 = extract_f32(&outputs, "high_res_feats_0", "encoder")?;
        let high_res_feats_1 = extract_f32(&outputs, "high_res_feats_1", "encoder")?;

        let embedding = SamEmbedding {
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
        };
        embedding.validate_shapes()
            .map_err(|e| format!("encoder output shape mismatch: {e}"))?;
        Ok(embedding)
    }

    pub fn decode(&self, embedding: &SamEmbedding, prompt: &SamPrompt) -> Result<SamDecoderOutput, String> {
        use ort::{inputs, value::TensorRef};

        let mut session = self.decoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

        // The 16 MB embedding is borrowed for the run — no per-click copy.
        let embed_tensor = TensorRef::from_array_view(
            ([1usize, 256, 64, 64], embedding.image_embed.as_slice()),
        ).map_err(|e| format!("decoder: image_embed tensor: {e}"))?;
        let feats0_tensor = TensorRef::from_array_view(
            ([1usize, 32, 256, 256], embedding.high_res_feats_0.as_slice()),
        ).map_err(|e| format!("decoder: high_res_feats_0 tensor: {e}"))?;
        let feats1_tensor = TensorRef::from_array_view(
            ([1usize, 64, 128, 128], embedding.high_res_feats_1.as_slice()),
        ).map_err(|e| format!("decoder: high_res_feats_1 tensor: {e}"))?;

        let n_points = prompt.point_labels.len();
        let coords_tensor = decoder_input("point_coords", [1, n_points, 2], prompt.point_coords.clone())?;
        let labels_tensor = decoder_input("point_labels", [1, n_points], prompt.point_labels.clone())?;
        let mask_tensor = decoder_input("mask_input", [1, 1, 256, 256], prompt.mask_input.clone())?;
        let has_mask_tensor = decoder_input("has_mask_input", [1], vec![prompt.has_mask_input])?;

        let outputs = session.run(inputs![
            "image_embed"       => embed_tensor,
            "high_res_feats_0"  => feats0_tensor,
            "high_res_feats_1"  => feats1_tensor,
            "point_coords"      => &coords_tensor,
            "point_labels"      => &labels_tensor,
            "mask_input"        => &mask_tensor,
            "has_mask_input"    => &has_mask_tensor,
        ]).map_err(|e| format!("decoder session.run: {e}"))?;

        // By name: masks [1, 3, 256, 256] and iou_predictions [1, 3].
        let masks = extract_f32(&outputs, "masks", "decoder")?;
        let iou_raw = extract_f32(&outputs, "iou_predictions", "decoder")?;

        if iou_raw.len() != 3 {
            return Err(format!("decoder: expected 3 IoU predictions, got {}", iou_raw.len()));
        }
        let iou_predictions = [iou_raw[0], iou_raw[1], iou_raw[2]];

        Ok(SamDecoderOutput { masks, iou_predictions })
    }

    /// `photo` encoded once, then each of `clicks` (fractions of its width
    /// and height) decoded: its best candidate's mask, 0 or 255 a cell, one
    /// after the other. For the bench.
    pub fn masks(&self, photo: &RgbaImage, clicks: &[(f32, f32)]) -> Result<Vec<u8>, String> {
        let embedding = self.encode(photo)?;
        let (w, h) = photo.dimensions();
        let mut masks = Vec::with_capacity(clicks.len() * crate::sam::SAM_MASK_CELLS);
        for &(fx, fy) in clicks {
            let prompt = crate::sam::prompt::build_click_prompt(fx * w as f32, fy * h as f32, w, h);
            let decoded = self.decode(&embedding, &prompt)?;
            masks.extend(decoded.mask(decoded.best()).iter().map(|&logit| if logit > 0.0 { 255 } else { 0 }));
        }
        Ok(masks)
    }
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    stage: &str,
) -> Result<Vec<f32>, String> {
    Ok(outputs[name]
        .try_extract_array::<f32>()
        .map_err(|e| format!("{stage}: extract {name}: {e}"))?
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset().0)
}

/// One decoder input of `shape`, named in its errors.
fn decoder_input<D: ndarray::Dimension + 'static>(
    name: &str,
    shape: impl ndarray::ShapeBuilder<Dim = D>,
    data: Vec<f32>,
) -> Result<ort::value::Tensor<f32>, String> {
    let array = ndarray::Array::from_shape_vec(shape, data).map_err(|e| format!("decoder: {name} shape: {e}"))?;
    ort::value::Tensor::from_array(array).map_err(|e| format!("decoder: {name} tensor: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a: a digest that stays the same across Rust releases.
    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    }

    /// A click on a disc and one on the ground around it, encoded and
    /// decoded on the CPU: each mask's selected cells and digest, pinned,
    /// so a change to how the sessions run shows as a changed mask.
    #[test]
    fn a_click_on_the_cpu_selects_the_mask_it_always_did() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let photo = RgbaImage::from_fn(96, 64, |x, y| {
            let inside = (x as f32 - 48.0).hypot(y as f32 - 32.0) < 20.0;
            image::Rgba(if inside { [40, 30, 90, 255] } else { [235, 225, 210, 255] })
        });
        let (encoder, pick) = encoder(true).expect("the encoder loads");
        assert_eq!(pick.provider, CPU_NAME);
        let sessions = Sessions::new(encoder, decoder().expect("the decoder loads"), pick.provider);
        let masks = sessions.masks(&photo, &[(0.5, 0.5), (0.1, 0.15)]).expect("the clicks decode");
        let pinned: Vec<(usize, u64)> = masks
            .chunks(crate::sam::SAM_MASK_CELLS)
            .map(|mask| (mask.iter().filter(|&&v| v == 255).count(), fnv1a(mask)))
            .collect();
        assert_eq!(pinned, [(13_418, 0xfc37_2ba9_1d56_3429), (50_067, 0xbe4f_a817_375b_9e96)]);
    }
}
