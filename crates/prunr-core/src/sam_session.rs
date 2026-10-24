//! The Magic Brush's ORT sessions (SAM 2.1); the tensors in and out are
//! `crate::sam`'s. The encoder goes to the GPU: on an RTX 3060 it encodes
//! a photo in 0.12 s against the CPU's 1.71 s, picking the same mask
//! candidate on 20 of 20 clicks (IoU 0.9998, summation order). The decoder
//! stays on the CPU, where a click decodes in under 0.1 s.

use std::borrow::Cow;

use ort::session::Session;
use prunr_models::ModelId;

use crate::engine::{EpSite, Miss, SessionPick, Weights, CPU_NAME};

const ID: ModelId = ModelId::Sam21HieraSmall;
const ENCODER: &str = "encoder";
const DECODER: &str = "decoder";

/// The encoder's session on the first GPU provider that builds it and
/// passes a test run, else on the CPU; with where it landed.
pub fn encoder(cpu_only: bool) -> Result<(Session, SessionPick), String> {
    let bytes = part_bytes(ENCODER)?;
    let eps = crate::engine::gpu_devices(crate::engine::gpu_eps_for(cpu_only), crate::engine::openvino_on_gpu());
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
        test_run(&mut session).map_err(|e| failed(format!("{ep} test run failed: {e}")))?;
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

/// One encode of a blank photo: a provider can build a session and still
/// fail its first run on an op.
fn test_run(session: &mut Session) -> Result<(), String> {
    let side = crate::sam::SAM_ENCODER_INPUT as usize;
    let input = ort::value::Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 3, side, side))).map_err(|e| e.to_string())?;
    let name = session.inputs()[0].name().to_string();
    session.run(ort::inputs![name.as_str() => &input]).map(drop).map_err(|e| e.to_string())
}
