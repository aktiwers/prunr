//! Animation-sweep export: render N frames of a processed item while
//! sweeping one knob, write them as `0000.png`, `0001.png`, ….
//!
//! Every frame is Tier 2/3 work over the cached seg + edge tensors — no
//! re-inference, so a 60-frame sweep is a handful of seconds on the worker
//! thread.

use std::path::PathBuf;
use std::sync::mpsc;

use image::{DynamicImage, RgbaImage};
use prunr_core::{
    BgEffect, ComposeMode, EdgeInferenceResult, EdgeSettings, FillStyle, LineMode, LineStyle,
    MaskSettings,
};

use super::item::BatchItem;
use super::item_settings::ItemSettings;
use super::worker::{EdgeBundle, SegBundle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepKnob {
    LineStrength,
    Gamma,
    ComposeModeCycle,
    LineStyleCycle,
    FillStyleCycle,
}

impl SweepKnob {
    pub const ALL: [Self; 5] = [
        Self::LineStrength,
        Self::Gamma,
        Self::ComposeModeCycle,
        Self::LineStyleCycle,
        Self::FillStyleCycle,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::LineStrength => "Line strength",
            Self::Gamma => "Gamma",
            Self::ComposeModeCycle => "Compose mode (cycle)",
            Self::LineStyleCycle => "Line style (cycle)",
            Self::FillStyleCycle => "Fill style (cycle)",
        }
    }

    /// Cycle knobs run one frame per enum variant; numeric sweeps lerp
    /// across a user-picked frame count.
    pub fn cycle_len(self) -> Option<usize> {
        match self {
            Self::LineStrength | Self::Gamma => None,
            Self::ComposeModeCycle => Some(ComposeMode::ALL.len()),
            Self::LineStyleCycle => Some(LineStyle::ALL.len()),
            Self::FillStyleCycle => Some(FillStyle::ALL.len()),
        }
    }
}

pub fn apply_frame(base: &ItemSettings, knob: SweepKnob, i: usize, total: usize) -> ItemSettings {
    let mut s = *base;
    match knob {
        SweepKnob::LineStrength => {
            s.line_strength = lerp(0.1, 0.9, i, total);
        }
        SweepKnob::Gamma => {
            s.gamma = lerp(0.5, 2.0, i, total);
        }
        SweepKnob::ComposeModeCycle => {
            s.compose_mode = ComposeMode::ALL[i % ComposeMode::ALL.len()];
        }
        SweepKnob::LineStyleCycle => {
            s.line_style = LineStyle::ALL[i % LineStyle::ALL.len()];
        }
        SweepKnob::FillStyleCycle => {
            s.fill_style = FillStyle::ALL[i % FillStyle::ALL.len()];
        }
    }
    s
}

fn lerp(start: f32, end: f32, i: usize, total: usize) -> f32 {
    if total <= 1 {
        return start;
    }
    let t = i as f32 / (total - 1) as f32;
    start + t * (end - start)
}

pub enum SweepEvent {
    Progress { done: usize, total: usize },
    Finished { frames_written: usize, dir: PathBuf },
    Failed(String),
}

pub struct SweepRequest {
    pub out_dir: PathBuf,
    pub knob: SweepKnob,
    pub frames: usize,
    pub source: DynamicImage,
    pub base_settings: ItemSettings,
    pub seg: Option<SegBundle>,
    pub edges: Option<EdgeBundle>,
    pub events: mpsc::Sender<SweepEvent>,
}

impl SweepRequest {
    /// Build a request from the selected item + UI form state. Decodes the
    /// source and decompresses cached tensors up-front so the spawned thread
    /// never touches `BatchItem`. Returns user-facing errors for the toast.
    pub(crate) fn from_item(
        item: &BatchItem,
        ui: &SweepUiState,
        events: mpsc::Sender<SweepEvent>,
    ) -> Result<Self, String> {
        let out_dir = ui.out_dir.clone().ok_or_else(|| "Pick an output folder first".to_string())?;
        let bytes = item.source.load_bytes().map_err(|e| format!("Source read failed: {e}"))?;
        let source = prunr_core::load_image_from_bytes(&bytes)
            .map_err(|e| format!("Decode failed: {e}"))?;
        let seg = item.cached_tensor.as_ref().and_then(|ct| ct.bundle());
        let edges = item.cached_edge_tensors.as_ref().and_then(|et| et.bundle_all());
        Ok(Self {
            out_dir,
            knob: ui.knob,
            frames: ui.effective_frames(),
            source,
            base_settings: item.settings,
            seg,
            edges,
            events,
        })
    }
}

pub struct SweepUiState {
    pub knob: SweepKnob,
    pub frames: usize,
    pub out_dir: Option<PathBuf>,
}

impl Default for SweepUiState {
    fn default() -> Self {
        Self { knob: SweepKnob::Gamma, frames: 12, out_dir: None }
    }
}

impl SweepUiState {
    pub fn effective_frames(&self) -> usize {
        self.knob.cycle_len().unwrap_or(self.frames)
    }

    pub fn is_ready(&self) -> bool {
        self.out_dir.is_some() && self.effective_frames() >= 1
    }
}

pub fn spawn_sweep(req: SweepRequest) {
    std::thread::Builder::new()
        .name("prunr-sweep".into())
        .spawn(move || run_sweep(req))
        .expect("failed to spawn sweep thread");
}

fn run_sweep(req: SweepRequest) {
    let total = req.frames;
    let events = req.events.clone();

    if let Err(err) = std::fs::create_dir_all(&req.out_dir) {
        let _ = events.send(SweepEvent::Failed(format!("create dir: {err}")));
        return;
    }

    // Build the borrow-ready EdgeInferenceResult once; `compose_subject_outline`
    // only reads it, so one clone amortises across all frames instead of one
    // per frame (~5 MB × frame count).
    let edge_res = req.edges.as_ref().map(|e| EdgeInferenceResult {
        tensors: e.tensors.clone(),
        height: e.height,
        width: e.width,
    });

    let mut written = 0;
    for i in 0..total {
        let settings = apply_frame(&req.base_settings, req.knob, i, total);
        let rgba = match render_frame(&req.source, &settings, req.seg.as_ref(), edge_res.as_ref()) {
            Ok(r) => r,
            Err(err) => {
                let _ = events.send(SweepEvent::Failed(format!("frame {i}: {err}")));
                return;
            }
        };
        let bytes = match prunr_core::encode_rgba_png(&rgba) {
            Ok(b) => b,
            Err(err) => {
                let _ = events.send(SweepEvent::Failed(format!("encode frame {i}: {err}")));
                return;
            }
        };
        let path = req.out_dir.join(format!("{i:04}.png"));
        if let Err(err) = std::fs::write(&path, &bytes) {
            let _ = events.send(SweepEvent::Failed(format!("write {}: {err}", path.display())));
            return;
        }
        written += 1;
        // Stop if the UI closed mid-sweep — receiver gone means no one cares
        // about the remaining frames. Avoids orphaning PNG writes.
        if events.send(SweepEvent::Progress { done: written, total }).is_err() {
            return;
        }
    }

    let _ = events.send(SweepEvent::Finished { frames_written: written, dir: req.out_dir });
}

fn render_frame(
    source: &DynamicImage,
    settings: &ItemSettings,
    seg: Option<&SegBundle>,
    edge_res: Option<&EdgeInferenceResult>,
) -> Result<RgbaImage, String> {
    let mask_settings = settings.mask_settings();
    let edge_settings = settings.edge_settings();

    match settings.line_mode {
        LineMode::Off => render_off(source, &mask_settings, seg),
        LineMode::EdgesOnly => render_edges_only(source, &edge_settings, edge_res),
        LineMode::SubjectOutline => {
            render_subject_outline(source, &mask_settings, &edge_settings, seg, edge_res)
        }
    }
}

fn render_off(
    source: &DynamicImage,
    mask: &MaskSettings,
    seg: Option<&SegBundle>,
) -> Result<RgbaImage, String> {
    let Some(seg) = seg else {
        return Err("LineMode::Off needs a cached seg tensor".into());
    };
    prunr_core::postprocess_from_flat(
        &seg.data, seg.height as usize, seg.width as usize,
        source, mask, seg.model,
    )
    .map_err(|e| e.to_string())
}

fn render_edges_only(
    source: &DynamicImage,
    edge: &EdgeSettings,
    edge_res: Option<&EdgeInferenceResult>,
) -> Result<RgbaImage, String> {
    let Some(edge_res) = edge_res else {
        return Err("LineMode::EdgesOnly needs a cached edge tensor".into());
    };
    let active = &edge_res.tensors[edge.edge_scale as usize];
    Ok(prunr_core::finalize_edges(active, edge_res.height, edge_res.width, source, edge))
}

fn render_subject_outline(
    source: &DynamicImage,
    mask: &MaskSettings,
    edge: &EdgeSettings,
    seg: Option<&SegBundle>,
    edge_res: Option<&EdgeInferenceResult>,
) -> Result<RgbaImage, String> {
    let seg = seg.ok_or("SubjectOutline needs a cached seg tensor")?;
    let edge_res = edge_res.ok_or("SubjectOutline needs a cached edge tensor")?;
    // Subject layer renders with transparent bg (no fill/bg effect) — the
    // compose pass needs the silhouette, not a decorated backdrop. Any
    // user-configured effect gets re-applied via the compose path itself.
    let subject_mask = MaskSettings {
        fill_style: FillStyle::None,
        bg_effect: BgEffect::None,
        ..*mask
    };
    let masked = prunr_core::postprocess_from_flat(
        &seg.data, seg.height as usize, seg.width as usize,
        source, &subject_mask, seg.model,
    )
    .map_err(|e| e.to_string())?;
    Ok(prunr_core::compose_subject_outline(edge_res, &masked, edge))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_knobs_have_fixed_len_matching_enum() {
        assert_eq!(SweepKnob::ComposeModeCycle.cycle_len(), Some(ComposeMode::ALL.len()));
        assert_eq!(SweepKnob::LineStyleCycle.cycle_len(), Some(LineStyle::ALL.len()));
        assert_eq!(SweepKnob::FillStyleCycle.cycle_len(), Some(FillStyle::ALL.len()));
    }

    #[test]
    fn numeric_knobs_have_no_cycle_len() {
        assert_eq!(SweepKnob::LineStrength.cycle_len(), None);
        assert_eq!(SweepKnob::Gamma.cycle_len(), None);
    }

    #[test]
    fn line_strength_sweep_endpoints() {
        let base = ItemSettings::default();
        let first = apply_frame(&base, SweepKnob::LineStrength, 0, 10);
        let last = apply_frame(&base, SweepKnob::LineStrength, 9, 10);
        assert!((first.line_strength - 0.1).abs() < 1e-5);
        assert!((last.line_strength - 0.9).abs() < 1e-5);
    }

    #[test]
    fn gamma_sweep_endpoints() {
        let base = ItemSettings::default();
        let first = apply_frame(&base, SweepKnob::Gamma, 0, 5);
        let last = apply_frame(&base, SweepKnob::Gamma, 4, 5);
        assert!((first.gamma - 0.5).abs() < 1e-5);
        assert!((last.gamma - 2.0).abs() < 1e-5);
    }

    #[test]
    fn single_frame_sweep_lands_on_start() {
        let base = ItemSettings::default();
        let only = apply_frame(&base, SweepKnob::LineStrength, 0, 1);
        assert!((only.line_strength - 0.1).abs() < 1e-5);
    }

    #[test]
    fn compose_mode_cycle_walks_all_variants_in_order() {
        let base = ItemSettings::default();
        for (i, expected) in ComposeMode::ALL.iter().enumerate() {
            let s = apply_frame(&base, SweepKnob::ComposeModeCycle, i, ComposeMode::ALL.len());
            assert_eq!(s.compose_mode, *expected);
        }
    }

    #[test]
    fn cycle_wraps_when_index_exceeds_variant_count() {
        let base = ItemSettings::default();
        let first = apply_frame(&base, SweepKnob::ComposeModeCycle, 0, 999);
        let wrapped = apply_frame(&base, SweepKnob::ComposeModeCycle, ComposeMode::ALL.len(), 999);
        assert_eq!(first.compose_mode, wrapped.compose_mode);
    }

    #[test]
    fn apply_frame_preserves_untouched_fields() {
        let mut base = ItemSettings::default();
        base.gamma = 1.75;
        base.edge_thickness = 3;
        let frame = apply_frame(&base, SweepKnob::LineStrength, 5, 10);
        assert_eq!(frame.gamma, 1.75);
        assert_eq!(frame.edge_thickness, 3);
    }
}
