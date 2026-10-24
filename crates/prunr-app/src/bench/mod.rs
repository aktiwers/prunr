//! `prunr bench`: each installed model run on the CPU provider and through
//! the GPU provider ladder the app uses, timed, with the two outputs
//! compared. One model at a time, its sessions built one after the other
//! (DirectML races concurrent builds) and released before the next model.
//!
//! The runs go through the same core entry points as the app, so a
//! mismatch here is a mismatch a user sees.

pub mod check;
pub mod input;
pub mod needs;
pub mod report;
pub mod stats;

use std::time::Instant;

use prunr_core::engine::{SessionPick, Skip};
use prunr_core::memory;
use prunr_models::{ModelCategory, ModelId};
use serde::Serialize;

pub use input::Inputs;
use stats::{Agreement, Timing, Tolerance};

/// The models only `--heavy` runs: multi-GB downloads that take minutes
/// on a CPU.
pub const HEAVY: [ModelId; 4] = [
    ModelId::SdV15InpaintFp16,
    ModelId::SdV15LcmInpaintFp16,
    ModelId::RoremFp16,
    ModelId::Nomos8kSchatL,
];

/// The diffusion erasers fill from this seed on both sides, so their
/// outputs can be compared.
const SEED: u64 = 42;

/// The provider name a session that fell back reports.
const CPU: &str = "CPU";

/// What a model makes, which decides how it runs and what is compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Segmentation,
    Upscale,
    Colorize,
    Depth,
    /// LaMa, Big-LaMa, MI-GAN.
    Eraser,
    /// The Stable Diffusion erasers.
    Diffusion,
    /// The Magic Brush.
    Selection,
    /// The lines (TEED). The app runs it on the CPU; the bench measures
    /// its GPU path, as for the face models.
    Lines,
    /// The face finder (YuNet).
    FaceFinder,
    /// The face restorers' networks (GFPGAN, RestoreFormer++).
    FaceRedraw,
}

impl Family {
    /// `None` for TAESD, which runs inside a Stable Diffusion erase.
    pub fn of(id: ModelId) -> Option<Self> {
        match Placement::of(id)? {
            Placement::Family(family) => Some(family),
            Placement::InsideDiffusion => None,
        }
    }

    pub fn compared_on(self) -> &'static str {
        match self {
            Self::Segmentation => "the cut-out's alpha",
            Self::Upscale | Self::Colorize => "RGB",
            Self::Depth => "the depth map",
            Self::Eraser | Self::Diffusion => "RGB inside the erased square",
            Self::Selection => "the masks of eight clicks",
            Self::Lines => "the line drawing",
            Self::FaceFinder => "the faces found (score, box)",
            Self::FaceRedraw => "RGB of a redrawn 512 px crop",
        }
    }

    /// Masks and pictures differ by a level or two between FP16 and FP32
    /// weights and between providers' kernels; the erasers invent what
    /// they fill, so small numeric drift moves more of it; diffusion
    /// compounds drift over its steps, so only a gross failure counts.
    pub fn tolerance(self) -> Tolerance {
        match self {
            Self::Segmentation => Tolerance { mean_abs: 3.0, over_2: 0.10 },
            Self::Upscale | Self::Colorize => Tolerance { mean_abs: 2.0, over_2: 0.10 },
            Self::Depth => Tolerance { mean_abs: 3.0, over_2: 0.15 },
            Self::Eraser => Tolerance { mean_abs: 4.0, over_2: 0.25 },
            Self::Diffusion => Tolerance { mean_abs: 24.0, over_2: 0.90 },
            // A cell is in or out: 0.5% of cells flipped across the clicks
            // (the CPU's masks on CUDA and on a Mac's GPU flipped under 0.1%).
            Self::Selection => Tolerance { mean_abs: 1.3, over_2: 0.005 },
            Self::Lines => Tolerance { mean_abs: 3.0, over_2: 0.10 },
            Self::FaceFinder | Self::FaceRedraw => Tolerance { mean_abs: 2.0, over_2: 0.10 },
        }
    }

    pub const ALL: [Family; 10] = [
        Self::Segmentation, Self::Upscale, Self::Colorize, Self::Depth, Self::Eraser, Self::Diffusion, Self::Selection,
        Self::Lines, Self::FaceFinder, Self::FaceRedraw,
    ];
}

/// Where a model sits in the bench, from one match over its category.
enum Placement {
    Family(Family),
    /// TAESD, run only as part of a Stable Diffusion erase.
    InsideDiffusion,
}

impl Placement {
    fn of(id: ModelId) -> Option<Self> {
        Some(match prunr_models::descriptor(id)?.category {
            ModelCategory::Segmentation => Self::Family(Family::Segmentation),
            ModelCategory::Upscale => Self::Family(Family::Upscale),
            ModelCategory::Colorize => Self::Family(Family::Colorize),
            ModelCategory::Depth => Self::Family(Family::Depth),
            ModelCategory::Inpaint if id.is_sd_family() => Self::Family(Family::Diffusion),
            ModelCategory::Inpaint if id == ModelId::TaesdFp16 => Self::InsideDiffusion,
            ModelCategory::Inpaint => Self::Family(Family::Eraser),
            ModelCategory::Selection => Self::Family(Family::Selection),
            ModelCategory::EdgeDetection => Self::Family(Family::Lines),
            ModelCategory::Face if id == ModelId::YuNet => Self::Family(Family::FaceFinder),
            ModelCategory::Face => Self::Family(Family::FaceRedraw),
        })
    }
}

fn in_catalog(id: ModelId) -> bool {
    Family::of(id).is_some()
}

/// The names `--models` takes: each model's stable name.
pub fn known_names() -> Vec<&'static str> {
    ModelId::ALL.iter().copied().filter(|&id| in_catalog(id)).map(|id| id.stable_name()).collect()
}

/// `name` as `--models` spells it: a stable name, any case, `-` for `_`.
pub fn model_named(name: &str) -> Option<ModelId> {
    let wanted = name.trim().to_ascii_lowercase().replace('-', "_");
    ModelId::ALL.iter().copied().find(|&id| in_catalog(id) && id.stable_name() == wanted)
}

/// The models to run: `names` in their order, or every installed one
/// (by `installed`) but the heavy ones unless `heavy`. A named heavy model
/// needs `heavy` too, so it never starts a multi-GB run by surprise.
pub fn select(names: &[String], heavy: bool, installed: impl Fn(ModelId) -> bool) -> Result<Vec<ModelId>, String> {
    let named: Vec<&String> = names.iter().filter(|n| !n.trim().is_empty()).collect();
    if named.is_empty() {
        return Ok(ModelId::ALL.iter().copied()
            .filter(|&id| in_catalog(id) && installed(id) && (heavy || !HEAVY.contains(&id)))
            .collect());
    }
    let mut picked = Vec::with_capacity(named.len());
    for name in named {
        let id = model_named(name)
            .ok_or_else(|| format!("unknown model '{name}'. Known: {}", known_names().join(", ")))?;
        if HEAVY.contains(&id) && !heavy {
            return Err(format!("{} is a heavy model (a multi-GB download, minutes on a CPU); add --heavy to run it", id.stable_name()));
        }
        if !picked.contains(&id) {
            picked.push(id);
        }
    }
    Ok(picked)
}

/// The GPU side when the ladder would try none of the providers in
/// `ladder` (`engine::gpu_ladder`); `None` when it would try one, so the
/// side builds a session, which may still land on the CPU.
fn gpu_passed_over(ladder: &[(&str, Option<Skip>)]) -> Option<Gpu> {
    if ladder.is_empty() {
        return Some(Gpu::NoProvider);
    }
    let skips: Vec<(&str, Skip)> = ladder.iter().map(|&(p, skip)| skip.map(|s| (p, s))).collect::<Option<_>>()?;
    let failed: Vec<String> = skips.iter().filter(|(_, s)| *s == Skip::FailedBefore).map(|(p, _)| p.to_string()).collect();
    Some(if failed.is_empty() {
        let names: Vec<&str> = skips.iter().map(|(p, _)| *p).collect();
        Gpu::ByDesign { reason: format!("{} declared incompatible", names.join(", ")) }
    } else {
        Gpu::FailedBefore { providers: failed }
    })
}

/// The provider the ladder tries first.
fn first_tried<'a>(ladder: &[(&'a str, Option<Skip>)]) -> Option<&'a str> {
    ladder.iter().find(|(_, skip)| skip.is_none()).map(|&(provider, _)| provider)
}

/// One side of a model's run: where its session landed and how long it took.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Side {
    pub provider: String,
    /// Some or all of the session runs on the CPU provider: a Stable
    /// Diffusion bundle names each part's provider in `provider`, and one
    /// part on the CPU makes it no GPU run.
    #[serde(skip)]
    pub on_cpu: bool,
    /// The weights file the session loaded (FP32, or an FP16 / INT8 variant).
    pub weights: Option<&'static str>,
    /// `None` when the session failed to build.
    pub timing: Option<Timing>,
    /// Each provider that failed on the way to `provider`, with its error.
    pub failed: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gpu {
    Ran(Side),
    NoProvider,
    ByDesign { reason: String },
    FailedBefore { providers: Vec<String> },
    /// Not run: it needs more memory than is free (`memory::Short`).
    NoRoom { short: String },
    NotInstalled,
}

/// The outcome a row reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Match,
    Mismatch,
    /// The GPU ran, with no CPU output to compare.
    NoReference,
    /// The ladder found no GPU provider that would take the model.
    FellBack,
    GpuFailed,
    CpuFailed,
    CpuByDesign,
    NoGpuProvider,
    FailedBefore,
    NoRoom,
    NotInstalled,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "MISMATCH",
            Self::NoReference => "no CPU reference",
            Self::FellBack => "fell back to CPU",
            Self::GpuFailed => "GPU run failed",
            Self::CpuFailed => "CPU run failed",
            Self::CpuByDesign => "CPU by design",
            Self::NoGpuProvider => "no GPU provider",
            Self::FailedBefore => "skipped: failed here before",
            Self::NoRoom => "not enough memory",
            Self::NotInstalled => "not installed",
        }
    }

    pub const ALL: [Verdict; 11] = [
        Self::Match, Self::Mismatch, Self::NoReference, Self::FellBack, Self::GpuFailed,
        Self::CpuFailed, Self::CpuByDesign, Self::NoGpuProvider, Self::FailedBefore, Self::NoRoom, Self::NotInstalled,
    ];
}

/// The verdict for a model of `family` from its two sides and how their
/// outputs compared (`sizes_differ` when they could not be).
pub fn verdict(family: Option<Family>, cpu: Option<&Side>, gpu: &Gpu, agreement: Option<&Agreement>, sizes_differ: bool) -> Verdict {
    match gpu {
        Gpu::NotInstalled => Verdict::NotInstalled,
        Gpu::NoProvider => Verdict::NoGpuProvider,
        Gpu::ByDesign { .. } => Verdict::CpuByDesign,
        Gpu::FailedBefore { .. } => Verdict::FailedBefore,
        Gpu::NoRoom { .. } => Verdict::NoRoom,
        Gpu::Ran(side) if side.error.is_some() => Verdict::GpuFailed,
        Gpu::Ran(side) if side.on_cpu => Verdict::FellBack,
        Gpu::Ran(_) if sizes_differ => Verdict::Mismatch,
        Gpu::Ran(_) => match agreement {
            Some(a) if family.is_some_and(|f| a.within(f.tolerance())) => Verdict::Match,
            Some(_) => Verdict::Mismatch,
            None if cpu.is_some_and(|c| c.error.is_some()) => Verdict::CpuFailed,
            None => Verdict::NoReference,
        },
    }
}

/// One model's line in the report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Row {
    pub model: &'static str,
    pub name: &'static str,
    pub family: Option<Family>,
    pub cpu: Option<Side>,
    pub gpu: Gpu,
    pub agreement: Option<Agreement>,
    pub note: Option<String>,
    pub verdict: Verdict,
}

/// What the runner is doing, for the terminal and the report written as
/// the bench goes.
pub enum Event {
    Stage { index: usize, total: usize, model: &'static str, stage: &'static str },
    Done(Box<Row>),
}

/// Run `models` one after the other on `inputs`, each side once untimed
/// and `runs` times timed.
pub fn run(models: &[ModelId], inputs: &Inputs, runs: u32, on: &mut dyn FnMut(Event)) {
    let total = models.len();
    for (i, &id) in models.iter().enumerate() {
        let model = display_name(id);
        let row = bench_model(id, inputs, Runs { cpu: runs, gpu: runs }, &mut |stage| {
            on(Event::Stage { index: i + 1, total, model, stage });
        });
        on(Event::Done(Box::new(row)));
    }
}

fn display_name(id: ModelId) -> &'static str {
    prunr_models::descriptor(id).map_or(id.stable_name(), |d| d.display_name)
}

/// Timed runs on each side after the first, untimed one.
#[derive(Debug, Clone, Copy)]
struct Runs {
    cpu: u32,
    gpu: u32,
}

fn bench_model(id: ModelId, inputs: &Inputs, runs: Runs, stage: &mut dyn FnMut(&'static str)) -> Row {
    let family = Family::of(id);
    let mut row = Row {
        model: display_name(id),
        name: id.stable_name(),
        family,
        cpu: None,
        gpu: Gpu::NotInstalled,
        agreement: None,
        note: None,
        verdict: Verdict::NotInstalled,
    };
    if !prunr_models::is_available(id) {
        return row;
    }
    let Some(family) = family else {
        row.gpu = Gpu::ByDesign { reason: "its sessions run on the CPU provider only".into() };
        row.verdict = Verdict::CpuByDesign;
        return row;
    };
    stage("reading weights");
    warm_weights(id);

    let cpu_reference = family != Family::Diffusion || id.sd_spec().is_some_and(|s| s.cpu_output_correct);
    // A heavy model's CPU run takes minutes; one is enough for a reference.
    let cpu_runs = if HEAVY.contains(&id) { 0 } else { runs.cpu };
    // A side that needs more memory than is free is not run: the bench
    // must not run the machine out of memory either.
    let (cpu, cpu_samples) = match (cpu_reference, memory::fits_now(id, CPU)) {
        (false, _) => {
            row.note = Some("no CPU reference: this model's output on the CPU provider is wrong by design".into());
            (None, None)
        }
        (true, Err(short)) => {
            row.note = Some(format!("no CPU reference: the CPU side {short}"));
            (None, None)
        }
        (true, Ok(())) => {
            let (side, samples) = measure(family, id, inputs, true, cpu_runs, stage);
            (Some(side), samples)
        }
    };
    let ladder = prunr_core::engine::gpu_ladder(id);
    let (gpu, gpu_samples) = match gpu_passed_over(&ladder) {
        Some(passed_over) => (passed_over, None),
        None => match first_tried(&ladder).map_or(Ok(()), |provider| memory::fits_now(id, provider)) {
            Err(short) => (Gpu::NoRoom { short: short.to_string() }, None),
            Ok(()) => {
                let (side, samples) = measure(family, id, inputs, false, runs.gpu, stage);
                (Gpu::Ran(side), samples)
            }
        },
    };
    let on_gpu = matches!(&gpu, Gpu::Ran(side) if !side.on_cpu);
    let mut sizes_differ = false;
    if let (true, Some(a), Some(b)) = (on_gpu, &cpu_samples, &gpu_samples) {
        row.agreement = Agreement::of(a, b);
        sizes_differ = a.len() != b.len();
        if sizes_differ {
            row.note = Some(format!("outputs differ in size: {} values on the CPU, {} on the GPU path", a.len(), b.len()));
        }
    }
    row.verdict = verdict(Some(family), cpu.as_ref(), &gpu, row.agreement.as_ref(), sizes_differ);
    row.cpu = cpu;
    row.gpu = gpu;
    row
}

/// Bring `id`'s own weights in before either side is timed, so neither
/// session time includes the disk: a download (its file, or a bundle's
/// parts) is read through once into the page cache, keeping nothing (holding it would pin hundreds of MB
/// per model for the rest of the run); bundled weights decompress into the
/// copy every session of the process shares.
fn warm_weights(id: ModelId) {
    let files: Vec<_> = prunr_models::download_path(id).into_iter()
        .chain(prunr_models::multi_part_paths(id).into_iter().flatten().map(|(_, path)| path))
        .collect();
    if files.is_empty() {
        let _ = prunr_models::resolve_bytes(id);
        return;
    }
    // More than the page cache can hold evicts itself as it reads, and the
    // session reads the disk again.
    let total: u64 = files.iter().filter_map(|path| std::fs::metadata(path).ok()).map(|m| m.len()).sum();
    if total > crate::hardware::available_ram_bytes_now() {
        return;
    }
    let mut buf = vec![0u8; 1 << 20];
    for path in files {
        let read = std::fs::File::open(&path).and_then(|mut f| {
            while std::io::Read::read(&mut f, &mut buf)? > 0 {}
            Ok(())
        });
        if let Err(e) = read {
            tracing::debug!(model = ?id, path = %path.display(), %e, "could not read the weights ahead");
        }
    }
}

/// Build one side's session, run it once, then `runs` timed times. The
/// samples are the first run's, picked as `family` compares them.
fn measure(family: Family, id: ModelId, inputs: &Inputs, cpu_only: bool, runs: u32, stage: &mut dyn FnMut(&'static str)) -> (Side, Option<Vec<u8>>) {
    let [session_stage, first_stage, timed_stage] = if cpu_only {
        ["CPU session", "CPU first run", "CPU timed runs"]
    } else {
        ["GPU session", "GPU first run", "GPU timed runs"]
    };
    stage(session_stage);
    let started = Instant::now();
    let opened = open(family, id, inputs, cpu_only);
    let session_ms = elapsed_ms(started);
    let mut session = match opened {
        Ok(session) => session,
        Err(error) => {
            let provider = if cpu_only { CPU } else { "none" };
            return (Side { provider: provider.into(), on_cpu: cpu_only, weights: None, timing: None, failed: Vec::new(), error: Some(error) }, None);
        }
    };
    let pick = session.pick();
    let mut side = Side {
        provider: pick.provider.clone(),
        on_cpu: session.on_cpu(),
        weights: session.weights(),
        timing: None,
        failed: pick.failed.clone(),
        error: None,
    };
    stage(first_stage);
    let started = Instant::now();
    let first = session.run();
    let mut timing = Timing { session_ms, first_ms: elapsed_ms(started), runs_ms: Vec::new() };
    let samples = match first {
        Ok(samples) => samples,
        Err(error) => {
            side.error = Some(error);
            side.timing = Some(timing);
            return (side, None);
        }
    };
    stage(timed_stage);
    for _ in 0..runs {
        let started = Instant::now();
        if let Err(error) = session.run() {
            side.error = Some(format!("a timed run failed: {error}"));
            break;
        }
        timing.runs_ms.push(elapsed_ms(started));
    }
    side.timing = Some(timing);
    (side, Some(samples))
}

/// To 0.01 ms, which is all a timer this coarse can say.
fn elapsed_ms(started: Instant) -> f64 {
    (started.elapsed().as_secs_f64() * 100_000.0).round() / 100.0
}

/// A loaded model, run on the bench input as the app runs it.
trait Run {
    fn pick(&self) -> &SessionPick;
    fn run(&mut self) -> Result<Vec<u8>, String>;
    fn weights(&self) -> Option<&'static str> {
        Some(self.pick().weights.label())
    }
    /// Some or all of it runs on the CPU provider.
    fn on_cpu(&self) -> bool {
        self.pick().provider == CPU
    }
}

fn open<'a>(family: Family, id: ModelId, inputs: &'a Inputs, cpu_only: bool) -> Result<Box<dyn Run + 'a>, String> {
    Ok(match family {
        Family::Eraser => Box::new(LamaRun::open(id, inputs, cpu_only)?),
        Family::Diffusion => Box::new(DiffusionRun::open(id, inputs, cpu_only)?),
        Family::Segmentation | Family::Upscale | Family::Colorize | Family::Depth => {
            Box::new(EngineRun::open(family, id, inputs, cpu_only)?)
        }
        Family::Selection => Box::new(SelectionRun::open(inputs, cpu_only)?),
        Family::Lines => Box::new(LinesRun::open(inputs, cpu_only)?),
        Family::FaceFinder => Box::new(FaceFinderRun::open(inputs, cpu_only)?),
        Family::FaceRedraw => Box::new(RedrawRun::open(id, inputs, cpu_only)?),
    })
}

/// The lines over the photo at the app's default look.
struct LinesRun {
    engine: prunr_core::EdgeEngine,
    pick: SessionPick,
    photo: image::DynamicImage,
}

impl LinesRun {
    fn open(inputs: &Inputs, cpu_only: bool) -> Result<Self, String> {
        let (engine, pick) = prunr_core::EdgeEngine::on(cpu_only).map_err(|e| e.to_string())?;
        Ok(Self { engine, pick, photo: image::DynamicImage::ImageRgba8(inputs.photo.clone()) })
    }
}

impl Run for LinesRun {
    fn pick(&self) -> &SessionPick {
        &self.pick
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        self.engine.detect(&self.photo, &prunr_core::EdgeSettings::default()).map(image::RgbaImage::into_raw).map_err(|e| e.to_string())
    }
}

/// The face finder over the photo: how many faces, then each one's
/// score and box in 0-255 of the photo.
struct FaceFinderRun<'a> {
    finder: prunr_core::face::FaceFinder,
    pick: SessionPick,
    inputs: &'a Inputs,
}

impl<'a> FaceFinderRun<'a> {
    fn open(inputs: &'a Inputs, cpu_only: bool) -> Result<Self, String> {
        let (finder, pick) = prunr_core::face::FaceFinder::on(cpu_only).map_err(|e| e.to_string())?;
        Ok(Self { finder, pick, inputs })
    }
}

impl Run for FaceFinderRun<'_> {
    fn pick(&self) -> &SessionPick {
        &self.pick
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        let photo = &self.inputs.photo;
        let faces = self.finder.find(photo).map_err(|e| e.to_string())?;
        let (w, h) = (photo.width() as f32, photo.height() as f32);
        let level = |v: f32| prunr_core::math::round_u8(v * 255.0);
        Ok(std::iter::once(faces.len().min(255) as u8)
            .chain(faces.iter().flat_map(|f| {
                let [x0, y0, x1, y1] = f.bbox;
                [level(f.score), level(x0 / w), level(y0 / h), level(x1 / w), level(y1 / h)]
            }))
            .collect())
    }
}

/// A face restorer's network on the photo's middle, as an aligned crop:
/// the bench picture has no face to find, and the network's work is the
/// same on any crop.
struct RedrawRun {
    redrawer: prunr_core::face::Redrawer,
    pick: SessionPick,
    crop: image::RgbImage,
}

impl RedrawRun {
    fn open(id: ModelId, inputs: &Inputs, cpu_only: bool) -> Result<Self, String> {
        let (redrawer, pick) = prunr_core::face::Redrawer::on(id, cpu_only).map_err(|e| e.to_string())?;
        let side = prunr_core::face::FACE_SIZE;
        let square = prunr_core::formats::resize_rgba(&inputs.photo, side, side, prunr_core::formats::ResizeFilter::Lanczos3);
        Ok(Self { redrawer, pick, crop: image::DynamicImage::ImageRgba8(square).into_rgb8() })
    }
}

impl Run for RedrawRun {
    fn pick(&self) -> &SessionPick {
        &self.pick
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        self.redrawer.redraw(&self.crop).map(image::RgbImage::into_raw).map_err(|e| e.to_string())
    }
}

/// The Magic Brush as the app runs it: the encoder on the side's
/// provider, the decoder on the CPU, a click at eight fixed places.
struct SelectionRun<'a> {
    sessions: std::sync::Arc<crate::gui::processor::SamSessions>,
    pick: SessionPick,
    inputs: &'a Inputs,
}

/// Where the clicks land, as fractions of the picture's width and height.
const CLICKS: [(f32, f32); 8] = [(0.5, 0.5), (0.2, 0.3), (0.8, 0.3), (0.3, 0.8), (0.7, 0.7), (0.5, 0.15), (0.15, 0.6), (0.85, 0.85)];

impl<'a> SelectionRun<'a> {
    fn open(inputs: &'a Inputs, cpu_only: bool) -> Result<Self, String> {
        let (sessions, provider) = crate::gui::processor::magic_brush_sessions(cpu_only)?;
        Ok(Self { sessions, pick: SessionPick { provider, ..SessionPick::default() }, inputs })
    }
}

impl Run for SelectionRun<'_> {
    fn pick(&self) -> &SessionPick {
        &self.pick
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        crate::gui::processor::magic_brush_masks(&self.sessions, &self.inputs.photo, &CLICKS)
    }
}

/// Background removal, upscale, colorize, depth: one `OrtEngine`.
struct EngineRun<'a> {
    engine: prunr_core::OrtEngine,
    family: Family,
    id: ModelId,
    inputs: &'a Inputs,
    /// The photo as background removal takes it, made once.
    decoded: Option<image::DynamicImage>,
}

impl<'a> EngineRun<'a> {
    fn open(family: Family, id: ModelId, inputs: &'a Inputs, cpu_only: bool) -> Result<Self, String> {
        let engine = prunr_core::OrtEngine::for_model(id, prunr_core::batch::ort_intra_threads(1), cpu_only).map_err(|e| e.to_string())?;
        let decoded = (family == Family::Segmentation).then(|| image::DynamicImage::ImageRgba8(inputs.photo.clone()));
        Ok(Self { engine, family, id, inputs, decoded })
    }
}

impl Run for EngineRun<'_> {
    fn pick(&self) -> &SessionPick {
        self.engine.pick()
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        let progress = prunr_core::Progress::none();
        let photo = &self.inputs.photo;
        let samples = match (self.family, &self.decoded) {
            (Family::Segmentation, Some(decoded)) => {
                prunr_core::process_image_from_decoded(decoded, &self.engine, &prunr_core::MaskSettings::default(), &progress)
                    .map(|r| stats::alpha_samples(&r.rgba_image))
            }
            (Family::Upscale, _) => {
                let scale = prunr_models::descriptor(self.id).and_then(|d| d.upscale).map_or(4, |k| k.native_scale);
                prunr_core::upscale::upscale_rgba_with_engine(&self.inputs.tile, &self.engine, self.id, scale, &progress, None)
                    .map(|img| stats::rgb_samples(&img))
            }
            (Family::Colorize, _) => prunr_core::colorize::colorize_plane(&self.engine, photo, &progress)
                .map(|plane| stats::rgb_samples(&prunr_core::colorize::colorized(photo, &plane, 1.0))),
            (Family::Depth, _) => prunr_core::depth::estimate_depth(&self.engine, photo, &progress)
                .map(|map| stats::gray16_samples(map.to_gray16(photo.width(), photo.height()).as_raw())),
            (family, _) => return Err(format!("{family:?} does not run on one engine")),
        };
        samples.map_err(|e| e.to_string())
    }
}

/// LaMa, Big-LaMa, MI-GAN: the eraser's cached session, released on drop.
struct LamaRun<'a> {
    id: ModelId,
    cpu_only: bool,
    pick: SessionPick,
    inputs: &'a Inputs,
}

impl<'a> LamaRun<'a> {
    fn open(id: ModelId, inputs: &'a Inputs, cpu_only: bool) -> Result<Self, String> {
        let pick = prunr_core::inpaint::load_lama(id, cpu_only).map_err(|e| e.to_string())?;
        Ok(Self { id, cpu_only, pick, inputs })
    }
}

impl Run for LamaRun<'_> {
    fn pick(&self) -> &SessionPick {
        &self.pick
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        let Inputs { photo, mask, .. } = self.inputs;
        prunr_core::inpaint::process_lama(photo, mask, self.id, self.cpu_only, prunr_core::inpaint::FillScale::Auto, &prunr_core::Progress::none())
            .map(|filled| stats::rgb_samples_inside(&filled.rgba, mask))
            .map_err(|e| e.to_string())
    }
}

impl Drop for LamaRun<'_> {
    fn drop(&mut self) {
        drop(prunr_core::inpaint::release_all());
    }
}

/// A Stable Diffusion eraser as the app's Eraser runs it at its default
/// settings, from a fixed seed, its bundle kept loaded between the runs
/// and released on drop.
struct DiffusionRun<'a> {
    id: ModelId,
    cpu_only: bool,
    request: prunr_core::inpaint_sd::SdInpaintRequest,
    pick: BundlePick,
    inputs: &'a Inputs,
}

impl<'a> DiffusionRun<'a> {
    fn open(id: ModelId, inputs: &'a Inputs, cpu_only: bool) -> Result<Self, String> {
        let request = crate::gui::processor::sd_request(&diffusion_tuning(id, cpu_only))
            .ok_or_else(|| format!("{id:?} is not a Stable Diffusion eraser"))?;
        let parts = prunr_core::inpaint_sd::load_bundle(id, &request.tuning).map_err(|e| e.to_string())?;
        Ok(Self { id, cpu_only, request, pick: BundlePick::of(parts), inputs })
    }
}

impl Run for DiffusionRun<'_> {
    fn pick(&self) -> &SessionPick {
        &self.pick.pick
    }

    fn weights(&self) -> Option<&'static str> {
        None
    }

    fn on_cpu(&self) -> bool {
        self.pick.on_cpu
    }

    fn run(&mut self) -> Result<Vec<u8>, String> {
        let Inputs { photo, mask, .. } = self.inputs;
        prunr_core::inpaint::process_inpaint_with(photo, mask, self.id, Some(self.request.clone()), self.cpu_only, &prunr_core::Progress::none())
            .map(|filled| stats::rgb_samples_inside(&filled.rgba, mask))
            .map_err(|e| e.to_string())
    }
}

impl Drop for DiffusionRun<'_> {
    fn drop(&mut self) {
        drop(prunr_core::inpaint::release_all());
    }
}

/// The Eraser's tuning for `id` at the app's default settings: the
/// quality the model is made for (Fast for the LCM bundle, Quality for
/// the full one), the default prompts, the fixed seed, kept loaded.
fn diffusion_tuning(id: ModelId, cpu_only: bool) -> crate::gui::processor::InpaintTuning {
    use crate::gui::brush_state::{BrushSettings, SdScheduler};
    let brush = BrushSettings::default();
    let scheduler = if id == ModelId::SdV15LcmInpaintFp16 { SdScheduler::Lcm } else { SdScheduler::DpmPlusPlus2MKarras };
    let quality = crate::gui::presets::SdSchedulerBundle::default_for(scheduler);
    crate::gui::processor::InpaintTuning {
        model: id,
        sd_prompt: brush.sd_prompt_for_run(),
        sd_negative_prompt: brush.sd_negative_prompt.clone(),
        sd_guidance_scale: quality.guidance_scale,
        sd_scheduler: scheduler,
        sd_steps: quality.steps,
        rorem_steps: brush.rorem_steps,
        sd_seed: Some(SEED),
        sd_tuning: prunr_core::inpaint_sd::plan_tuning(
            true,
            prunr_core::inpaint_sd::SD_DEFAULT_MARGIN_MB,
        )
        .with_env_overrides(),
        cpu_only,
        ..Default::default()
    }
}

/// One pick for a bundle of parts: their provider when they share one,
/// else each part's; every part's failures, named by part. `on_cpu`: a
/// part runs on the CPU provider.
struct BundlePick {
    pick: SessionPick,
    on_cpu: bool,
}

impl BundlePick {
    fn of(parts: Vec<(&'static str, SessionPick)>) -> Self {
        let on_cpu = parts.iter().any(|(_, p)| p.provider == CPU);
        let shared = parts.first().map(|(_, p)| p.provider.clone())
            .filter(|first| parts.iter().all(|(_, p)| &p.provider == first));
        let provider = shared.unwrap_or_else(|| {
            parts.iter().map(|(part, p)| format!("{part} {}", p.provider)).collect::<Vec<_>>().join(", ")
        });
        let failed = parts.into_iter()
            .flat_map(|(part, p)| p.failed.into_iter().map(move |f| format!("{part}: {f}")))
            .collect();
        Self { pick: SessionPick { provider, failed, ..Default::default() }, on_cpu }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_is_in_a_family() {
        for &id in ModelId::ALL {
            if id != ModelId::TaesdFp16 {
                assert!(in_catalog(id), "{id:?} is missing from the bench");
            }
        }
        assert_eq!(Family::of(ModelId::Silueta), Some(Family::Segmentation));
        assert_eq!(Family::of(ModelId::Migan), Some(Family::Eraser));
        assert_eq!(Family::of(ModelId::RoremFp16), Some(Family::Diffusion));
        assert_eq!(Family::of(ModelId::RealEsrganX2Plus), Some(Family::Upscale));
        assert_eq!(Family::of(ModelId::Sam21HieraSmall), Some(Family::Selection));
        assert_eq!(Family::of(ModelId::Teed), Some(Family::Lines));
        assert_eq!(Family::of(ModelId::YuNet), Some(Family::FaceFinder));
        assert_eq!(Family::of(ModelId::FaceGfpgan), Some(Family::FaceRedraw));
        assert_eq!(Family::of(ModelId::FaceRestoreFormerPp), Some(Family::FaceRedraw));
        assert!(!in_catalog(ModelId::TaesdFp16), "TAESD runs inside a Stable Diffusion erase");
    }

    #[test]
    fn models_are_named_by_their_stable_names_in_any_case_with_dashes() {
        assert_eq!(model_named("silueta"), Some(ModelId::Silueta));
        assert_eq!(model_named("Real-ESRGAN_x2plus"), Some(ModelId::RealEsrganX2Plus));
        assert_eq!(model_named(" migan "), Some(ModelId::Migan));
        assert_eq!(model_named("taesd_fp16"), None);
        assert_eq!(model_named("lama"), None);
        assert!(known_names().contains(&"lama_fp32"));
    }

    /// No `--models`: every installed model but the heavy ones; with
    /// names: those, in order, once each; an unknown name or a heavy one
    /// without `--heavy` stops the run before it starts.
    #[test]
    fn selection_takes_the_named_models_or_every_installed_light_one() {
        let installed = |id: ModelId| !matches!(id, ModelId::U2net | ModelId::SdV15InpaintFp16);
        let all = select(&[], false, installed).unwrap();
        assert!(all.contains(&ModelId::Silueta) && all.contains(&ModelId::Teed));
        assert!(!all.contains(&ModelId::U2net), "not installed");
        assert!(!all.iter().any(|id| HEAVY.contains(id)), "heavy needs --heavy");
        assert!(!all.contains(&ModelId::TaesdFp16));
        let heavy = select(&[], true, installed).unwrap();
        assert!(heavy.contains(&ModelId::RoremFp16) && !heavy.contains(&ModelId::SdV15InpaintFp16));

        let names = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            select(&names(&["migan", "silueta", "migan", "u2net"]), false, installed).unwrap(),
            [ModelId::Migan, ModelId::Silueta, ModelId::U2net],
            "named models run in the given order, once, installed or not",
        );
        let unknown = select(&names(&["silueta", "nope"]), false, installed).unwrap_err();
        assert!(unknown.contains("'nope'") && unknown.contains("real_esrgan_x2plus"), "{unknown}");
        let needs_heavy = select(&names(&["rorem_fp16"]), false, installed).unwrap_err();
        assert!(needs_heavy.contains("--heavy"), "{needs_heavy}");
        assert_eq!(select(&names(&["rorem_fp16"]), true, installed).unwrap(), [ModelId::RoremFp16]);
    }

    #[test]
    fn the_gpu_side_builds_when_the_ladder_would_try_any_provider() {
        assert_eq!(gpu_passed_over(&[]), Some(Gpu::NoProvider));
        assert_eq!(gpu_passed_over(&[("OpenVINO", Some(Skip::Declared)), ("DirectML", None)]), None);
        assert_eq!(
            gpu_passed_over(&[("OpenVINO", Some(Skip::Declared))]),
            Some(Gpu::ByDesign { reason: "OpenVINO declared incompatible".into() }),
        );
        assert_eq!(
            gpu_passed_over(&[("OpenVINO", Some(Skip::Declared)), ("CUDA", Some(Skip::FailedBefore))]),
            Some(Gpu::FailedBefore { providers: vec!["CUDA".into()] }),
            "OpenVINO is declared out; CUDA is allowed but failed before",
        );
        let ladder = [("OpenVINO", Some(Skip::Declared)), ("DirectML", None), ("CUDA", None)];
        assert_eq!(first_tried(&ladder), Some("DirectML"), "the GPU side's memory is checked where it lands first");
    }

    fn side(provider: &str) -> Side {
        Side { provider: provider.into(), on_cpu: provider == CPU, weights: Some("FP32"), timing: Some(Timing::default()), failed: Vec::new(), error: None }
    }

    /// One row per way a model's run ends.
    #[test]
    fn every_outcome_has_its_verdict() {
        let seg = Some(Family::Segmentation);
        let cpu = side("CPU");
        let gpu = Gpu::Ran(side("DirectML"));
        let close = Agreement { max_abs: 40, mean_abs: 0.4, over_2: 0.01, samples: 10 };
        let far = Agreement { mean_abs: 9.0, ..close };
        let failed = Side { error: Some("boom".into()), ..side("DirectML") };
        let part_on_cpu = Side { on_cpu: true, ..side("unet DirectML, vae_decoder CPU") };
        type Case<'a> = (Gpu, Option<&'a Side>, Option<&'a Agreement>, bool, Verdict);
        let cases: [Case; 13] = [
            (gpu.clone(), Some(&cpu), Some(&close), false, Verdict::Match),
            (gpu.clone(), Some(&cpu), Some(&far), false, Verdict::Mismatch),
            (gpu.clone(), Some(&cpu), None, true, Verdict::Mismatch),
            (gpu.clone(), None, None, false, Verdict::NoReference),
            (gpu.clone(), Some(&failed), None, false, Verdict::CpuFailed),
            (Gpu::Ran(side("CPU")), Some(&cpu), None, false, Verdict::FellBack),
            (Gpu::Ran(part_on_cpu), Some(&cpu), Some(&close), false, Verdict::FellBack),
            (Gpu::Ran(failed.clone()), Some(&cpu), None, false, Verdict::GpuFailed),
            (Gpu::NoProvider, Some(&cpu), None, false, Verdict::NoGpuProvider),
            (Gpu::ByDesign { reason: String::new() }, Some(&cpu), None, false, Verdict::CpuByDesign),
            (Gpu::FailedBefore { providers: vec!["DirectML".into()] }, Some(&cpu), None, false, Verdict::FailedBefore),
            (Gpu::NoRoom { short: String::new() }, Some(&cpu), None, false, Verdict::NoRoom),
            (Gpu::NotInstalled, None, None, false, Verdict::NotInstalled),
        ];
        for (gpu, cpu, agreement, sizes_differ, expected) in &cases {
            assert_eq!(verdict(seg, *cpu, gpu, *agreement, *sizes_differ), *expected, "{gpu:?}");
        }
        for v in Verdict::ALL {
            assert!(cases.iter().any(|case| case.4 == v), "{v:?} has no case");
        }
    }

    /// The same difference passes for a diffusion eraser and fails for a
    /// mask: each family has its own tolerance.
    #[test]
    fn the_tolerance_is_per_family() {
        let drift = Agreement { max_abs: 120, mean_abs: 12.0, over_2: 0.6, samples: 10 };
        let gpu = Gpu::Ran(side("CoreML"));
        let cpu = side("CPU");
        assert_eq!(verdict(Some(Family::Diffusion), Some(&cpu), &gpu, Some(&drift), false), Verdict::Match);
        assert_eq!(verdict(Some(Family::Segmentation), Some(&cpu), &gpu, Some(&drift), false), Verdict::Mismatch);
        for family in Family::ALL {
            let t = family.tolerance();
            assert!(t.mean_abs > 0.0 && t.over_2 > 0.0 && t.over_2 <= 1.0, "{family:?}");
        }
    }

    #[test]
    fn a_bundle_names_one_provider_when_its_parts_share_it() {
        let pick = |provider: &str, failed: &[&str]| SessionPick {
            provider: provider.into(),
            failed: failed.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let same = BundlePick::of(vec![("unet", pick("DirectML", &[])), ("vae_decoder", pick("DirectML", &[]))]);
        assert_eq!((same.pick.provider.as_str(), same.on_cpu), ("DirectML", false));
        let mixed = BundlePick::of(vec![("unet", pick("DirectML", &[])), ("vae_decoder", pick("CPU", &["DirectML test run failed: NaN"]))]);
        assert_eq!(mixed.pick.provider, "unet DirectML, vae_decoder CPU");
        assert_eq!(mixed.pick.failed, ["vae_decoder: DirectML test run failed: NaN"]);
        assert!(mixed.on_cpu, "a part on the CPU makes the bundle no GPU run");
    }
}
