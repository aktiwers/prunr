use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use crate::types::{CoreError, ModelKind};
pub use ort::session::builder::GraphOptimizationLevel;
use ort::{
    ep::{ExecutionProviderDispatch, CPU},
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
};

/// GPU execution providers, platform-gated so ladder matches stay
/// exhaustive without dead arms. `Display` keeps the historic names:
/// they are the persistent `ep_compat.json` cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EpKind {
    #[cfg(not(target_os = "macos"))]
    OpenVino,
    #[cfg(not(target_os = "macos"))]
    Cuda,
    /// AMD's provider (ROCm 7); its runtime has only a Linux wheel.
    #[cfg(target_os = "linux")]
    MiGraphX,
    #[cfg(target_os = "macos")]
    CoreMl,
    #[cfg(windows)]
    DirectMl,
}

/// OpenVINO's provider name, defined on every platform so code that
/// reads a provider string needs no per-platform branch.
pub(crate) const OPENVINO_NAME: &str = "OpenVINO";

/// CUDA's provider name, defined on every platform like OpenVINO's.
pub const CUDA_NAME: &str = "CUDA";

/// CoreML's provider name, defined on every platform like CUDA's.
pub(crate) const COREML_NAME: &str = "CoreML";

/// The CPU provider's name: its label, its cache-dir key and the
/// fallback when no GPU provider is available.
pub const CPU_NAME: &str = "CPU";

impl EpKind {
    /// Stable name: also the `ep_compat.json` cache key.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => OPENVINO_NAME,
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => CUDA_NAME,
            #[cfg(target_os = "linux")]
            EpKind::MiGraphX => "MIGraphX",
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => COREML_NAME,
            #[cfg(windows)]
            EpKind::DirectMl => "DirectML",
        }
    }
}

impl std::fmt::Display for EpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The GPU EP ladder filtered by `is_available()`: a `load-dynamic`
/// runtime may lack EPs (the OpenVINO wheel has CPU + OpenVINO only).
/// Probed once per process.
///
/// OpenVINO goes before CUDA so Intel users get it; an NVIDIA machine
/// has no OpenVINO runtime, so it still reaches CUDA.
pub(crate) fn available_gpu_eps() -> &'static [EpKind] {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Vec<EpKind>> = OnceLock::new();
    CACHED.get_or_init(|| {
        use ort::ep::ExecutionProvider;
        let mut eps: Vec<EpKind> = Vec::new();
        // ort panics on any call before the runtime is loaded.
        if crate::ort_runtime::ensure_initialized().is_err() {
            return eps;
        }
        #[cfg(target_os = "macos")]
        {
            if ort::ep::CoreML::is_available().unwrap_or(false) { eps.push(EpKind::CoreMl); }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if ort::ep::OpenVINO::is_available().unwrap_or(false) { eps.push(EpKind::OpenVino); }
            if ort::ep::CUDA::is_available().unwrap_or(false) {
                crate::ort_runtime::preload_nvidia_libraries();
                match provider_starts(ort::ep::CUDA::default().with_device_id(0).build().error_on_failure()) {
                    Ok(()) => eps.push(EpKind::Cuda),
                    Err(e) => {
                        tracing::info!(%e, "CUDA cannot start; sessions run on the CPU");
                        let _ = GPU_START_FAILURE.set(GpuStartFailure::of(GpuStack::Cuda, &e));
                    }
                }
            }
            #[cfg(target_os = "linux")]
            if ort::ep::MIGraphX::is_available().unwrap_or(false) {
                match provider_starts(ort::ep::MIGraphX::default().with_device_id(0).build().error_on_failure()) {
                    Ok(()) => eps.push(EpKind::MiGraphX),
                    Err(e) => {
                        tracing::info!(%e, "MIGraphX cannot start; sessions run on the CPU");
                        let _ = GPU_START_FAILURE.set(GpuStartFailure::of(GpuStack::Rocm, &e));
                    }
                }
            }
            #[cfg(windows)]
            if ort::ep::DirectML::is_available().unwrap_or(false) && directml_starts() { eps.push(EpKind::DirectMl); }
        }
        let eps_str: Vec<&'static str> = eps.iter().map(EpKind::as_str).collect();
        tracing::info!(eps = ?eps_str, "Available GPU execution providers");
        eps
    }).as_slice()
}

/// Why the loaded runtime's GPU provider could not start, when it could not.
static GPU_START_FAILURE: std::sync::OnceLock<GpuStartFailure> = std::sync::OnceLock::new();

/// Why the GPU provider the runtime carries does not run here, once the
/// providers have been probed; `None` when it runs or there is none.
pub fn gpu_start_failure() -> Option<&'static GpuStartFailure> {
    GPU_START_FAILURE.get()
}

/// The vendor stack a GPU runtime needs installed beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuStack {
    /// NVIDIA's driver with CUDA 12 and cuDNN 9.
    Cuda,
    /// AMD's ROCm 7 with MIGraphX.
    Rocm,
}

impl GpuStack {
    fn provider(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "MIGraphX",
        }
    }

    fn needs(self) -> &'static str {
        match self {
            Self::Cuda => "NVIDIA's CUDA 12 and cuDNN 9 libraries",
            Self::Rocm => "AMD's ROCm 7 with MIGraphX installed",
        }
    }
}

/// Why a GPU runtime's provider did not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuStartFailure {
    pub stack: GpuStack,
    pub why: StartFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartFailure {
    /// The loader could not find this library of the vendor's stack.
    MissingLibrary(String),
    /// No driver, or one older than the stack needs.
    NoDriver,
    /// The driver runs but finds no graphics card it serves.
    NoDevice,
    Other,
}

impl GpuStartFailure {
    /// A CUDA start that failed on `err`, for tests outside the crate.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn cuda_for_test(err: &str) -> Self {
        Self::of(GpuStack::Cuda, err)
    }

    /// `err` names the first library the loader could not find, or the
    /// vendor's error code. macOS starts no CUDA or ROCm provider.
    #[cfg(any(not(target_os = "macos"), test, feature = "test-utils"))]
    pub(crate) fn of(stack: GpuStack, err: &str) -> Self {
        let missing = err
            .split(": cannot open shared object file")
            .next()
            .and_then(|head| head.rsplit(' ').next())
            .filter(|lib| lib.starts_with("lib") && lib.contains(".so"));
        let why = match missing {
            Some(lib) if lib.starts_with("libcuda.so") => StartFailure::NoDriver,
            Some(lib) => StartFailure::MissingLibrary(lib.to_string()),
            // cudaErrorInsufficientDriver, which is also what a missing
            // driver gives once CUDA's own libraries load.
            None if err.contains("CUDA failure 35:") => StartFailure::NoDriver,
            // cudaErrorNoDevice / hipErrorNoDevice.
            None if err.contains("CUDA failure 100:") || err.contains("hipErrorNoDevice") => StartFailure::NoDevice,
            None => StartFailure::Other,
        };
        Self { stack, why }
    }

    /// The failure in the user's words.
    pub fn text(&self) -> String {
        let provider = self.stack.provider();
        match &self.why {
            StartFailure::MissingLibrary(lib) => format!("{provider} could not start: {lib} is missing. It needs {}.", self.stack.needs()),
            StartFailure::NoDriver => match self.stack {
                GpuStack::Cuda => "CUDA could not start: the NVIDIA driver is missing or older than CUDA 12 needs (525 or newer).".to_string(),
                GpuStack::Rocm => "MIGraphX could not start: the AMD graphics driver is missing or too old for ROCm 7.".to_string(),
            },
            StartFailure::NoDevice => match self.stack {
                GpuStack::Cuda => "CUDA could not start: no NVIDIA graphics card was found.".to_string(),
                GpuStack::Rocm => "MIGraphX could not start: no AMD graphics card that ROCm supports was found.".to_string(),
            },
            StartFailure::Other => format!("{provider} could not start, so models run on the CPU. The log has the details."),
        }
    }
}

/// Whether `provider` registers on a session builder. A runtime lists a
/// provider it was built with even where it cannot run; registering is
/// what loads its libraries and finds the device.
#[cfg(not(target_os = "macos"))]
fn provider_starts(provider: ExecutionProviderDispatch) -> Result<(), String> {
    crate::ort_runtime::session_builder()
        .map_err(|e| e.to_string())
        .and_then(|b| b.with_execution_providers([provider]).map(drop).map_err(|e| e.to_string()))
}

/// Whether DirectML can start a device here. The DirectML runtime, which
/// Windows ships built in, lists DirectML on every machine; registering
/// it is what creates the device, and that fails without a DirectX 12
/// GPU (a VM, a remote desktop), where sessions then run on the CPU of
/// the same runtime and the label says so.
#[cfg(windows)]
fn directml_starts() -> bool {
    let started = provider_starts(directml_provider());
    if let Err(e) = &started {
        tracing::info!(%e, "DirectML cannot start a device; sessions run on the CPU");
    }
    started.is_ok()
}

/// No device id: DXCore's high-performance order picks the discrete card
/// on a hybrid laptop, where DXGI's index 0 is usually the integrated one.
#[cfg(windows)]
fn directml_provider() -> ExecutionProviderDispatch {
    ort::ep::DirectML::default().build().error_on_failure()
}

/// The GPU providers a session may try: none when the user chose
/// "CPU only" (Settings › Processor).
pub(crate) fn gpu_eps_for(cpu_only: bool) -> &'static [EpKind] {
    if cpu_only { &[] } else { available_gpu_eps() }
}

/// Set by `prunr bench --table`: this process tries the places the
/// registry keeps on the CPU for a slower, different or no-gain result,
/// so they are measured again. A place that fails stays skipped: some
/// fail by hanging or by taking the machine's memory.
static REMEASURE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub fn remeasure_places() {
    let _ = REMEASURE.set(());
}

/// Why `model` runs on the CPU rather than `ep`: its place there, or a
/// provider that cannot serve its kind of session.
fn kept_off(model: prunr_models::ModelId, ep: EpKind) -> Option<prunr_models::Why> {
    if compiles_per_shape(model, ep) {
        return Some(COMPILES_PER_SHAPE);
    }
    match prunr_models::place(model, ep.as_str(), openvino_on_gpu()) {
        place @ prunr_models::Place::Cpu(why) if place.kept_off(REMEASURE.get().is_some()) => Some(why),
        _ => None,
    }
}

/// A session of `id` on the first accelerator that takes it (the
/// registry's `places`) and builds it, else on the CPU, with where it
/// landed: for a model whose caller runs the session itself (lines,
/// faces). `commit` loads the model into the builder it is handed,
/// provider attached.
pub fn session_on_ladder(
    id: prunr_models::ModelId,
    cpu_only: bool,
    commit: impl Fn(ort::session::builder::SessionBuilder) -> Result<Session, String>,
) -> Result<(Session, SessionPick), CoreError> {
    let eps = gpu_eps_for(cpu_only);
    let mut failed = Vec::new();
    if !eps.is_empty() {
        let landed = walk_ladder("", id, Weights::Fp32, eps, |ep| {
            let builder = crate::ort_runtime::session_builder().map_err(|e| Miss::Unusable(e.to_string()))?;
            let built = ep.attach(builder, id, EpSite::Engine { threads: num_cpus::get(), part: None })?;
            commit(built).map_err(|e| Miss::Failed { error: format!("ORT session creation failed ({ep}): {e}"), suspect_cache: None })
        });
        match landed {
            Ok((session, ep, failed)) => return Ok((session, SessionPick { failed, ..SessionPick::on(ep.as_str()) })),
            Err(walk) => failed = walk.failed,
        }
    }
    let session = commit(crate::ort_runtime::session_builder()?).map_err(CoreError::Inference)?;
    Ok((session, SessionPick { failed, ..SessionPick::on(CPU_NAME) }))
}

/// The session family a provider is registered for; each keeps its own
/// tuning of the providers.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // each field tunes a provider that only some platforms compile in
pub(crate) enum EpSite<'a> {
    /// Segmentation and upscale engines; `part` names one part of a model
    /// run part by part, so what it keeps of its own (CoreML's compiled
    /// model) stays apart from the other parts'.
    Engine { threads: usize, part: Option<&'a str> },
    /// LaMa-class fills.
    Fill { threads: usize },
    /// One part of a Stable Diffusion bundle. `ov_device` pins OpenVINO's
    /// device (bench only).
    SdPart { part: &'a str, ov_device: Option<&'a str> },
    /// One part of the Magic Brush's model (SAM 2.1).
    Selection { part: &'a str },
}

fn disabled_optimizers_on(provider: &str) -> &'static [&'static str] {
    if provider == COREML_NAME { &["MatMulAddFusion"] } else { &[] }
}

/// OpenVINO's device for engines once the user lets it use the Intel GPU.
/// AUTO compiles for the GPU while the CPU serves the first runs, and
/// stays on the CPU where no GPU or OpenCL driver is. Opt-in because it
/// is not a win everywhere: on an HD 530 (prunr bench, 2026-10) U2Net ran
/// 441 → 185 ms, BiRefNet-lite 9.1 → 12.9 s, and Real-ESRGAN x2's FP16
/// output drifted (10.8% of values off by more than 2/255).
#[cfg(not(target_os = "macos"))]
const OV_ENGINE_DEVICE: &str = "AUTO:GPU,CPU";

/// The user's Intel GPU choice, read once per process.
#[cfg(not(target_os = "macos"))]
fn intel_gpu_chosen() -> bool {
    static CHOSEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| {
        let chosen = crate::ort_runtime::intel_gpu_chosen();
        tracing::info!(chosen, "OpenVINO may use the Intel GPU");
        chosen
    })
}

#[cfg(not(target_os = "macos"))]
impl EpSite<'_> {
    /// The OpenVINO device this site asks for; `None` is OpenVINO's CPU.
    /// Fills stay on the CPU: the GPU plugin cannot build LaMa (dynamic
    /// shapes), and under AUTO it ran ~10% slower than the CPU.
    fn ov_device(&self, intel_gpu: bool) -> Option<&str> {
        match self {
            EpSite::Engine { .. } | EpSite::Selection { .. } => intel_gpu.then_some(OV_ENGINE_DEVICE),
            EpSite::Fill { .. } => None,
            EpSite::SdPart { ov_device, .. } => *ov_device,
        }
    }
}

impl EpKind {
    /// `builder` with this provider tuned for `site` (`provider`) and the
    /// session options it needs.
    pub(crate) fn attach(self, builder: ort::session::builder::SessionBuilder, model: prunr_models::ModelId, site: EpSite<'_>) -> Result<ort::session::builder::SessionBuilder, Miss> {
        crate::ort_runtime::without_optimizers(builder, self.disabled_optimizers())
            .map_err(|e| Miss::Unusable(e.to_string()))?
            .with_execution_providers([self.provider(model, site)])
            .map_err(|e| Miss::Unusable(format!("ORT register {self} EP failed: {e}")))
    }

    /// ORT optimizers this provider's sessions go without. CoreML: with
    /// MatMul and Add fused into Gemm, ORT writes each transformer weight
    /// into the compiled program inline instead of its weight file; such
    /// models took 27-49 s to load on a MacBook Air instead of 1-2 s, at
    /// the same speed and output.
    fn disabled_optimizers(self) -> &'static [&'static str] {
        disabled_optimizers_on(self.as_str())
    }

    /// This provider tuned for `site`, set to fail its registration
    /// loudly. ort's default logs a failed registration and commits the
    /// session on the CPU, which the ladder would then label with this
    /// provider's name. A failed registration says the machine lacks the
    /// provider, not that the model breaks on it, so the ladders move on
    /// without an `ep_compat` record.
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] // only CoreML's compiled-model cache is per model
    pub(crate) fn provider(self, model: prunr_models::ModelId, site: EpSite<'_>) -> ExecutionProviderDispatch {
        let dispatch = match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => {
                let p = ort::ep::CUDA::default().with_device_id(0);
                match site {
                    // No CUDA graph: it replays the first run's shapes and
                    // buffer addresses, and tiles and photos change both.
                    // Heuristic conv search for the same reason: exhaustive
                    // benchmarks every new shape, the fixed default
                    // algorithm is often the slowest.
                    EpSite::Engine { .. } => p
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                        .with_tf32(true)
                        .build(),
                    // The pool and search of the engines, for the same
                    // reasons, without TF32's lower precision: SD 1.5 on a
                    // 12 GB RTX 3060 took 7.1 GB for 512 px tiles instead
                    // of 9.4, at the same 4.0 s, and its pixels differ from
                    // the defaults' no more than two default runs do. The
                    // Magic Brush's masks on it match the CPU's (same pick
                    // on 20 of 20 clicks, IoU 0.9998).
                    EpSite::SdPart { .. } | EpSite::Selection { .. } => p
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                        .build(),
                    EpSite::Fill { .. } => p.build(),
                }
            }
            // One device, as for CUDA: the first card ROCm lists.
            #[cfg(target_os = "linux")]
            EpKind::MiGraphX => ort::ep::MIGraphX::default().with_device_id(0).build(),
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => {
                let dir = match site {
                    EpSite::SdPart { part, .. } | EpSite::Engine { part: Some(part), .. } | EpSite::Selection { part } => {
                        crate::cache::cache_dir_for_part(model, self.as_str(), part)
                    }
                    EpSite::Engine { part: None, .. } | EpSite::Fill { .. } => crate::cache::cache_dir_for(model, self.as_str()),
                };
                coreml_provider(dir.as_deref()).build()
            }
            #[cfg(windows)]
            EpKind::DirectMl => directml_provider(),
            // NOT using `with_cache_dir`: on ort 2.0.0-rc.12 + OpenVINO
            // 1.24.1 cold and warm SD UNet commits match (23.7 / 23.2 s)
            // because OpenVINO keys the cache differently each launch and
            // rewrites ~3.3 GB every build.
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => match site {
                // Uncapped, OpenVINO's TBB pool takes every logical core
                // and oversubscribes against rayon (`sched_yield` ~2% of
                // CLI batch wall time).
                EpSite::Engine { threads, .. } | EpSite::Fill { threads } => {
                    let provider = ort::ep::OpenVINO::default().with_num_threads(threads.max(1));
                    match site.ov_device(intel_gpu_chosen()) {
                        Some(device) => provider.with_device_type(device),
                        None => provider,
                    }
                    .build()
                }
                // Only on the Intel GPU: the Magic Brush leaves OpenVINO's
                // CPU device to ORT's own CPU provider.
                EpSite::Selection { .. } => {
                    let provider = ort::ep::OpenVINO::default();
                    match site.ov_device(intel_gpu_chosen()) {
                        Some(device) => provider.with_device_type(device),
                        None => provider,
                    }
                    .build()
                }
                // `num_streams=1` disables per-stream buffer duplication
                // (the bundle sits at ~16 GB RSS on an i7-6700 either
                // way; measured 2026-10). Dynamic shapes cost nothing per
                // step, so one bundle serves 512² tiles and the 512×768
                // crop.
                EpSite::SdPart { .. } => {
                    let provider = ort::ep::OpenVINO::default()
                        .with_num_streams(1)
                        .with_dynamic_shapes(true);
                    match site.ov_device(intel_gpu_chosen()) {
                        Some(device) => provider.with_device_type(device),
                        None => provider,
                    }
                    .build()
                }
            },
        };
        dispatch.error_on_failure()
    }

    /// Whether an engine on this provider compiles a model for each input
    /// size it meets (`ModelDescriptor::free_dims`). CoreML: a free size
    /// compiles to a program for any size, slower and many times larger.
    pub(crate) fn sizes_each_input(self) -> bool {
        self.as_str() == COREML_NAME
    }

    /// Whether sessions on this provider load and write ORT's
    /// optimized-graph cache. Never DirectML: ORT turns its graph fusion
    /// off whenever an optimized model path is set.
    pub(crate) fn keeps_optimized_graph(self) -> bool {
        match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => true,
            _ => false,
        }
    }
}

/// CoreML set to compile ML Programs: NeuralNetwork, its default, takes no
/// fp16 or int64 inputs, ConvTranspose or LayerNorm. A float32 ML
/// Program runs on the GPU in fp32, never on the Neural Engine, so the
/// units are the CPU and the GPU: with all of them CoreML also planned
/// for the Neural Engine and failed Silueta and Real-ESRGAN x2 on a
/// MacBook Air ("Error in building plan").
///
/// CoreML reuses what it compiled under `cache_dir` without checking
/// input shapes: a session that pins its input dims must key the dir on
/// them, or a model compiled for one tile size is reused for another.
#[cfg(any(target_os = "macos", test))]
fn coreml_provider(cache_dir: Option<&std::path::Path>) -> ort::ep::CoreML {
    let p = ort::ep::CoreML::default()
        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU);
    match cache_dir {
        Some(dir) => p.with_model_cache_dir(dir.to_string_lossy()),
        None => p,
    }
}

/// Why a provider gave no session.
#[derive(Debug)]
pub(crate) enum Miss {
    /// The provider could not be set up: says nothing about the model.
    Unusable(String),
    /// The session failed to build or to run. `suspect_cache`: the cached
    /// optimized graph it loaded, which is then the suspect and is deleted
    /// instead of the failure being held against the provider.
    Failed { error: String, suspect_cache: Option<PathBuf> },
}

/// Why the ladder passes a provider over without trying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// It runs on the CPU rather than there, and why: its place in the
    /// registry (`prunr_models::Places`), or a provider that cannot serve
    /// its kind of session.
    KeptOff(prunr_models::Why),
    /// It failed these weights of the model on an earlier run, under the
    /// same drivers and runtime; skipped until they change or the list is
    /// reset (Settings › Hardware, `prunr --clear-ep-cache`).
    FailedBefore,
}

impl Skip {
    fn text(self, ep: EpKind) -> String {
        match self {
            Skip::KeptOff(why) => format!("{ep} is kept off: {}", why.text()),
            Skip::FailedBefore => format!("{ep} failed this model before"),
        }
    }
}

/// The CPU provider every session falls back to: no arena, for a lower
/// memory baseline (the subprocess handles OOM).
fn cpu_provider() -> ExecutionProviderDispatch {
    CPU::default().with_arena_allocator(false).build()
}

/// Whether, and why, the ladder passes `ep` over for these weights.
pub(crate) fn skip_reason(model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<Skip> {
    if let Some(why) = kept_off(model, ep) {
        Some(Skip::KeptOff(why))
    } else if crate::ep_compat::is_known_failure(ep, model, weights) {
        Some(Skip::FailedBefore)
    } else {
        None
    }
}

/// MIGraphX compiles a static graph for each input shape. A fill meets a
/// new shape with every crop and SD with each tile layout, so each stroke
/// would recompile for seconds to minutes; a model run in parts sizes its
/// crops to the image, so each new size would recompile every part. They
/// stay off it. Engines see one shape per model and keep it.
const COMPILES_PER_SHAPE: prunr_models::Why =
    prunr_models::Why::Fails("MIGraphX compiles a graph per input shape: each crop or part would compile again");

fn compiles_per_shape(model: prunr_models::ModelId, ep: EpKind) -> bool {
    #[cfg(target_os = "linux")]
    {
        ep == EpKind::MiGraphX
            && prunr_models::descriptor(model).is_some_and(|d| d.category == prunr_models::ModelCategory::Inpaint || d.runs_in_parts())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (model, ep);
        false
    }
}

/// What a walk down the ladder met: each provider that failed, with its
/// error, in order, and each it passed over without a try.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Walk {
    pub(crate) failed: Vec<String>,
    skipped: Vec<String>,
}

impl Walk {
    /// The walk for `weights` when it would try none of `eps`: each one's
    /// skip. `None` when it would try one.
    fn without_a_try(model: prunr_models::ModelId, weights: Weights, eps: &[EpKind]) -> Option<Self> {
        let skipped = eps.iter().map(|&ep| skip_reason(model, weights, ep).map(|s| s.text(ep))).collect::<Option<_>>()?;
        Some(Self { failed: Vec::new(), skipped })
    }

    /// Why the walk landed nowhere, in one line.
    pub(crate) fn reason(&self) -> String {
        if !self.failed.is_empty() {
            self.failed.join("; ")
        } else if !self.skipped.is_empty() {
            format!("no GPU provider to try: {}", self.skipped.join(", "))
        } else {
            "No GPU EPs available on this platform".to_string()
        }
    }
}

/// Each provider in `eps` with `weights` until `commit` builds a session,
/// returned with the failures on the way (`SessionPick::failed`). Pairs
/// declared or known to fail are skipped; a new failure is recorded
/// against these weights only, so FP16 failing never bars FP32. `prefix`
/// leads each log line ("LaMa: ").
pub(crate) fn walk_ladder<S>(
    prefix: &str,
    model: prunr_models::ModelId,
    weights: Weights,
    eps: &[EpKind],
    mut commit: impl FnMut(EpKind) -> Result<S, Miss>,
) -> Result<(S, EpKind, Vec<String>), Walk> {
    crate::ep_compat::refresh();
    let mut walk = Walk::default();
    for &ep in eps {
        if let Some(skip) = skip_reason(model, weights, ep) {
            tracing::debug!(?model, ?weights, ep = %ep, ?skip, "{prefix}EP skipped");
            walk.skipped.push(skip.text(ep));
            continue;
        }
        let drafted = crate::cache::drafts_attached();
        let started = std::time::SystemTime::now();
        let mut committed = commit(ep);
        // A build that failed while writing its optimized graph (the disk
        // filled meanwhile) says nothing about the provider until it fails
        // without one too.
        if matches!(committed, Err(Miss::Failed { suspect_cache: None, .. })) && crate::cache::drafts_attached() > drafted {
            tracing::warn!(?model, ep = %ep, "{prefix}session failed while writing its optimized graph; building it without");
            committed = crate::cache::without_drafts(|| commit(ep));
        }
        match committed {
            Ok(session) => return Ok((session, ep, walk.failed)),
            Err(Miss::Unusable(error)) => {
                tracing::warn!(?model, ep = %ep, %error, "{prefix}GPU EP unusable — trying next EP");
                walk.failed.push(error);
            }
            Err(Miss::Failed { error, suspect_cache }) => {
                tracing::warn!(?model, ?weights, ep = %ep, %error, "{prefix}GPU session failed — trying next EP");
                match suspect_cache.or_else(|| compiled_cache_named(model, ep.as_str(), &error, started)) {
                    Some(path) => clear_suspect_cache(&path),
                    None => crate::ep_compat::record_failure(ep, model, weights, &error),
                }
                walk.failed.push(error);
            }
        }
    }
    Err(walk)
}

/// The GPU providers the loaded runtime offers, in the order a session
/// tries them.
pub fn gpu_provider_names() -> Vec<String> {
    available_gpu_eps().iter().map(|&ep| match ep {
        #[cfg(not(target_os = "macos"))]
        EpKind::OpenVino if intel_gpu_chosen() => prunr_models::Column::OpenVinoGpu.name().to_string(),
        _ => ep.as_str().to_string(),
    }).collect()
}

/// The lighter copy of `id`'s weights a session tries before its own,
/// when one is on disk: FP16 on a GPU, INT8 on the CPU. Never on macOS,
/// where CoreML converts to FP16 itself.
fn lighter_weights(id: prunr_models::ModelId, cpu_only: bool) -> Option<Weights> {
    if cfg!(target_os = "macos") {
        None
    } else if cpu_only {
        prunr_models::has_int8_variant(id).then_some(Weights::Int8)
    } else {
        prunr_models::has_fp16_variant(id).then_some(Weights::Fp16)
    }
}

/// Why a GPU session of `model` passes `ep` over for every weights it
/// tries (`None`: it tries `ep`).
fn gpu_skip(model: prunr_models::ModelId, ep: EpKind) -> Option<Skip> {
    if let Some(why) = kept_off(model, ep) {
        return Some(Skip::KeptOff(why));
    }
    let mut why = None;
    for weights in [lighter_weights(model, false), Some(Weights::Fp32)].into_iter().flatten() {
        why = Some(skip_reason(model, weights, ep)?);
    }
    why
}

/// Run options that shrink CUDA device 0's pool as a run ends; `None`
/// when the runtime refuses them, and the run goes without.
pub fn shrink_cuda_pool() -> Option<&'static ort::session::RunOptions> {
    static OPTIONS: std::sync::OnceLock<Option<ort::session::RunOptions>> = std::sync::OnceLock::new();
    OPTIONS.get_or_init(|| {
        let mut options = ort::session::RunOptions::new().ok()?;
        options.set("memory.enable_memory_arena_shrinkage", "gpu:0")
            .map_err(|e| tracing::warn!(%e, "CUDA pool shrinking unavailable"))
            .ok()?;
        Some(options)
    }).as_ref()
}

/// The first GPU provider a session of `model` tries, as the ladder walks
/// them; `None`: it runs on the CPU.
pub(crate) fn first_gpu_provider(model: prunr_models::ModelId, cpu_only: bool) -> Option<&'static str> {
    gpu_eps_for(cpu_only).iter().find(|&&ep| gpu_skip(model, ep).is_none()).map(EpKind::as_str)
}

/// Each GPU provider the loaded runtime offers, in ladder order, with why
/// a session of `model` would pass it over (`None`: it would try it).
pub fn gpu_ladder(model: prunr_models::ModelId) -> Vec<(&'static str, Option<Skip>)> {
    available_gpu_eps().iter().map(|&ep| (ep.as_str(), gpu_skip(model, ep))).collect()
}

/// The weights file a session loaded: the model's own (FP32 for
/// segmentation and upscale; an SD bundle's own half-precision weights
/// count as these too), or the FP16 (GPU) or INT8 (CPU) variant beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Weights {
    #[default]
    Fp32,
    Fp16,
    Int8,
}

impl Weights {
    pub const ALL: [Weights; 3] = [Weights::Fp32, Weights::Fp16, Weights::Int8];

    pub fn label(self) -> &'static str {
        match self {
            Weights::Fp32 => "FP32",
            Weights::Fp16 => "FP16",
            Weights::Int8 => "INT8",
        }
    }
}

/// Where a session ended up: the provider it runs on, the weights it
/// loaded, and each earlier try that failed, in order, with the
/// provider's own error text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionPick {
    pub provider: String,
    pub weights: Weights,
    pub failed: Vec<String>,
}

impl SessionPick {
    pub(crate) fn on(provider: &str) -> Self {
        Self { provider: provider.to_string(), ..Default::default() }
    }

    /// `earlier` failures go ahead of this pick's own.
    fn put_first(&mut self, mut earlier: Vec<String>) {
        earlier.append(&mut self.failed);
        self.failed = earlier;
    }
}

/// ORT-backed inference engine. Create once per model and reuse across
/// images; never per image.
///
/// The session is behind a Mutex because `Session::run()` takes `&mut self`.
pub struct OrtEngine {
    sessions: Sessions,
    pick: SessionPick,
    model_kind: ModelKind,
}

/// One session, one per input size, or a model cut into parts that run
/// in turn.
enum Sessions {
    One(Mutex<Session>),
    Sized(Box<SizedSessions>),
    Parts(Box<crate::upscale::blocks::BlockNet>),
}

/// A model compiled for one input size at a time, on a provider that runs
/// it best that way (CoreML): the session for the size last run, compiled
/// again when another size comes.
struct SizedSessions {
    recipe: SizeRecipe,
    weights: Weights,
    /// The size the engine was built at, compiled again when a new size
    /// failed to compile and left no session.
    first: Vec<usize>,
    current: Mutex<Option<(Vec<usize>, Session)>>,
}

impl SizedSessions {
    /// `f` on the session for an input of `shape`, compiled first when
    /// the size is new. The old size's session is freed before the new one
    /// builds, so the engine holds one at a time.
    fn with_session<T>(&self, shape: &[usize], f: impl FnOnce(&mut Session) -> Result<T, CoreError>) -> Result<T, CoreError> {
        let mut current = self.current.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, session)) = current.as_mut().filter(|(size, _)| size.as_slice() == shape) {
            return f(session);
        }
        drop(current.take());
        let (_, session) = current.insert((shape.to_vec(), self.build(shape)?));
        f(session)
    }

    /// `f` on the session for the size last run, or the first size's when
    /// none is compiled.
    fn with_current<T>(&self, f: impl FnOnce(&mut Session) -> Result<T, CoreError>) -> Result<T, CoreError> {
        let shape = self.current.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or_else(|| self.first.clone(), |(size, _)| size.clone());
        self.with_session(&shape, f)
    }

    /// A session for `shape`; a compile it loaded from the cache that
    /// fails is deleted and compiled afresh, once.
    fn build(&self, shape: &[usize]) -> Result<Session, CoreError> {
        let SizeRecipe { model, ep, .. } = self.recipe;
        let bytes = OrtEngine::weights_bytes(model, self.weights)?
            .ok_or_else(|| CoreError::Inference(format!("{model:?}: its {} weights are gone from disk", self.weights.label())))?;
        let started = std::time::SystemTime::now();
        self.recipe.session(&bytes, shape).or_else(|miss| {
            let error = miss_text(miss);
            let cached = compiled_cache_named(model, ep.as_str(), &error, started).ok_or_else(|| CoreError::Inference(error.clone()))?;
            tracing::warn!(?model, %ep, %error, "a compiled size failed to load; compiling it again");
            clear_suspect_cache(&cached);
            self.recipe.session(&bytes, shape).map_err(|miss| CoreError::Inference(miss_text(miss)))
        })
    }
}

fn miss_text(miss: Miss) -> String {
    match miss {
        Miss::Unusable(error) | Miss::Failed { error, .. } => error,
    }
}

/// What a session of a model compiled for one input size is built with.
#[derive(Clone, Copy)]
struct SizeRecipe {
    model: prunr_models::ModelId,
    ep: EpKind,
    intra_threads: usize,
    level: GraphOptimizationLevel,
}

impl SizeRecipe {
    /// A session of `model_bytes` compiled for `shape`, its compile kept
    /// apart from other sizes'.
    fn session(self, model_bytes: &[u8], shape: &[usize]) -> Result<Session, Miss> {
        let Self { model, ep, intra_threads, level } = self;
        let builder = OrtEngine::builder_with_base(intra_threads, level).map_err(|e| Miss::Unusable(e.to_string()))?;
        let (builder, coverage) = crate::ep_coverage::watch(ep, fix_sizes(builder, model, shape)?);
        let key = size_key(shape);
        let mut built = ep.attach(builder, model, EpSite::Engine { threads: intra_threads, part: Some(&key) })?;
        let started = Instant::now();
        let session = built.commit_from_memory(model_bytes)
            .map_err(|e| Miss::Failed { error: format!("ORT session creation failed ({ep}, input {key}): {e}"), suspect_cache: None })?;
        tracing::info!(?model, %ep, size = %key, elapsed_ms = started.elapsed().as_millis() as u64, "session compiled for one input size");
        coverage.report(model, None);
        crate::cache::keep_newest_parts(model, ep.as_str(), &key, SIZE_KEY_PREFIX, SIZES_KEPT);
        Ok(session)
    }
}

/// The input an engine of `model` on `provider` is first compiled for,
/// when it compiles each size: an upscaler's tile there, Depth's square.
/// `None` for a model with a fixed input.
fn first_input(model: prunr_models::ModelId, provider: &str) -> Option<[usize; 4]> {
    let descriptor = prunr_models::descriptor(model).filter(|d| !d.free_dims.is_empty())?;
    let side = match descriptor.category {
        prunr_models::ModelCategory::Upscale => crate::upscale::tile_for_provider(descriptor.recommended_tile?, provider),
        prunr_models::ModelCategory::Depth => crate::depth::DEPTH_LONG_SIDE,
        _ => return None,
    };
    Some([1, 3, side as usize, side as usize])
}

/// What a size's compile is kept under, beside the model's other sizes.
fn size_key(shape: &[usize]) -> String {
    format!("{SIZE_KEY_PREFIX}{}", shape.iter().map(usize::to_string).collect::<Vec<_>>().join("-"))
}

const SIZE_KEY_PREFIX: &str = "size-";

/// How many sizes of a model keep their compile on disk (Real-ESRGAN x4:
/// 130 MB each, Depth 190 MB).
const SIZES_KEPT: usize = 4;

/// `builder` with `model`'s free input sizes fixed to `shape`'s.
fn fix_sizes(builder: ort::session::builder::SessionBuilder, model: prunr_models::ModelId, shape: &[usize]) -> Result<ort::session::builder::SessionBuilder, Miss> {
    let dims = prunr_models::descriptor(model).map_or(&[][..], |d| d.free_dims);
    dims.iter().try_fold(builder, |builder, &(name, axis)| {
        let size = shape.get(axis).ok_or_else(|| Miss::Unusable(format!("{model:?}: input has no axis {axis} for {name}")))?;
        builder.with_dimension_override(name, *size as i64).map_err(|e| Miss::Unusable(format!("ORT fix input size {name} failed: {e}")))
    })
}

impl OrtEngine {
    /// Tries each available GPU provider in turn, then the CPU.
    /// `intra_threads`: ORT intra-op threads; for batch use
    /// num_cpus / rayon_workers.
    pub fn new(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::with_threads(model, intra_threads, false)
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Create a CPU-only engine. Instant — no GPU compilation.
    pub fn new_cpu_only(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::with_threads(model, intra_threads, true)
    }

    /// The GPU providers unless `cpu_only`, then the CPU.
    pub fn with_threads(model: ModelKind, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        Self::build_with(model, intra_threads, gpu_eps_for(cpu_only), cpu_only, GraphOptimizationLevel::Level3)
    }

    /// Each weights a session tries in turn (its lighter copy, when on
    /// disk, then its own) on the providers in `eps`, or on the CPU when
    /// `cpu_only` (`eps` is then empty); a GPU session that fails every
    /// weights retries on the CPU, which tries its own lighter copy (INT8)
    /// first. Weights no provider in `eps` would try are never read: on a
    /// machine without a GPU provider a session goes to the CPU without a
    /// read.
    fn build_with(
        model: ModelKind,
        intra_threads: usize,
        eps: &[EpKind],
        cpu_only: bool,
        level: GraphOptimizationLevel,
    ) -> Result<Self, CoreError> {
        tracing::debug!(?model, intra_threads, cpu_only, "OrtEngine init");
        let id: prunr_models::ModelId = model.into();
        let mut failed = Vec::new();
        for weights in [lighter_weights(id, cpu_only), Some(Weights::Fp32)].into_iter().flatten() {
            if let Some(walk) = (!cpu_only).then(|| Walk::without_a_try(id, weights, eps)).flatten() {
                failed.push(format!("{} weights: {}", weights.label(), CoreError::Inference(walk.reason())));
                continue;
            }
            let Some(bytes) = Self::weights_bytes(id, weights)? else {
                tracing::debug!(?model, ?weights, "lighter weights gone from disk — skipping them");
                continue;
            };
            match Self::build_session(&bytes, weights, intra_threads, model, eps, level) {
                Ok(mut engine) => {
                    tracing::info!(?model, ?weights, provider = %engine.pick.provider, "OrtEngine ready");
                    engine.pick.put_first(failed);
                    return Ok(engine);
                }
                Err(e) if cpu_only && weights == Weights::Fp32 => return Err(e),
                Err(e) => {
                    tracing::warn!(?model, ?weights, error = %e, "session failed — trying the next weights");
                    failed.push(format!("{} weights: {e}", weights.label()));
                }
            }
        }
        let mut engine = Self::build_with(model, intra_threads, &[], true, level)?;
        tracing::info!(?model, provider = %engine.pick.provider, "OrtEngine ready (CPU fallback after GPU failure)");
        engine.pick.put_first(failed);
        Ok(engine)
    }

    /// `model`'s session as the app builds it: the model's graph
    /// optimisation level, through the GPU providers unless `cpu_only`.
    pub fn for_model(model: prunr_models::ModelId, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        let descriptor = prunr_models::descriptor(model)
            .ok_or_else(|| CoreError::Model(format!("{model:?} is not a known model")))?;
        let kind = ModelKind::try_from(model)
            .map_err(|id| CoreError::Model(format!("{id:?} has no session kind")))?;
        if let Some(prunr_models::UpscaleModelKnobs { parts: Some(layout), input_name, .. }) = descriptor.upscale {
            let net = crate::upscale::blocks::NetShape { layout, image_input: input_name };
            return Self::build_parts(kind, net, intra_threads, cpu_only, crate::upscale::pick_optimization_level(descriptor));
        }
        Self::build_with(kind, intra_threads, gpu_eps_for(cpu_only), cpu_only, crate::upscale::pick_optimization_level(descriptor))
    }

    /// The provider this session runs on, its weights, and what failed
    /// before it.
    pub fn pick(&self) -> &SessionPick {
        &self.pick
    }

    /// `id`'s `weights` as a session loads them; `None` when a lighter
    /// copy has gone from disk. Bundled weights are shared with every
    /// session building at the time; a download missing its own weights
    /// is the "not installed" error.
    fn weights_bytes(id: prunr_models::ModelId, weights: Weights) -> Result<Option<std::sync::Arc<Vec<u8>>>, CoreError> {
        Ok(match weights {
            Weights::Fp32 => Some(
                prunr_models::resolve_bytes(id).ok_or_else(|| CoreError::Inference(prunr_models::not_installed_error(id)))?,
            ),
            Weights::Fp16 => prunr_models::model_fp16_bytes(id).map(std::sync::Arc::new),
            Weights::Int8 => prunr_models::model_int8_bytes(id).map(std::sync::Arc::new),
        })
    }

    /// A session of `model_bytes` on the first provider in `eps` that
    /// takes it, or on the CPU when `eps` is empty.
    fn build_session(model_bytes: &[u8], weights: Weights, intra_threads: usize, model: ModelKind, eps: &[EpKind], level: GraphOptimizationLevel) -> Result<Self, CoreError> {
        let model_id: prunr_models::ModelId = model.into();
        if eps.is_empty() {
            crate::cache::gc_stale_for_model(model_id, CPU_NAME);
            let session = commit_past_bad_cache(|| {
                let builder = Self::builder_with_base(intra_threads, level).map_err(|e| (e, None))?;
                let (builder, bytes, draft) = apply_ort_graph_cache(builder, model_bytes, model_id, CPU_NAME, weights, None);
                let suspect = cached_graph(matches!(bytes, Cow::Owned(_)), model_id, CPU_NAME, weights, None);
                let started = Instant::now();
                let session = builder
                    .with_execution_providers([cpu_provider()])
                    .map_err(|e| (CoreError::Inference(format!("ORT set CPU EP failed: {e}")), None))?
                    .commit_from_memory(&bytes)
                    .map_err(|e| (CoreError::Inference(format!("ORT session creation failed (CPU): {e}")), suspect))?;
                if let Some(draft) = draft { draft.publish(); }
                tracing::info!(?model, ep = CPU_NAME, elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
                Ok(session)
            })?;
            return Ok(Self {
                sessions: Sessions::One(Mutex::new(session)),
                pick: SessionPick { weights, ..SessionPick::on(CPU_NAME) },
                model_kind: model,
            });
        }

        // One EP per try, not a list: a list hides which EP won and lets a
        // crashing EP (DirectML's AbiCustomRegistry) abort before the CPU
        // fallback is reached.
        let (sessions, ep, failed) = walk_ladder("", model_id, weights, eps, |ep| {
            crate::cache::gc_stale_for_model(model_id, ep.as_str());
            if let Some(shape) = ep.sizes_each_input().then(|| first_input(model_id, ep.as_str())).flatten() {
                let recipe = SizeRecipe { model: model_id, ep, intra_threads, level };
                let session = recipe.session(model_bytes, &shape)?;
                return Ok(Sessions::Sized(Box::new(SizedSessions {
                    recipe,
                    weights,
                    first: shape.to_vec(),
                    current: Mutex::new(Some((shape.to_vec(), session))),
                })));
            }
            let builder = Self::builder_with_base(intra_threads, level).map_err(|e| Miss::Unusable(e.to_string()))?;
            let (builder, bytes_owner, draft) = if ep.keeps_optimized_graph() {
                apply_ort_graph_cache(builder, model_bytes, model_id, ep.as_str(), weights, None)
            } else {
                (builder, Cow::Borrowed(model_bytes), None)
            };
            let (builder, coverage) = crate::ep_coverage::watch(ep, builder);
            let mut built = ep.attach(builder, model_id, EpSite::Engine { threads: intra_threads, part: None })?;
            let started = Instant::now();
            let session = built.commit_from_memory(&bytes_owner).map_err(|e| Miss::Failed {
                error: format!("ORT session creation failed ({ep}): {e}"),
                suspect_cache: cached_graph(matches!(bytes_owner, Cow::Owned(_)), model_id, ep.as_str(), weights, None),
            })?;
            if let Some(draft) = draft { draft.publish(); }
            tracing::info!(?model, ?weights, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
            coverage.report(model_id, None);
            Ok(Sessions::One(Mutex::new(session)))
        }).map_err(|walk| CoreError::Inference(walk.reason()))?;
        Ok(Self {
            sessions,
            pick: SessionPick { weights, failed, ..SessionPick::on(ep.as_str()) },
            model_kind: model,
        })
    }

    /// A model cut into parts: every part on one provider, the first in
    /// the GPU ladder that takes them all, else the CPU. Parts skip the
    /// optimized-graph cache, which keys one graph per model.
    fn build_parts(model: ModelKind, net: crate::upscale::blocks::NetShape, intra_threads: usize, cpu_only: bool, level: GraphOptimizationLevel) -> Result<Self, CoreError> {
        let id: prunr_models::ModelId = model.into();
        let rss_before = crate::sysmem::process_rss_mb();
        let (onnx, norms) = crate::upscale::blocks::read_parts(id, net.layout.features)?;
        let commit_all = |ep: Option<EpKind>| -> Result<Vec<(&'static str, Session)>, Miss> {
            onnx.iter().map(|(key, bytes)| {
                let builder = Self::builder_with_base(intra_threads, level).map_err(|e| Miss::Unusable(e.to_string()))?;
                let (mut built, coverage) = match ep {
                    Some(ep) => {
                        let (builder, coverage) = crate::ep_coverage::watch(ep, builder);
                        (ep.attach(builder, id, EpSite::Engine { threads: intra_threads, part: Some(key) })?, coverage)
                    }
                    None => (
                        builder.with_execution_providers([cpu_provider()])
                            .map_err(|e| Miss::Unusable(format!("ORT register {CPU_NAME} EP failed: {e}")))?,
                        crate::ep_coverage::CoverageWatch::default(),
                    ),
                };
                let session = built.commit_from_memory(bytes)
                    .map_err(|e| Miss::Failed { error: format!("ORT session creation failed ({key}): {e}"), suspect_cache: None })?;
                coverage.report(id, Some(key));
                Ok((*key, session))
            }).collect()
        };
        let started = Instant::now();
        let (sessions, pick) = match walk_ladder("", id, Weights::Fp32, gpu_eps_for(cpu_only), |ep| commit_all(Some(ep))) {
            Ok((sessions, ep, failed)) => (sessions, SessionPick { failed, ..SessionPick::on(ep.as_str()) }),
            Err(walk) => {
                let sessions = commit_all(None).map_err(|miss| CoreError::Inference(miss_text(miss)))?;
                (sessions, SessionPick { failed: walk.failed, ..SessionPick::on(CPU_NAME) })
            }
        };
        tracing::info!(?model, parts = sessions.len(), provider = %pick.provider, elapsed_ms = started.elapsed().as_millis() as u64, "part sessions committed");
        let resident_mb = rss_before.zip(crate::sysmem::process_rss_mb()).map_or(0, |(before, after)| after.saturating_sub(before));
        let net = crate::upscale::blocks::BlockNet::new(sessions, norms, net, resident_mb * 1024 * 1024)?;
        Ok(Self { sessions: Sessions::Parts(Box::new(net)), pick, model_kind: model })
    }

    /// The model's parts, when it runs part by part.
    pub(crate) fn parts(&self) -> Option<&crate::upscale::blocks::BlockNet> {
        match &self.sessions {
            Sessions::Parts(net) => Some(net),
            Sessions::One(_) | Sessions::Sized(_) => None,
        }
    }

    fn builder_with_base(intra_threads: usize, level: GraphOptimizationLevel) -> Result<ort::session::builder::SessionBuilder, CoreError> {
        crate::ort_runtime::session_builder()?
            .with_optimization_level(level)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))
    }

    /// A startup guess at the provider, before any session exists: the
    /// first GPU provider the loaded runtime offers, else the CPU. Each
    /// session reports the real one.
    pub fn detect_active_provider() -> String {
        DETECTED_PROVIDER.get_or_init(|| {
            // The UI walks the ladder only once this has answered, so the
            // failure store it reads is read here, off the UI thread.
            crate::ep_compat::warm();
            available_gpu_eps()
                .first()
                .map(|ep| ep.as_str().to_string())
                .unwrap_or_else(|| CPU_NAME.to_string())
        }).clone()
    }

    /// The probe's answer once it is in, without waiting for it: a GPU
    /// driver can hold the probe for good.
    pub fn detected_provider() -> Option<String> {
        DETECTED_PROVIDER.get().cloned()
    }

    /// Whether the probe has answered.
    pub fn probed() -> bool {
        DETECTED_PROVIDER.get().is_some()
    }

}

static DETECTED_PROVIDER: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Whether sessions on `provider` run on the host CPU: the CPU provider,
/// and OpenVINO unless the user let its engines use the Intel GPU.
pub fn runs_on_host_cpu(provider: &str) -> bool {
    on_host_cpu(provider, openvino_on_gpu())
}

/// Whether OpenVINO engines in this process may use the Intel GPU.
pub fn openvino_on_gpu() -> bool {
    #[cfg(not(target_os = "macos"))]
    { intel_gpu_chosen() }
    #[cfg(target_os = "macos")]
    { false }
}

pub(crate) fn on_host_cpu(provider: &str, openvino_on_gpu: bool) -> bool {
    provider.eq_ignore_ascii_case(CPU_NAME)
        || (provider.eq_ignore_ascii_case(OPENVINO_NAME) && !openvino_on_gpu)
}

/// Whether `model` runs on the accelerator `provider`: the registry does
/// not place it on the CPU there, and some weights the ladder would load
/// there (FP16 first where a copy exists, then the model's own) have not
/// failed there on an earlier run. A model it does not run falls back to
/// the CPU by design.
pub fn model_runs_on(model: prunr_models::ModelId, provider: &str) -> bool {
    match available_gpu_eps().iter().find(|ep| ep.as_str().eq_ignore_ascii_case(provider)) {
        Some(&ep) => gpu_skip(model, ep).is_none(),
        None => !prunr_models::place(model, provider, openvino_on_gpu()).kept_off(false),
    }
}

/// Engines of one model build one after another: DirectML's session
/// creation is not thread-safe, and CoreML compiles each engine into the
/// model's one cache folder, where two at once left a half-written cache
/// that was then recorded as CoreML failing the model.
pub(crate) fn engines_build_one_at_a_time() -> bool {
    directml_active() || OrtEngine::detect_active_provider().eq_ignore_ascii_case(COREML_NAME)
}

/// Whether DirectML is the active GPU provider. Parallel session builds
/// go sequential under it: its `commit_from_memory` races in
/// AbiCustomRegistry on concurrent calls.
pub(crate) fn directml_active() -> bool {
    #[cfg(windows)]
    {
        OrtEngine::detect_active_provider()
            .eq_ignore_ascii_case(EpKind::DirectMl.as_str())
    }
    #[cfg(not(windows))]
    {
        false
    }
}

impl OrtEngine {
    /// Lock the ORT session: for a model whose input size is fixed, or to
    /// read what the session holds (input types). A run of a model whose
    /// size varies goes through `with_session_for`.
    pub(crate) fn with_session<T, F>(&self, f: F) -> Result<T, CoreError>
    where
        F: FnOnce(&mut Session) -> Result<T, CoreError>,
    {
        match &self.sessions {
            Sessions::Sized(sized) => sized.with_current(f),
            _ => f(&mut self.one_session()?.lock().unwrap_or_else(std::sync::PoisonError::into_inner)),
        }
    }

    /// Lock the session that runs an input of `shape`: on a provider that
    /// compiles each size, the one compiled for it, built first when the
    /// size is new.
    pub(crate) fn with_session_for<T, F>(&self, shape: &[usize], f: F) -> Result<T, CoreError>
    where
        F: FnOnce(&mut Session) -> Result<T, CoreError>,
    {
        match &self.sessions {
            Sessions::Sized(sized) => sized.with_session(shape, f),
            _ => self.with_session(f),
        }
    }

    /// The session, for a model that runs whole at one size.
    fn one_session(&self) -> Result<&Mutex<Session>, CoreError> {
        match &self.sessions {
            Sessions::One(session) => Ok(session),
            Sessions::Sized(_) => Err(CoreError::Inference(format!("{:?} runs a session per input size", self.model_kind))),
            Sessions::Parts(_) => Err(CoreError::Inference(format!("{:?} runs part by part, not as one session", self.model_kind))),
        }
    }

    /// Run a one-input f32 model on `data` shaped `shape`, returning its
    /// first output flat; errors unless it holds `expected_len` values.
    /// Bails before and after the run once `progress` is cancelled.
    pub(crate) fn run_single_f32(
        &self,
        label: &str,
        input: &str,
        shape: [usize; 4],
        data: Vec<f32>,
        expected_len: usize,
        progress: &crate::progress::Progress,
    ) -> Result<Vec<f32>, CoreError> {
        let array = ndarray::Array4::from_shape_vec(shape, data)
            .map_err(|e| CoreError::Inference(format!("{label}: input shape: {e}")))?;
        let tensor = Tensor::from_array(array).map_err(|e| CoreError::Inference(format!("{label}: input tensor: {e}")))?;
        progress.check_cancelled()?;
        let out = self.with_session_for(&shape, |session| {
            let outputs = session.run(ort::inputs![input => &tensor])
                .map_err(|e| CoreError::Inference(format!("{label}: inference failed: {e}")))?;
            let (_, values) = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| CoreError::Inference(format!("{label}: output: {e}")))?;
            Ok(values.to_vec())
        })?;
        if out.len() != expected_len {
            return Err(CoreError::Inference(format!("{label}: output has {} values, expected {expected_len}", out.len())));
        }
        progress.check_cancelled()?;
        Ok(out)
    }

    /// Run inference via `IoBinding` and pass the output to `f` as a
    /// borrow, alive only for `f`'s call. Skips the output clone of the
    /// `with_session` path (4 MB at BiRefNet 1024², ~11.9% of memcpy in
    /// the perf trace); use it when the caller can consume the view in
    /// place.
    pub(crate) fn infer_into<R, F>(
        &self,
        input_array: ndarray::Array4<f32>,
        f: F,
    ) -> Result<R, CoreError>
    where
        F: FnOnce(ndarray::ArrayView4<'_, f32>) -> Result<R, CoreError>,
    {
        let mut session = self.one_session()?.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| CoreError::Inference(format!("Failed to create input tensor: {e}")))?;

        let mut binding = session.create_binding()
            .map_err(|e| CoreError::Inference(format!("Failed to create IoBinding: {e}")))?;
        binding.bind_input(&input_name, &input_tensor)
            .map_err(|e| CoreError::Inference(format!("Failed to bind input: {e}")))?;

        // CPU output memory works for every EP we register today: CPU,
        // CUDA, CoreML, DirectML, OpenVINO. ORT inserts the device→host
        // copy before writing here.
        let mem_info = MemoryInfo::new(
            AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput,
        ).map_err(|e| CoreError::Inference(format!("MemoryInfo: {e}")))?;
        binding.bind_output_to_device(&output_name, &mem_info)
            .map_err(|e| CoreError::Inference(format!("Failed to bind output: {e}")))?;

        let outputs = session.run_binding(&binding)
            .map_err(|e| CoreError::Inference(format!("ORT run_binding failed: {e}")))?;
        let view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("Failed to extract output: {e}")))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("Output reshape error: {e}")))?;

        f(view)
    }
}

impl OrtEngine {
    /// The active execution provider's name ("CUDA", "CoreML", "CPU").
    pub fn active_provider(&self) -> &str {
        &self.pick.provider
    }
}

/// CoreML's compiled-model cache of `model` when `error` names a path in
/// it and the failed build, started at `since`, compiled nothing into it:
/// a compiled model left half-written is the cache's fault, not the
/// provider's. One the build had just compiled failed on its own, and
/// CoreML names its fresh files then too.
fn compiled_cache_named(model: prunr_models::ModelId, ep_name: &str, error: &str, since: std::time::SystemTime) -> Option<PathBuf> {
    let dir = (ep_name == COREML_NAME).then(|| crate::cache::cache_path_for(model, ep_name)).flatten();
    named_in(error, dir).filter(|dir| !written_since(dir, since))
}

fn named_in(error: &str, dir: Option<PathBuf>) -> Option<PathBuf> {
    dir.filter(|dir| error.contains(dir.to_string_lossy().as_ref()))
}

/// Whether a file under `dir` was written at `since` or later.
fn written_since(dir: &std::path::Path, since: std::time::SystemTime) -> bool {
    std::fs::read_dir(dir).into_iter().flatten().flatten().any(|entry| {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => written_since(&path, since),
            _ => entry.metadata().and_then(|meta| meta.modified()).is_ok_and(|modified| modified >= since),
        }
    })
}

/// Delete a cached optimized graph a session failed on, so the next
/// build rebuilds it. `NotFound` is silent: processes race on the path.
fn clear_suspect_cache(path: &std::path::Path) {
    let removed = if path.is_dir() { std::fs::remove_dir_all(path) } else { std::fs::remove_file(path) };
    match removed {
        Ok(()) => tracing::warn!(cache = %path.display(),
            "deleted suspect cache file after commit failure"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(cache = %path.display(), %e,
            "failed to delete suspect cache file"),
    }
}

/// The cached optimized graph a session loaded, when `hit` (an
/// `apply_ort_graph_cache` hit comes back as owned bytes).
pub(crate) fn cached_graph(hit: bool, model: prunr_models::ModelId, ep_name: &str, weights: Weights, part: Option<&str>) -> Option<PathBuf> {
    hit.then(|| optimized_graph_path(model, ep_name, weights, part)).flatten()
}

/// Where the optimized graph of `model` (or of its `part`) is kept.
fn optimized_graph_path(model: prunr_models::ModelId, ep_name: &str, weights: Weights, part: Option<&str>) -> Option<PathBuf> {
    match part {
        Some(part) => crate::cache::optimized_model_path_for_part(model, ep_name, part),
        None => crate::cache::optimized_model_path(model, ep_name, weights),
    }
}

/// Commit a CPU session, where `commit` answers a failure with the cached
/// graph it loaded, if any. A cached graph the CPU fails on (damaged, or
/// written on another machine) is deleted and the session committed again,
/// from the model: the CPU is the last provider, so a bad cache must not
/// stop a model loading. Running out of memory keeps the graph.
pub(crate) fn commit_past_bad_cache<T, E: std::fmt::Display>(mut commit: impl FnMut() -> Result<T, (E, Option<PathBuf>)>) -> Result<T, E> {
    let drafted = crate::cache::drafts_attached();
    match commit() {
        // Out of memory says nothing against the graph, and a retry would
        // meet the same shortage.
        Err((error, Some(_))) if crate::ep_compat::is_transient(&error.to_string()) => Err(error),
        Err((error, Some(cached))) => {
            tracing::warn!(%error, cache = %cached.display(), "CPU session failed on its cached graph; building it again");
            clear_suspect_cache(&cached);
            commit().map_err(|(error, _)| error)
        }
        Err((error, None)) if crate::cache::drafts_attached() > drafted => {
            tracing::warn!(%error, "CPU session failed while writing its optimized graph; building it without");
            crate::cache::without_drafts(&mut commit).map_err(|(error, _)| error)
        }
        done => done.map_err(|(error, _)| error),
    }
}

/// Per-(model, EP, weights) optimized-graph cache: a hit returns the
/// optimized bytes; a miss has ORT write them for the next launch.
/// Best-effort: any failure logs and falls back to the uncached path.
pub(crate) fn apply_ort_graph_cache<'a>(
    builder: ort::session::builder::SessionBuilder,
    model_bytes: &'a [u8],
    model_id: prunr_models::ModelId,
    ep_name: &str,
    weights: Weights,
    part: Option<&str>,
) -> (ort::session::builder::SessionBuilder, Cow<'a, [u8]>, Option<crate::cache::GraphDraft>) {
    let Some(path) = optimized_graph_path(model_id, ep_name, weights, part) else {
        return (builder, Cow::Borrowed(model_bytes), None);
    };
    match std::fs::read(&path) {
        Ok(cached) => {
            tracing::debug!(?model_id, ep = %ep_name, bytes = cached.len(), "cache hit (optimized graph)");
            return (builder, Cow::Owned(cached), None);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(?model_id, ep = %ep_name, %e, path = %path.display(),
                "cached optimized graph unreadable; rebuilding");
        }
    }
    let dir = match part {
        Some(part) => crate::cache::cache_dir_for_part(model_id, ep_name, part),
        None => crate::cache::cache_dir_for(model_id, ep_name),
    };
    if dir.is_none() {
        return (builder, Cow::Borrowed(model_bytes), None);
    }
    let (builder, draft) = crate::cache::attach_draft(builder, &path, model_bytes.len() as u64, model_id, ep_name, part);
    (builder, Cow::Borrowed(model_bytes), draft)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cached CPU graph the session cannot load (damaged, or written on
    /// another machine) is deleted and the session built from the model:
    /// the CPU is the last provider, so a bad cache must not stop a model.
    #[test]
    fn a_cpu_session_builds_past_a_broken_cached_graph() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let id = prunr_models::ModelId::Silueta;
        assert!(crate::cache::cache_dir_for(id, CPU_NAME).is_some());
        let broken: Vec<PathBuf> = Weights::ALL
            .into_iter()
            .filter_map(|w| crate::cache::optimized_model_path(id, CPU_NAME, w))
            .collect();
        for path in &broken {
            std::fs::write(path, b"not a graph").unwrap();
        }
        let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("the model loads past its broken cache");
        let used = crate::cache::optimized_model_path(id, CPU_NAME, engine.pick.weights).unwrap();
        assert_ne!(std::fs::read(&used).ok().as_deref(), Some(&b"not a graph"[..]), "the broken graph is gone");
    }

    /// Running out of memory on a cached graph says nothing against the
    /// graph: it stays for the next build, and the build is not retried
    /// into the same shortage.
    #[test]
    fn a_cpu_session_out_of_memory_keeps_its_cached_graph() {
        let dir = tempfile::tempdir().unwrap();
        let cached = dir.path().join("optimized.onnx");
        std::fs::write(&cached, b"graph").unwrap();
        let tries = std::cell::Cell::new(0);
        let built = commit_past_bad_cache(|| {
            tries.set(tries.get() + 1);
            Err::<(), _>(("Failed to allocate memory for requested buffer of size 1073741824", Some(cached.clone())))
        });
        assert!(built.is_err());
        assert!(cached.exists(), "the graph stays");
        assert_eq!(tries.get(), 1);
    }

    /// Erasers and models run in parts never land on MIGraphX, which
    /// recompiles per crop size; engines still use it.
    #[cfg(target_os = "linux")]
    #[test]
    fn erasers_and_parted_models_pass_migraphx_over_and_engines_keep_it() {
        for d in prunr_models::REGISTRY.iter() {
            let per_shape = d.category == prunr_models::ModelCategory::Inpaint || d.runs_in_parts();
            let kept = per_shape || prunr_models::place(d.id, "MIGraphX", false).kept_off(false);
            assert_eq!(matches!(skip_reason(d.id, Weights::Fp32, EpKind::MiGraphX), Some(Skip::KeptOff(_))), kept, "{:?}", d.id);
            assert!(!compiles_per_shape(d.id, EpKind::Cuda));
        }
    }

    /// With Intel graphics on, OpenVINO engines are sized like a GPU's;
    /// without it they share the host CPU.
    #[test]
    fn openvino_is_the_host_cpu_unless_it_may_use_the_intel_gpu() {
        assert!(on_host_cpu(OPENVINO_NAME, false));
        assert!(!on_host_cpu(OPENVINO_NAME, true));
        assert!(on_host_cpu(CPU_NAME, true), "the CPU provider is the CPU either way");
        assert!(!on_host_cpu("CUDA", false));
    }

    /// Cut-out and upscale engines ask OpenVINO for the GPU, with the CPU
    /// behind it, only when the user chose it; fills never do, and an SD
    /// part keeps the device it pins.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn only_engines_ask_openvino_for_the_gpu_and_only_when_chosen() {
        for chosen in [false, true] {
            assert_eq!(EpSite::Engine { threads: 4, part: None }.ov_device(chosen), chosen.then_some("AUTO:GPU,CPU"));
            assert_eq!(EpSite::Fill { threads: 4 }.ov_device(chosen), None);
            assert_eq!(EpSite::SdPart { part: "unet", ov_device: None }.ov_device(chosen), None);
            assert_eq!(EpSite::SdPart { part: "unet", ov_device: Some("GPU") }.ov_device(chosen), Some("GPU"));
        }
    }

    /// A GPU runtime that cannot start knows which library is missing, so
    /// the app can offer it, and tells a missing driver or card apart.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_gpu_runtime_that_cannot_start_names_what_is_missing() {
        let why = |stack, err: &str| GpuStartFailure::of(stack, err).why;
        let failed = |lib: &str| format!("ORT register CUDA EP failed: provider_bridge_ort.cc:1957 [ONNXRuntimeError] : 1 : FAIL : Failed to load library /x/libonnxruntime_providers_cuda.so with error: {lib}: cannot open shared object file: No such file or directory");
        assert_eq!(why(GpuStack::Cuda, &failed("libcublasLt.so.12")), StartFailure::MissingLibrary("libcublasLt.so.12".into()));
        assert_eq!(why(GpuStack::Cuda, &failed("libcuda.so.1")), StartFailure::NoDriver);
        let parse_failed = |cuda: &str| format!("Parse Failed to parse provider option \"device_id\": {cuda} ; GPU=-1 ; expr=cudaGetDeviceCount(&num_devices);");
        assert_eq!(why(GpuStack::Cuda, &parse_failed("CUDA failure 35: CUDA driver version is insufficient for CUDA runtime version")), StartFailure::NoDriver);
        assert_eq!(why(GpuStack::Cuda, &parse_failed("CUDA failure 100: no CUDA-capable device is detected")), StartFailure::NoDevice);
        assert_eq!(why(GpuStack::Rocm, &failed("libmigraphx_c.so.3")), StartFailure::MissingLibrary("libmigraphx_c.so.3".into()));
        assert_eq!(why(GpuStack::Rocm, "hipErrorNoDevice"), StartFailure::NoDevice);
        assert_eq!(why(GpuStack::Cuda, "out of memory"), StartFailure::Other);
        assert_eq!(
            GpuStartFailure::of(GpuStack::Cuda, &failed("libcudnn.so.9")).text(),
            "CUDA could not start: libcudnn.so.9 is missing. It needs NVIDIA's CUDA 12 and cuDNN 9 libraries.",
        );
        assert_eq!(
            GpuStartFailure::of(GpuStack::Rocm, &failed("libamdhip64.so.7")).text(),
            "MIGraphX could not start: libamdhip64.so.7 is missing. It needs AMD's ROCm 7 with MIGraphX installed.",
        );
    }

    /// A CoreML failure that names a path in the model's compiled cache
    /// (seen: "A valid manifest does not exist at path: <cache>/…") makes
    /// the cache the suspect; any other failure leaves the provider's.
    #[test]
    fn a_coreml_error_naming_its_cache_makes_the_cache_the_suspect() {
        let dir = std::path::PathBuf::from("/c/prunr/ep_cache/coreml/silueta-1.0.0-v1");
        let manifest = format!("Exception during initialization: A valid manifest does not exist at path: {}/0_dynamic_mlprogram/model/Manifest.json", dir.display());
        assert_eq!(named_in(&manifest, Some(dir.clone())), Some(dir.clone()));
        assert_eq!(named_in("Error in building plan", Some(dir.clone())), None);
        assert_eq!(named_in(&manifest, None), None, "no cache for the provider");
        let cache_dir = std::env::temp_dir().join(format!("prunr-suspect-dir-{}", std::process::id()));
        std::fs::create_dir_all(cache_dir.join("0_dynamic_mlprogram")).unwrap();
        clear_suspect_cache(&cache_dir);
        assert!(!cache_dir.exists(), "a suspect cache folder is removed whole");
    }

    /// A model CoreML cannot plan fails each fresh compile with the
    /// compiled file's path in the error; were that read as a bad cache,
    /// the failure would never be recorded and every launch would
    /// compile the model again.
    #[test]
    fn a_coreml_model_that_fails_its_fresh_compile_is_not_a_bad_cache() {
        let dir = std::env::temp_dir().join(format!("prunr-fresh-compile-{}", std::process::id()));
        let compiled = dir.join("1452/1_dynamic_mlprogram/model/compiled_model.mlmodelc");
        std::fs::create_dir_all(&compiled).unwrap();
        let mil = compiled.join("model.mil");
        let build_started = std::time::SystemTime::now() - std::time::Duration::from_secs(5);

        std::fs::write(&mil, b"program").unwrap();
        assert!(written_since(&dir, build_started), "compiled by the failed build: the provider's failure");

        let cached = std::fs::File::options().write(true).open(&mil).unwrap();
        cached.set_modified(build_started - std::time::Duration::from_secs(3600)).unwrap();
        assert!(!written_since(&dir, build_started), "compiled before the build: the cache is the suspect");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A size that fails to compile leaves no session; reading the session
    /// then compiles the first size again instead of failing for good.
    #[test]
    fn a_sized_engine_compiles_its_first_size_again_after_a_size_fails() {
        let id = prunr_models::ModelId::DepthAnythingV2Small;
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let Some(&ep) = available_gpu_eps().first().filter(|_| prunr_models::is_available(id)) else {
            eprintln!("SKIP: no GPU provider or no depth model");
            return;
        };
        let first = vec![1, 3, 518, 518];
        let sized = SizedSessions {
            recipe: SizeRecipe { model: id, ep, intra_threads: 2, level: GraphOptimizationLevel::Level3 },
            weights: Weights::Fp32,
            first: first.clone(),
            current: Mutex::new(None),
        };
        let input_name = |session: &mut Session| Ok(session.inputs()[0].name().to_string());
        assert!(sized.with_session(&[1, 3], input_name).is_err(), "a size without the input's axes fails");
        assert_eq!(sized.with_current(input_name).expect("the first size compiles again"), "pixel_values");
        let current = sized.current.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().map(|(size, _)| size.clone());
        assert_eq!(current, Some(first));
    }

    /// A model given free sizes in the registry that the engine has no
    /// first size for would compile for any size again, silently.
    #[test]
    fn every_model_with_free_sizes_has_a_first_size() {
        for d in prunr_models::REGISTRY.iter().filter(|d| !d.free_dims.is_empty()) {
            assert!(first_input(d.id, COREML_NAME).is_some(), "{:?} has free sizes and no first size", d.id);
        }
    }

    #[test]
    fn a_size_is_compiled_first_at_the_input_its_engine_starts_with() {
        use prunr_models::ModelId;
        assert_eq!(first_input(ModelId::DepthAnythingV2Small, COREML_NAME), Some([1, 3, 518, 518]));
        let x4_tile = crate::upscale::tile_for_provider(prunr_models::descriptor(ModelId::RealEsrganX4Plus).and_then(|d| d.recommended_tile).unwrap(), COREML_NAME) as usize;
        assert_eq!(first_input(ModelId::RealEsrganX4Plus, COREML_NAME), Some([1, 3, x4_tile, x4_tile]));
        assert_eq!(first_input(ModelId::U2net, COREML_NAME), None, "a fixed input is never resized");
        assert_eq!(size_key(&[1, 3, 392, 518]), "size-1-3-392-518");
    }

    #[test]
    fn clear_suspect_cache_removes_existing_file_and_handles_missing() {
        let dir = std::env::temp_dir().join("prunr-clear-suspect-cache-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("optimized.onnx");
        std::fs::write(&path, b"stale").unwrap();
        clear_suspect_cache(&path);
        assert!(!path.exists(), "existing file must be removed");
        // NotFound must be a silent no-op — concurrent deletions are
        // expected, and a panic here would abort the whole session build
        // instead of letting it rebuild from source.
        clear_suspect_cache(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Boundary contract: a session that failed on a cached optimized
    /// graph deletes the graph and does NOT record the provider as
    /// failing the model. A malformed compiled.onnx from the EPContext
    /// path would otherwise persistently disable OpenVINO for SD, until
    /// `ep_compat.json` was removed by hand.
    ///
    /// Uses `(EpKind::Cuda, ModelId::TaesdFp16)` to dodge other tests
    /// that touch `ep_compat` under the parallel runner.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_failure_on_a_cached_graph_deletes_it_instead_of_recording_the_provider() {
        use prunr_models::ModelId;
        let id = ModelId::TaesdFp16;
        let ep = EpKind::Cuda;
        let Some(dir) = crate::cache::cache_dir_for(id, ep.as_str()) else { return };
        let cache_file = dir.join("optimized.onnx");
        std::fs::write(&cache_file, b"stale").unwrap();
        let initial = crate::ep_compat::is_known_failure(ep, id, Weights::Fp32);

        let walk = walk_ladder("", id, Weights::Fp32, &[ep], |_| {
            Err::<(), _>(Miss::Failed { error: "test cache failure".into(), suspect_cache: Some(cache_file.clone()) })
        });

        assert_eq!(walk.map_err(|w| w.failed), Err(vec!["test cache failure".to_string()]));
        assert!(!cache_file.exists(), "cache file must be deleted");
        assert_eq!(
            crate::ep_compat::is_known_failure(ep, id, Weights::Fp32), initial,
            "a failure on a cached graph must not be recorded",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn coreml_compiles_ml_programs_into_the_given_cache() {
        let dir = std::path::Path::new("/Users/ada/Library/Caches/prunr/ep_cache/coreml/u2net-1-ort1.24.1-v1");
        let options = format!("{:?}", coreml_provider(Some(dir)));
        assert!(options.contains(r#""ModelFormat": "MLProgram""#), "{options}");
        assert!(options.contains(r#""MLComputeUnits": "CPUAndGPU""#), "{options}");
        assert!(options.contains(&format!(r#""ModelCacheDirectory": {:?}"#, dir.to_string_lossy())), "{options}");
        assert!(!format!("{:?}", coreml_provider(None)).contains("ModelCacheDirectory"));
    }

    #[test]
    fn coreml_sessions_skip_the_fusion_that_inlines_weights() {
        assert_eq!(disabled_optimizers_on(COREML_NAME), ["MatMulAddFusion"]);
        assert!(disabled_optimizers_on(CUDA_NAME).is_empty());
        assert!(disabled_optimizers_on(OPENVINO_NAME).is_empty());
    }

    /// A provider the loaded runtime lacks must fail its registration,
    /// not hand back a builder that commits on the CPU while the ladder
    /// labels the session with the provider's name. Every site counts:
    /// each ladder registers through `EpKind::provider`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn missing_provider_fails_registration_instead_of_running_on_cpu() {
        use ort::ep::ExecutionProvider;
        if crate::ort_runtime::ensure_initialized().is_err()
            || ort::ep::CUDA::is_available().unwrap_or(true)
        {
            return;
        }
        let sites = [
            EpSite::Engine { threads: 1, part: None },
            EpSite::Engine { threads: 1, part: Some("block01") },
            EpSite::Fill { threads: 1 },
            EpSite::SdPart { part: "unet", ov_device: None },
        ];
        for site in sites {
            let builder = crate::ort_runtime::session_builder().expect("runtime loaded above");
            let registered = builder.with_execution_providers([
                EpKind::Cuda.provider(prunr_models::ModelId::Silueta, site),
            ]);
            assert!(registered.is_err(), "{site:?}: CUDA registered on a runtime without it");
        }
    }

    /// The weights fallback within one provider: FP16 failing there moves
    /// on to the FP32 weights on the same provider, and only the FP16
    /// failure is remembered.
    #[test]
    fn fp16_failing_on_a_provider_falls_back_to_fp32_there() {
        use prunr_models::ModelId;
        #[cfg(not(target_os = "macos"))]
        let ep = EpKind::OpenVino;
        #[cfg(target_os = "macos")]
        let ep = EpKind::CoreMl;
        let model = ModelId::FourXNmkdSuperscale;
        let tried = std::cell::RefCell::new(Vec::new());
        let commit_fp16_fails = |weights: Weights| {
            let tried = &tried;
            move |ep: EpKind| {
                tried.borrow_mut().push((ep, weights));
                if weights == Weights::Fp16 {
                    Err(Miss::Failed { error: "tensor(float16) is invalid".into(), suspect_cache: None })
                } else {
                    Ok("session")
                }
            }
        };
        assert!(walk_ladder("", model, Weights::Fp16, &[ep], commit_fp16_fails(Weights::Fp16)).is_err());
        let picked = walk_ladder("", model, Weights::Fp32, &[ep], commit_fp16_fails(Weights::Fp32));
        assert_eq!(picked, Ok(("session", ep, Vec::new())), "FP32 ran on the provider FP16 failed on");
        assert_eq!(*tried.borrow(), [(ep, Weights::Fp16), (ep, Weights::Fp32)]);
        assert!(crate::ep_compat::is_known_failure(ep, model, Weights::Fp16), "the FP16 failure is remembered");
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32), "and only that");
    }

    /// A session says where it landed: a CPU-only one on the CPU, with
    /// nothing failed on the way and never the GPU's FP16 weights.
    #[test]
    fn a_cpu_session_names_the_cpu_and_its_weights() {
        use prunr_models::ModelId;
        assert!(OrtEngine::for_model(ModelId::Teed, 1, true).is_err(), "TEED has no engine kind");
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let engine = OrtEngine::for_model(ModelId::Silueta, 1, true).expect("Silueta is bundled");
        let pick = engine.pick();
        assert_eq!((pick.provider.as_str(), engine.active_provider()), ("CPU", "CPU"));
        assert!(pick.failed.is_empty(), "{:?}", pick.failed);
        assert_ne!(pick.weights, Weights::Fp16);
        assert_eq!(engine.model_kind(), ModelKind::Silueta);
    }

    /// A session that fails while writing its optimized graph (a disk
    /// filled up meanwhile) is built again without it, and the failure is
    /// not held against the provider: the cache must never cost the model.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_session_that_fails_writing_its_graph_builds_again_without_it() {
        use prunr_models::ModelId;
        let (ep, model) = (EpKind::Cuda, ModelId::Migan);
        let tries = std::cell::Cell::new(0);
        let walk = walk_ladder("", model, Weights::Fp32, &[ep], |_| {
            tries.set(tries.get() + 1);
            if crate::cache::take_draft_slot() {
                Err(Miss::Failed { error: "ORT session creation failed (CUDA): Protobuf serialization failed.".into(), suspect_cache: None })
            } else {
                Ok("session")
            }
        });
        assert_eq!(walk, Ok(("session", ep, Vec::new())));
        assert_eq!(tries.get(), 2);
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32));
        let cpu = commit_past_bad_cache(|| if crate::cache::take_draft_slot() { Err(("write failed", None)) } else { Ok("cpu session") });
        assert_eq!(cpu, Ok("cpu session"), "the CPU's commit too");
    }

    /// A provider that cannot start on this machine (DirectML without a
    /// DirectX 12 GPU) is passed over without a cache record, and the walk
    /// fails, so the engine builds its CPU session from the same runtime.
    #[test]
    fn a_provider_that_cannot_start_is_not_held_against_the_model() {
        use prunr_models::ModelId;
        #[cfg(not(target_os = "macos"))]
        let ep = EpKind::OpenVino;
        #[cfg(target_os = "macos")]
        let ep = EpKind::CoreMl;
        let model = ModelId::FourXNmkdSiaxCx;
        let walk = walk_ladder("", model, Weights::Fp32, &[ep], |_| Err::<(), _>(Miss::Unusable("no DirectX 12 device".into())));
        assert_eq!(walk.map_err(|w| w.reason()), Err("no DirectX 12 device".to_string()));
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32), "the machine's lack, not the model's");
    }

    /// A provider is passed over without a try when the registry keeps
    /// the model off it or it failed before; the walk says so when it
    /// tried none.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_walk_that_tries_no_provider_says_why() {
        use prunr_models::ModelId;
        let model = ModelId::Silueta;
        let prunr_models::Place::Cpu(why) = prunr_models::place(model, "OpenVINO", false) else { panic!("Silueta stays off OpenVINO") };
        let line = format!("no GPU provider to try: OpenVINO is kept off: {}", why.text());
        let declared = walk_ladder("", model, Weights::Fp32, &[EpKind::OpenVino], |_| -> Result<(), Miss> { panic!("tried a declared pair") });
        assert_eq!(declared.map_err(|w| w.reason()), Err(line.clone()));
        let none = walk_ladder("", model, Weights::Fp32, &[], |_| -> Result<(), Miss> { unreachable!() });
        assert_eq!(none.map_err(|w| w.reason()), Err("No GPU EPs available on this platform".to_string()));
        assert_eq!(gpu_skip(model, EpKind::OpenVino), Some(Skip::KeptOff(why)));
        assert_eq!(
            Walk::without_a_try(model, Weights::Fp32, &[EpKind::OpenVino]).map(|w| w.reason()),
            Some(line),
            "the same line, known before any weights are read",
        );
        assert_eq!(Walk::without_a_try(ModelId::RealEsrganX4Plus, Weights::Fp32, &[EpKind::OpenVino]), None, "OpenVINO would be tried");
    }

    /// Sessions of one model built at once all build: none reads the
    /// optimized graph another is still writing, and the graph cached
    /// after them loads.
    #[test]
    fn sessions_built_at_once_never_read_a_half_written_graph() {
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let build = || OrtEngine::build_with(ModelKind::Silueta, 1, &[], false, GraphOptimizationLevel::Level3).map(|_| ());
        let results: Vec<_> = std::thread::scope(|s| {
            // Staggered: later builds start while the first writes.
            let handles: Vec<_> = (0..12u64).map(|i| s.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(i * 15));
                build()
            })).collect();
            handles.into_iter().map(|h| h.join().expect("no panic")).collect()
        });
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        assert!(build().is_ok(), "the cached graph loads");
    }

    /// A GPU session on a machine without a GPU provider lands on the CPU,
    /// naming for each weights it would have tried why none was, and reads
    /// none of them on the way.
    #[test]
    fn a_gpu_session_without_a_gpu_provider_lands_on_the_cpu_and_says_why() {
        use prunr_models::ModelId;
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let engine = OrtEngine::build_with(ModelKind::Silueta, 1, &[], false, GraphOptimizationLevel::Level3)
            .expect("Silueta is bundled");
        let none = "Inference error: No GPU EPs available on this platform";
        let mut expected = Vec::new();
        if lighter_weights(ModelId::Silueta, false).is_some() {
            expected.push(format!("FP16 weights: {none}"));
        }
        expected.push(format!("FP32 weights: {none}"));
        assert_eq!(engine.pick().provider, "CPU");
        assert_eq!(engine.pick().failed, expected);
        assert_eq!(engine.pick().weights, lighter_weights(ModelId::Silueta, true).unwrap_or(Weights::Fp32));
    }

    /// Level2 is a valid value for all bundled models — the constructor must not
    /// reject it for non-HAT models. Exercises the no-regression case: explicitly
    /// passing Level2 to a segmentation model succeeds just as Level3 does.
    ///
    /// Gated on `dev-models` because Silueta is bundled only in that feature.
    #[cfg(feature = "dev-models")]
    #[test]
    fn level2_constructs_for_existing_model() {
        let engine = OrtEngine::build_with(
            ModelKind::Silueta,
            1,
            gpu_eps_for(false),
            false,
            GraphOptimizationLevel::Level2,
        )
        .expect("Level2 must succeed for Silueta");
        assert!(!engine.active_provider().is_empty());
    }
}
