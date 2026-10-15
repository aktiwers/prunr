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

/// Trait for inference backends. Implemented by OrtEngine.
/// Send + Sync required because worker threads own the engine instance.
pub trait InferenceEngine: Send + Sync {
    /// Returns the name of the active execution provider (e.g., "CUDA", "CoreML", "CPU").
    fn active_provider(&self) -> &str;
}

/// GPU execution-provider variants. Variants are platform-gated so the
/// EP-ladder match sites stay exhaustive without dead arms — CoreML
/// only exists on macOS, DirectML on Windows, CUDA + OpenVINO on
/// non-macOS targets. `Display` mirrors the historic string names so
/// log output and the persistent `ep_compat.json` cache keys stay
/// stable across this refactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EpKind {
    #[cfg(not(target_os = "macos"))]
    OpenVino,
    #[cfg(not(target_os = "macos"))]
    Cuda,
    #[cfg(target_os = "macos")]
    CoreMl,
    #[cfg(windows)]
    DirectMl,
}

/// OpenVINO's provider name, defined on every platform so code that
/// reads a provider string needs no per-platform branch.
pub(crate) const OPENVINO_NAME: &str = "OpenVINO";

impl EpKind {
    /// Stable display string used by logs, the active-provider label,
    /// and the `ep_compat.json` cache keys. Returned as `&'static str`
    /// so callers (e.g. `is_ep_compatible`) don't trigger an allocation
    /// just to read the EP name.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => OPENVINO_NAME,
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => "CUDA",
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => "CoreML",
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

/// Filter the GPU EP ladder by `is_available()` — on `load-dynamic`
/// builds the loaded `libonnxruntime` may not have all EPs compiled in
/// (e.g. the OpenVINO PyPI wheel has CPU + OpenVINO only). Cached for
/// the process lifetime; first call probes ort's EP table.
///
/// Order on Linux: OpenVINO before CUDA so Intel users (the most common
/// non-NVIDIA Linux desktop population) hit OpenVINO first when their
/// Runtime Store install wins. NVIDIA users still get CUDA because
/// OpenVINO Runtime won't be installed and `is_available()` returns false.
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
            if ort::ep::CoreML::default()
                .is_available().unwrap_or(false) { eps.push(EpKind::CoreMl); }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if ort::ep::OpenVINO::default()
                .is_available().unwrap_or(false) { eps.push(EpKind::OpenVino); }
            if ort::ep::CUDA::default()
                .is_available().unwrap_or(false) { eps.push(EpKind::Cuda); }
            #[cfg(windows)]
            if ort::ep::DirectML::default()
                .is_available().unwrap_or(false) && directml_starts() { eps.push(EpKind::DirectMl); }
        }
        let eps_str: Vec<&'static str> = eps.iter().map(EpKind::as_str).collect();
        tracing::info!(eps = ?eps_str, "Available GPU execution providers");
        eps
    }).as_slice()
}

/// Whether DirectML can start a device here. The DirectML runtime, which
/// Windows ships built in, lists DirectML on every machine; registering
/// it is what creates the device, and that fails without a DirectX 12
/// GPU (a VM, a remote desktop), where sessions then run on the CPU of
/// the same runtime and the label says so.
#[cfg(windows)]
fn directml_starts() -> bool {
    let started = crate::ort_runtime::session_builder()
        .map_err(|e| e.to_string())
        .and_then(|b| b.with_execution_providers([directml_provider()]).map(drop).map_err(|e| e.to_string()));
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

/// The session family a provider is registered for; each keeps its own
/// tuning of the providers.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // each field tunes a provider that only some platforms compile in
pub(crate) enum EpSite<'a> {
    /// Segmentation and upscale engines.
    Engine { threads: usize },
    /// LaMa-class fills.
    Fill { threads: usize },
    /// One part of a Stable Diffusion bundle. `ov_device` pins OpenVINO's
    /// device (bench only).
    SdPart { part: &'a str, ov_device: Option<&'a str> },
}

impl EpKind {
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
                    EpSite::Engine { .. } => p
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default)
                        .with_cuda_graph(true)
                        .with_tf32(true)
                        .build(),
                    EpSite::Fill { .. } | EpSite::SdPart { .. } => p.build(),
                }
            }
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => {
                let dir = match site {
                    EpSite::SdPart { part, .. } => crate::cache::cache_dir_for_part(model, self.as_str(), part),
                    EpSite::Engine { .. } | EpSite::Fill { .. } => crate::cache::cache_dir_for(model, self.as_str()),
                };
                coreml_provider(dir.as_deref()).build()
            }
            #[cfg(windows)]
            EpKind::DirectMl => directml_provider(),
            // NOT using `with_cache_dir`: retested against SD UNet on
            // ort=2.0.0-rc.12 + openvino-1.24.1, Intel HD 530. Cold and
            // warm session-commit elapsed are identical within noise
            // (cold 23735 ms / warm 23224 ms on UNet), and every blob's
            // mtime advances on the warm run — OpenVINO computes a
            // different cache key each launch and rewrites ~3.3 GB on
            // every cold build. Re-evaluate when ort bumps.
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => match site {
                // Cap the EP-internal TBB pool to our outer rayon budget.
                // Uncapped, OpenVINO sizes its pool to all logical cores
                // and oversubscribes against the rayon workers —
                // `sched_yield` was ~2% of CLI batch wall time.
                EpSite::Engine { threads } | EpSite::Fill { threads } => ort::ep::OpenVINO::default()
                    .with_num_threads(threads.max(1))
                    .build(),
                // `num_streams=1` disables per-stream buffer duplication
                // (the bundle sits at ~16 GB RSS on an i7-6700 either
                // way; measured 2026-10). Dynamic shapes cost nothing per
                // step, so one bundle serves 512² tiles and the 512×768
                // crop.
                EpSite::SdPart { ov_device, .. } => {
                    let mut p = ort::ep::OpenVINO::default()
                        .with_num_streams(1)
                        .with_dynamic_shapes(true);
                    if let Some(dev) = ov_device {
                        p = p.with_device_type(dev);
                    }
                    p.build()
                }
            },
        };
        dispatch.error_on_failure()
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
/// Program runs on the GPU in fp32, never on the Neural Engine.
///
/// CoreML reuses what it compiled under `cache_dir` without checking
/// input shapes: a session that pins its input dims must key the dir on
/// them, or a model compiled for one tile size is reused for another.
#[cfg(any(target_os = "macos", test))]
fn coreml_provider(cache_dir: Option<&std::path::Path>) -> ort::ep::CoreML {
    let p = ort::ep::CoreML::default().with_model_format(ort::ep::coreml::ModelFormat::MLProgram);
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
    /// The registry declares the model incompatible with it.
    Declared,
    /// It failed these weights of the model on an earlier run; skipped
    /// until `prunr --clear-ep-cache`.
    FailedBefore,
}

impl Skip {
    fn text(self, ep: EpKind) -> String {
        match self {
            Skip::Declared => format!("{ep} is declared incompatible"),
            Skip::FailedBefore => format!("{ep} failed this model before"),
        }
    }
}

/// Whether, and why, the ladder passes `ep` over for these weights.
fn skip_reason(model: prunr_models::ModelId, weights: Weights, ep: EpKind) -> Option<Skip> {
    if !prunr_models::is_ep_compatible(model, ep.as_str()) {
        Some(Skip::Declared)
    } else if crate::ep_compat::is_known_failure(ep, model, weights) {
        Some(Skip::FailedBefore)
    } else {
        None
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
    let mut walk = Walk::default();
    for &ep in eps {
        if let Some(skip) = skip_reason(model, weights, ep) {
            tracing::debug!(?model, ?weights, ep = %ep, ?skip, "{prefix}EP skipped");
            walk.skipped.push(skip.text(ep));
            continue;
        }
        match commit(ep) {
            Ok(session) => return Ok((session, ep, walk.failed)),
            Err(Miss::Unusable(error)) => {
                tracing::warn!(?model, ep = %ep, %error, "{prefix}GPU EP unusable — trying next EP");
                walk.failed.push(error);
            }
            Err(Miss::Failed { error, suspect_cache }) => {
                tracing::warn!(?model, ?weights, ep = %ep, %error, "{prefix}GPU session failed — trying next EP");
                match suspect_cache {
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
pub fn gpu_provider_names() -> Vec<&'static str> {
    available_gpu_eps().iter().map(EpKind::as_str).collect()
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
    let mut why = None;
    for weights in [lighter_weights(model, false), Some(Weights::Fp32)].into_iter().flatten() {
        why = Some(skip_reason(model, weights, ep)?);
    }
    why
}

/// Each GPU provider the loaded runtime offers, in ladder order, with why
/// a session of `model` would pass it over (`None`: it would try it).
pub fn gpu_ladder(model: prunr_models::ModelId) -> Vec<(&'static str, Option<Skip>)> {
    available_gpu_eps().iter().map(|&ep| (ep.as_str(), gpu_skip(model, ep))).collect()
}

/// The weights file a session loaded: the model's own (FP32 for
/// segmentation and upscale; an SD bundle's own half-precision weights
/// count as these too), or the FP16 (GPU) or INT8 (CPU) variant beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Weights {
    #[default]
    Fp32,
    Fp16,
    Int8,
}

impl Weights {
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

/// ORT-backed inference engine. Holds one Session per model selection.
/// Create once per model; reuse across all images — never instantiate per-image.
///
/// The session is behind a Mutex because ort's `Session::run()` takes `&mut self`
/// internally (ORT updates state during inference). The Mutex enables shared `&OrtEngine`
/// references while still satisfying ort's mutability requirement.
pub struct OrtEngine {
    session: Mutex<Session>,
    pick: SessionPick,
    model_kind: ModelKind,
}

impl OrtEngine {
    /// Create a new OrtEngine for the given model.
    ///
    /// - `model`: Which ONNX model to load (Silueta = ~4MB fast, U2net = ~170MB quality)
    /// - `intra_threads`: ORT intra-op thread count. For batch use: num_cpus / rayon_workers.
    ///   Pass 1 for single-image use; the thread count is set at session creation time.
    ///
    /// Tries each available GPU provider in turn, then the CPU. Call
    /// active_provider() after creation to see which one won.
    pub fn new(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::new_with_fallback(model, intra_threads, false)
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Create a CPU-only engine. Instant — no GPU compilation.
    pub fn new_cpu_only(model: ModelKind, intra_threads: usize) -> Result<Self, CoreError> {
        Self::new_with_fallback(model, intra_threads, true)
    }

    fn new_with_fallback(model: ModelKind, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        Self::new_with_fallback_and_level(model, intra_threads, cpu_only, GraphOptimizationLevel::Level3)
    }

    /// Core session-creation path shared by `new`, `new_cpu_only`, and
    /// `for_model`: the GPU providers unless `cpu_only`, then the CPU.
    fn new_with_fallback_and_level(
        model: ModelKind,
        intra_threads: usize,
        cpu_only: bool,
        level: GraphOptimizationLevel,
    ) -> Result<Self, CoreError> {
        Self::build_with(model, intra_threads, gpu_eps_for(cpu_only), cpu_only, level)
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
        Self::new_with_fallback_and_level(kind, intra_threads, cpu_only, crate::upscale::pick_optimization_level(descriptor))
    }

    /// The provider this session runs on, its weights, and what failed
    /// before it.
    pub fn pick(&self) -> &SessionPick {
        &self.pick
    }

    /// `id`'s `weights` as a session loads them; `None` when a lighter
    /// copy has gone from disk. Bundled weights are zero-copy; a download
    /// missing its own weights is the "not installed" error.
    fn weights_bytes(id: prunr_models::ModelId, weights: Weights) -> Result<Option<Cow<'static, [u8]>>, CoreError> {
        Ok(match weights {
            Weights::Fp32 => Some(
                prunr_models::resolve_bytes(id).ok_or_else(|| CoreError::Inference(prunr_models::not_installed_error(id)))?,
            ),
            Weights::Fp16 => prunr_models::model_fp16_bytes(id).map(Cow::Owned),
            Weights::Int8 => prunr_models::model_int8_bytes(id).map(Cow::Owned),
        })
    }

    /// A session of `model_bytes` on the first provider in `eps` that
    /// takes it, or on the CPU when `eps` is empty.
    fn build_session(model_bytes: &[u8], weights: Weights, intra_threads: usize, model: ModelKind, eps: &[EpKind], level: GraphOptimizationLevel) -> Result<Self, CoreError> {
        let model_id: prunr_models::ModelId = model.into();
        if eps.is_empty() {
            crate::cache::gc_stale_for_model(model_id, "CPU");
            let builder = Self::builder_with_base(intra_threads, level)?;
            let (builder, bytes) = apply_ort_graph_cache(builder, model_bytes, model_id, "CPU");
            let started = Instant::now();
            let session = builder
                .with_execution_providers([
                    CPU::default()
                        .with_arena_allocator(false) // lower memory baseline; subprocess handles OOM
                        .build(),
                ])
                .map_err(|e| CoreError::Inference(format!("ORT set CPU EP failed: {e}")))?
                .commit_from_memory(&bytes)
                .map_err(|e| CoreError::Inference(format!("ORT session creation failed (CPU): {e}")))?;
            tracing::info!(?model, ep = "CPU", elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
            return Ok(Self {
                session: Mutex::new(session),
                pick: SessionPick { weights, ..SessionPick::on("CPU") },
                model_kind: model,
            });
        }

        // GPU path: try platform GPU EPs one by one so we know which succeeded
        // (registering them as a list hides which EP was actually selected and
        // lets a crashing EP abort session creation before the CPU fallback is
        // reached — exactly the DirectML AbiCustomRegistry failure seen in the
        // wild). Fall through on error; the caller retries on the CPU if all
        // GPU EPs fail.
        let (session, ep, failed) = walk_ladder("", model_id, weights, eps, |ep| {
            crate::cache::gc_stale_for_model(model_id, ep.as_str());
            let builder = Self::builder_with_base(intra_threads, level).map_err(|e| Miss::Unusable(e.to_string()))?;
            let (builder, bytes_owner) = if ep.keeps_optimized_graph() {
                apply_ort_graph_cache(builder, model_bytes, model_id, ep.as_str())
            } else {
                (builder, Cow::Borrowed(model_bytes))
            };
            let (builder, coverage) = crate::ep_coverage::watch(ep, builder);
            let mut built = builder
                .with_execution_providers([ep.provider(model_id, EpSite::Engine { threads: intra_threads })])
                .map_err(|e| Miss::Unusable(format!("ORT register {ep} EP failed: {e}")))?;
            let started = Instant::now();
            let session = built.commit_from_memory(&bytes_owner).map_err(|e| Miss::Failed {
                error: format!("ORT session creation failed ({ep}): {e}"),
                suspect_cache: cached_graph(matches!(bytes_owner, Cow::Owned(_)), model_id, ep),
            })?;
            tracing::info!(?model, ?weights, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
            coverage.report(model_id, None);
            Ok(session)
        }).map_err(|walk| CoreError::Inference(walk.reason()))?;
        Ok(Self {
            session: Mutex::new(session),
            pick: SessionPick { weights, failed, ..SessionPick::on(ep.as_str()) },
            model_kind: model,
        })
    }

    fn builder_with_base(intra_threads: usize, level: GraphOptimizationLevel) -> Result<ort::session::builder::SessionBuilder, CoreError> {
        crate::ort_runtime::session_builder()?
            .with_optimization_level(level)
            .map_err(|e| CoreError::Inference(format!("ORT set optimization level failed: {e}")))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| CoreError::Inference(format!("ORT set intra threads failed: {e}")))
    }

    /// Detect the runtime provider (cached — runs once per process).
    ///
    /// This is a best-effort *startup guess* for the UI label before any
    /// subprocess is spawned. Windows/Linux probe `nvidia-smi` for CUDA;
    /// macOS always returns CoreML. The real EP is stamped onto each
    /// `OrtEngine` at session creation and propagated through the subprocess
    /// `Ready` event, so the UI corrects itself if this guess was wrong.
    pub fn detect_active_provider() -> String {
        static CACHED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        CACHED.get_or_init(|| {
            // First entry from `available_gpu_eps` is the most likely
            // winner — same probe drives the actual EP ladder. CPU is
            // the universal fallback when the loaded libonnxruntime
            // has no GPU EPs compiled in.
            available_gpu_eps()
                .first()
                .map(|ep| ep.as_str().to_string())
                .unwrap_or_else(|| "CPU".to_string())
        }).clone()
    }

}

/// Whether sessions on `provider` run on the host CPU: the CPU provider,
/// and OpenVINO, which this app configures for no other device.
pub fn runs_on_host_cpu(provider: &str) -> bool {
    provider.eq_ignore_ascii_case("CPU") || provider.eq_ignore_ascii_case(OPENVINO_NAME)
}

/// Whether `model` runs on the accelerator `provider`: not declared
/// incompatible in the registry, and some weights the ladder would load
/// there (FP16 first where a copy exists, then the model's own) have not
/// failed there on an earlier run. A model it does not run falls back to
/// the CPU by design.
pub fn model_runs_on(model: prunr_models::ModelId, provider: &str) -> bool {
    match available_gpu_eps().iter().find(|ep| ep.as_str().eq_ignore_ascii_case(provider)) {
        Some(&ep) => gpu_skip(model, ep).is_none(),
        None => prunr_models::is_ep_compatible(model, provider),
    }
}

/// Whether DirectML is the active GPU provider on this platform.
///
/// Used to gate parallel session-build paths: DirectML's
/// `commit_from_memory` has historic AbiCustomRegistry races on
/// concurrent calls, so callers fall back to sequential build when
/// DirectML is active. Always `false` outside Windows since the
/// `EpKind::DirectMl` variant is `cfg(windows)`-gated.
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
    /// Lock the underlying ORT Session for inference.
    /// Used by pipeline.rs — not part of the public trait API.
    /// ORT requires &mut Session for run(); Mutex provides interior mutability.
    pub(crate) fn with_session<T, F>(&self, f: F) -> Result<T, CoreError>
    where
        F: FnOnce(&mut Session) -> Result<T, CoreError>,
    {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut session)
    }

    /// Run inference via `IoBinding` and pass the output **as a borrow**
    /// to `f`. The borrow is alive only for the duration of `f`'s call.
    ///
    /// Compared to the `with_session` + `try_extract_array().to_owned()`
    /// path, this skips one full output-tensor clone (4 MB at BiRefNet
    /// 1024² → ~11.9% of `_memcpy_avx_unaligned_erms` in the perf trace).
    /// `bind_output_to_device(CPU)` lets ORT allocate the output in CPU
    /// memory directly; the post-run `try_extract_array` returns a view
    /// into that buffer without copying.
    ///
    /// Use this when the caller can consume the view in-place
    /// (e.g. postprocess running inside the closure). For paths that
    /// need an owned `Vec<f32>` regardless (Tier-2 cache write,
    /// IPC tensor packing), the `with_session` path is fine — the
    /// owning copy is unavoidable there.
    pub(crate) fn infer_into<R, F>(
        &self,
        input_array: ndarray::Array4<f32>,
        f: F,
    ) -> Result<R, CoreError>
    where
        F: FnOnce(ndarray::ArrayView4<'_, f32>) -> Result<R, CoreError>,
    {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        // copy before writing here. If a future GPU-resident postprocess
        // path lands, switch to the EP's native device.
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

impl InferenceEngine for OrtEngine {
    fn active_provider(&self) -> &str {
        &self.pick.provider
    }
}

/// Delete a cached optimized graph a session failed on (stale, or a
/// malformed artifact), so the next build rebuilds it from source.
/// `NotFound` is silent — concurrent processes legitimately race on the
/// same path.
fn clear_suspect_cache(path: &std::path::Path) {
    match std::fs::remove_file(path) {
        Ok(()) => tracing::warn!(cache = %path.display(),
            "deleted suspect cache file after commit failure"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(cache = %path.display(), %e,
            "failed to delete suspect cache file"),
    }
}

/// The cached optimized graph a session loaded, when `hit` (an
/// `apply_ort_graph_cache` hit comes back as owned bytes).
pub(crate) fn cached_graph(hit: bool, model: prunr_models::ModelId, ep: EpKind) -> Option<PathBuf> {
    hit.then(|| crate::cache::optimized_model_path(model, ep.as_str())).flatten()
}

/// Per-(model, EP) ORT-graph-optimization cache. On hit returns the
/// optimized bytes so commit skips optimization; on miss sets
/// `with_optimized_model_path` so ORT writes the optimized graph for
/// the next launch. Cache failures are best-effort — any I/O or ORT
/// error logs and falls back to the uncached path.
pub(crate) fn apply_ort_graph_cache<'a>(
    builder: ort::session::builder::SessionBuilder,
    model_bytes: &'a [u8],
    model_id: prunr_models::ModelId,
    ep_name: &str,
) -> (ort::session::builder::SessionBuilder, Cow<'a, [u8]>) {
    let Some(path) = crate::cache::optimized_model_path(model_id, ep_name) else {
        return (builder, Cow::Borrowed(model_bytes));
    };
    match std::fs::read(&path) {
        Ok(cached) => {
            tracing::debug!(?model_id, ep = %ep_name, bytes = cached.len(), "cache hit (optimized graph)");
            return (builder, Cow::Owned(cached));
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(?model_id, ep = %ep_name, %e, path = %path.display(),
                "cached optimized graph unreadable; rebuilding");
        }
    }
    if crate::cache::cache_dir_for(model_id, ep_name).is_none() {
        return (builder, Cow::Borrowed(model_bytes));
    }
    match builder.with_optimized_model_path(&path) {
        Ok(builder) => {
            tracing::debug!(?model_id, ep = %ep_name, path = %path.display(), "cache miss (writing optimized graph)");
            (builder, Cow::Borrowed(model_bytes))
        }
        Err(e) => {
            tracing::warn!(?model_id, ep = %ep_name, %e, "with_optimized_model_path failed; cache disabled for this build");
            (e.recover(), Cow::Borrowed(model_bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEngine;
    impl InferenceEngine for MockEngine {
        fn active_provider(&self) -> &str { "CPU" }
    }

    #[test]
    fn test_inference_engine_trait_is_object_safe() {
        let engine: Box<dyn InferenceEngine> = Box::new(MockEngine);
        assert_eq!(engine.active_provider(), "CPU");
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
        assert!(options.contains(&format!(r#""ModelCacheDirectory": {:?}"#, dir.to_string_lossy())), "{options}");
        assert!(!format!("{:?}", coreml_provider(None)).contains("ModelCacheDirectory"));
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
            || ort::ep::CUDA::default().is_available().unwrap_or(true)
        {
            return;
        }
        let sites = [
            EpSite::Engine { threads: 1 },
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

    /// A provider is passed over when declared incompatible or failed
    /// before, without a try; the walk says so when it tried none.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_walk_that_tries_no_provider_says_why() {
        use prunr_models::ModelId;
        let model = ModelId::Silueta;
        let declared = walk_ladder("", model, Weights::Fp32, &[EpKind::OpenVino], |_| -> Result<(), Miss> { panic!("tried a declared pair") });
        assert_eq!(declared.map_err(|w| w.reason()), Err("no GPU provider to try: OpenVINO is declared incompatible".to_string()));
        let none = walk_ladder("", model, Weights::Fp32, &[], |_| -> Result<(), Miss> { unreachable!() });
        assert_eq!(none.map_err(|w| w.reason()), Err("No GPU EPs available on this platform".to_string()));
        assert_eq!(gpu_skip(model, EpKind::OpenVino), Some(Skip::Declared));
        assert_eq!(
            Walk::without_a_try(model, Weights::Fp32, &[EpKind::OpenVino]).map(|w| w.reason()),
            Some("no GPU provider to try: OpenVINO is declared incompatible".to_string()),
            "the same line, known before any weights are read",
        );
        assert_eq!(Walk::without_a_try(ModelId::U2net, Weights::Fp32, &[EpKind::OpenVino]), None, "OpenVINO would be tried");
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

    // Integration tests require dev-models feature and downloaded models.
    // Run with: cargo test -p prunr-core --features dev-models
    #[cfg(feature = "dev-models")]
    #[test]
    fn test_ort_engine_silueta_active_provider_non_empty() {
        let engine = OrtEngine::new(ModelKind::Silueta, 1)
            .expect("OrtEngine::new should succeed with dev-models and downloaded models");
        let provider = engine.active_provider();
        assert!(!provider.is_empty(), "active_provider() returned empty string");
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_ort_engine_u2net_creates_session() {
        // U2Net is OnDemand — only run when the user has downloaded it.
        // Mirrors the skip-if-missing pattern in tests/reference_test.rs.
        if !prunr_models::is_available(prunr_models::ModelId::U2net) {
            eprintln!("Skipping: U2Net not in user data dir (download via Model Store)");
            return;
        }
        let engine = OrtEngine::new(ModelKind::U2net, 1)
            .expect("OrtEngine::new(U2net) should succeed when U2Net is installed");
        let provider = engine.active_provider();
        assert!(!provider.is_empty());
    }

    /// Level2 is a valid value for all bundled models — the constructor must not
    /// reject it for non-HAT models. Exercises the no-regression case: explicitly
    /// passing Level2 to a segmentation model succeeds just as Level3 does.
    ///
    /// Gated on `dev-models` because Silueta is bundled only in that feature.
    #[cfg(feature = "dev-models")]
    #[test]
    fn level2_constructs_for_existing_model() {
        let engine = OrtEngine::new_with_fallback_and_level(
            ModelKind::Silueta,
            1,
            false,
            GraphOptimizationLevel::Level2,
        )
        .expect("Level2 must succeed for Silueta");
        assert!(!engine.active_provider().is_empty());
    }
}
