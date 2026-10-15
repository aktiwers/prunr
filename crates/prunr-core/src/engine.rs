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
                let mut p = ort::ep::CoreML::default();
                if let Some(dir) = dir {
                    p = p.with_model_cache_dir(dir.to_string_lossy().into_owned());
                }
                p.build()
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

/// Why a provider gave no session.
#[derive(Debug)]
pub(crate) enum Miss {
    /// The provider could not be set up: says nothing about the model.
    Unusable(String),
    /// The session failed to build. `from_cache`: it loaded a cached
    /// optimized graph, which is then the suspect.
    Failed { error: String, from_cache: bool },
}

/// Each provider in `eps` with `weights` until `commit` builds a session,
/// returned with the failures on the way (`SessionPick::failed`). Pairs
/// declared or known to fail are skipped; a new failure is recorded
/// against these weights only, so FP16 failing never bars FP32.
pub(crate) fn walk_ladder<S>(
    model: prunr_models::ModelId,
    weights: Weights,
    eps: &[EpKind],
    mut commit: impl FnMut(EpKind) -> Result<S, Miss>,
) -> Result<(S, EpKind, Vec<String>), String> {
    let mut failed = Vec::new();
    let mut skipped = Vec::new();
    for &ep in eps {
        if !prunr_models::is_ep_compatible(model, ep.as_str()) {
            tracing::debug!(?model, ep = %ep, "EP statically incompatible; skipping");
            skipped.push(format!("{ep} is declared incompatible"));
            continue;
        }
        if crate::ep_compat::is_known_failure(ep, model, weights) {
            tracing::debug!(?model, ?weights, ep = %ep, "EP cached as incompatible; skipping");
            skipped.push(format!("{ep} failed this model before"));
            continue;
        }
        match commit(ep) {
            Ok(session) => return Ok((session, ep, failed)),
            Err(Miss::Unusable(error)) => {
                tracing::warn!(?model, ep = %ep, %error, "GPU EP unusable — trying next EP");
                failed.push(error);
            }
            Err(Miss::Failed { error, from_cache }) => {
                tracing::warn!(?model, ?weights, ep = %ep, %error, "GPU session creation failed — trying next EP");
                handle_commit_failure(
                    from_cache, ep, model, weights,
                    || crate::cache::optimized_model_path(model, ep.as_str()),
                    &error,
                );
                failed.push(error);
            }
        }
    }
    Err(if !failed.is_empty() {
        failed.join("; ")
    } else if !skipped.is_empty() {
        format!("no GPU provider to try: {}", skipped.join(", "))
    } else {
        "No GPU EPs available on this platform".to_string()
    })
}

/// The GPU providers the loaded runtime offers, in the order a session
/// tries them.
pub fn gpu_provider_names() -> Vec<&'static str> {
    available_gpu_eps().iter().map(EpKind::as_str).collect()
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

    /// Same as [`Self::new`] but with an explicit optimization-level override.
    ///
    /// Use `Level2` for window-attention transformers (HAT / Swin) where
    /// `Level3` bakes the first tile's input shape into the graph during
    /// session init, causing an irrecoverable shape-mismatch on the second
    /// tile in the same session (observed against 4xNomos8kSCHAT-L: Level3
    /// panics with `Attempting to get index by a name which does not exist:
    /// InsertedPrecisionFreeCast_…`; Level2 succeeds across all tile sizes).
    pub fn new_with_optimization_level(
        model: ModelKind,
        intra_threads: usize,
        level: GraphOptimizationLevel,
    ) -> Result<Self, CoreError> {
        Self::new_with_fallback_and_level(model, intra_threads, false, level)
    }

    /// CPU-only engine with an explicit optimization-level override.
    /// Used by the upscale dispatcher to avoid OpenVINO's per-input-shape
    /// lazy graph compilation, which on RRDB-class models (Real-ESRGAN)
    /// can stall a single dispatch for tens of minutes the first time a
    /// new tile dimension is seen — the EP cache for upscale wasn't
    /// hitting in practice.
    pub fn new_cpu_only_with_optimization_level(
        model: ModelKind,
        intra_threads: usize,
        level: GraphOptimizationLevel,
    ) -> Result<Self, CoreError> {
        Self::new_with_fallback_and_level(model, intra_threads, true, level)
    }

    fn new_with_fallback(model: ModelKind, intra_threads: usize, cpu_only: bool) -> Result<Self, CoreError> {
        Self::new_with_fallback_and_level(model, intra_threads, cpu_only, GraphOptimizationLevel::Level3)
    }

    /// Core session-creation path shared by `new`, `new_cpu_only`, and
    /// `new_with_optimization_level`. Tries optimized variant (FP16/INT8)
    /// first; falls back to FP32 on failure. When `cpu_only=false`, also
    /// falls back to CPU-only if the GPU EP crashes at init.
    fn new_with_fallback_and_level(
        model: ModelKind,
        intra_threads: usize,
        cpu_only: bool,
        level: GraphOptimizationLevel,
    ) -> Result<Self, CoreError> {
        tracing::debug!(?model, intra_threads, cpu_only, "OrtEngine init");
        let mut failed = Vec::new();
        // Check if an optimized variant exists (loaded from filesystem, so Vec<u8>).
        if let Some((optimized, weights)) = Self::optimized_variant_bytes(model, cpu_only) {
            tracing::debug!(?model, variant_bytes = optimized.len(), "OrtEngine trying optimized variant");
            match Self::build_session(&optimized, weights, intra_threads, model, cpu_only, level) {
                Ok(engine) => {
                    tracing::info!(?model, provider = %engine.pick.provider, "OrtEngine ready (optimized variant)");
                    return Ok(engine);
                }
                Err(e) => {
                    tracing::warn!(?model, error = %e, "optimized variant failed — falling back to FP32");
                    failed.push(format!("{} weights: {e}", weights.label()));
                }
            }
        } else {
            tracing::debug!(?model, "no optimized variant on disk — using embedded FP32");
        }
        // Fall back to the FP32 model bytes. Bundled models are zero-copy
        // (`Cow::Borrowed`); on-demand models load from the user data dir
        // (`Cow::Owned`) and surface a clear "not installed" error here
        // when the file is missing.
        let id: prunr_models::ModelId = model.into();
        let fp32 = prunr_models::resolve_bytes(id)
            .ok_or_else(|| CoreError::Inference(prunr_models::not_installed_error(id)))?;
        match Self::build_session(&fp32, Weights::Fp32, intra_threads, model, cpu_only, level) {
            Ok(mut engine) => {
                tracing::info!(?model, provider = %engine.pick.provider, "OrtEngine ready (FP32)");
                failed.append(&mut engine.pick.failed);
                engine.pick.failed = failed;
                Ok(engine)
            }
            Err(e) if !cpu_only => {
                tracing::warn!(?model, error = %e, "GPU session creation failed — retrying CPU-only");
                failed.push(format!("FP32 weights: {e}"));
                // Recurse so the cpu_only=true path also tries its
                // CPU-targeted optimized variant (INT8) before falling
                // back to FP32. Otherwise the GPU-fail-then-CPU path
                // ends up on FP32 even when an INT8 variant is on disk.
                let mut engine = Self::new_with_fallback_and_level(model, intra_threads, true, level)?;
                tracing::info!(
                    ?model, provider = %engine.pick.provider,
                    "OrtEngine ready (CPU fallback after GPU failure)",
                );
                failed.append(&mut engine.pick.failed);
                engine.pick.failed = failed;
                Ok(engine)
            }
            Err(e) => Err(e),
        }
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

    /// Try to load an optimized model variant from the filesystem.
    /// Returns None if no variant is available (falls back to embedded FP32).
    fn optimized_variant_bytes(model: ModelKind, cpu_only: bool) -> Option<(Vec<u8>, Weights)> {
        #[cfg(target_os = "macos")]
        { return None; } // macOS uses FP32 — CoreML does its own FP16 conversion

        #[cfg(not(target_os = "macos"))]
        {
            let id: prunr_models::ModelId = model.into();
            if cpu_only {
                prunr_models::model_int8_bytes(id).map(|b| (b, Weights::Int8))
            } else {
                prunr_models::model_fp16_bytes(id).map(|b| (b, Weights::Fp16))
            }
        }
    }

    fn build_session(model_bytes: &[u8], weights: Weights, intra_threads: usize, model: ModelKind, cpu_only: bool, level: GraphOptimizationLevel) -> Result<Self, CoreError> {
        let model_id: prunr_models::ModelId = model.into();
        // CPU-only path: straight shot.
        if cpu_only {
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
        // wild). Fall through on error; the caller retries with cpu_only=true
        // if all GPU EPs fail.
        let (session, ep, failed) = walk_ladder(model_id, weights, available_gpu_eps(), |ep| {
            crate::cache::gc_stale_for_model(model_id, ep.as_str());
            let builder = Self::builder_with_base(intra_threads, level).map_err(|e| Miss::Unusable(e.to_string()))?;
            let (builder, bytes_owner) = if ep.keeps_optimized_graph() {
                apply_ort_graph_cache(builder, model_bytes, model_id, ep.as_str())
            } else {
                (builder, Cow::Borrowed(model_bytes))
            };
            let mut built = builder
                .with_execution_providers([ep.provider(model_id, EpSite::Engine { threads: intra_threads })])
                .map_err(|e| Miss::Unusable(format!("ORT register {ep} EP failed: {e}")))?;
            let started = Instant::now();
            let session = built.commit_from_memory(&bytes_owner).map_err(|e| Miss::Failed {
                error: format!("ORT session creation failed ({ep}): {e}"),
                from_cache: matches!(bytes_owner, Cow::Owned(_)),
            })?;
            tracing::info!(?model, ?weights, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
            Ok(session)
        }).map_err(CoreError::Inference)?;
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
    if !prunr_models::is_ep_compatible(model, provider) {
        return false;
    }
    let Some(&ep) = available_gpu_eps().iter().find(|ep| ep.as_str().eq_ignore_ascii_case(provider)) else {
        return true;
    };
    let fp16_tried = cfg!(not(target_os = "macos")) && prunr_models::has_fp16_variant(model);
    !crate::ep_compat::is_known_failure(ep, model, Weights::Fp32)
        || (fp16_tried && !crate::ep_compat::is_known_failure(ep, model, Weights::Fp16))
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

/// Delete a suspect cache file. `NotFound` is silent — concurrent
/// processes legitimately race on the same path.
pub(crate) fn clear_suspect_cache(cache_path: Option<&std::path::Path>) {
    let Some(p) = cache_path else { return };
    match std::fs::remove_file(p) {
        Ok(()) => tracing::warn!(cache = %p.display(),
            "deleted suspect cache file after commit failure"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(cache = %p.display(), %e,
            "failed to delete suspect cache file"),
    }
}

/// Route a session-commit failure based on whether the load came from
/// cache. Cached load → cache file is the suspect (stale optimized
/// graph, malformed spike artifact); delete the file so the next
/// session build hits a cache miss and rebuilds from source.
/// Non-cached load → the EP itself failed for this model; record it
/// in `ep_compat` so future runs skip the doomed attempt.
///
/// `cache_path_fn` is a closure so the path-resolution allocation
/// only fires on the cached branch.
pub(crate) fn handle_commit_failure(
    was_cached: bool,
    ep: EpKind,
    id: prunr_models::ModelId,
    weights: Weights,
    cache_path_fn: impl FnOnce() -> Option<PathBuf>,
    error: &str,
) {
    if was_cached {
        clear_suspect_cache(cache_path_fn().as_deref());
    } else {
        crate::ep_compat::record_failure(ep, id, weights, error);
    }
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
        clear_suspect_cache(Some(&path));
        assert!(!path.exists(), "existing file must be removed");
        // NotFound + None must be silent no-ops — concurrent deletions
        // are expected, and a panic here would abort the whole session
        // build instead of letting it rebuild from source.
        clear_suspect_cache(Some(&path));
        clear_suspect_cache(None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Boundary contract: when `was_cached=true`, the cache file is
    /// deleted AND `ep_compat::record_failure` is NOT called. This is
    /// A malformed compiled.onnx from the EPContext path would otherwise
    /// trigger `record_failure` and persistently disable OpenVINO for SD,
    /// until `ep_compat.json` was removed by hand.
    ///
    /// Uses `(EpKind::Cuda, ModelId::TaesdFp16)` to dodge other tests
    /// that touch `ep_compat` under the parallel runner.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn handle_commit_failure_skips_record_failure_when_cached() {
        use prunr_models::ModelId;
        let id = ModelId::TaesdFp16;
        let ep = EpKind::Cuda;
        let Some(dir) = crate::cache::cache_dir_for(id, ep.as_str()) else { return };
        let cache_file = dir.join("optimized.onnx");
        std::fs::write(&cache_file, b"stale").unwrap();
        let initial = crate::ep_compat::is_known_failure(ep, id, Weights::Fp32);

        handle_commit_failure(
            true, ep, id, Weights::Fp32,
            || crate::cache::optimized_model_path(id, ep.as_str()),
            "test cache failure",
        );

        assert!(!cache_file.exists(), "cache file must be deleted");
        assert_eq!(
            crate::ep_compat::is_known_failure(ep, id, Weights::Fp32), initial,
            "record_failure must NOT fire on was_cached=true",
        );
        let _ = std::fs::remove_dir_all(&dir);
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
                    Err(Miss::Failed { error: "tensor(float16) is invalid".into(), from_cache: false })
                } else {
                    Ok("session")
                }
            }
        };
        assert!(walk_ladder(model, Weights::Fp16, &[ep], commit_fp16_fails(Weights::Fp16)).is_err());
        let picked = walk_ladder(model, Weights::Fp32, &[ep], commit_fp16_fails(Weights::Fp32));
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
        let walk = walk_ladder(model, Weights::Fp32, &[ep], |_| Err::<(), _>(Miss::Unusable("no DirectX 12 device".into())));
        assert_eq!(walk, Err("no DirectX 12 device".to_string()));
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32), "the machine's lack, not the model's");
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
    fn new_with_optimization_level_constructs_for_existing_model() {
        let engine = OrtEngine::new_with_optimization_level(
            ModelKind::Silueta,
            1,
            GraphOptimizationLevel::Level2,
        )
        .expect("Level2 must succeed for Silueta");
        assert!(!engine.active_provider().is_empty());
    }
}
