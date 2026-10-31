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
    value::{DynValue, Outlet, Tensor, TensorElementType},
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

/// DirectML's provider name, defined on every platform like CUDA's.
pub const DIRECTML_NAME: &str = "DirectML";

/// MIGraphX's provider name, defined on every platform like CUDA's.
pub(crate) const MIGRAPHX_NAME: &str = "MIGraphX";

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
            EpKind::MiGraphX => MIGRAPHX_NAME,
            #[cfg(target_os = "macos")]
            EpKind::CoreMl => COREML_NAME,
            #[cfg(windows)]
            EpKind::DirectMl => DIRECTML_NAME,
        }
    }
}

impl std::fmt::Display for EpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The GPU providers this process offers, once probed; an add-on joins
/// them when it installs (`start_addon`).
static AVAILABLE: std::sync::RwLock<Option<std::sync::Arc<[EpKind]>>> = std::sync::RwLock::new(None);

/// The GPU EP ladder filtered by `is_available()`: a `load-dynamic`
/// runtime may lack EPs (the OpenVINO wheel has CPU + OpenVINO only).
/// Probed once per process; an add-on installed later joins it.
pub(crate) fn available_gpu_eps() -> std::sync::Arc<[EpKind]> {
    use std::sync::{Arc, PoisonError};
    if let Some(eps) = AVAILABLE.read().unwrap_or_else(PoisonError::into_inner).as_ref() {
        return Arc::clone(eps);
    }
    let mut slot = AVAILABLE.write().unwrap_or_else(PoisonError::into_inner);
    Arc::clone(slot.get_or_insert_with(|| probe_gpu_eps().into()))
}

/// Start `addon`, installed after this process probed its providers:
/// sessions built from now on may take it. Off the UI thread: the CUDA
/// add-on loads NVIDIA's libraries, seconds the first time.
#[cfg(not(target_os = "macos"))]
pub fn start_addon(addon: crate::ort_runtime::Addon) -> Result<(), String> {
    use std::sync::PoisonError;
    let ep = addon.ep();
    if available_gpu_eps().contains(&ep) {
        return Ok(());
    }
    match addon_starts(addon) {
        Some(Ok(())) => {}
        Some(Err(failure)) => {
            let text = failure.text();
            START_FAILURES.record(failure);
            return Err(text);
        }
        None => return Err(format!("the {ep} add-on did not register; the log says why")),
    }
    let mut slot = AVAILABLE.write().unwrap_or_else(PoisonError::into_inner);
    let mut eps: Vec<EpKind> = slot.as_deref().unwrap_or_default().to_vec();
    eps.insert(addon_position(&eps, ep), ep);
    *slot = Some(eps.into());
    tracing::info!(%ep, "add-on started; sessions may take it from now on");
    Ok(())
}

/// Where the add-on `ep` goes among `eps`, in the probe's order: CUDA
/// first, OpenVINO last.
#[cfg(not(target_os = "macos"))]
fn addon_position(eps: &[EpKind], ep: EpKind) -> usize {
    if ep == EpKind::Cuda { 0 } else { eps.len() }
}

fn probe_gpu_eps() -> Vec<EpKind> {
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
        use crate::ort_runtime::Addon;
        let cuda = if ort::ep::CUDA::is_available().unwrap_or(false) {
            crate::ort_runtime::preload_nvidia_libraries();
            Some(provider_starts(ort::ep::CUDA::default().with_device_id(0).build().error_on_failure()).map_err(|e| GpuStartFailure::of(GpuStack::Cuda, &e)))
        } else {
            addon_starts(Addon::Cuda)
        };
        if cuda.is_some_and(|started| START_FAILURES.started(started)) {
            eps.push(EpKind::Cuda);
        }
        #[cfg(target_os = "linux")]
        if ort::ep::MIGraphX::is_available().unwrap_or(false) {
            let migraphx = provider_starts(ort::ep::MIGraphX::default().with_device_id(0).build().error_on_failure()).map_err(|e| GpuStartFailure::of(GpuStack::Rocm, &e));
            if START_FAILURES.started(migraphx) {
                eps.push(EpKind::MiGraphX);
            }
        }
        #[cfg(windows)]
        if ort::ep::DirectML::is_available().unwrap_or(false) && directml_starts() { eps.push(EpKind::DirectMl); }
        // Last: a model no test measured tries a graphics card first, and
        // OpenVINO runs on the processor unless the user lets it use Intel
        // graphics, an integrated chip.
        let openvino = if ort::ep::OpenVINO::is_available().unwrap_or(false) { Some(Ok(())) } else { addon_starts(Addon::OpenVino) };
        if openvino.is_some_and(|started| START_FAILURES.started(started)) {
            eps.push(EpKind::OpenVino);
        }
    }
    let eps_str: Vec<&'static str> = eps.iter().map(EpKind::as_str).collect();
    tracing::info!(eps = ?eps_str, "Available GPU execution providers");
    eps
}

/// Why each provider that could not start here did not, once the
/// providers have been probed or an add-on started: CUDA and OpenVINO can
/// be installed side by side, and each fails on its own.
struct StartFailures(PerStack<std::sync::OnceLock<GpuStartFailure>>);

impl StartFailures {
    const fn new() -> Self {
        Self([const { std::sync::OnceLock::new() }; GpuStack::ALL.len()])
    }

    /// Keep `failure`, the first of its stack.
    #[cfg(not(target_os = "macos"))]
    fn record(&self, failure: GpuStartFailure) {
        tracing::info!(?failure, "a provider cannot start; sessions run on the CPU");
        let _ = self.0[failure.stack.index()].set(failure);
    }

    /// Whether a start succeeded; the failure is kept when it did not.
    #[cfg(not(target_os = "macos"))]
    fn started(&self, outcome: Result<(), GpuStartFailure>) -> bool {
        outcome.map_err(|failure| self.record(failure)).is_ok()
    }

    fn each(&self) -> PerStack<Option<&GpuStartFailure>> {
        std::array::from_fn(|i| self.0[i].get())
    }
}

static START_FAILURES: StartFailures = StartFailures::new();

/// Why each provider does not run here, by its stack; `None` where it
/// runs or there is none.
pub fn gpu_start_failures() -> PerStack<Option<&'static GpuStartFailure>> {
    START_FAILURES.each()
}

/// One value per vendor stack, indexed by `GpuStack::index`.
pub type PerStack<T> = [T; GpuStack::ALL.len()];

/// The vendor stack a provider needs installed beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuStack {
    /// NVIDIA's driver with CUDA 12 and cuDNN 9.
    Cuda,
    /// AMD's ROCm 7 with MIGraphX.
    Rocm,
    /// Intel's OpenVINO, which brings its own libraries.
    OpenVino,
}

impl GpuStack {
    pub const ALL: [Self; 3] = [Self::Cuda, Self::Rocm, Self::OpenVino];

    pub fn index(self) -> usize {
        self as usize
    }

    fn provider(self) -> &'static str {
        match self {
            Self::Cuda => CUDA_NAME,
            Self::Rocm => MIGRAPHX_NAME,
            Self::OpenVino => OPENVINO_NAME,
        }
    }

    fn needs(self) -> &'static str {
        match self {
            Self::Cuda => "NVIDIA's CUDA 12 and cuDNN 9 libraries",
            Self::Rocm => "AMD's ROCm 7 with MIGraphX installed",
            Self::OpenVino => "its own files, which installing it again brings back",
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

    /// `failures` in their stacks' slots, as `gpu_start_failures` gives
    /// them, for tests.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn by_stack<'a>(failures: &[&'a Self]) -> PerStack<Option<&'a Self>> {
        std::array::from_fn(|i| failures.iter().copied().find(|f| f.stack.index() == i))
    }

    /// The driver runs but serves no device the provider can use.
    pub const fn no_device(stack: GpuStack) -> Self {
        Self { stack, why: StartFailure::NoDevice }
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
            StartFailure::NoDriver => format!("{provider} could not start: {}", match self.stack {
                GpuStack::Cuda => "the NVIDIA driver is missing or older than CUDA 12 needs (525 or newer).",
                GpuStack::Rocm => "the AMD graphics driver is missing or too old for ROCm 7.",
                GpuStack::OpenVino => "Intel's graphics driver is missing.",
            }),
            StartFailure::NoDevice => format!("{provider} could not start: {}", match self.stack {
                GpuStack::Cuda => "no NVIDIA graphics card was found.",
                GpuStack::Rocm => "no AMD graphics card that ROCm supports was found.",
                GpuStack::OpenVino => "it found no Intel processor or Intel graphics it can use.",
            }),
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

/// Whether the CUDA add-on can start here: it lists a device only for an
/// NVIDIA card its CUDA runs on (none without one, or with a driver too
/// old for CUDA 12), and a session takes that device.
#[cfg(not(target_os = "macos"))]
fn cuda_addon_starts() -> Result<(), GpuStartFailure> {
    let devices = crate::ort_runtime::addon_devices(crate::ort_runtime::Addon::Cuda);
    if devices.is_empty() {
        return Err(GpuStartFailure::no_device(GpuStack::Cuda));
    }
    devices_start(devices, None).map_err(|e| GpuStartFailure::of(GpuStack::Cuda, &e))
}

/// Whether a session takes an add-on's `devices`, with `options`: what
/// it takes for the add-on to start.
#[cfg(not(target_os = "macos"))]
fn devices_start(devices: Vec<ort::device::Device<'static>>, options: Option<&[(String, String)]>) -> Result<(), String> {
    crate::ort_runtime::session_builder()
        .map_err(|e| e.to_string())
        .and_then(|b| b.with_devices(devices, options).map(drop).map_err(|e| e.to_string()))
}

/// Whether the OpenVINO add-on can start here: it lists a device only for
/// an Intel processor, or Intel graphics with Intel's driver once the user
/// lets it use them, and a session takes the one engines use first.
#[cfg(not(target_os = "macos"))]
fn openvino_addon_starts() -> Result<(), GpuStartFailure> {
    let session = OpenVinoTuning::default().addon_session().map_err(|_| GpuStartFailure::no_device(GpuStack::OpenVino))?;
    devices_start(session.devices, Some(&session.options)).map_err(|e| GpuStartFailure::of(GpuStack::OpenVino, &e))
}

/// Whether `addon` starts here: registered, it lists a device and a
/// session takes it. `None` when it is not installed, or waits for a newer
/// runtime, which Settings says by itself.
#[cfg(not(target_os = "macos"))]
fn addon_starts(addon: crate::ort_runtime::Addon) -> Option<Result<(), GpuStartFailure>> {
    use crate::ort_runtime::Addon;
    if !crate::ort_runtime::register_addon(addon) {
        return crate::ort_runtime::registration_error(addon).map(|e| Err(GpuStartFailure::of(addon.stack(), e)));
    }
    Some(match addon {
        Addon::Cuda => cuda_addon_starts(),
        Addon::OpenVino => openvino_addon_starts(),
    })
}

impl crate::ort_runtime::Addon {
    /// The vendor stack its provider runs on, which a start failure names.
    pub fn stack(self) -> GpuStack {
        match self {
            Self::Cuda => GpuStack::Cuda,
            Self::OpenVino => GpuStack::OpenVino,
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn ep(self) -> EpKind {
        match self {
            Self::Cuda => EpKind::Cuda,
            Self::OpenVino => EpKind::OpenVino,
        }
    }
}

/// Where a session goes on the OpenVINO add-on: the plugin's devices,
/// with the settings it takes.
#[cfg(not(target_os = "macos"))]
struct OpenVinoSession {
    devices: Vec<ort::device::Device<'static>>,
    options: Vec<(String, String)>,
}

/// How an OpenVINO session is set up for its site, whether the loaded
/// runtime carries OpenVINO or its add-on brings it.
#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct OpenVinoTuning<'a> {
    /// As OpenVINO names it (`AUTO:GPU,CPU`, `GPU`); `None` is the
    /// processor.
    device: Option<&'a str>,
    /// OpenVINO's thread cap on the processor. Uncapped, its TBB pool
    /// takes every logical core and oversubscribes against rayon
    /// (`sched_yield` ~2% of CLI batch wall time).
    threads: Option<usize>,
    /// One stream and dynamic shapes: a Stable Diffusion part. One stream
    /// keeps no buffers per stream (the bundle sits at ~16 GB RSS on an
    /// i7-6700 either way; measured 2026-10), and dynamic shapes cost
    /// nothing per step, so one bundle serves 512² tiles and the 512×768
    /// crop.
    sd_part: bool,
}

#[cfg(not(target_os = "macos"))]
impl<'a> OpenVinoTuning<'a> {
    fn for_site(site: EpSite<'a>, model: prunr_models::ModelId) -> Self {
        let device = site.ov_device(on_intel_gpu(model));
        match site {
            EpSite::Engine { threads, .. } | EpSite::Fill { threads } => Self { device, threads: Some(threads), sd_part: false },
            EpSite::Selection { .. } => Self { device, threads: None, sd_part: false },
            EpSite::SdPart { .. } => Self { device, threads: None, sd_part: true },
        }
    }

    /// The loaded runtime's own OpenVINO provider, so tuned. Not given a
    /// cache dir: on ort 2.0.0-rc.12 + OpenVINO 1.24.1 cold and warm SD
    /// UNet commits match (23.7 / 23.2 s) because OpenVINO keys the cache
    /// differently each launch and rewrites ~3.3 GB every build.
    fn provider(self) -> ExecutionProviderDispatch {
        let mut provider = ort::ep::OpenVINO::default();
        if let Some(threads) = self.threads {
            provider = provider.with_num_threads(threads.max(1));
        }
        if self.sd_part {
            provider = provider.with_num_streams(1).with_dynamic_shapes(true);
        }
        if let Some(device) = self.device {
            provider = provider.with_device_type(device);
        }
        provider.build()
    }

    /// The add-on's devices for the device, with the settings it takes:
    /// AUTO's factory for a device AUTO picks among. Its shapes are
    /// dynamic without asking.
    fn addon_session(self) -> Result<OpenVinoSession, String> {
        let device = self.device.unwrap_or("CPU");
        let (meta, wanted) = match device.split_once(':') {
            Some((meta, devices)) => (Some(meta), devices.split(',').collect::<Vec<_>>()),
            None => (None, vec![device]),
        };
        let addon = crate::ort_runtime::Addon::OpenVino.ep_name();
        let ep = meta.map_or_else(|| addon.to_string(), |meta| format!("{addon}.{meta}"));
        let mut listed = crate::ort_runtime::devices_named(&ep);
        let devices: Vec<_> = wanted
            .iter()
            .filter_map(|&name| {
                let at = listed.iter().position(|d| format!("{:?}", d.hardware_device().ty()) == name)?;
                Some(listed.remove(at))
            })
            .collect();
        if devices.len() != wanted.len() {
            return Err(format!("OpenVINO lists no {device} device here"));
        }
        let options = openvino_load_config(meta, &wanted, self.threads, self.sd_part)
            .map(|config| vec![(format!("{ep}.load_config"), config)])
            .unwrap_or_default();
        Ok(OpenVinoSession { devices, options })
    }
}

/// OpenVINO's settings for a session on `devices`, as the plugin's
/// `load_config` takes them: per device, and under `meta`'s name for
/// devices a meta device (AUTO) picks among, which is all the plugin
/// reads for it. The legacy option names are ignored by the plugin.
/// `None` when nothing is set.
#[cfg(not(target_os = "macos"))]
fn openvino_load_config(meta: Option<&str>, devices: &[&str], threads: Option<usize>, one_stream: bool) -> Option<String> {
    let mut per_device = serde_json::Map::new();
    for &device in devices {
        let mut settings = serde_json::Map::new();
        if let Some(threads) = threads.filter(|_| device == "CPU") {
            settings.insert("INFERENCE_NUM_THREADS".into(), threads.max(1).to_string().into());
        }
        if one_stream {
            settings.insert("NUM_STREAMS".into(), "1".into());
        }
        if !settings.is_empty() {
            per_device.insert(device.into(), settings.into());
        }
    }
    if per_device.is_empty() {
        return None;
    }
    let config = match meta {
        Some(meta) => serde_json::Map::from_iter([(meta.to_string(), per_device.into())]),
        None => per_device,
    };
    Some(serde_json::Value::Object(config).to_string())
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

/// The GPU providers a session of `model` may try, fastest first: none
/// when the user chose "CPU only" (Settings › Processor); the one
/// `try_only` names while a test measures it.
pub(crate) fn gpu_eps_for(model: prunr_models::ModelId, cpu_only: bool) -> Vec<EpKind> {
    if cpu_only {
        return Vec::new();
    }
    let mut eps = available_gpu_eps().to_vec();
    keep_only(&mut eps, *ONLY.read().unwrap_or_else(std::sync::PoisonError::into_inner), |ep: &EpKind| ep.as_str());
    let places = places_here(model);
    let (intel_gpu, remeasuring) = (openvino_on_gpu(), remeasuring());
    fastest_first(&eps, |ep| prunr_models::place_in(&places, model, ep.as_str(), intel_gpu, remeasuring))
}

/// The one GPU provider sessions may try while a test measures it alone.
/// Process-wide: only the test process (`prunr bench --here`) sets it.
static ONLY: std::sync::RwLock<Option<&'static str>> = std::sync::RwLock::new(None);

/// Sessions try only `provider` (ORT's name) of the GPU providers until
/// the guard drops; one this process does not offer leaves none.
#[must_use = "the pin lasts while the guard lives"]
pub fn try_only(provider: &'static str) -> TryingOnly {
    *ONLY.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(provider);
    TryingOnly(())
}

pub struct TryingOnly(());

fn keep_only<T>(eps: &mut Vec<T>, only: Option<&str>, name: impl Fn(&T) -> &str) {
    if let Some(only) = only {
        eps.retain(|ep| name(ep).eq_ignore_ascii_case(only));
    }
}

impl Drop for TryingOnly {
    fn drop(&mut self) {
        *ONLY.write().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

/// `eps` with those measured faster first, quickest first, then the
/// unmeasured, then those that keep the model on the CPU; each group in
/// `eps`' own order. Two providers that both run a model can differ
/// 19-fold: Silueta takes 30 ms on DirectML and 566 ms on CUDA on one
/// RTX 3060.
fn fastest_first<T: Copy>(eps: &[T], place_of: impl Fn(T) -> prunr_models::Place) -> Vec<T> {
    use prunr_models::Place;
    let mut ranked: Vec<(T, u8, f32)> = eps
        .iter()
        .map(|&ep| match place_of(ep) {
            Place::There(speed) => (ep, 0, speed.ms),
            Place::Unmeasured => (ep, 1, 0.0),
            Place::Cpu(_) => (ep, 2, 0.0),
        })
        .collect();
    ranked.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
    ranked.into_iter().map(|(ep, ..)| ep).collect()
}

/// Set by `prunr bench --table`: this process tries the places the
/// registry keeps on the CPU for a slower, different or no-gain result,
/// so they are measured again. A place that fails stays skipped: some
/// fail by hanging or by taking the machine's memory.
static REMEASURE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub fn remeasure_places() {
    let _ = REMEASURE.set(());
}

/// Why `model` runs on the CPU rather than `ep`: its place there (while
/// places are measured again, `remeasuring`, only a failing one), or a
/// provider that cannot serve its kind of session.
fn kept_off(model: prunr_models::ModelId, ep: EpKind, remeasuring: bool) -> Option<prunr_models::Why> {
    if compiles_per_shape(model, ep) {
        return Some(COMPILES_PER_SHAPE);
    }
    match place_here(model, ep.as_str(), openvino_on_gpu(), remeasuring) {
        place @ prunr_models::Place::Cpu(why) if place.kept_off(remeasuring) => Some(why),
        _ => None,
    }
}

fn remeasuring() -> bool {
    REMEASURE.get().is_some()
}

/// Where `model` runs on this machine: the registry's places, each one a
/// test measured here (`ep_compat::measured`) standing in for it. Never
/// over one the registry knows fails: it may hang or take the machine's
/// memory, and the registry may have learned it after the test.
pub fn places_here(model: prunr_models::ModelId) -> prunr_models::Places {
    with_measured(prunr_models::registry_places(model), crate::ep_compat::measured_for(model))
}

fn with_measured(
    mut places: prunr_models::Places,
    measured: [Option<prunr_models::Place>; prunr_models::Column::ALL.len()],
) -> prunr_models::Places {
    for column in prunr_models::Column::ALL {
        if places.get(column).fails() {
            continue;
        }
        if let Some(place) = measured[column.index()] {
            places.set(column, place);
        }
    }
    places
}

fn place_here(model: prunr_models::ModelId, provider: &str, intel_gpu: bool, remeasuring: bool) -> prunr_models::Place {
    prunr_models::place_in(&places_here(model), model, provider, intel_gpu, remeasuring)
}

/// The table's column a session of `model` on `provider` reads here
/// (`prunr_models::column_in`).
pub fn column_for(model: prunr_models::ModelId, provider: &str) -> Option<prunr_models::Column> {
    prunr_models::column_in(&places_here(model), model, provider, openvino_on_gpu(), remeasuring())
}


/// Whether OpenVINO runs `model` on the Intel graphics here.
#[cfg(not(target_os = "macos"))]
pub(crate) fn on_intel_gpu(model: prunr_models::ModelId) -> bool {
    column_for(model, OPENVINO_NAME) == Some(prunr_models::Column::OpenVinoGpu)
}

/// Whether `model`'s sessions on `provider` run on the host CPU: the CPU
/// provider, a provider the table keeps it off, or OpenVINO's processor.
/// Engines are sized by it, so a model the Intel graphics leave to the
/// processor is not sized like graphics work.
pub fn model_on_host_cpu(model: prunr_models::ModelId, provider: &str) -> bool {
    on_host_cpu_at(model, provider, openvino_on_gpu())
}

fn on_host_cpu_at(model: prunr_models::ModelId, provider: &str, intel_gpu: bool) -> bool {
    let places = places_here(model);
    let Some(column) = prunr_models::column_in(&places, model, provider, intel_gpu, false) else { return true };
    column == prunr_models::Column::OpenVino || places.get(column).kept_off(false)
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
    let eps = gpu_eps_for(id, cpu_only);
    let mut failed = Vec::new();
    if !eps.is_empty() {
        let landed = walk_ladder("", id, Weights::Fp32, &eps, |ep| {
            let builder = crate::ort_runtime::session_builder().map_err(|e| Miss::Unusable(e.to_string()))?;
            let built = ep.attach(builder, id, EpSite::Engine { threads: num_cpus::get(), part: None })?;
            let mut session = commit(built).map_err(|e| Miss::failed(format!("ORT session creation failed ({ep}): {e}")))?;
            kept_after_a_run(id, ep, &mut session).map_err(Miss::failed)?;
            Ok(session)
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

#[cfg(not(target_os = "macos"))]
impl<'a> EpSite<'a> {
    /// The OpenVINO device this site asks for, `intel_gpu` when the model
    /// runs on the Intel graphics (`on_intel_gpu`); `None` is OpenVINO's
    /// CPU. Fills stay there.
    fn ov_device(self, intel_gpu: bool) -> Option<&'a str> {
        match self {
            EpSite::Engine { .. } | EpSite::Selection { .. } => intel_gpu.then_some(OV_ENGINE_DEVICE),
            EpSite::Fill { .. } => None,
            EpSite::SdPart { ov_device, .. } => ov_device,
        }
    }
}

/// How a CUDA session is set up for its site, whether the loaded runtime
/// carries CUDA or the CUDA add-on brings it. Each `true` sets an option;
/// the rest keep CUDA's defaults.
#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaTuning {
    /// The pool grows by what is asked, not to the next power of two.
    pool_as_asked: bool,
    heuristic_search: bool,
    tf32: bool,
    /// ORT's memory pattern off. On, it reserves one block of a run's peak
    /// from the second run on, on top of the pool the first run filled,
    /// which can take a model past the card's memory; off, a model holds
    /// its first run's memory at the same speed.
    no_memory_pattern: bool,
}

#[cfg(not(target_os = "macos"))]
impl CudaTuning {
    fn for_site(site: EpSite<'_>) -> Self {
        match site {
            // No CUDA graph: it replays the first run's shapes and buffer
            // addresses, and tiles and photos change both. Heuristic conv
            // search for the same reason: exhaustive benchmarks every new
            // shape, the fixed default algorithm is often the slowest.
            EpSite::Engine { .. } => Self { pool_as_asked: true, heuristic_search: true, tf32: true, no_memory_pattern: true },
            // The pool and search of the engines, for the same reasons,
            // without TF32's lower precision: SD 1.5 on a 12 GB RTX 3060
            // took 7.1 GB for 512 px tiles instead of 9.4, at the same
            // 4.0 s, and its pixels differ from the defaults' no more than
            // two default runs do. The Magic Brush's masks on it match the
            // CPU's (same pick on 20 of 20 clicks, IoU 0.9998).
            // A Stable Diffusion part sets its memory pattern from its own
            // memory plan.
            EpSite::SdPart { .. } => Self { pool_as_asked: true, heuristic_search: true, tf32: false, no_memory_pattern: false },
            EpSite::Selection { .. } => Self { pool_as_asked: true, heuristic_search: true, tf32: false, no_memory_pattern: true },
            EpSite::Fill { .. } => Self { pool_as_asked: false, heuristic_search: false, tf32: false, no_memory_pattern: true },
        }
    }

    /// The loaded runtime's own CUDA provider, so tuned.
    fn provider(self) -> ExecutionProviderDispatch {
        let mut p = ort::ep::CUDA::default().with_device_id(0);
        if self.pool_as_asked {
            p = p.with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
        }
        if self.heuristic_search {
            p = p.with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic);
        }
        if self.tf32 {
            p = p.with_tf32(true);
        }
        p.build()
    }

    /// The same for the add-on, as `with_devices` takes it: each key
    /// prefixed with the provider's name; the pool's under `arena.`, the
    /// only spelling ORT 1.30 hands a plugin's pool (an unknown key is
    /// ignored without a word, and the default power-of-two growth took
    /// a 12 GB card for BiRefNet-lite).
    fn addon_options(self) -> Vec<(String, String)> {
        [
            (self.pool_as_asked, "arena.extend_strategy", "1"),
            (self.heuristic_search, "cudnn_conv_algo_search", "HEURISTIC"),
            (self.tf32, "use_tf32", "1"),
        ]
        .into_iter()
        .filter(|&(set, ..)| set)
        .map(|(_, key, value)| (format!("{}.{key}", crate::ort_runtime::Addon::Cuda.ep_name()), value.to_string()))
        .collect()
    }
}

impl EpKind {
    /// `builder` with this provider tuned for `site` (`provider`) and the
    /// session options it needs.
    pub(crate) fn attach(self, builder: ort::session::builder::SessionBuilder, model: prunr_models::ModelId, site: EpSite<'_>) -> Result<ort::session::builder::SessionBuilder, Miss> {
        let builder = crate::ort_runtime::without_optimizers(builder, self.disabled_optimizers()).map_err(|e| Miss::Unusable(e.to_string()))?;
        let builder = match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda if CudaTuning::for_site(site).no_memory_pattern => builder.with_memory_pattern(false).map_err(|e| Miss::Unusable(e.to_string()))?,
            _ => builder,
        };
        let registered = match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda if crate::ort_runtime::Addon::Cuda.registered() => {
                builder.with_devices(crate::ort_runtime::addon_devices(crate::ort_runtime::Addon::Cuda), Some(&CudaTuning::for_site(site).addon_options()))
            }
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino if crate::ort_runtime::Addon::OpenVino.registered() => {
                let session = OpenVinoTuning::for_site(site, model).addon_session().map_err(Miss::Unusable)?;
                builder.with_devices(session.devices, Some(&session.options))
            }
            _ => builder.with_execution_providers([self.provider(model, site)]),
        };
        registered.map_err(|e| Miss::Unusable(format!("ORT register {self} EP failed: {e}")))
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
            EpKind::Cuda => CudaTuning::for_site(site).provider(),
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
            #[cfg(not(target_os = "macos"))]
            EpKind::OpenVino => OpenVinoTuning::for_site(site, model).provider(),
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

    /// Whether this provider's sessions build one at a time in a process,
    /// whichever model or thread builds them: DirectML's
    /// `commit_from_memory` races in AbiCustomRegistry when two run at once.
    fn builds_one_at_a_time(self) -> bool {
        match self {
            #[cfg(windows)]
            EpKind::DirectMl => true,
            _ => false,
        }
    }

    /// Whether a session this provider took can still fail its first run:
    /// CUDA loads cuDNN's parts on first use, DirectML builds its fused
    /// nodes then, and MIGraphX compiles a free size when it first meets
    /// it. OpenVINO and CoreML compile the graph for its sizes as the
    /// session builds.
    pub(crate) fn may_fail_its_first_run(self) -> bool {
        match self {
            #[cfg(not(target_os = "macos"))]
            EpKind::Cuda => true,
            #[cfg(windows)]
            EpKind::DirectMl => true,
            #[cfg(target_os = "linux")]
            EpKind::MiGraphX => true,
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

impl Miss {
    /// A failure with no cached graph to suspect.
    pub(crate) fn failed(error: String) -> Self {
        Self::Failed { error, suspect_cache: None }
    }
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
    if let Some(why) = kept_off(model, ep, remeasuring()) {
        Some(Skip::KeptOff(why))
    } else if crate::ep_compat::is_known_failure(ep, model, weights, asks_intel_gpu(ep, model)) {
        Some(Skip::FailedBefore)
    } else {
        None
    }
}

/// Whether `model`'s session on `ep` asks for the Intel GPU: OpenVINO
/// engines with the switch on. Fills stay on the CPU, and SD parts carry
/// their own device. A failure there is its own (`ep_compat`).
pub(crate) fn asks_intel_gpu(ep: EpKind, model: prunr_models::ModelId) -> bool {
    asks_intel_gpu_as(ep, model, remeasuring())
}

/// `asks_intel_gpu` with places measured again (`remeasuring`) or not, as
/// the failure a test looks up is keyed the way the test reads places.
fn asks_intel_gpu_as(ep: EpKind, model: prunr_models::ModelId, remeasuring: bool) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        ep == EpKind::OpenVino
            && prunr_models::column_in(&places_here(model), model, OPENVINO_NAME, openvino_on_gpu(), remeasuring) == Some(prunr_models::Column::OpenVinoGpu)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (ep, model, remeasuring);
        false
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

/// The size a test run gives a free input size the model names no size
/// for (`first_input`).
const TEST_SIDE: usize = 256;

/// The side a test run of `model` on `ep` gives its free sizes.
fn test_side(model: prunr_models::ModelId, ep: EpKind) -> usize {
    first_input(model, ep.as_str()).map_or(TEST_SIDE, |shape| shape[2])
}

/// Whether `session` of `model` may stay on `ep`: where a provider can
/// take a session and fail its first run (`EpKind::may_fail_its_first_run`)
/// and this machine's test has not run the model there, one run on zeros,
/// so a model the provider cannot run lands on the next one instead of
/// failing the user's run.
pub(crate) fn kept_after_a_run(model: prunr_models::ModelId, ep: EpKind, session: &mut Session) -> Result<(), String> {
    if !ep.may_fail_its_first_run() || ran_here(model, ep) {
        return Ok(());
    }
    run_on_zeros(session, test_side(model, ep)).map_err(|e| format!("{ep} test run failed: {e}"))
}

/// Whether this machine's test ran `model` on `ep` and kept it there.
fn ran_here(model: prunr_models::ModelId, ep: EpKind) -> bool {
    column_for(model, ep.as_str()).is_some_and(|column| crate::ep_compat::measured_for(model)[column.index()].is_some_and(prunr_models::Place::runs_there))
}

/// One run of `session` on zeros. An input it cannot fill (not a float
/// tensor, or a free channel axis) is no verdict: the run is skipped.
fn run_on_zeros(session: &mut Session, side: usize) -> Result<(), String> {
    let Some(inputs) = session.inputs().iter().map(|input| Some((input.name().to_string(), zeros_for(input, side)?))).collect::<Option<Vec<_>>>() else {
        tracing::debug!("a session's inputs take no test input; not run");
        return Ok(());
    };
    session.run(inputs).map(drop).map_err(|e| e.to_string())
}

fn zeros_for(input: &Outlet, side: usize) -> Option<DynValue> {
    let ty = input.dtype();
    let shape = concrete_shape(ty.tensor_shape()?, side)?;
    let count = shape.iter().product();
    match ty.tensor_type()? {
        TensorElementType::Float32 => Tensor::from_array((shape, vec![0f32; count])).ok().map(Tensor::into_dyn),
        TensorElementType::Float16 => Tensor::from_array((shape, vec![half::f16::ZERO; count])).ok().map(Tensor::into_dyn),
        _ => None,
    }
}

/// `dims` with a free batch 1 and each free size after the channels
/// `side`; `None` for a free channel axis, whose count no size says.
fn concrete_shape(dims: &[i64], side: usize) -> Option<Vec<usize>> {
    dims.iter().enumerate().map(|(axis, &d)| match axis {
        _ if d > 0 => Some(d as usize),
        0 => Some(1),
        1 => None,
        _ => Some(side),
    }).collect()
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
        let committed = {
            let _turn = ep.builds_one_at_a_time().then(|| ONE_BUILD_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
            let mut committed = commit(ep);
            // A build that failed while writing its optimized graph (the
            // disk filled meanwhile) says nothing about the provider until
            // it fails without one too.
            if matches!(committed, Err(Miss::Failed { suspect_cache: None, .. })) && crate::cache::drafts_attached() > drafted {
                tracing::warn!(?model, ep = %ep, "{prefix}session failed while writing its optimized graph; building it without");
                committed = crate::cache::without_drafts(|| commit(ep));
            }
            committed
        };
        match committed {
            Ok(session) => return Ok((session, ep, walk.failed)),
            Err(Miss::Unusable(error)) => {
                tracing::warn!(?model, ep = %ep, %error, "{prefix}GPU EP unusable — trying next EP");
                walk.failed.push(error);
            }
            Err(Miss::Failed { error, suspect_cache }) => {
                tracing::warn!(?model, ?weights, ep = %ep, %error, "{prefix}GPU session failed — trying next EP");
                match suspect_cache {
                    // Out of memory or a lost device says nothing against the graph.
                    Some(_) if crate::ep_compat::is_transient(&error) => {}
                    Some(graph) => drop_cached_graph(&graph),
                    None => match compiled_cache_named(model, ep.as_str(), &error, started) {
                        Some(compiled) => clear_suspect_cache(&compiled),
                        None => crate::ep_compat::record_failure(ep, model, weights, asks_intel_gpu(ep, model), &error),
                    },
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
        EpKind::OpenVino if crate::ort_runtime::openvino_may_use_graphics() => prunr_models::Column::OpenVinoGpu.name().to_string(),
        _ => ep.as_str().to_string(),
    }).collect()
}

/// `on_engine`, a run of `model` on `provider`; when that ran out of
/// memory on a graphics card, `on_cpu`: a run the card has no room for
/// still finishes, on the processor, when the processor has the room.
pub fn retry_on_cpu<T>(
    model: prunr_models::ModelId,
    provider: &str,
    progress: &crate::Progress,
    on_engine: impl FnOnce() -> Result<T, CoreError>,
    on_cpu: impl FnOnce() -> Result<T, CoreError>,
) -> Result<T, CoreError> {
    let on_card = !runs_on_host_cpu(provider);
    if on_card && crate::memory::ran_short(model, provider) {
        tracing::info!(?model, provider, "the card had no room for it and has gained none: on the processor");
        progress.step(crate::progress::Step::OnTheProcessor);
        return on_cpu();
    }
    match on_engine() {
        Err(e) if on_card && e.is_out_of_memory() => {
            if e.is_card_out_of_memory() {
                crate::memory::note_ran_short(model, provider);
            }
            if let Err(short) = crate::memory::fits_now(model, CPU_NAME) {
                tracing::warn!(?model, %e, %short, "out of graphics memory, and the processor has no room either");
                return Err(e);
            }
            tracing::warn!(?model, %e, "out of graphics memory; the run goes on on the processor");
            progress.step(crate::progress::Step::OnTheProcessor);
            on_cpu()
        }
        other => other,
    }
}

/// The GPU providers this process runs models on, by their names, once
/// the probe has answered; `None` before, as the UI thread must not wait
/// for a GPU driver.
pub fn gpu_providers_now() -> Option<Vec<&'static str>> {
    OrtEngine::probed().then(|| available_gpu_eps().iter().map(|ep| ep.as_str()).collect())
}

/// The lighter copy of `id`'s weights on disk a session may try before
/// its own: FP16 on a GPU in `eps` that gains from it, INT8 on the CPU.
/// Never on macOS, where CoreML converts to FP16 itself.
fn lighter_copy(id: prunr_models::ModelId, cpu_only: bool, eps: &[EpKind]) -> Option<Weights> {
    if cfg!(target_os = "macos") {
        None
    } else if cpu_only {
        prunr_models::has_int8_variant(id).then_some(Weights::Int8)
    } else {
        (prunr_models::has_fp16_variant(id) && !fp16_eps(id, eps).is_empty()).then_some(Weights::Fp16)
    }
}

/// `lighter_copy` as a session built now takes it: a copy kept for small
/// cards only where the card cannot hold the model's own weights. Reads
/// the card, so only a session build asks.
fn lighter_weights(id: prunr_models::ModelId, cpu_only: bool, eps: &[EpKind]) -> Option<Weights> {
    lighter_copy(id, cpu_only, eps).filter(|&weights| {
        weights != Weights::Fp16
            || prunr_models::half_precision_copy(id).is_none()
            || fp16_eps(id, eps).first().is_some_and(|ep| crate::memory::copy_in_use(id, ep.as_str(), || crate::memory::card_free_mb(ep.as_str())).is_some())
    })
}

/// The providers in `eps` where `id` gains from its FP16 copy. One on the
/// host CPU widens FP16 back to FP32: OpenVINO on the processor ran
/// Real-ESRGAN x4 no faster (7.78 s a 256 tile against 7.69 s on an
/// i7-6700) and 0.8/255 off on average, up to 10.
fn fp16_eps(id: prunr_models::ModelId, eps: &[EpKind]) -> Vec<EpKind> {
    eps.iter().copied().filter(|&ep| !model_on_host_cpu(id, ep.as_str())).collect()
}

/// Why a GPU session of `model` passes `ep` over for every weights it
/// tries (`None`: it tries `ep`).
fn gpu_skip(model: prunr_models::ModelId, ep: EpKind, remeasuring: bool) -> Option<Skip> {
    if let Some(why) = kept_off(model, ep, remeasuring) {
        return Some(Skip::KeptOff(why));
    }
    let intel_gpu = asks_intel_gpu_as(ep, model, remeasuring);
    [lighter_copy(model, false, &[ep]), Some(Weights::Fp32)].into_iter().flatten()
        .all(|weights| crate::ep_compat::is_known_failure(ep, model, weights, intel_gpu))
        .then_some(Skip::FailedBefore)
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
    gpu_eps_for(model, cpu_only).into_iter().find(|&ep| gpu_skip(model, ep, remeasuring()).is_none()).map(|ep| ep.as_str())
}

/// Each GPU provider the loaded runtime offers, in the order a session of
/// `model` tries them, with why it would pass one over (`None`: it would
/// try it).
pub fn gpu_ladder(model: prunr_models::ModelId) -> Vec<(&'static str, Option<Skip>)> {
    ladder(model, remeasuring())
}

/// What a test of this computer runs `model` on, in the order it tries
/// them: each GPU provider it would not pass over, with the column its
/// time goes in. A test tries again the places Prunr's own tests keep on
/// the CPU (`remeasure_places`), and this reads them so in any process.
pub fn tries_in_a_test(model: prunr_models::ModelId) -> Vec<(&'static str, prunr_models::Column)> {
    let places = places_here(model);
    ladder(model, true)
        .into_iter()
        .filter(|(_, skip)| skip.is_none())
        .filter_map(|(provider, _)| Some((provider, prunr_models::column_in(&places, model, provider, openvino_on_gpu(), true)?)))
        .collect()
}

fn ladder(model: prunr_models::ModelId, remeasuring: bool) -> Vec<(&'static str, Option<Skip>)> {
    gpu_eps_for(model, false).into_iter().map(|ep| (ep.as_str(), gpu_skip(model, ep, remeasuring))).collect()
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
    /// A smaller tile than the model's own an upscale on it last ran whole
    /// on, and the card's free memory then: the next starts there, so a
    /// card that ran out at the model's own tile is not tried at it again
    /// for every image, until the card has gained room.
    upscale_tile: Mutex<Option<(u32, Option<u64>)>>,
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
            .map_err(|e| Miss::failed(format!("ORT session creation failed ({ep}, input {key}): {e}")))?;
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
        prunr_models::ModelCategory::Depth => crate::depth::DEPTH_SHORT_SIDE,
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
        Self::build_with(model, intra_threads, &gpu_eps_for(model.into(), cpu_only), cpu_only, GraphOptimizationLevel::Level3)
    }

    /// Each weights a session tries in turn (its lighter copy, when on
    /// disk, then its own) on the providers in `eps` (FP16 on those that
    /// gain from it, `fp16_eps`), or on the CPU when
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
        for weights in [lighter_weights(id, cpu_only, eps), Some(Weights::Fp32)].into_iter().flatten() {
            let eps = if weights == Weights::Fp16 { Cow::Owned(fp16_eps(id, eps)) } else { Cow::Borrowed(eps) };
            if let Some(walk) = (!cpu_only).then(|| Walk::without_a_try(id, weights, &eps)).flatten() {
                failed.push(format!("{} weights: {}", weights.label(), CoreError::Inference(walk.reason())));
                continue;
            }
            let Some(bytes) = Self::weights_bytes(id, weights)? else {
                tracing::debug!(?model, ?weights, "lighter weights gone from disk — skipping them");
                continue;
            };
            match Self::build_session(&bytes, weights, intra_threads, model, &eps, level) {
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
        Self::build_with(kind, intra_threads, &gpu_eps_for(model, cpu_only), cpu_only, crate::upscale::pick_optimization_level(descriptor))
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
                upscale_tile: Default::default(),
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
            let suspect = || cached_graph(matches!(bytes_owner, Cow::Owned(_)), model_id, ep.as_str(), weights, None);
            let mut session = built.commit_from_memory(&bytes_owner).map_err(|e| Miss::Failed {
                error: format!("ORT session creation failed ({ep}): {e}"),
                suspect_cache: suspect(),
            })?;
            kept_after_a_run(model_id, ep, &mut session).map_err(|error| Miss::Failed { error, suspect_cache: suspect() })?;
            if let Some(draft) = draft { draft.publish(); }
            tracing::info!(?model, ?weights, ep = %ep, elapsed_ms = started.elapsed().as_millis() as u64, "session committed");
            coverage.report(model_id, None);
            Ok(Sessions::One(Mutex::new(session)))
        }).map_err(|walk| CoreError::Inference(walk.reason()))?;
        Ok(Self {
            sessions,
            pick: SessionPick { weights, failed, ..SessionPick::on(ep.as_str()) },
            model_kind: model,
            upscale_tile: Default::default(),
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
                let mut session = built.commit_from_memory(bytes)
                    .map_err(|e| Miss::failed(format!("ORT session creation failed ({key}): {e}")))?;
                ep.map_or(Ok(()), |ep| kept_after_a_run(id, ep, &mut session)).map_err(|error| Miss::failed(format!("{key}: {error}")))?;
                coverage.report(id, Some(key));
                Ok((*key, session))
            }).collect()
        };
        let started = Instant::now();
        let (sessions, pick) = match walk_ladder("", id, Weights::Fp32, &gpu_eps_for(id, cpu_only), |ep| commit_all(Some(ep))) {
            Ok((sessions, ep, failed)) => (sessions, SessionPick { failed, ..SessionPick::on(ep.as_str()) }),
            Err(walk) => {
                let sessions = commit_all(None).map_err(|miss| CoreError::Inference(miss_text(miss)))?;
                (sessions, SessionPick { failed: walk.failed, ..SessionPick::on(CPU_NAME) })
            }
        };
        tracing::info!(?model, parts = sessions.len(), provider = %pick.provider, elapsed_ms = started.elapsed().as_millis() as u64, "part sessions committed");
        let resident_mb = rss_before.zip(crate::sysmem::process_rss_mb()).map_or(0, |(before, after)| after.saturating_sub(before));
        let net = crate::upscale::blocks::BlockNet::new(sessions, norms, net, resident_mb * 1024 * 1024)?;
        Ok(Self { sessions: Sessions::Parts(Box::new(net)), pick, model_kind: model, upscale_tile: Default::default() })
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
    { crate::ort_runtime::openvino_may_use_graphics() }
    #[cfg(target_os = "macos")]
    { false }
}

pub(crate) fn on_host_cpu(provider: &str, openvino_on_gpu: bool) -> bool {
    provider.eq_ignore_ascii_case(CPU_NAME)
        || (provider.eq_ignore_ascii_case(OPENVINO_NAME) && !openvino_on_gpu)
}

/// Held while a provider that `builds_one_at_a_time` builds a session
/// (and runs its test): the ladder takes it, so every builder does.
static ONE_BUILD_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Engines of one model build one after another under CoreML: it compiles
/// each engine into the model's one cache folder, where two at once left a
/// half-written cache that was then recorded as CoreML failing the model.
pub(crate) fn engines_build_one_at_a_time() -> bool {
    OrtEngine::detect_active_provider().eq_ignore_ascii_case(COREML_NAME)
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

    /// The tile an upscale on this engine starts at: the smaller one its
    /// last image fit, while the card has gained no room since; else the
    /// model's `own`.
    pub(crate) fn upscale_tile(&self, own: u32) -> u32 {
        let fitted = *self.upscale_tile.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        start_tile(own, fitted, || crate::memory::card_room_mb(self.active_provider()))
    }

    /// An upscale on this engine ran whole on `tile`. The card's room is
    /// kept from when the tile first had to shrink, so gains add up.
    pub(crate) fn set_upscale_tile(&self, tile: u32, own: u32) {
        let mut fitted = self.upscale_tile.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *fitted = fitted_after(*fitted, tile, own, || crate::memory::card_room_mb(self.active_provider()));
    }
}

/// What an engine keeps after an image ran whole on `tile`: nothing at the
/// model's `own` tile; a smaller one with the card's room, kept from when
/// that tile was first needed.
fn fitted_after(fitted: Option<(u32, Option<u64>)>, tile: u32, own: u32, room_now: impl FnOnce() -> Option<u64>) -> Option<(u32, Option<u64>)> {
    match fitted {
        _ if tile >= own => None,
        Some((held, room)) if held == tile => Some((held, room)),
        _ => Some((tile, room_now())),
    }
}

/// The tile to start at: `fitted`'s, a smaller one the last image fit and
/// the card's room then, while `room_now` says the card has gained none
/// since; else `own`.
fn start_tile(own: u32, fitted: Option<(u32, Option<u64>)>, room_now: impl FnOnce() -> Option<u64>) -> u32 {
    match fitted {
        Some((tile, then)) if crate::memory::still_short(then, room_now()) => tile,
        _ => own,
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

/// An optimized graph a session failed on: deleted, and saved again by
/// the next build. One this machine saved will not load back, so it is
/// no longer read; it is still written, which keeps Nomos8k's run in
/// 0.8 GB, not 2.7.
fn drop_cached_graph(path: &std::path::Path) {
    clear_suspect_cache(path);
    crate::cache::note_graph_failed(path);
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
            drop_cached_graph(&cached);
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
    if !crate::cache::graph_not_read(&path) {
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
    /// A run a card ran out of memory for goes on on the processor and
    /// says so; any other failure, or a run already on the processor,
    /// keeps its error. Each case its own model: the note on the card is
    /// the process's.
    #[test]
    fn a_card_out_of_memory_goes_on_on_the_processor() {
        use crate::types::CoreError;
        use prunr_models::ModelId;
        use std::sync::Arc;
        let oom = || Err::<u8, _>(CoreError::Inference("CUDA error: out of memory".into()));
        let card = "CUDAExecutionProvider";
        let sink = Arc::new(crate::progress::RecordingSink::default());
        assert_eq!(super::retry_on_cpu(ModelId::Silueta, card, &crate::Progress::new(sink.clone()), oom, || Ok(7)).ok(), Some(7));
        assert!(sink.updates().contains(&crate::ProgressUpdate::Step(crate::Step::OnTheProcessor)));
        crate::progress::check_contract(&sink.updates()).expect("it names its step");
        assert!(super::retry_on_cpu(ModelId::U2net, super::CPU_NAME, &crate::Progress::none(), oom, || Ok(7)).is_err(), "already on the processor");
        let broken = || Err::<u8, _>(CoreError::Inference("Graph contains cycles".into()));
        assert!(super::retry_on_cpu(ModelId::Teed, card, &crate::Progress::none(), broken, || Ok(7)).is_err(), "not a memory failure");
        assert!(super::retry_on_cpu(ModelId::Migan, card, &crate::Progress::none(), oom, oom).is_err_and(|e| e.is_out_of_memory()), "out of memory on the processor too");
    }

    /// The card's room is kept from when a smaller tile was first needed,
    /// so small gains image after image add up; the model's own tile
    /// keeps nothing.
    #[test]
    fn a_smaller_tile_keeps_the_room_it_was_first_needed_at() {
        let first = super::fitted_after(None, 256, 512, || Some(2_000));
        assert_eq!(first, Some((256, Some(2_000))));
        assert_eq!(super::fitted_after(first, 256, 512, || Some(2_600)), Some((256, Some(2_000))), "same tile: the first room stays");
        assert_eq!(super::fitted_after(first, 128, 512, || Some(1_500)), Some((128, Some(1_500))), "a smaller one: its own room");
        assert_eq!(super::fitted_after(first, 512, 512, || Some(9_000)), None);
    }

    /// An upscale starts at the smaller tile its last image fit while the
    /// card has gained less than a gigabyte, or there is no reading.
    #[test]
    fn a_smaller_tile_holds_until_the_card_gains_room() {
        assert_eq!(super::start_tile(512, None, || Some(9_000)), 512);
        assert_eq!(super::start_tile(512, Some((128, Some(2_000))), || Some(2_500)), 128);
        assert_eq!(super::start_tile(512, Some((128, Some(2_000))), || Some(3_100)), 512, "a GB more is free");
        assert_eq!(super::start_tile(512, Some((128, None)), || Some(9_000)), 128, "no reading then");
        assert_eq!(super::start_tile(512, Some((256, Some(2_000))), || None), 256, "no reading now");
    }
    use super::*;

    /// A model reshaping its one input `x` ([batch, 3, h, w], each free)
    /// to `to`, hand-encoded: a session takes it whatever `to` is, and its
    /// run fails when the input's size does not divide into `to`.
    fn reshape_model(to: [i64; 2]) -> Vec<u8> {
        fn varint(mut v: u64) -> Vec<u8> {
            let mut out = Vec::new();
            loop {
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(byte);
                    return out;
                }
                out.push(byte | 0x80);
            }
        }
        let number = |field: u64, value: u64| [varint(field << 3), varint(value)].concat();
        let bytes = |field: u64, payload: &[u8]| [varint(field << 3 | 2), varint(payload.len() as u64), payload.to_vec()].concat();
        let free = |name: &str| bytes(1, &bytes(2, name.as_bytes()));
        let fixed = |size: u64| bytes(1, &number(1, size));
        let float_tensor = |shape: Vec<u8>| bytes(1, &[number(1, 1), shape].concat());
        let x = [bytes(1, b"x"), bytes(2, &float_tensor(bytes(2, &[free("n"), fixed(3), free("h"), free("w")].concat())))].concat();
        let y = [bytes(1, b"y"), bytes(2, &float_tensor(Vec::new()))].concat();
        let target: Vec<u8> = to.iter().flat_map(|v| v.to_le_bytes()).collect();
        let to = [number(1, 2), number(2, 7), bytes(8, b"to"), bytes(9, &target)].concat();
        let node = [bytes(1, b"x"), bytes(1, b"to"), bytes(2, b"y"), bytes(4, b"Reshape")].concat();
        let graph = [bytes(1, &node), bytes(2, b"g"), bytes(5, &to), bytes(11, &x), bytes(12, &y)].concat();
        [number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat()
    }

    /// A CUDA or DirectML session stays only once it has run: one whose
    /// run fails goes to the next provider. Providers that compile the
    /// whole graph as the session builds are not run. Silueta's input has
    /// no free size, so the model's is 256: 3·256·256 values make no rows
    /// of 7.
    #[test]
    fn a_session_that_cannot_run_is_not_kept() {
        if crate::ort_runtime::runtime_for_tests().is_err() {
            return;
        }
        let session = |to| crate::ort_runtime::session_builder().unwrap().commit_from_memory(&reshape_model(to)).expect("any reshape commits");
        let silueta = prunr_models::ModelId::Silueta;
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(kept_after_a_run(silueta, EpKind::Cuda, &mut session([3, -1])), Ok(()));
            let refused = kept_after_a_run(silueta, EpKind::Cuda, &mut session([-1, 7])).expect_err("no rows of 7");
            assert!(refused.starts_with("CUDA test run failed"), "{refused}");
            assert_eq!(kept_after_a_run(silueta, EpKind::OpenVino, &mut session([-1, 7])), Ok(()));
        }
        #[cfg(target_os = "macos")]
        assert_eq!(kept_after_a_run(silueta, EpKind::CoreMl, &mut session([-1, 7])), Ok(()));
    }

    /// A test run's free sizes take the side and a free batch 1
    /// (3·16·16 = 768 values make rows of 256 only then).
    #[test]
    fn a_test_run_is_one_image_of_the_side() {
        if crate::ort_runtime::runtime_for_tests().is_err() {
            return;
        }
        let mut session = crate::ort_runtime::session_builder().unwrap().commit_from_memory(&reshape_model([3, 256])).unwrap();
        assert_eq!(run_on_zeros(&mut session, 16), Ok(()));
        assert!(run_on_zeros(&mut session, 17).is_err());
    }

    /// With the OpenVINO add-on registered, a session that names no
    /// provider runs on ORT's own CPU provider: ORT's choice for such a
    /// session put it on OpenVINO's processor under every policy, the
    /// default too.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_session_naming_no_provider_stays_on_ort_s_cpu() {
        use crate::ort_runtime::{addon_devices, register_addon, use_addon, Addon};
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let Some(plugin) = crate::ort_runtime::openvino_plugin_for_tests() else { return };
        use_addon(Addon::OpenVino, plugin.into());
        if !register_addon(Addon::OpenVino) || addon_devices(Addon::OpenVino).is_empty() {
            eprintln!("SKIP: OpenVINO lists no device here");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let ran_on = |devices: Vec<ort::device::Device<'static>>, run: &str| -> std::collections::BTreeSet<String> {
            let mut built = crate::ort_runtime::session_builder().unwrap().with_profiling(dir.path().join(run)).unwrap();
            if !devices.is_empty() {
                built = built.with_devices(devices, None).unwrap();
            }
            let mut session = built.commit_from_memory(&reshape_model([3, -1])).unwrap();
            assert_eq!(run_on_zeros(&mut session, 2), Ok(()));
            let profile: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(session.end_profiling().unwrap()).unwrap()).unwrap();
            profile.as_array().unwrap().iter().filter_map(|event| Some(event["args"]["provider"].as_str()?.to_string())).collect()
        };
        assert_eq!(ran_on(Vec::new(), "none"), ["CPUExecutionProvider".to_string()].into());
        let processor: Vec<_> = addon_devices(Addon::OpenVino).into_iter().filter(|d| d.hardware_device().ty() == ort::memory::DeviceType::CPU).collect();
        if !processor.is_empty() {
            assert_eq!(ran_on(processor, "named"), ["OpenVINOExecutionProvider".to_string()].into(), "a session naming OpenVINO keeps it");
        }
    }

    /// OpenVINO is tuned per site, for its own provider and its add-on
    /// alike: the thread cap for engines and fills, one stream and dynamic
    /// shapes for a Stable Diffusion part, neither for the Magic Brush.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn openvino_is_tuned_per_site() {
        let lama = prunr_models::ModelId::LaMaFp32;
        let tuned = |site| OpenVinoTuning::for_site(site, lama);
        assert_eq!(tuned(EpSite::Fill { threads: 3 }), OpenVinoTuning { device: None, threads: Some(3), sd_part: false });
        assert_eq!(tuned(EpSite::Engine { threads: 4, part: None }).threads, Some(4));
        assert_eq!(tuned(EpSite::Selection { part: "encoder" }), OpenVinoTuning::default());
        assert_eq!(tuned(EpSite::SdPart { part: "unet", ov_device: Some("GPU") }), OpenVinoTuning { device: Some("GPU"), threads: None, sd_part: true });
    }

    /// OpenVINO's settings reach the device they are for: the thread cap
    /// the processor only, under AUTO's name when AUTO picks the device,
    /// and nothing when nothing is set.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn openvino_settings_name_the_device_they_are_for() {
        let config = |meta, devices: &[&str], threads, one_stream| openvino_load_config(meta, devices, threads, one_stream);
        assert_eq!(config(None, &["CPU"], Some(4), false).as_deref(), Some(r#"{"CPU":{"INFERENCE_NUM_THREADS":"4"}}"#));
        assert_eq!(config(Some("AUTO"), &["GPU", "CPU"], Some(4), false).as_deref(), Some(r#"{"AUTO":{"CPU":{"INFERENCE_NUM_THREADS":"4"}}}"#));
        assert_eq!(config(None, &["GPU"], None, true).as_deref(), Some(r#"{"GPU":{"NUM_STREAMS":"1"}}"#));
        assert_eq!(config(None, &["GPU"], Some(4), false), None, "no thread cap off the processor");
        assert_eq!(config(None, &["CPU"], None, false), None);
    }

    /// With the OpenVINO add-on started, an engine runs on it.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn an_engine_runs_on_the_openvino_add_on() {
        use crate::ort_runtime::{use_addon, Addon};
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let Some(plugin) = crate::ort_runtime::openvino_plugin_for_tests() else { return };
        use_addon(Addon::OpenVino, plugin.into());
        if let Err(why) = start_addon(Addon::OpenVino) {
            eprintln!("SKIP: {why}");
            return;
        }
        assert_eq!(available_gpu_eps().last(), Some(&EpKind::OpenVino), "OpenVINO goes last");
        let engine = OrtEngine::for_model(prunr_models::ModelId::RealEsrganX2Plus, 2, false).expect("Real-ESRGAN x2 loads");
        assert_eq!(engine.active_provider(), OPENVINO_NAME);
    }

    /// A free channel axis says no count, so a session with one is not run.
    #[test]
    fn a_free_channel_axis_takes_no_test_input() {
        assert_eq!(concrete_shape(&[-1, 3, -1, -1], 64), Some(vec![1, 3, 64, 64]));
        assert_eq!(concrete_shape(&[1, 3, 512, 512], 64), Some(vec![1, 3, 512, 512]));
        assert_eq!(concrete_shape(&[-1, -1, -1, -1], 64), None);
    }

    /// On OpenVINO's processor a session loads the model's own weights,
    /// not its FP16 copy, which is widened back there.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn openvino_on_the_processor_loads_full_precision() {
        let id = prunr_models::ModelId::RealEsrganX4Plus;
        if crate::ort_runtime::runtime_for_tests().is_err() || !available_gpu_eps().contains(&EpKind::OpenVino) || on_intel_gpu(id) || !prunr_models::has_fp16_variant(id) {
            eprintln!("SKIP: needs OpenVINO on the processor and Real-ESRGAN x4's FP16 copy");
            return;
        }
        let engine = OrtEngine::for_model(id, 2, false).unwrap();
        assert_eq!((engine.active_provider(), engine.pick().weights), (OPENVINO_NAME, Weights::Fp32));
    }

    /// Real-ESRGAN x2 pads to a multiple but has no window attention, so
    /// it takes Level3: two picture shapes in one session, as Level2 does.
    #[test]
    fn real_esrgan_x2_at_level3_matches_level2_across_shapes() {
        let id = prunr_models::ModelId::RealEsrganX2Plus;
        if crate::ort_runtime::runtime_for_tests().is_err() || !prunr_models::is_available(id) {
            eprintln!("SKIP: needs Real-ESRGAN x2");
            return;
        }
        let _own = crate::cache::OwnCacheRoot::new("x2-levels");
        let kind = ModelKind::try_from(id).unwrap();
        let pictures = [crate::upscale::ripple_picture(64, 48), crate::upscale::ripple_picture(40, 90)];
        let upscaled = |level| {
            crate::cache::clear_for_model(id);
            let engine = OrtEngine::build_with(kind, 2, &[], true, level).unwrap();
            pictures.iter().map(|p| crate::upscale::upscale_rgba_with_engine(p, &engine, id, 2, &crate::Progress::none(), None).unwrap()).collect::<Vec<_>>()
        };
        for (two, three) in upscaled(GraphOptimizationLevel::Level2).iter().zip(&upscaled(GraphOptimizationLevel::Level3)) {
            assert_eq!(two.dimensions(), three.dimensions());
            assert!(two.pixels().zip(three.pixels()).all(|(a, b)| (0..3).all(|k| a[k].abs_diff(b[k]) <= 1)), "{:?}", two.dimensions());
        }
    }

    /// DirectML on models it refuses or breaks on, under each set of
    /// session switches: does each load, run, and land near the CPU?
    /// `PRUNR_DML_PROBE` lists the files, `;`-separated; a folder stands
    /// for every .onnx in it. Free dimensions take `PRUNR_DML_PROBE_SIDE`.
    #[cfg(windows)]
    #[test]
    #[ignore = "real models on DirectML"]
    fn directml_switches_probe() {
        let Ok(list) = std::env::var("PRUNR_DML_PROBE") else { return };
        let side: usize = std::env::var("PRUNR_DML_PROBE_SIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(TEST_SIDE);
        let files: Vec<PathBuf> = list.split(';').map(PathBuf::from).flat_map(|p| {
            if p.is_dir() {
                let mut v: Vec<PathBuf> = std::fs::read_dir(&p).unwrap().flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "onnx")).collect();
                v.sort();
                v
            } else {
                vec![p]
            }
        }).collect();
        let build = |bytes: &[u8], dml: bool, folding: bool, no_fusion: bool, no_mem_pattern: bool, level: GraphOptimizationLevel| -> Result<Session, String> {
            let mut b = crate::ort_runtime::session_builder().map_err(|e| e.to_string())?
                .with_optimization_level(level).map_err(|e| e.to_string())?
                .with_intra_threads(4).map_err(|e| e.to_string())?;
            if folding {
                b = b.with_disabled_optimizers("NoSuchOptimizer").map_err(|e| e.to_string())?;
            }
            if no_fusion {
                b = b.with_config_entry("ep.dml.disable_graph_fusion", "1").map_err(|e| e.to_string())?;
            }
            if no_mem_pattern {
                b = b.with_memory_pattern(false).map_err(|e| e.to_string())?;
            }
            if dml {
                b = b.with_execution_providers([directml_provider()]).map_err(|e| e.to_string())?;
            }
            b.commit_from_memory(bytes).map_err(|e| e.to_string().lines().next().unwrap_or_default().to_string())
        };
        let inputs_of = |session: &Session| -> Vec<(String, DynValue)> {
            session.inputs().iter().enumerate().map(|(n, input)| {
                let ty = input.dtype();
                let shape = ty.tensor_shape().and_then(|dims| concrete_shape(dims, side)).expect("an input with a fixed channel count");
                let count: usize = shape.iter().product();
                let data: Vec<f32> = (0..count).map(|i| (((i * 7919 + n * 104729) % 1000) as f32) / 1000.0).collect();
                let value: DynValue = match ty.tensor_type() {
                    Some(TensorElementType::Float16) => Tensor::from_array((shape, data.into_iter().map(half::f16::from_f32).collect::<Vec<_>>())).unwrap().into_dyn(),
                    _ => Tensor::from_array((shape, data)).unwrap().into_dyn(),
                };
                (input.name().to_string(), value)
            }).collect()
        };
        let first_output = |session: &mut Session, inputs: &[(String, DynValue)]| -> Result<Vec<f32>, String> {
            let feed: Vec<(String, &DynValue)> = inputs.iter().map(|(n, v)| (n.clone(), v)).collect();
            let outputs = session.run(feed).map_err(|e| e.to_string().lines().next().unwrap_or_default().to_string())?;
            let out = &outputs[0];
            Ok(match out.try_extract_tensor::<f32>() {
                Ok((_, v)) => v.to_vec(),
                Err(_) => out.try_extract_tensor::<half::f16>().map_err(|e| e.to_string())?.1.iter().map(|x| x.to_f32()).collect(),
            })
        };
        let variants: [(&str, bool, bool, bool, GraphOptimizationLevel); 7] = [
            ("as the app", false, false, false, GraphOptimizationLevel::Level3),
            ("no DML fusion", false, true, false, GraphOptimizationLevel::Level3),
            ("no memory pattern", false, false, true, GraphOptimizationLevel::Level3),
            ("no DML fusion, no memory pattern", false, true, true, GraphOptimizationLevel::Level3),
            ("folding on", true, false, false, GraphOptimizationLevel::Level3),
            ("folding on, no DML fusion", true, true, false, GraphOptimizationLevel::Level3),
            ("Level1, folding on", true, false, false, GraphOptimizationLevel::Level1),
        ];
        for path in files {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let bytes = std::fs::read(&path).unwrap();
            let mut cpu = match build(&bytes, false, false, false, false, GraphOptimizationLevel::Level3) {
                Ok(s) => s,
                Err(e) => { eprintln!("{name}: CPU: {e}"); continue; }
            };
            let inputs = inputs_of(&cpu);
            let reference = match first_output(&mut cpu, &inputs) {
                Ok(r) => r,
                Err(e) => { eprintln!("{name}: CPU run: {e}"); continue; }
            };
            let range = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            for (label, folding, no_fusion, no_mem, level) in variants {
                let started = Instant::now();
                let outcome = build(&bytes, true, folding, no_fusion, no_mem, level).and_then(|mut s| {
                    let built = started.elapsed().as_millis();
                    first_output(&mut s, &inputs).map(|out| (built, out))
                });
                match outcome {
                    Ok((built, out)) => {
                        let diffs: Vec<f32> = out.iter().zip(&reference).map(|(a, b)| (a - b).abs() / range).collect();
                        let max = diffs.iter().fold(0f32, |m, &d| m.max(d));
                        let mean = diffs.iter().sum::<f32>() / diffs.len().max(1) as f32;
                        eprintln!("{name}: {label}: runs (built {built} ms), off the CPU by max {:.4}, mean {:.5} of its range", max, mean);
                    }
                    Err(e) => eprintln!("{name}: {label}: FAILS: {}", e.chars().take(160).collect::<String>()),
                }
            }
        }
    }

    /// The real case: Nomos8k's FP16 network on the CPU, whose saved graph
    /// never loads back (ORT's optimizers, run again, look for casts the
    /// first run folded away).
    #[test]
    #[ignore = "optimizes 4xNomos8kSCHAT-L on the CPU three times: about two minutes"]
    fn nomos8k_s_graph_is_no_longer_read_once_it_fails_to_load_back() {
        let id = prunr_models::ModelId::Nomos8kSchatL;
        if crate::ort_runtime::runtime_for_tests().is_err() || !prunr_models::is_available(id) {
            return;
        }
        crate::cache::clear_for_model(id);
        let graph = crate::cache::optimized_model_path(id, CPU_NAME, Weights::Fp32).unwrap();
        for _ in 0..3 {
            OrtEngine::for_model(id, 2, true).expect("Nomos8k loads on the CPU");
            assert!(graph.exists(), "its graph is written each time");
        }
        assert!(crate::cache::graph_not_read(&graph), "and no longer read");
    }

    /// A CUDA session holds the card memory its first run took: ORT's
    /// memory pattern, on by default, reserves one block of the run's
    /// peak from the second run on, on top of the pool the first run
    /// filled. BiRefNet-lite at 1024 went from 7.7 to 12.0 GB on a 12 GB
    /// RTX 3060 and spilled into system memory. `PRUNR_CUDA_ADDON` names
    /// the add-on's library where the runtime has no CUDA of its own.
    #[cfg(not(target_os = "macos"))]
    #[test]
    #[ignore = "BiRefNet-lite twice at 1024 on a CUDA card"]
    fn a_cuda_session_takes_no_more_card_memory_from_its_second_run() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        if let Some(addon) = std::env::var_os("PRUNR_CUDA_ADDON") {
            crate::ort_runtime::use_addon(crate::ort_runtime::Addon::Cuda, addon.into());
        }
        let engine = OrtEngine::for_model(prunr_models::ModelId::BiRefNetLite, 4, false).expect("BiRefNet-lite loads");
        if engine.active_provider() != CUDA_NAME {
            eprintln!("SKIP: it runs on {}", engine.active_provider());
            return;
        }
        let used = || crate::card_memory::cuda_card_memory().map(|c| c.total_mb - c.free_mb).expect("the card's memory");
        let photo = image::DynamicImage::new_rgb8(1024, 1024);
        let progress = crate::Progress::none();
        crate::pipeline::infer_only(&photo, &engine, &progress).expect("run 1");
        let after_one = used();
        crate::pipeline::infer_only(&photo, &engine, &progress).expect("run 2");
        let after_two = used();
        assert!(after_two <= after_one + 256, "run 2 took {} MB more than run 1's {after_one} MB", after_two.saturating_sub(after_one));
    }

    /// A cached CPU graph the session cannot load (one written elsewhere)
    /// is deleted and the model built and saved again: the CPU is the last
    /// provider, so a bad cache must not stop a model. One saved here that
    /// fails is no longer read, still written.
    #[test]
    fn a_cpu_session_builds_past_a_broken_cached_graph() {
        if let Err(why) = crate::ort_runtime::runtime_for_tests() {
            eprintln!("SKIP: {why}");
            return;
        }
        let _own = crate::cache::OwnCacheRoot::new("broken-graph");
        let id = prunr_models::ModelId::Silueta;
        assert!(crate::cache::cache_dir_for(id, CPU_NAME).is_some());
        let broken: Vec<PathBuf> = Weights::ALL
            .into_iter()
            .filter_map(|w| crate::cache::optimized_model_path(id, CPU_NAME, w))
            .collect();
        let break_them = || broken.iter().for_each(|path| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"not a graph").unwrap();
        });
        crate::cache::clear_for_model(id);
        break_them();
        let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("the model loads past its broken cache");
        let used = crate::cache::optimized_model_path(id, CPU_NAME, engine.pick.weights).unwrap();
        assert_ne!(std::fs::read(&used).ok().as_deref(), Some(&b"not a graph"[..]), "the broken graph is gone");
        assert!(used.exists() && !crate::cache::graph_not_read(&used), "and saved again, here");
        break_them();
        OrtEngine::new_cpu_only(ModelKind::Silueta, 1).expect("it loads past the second one");
        assert!(crate::cache::graph_not_read(&used), "the second failure stops reading it");
        let builder = OrtEngine::builder_with_base(1, GraphOptimizationLevel::Level3).unwrap();
        let (_, bytes, draft) = apply_ort_graph_cache(builder, &[], id, CPU_NAME, engine.pick.weights, None);
        assert!(matches!(bytes, Cow::Borrowed(_)) && draft.is_some(), "not read, still written");
        drop(draft);
        crate::cache::clear_for_model(id);
        assert!(!crate::cache::graph_not_read(&used), "clearing the cache starts afresh");
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
            let kept = per_shape || place_here(d.id, "MIGraphX", false, false).kept_off(false);
            assert_eq!(matches!(skip_reason(d.id, Weights::Fp32, EpKind::MiGraphX), Some(Skip::KeptOff(_))), kept, "{:?}", d.id);
            assert!(!compiles_per_shape(d.id, EpKind::Cuda));
        }
    }

    /// DirectML sessions of any model, built from several threads at once,
    /// commit one at a time.
    #[cfg(windows)]
    #[test]
    fn directml_sessions_build_one_at_a_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static BUILDING: AtomicUsize = AtomicUsize::new(0);
        static MOST: AtomicUsize = AtomicUsize::new(0);
        let walk = || {
            walk_ladder("", prunr_models::ModelId::U2net, Weights::Fp32, &[EpKind::DirectMl], |_| {
                MOST.fetch_max(BUILDING.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(20));
                BUILDING.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            })
            .is_ok()
        };
        let threads: Vec<_> = (0..4).map(|_| std::thread::spawn(walk)).collect();
        assert!(threads.into_iter().all(|t| t.join().expect("no panic")), "each built");
        assert_eq!(MOST.load(Ordering::SeqCst), 1);
    }

    /// An add-on started after the probe takes its place in the probe's
    /// order: CUDA before every other provider, OpenVINO after them, so a
    /// model no test measured tries a graphics card before OpenVINO, which
    /// runs on the processor unless the user lets it use Intel graphics.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_started_add_on_joins_the_providers_where_the_probe_puts_it() {
        assert_eq!(addon_position(&[], EpKind::Cuda), 0);
        assert_eq!(addon_position(&[EpKind::OpenVino], EpKind::Cuda), 0, "before OpenVINO");
        #[cfg(windows)]
        assert_eq!(addon_position(&[EpKind::DirectMl], EpKind::Cuda), 0, "before DirectML");
        #[cfg(target_os = "linux")]
        {
            assert_eq!(addon_position(&[EpKind::MiGraphX, EpKind::OpenVino], EpKind::Cuda), 0, "before MIGraphX");
            assert_eq!(addon_position(&[EpKind::Cuda, EpKind::MiGraphX], EpKind::OpenVino), 2, "OpenVINO last");
        }
    }

    /// A test of this computer tries again what Prunr's own tests keep on
    /// the CPU for a slower, different or no-gain result, from any
    /// process; a known failure stays off in both views.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_test_tries_again_what_the_registry_keeps_on_the_cpu() {
        use prunr_models::{Column, Place, Why};
        let mut seen = 0;
        for d in prunr_models::REGISTRY {
            let Place::Cpu(why) = place_here(d.id, OPENVINO_NAME, false, false) else { continue };
            if compiles_per_shape(d.id, EpKind::OpenVino) || openvino_on_gpu() {
                continue;
            }
            assert_eq!(prunr_models::registry_places(d.id).get(Column::OpenVino), Place::Cpu(why));
            assert!(kept_off(d.id, EpKind::OpenVino, false).is_some(), "{:?}", d.id);
            assert_eq!(kept_off(d.id, EpKind::OpenVino, true).is_some(), matches!(why, Why::Fails(_)), "{:?}: {why:?}", d.id);
            seen += 1;
        }
        assert!(seen > 0);
    }

    /// While a test measures one provider, a model tries that one alone; a
    /// provider the process lacks leaves none.
    #[test]
    fn a_test_pass_lets_a_model_try_one_provider() {
        let all = vec!["CUDAExecutionProvider", "DmlExecutionProvider"];
        let only = |name| {
            let mut eps = all.clone();
            keep_only(&mut eps, name, |ep| ep);
            eps
        };
        assert_eq!(only(None), all);
        assert_eq!(only(Some("DMLExecutionProvider")), ["DmlExecutionProvider"]);
        assert!(only(Some("NoSuchExecutionProvider")).is_empty(), "named but missing");
    }

    /// A model tries the providers measured faster first, quickest first;
    /// then the unmeasured, then those that keep it on the CPU, each group
    /// in the runtime's order.
    #[test]
    fn a_model_tries_the_fastest_provider_first() {
        use prunr_models::{Place, Speed, Why};
        let on = |ms| Place::There(Speed::on(ms, 600.0, "test"));
        let order = |places: [Place; 3]| fastest_first(&[0, 1, 2], |i: usize| places[i]);
        assert_eq!(order([on(394.0), on(36.0), on(120.0)]), [1, 2, 0], "quickest first");
        assert_eq!(order([Place::Unmeasured, on(36.0), Place::Unmeasured]), [1, 0, 2], "measured before unmeasured, in order");
        assert_eq!(order([Place::Cpu(Why::Fails("x")), Place::Unmeasured, on(500.0)]), [2, 1, 0], "the CPU's last");
        assert_eq!(order([Place::Unmeasured; 3]), [0, 1, 2], "nothing measured: the runtime's order");
    }

    /// A place a test measured here stands in for the registry's; one the
    /// registry knows fails is never measured over. A measured failure
    /// stays skipped while places are measured again, the rest are tried.
    #[test]
    fn a_place_measured_here_stands_in_for_the_registrys() {
        use crate::ep_compat::MEASURED_HERE;
        use prunr_models::{Column, Place, Places, Speed, Why};
        let registry = Places { openvino: Place::Cpu(Why::Fails("hangs")), cuda: Place::There(Speed::on(100.0, 200.0, "test")), ..Places::UNMEASURED };
        let slower = Column::ALL.map(|column| (column != Column::CoreMl).then_some(Place::Cpu(Why::Slower(MEASURED_HERE))));
        let here = with_measured(registry, slower);
        assert_eq!(here.cuda, Place::Cpu(Why::Slower(MEASURED_HERE)), "measured over the registry's place");
        assert_eq!(here.directml, Place::Cpu(Why::Slower(MEASURED_HERE)), "measured where the registry has none");
        assert_eq!(here.coreml, Place::Unmeasured, "not measured: the registry's");
        assert_eq!(here.openvino, Place::Cpu(Why::Fails("hangs")), "never over a known failure");
        assert!(here.cuda.kept_off(false) && !here.cuda.kept_off(true), "tried again when measuring");
        let failed = with_measured(Places::UNMEASURED, [Some(Place::Cpu(Why::Fails(MEASURED_HERE))); Column::ALL.len()]);
        assert!(failed.cuda.kept_off(true), "a measured failure stays off");
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

    /// A model's engines are sized as host-CPU work where its sessions run
    /// there: the CPU, a provider the table keeps it off, or OpenVINO's
    /// processor, which the Intel graphics leave some models on.
    #[test]
    fn a_model_left_on_the_processor_is_sized_as_host_cpu_work() {
        use prunr_models::ModelId;
        assert!(on_host_cpu_at(ModelId::BiRefNetLite, "OpenVINO", true), "kept off OpenVINO: ORT's CPU");
        assert!(on_host_cpu_at(ModelId::DepthAnythingV2Small, "OpenVINO", true), "slower on the graphics: the processor");
        assert!(!on_host_cpu_at(ModelId::RealEsrganX4Plus, "OpenVINO", true), "on the graphics");
        assert!(on_host_cpu_at(ModelId::RealEsrganX4Plus, "OpenVINO", false), "the switch is off");
        assert!(!on_host_cpu_at(ModelId::BiRefNetLite, "CUDA", false));
        assert!(on_host_cpu_at(ModelId::Nomos8kSchatL, "CUDA", false), "kept off CUDA");
        assert!(on_host_cpu_at(ModelId::BiRefNetLite, CPU_NAME, true));
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

    /// The OpenVINO add-on that cannot start says why in the same words:
    /// one of its own files gone, or no Intel device it can use.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn openvino_that_cannot_start_says_why() {
        let registration = "Failed to load library /x/libonnxruntime_providers_openvino_plugin.so with error: libtbb.so.12: cannot open shared object file: No such file or directory";
        assert_eq!(
            GpuStartFailure::of(GpuStack::OpenVino, registration).text(),
            "OpenVINO could not start: libtbb.so.12 is missing. It needs its own files, which installing it again brings back.",
        );
        assert_eq!(GpuStartFailure::no_device(GpuStack::OpenVino).text(), "OpenVINO could not start: it found no Intel processor or Intel graphics it can use.");
    }

    /// Each provider's start failure is kept on its own, the first of its
    /// stack: with CUDA and OpenVINO beside each other, both say theirs.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn each_provider_keeps_its_own_start_failure() {
        let failures = StartFailures::new();
        let failure = |stack, why| GpuStartFailure { stack, why };
        assert!(!failures.started(Err(failure(GpuStack::Cuda, StartFailure::NoDriver))));
        assert!(!failures.started(Err(failure(GpuStack::OpenVino, StartFailure::NoDevice))));
        assert!(!failures.started(Err(failure(GpuStack::Cuda, StartFailure::Other))));
        assert!(failures.started(Ok(())), "only a start that worked lists its provider");
        let kept = failures.each();
        assert_eq!(kept[GpuStack::Cuda.index()].map(|f| &f.why), Some(&StartFailure::NoDriver), "the first of its stack");
        assert_eq!(kept[GpuStack::OpenVino.index()].map(|f| &f.why), Some(&StartFailure::NoDevice));
        assert_eq!(kept[GpuStack::Rocm.index()], None);
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
        let initial = crate::ep_compat::is_known_failure(ep, id, Weights::Fp32, crate::engine::asks_intel_gpu(ep, id));

        let walk = walk_ladder("", id, Weights::Fp32, &[ep], |_| {
            Err::<(), _>(Miss::Failed { error: "test cache failure".into(), suspect_cache: Some(cache_file.clone()) })
        });

        assert_eq!(walk.map_err(|w| w.failed), Err(vec!["test cache failure".to_string()]));
        assert!(!cache_file.exists(), "cache file must be deleted");
        assert_eq!(
            crate::ep_compat::is_known_failure(ep, id, Weights::Fp32, crate::engine::asks_intel_gpu(ep, id)), initial,
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
        assert!(crate::ep_compat::is_known_failure(ep, model, Weights::Fp16, crate::engine::asks_intel_gpu(ep, model)), "the FP16 failure is remembered");
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32, crate::engine::asks_intel_gpu(ep, model)), "and only that");
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
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32, crate::engine::asks_intel_gpu(ep, model)));
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
        assert!(!crate::ep_compat::is_known_failure(ep, model, Weights::Fp32, crate::engine::asks_intel_gpu(ep, model)), "the machine's lack, not the model's");
    }

    /// A provider is passed over without a try when the registry keeps
    /// the model off it or it failed before; the walk says so when it
    /// tried none.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_walk_that_tries_no_provider_says_why() {
        use prunr_models::ModelId;
        let model = ModelId::Silueta;
        let prunr_models::Place::Cpu(why) = place_here(model, "OpenVINO", false, false) else { panic!("Silueta stays off OpenVINO") };
        let line = format!("no GPU provider to try: OpenVINO is kept off: {}", why.text());
        let declared = walk_ladder("", model, Weights::Fp32, &[EpKind::OpenVino], |_| -> Result<(), Miss> { panic!("tried a declared pair") });
        assert_eq!(declared.map_err(|w| w.reason()), Err(line.clone()));
        let none = walk_ladder("", model, Weights::Fp32, &[], |_| -> Result<(), Miss> { unreachable!() });
        assert_eq!(none.map_err(|w| w.reason()), Err("No GPU EPs available on this platform".to_string()));
        assert_eq!(gpu_skip(model, EpKind::OpenVino, remeasuring()), Some(Skip::KeptOff(why)));
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
    /// naming why its own weights were not tried there, and reads none on
    /// the way. Its FP16 copy is not named: no provider gains from it.
    #[test]
    fn a_gpu_session_without_a_gpu_provider_lands_on_the_cpu_and_says_why() {
        use prunr_models::ModelId;
        if crate::ort_runtime::ensure_initialized().is_err() {
            return;
        }
        let engine = OrtEngine::build_with(ModelKind::Silueta, 1, &[], false, GraphOptimizationLevel::Level3)
            .expect("Silueta is bundled");
        let none = "Inference error: No GPU EPs available on this platform";
        assert_eq!(engine.pick().provider, "CPU");
        assert_eq!(engine.pick().failed, vec![format!("FP32 weights: {none}")]);
        assert_eq!(engine.pick().weights, lighter_weights(ModelId::Silueta, true, &[]).unwrap_or(Weights::Fp32));
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
            &gpu_eps_for(prunr_models::ModelId::Silueta, false),
            false,
            GraphOptimizationLevel::Level2,
        )
        .expect("Level2 must succeed for Silueta");
        assert!(!engine.active_provider().is_empty());
    }
}
