//! GUI-side runtime install glue over `prunr-runtime-install`: the
//! `RuntimeId` enum and channel-based progress events.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};

use prunr_runtime_install as ri;

/// User agent for every HTTP request the app makes.
pub const USER_AGENT: &str = concat!("prunr/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone)]
pub enum InstallEvent {
    /// Pre-download work: the PyPI metadata query.
    Preparing,
    Downloading { bytes_so_far: u64, total_bytes: u64 },
    Extracting,
    /// `started`: an add-on starts in this process as it installs, or says
    /// why it could not; a runtime loads from the next start (`None`).
    Done { install_dir: PathBuf, started: Option<Result<(), String>> },
    Failed { error: String },
}

impl InstallEvent {
    pub fn status_text(&self) -> String {
        match self {
            InstallEvent::Preparing => "Preparing…".to_string(),
            InstallEvent::Downloading { bytes_so_far, total_bytes } if *total_bytes > 0 => {
                let pct = bytes_so_far * 100 / total_bytes;
                format!(
                    "Downloading {pct}% ({:.0}/{:.0} MB)",
                    *bytes_so_far as f64 / 1024.0 / 1024.0,
                    *total_bytes as f64 / 1024.0 / 1024.0,
                )
            }
            InstallEvent::Downloading { .. } => "Downloading…".to_string(),
            InstallEvent::Extracting => "Extracting…".to_string(),
            InstallEvent::Done { .. } => "Installed".to_string(),
            InstallEvent::Failed { error } => format!("Failed: {error}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeId {
    OpenVino,
    Cuda,
    MiGraphX,
}

impl RuntimeId {
    /// The runtimes Settings › Hardware offers to download. DirectML is
    /// not one: Windows ships it built in. OpenVINO is not offered on
    /// Windows: its `onnxruntime-openvino` wheel there ships the provider
    /// DLL without `openvino.dll` (Python loads that from the separate
    /// `openvino` package), so the provider never registers and every
    /// session would run on the CPU, where DirectML runs the same iGPU.
    /// CUDA is offered on Windows too, as an add-on beside the built-in
    /// DirectML (`is_addon`): DirectML fails models on an NVIDIA card that
    /// CUDA runs, and is faster for others. MIGraphX has a Linux wheel only.
    pub const OFFERED: &'static [RuntimeId] = if cfg!(all(windows, target_arch = "x86_64")) {
        &[Self::Cuda]
    } else if cfg!(windows) {
        &[]
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        &[Self::OpenVino, Self::Cuda, Self::MiGraphX]
    } else {
        &[Self::OpenVino]
    };

    /// Whether it loads into the runtime the app ships, as a provider
    /// add-on, rather than replacing it: CUDA on Windows, where the
    /// built-in runtime keeps DirectML beside it and each model takes the
    /// faster, and on Linux.
    pub fn is_addon(self) -> bool {
        self.addon().is_some()
    }

    fn addon(self) -> Option<&'static ri::AddOn> {
        match self {
            Self::Cuda if cfg!(all(windows, target_arch = "x86_64")) => Some(&ri::CUDA_ADDON_WINDOWS),
            Self::Cuda if cfg!(all(target_os = "linux", target_arch = "x86_64")) => Some(&ri::CUDA_ADDON_LINUX),
            _ => None,
        }
    }

    /// An add-on joining a built-in accelerator: CUDA beside DirectML.
    fn beside_directml(self) -> bool {
        self.is_addon() && prunr_core::ort_runtime::BUILT_IN_HAS_GPU
    }

    fn pypi_package(self) -> &'static str {
        match self {
            Self::OpenVino => "onnxruntime-openvino",
            Self::Cuda => "onnxruntime-gpu",
            Self::MiGraphX => "onnxruntime-migraphx",
        }
    }

    /// The ORT release this runtime installs. The MIGraphX wheel skipped
    /// 1.24.1; any 1.24 release serves the same API.
    fn ort_version(self) -> &'static str {
        match self {
            Self::OpenVino | Self::Cuda => ri::PINNED_ORT_VERSION,
            Self::MiGraphX => "1.24.2",
        }
    }

    pub fn install_subdir(self) -> String {
        match self.addon() {
            Some(addon) => ri::install_subdir(&format!("{}-addon", self.short_name()), addon.version),
            None => ri::install_subdir(self.short_name(), self.ort_version()),
        }
    }

    /// The offered runtime installed under `subdir` of the runtime store.
    pub fn from_subdir(subdir: &str) -> Option<Self> {
        Self::OFFERED.iter().copied().find(|rt| rt.install_subdir() == subdir)
    }

    pub fn install_dir(self) -> Option<PathBuf> {
        prunr_models::runtime_store_dir().map(|d| d.join(self.install_subdir()))
    }

    /// `install_dir`, its name checked to stay inside the runtime store
    /// (no path separators), so removing or replacing it cannot reach
    /// anything else.
    fn checked_install_dir(self) -> Result<PathBuf, String> {
        ri::validate_subdir(&self.install_subdir()).map_err(|e| format!("install_subdir invariant broken: {e}"))?;
        self.install_dir().ok_or_else(|| "could not resolve user data dir".to_string())
    }

    pub fn is_installed(self) -> bool {
        self.library().is_some_and(|p| p.is_file()) && !self.removal_pending()
    }

    /// An add-on uninstalled while loaded leaves this mark beside its
    /// install, and goes at the next start (`finish_uninstalls`).
    fn removal_mark(self) -> Option<PathBuf> {
        self.install_dir().and_then(|dir| beside(&dir, "remove").ok())
    }

    fn removal_pending(self) -> bool {
        self.removal_mark().is_some_and(|mark| mark.is_file())
    }

    /// The library its install holds: the runtime's, or the add-on's.
    fn library(self) -> Option<PathBuf> {
        let name = self.addon().map_or(prunr_core::ort_runtime::DYLIB_NAME, |addon| addon.library);
        self.install_dir().map(|d| d.join(name))
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::OpenVino => "openvino",
            Self::Cuda => "cuda",
            Self::MiGraphX => "migraphx",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::OpenVino => "OpenVINO Runtime",
            Self::Cuda if self.is_addon() => "CUDA",
            Self::Cuda => "CUDA Runtime",
            Self::MiGraphX => "MIGraphX Runtime",
        }
    }

    /// What installing it downloads, as a button says it: the add-on alone
    /// where NVIDIA's libraries are already here (`nvidia_here`).
    pub fn download_label(self, nvidia_here: bool) -> std::borrow::Cow<'static, str> {
        use std::borrow::Cow;
        match self {
            Self::OpenVino => Cow::Borrowed("80 MB"),
            Self::Cuda => match self.addon() {
                Some(addon) if nvidia_here => Cow::Owned(crate::gui::views::format_byte_size(addon.size_bytes)),
                _ => Cow::Borrowed("about 2 GB"),
            },
            Self::MiGraphX => Cow::Borrowed("20 MB"),
        }
    }

    /// `download_label`, as a sentence says it: what the download brings
    /// beside the runtime.
    pub fn download_size(self, nvidia_here: bool) -> std::borrow::Cow<'static, str> {
        let label = self.download_label(nvidia_here);
        match self {
            Self::Cuda if !(nvidia_here && self.is_addon()) => format!("{label} with NVIDIA's CUDA and cuDNN libraries").into(),
            _ => label,
        }
    }

    /// Settings › Hardware's line while this runtime is not installed: its
    /// download, and why to get it when the hardware suits it.
    pub fn not_installed(self, recommended: bool, nvidia_here: bool) -> String {
        let why = match (self, recommended) {
            (Self::OpenVino, false) => "",
            (Self::OpenVino, true) => " Recommended for Intel hardware: the models run faster, and you can try running them on the Intel graphics chip.",
            (Self::Cuda, false) if self.beside_directml() => " Runs models on an NVIDIA graphics card beside DirectML; Prunr uses the faster for each model.",
            (Self::Cuda, true) if self.beside_directml() => " Recommended for your NVIDIA graphics card: it runs models DirectML cannot, and Prunr uses the faster for each model.",
            (Self::Cuda, false) => " Runs the models on an NVIDIA graphics card.",
            (Self::Cuda, true) => " Recommended for your NVIDIA graphics card: runs the models on it.",
            (Self::MiGraphX, false) => " Runs the models on an AMD graphics card; needs AMD's ROCm 7 with MIGraphX.",
            (Self::MiGraphX, true) => " Recommended for your AMD graphics card: runs the models on it. Needs AMD's ROCm 7 with MIGraphX.",
        };
        format!("Not installed ({}).{why}", self.download_size(nvidia_here))
    }

    /// Settings › Hardware's line for an add-on installed while an older
    /// runtime from the Runtime Store is loaded.
    pub fn waiting_line(self) -> &'static str {
        "Installed. It starts once Prunr runs on its own runtime (Built in, under Runtime) after a restart."
    }

    /// Settings › Hardware's line once it is installed.
    pub fn installed_line(self) -> &'static str {
        match self {
            Self::Cuda if self.beside_directml() => "Installed. Runs beside DirectML on your NVIDIA graphics card.",
            _ => "Installed.",
        }
    }

    pub fn uninstall_label(self) -> &'static str {
        match self {
            Self::OpenVino => "Uninstall OpenVINO Runtime",
            Self::Cuda if self.is_addon() => "Uninstall CUDA",
            Self::Cuda => "Uninstall CUDA Runtime",
            Self::MiGraphX => "Uninstall MIGraphX Runtime",
        }
    }

    /// The first-launch prompt's body.
    pub fn first_launch_prompt_body(self, nvidia_here: bool) -> String {
        let size = self.download_size(nvidia_here);
        match self {
            Self::OpenVino => format!("We detected Intel hardware that can use OpenVINO Runtime. \
                Installing it ({size}) makes the models run faster, and lets you \
                try running them on the Intel graphics chip (Settings › Hardware)."),
            Self::Cuda if self.beside_directml() => format!("We detected an NVIDIA graphics card. Installing CUDA \
                ({size}) runs more models on it than DirectML can, and each model \
                on the faster of the two."),
            Self::Cuda => format!("We detected an NVIDIA graphics card. Installing {} \
                ({size}) runs the models on it instead of the processor.", self.display_name()),
            Self::MiGraphX => format!("We detected an AMD graphics card. Installing MIGraphX Runtime \
                ({size}) runs the models on it instead of the processor. It needs \
                AMD's ROCm 7 with MIGraphX on this computer."),
        }
    }

    /// The first-launch prompt's install-button label.
    pub fn install_button_label(self) -> &'static str {
        match self {
            Self::OpenVino => "Install OpenVINO Runtime",
            Self::Cuda if self.is_addon() => "Install CUDA",
            Self::Cuda => "Install CUDA Runtime",
            Self::MiGraphX => "Install MIGraphX Runtime",
        }
    }

    /// Stable key for persisted settings — decoupled from `Debug` output
    /// so we can rename the variant without breaking user state.
    pub fn settings_key(self) -> &'static str {
        match self {
            Self::OpenVino => "OpenVino",
            Self::Cuda => "Cuda",
            Self::MiGraphX => "MiGraphX",
        }
    }
}

/// NVIDIA's CUDA 12 and cuDNN 9 libraries the CUDA runtime links, as the
/// PyPI wheels that carry them: (package, pinned version). ORT asks for
/// any 12.x / 9.x; these are the newest at pinning.
pub const NVIDIA_WHEELS: &[(&str, &str)] = &[
    ("nvidia-cuda-runtime-cu12", "12.9.79"),
    ("nvidia-cublas-cu12", "12.9.2.10"),
    ("nvidia-cudnn-cu12", "9.27.0.42"),
    ("nvidia-cufft-cu12", "11.4.1.4"),
    ("nvidia-curand-cu12", "10.3.10.19"),
    ("nvidia-cuda-nvrtc-cu12", "12.9.86"),
];

/// The NVIDIA libraries' download, for the button that offers them
/// again when they are missing.
pub const NVIDIA_DOWNLOAD_LABEL: &str = "1.7 GB";

/// The settings key the NVIDIA libraries' license acceptance is kept under.
pub const NVIDIA_LICENSE_KEY: &str = "NvidiaCudaLibraries";

/// The licenses the NVIDIA libraries come under: the CUDA EULA and the
/// cuDNN license.
pub const NVIDIA_LICENSE_URLS: [&str; 2] = [
    "https://docs.nvidia.com/cuda/eula/index.html",
    "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html",
];

/// Where the NVIDIA libraries go: inside the CUDA install (the runtime or
/// the add-on), so uninstalling it removes them too.
pub fn nvidia_libraries_dir() -> Option<PathBuf> {
    RuntimeId::Cuda.install_dir().map(|d| d.join(prunr_core::ort_runtime::NVIDIA_LIBS_DIR))
}

pub fn nvidia_libraries_installed() -> bool {
    nvidia_libraries_dir().is_some_and(|d| d.join(prunr_core::ort_runtime::CUDNN_LIBRARY).is_file())
}

/// What one install fetches: a runtime, or the NVIDIA libraries the CUDA
/// runtime needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallTarget {
    Runtime(RuntimeId),
    NvidiaLibraries,
}

impl InstallTarget {
    /// It downloads NVIDIA's libraries, under their license.
    pub fn needs_nvidia_license(self) -> bool {
        matches!(self, Self::Runtime(RuntimeId::Cuda) | Self::NvidiaLibraries)
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Runtime(rt) => rt.display_name(),
            Self::NvidiaLibraries => "NVIDIA libraries",
        }
    }
}

/// Which runtime every process loads: the one built into the app (the
/// DirectML build on Windows) or a Runtime Store install.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeChoice {
    BuiltIn,
    Store(RuntimeId),
}

impl RuntimeChoice {
    /// Every choice Settings can show, in its order.
    pub const ALL: [RuntimeChoice; 4] = [
        Self::BuiltIn,
        Self::Store(RuntimeId::OpenVino),
        Self::Store(RuntimeId::Cuda),
        Self::Store(RuntimeId::MiGraphX),
    ];

    fn preference(self) -> String {
        match self {
            Self::BuiltIn => prunr_core::ort_runtime::PREFER_BUILT_IN.to_string(),
            Self::Store(rt) => rt.install_subdir(),
        }
    }

    /// The runtime this process loaded, once it has.
    pub fn loaded() -> Option<Self> {
        use prunr_core::ort_runtime::{loaded_dylib, loaded_source};
        Self::of_library(loaded_dylib()?, loaded_source()?)
    }

    /// The runtime every process loads from its next start: the
    /// resolver's own pick, so a preferred install that has gone is not
    /// shown as the one in use.
    pub fn next() -> Option<Self> {
        let (library, source) = prunr_core::ort_runtime::resolve_dylib_path().ok()?;
        Self::of_library(&library, source)
    }

    /// The choice a runtime library is; `None` for one `ORT_DYLIB_PATH`
    /// names, which no choice in Settings overrides.
    fn of_library(library: &Path, source: prunr_core::ort_runtime::DylibSource) -> Option<Self> {
        use prunr_core::ort_runtime::DylibSource;
        match source {
            DylibSource::Bundled => Some(Self::BuiltIn),
            DylibSource::RuntimeStore => RuntimeId::from_subdir(library.parent()?.file_name()?.to_str()?).map(Self::Store),
            DylibSource::EnvVar => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BuiltIn => "Built in",
            Self::Store(RuntimeId::OpenVino) => "OpenVINO",
            Self::Store(RuntimeId::Cuda) => "CUDA",
            Self::Store(RuntimeId::MiGraphX) => "MIGraphX",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::BuiltIn if prunr_core::ort_runtime::BUILT_IN_HAS_GPU => "DirectML: any DirectX 12 graphics card, else the processor",
            Self::BuiltIn => "Prunr's own: the processor, and CUDA when installed",
            Self::Store(RuntimeId::OpenVino) => "Tuned for Intel processors and graphics",
            Self::Store(RuntimeId::Cuda) => "NVIDIA graphics cards",
            Self::Store(RuntimeId::MiGraphX) => "AMD graphics cards",
        }
    }
}

/// Make `choice` the runtime every process loads from its next start.
pub fn prefer(choice: RuntimeChoice) -> Result<(), String> {
    if let RuntimeChoice::Store(rt) = choice {
        rt.checked_install_dir()?;
    }
    prunr_core::ort_runtime::set_preferred_runtime(&choice.preference())
}

/// Started install handle: progress receiver + cancel flag. Caller
/// flips `cancel` to abort an in-flight install — the worker thread
/// notices on the next chunk read or backoff tick (≤50 ms latency).
pub struct InstallHandle {
    pub events: mpsc::Receiver<InstallEvent>,
    pub cancel: Arc<AtomicBool>,
}

/// Final event is either `Done` or `Failed`; channel closes after.
pub fn start_install(target: InstallTarget) -> InstallHandle {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_thread = Arc::clone(&cancel);
    std::thread::spawn(move || {
        let installed = match target {
            InstallTarget::Runtime(rt) => run_install(rt, &tx, cancel_for_thread),
            InstallTarget::NvidiaLibraries => run_nvidia_install(&tx, cancel_for_thread),
        };
        if let Err(e) = installed {
            let _ = tx.send(InstallEvent::Failed { error: e });
        }
    });
    InstallHandle { events: rx, cancel }
}

/// When an uninstall takes effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uninstalled {
    Now,
    /// The add-on is loaded in this process, and Windows deletes no
    /// library in use: it goes at the next start.
    AtNextStart,
}

/// Remove an installed runtime or add-on. Refuses when this is the only
/// ORT source (uninstalling would brick startup) and when the subdir name
/// could escape `<data>/runtimes/`.
pub fn uninstall(runtime: RuntimeId) -> Result<Uninstalled, String> {
    let dir = runtime.checked_install_dir()?;
    if !dir.exists() {
        return Ok(Uninstalled::Now); // already absent — nothing to do
    }
    // Installing the add-on made Prunr's own runtime the choice; where
    // that runtime has no accelerator of its own, one chosen from the
    // store before comes back.
    if runtime.is_addon() && !prunr_core::ort_runtime::BUILT_IN_HAS_GPU {
        prunr_core::ort_runtime::forget_preferred_runtime(prunr_core::ort_runtime::PREFER_BUILT_IN);
    }
    if runtime.is_addon() && prunr_core::ort_runtime::cuda_addon_registered() {
        let mark = runtime.removal_mark().ok_or("could not resolve user data dir")?;
        std::fs::write(&mark, b"").map_err(|e| format!("mark {} for removal: {e}", dir.display()))?;
        return Ok(Uninstalled::AtNextStart);
    }
    if !runtime.is_addon() && !prunr_core::ort_runtime::has_fallback_excluding(&dir) {
        return Err(
            "this is the only ONNX Runtime installed — uninstalling would leave \
             the app unable to start. Install another runtime first, or set \
             ORT_DYLIB_PATH to a system ORT before removing this one."
                .to_string()
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("remove {}: {e}", dir.display()))?;
    prunr_core::ort_runtime::forget_preferred_runtime(&runtime.install_subdir());
    Ok(Uninstalled::Now)
}

/// Remove the add-ons uninstalled while they were loaded. Off the main
/// thread: an add-on is about 2 GB, and nothing loads it meanwhile, as
/// its mark stays until it is gone (`is_installed`).
pub fn finish_uninstalls() {
    let Some(store) = prunr_models::runtime_store_dir() else { return };
    let marked: Vec<String> = RuntimeId::OFFERED.iter().filter(|rt| rt.is_addon() && rt.removal_pending()).map(|rt| rt.install_subdir()).collect();
    if !marked.is_empty() {
        std::thread::spawn(move || remove_marked(&store, marked));
    }
}

/// Remove each of `subdirs` in `store` that has its removal mark beside
/// it, then the mark; nothing else in `store`.
fn remove_marked(store: &Path, subdirs: impl IntoIterator<Item = String>) {
    for subdir in subdirs.into_iter().filter(|subdir| ri::validate_subdir(subdir).is_ok()) {
        let dir = store.join(&subdir);
        let Ok(mark) = beside(&dir, "remove") else { continue };
        if !mark.is_file() {
            continue;
        }
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {
                let _ = std::fs::remove_file(&mark);
                tracing::info!(dir = %dir.display(), "removed an add-on uninstalled while it was loaded");
            }
            Err(e) => tracing::warn!(dir = %dir.display(), %e, "could not remove an uninstalled add-on; trying at the next start"),
        }
    }
}

/// The CUDA runtime or add-on brings NVIDIA's libraries in the same
/// install, never the system's; a reinstall keeps the ones already in
/// place.
fn run_install(
    rt: RuntimeId,
    tx: &mpsc::Sender<InstallEvent>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    let _ = tx.send(InstallEvent::Preparing);
    // Uninstalled while loaded and still here: installing keeps it.
    if let Some(mark) = rt.removal_mark().filter(|mark| mark.is_file()) {
        std::fs::remove_file(&mark).map_err(|e| format!("keep {}: {e}", rt.display_name()))?;
        let install_dir = rt.checked_install_dir()?;
        let _ = tx.send(InstallEvent::Done { install_dir, started: Some(start_addon()) });
        return Ok(());
    }
    let wheel = match rt.addon() {
        Some(addon) => addon.download(),
        None => query_pypi(rt, &cancel)?,
    };
    let nvidia = if rt == RuntimeId::Cuda && !nvidia_libraries_installed() { nvidia_wheels(&cancel)? } else { Vec::new() };
    let total = wheel.size_bytes + nvidia.iter().map(|w| w.size_bytes).sum::<u64>();

    let _ = tx.send(InstallEvent::Downloading { bytes_so_far: 0, total_bytes: total });
    let mut on_progress = |so_far: u64, _: u64| {
        let _ = tx.send(InstallEvent::Downloading { bytes_so_far: so_far, total_bytes: total });
    };
    let mut hooks = ri::DownloadHooks { progress: &mut on_progress, cancel: Arc::clone(&cancel) };
    let target = rt.checked_install_dir()?;
    let spool = spool_for(&target)?;
    // Already SHA-verified inside the retry loop.
    let installed = ri::download_wheel(&wheel, &mut hooks, &spool).and_then(|file| {
        let _ = tx.send(InstallEvent::Extracting);
        extract_into_place(file, &target, rt.addon())
    });
    let _ = std::fs::remove_file(&spool);
    let install_dir = installed?;
    if !nvidia.is_empty() {
        let target = rt.checked_install_dir()?.join(prunr_core::ort_runtime::NVIDIA_LIBS_DIR);
        fetch_nvidia_libraries(&nvidia, &target, tx, &cancel, (wheel.size_bytes, total))?;
    }
    // The runtime just installed is the one the user wants next start;
    // an add-on starts now, in the runtime already loaded, when that is
    // as new as the add-on (Prunr's own), else at the next start, in it.
    let choice = if rt.is_addon() { RuntimeChoice::BuiltIn } else { RuntimeChoice::Store(rt) };
    if let Err(e) = prefer(choice) {
        tracing::warn!(runtime = rt.settings_key(), %e, "could not mark the runtime it needs as preferred");
    }
    let started = rt.is_addon().then(|| {
        use_installed_addons();
        prunr_core::ort_runtime::hosts_addons().then(start_addon)
    }).flatten();

    let _ = tx.send(InstallEvent::Done { install_dir, started });
    Ok(())
}

/// Tell this process where the installed add-ons are, before it loads any
/// session: each one registers when the providers are probed.
pub fn use_installed_addons() {
    let cuda = RuntimeId::Cuda;
    if let Some(library) = cuda.library().filter(|_| cuda.is_addon() && cuda.is_installed()) {
        prunr_core::ort_runtime::use_cuda_addon(library);
    }
}

/// Start the installed add-on in this process.
fn start_addon() -> Result<(), String> {
    #[cfg(not(target_os = "macos"))]
    {
        prunr_core::engine::start_cuda_addon()
    }
    #[cfg(target_os = "macos")]
    {
        Err("no add-on runs on macOS".to_string())
    }
}

/// Download each NVIDIA wheel in turn (one in memory at a time, the
/// largest 766 MB), extract its libraries into a staging folder, and move
/// that into place once all are there.
fn run_nvidia_install(tx: &mpsc::Sender<InstallEvent>, cancel: Arc<AtomicBool>) -> Result<(), String> {
    let _ = tx.send(InstallEvent::Preparing);
    let target = nvidia_libraries_dir().ok_or("could not resolve user data dir")?;
    if !RuntimeId::Cuda.is_installed() {
        return Err("install CUDA Runtime first".to_string());
    }
    let wheels = nvidia_wheels(&cancel)?;
    let total = wheels.iter().map(|w| w.size_bytes).sum();
    fetch_nvidia_libraries(&wheels, &target, tx, &cancel, (0, total))?;
    let _ = tx.send(InstallEvent::Done { install_dir: target, started: None });
    Ok(())
}

fn nvidia_wheels(cancel: &Arc<AtomicBool>) -> Result<Vec<ri::WheelInfo>, String> {
    NVIDIA_WHEELS.iter().map(|(package, version)| query_pypi_package(package, version, cancel)).collect()
}

/// `wheels`' libraries into `target`, through a staging folder moved into
/// place once all are there; `(done, total)` places their bytes in the
/// install's progress.
fn fetch_nvidia_libraries(
    wheels: &[ri::WheelInfo],
    target: &Path,
    tx: &mpsc::Sender<InstallEvent>,
    cancel: &Arc<AtomicBool>,
    (done_before, total): (u64, u64),
) -> Result<(), String> {
    let staging = fresh_staging(target)?;
    let spool = spool_for(target)?;
    let fetched = (|| {
        let mut done = done_before;
        for wheel in wheels {
            let mut on_progress = |so_far: u64, _: u64| {
                let _ = tx.send(InstallEvent::Downloading { bytes_so_far: done + so_far, total_bytes: total });
            };
            let mut hooks = ri::DownloadHooks { progress: &mut on_progress, cancel: Arc::clone(cancel) };
            let file = ri::download_wheel(wheel, &mut hooks, &spool)?;
            let _ = tx.send(InstallEvent::Extracting);
            ri::extract_nvidia_wheel(file, &staging)?;
            done += wheel.size_bytes;
        }
        Ok::<(), String>(())
    })();
    let _ = std::fs::remove_file(&spool);
    if let Err(e) = fetched {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    move_into_place(&staging, target, &[])
}

fn query_pypi(rt: RuntimeId, cancel: &Arc<AtomicBool>) -> Result<ri::WheelInfo, String> {
    query_pypi_package(rt.pypi_package(), rt.ort_version(), cancel)
}

/// The files PyPI lists at `url`, asked again after a dropped connection
/// or a server error (5xx), as downloads are.
fn pypi_files(url: &str, cancel: &Arc<AtomicBool>) -> Result<Vec<serde_json::Value>, String> {
    ri::retry_with_backoff(cancel, ri::DOWNLOAD_ATTEMPTS, ri::DOWNLOAD_BACKOFF_BASE_MS, |_attempt| {
        let client = ri::http_client(USER_AGENT)
            .map_err(|e| ri::DlError::fatal(format!("HTTP client: {}", ri::with_causes(&e))))?;
        let resp = client.get(url).send().map_err(|e| ri::DlError::request(&e))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(ri::DlError::for_status(status, format!("PyPI answered {status}")));
        }
        let mut metadata: serde_json::Value = resp.json()
            .map_err(|e| ri::DlError::transient(format!("PyPI JSON parse: {}", ri::with_causes(&e))))?;
        match metadata["urls"].take() {
            serde_json::Value::Array(urls) => Ok(urls),
            _ => Err(ri::DlError::fatal("PyPI metadata missing `urls`")),
        }
    })
    .map_err(|e| e.message)
}

/// `package` at `version`: its wheel for this host.
fn query_pypi_package(package: &str, version: &str, cancel: &Arc<AtomicBool>) -> Result<ri::WheelInfo, String> {
    ri::pick_wheel_for_host(&pypi_files(&format!("https://pypi.org/pypi/{package}/{version}/json"), cancel)?)
}

/// Extract into a hidden folder beside the install and move it into place
/// once whole: an install cut short (a full disk, the app closed) leaves
/// the runtime that was there, or none, never half of one to load. An
/// add-on's archive gives up its library; a wheel, its runtime's.
fn extract_into_place(archive: impl std::io::Read + std::io::Seek, target_dir: &Path, addon: Option<&ri::AddOn>) -> Result<PathBuf, String> {
    let staging = fresh_staging(target_dir)?;
    let extracted = match addon {
        Some(addon) => ri::extract_addon(archive, &staging, addon),
        None => ri::extract_wheel(archive, &staging),
    };
    let library = extracted.inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&staging);
    })?;
    // The NVIDIA libraries live inside the CUDA install and come from
    // their own download; a reinstall of the runtime keeps them.
    move_into_place(&staging, target_dir, &[prunr_core::ort_runtime::NVIDIA_LIBS_DIR])?;
    Ok(target_dir.join(library.file_name().ok_or("extracted library has no name")?))
}

/// An empty hidden folder beside `target`, for an install to fill before
/// it moves into place.
fn fresh_staging(target: &Path) -> Result<PathBuf, String> {
    let staging = beside(target, "staging")?;
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(|e| format!("clean staging dir: {e}"))?;
    }
    std::fs::create_dir_all(&staging).map_err(|e| format!("create staging dir: {e}"))?;
    Ok(staging)
}

/// The file a wheel for `target` downloads to, beside it; the folder
/// they share is made first, as the first install meets none.
fn spool_for(target: &Path) -> Result<PathBuf, String> {
    let spool = beside(target, "wheel")?;
    if let Some(dir) = spool.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    Ok(spool)
}

/// `.{name}.{what}` beside `target`, named after it.
fn beside(target: &Path, what: &str) -> Result<PathBuf, String> {
    let name = target.file_name().ok_or("install dir has no name")?.to_string_lossy();
    Ok(target.with_file_name(format!(".{name}.{what}")))
}

/// Replace `target` with the whole `staging` folder, carrying the `kept`
/// subfolders of the old `target` over.
fn move_into_place(staging: &Path, target: &Path, kept: &[&str]) -> Result<(), String> {
    for name in kept {
        let old = target.join(name);
        if old.exists() {
            std::fs::rename(&old, staging.join(name)).map_err(|e| format!("keep {name}: {e}"))?;
        }
    }
    if target.exists() {
        if let Err(e) = std::fs::remove_dir_all(target) {
            for name in kept {
                let _ = std::fs::rename(staging.join(name), target.join(name));
            }
            return Err(format!("clean install dir: {e}"));
        }
    }
    std::fs::rename(staging, target).map_err(|e| format!("move install into place: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Answer each connection to the returned URL with the next of
    /// `responses`; the handle tells how many were asked for.
    fn serve(responses: Vec<&'static str>) -> (String, std::thread::JoinHandle<usize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/pypi/onnxruntime/json", listener.local_addr().unwrap());
        let served = std::thread::spawn(move || {
            let mut served = 0;
            // Replies first: zip asks its first side first, and the
            // listener would wait for a connection past the last reply.
            for (reply, stream) in responses.into_iter().zip(listener.incoming()) {
                let mut stream = stream.unwrap();
                let mut request = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut request);
                let _ = std::io::Write::write_all(&mut stream, reply.as_bytes());
                served += 1;
            }
            served
        });
        (url, served)
    }

    const UNAVAILABLE: &str = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    const NOT_FOUND: &str = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found";
    const LISTED: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"urls\":[]}";

    /// PyPI's passing trouble (a 5xx) is asked again, as downloads are;
    /// a refusal (a 4xx) says what PyPI answered, at once.
    #[test]
    fn the_pypi_query_retries_a_server_error_and_names_a_refusal() {
        let cancel = Arc::new(AtomicBool::new(false));
        let (url, served) = serve(vec![UNAVAILABLE, LISTED]);
        assert_eq!(pypi_files(&url, &cancel), Ok(Vec::new()));
        assert_eq!(served.join().unwrap(), 2);

        let (url, served) = serve(vec![NOT_FOUND]);
        let refused = pypi_files(&url, &cancel).expect_err("a 404");
        assert!(refused.contains("404"), "{refused}");
        assert_eq!(served.join().unwrap(), 1, "a refusal is not asked again");
    }

    /// The installer checks the extracted library under its name and the
    /// loader looks for it under core's; neither crate can see the other.
    #[test]
    fn installer_and_loader_agree_on_the_library_name() {
        assert_eq!(ri::DYLIB_NAME, prunr_core::ort_runtime::DYLIB_NAME);
    }

    /// The preference file names each offered install by its dir, which
    /// names the runtime back, and the built-in value can never name an
    /// install dir.
    #[test]
    fn every_runtime_choice_round_trips_through_the_preference_file() {
        use prunr_core::ort_runtime::PREFER_BUILT_IN;
        for &rt in RuntimeId::OFFERED {
            assert_eq!(RuntimeId::from_subdir(&RuntimeChoice::Store(rt).preference()), Some(rt));
        }
        assert_eq!(RuntimeChoice::BuiltIn.preference(), PREFER_BUILT_IN);
        assert!(ri::validate_subdir(PREFER_BUILT_IN).is_ok());
        assert_eq!(RuntimeId::from_subdir(PREFER_BUILT_IN), None);
        assert_eq!(RuntimeId::from_subdir("onnxruntime-1.24.1-linux-x64"), None, "a store entry Settings does not offer");
    }

    /// A runtime library is the built-in choice, an offered install by
    /// its dir's name, or no choice at all when the environment names it.
    #[test]
    fn a_runtime_library_is_the_choice_its_place_names() {
        use prunr_core::ort_runtime::{DylibSource, DYLIB_NAME};
        let store = Path::new("/data/prunr/runtimes");
        let openvino = store.join(RuntimeId::OpenVino.install_subdir()).join(DYLIB_NAME);
        let offered = RuntimeId::OFFERED.contains(&RuntimeId::OpenVino);
        assert_eq!(
            RuntimeChoice::of_library(&openvino, DylibSource::RuntimeStore),
            offered.then_some(RuntimeChoice::Store(RuntimeId::OpenVino)),
            "an install Settings does not offer is no choice",
        );
        assert_eq!(RuntimeChoice::of_library(&store.join("onnxruntime-1.24.1-linux-x64").join(DYLIB_NAME), DylibSource::RuntimeStore), None);
        assert_eq!(RuntimeChoice::of_library(Path::new("/opt/prunr/runtime").join(DYLIB_NAME).as_path(), DylibSource::Bundled), Some(RuntimeChoice::BuiltIn));
        assert_eq!(RuntimeChoice::of_library(&openvino, DylibSource::EnvVar), None);
    }

    /// A download that will not extract leaves the install that was there
    /// and no half-made folder for the runtime picker to find.
    #[test]
    fn a_broken_download_leaves_the_old_install() {
        let store = tempfile::tempdir().unwrap();
        let target = store.path().join("openvino-1.24.1-linux-x64");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("libonnxruntime.so"), b"the working one").unwrap();
        assert!(extract_into_place(std::io::Cursor::new(b"not a zip"), &target, None).is_err());
        assert_eq!(std::fs::read(target.join("libonnxruntime.so")).unwrap(), b"the working one");
        let left: Vec<_> = std::fs::read_dir(store.path()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("openvino-1.24.1-linux-x64")], "no staging folder left");
    }

    /// Reinstalling the CUDA runtime keeps the NVIDIA libraries inside it
    /// and replaces everything else.
    #[test]
    fn a_reinstall_keeps_the_nvidia_libraries() {
        let store = tempfile::tempdir().unwrap();
        let target = store.path().join("cuda-1.24.1-linux-x64");
        let staging = store.path().join(".cuda-1.24.1-linux-x64.staging");
        let nvidia = prunr_core::ort_runtime::NVIDIA_LIBS_DIR;
        std::fs::create_dir_all(target.join(nvidia)).unwrap();
        std::fs::write(target.join(nvidia).join("libcudnn.so.9"), b"downloaded").unwrap();
        std::fs::write(target.join("libonnxruntime.so"), b"old").unwrap();
        std::fs::write(target.join("stale"), b"old").unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("libonnxruntime.so"), b"new").unwrap();
        move_into_place(&staging, &target, &[nvidia]).unwrap();
        assert_eq!(std::fs::read(target.join(nvidia).join("libcudnn.so.9")).unwrap(), b"downloaded");
        assert_eq!(std::fs::read(target.join("libonnxruntime.so")).unwrap(), b"new");
        assert!(!target.join("stale").exists(), "the rest is replaced");
        assert!(!staging.exists());
    }

    /// The first install on a computer meets no runtime folder yet: the
    /// download it writes beside its install has somewhere to go (Windows,
    /// 2026-10-26: "The system cannot find the path specified").
    #[test]
    fn the_first_install_has_a_folder_to_download_into() {
        let store = tempfile::tempdir().unwrap();
        let target = store.path().join("runtimes").join(RuntimeId::Cuda.install_subdir());
        let spool = super::spool_for(&target).unwrap();
        assert!(std::fs::File::create(&spool).is_ok(), "{}", spool.display());
    }

    /// Windows offers CUDA alone (DirectML is built in, OpenVINO's wheel
    /// there lacks openvino.dll); CUDA only where its wheel exists.
    #[test]
    fn each_platform_offers_the_runtimes_it_has_a_wheel_for() {
        let expected: &[RuntimeId] = if cfg!(all(windows, target_arch = "x86_64")) {
            &[RuntimeId::Cuda]
        } else if cfg!(windows) {
            &[]
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            &[RuntimeId::OpenVino, RuntimeId::Cuda, RuntimeId::MiGraphX]
        } else {
            &[RuntimeId::OpenVino]
        };
        assert_eq!(RuntimeId::OFFERED, expected);
        assert_eq!(RuntimeId::Cuda.is_addon(), cfg!(all(any(windows, target_os = "linux"), target_arch = "x86_64")), "CUDA joins Prunr's own runtime");
    }

    /// At the next start only the add-on marked for removal goes, with its
    /// mark: not one without a mark, not a marked folder no add-on names.
    #[test]
    fn only_a_marked_add_on_is_removed_at_the_next_start() {
        let store = tempfile::tempdir().unwrap();
        for name in ["cuda-addon-0.2.0-windows-x64", "cuda-1.24.1-windows-x64", "other"] {
            std::fs::create_dir_all(store.path().join(name).join("nvidia")).unwrap();
        }
        std::fs::write(store.path().join(".cuda-addon-0.2.0-windows-x64.remove"), b"").unwrap();
        std::fs::write(store.path().join(".other.remove"), b"").unwrap();
        remove_marked(store.path(), ["cuda-addon-0.2.0-windows-x64".to_string(), "cuda-1.24.1-windows-x64".to_string(), "../escape".to_string()]);
        assert!(!store.path().join("cuda-addon-0.2.0-windows-x64").exists(), "marked: removed");
        assert!(!store.path().join(".cuda-addon-0.2.0-windows-x64.remove").exists(), "its mark too");
        assert!(store.path().join("cuda-1.24.1-windows-x64").exists(), "not marked: kept");
        assert!(store.path().join("other").exists() && store.path().join(".other.remove").exists(), "not an add-on: kept");
    }

    /// An add-on's download leaves its library in the install, beside the
    /// NVIDIA libraries a reinstall keeps, and no runtime of its own.
    #[test]
    fn an_addon_install_holds_its_library_beside_the_nvidia_ones() {
        use std::io::Write;
        let mut zipped = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut zipped);
            for (name, body) in [(ri::CUDA_ADDON_WINDOWS.library, "plugin"), ("version.json", "{}")] {
                zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
                zip.write_all(body.as_bytes()).unwrap();
            }
            zip.finish().unwrap();
        }
        let store = tempfile::tempdir().unwrap();
        let target = store.path().join("cuda-addon-0.2.0-windows-x64");
        let nvidia = prunr_core::ort_runtime::NVIDIA_LIBS_DIR;
        std::fs::create_dir_all(target.join(nvidia)).unwrap();
        std::fs::write(target.join(nvidia).join("cudnn64_9.dll"), b"downloaded").unwrap();
        let library = extract_into_place(std::io::Cursor::new(zipped.into_inner()), &target, Some(&ri::CUDA_ADDON_WINDOWS)).unwrap();
        assert_eq!(library, target.join(ri::CUDA_ADDON_WINDOWS.library));
        assert_eq!(std::fs::read(target.join(nvidia).join("cudnn64_9.dll")).unwrap(), b"downloaded");
        assert!(!target.join(prunr_core::ort_runtime::DYLIB_NAME).exists(), "no runtime of its own");
    }
}
