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
    Done { install_dir: PathBuf },
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
    /// CUDA and MIGraphX are offered where DirectML is not and their
    /// runtimes have a wheel: Linux on x86-64.
    pub const OFFERED: &'static [RuntimeId] = if cfg!(windows) {
        &[]
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        &[Self::OpenVino, Self::Cuda, Self::MiGraphX]
    } else {
        &[Self::OpenVino]
    };

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
        ri::install_subdir(self.short_name(), self.ort_version())
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
        self.install_dir()
            .map(|d| d.join(prunr_core::ort_runtime::DYLIB_NAME))
            .is_some_and(|p| p.is_file())
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
            Self::Cuda => "CUDA Runtime",
            Self::MiGraphX => "MIGraphX Runtime",
        }
    }

    /// Settings › Hardware's line while this runtime is not installed: its
    /// download, and why to get it when the hardware suits it.
    pub fn not_installed(self, recommended: bool) -> &'static str {
        match (self, recommended) {
            (Self::OpenVino, false) => "Not installed (80 MB).",
            (Self::OpenVino, true) => "Not installed (80 MB). Recommended for Intel hardware: the models run faster, and you can try running them on the Intel graphics chip.",
            (Self::Cuda, false) => "Not installed (250 MB). Runs the models on an NVIDIA graphics card; needs NVIDIA's CUDA 12 and cuDNN 9 libraries.",
            (Self::Cuda, true) => "Not installed (250 MB). Recommended for your NVIDIA graphics card: runs the models on it. Needs NVIDIA's CUDA 12 and cuDNN 9 libraries.",
            (Self::MiGraphX, false) => "Not installed (20 MB). Runs the models on an AMD graphics card; needs AMD's ROCm 7 with MIGraphX.",
            (Self::MiGraphX, true) => "Not installed (20 MB). Recommended for your AMD graphics card: runs the models on it. Needs AMD's ROCm 7 with MIGraphX.",
        }
    }

    pub fn uninstall_label(self) -> &'static str {
        match self {
            Self::OpenVino => "Uninstall OpenVINO Runtime",
            Self::Cuda => "Uninstall CUDA Runtime",
            Self::MiGraphX => "Uninstall MIGraphX Runtime",
        }
    }

    /// The first-launch prompt's body.
    pub fn first_launch_prompt_body(self) -> &'static str {
        match self {
            Self::OpenVino => "We detected Intel hardware that can use OpenVINO Runtime. \
                Installing it (~80 MB) makes the models run faster, and lets you \
                try running them on the Intel graphics chip (Settings › Hardware).",
            Self::Cuda => "We detected an NVIDIA graphics card. Installing CUDA Runtime \
                (~250 MB) runs the models on it instead of the processor. It needs \
                NVIDIA's CUDA 12 and cuDNN 9 libraries on this computer.",
            Self::MiGraphX => "We detected an AMD graphics card. Installing MIGraphX Runtime \
                (~20 MB) runs the models on it instead of the processor. It needs \
                AMD's ROCm 7 with MIGraphX on this computer.",
        }
    }

    /// The first-launch prompt's install-button label.
    pub fn install_button_label(self) -> &'static str {
        match self {
            Self::OpenVino => "Install OpenVINO Runtime",
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
            Self::BuiltIn => "DirectML: any DirectX 12 graphics card, else the processor",
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
pub fn start_install(runtime: RuntimeId) -> InstallHandle {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_thread = Arc::clone(&cancel);
    std::thread::spawn(move || {
        if let Err(e) = run_install(runtime, &tx, cancel_for_thread) {
            let _ = tx.send(InstallEvent::Failed { error: e });
        }
    });
    InstallHandle { events: rx, cancel }
}

/// Remove an installed runtime. Refuses when this is the only ORT source
/// (uninstalling would brick startup) and when the subdir name could
/// escape `<data>/runtimes/`.
pub fn uninstall(runtime: RuntimeId) -> Result<(), String> {
    let dir = runtime.checked_install_dir()?;
    if !dir.exists() {
        return Ok(()); // already absent — nothing to do
    }
    if !prunr_core::ort_runtime::has_fallback_excluding(&dir) {
        return Err(
            "this is the only ONNX Runtime installed — uninstalling would leave \
             the app unable to start. Install another runtime first, or set \
             ORT_DYLIB_PATH to a system ORT before removing this one."
                .to_string()
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("remove {}: {e}", dir.display()))?;
    prunr_core::ort_runtime::forget_preferred_runtime(&runtime.install_subdir());
    Ok(())
}

fn run_install(
    rt: RuntimeId,
    tx: &mpsc::Sender<InstallEvent>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    let _ = tx.send(InstallEvent::Preparing);
    let wheel = query_pypi(rt)?;

    let _ = tx.send(InstallEvent::Downloading {
        bytes_so_far: 0,
        total_bytes: wheel.size_bytes,
    });
    let mut on_progress = |so_far: u64, total: u64| {
        let _ = tx.send(InstallEvent::Downloading {
            bytes_so_far: so_far, total_bytes: total,
        });
    };
    let mut hooks = ri::DownloadHooks { progress: &mut on_progress, cancel };
    // Already SHA-verified inside the retry loop.
    let bytes = ri::download_wheel(&wheel, &mut hooks)?;

    let _ = tx.send(InstallEvent::Extracting);
    let install_dir = extract_into_place(&bytes, &rt.checked_install_dir()?)?;
    // The runtime just installed is the one the user wants next start.
    if let Err(e) = prefer(RuntimeChoice::Store(rt)) {
        tracing::warn!(runtime = rt.settings_key(), %e, "could not mark the new runtime as preferred");
    }

    let _ = tx.send(InstallEvent::Done { install_dir });
    Ok(())
}

fn query_pypi(rt: RuntimeId) -> Result<ri::WheelInfo, String> {
    let url = format!("https://pypi.org/pypi/{}/{}/json", rt.pypi_package(), rt.ort_version());
    let client = ri::http_client(USER_AGENT).map_err(|e| format!("HTTP client: {}", ri::with_causes(&e)))?;
    let resp = client.get(&url).send()
        .map_err(|e| format!("PyPI query failed: {}", ri::with_causes(&e)))?;
    let metadata: serde_json::Value = resp.json()
        .map_err(|e| format!("PyPI JSON parse: {}", ri::with_causes(&e)))?;
    let urls = metadata["urls"].as_array()
        .ok_or_else(|| "PyPI metadata missing `urls`".to_string())?;
    ri::pick_wheel_for_host(urls)
}

/// Extract into a hidden folder beside the install and move it into place
/// once whole: an install cut short (a full disk, the app closed) leaves
/// the runtime that was there, or none, never half of one to load.
fn extract_into_place(bytes: &[u8], target_dir: &Path) -> Result<PathBuf, String> {
    let name = target_dir.file_name().ok_or("install dir has no name")?.to_string_lossy().into_owned();
    let staging = target_dir.with_file_name(format!(".{name}.staging"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(|e| format!("clean staging dir: {e}"))?;
    }
    std::fs::create_dir_all(&staging).map_err(|e| format!("create staging dir: {e}"))?;
    let library = ri::extract_wheel(bytes, &staging).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&staging);
    })?;
    if target_dir.exists() {
        std::fs::remove_dir_all(target_dir).map_err(|e| format!("clean install dir: {e}"))?;
    }
    std::fs::rename(&staging, target_dir).map_err(|e| format!("move install into place: {e}"))?;
    Ok(target_dir.join(library.file_name().ok_or("extracted library has no name")?))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(extract_into_place(b"not a zip", &target).is_err());
        assert_eq!(std::fs::read(target.join("libonnxruntime.so")).unwrap(), b"the working one");
        let left: Vec<_> = std::fs::read_dir(store.path()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("openvino-1.24.1-linux-x64")], "no staging folder left");
    }

    /// Windows offers no download (DirectML is built in, OpenVINO's wheel
    /// there lacks openvino.dll); CUDA only where its wheel exists.
    #[test]
    fn each_platform_offers_the_runtimes_it_has_a_wheel_for() {
        let expected: &[RuntimeId] = if cfg!(windows) {
            &[]
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            &[RuntimeId::OpenVino, RuntimeId::Cuda, RuntimeId::MiGraphX]
        } else {
            &[RuntimeId::OpenVino]
        };
        assert_eq!(RuntimeId::OFFERED, expected);
    }
}
