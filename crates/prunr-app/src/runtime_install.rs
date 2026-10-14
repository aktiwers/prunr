//! GUI-side runtime install glue. The cross-cutting helpers (host RID,
//! wheel selection, SHA256 verify, repackaging, install-dir naming) live
//! in the `prunr-runtime-install` crate and are shared with `xtask`.
//! This module wires those into a `RuntimeId` enum + `mpsc::Sender`-based
//! progress events for the GUI.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};

use prunr_runtime_install as ri;

/// User agent for every HTTP request the app makes.
pub const USER_AGENT: &str = concat!("prunr/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone)]
pub enum InstallEvent {
    /// Pre-download work (PyPI metadata query, post-download SHA verify).
    /// Rendered as a single "Preparing…" state since both legs are
    /// sub-second and flicker as distinct UI labels.
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
    DirectMl,
}

impl RuntimeId {
    /// The runtimes this platform offers. DirectML is a Windows API.
    pub const OFFERED: &'static [RuntimeId] = if cfg!(windows) {
        &[Self::DirectMl, Self::OpenVino]
    } else {
        &[Self::OpenVino]
    };

    fn pypi_package(self) -> &'static str {
        match self {
            Self::OpenVino => "onnxruntime-openvino",
            Self::DirectMl => "onnxruntime-directml",
        }
    }

    fn pypi_version(self) -> &'static str {
        match self {
            Self::OpenVino | Self::DirectMl => prunr_runtime_install::PINNED_ORT_VERSION,
        }
    }

    pub fn install_subdir(self) -> String {
        ri::install_subdir(self.short_name(), self.pypi_version())
    }

    pub fn install_dir(self) -> Option<PathBuf> {
        prunr_models::runtime_store_dir().map(|d| d.join(self.install_subdir()))
    }

    pub fn is_installed(self) -> bool {
        self.install_dir()
            .map(|d| d.join(prunr_core::ort_runtime::DYLIB_NAME))
            .is_some_and(|p| p.is_file())
    }

    /// The installed runtime every process loads first, when one is set.
    pub fn preferred() -> Option<Self> {
        let subdir = prunr_core::ort_runtime::preferred_runtime()?;
        Self::OFFERED.iter().copied().find(|rt| rt.install_subdir() == subdir)
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::OpenVino => "openvino",
            Self::DirectMl => "directml",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::OpenVino => "OpenVINO Runtime",
            Self::DirectMl => "DirectML Runtime",
        }
    }

    pub fn approx_download_mb(self) -> u32 {
        match self {
            Self::OpenVino => 80,
            Self::DirectMl => 25,
        }
    }

    /// Pre-baked first-launch prompt body. Static-str instead of a
    /// per-frame `format!` so the modal doesn't churn ~300 B per
    /// repaint while the user reads the prompt.
    pub fn first_launch_prompt_body(self) -> &'static str {
        match self {
            Self::OpenVino => "We detected hardware that can use OpenVINO Runtime. \
                Installing it (~80 MB) unlocks 2-3× faster background removal and \
                makes Stable Diffusion inpaint usable on your machine.",
            Self::DirectMl => "Your graphics card can run Prunr's models. \
                Installing DirectML Runtime (~25 MB) moves background removal, \
                upscaling and the erasers from the processor to the card.",
        }
    }

    /// Pre-baked install-button label for the prompt. Mirrors
    /// `first_launch_prompt_body` — the variant count is small enough
    /// to enumerate, and the `display_name` interpolation is the only
    /// reason this needed a per-frame `format!` before.
    pub fn install_button_label(self) -> &'static str {
        match self {
            Self::OpenVino => "Install OpenVINO Runtime",
            Self::DirectMl => "Install DirectML Runtime",
        }
    }

    /// Stable key for persisted settings — decoupled from `Debug` output
    /// so we can rename the variant without breaking user state.
    pub fn settings_key(self) -> &'static str {
        match self {
            Self::OpenVino => "OpenVino",
            Self::DirectMl => "DirectMl",
        }
    }
}

/// Make `runtime` the one every process loads from its next start.
pub fn prefer(runtime: RuntimeId) -> Result<(), String> {
    let subdir = runtime.install_subdir();
    ri::validate_subdir(&subdir).map_err(|e| format!("install_subdir invariant broken: {e}"))?;
    prunr_core::ort_runtime::set_preferred_runtime(&subdir)
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
/// (uninstalling would brick startup) and validates the subdir name so
/// a future refactor of `install_subdir()` (or a future variant whose
/// short-name becomes user-supplied) can't escape `<data>/runtimes/` via
/// embedded path separators.
pub fn uninstall(runtime: RuntimeId) -> Result<(), String> {
    ri::validate_subdir(&runtime.install_subdir())
        .map_err(|e| format!("install_subdir invariant broken: {e}"))?;
    let dir = runtime.install_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?;
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
    // SHA verification is folded into `download_wheel`'s retry loop —
    // bytes are already verified when this returns Ok. The Preparing
    // event is the lifecycle marker; no separate verify call needed.
    let bytes = ri::download_wheel(&wheel, &mut hooks)?;
    let _ = tx.send(InstallEvent::Preparing);

    let _ = tx.send(InstallEvent::Extracting);
    let install_dir = prepare_and_extract(&bytes, rt)?;
    // The runtime just installed is the one the user wants next start.
    if let Err(e) = prefer(rt) {
        tracing::warn!(runtime = rt.settings_key(), %e, "could not mark the new runtime as preferred");
    }

    let _ = tx.send(InstallEvent::Done { install_dir });
    Ok(())
}

fn query_pypi(rt: RuntimeId) -> Result<ri::WheelInfo, String> {
    let url = format!(
        "https://pypi.org/pypi/{}/{}/json",
        rt.pypi_package(), rt.pypi_version(),
    );
    let client = ri::http_client(USER_AGENT).map_err(|e| format!("HTTP client: {e}"))?;
    let resp = client.get(&url).send()
        .map_err(|e| format!("PyPI query failed: {e}"))?;
    let metadata: serde_json::Value = resp.json()
        .map_err(|e| format!("PyPI JSON parse: {e}"))?;
    let urls = metadata["urls"].as_array()
        .ok_or_else(|| "PyPI metadata missing `urls`".to_string())?;
    ri::pick_wheel_for_host(urls)
}

fn prepare_and_extract(bytes: &[u8], rt: RuntimeId) -> Result<PathBuf, String> {
    ri::validate_subdir(&rt.install_subdir())
        .map_err(|e| format!("install_subdir invariant broken: {e}"))?;
    let target_dir = prunr_models::runtime_store_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?
        .join(rt.install_subdir());
    if target_dir.exists() {
        std::fs::remove_dir_all(&target_dir)
            .map_err(|e| format!("clean install dir: {e}"))?;
    }
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("create install dir: {e}"))?;
    ri::extract_wheel(bytes, &target_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directml_installs_from_its_own_wheel_into_its_own_folder() {
        let rt = RuntimeId::DirectMl;
        assert_eq!(rt.pypi_package(), "onnxruntime-directml");
        assert_eq!(rt.install_subdir(), format!("directml-{}-{}", ri::PINNED_ORT_VERSION, ri::host_rid()));
        assert_ne!(rt.install_subdir(), RuntimeId::OpenVino.install_subdir());
        assert_eq!(rt.settings_key(), "DirectMl");
    }

    #[test]
    fn directml_is_offered_only_on_windows() {
        assert_eq!(RuntimeId::OFFERED.contains(&RuntimeId::DirectMl), cfg!(windows));
        assert!(RuntimeId::OFFERED.contains(&RuntimeId::OpenVino));
    }
}
