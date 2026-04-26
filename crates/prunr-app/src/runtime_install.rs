//! GUI-side runtime installer. Mirrors `xtask::install_runtime` —
//! `host_rid` / `host_pypi_token` / `repackage_target_filename` /
//! `pick_wheel_for_host` are duplicated and **must stay in sync** with
//! that file until they're extracted into a shared crate.

use std::path::PathBuf;
use std::sync::mpsc;

use sha2::{Digest, Sha256};

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
    /// User-facing status label. Lives on the variant so the view layer
    /// doesn't reach into our event shape.
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
}

impl RuntimeId {
    fn pypi_package(self) -> &'static str {
        match self { Self::OpenVino => "onnxruntime-openvino" }
    }

    fn pypi_version(self) -> &'static str {
        match self { Self::OpenVino => "1.24.1" }
    }

    pub fn install_subdir(self) -> String {
        format!("{}-{}-{}", self.short_name(), self.pypi_version(), host_rid())
    }

    pub fn install_dir(self) -> Option<PathBuf> {
        prunr_models::data_dir().map(|d| d.join("runtimes").join(self.install_subdir()))
    }

    pub fn is_installed(self) -> bool {
        self.install_dir()
            .map(|d| d.join(crate::ort_runtime::DYLIB_NAME))
            .is_some_and(|p| p.is_file())
    }

    fn short_name(self) -> &'static str {
        match self { Self::OpenVino => "openvino" }
    }

    pub fn display_name(self) -> &'static str {
        match self { Self::OpenVino => "OpenVINO Runtime" }
    }

    pub fn approx_download_mb(self) -> u32 {
        match self { Self::OpenVino => 80 }
    }

    /// Stable key for persisted settings (snooze map, future
    /// per-runtime preferences). Decoupled from `Debug` output so we
    /// can rename the variant without breaking user state.
    pub fn settings_key(self) -> &'static str {
        match self { Self::OpenVino => "OpenVino" }
    }
}

/// Final event is either `Done` or `Failed`; channel closes after.
pub fn start_install(runtime: RuntimeId) -> mpsc::Receiver<InstallEvent> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        if let Err(e) = run_install(runtime, &tx) {
            let _ = tx.send(InstallEvent::Failed { error: e });
        }
    });
    rx
}

/// Remove an installed runtime. Refuses when this is the app's only
/// ORT source — uninstalling the last fallback bricks startup. Also
/// includes the parent-dir safety guard from the install path so a
/// future refactor of `install_dir()` can't nuke the wrong tree.
pub fn uninstall(runtime: RuntimeId) -> Result<(), String> {
    let dir = runtime.install_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?;
    if !dir.parent().is_some_and(|p| p.ends_with("runtimes")) {
        return Err(format!(
            "refusing to wipe non-runtimes path: {}", dir.display(),
        ));
    }
    if !dir.exists() {
        return Ok(()); // already absent — nothing to do
    }
    if !crate::ort_runtime::has_fallback_excluding(&dir) {
        return Err(
            "this is the only ONNX Runtime installed — uninstalling would leave \
             the app unable to start. Install another runtime first, or set \
             ORT_DYLIB_PATH to a system ORT before removing this one."
                .to_string()
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("remove {}: {e}", dir.display()))
}

fn run_install(rt: RuntimeId, tx: &mpsc::Sender<InstallEvent>) -> Result<(), String> {
    let _ = tx.send(InstallEvent::Preparing);
    let url_info = query_pypi(rt)?;

    let _ = tx.send(InstallEvent::Downloading {
        bytes_so_far: 0,
        total_bytes: url_info.size_bytes,
    });
    let bytes = download_with_progress(&url_info, tx)?;

    let _ = tx.send(InstallEvent::Preparing);
    verify_sha256(&bytes, &url_info.sha256)?;

    let _ = tx.send(InstallEvent::Extracting);
    let install_dir = extract_wheel(&bytes, rt)?;

    let _ = tx.send(InstallEvent::Done { install_dir });
    Ok(())
}

struct WheelInfo {
    url: String,
    sha256: String,
    size_bytes: u64,
}

fn query_pypi(rt: RuntimeId) -> Result<WheelInfo, String> {
    let url = format!(
        "https://pypi.org/pypi/{}/{}/json",
        rt.pypi_package(), rt.pypi_version(),
    );
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("prunr/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;
    let resp = client.get(&url).send()
        .map_err(|e| format!("PyPI query failed: {e}"))?;
    let metadata: serde_json::Value = resp.json()
        .map_err(|e| format!("PyPI JSON parse: {e}"))?;
    let urls = metadata["urls"].as_array()
        .ok_or_else(|| "PyPI metadata missing `urls`".to_string())?;
    pick_wheel_for_host(urls)
}

fn pick_wheel_for_host(urls: &[serde_json::Value]) -> Result<WheelInfo, String> {
    let host_token = host_pypi_token().ok_or_else(||
        format!("unsupported host platform `{}` for OpenVINO runtime", host_rid())
    )?;
    let pick = |require_cp313: bool| urls.iter().find_map(|u| {
        let name = u["filename"].as_str()?;
        if !name.contains(host_token) { return None; }
        if require_cp313 && !name.contains("cp313") { return None; }
        Some(WheelInfo {
            url: u["url"].as_str()?.to_string(),
            sha256: u["digests"]["sha256"].as_str()?.to_string(),
            size_bytes: u["size"].as_u64().unwrap_or(0),
        })
    });
    pick(true).or_else(|| pick(false)).ok_or_else(||
        format!("no wheel for host platform `{}`", host_rid()),
    )
}

fn download_with_progress(
    info: &WheelInfo,
    tx: &mpsc::Sender<InstallEvent>,
) -> Result<bytes::Bytes, String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("prunr/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;
    let mut response = client.get(&info.url).send()
        .map_err(|e| format!("download: {e}"))?;
    let total = response.content_length().unwrap_or(info.size_bytes);

    // Don't pre-allocate from `total` — the server-supplied
    // Content-Length is untrusted; a bogus value would be a one-shot
    // OOM. `Vec` grows; the resize cost on 80 MB is microseconds.
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut last_pct: u64 = u64::MAX;
    loop {
        let n = std::io::Read::read(&mut response, &mut chunk)
            .map_err(|e| format!("download read: {e}"))?;
        if n == 0 { break; }
        buf.extend_from_slice(&chunk[..n]);
        // Throttle progress to 1% boundaries — at 64 KB chunks an 80 MB
        // download fires 1280 events otherwise, swamping the egui poll.
        let pct = if total > 0 { buf.len() as u64 * 100 / total } else { 0 };
        if pct != last_pct {
            last_pct = pct;
            let _ = tx.send(InstallEvent::Downloading {
                bytes_so_far: buf.len() as u64,
                total_bytes: total,
            });
        }
    }
    Ok(bytes::Bytes::from(buf))
}

fn verify_sha256(bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex::encode(Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!(
            "SHA256 mismatch — expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn extract_wheel(bytes: &[u8], rt: RuntimeId) -> Result<PathBuf, String> {
    let target_dir = prunr_models::data_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?
        .join("runtimes")
        .join(rt.install_subdir());
    if !target_dir.parent().is_some_and(|p| p.ends_with("runtimes")) {
        return Err(format!(
            "refusing to wipe non-runtimes path: {}",
            target_dir.display(),
        ));
    }
    if target_dir.exists() {
        std::fs::remove_dir_all(&target_dir)
            .map_err(|e| format!("clean install dir: {e}"))?;
    }
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("create install dir: {e}"))?;

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("zip parse: {e}"))?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| format!("zip entry: {e}"))?;
        let name = entry.name().to_string();
        let Some(target_filename) = repackage_target_filename(&name) else {
            continue;
        };
        let dest = target_dir.join(target_filename);
        let mut out = std::fs::File::create(&dest)
            .map_err(|e| format!("create {}: {e}", dest.display()))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("extract {}: {e}", dest.display()))?;
    }

    let dylib = target_dir.join(crate::ort_runtime::DYLIB_NAME);
    if !dylib.is_file() {
        return Err(format!(
            "{} missing after extract — wheel layout may have changed",
            crate::ort_runtime::DYLIB_NAME,
        ));
    }
    Ok(target_dir)
}

/// Filter + rename for files extracted from an `onnxruntime-*` wheel.
/// Mirrors `xtask::repackage_target_filename` exactly — keep them in
/// sync if the wheel layout changes upstream.
fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if stripped.contains('/') { return None; }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
    if stripped.ends_with(".py") { return None; }
    if stripped.starts_with("libonnxruntime.so.") {
        return Some("libonnxruntime.so".to_string());
    }
    Some(stripped.to_string())
}

fn host_rid() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) { "linux-x64" }
    else if cfg!(all(target_os = "linux", target_arch = "aarch64")) { "linux-arm64" }
    else if cfg!(all(target_os = "windows", target_arch = "x86_64")) { "windows-x64" }
    else if cfg!(all(target_os = "windows", target_arch = "aarch64")) { "windows-arm64" }
    else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") { "macos-arm64" } else { "macos-x64" }
    }
    else { "unknown" }
}

fn host_pypi_token() -> Option<&'static str> {
    Some(match host_rid() {
        "linux-x64" => "manylinux_2_28_x86_64",
        "linux-arm64" => "manylinux_2_28_aarch64",
        "windows-x64" => "win_amd64",
        "windows-arm64" => "win_arm64",
        "macos-arm64" => "macosx_11_0_arm64",
        _ => return None,
    })
}
