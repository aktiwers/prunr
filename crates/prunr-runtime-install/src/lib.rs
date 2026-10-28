//! Shared host-detection + PyPI-wheel install logic for ORT runtimes.
//!
//! Used by `xtask install-runtime` (build-side CLI) and the GUI Settings →
//! Hardware → Install button.

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// The ONNX Runtime version of the accelerator builds the runtime store
/// offers (OpenVINO, CUDA). Their wheels lag the plain one: OpenVINO's
/// stop at 1.24.1, and CUDA's need CUDA 13 from 1.27.
/// The pykeio/ort `api-24` feature refuses a runtime older than 1.24 at
/// load.
///
/// **Upper bound:** past the 1.x line the ABI is expected to change; a
/// bump without switching the `ort` API feature compiles but crashes at
/// session creation. Run `cargo xtask probe-load-dynamic <new-dylib>`
/// before bumping either version.
pub const PINNED_ORT_VERSION: &str = "1.24.1";

/// The version of the plain `onnxruntime` wheel, which the Linux build
/// ships (CPU) and the macOS one (CPU and CoreML). From 1.24 to 1.28 its
/// constant folding corrupts weights, and with folding off CoreML loses
/// the Magic Brush, DDColor and BiRefNet. Not 1.31: it moved CoreML onto
/// device-based provider selection, and on a MacBook Air our CoreML
/// options stopped applying (Silueta's output came out wrong). The macOS
/// wheel's minimum OS is
/// the Mac app's floor (`LSMinimumSystemVersion`, 14.0 for 1.30), so a
/// bump checks that too.
pub const PLAIN_ORT_VERSION: &str = "1.30.0";

/// The Windows ML build of ONNX Runtime the Windows release ships
/// (`onnxruntime-windowsml`: `onnxruntime.dll` and `DirectML.dll`).
/// Microsoft's DirectML builds (`onnxruntime-directml`) end at 1.24.4,
/// where constant folding corrupts weights and runs off: DDColor then
/// gives a wrong result on DirectML, and runs right on this 1.30 build.
pub const WINDOWS_ML_VERSION: &str = "1.30.0.202609102321";

/// A provider library that loads into the runtime the app ships (an ORT
/// plugin execution provider), as its release publishes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddOn {
    pub version: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    /// The library inside the archive: at its root, or in `folder`.
    pub library: &'static str,
    /// The archive's folder the library loads its own libraries from,
    /// kept beside it; `None` when the library comes alone.
    pub folder: Option<&'static str>,
    /// The beginnings of the names in `folder` left out: parts the app
    /// never loads.
    pub left_out: &'static [&'static str],
    pub packing: Packing,
}

/// How a release packs its files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    Zip,
    /// A gzip-compressed tar.
    TarGz,
}

impl AddOn {
    /// Its download, checked as a wheel's is.
    pub fn download(&self) -> WheelInfo {
        WheelInfo { url: self.url.to_string(), sha256: self.sha256.to_string(), size_bytes: self.size_bytes }
    }
}

/// Microsoft's CUDA provider as a plugin (CUDA 12, cuDNN 9), the release
/// matched to ONNX Runtime 1.30, which the Windows ML build is. It brings
/// no NVIDIA libraries; the app downloads those beside it. 0.3.0 is
/// matched to 1.31 and CUDA 13 only.
pub const CUDA_ADDON_WINDOWS: AddOn = AddOn {
    version: "0.2.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.2.0/cuda_ep_cuda12_0.2.0_win-x64.zip",
    sha256: "ba698427418129d0215c1d51367a1a97995cc1eed56056479288e3245d19947c",
    size_bytes: 238_621_368,
    library: "onnxruntime_providers_cuda.dll",
    folder: None,
    left_out: &[],
    packing: Packing::Zip,
};

/// `CUDA_ADDON_WINDOWS`'s Linux build. It needs CUDA 12.5 or later, which
/// the libraries the app downloads beside it are; Ubuntu's own (12.4)
/// lack what it calls.
pub const CUDA_ADDON_LINUX: AddOn = AddOn {
    version: "0.2.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.2.0/cuda_ep_cuda12_0.2.0_linux-x64.tar.gz",
    sha256: "22cfe716fad99d338cad73e325611f8ba59eb9b0d1932c43701ac998a5037ff3",
    size_bytes: 436_640_825,
    library: "libonnxruntime_providers_cuda.so",
    folder: None,
    left_out: &[],
    packing: Packing::TarGz,
};

/// Intel's OpenVINO provider as a plugin, built against ONNX Runtime 1.30.
/// It brings OpenVINO 2026.4.1 with oneTBB and hwloc in its folder, and
/// finds them there; the Intel graphics also need Intel's own driver. Its
/// NPU part is left out: the app names the processor and the graphics
/// only, so OpenVINO never loads it.
pub const OPENVINO_ADDON_LINUX: AddOn = AddOn {
    version: "1.8.1",
    url: "https://files.pythonhosted.org/packages/88/01/4e72a8a65b584f01d6e59eb453bae397b82b2e51df1b5c94efa05e17800b/onnxruntime_ep_openvino-1.8.1-py3-none-manylinux_2_28_x86_64.whl",
    sha256: "3832643158c8c187fcab6db661cde2a8c06a2b3559d95d0f2411ac0e51c5e924",
    size_bytes: 52_587_628,
    library: "libonnxruntime_providers_openvino_plugin.so",
    folder: Some("onnxruntime_ep_openvino"),
    left_out: &["libopenvino_intel_npu"],
    packing: Packing::Zip,
};

/// `OPENVINO_ADDON_LINUX`'s Windows build. Windows finds its DLLs beside
/// it: ONNX Runtime loads a plugin with its own folder searched first.
/// Without its NPU part it keeps no networking import at all.
pub const OPENVINO_ADDON_WINDOWS: AddOn = AddOn {
    version: "1.8.1",
    url: "https://files.pythonhosted.org/packages/75/98/224ead07344390258ae0515ed285ba2951e9c1405f4507443a03bb7dd5c7/onnxruntime_ep_openvino-1.8.1-py3-none-win_amd64.whl",
    sha256: "1acd878be173b02bb8b953460f2f59b121e5f6e47ebc6aad490c0590cf6718fb",
    size_bytes: 75_824_318,
    library: "onnxruntime_providers_openvino_plugin.dll",
    folder: Some("onnxruntime_ep_openvino"),
    left_out: &["openvino_intel_npu"],
    packing: Packing::Zip,
};

/// Host runtime identifier — `linux-x64`, `macos-arm64`, etc. `unknown`
/// for unsupported platforms, for which no wheel fits.
pub fn host_rid() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) { "linux-x64" }
    else if cfg!(all(target_os = "linux", target_arch = "aarch64")) { "linux-arm64" }
    else if cfg!(all(target_os = "windows", target_arch = "x86_64")) { "windows-x64" }
    else if cfg!(all(target_os = "windows", target_arch = "aarch64")) { "windows-arm64" }
    else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") { "macos-arm64" } else { "macos-x64" }
    }
    else { "unknown" }
}

/// This machine's glibc: the newest a Linux wheel may need to load here.
/// (0, 0) off glibc, where no manylinux wheel fits.
pub fn host_glibc() -> (u32, u32) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: glibc returns a static NUL-terminated version string.
        let version = unsafe { std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()) };
        if let Some((major, minor)) = version.to_str().ok().and_then(|v| v.split_once('.')) {
            let minor = minor.split(|c: char| !c.is_ascii_digit()).next().unwrap_or("");
            if let (Ok(major), Ok(minor)) = (major.parse(), minor.parse()) {
                return (major, minor);
            }
        }
    }
    (0, 0)
}

/// The glibc a manylinux tag for `arch` needs: `manylinux_2_27_x86_64`,
/// or a legacy alias (`manylinux2014_x86_64` is 2.17).
fn manylinux_glibc(tag: &str, arch: &str) -> Option<(u32, u32)> {
    let rest = tag.strip_prefix("manylinux")?.strip_suffix(arch)?.strip_suffix('_')?;
    match rest {
        "1" => Some((2, 5)),
        "2010" => Some((2, 12)),
        "2014" => Some((2, 17)),
        _ => {
            let (major, minor) = rest.strip_prefix('_')?.split_once('_')?;
            Some((major.parse().ok()?, minor.parse().ok()?))
        }
    }
}

/// Whether a wheel's platform tag runs on `rid`. A macOS tag carries
/// the wheel's minimum OS, which moves between ORT releases
/// (`macosx_11_0_arm64` before 1.24, `macosx_14_0_arm64` from it), so
/// only its arch is fixed. A Linux tag may need any glibc up to `glibc`,
/// the host's (NVIDIA's wheels carry 2.12 to 2.27, MIGraphX's 2.34).
fn tag_fits(tag: &str, rid: &str, glibc: (u32, u32)) -> bool {
    let number = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let macos = |arch: &str| tag.strip_prefix("macosx_")
        .and_then(|t| t.strip_suffix(arch))
        .and_then(|t| t.strip_suffix('_'))
        .and_then(|version| version.split_once('_'))
        .is_some_and(|(major, minor)| number(major) && number(minor));
    match rid {
        "linux-x64" => manylinux_glibc(tag, "x86_64").is_some_and(|v| v <= glibc),
        "linux-arm64" => manylinux_glibc(tag, "aarch64").is_some_and(|v| v <= glibc),
        "windows-x64" => tag == "win_amd64",
        "windows-arm64" => tag == "win_arm64",
        "macos-arm64" => macos("arm64"),
        "macos-x64" => macos("x86_64"),
        _ => false,
    }
}

/// A wheel filename's platform tags: its last `-` field, which may
/// join several with `.` (`manylinux_2_27_x86_64.manylinux_2_28_x86_64`).
fn platform_tags(filename: &str) -> impl Iterator<Item = &str> {
    filename.strip_suffix(".whl")
        .and_then(|stem| stem.rsplit('-').next())
        .into_iter()
        .flat_map(|field| field.split('.'))
}

/// Install-dir name: `<short>-<version>-<rid>`. Locked — existing
/// user installs at `<data>/prunr/runtimes/<this>` must keep working.
pub fn install_subdir(short_name: &str, version: &str) -> String {
    format!("{short_name}-{version}-{}", host_rid())
}

/// Reject names that would escape `<data_dir>/runtimes/` when joined
/// as a single path segment.
pub fn validate_subdir(name: &str) -> Result<(), String> {
    if !one_segment(name) {
        return Err(format!("invalid runtime subdirectory name: {name:?}"));
    }
    Ok(())
}

/// Whether `name`, joined onto a folder, names a file or folder in it:
/// the guard every extracted name and install folder passes, so none
/// reaches outside.
fn one_segment(name: &str) -> bool {
    !matches!(name, "" | "." | "..") && !name.contains('/') && !name.contains('\\')
}

#[derive(Debug, Clone)]
pub struct WheelInfo {
    pub url: String,
    pub sha256: String,
    /// `0` when PyPI metadata didn't carry `size`.
    pub size_bytes: u64,
}

/// Pick the right wheel for the host platform. Prefers cp313, falls
/// back to any cp tag.
pub fn pick_wheel_for_host(urls: &[serde_json::Value]) -> Result<WheelInfo, String> {
    pick_wheel_on(urls, host_rid(), host_glibc())
}

fn pick_wheel_on(urls: &[serde_json::Value], rid: &str, glibc: (u32, u32)) -> Result<WheelInfo, String> {
    let pick = |require_cp313: bool| urls.iter().find_map(|u| {
        let name = u["filename"].as_str()?;
        // An sdist whose name carries a wheel-flavour token would
        // otherwise fail confusingly in `extract_wheel`.
        if !name.ends_with(".whl") { return None; }
        if !platform_tags(name).any(|tag| tag_fits(tag, rid, glibc)) { return None; }
        if require_cp313 && !name.contains("cp313") { return None; }
        Some(WheelInfo {
            url: u["url"].as_str()?.to_string(),
            sha256: u["digests"]["sha256"].as_str()?.to_string(),
            size_bytes: u["size"].as_u64().unwrap_or(0),
        })
    });
    pick(true).or_else(|| pick(false)).ok_or_else(||
        format!("no wheel for host platform `{rid}`"),
    )
}

/// Filter + rename for files extracted from an `onnxruntime-*` wheel.
///
/// Path-traversal guard: the result is joined onto `target_dir` as one
/// filename, so anything containing `/` or `\\` is rejected.
pub fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if !one_segment(stripped) {
        return None;
    }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
    // The GPU wheel's TensorRT provider: no session registers it.
    if stripped.trim_start_matches("lib").starts_with("onnxruntime_providers_tensorrt") { return None; }
    if stripped.ends_with(".py") { return None; }
    if stripped.starts_with("libonnxruntime.so.") {
        return Some("libonnxruntime.so".to_string());
    }
    // The third condition skips the canonical un-versioned form so it
    // falls through to the pass-through return below.
    if stripped.starts_with("libonnxruntime.")
        && stripped.ends_with(".dylib")
        && stripped != "libonnxruntime.dylib"
    {
        return Some("libonnxruntime.dylib".to_string());
    }
    Some(stripped.to_string())
}

fn verify_digest(digest: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex::encode(digest);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!("SHA256 mismatch — expected {expected}, got {actual}"));
    }
    Ok(())
}

/// Cancel + per-chunk progress hooks supplied by the caller.
pub struct DownloadHooks<'a> {
    pub progress: &'a mut dyn FnMut(u64, u64),
    pub cancel: Arc<AtomicBool>,
}

impl<'a> DownloadHooks<'a> {
    /// Hooks with a no-op cancel — for non-cancellable callers (xtask).
    pub fn progress_only(progress: &'a mut dyn FnMut(u64, u64)) -> Self {
        Self { progress, cancel: Arc::new(AtomicBool::new(false)) }
    }
}

/// Retry policy the app's downloads share, so runtime and model installs
/// give up after the same number of tries.
pub const DOWNLOAD_ATTEMPTS: u32 = 3;
pub const DOWNLOAD_BACKOFF_BASE_MS: u64 = 500;

/// Wraps `download_wheel_attempt` in `DOWNLOAD_ATTEMPTS` tries with
/// exponential backoff. Transient errors (network timeout, 5xx, broken
/// connection, SHA256 mismatch) retry; fatal errors (404, user cancel)
/// fail fast.
///
/// SHA verification runs inside the attempt so a transient corruption
/// (partial CDN payload, bit flip) gets retried rather than failing.
/// The wheel downloaded to the file `path` (made or replaced), checked
/// against its SHA-256 as it streams, never held in memory whole (cuDNN's
/// is 766 MB); open for reading. The caller removes the file.
pub fn download_wheel(info: &WheelInfo, hooks: &mut DownloadHooks<'_>, path: &Path) -> Result<std::fs::File, String> {
    let cancel = Arc::clone(&hooks.cancel);
    retry_with_backoff(&cancel, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS, |attempt| {
        if attempt > 0 {
            tracing::info!(attempt, url = %info.url, "retrying wheel download");
        }
        let digest = download_wheel_attempt(info, hooks, path)?;
        verify_digest(&digest, &info.sha256).map_err(DlError::transient)?;
        std::fs::File::open(path).map_err(|e| DlError::fatal(format!("open {}: {e}", path.display())))
    }).map_err(|e| e.message)
}

/// An error that tells transient from fatal. `cancelled()` builds the
/// value for a cancel during backoff, so each type picks its own wording.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
    fn cancelled() -> Self;
    /// How many times the usual wait before the next try.
    fn patience(&self) -> u64 {
        1
    }
}

/// A failed download attempt: what to tell the user, and whether another
/// attempt could succeed. Every downloader (runtime wheels, on-demand
/// models, xtask's model fetch) shares it.
#[derive(Debug)]
pub struct DlError {
    pub message: String,
    retryable: bool,
    /// The network itself is down: no name lookup, no route.
    offline: bool,
}

/// The wait before the next try, against the usual, when the network is
/// down: a Wi-Fi coming back after sleep or a VPN reconnecting takes
/// seconds. A timeout has waited its own 15-60 s already, and a refused
/// connection will not clear by waiting.
const OFFLINE_PATIENCE: u64 = 8;

impl DlError {
    pub fn fatal(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: false, offline: false } }
    pub fn transient(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: true, offline: false } }

    /// A request that got no answer, said in the person's words, worth
    /// another try.
    pub fn request(error: &reqwest::Error) -> Self {
        let host = error.url().and_then(|u| u.host_str()).unwrap_or("the server");
        Self::unanswered(host, &with_causes(error), error.is_timeout(), error.is_connect())
    }

    /// A download that broke off part way, worth another try.
    pub fn broke_off(error: &std::io::Error) -> Self {
        let causes = with_causes(error);
        Self::transient(format!("the download broke off ({})", last_cause(&causes)))
    }

    // hyper-util words every failed name lookup "dns error"; the system's
    // own text under it differs per platform.
    fn unanswered(host: &str, causes: &str, timeout: bool, connect: bool) -> Self {
        let lower = causes.to_ascii_lowercase();
        let unreachable = lower.contains("unreachable") || lower.contains("no route to host");
        let message = if causes.contains("dns error") {
            format!("no internet connection ({host} could not be found)")
        } else if unreachable {
            format!("no internet connection ({host} could not be reached)")
        } else if timeout {
            format!("{host} did not answer in time")
        } else if connect {
            format!("could not reach {host} ({})", last_cause(causes))
        } else {
            format!("the request to {host} failed: {causes}")
        };
        Self { message, retryable: true, offline: causes.contains("dns error") || unreachable }
    }

    /// A response that was not a success: a 5xx is the server's passing
    /// trouble and is worth another attempt; anything else (a 4xx: bad
    /// URL, gone file) fails at once.
    pub fn for_status(status: reqwest::StatusCode, msg: impl Into<String>) -> Self {
        if status.is_server_error() { Self::transient(msg) } else { Self::fatal(msg) }
    }
}

impl std::fmt::Display for DlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Retryable for DlError {
    fn is_retryable(&self) -> bool { self.retryable }
    fn cancelled() -> Self { DlError::fatal("cancelled") }
    fn patience(&self) -> u64 { if self.offline { OFFLINE_PATIENCE } else { 1 } }
}

/// The innermost of `with_causes`' "a: b: c".
fn last_cause(causes: &str) -> &str {
    causes.rsplit_once(": ").map_or(causes, |(_, last)| last)
}

/// `error` with the causes under it, "a: b: c". A failed request's own
/// text names only the request ("error sending request for url ..."); why
/// it failed (a refused connection, an untrusted certificate, a proxy
/// that would not answer) is a cause beneath it.
pub fn with_causes(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(e) = cause {
        let more = e.to_string();
        if !text.contains(&more) {
            text.push_str(": ");
            text.push_str(&more);
        }
        cause = e.source();
    }
    text
}

/// The HTTP client every download site shares. The blocking client
/// applies `timeout` per operation: once for the request headers, then
/// again for each `Read` call on a streamed body, so a slow link still
/// finishes and a stalled one fails within a minute. Whole-body helpers
/// such as `json()` get it as a total, which only the small PyPI
/// metadata query uses.
pub fn http_client(user_agent: &str) -> reqwest::Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(user_agent)
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(60))
        .build()
}

/// Streams chunks into `path`, hashing them on the way; fires progress
/// per percentage point, polls cancel per chunk. The SHA-256 digest of
/// what was written.
fn download_wheel_attempt(
    info: &WheelInfo,
    hooks: &mut DownloadHooks<'_>,
    path: &Path,
) -> Result<Vec<u8>, DlError> {
    let client = http_client(concat!("prunr-runtime-install/", env!("CARGO_PKG_VERSION")))
        .map_err(|e| DlError::fatal(format!("HTTP client: {}", with_causes(&e))))?;
    let response = client.get(&info.url).send().map_err(|e| DlError::request(&e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(DlError::for_status(status, format!("HTTP {status}")));
    }
    let total = response.content_length().unwrap_or(info.size_bytes);

    let mut response = response;
    let file = std::fs::File::create(path).map_err(|e| DlError::fatal(format!("create {}: {e}", path.display())))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut done: u64 = 0;
    let mut last_pct: u64 = u64::MAX;
    loop {
        if hooks.cancel.load(Ordering::Relaxed) {
            return Err(DlError::cancelled());
        }
        let n = match response.read(&mut chunk) {
            Ok(n) => n,
            Err(e) => return Err(DlError::broke_off(&e)),
        };
        if n == 0 { break; }
        hasher.update(&chunk[..n]);
        out.write_all(&chunk[..n]).map_err(|e| DlError::fatal(format!("write {}: {e}", path.display())))?;
        done += n as u64;
        let pct = if total > 0 { done * 100 / total } else { 0 };
        if pct != last_pct {
            last_pct = pct;
            (hooks.progress)(done, total);
        }
    }
    out.flush().map_err(|e| DlError::fatal(format!("write {}: {e}", path.display())))?;
    Ok(hasher.finalize().to_vec())
}

/// Run `attempt` up to `max_attempts` times. Returns immediately on
/// non-retryable errors (per `Retryable::is_retryable`). Between
/// attempts, sleeps an exponentially-growing duration starting at
/// `base_ms` (500 → 1000 → 2000…, times the error's `patience`), polling
/// `cancel` every 50 ms so a user-cancel during backoff fires promptly.
/// The closure gets the 0-based attempt index.
pub fn retry_with_backoff<F, T, E>(
    cancel: &Arc<AtomicBool>,
    max_attempts: u32,
    base_ms: u64,
    mut attempt: F,
) -> Result<T, E>
where
    F: FnMut(u32) -> Result<T, E>,
    E: Retryable,
{
    let mut tries: u32 = 0;
    loop {
        match attempt(tries) {
            Ok(v) => return Ok(v),
            Err(e) if !e.is_retryable() => return Err(e),
            Err(e) if tries + 1 >= max_attempts => return Err(e),
            Err(e) => {
                tries += 1;
                let delay = backoff(base_ms, tries, e.patience());
                tracing::warn!(tries, ?delay, "transient error — retrying");
                let deadline = std::time::Instant::now() + delay;
                while std::time::Instant::now() < deadline {
                    if cancel.load(Ordering::Acquire) {
                        return Err(E::cancelled());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        }
    }
}

/// The wait before try `tries` (1-based): doubling from `base_ms`, times
/// `patience`.
fn backoff(base_ms: u64, tries: u32, patience: u64) -> std::time::Duration {
    std::time::Duration::from_millis(base_ms.saturating_mul(1 << tries.min(6)).saturating_mul(patience))
}

/// An archive's entries that `rename` keeps, written flat into `target_dir`
/// under the names it gives. Caller owns dir creation + safety guards
/// (e.g. refusing non-`runtimes/` paths), which are install-time policy.
///
/// When two entries map to the same target (`libonnxruntime.so` and
/// `libonnxruntime.so.1.24.1`), the first write wins: otherwise a 0-byte
/// symlink-flattened entry could shadow the real dylib.
struct Flat<'a, R> {
    target_dir: &'a Path,
    rename: R,
    written: std::collections::HashSet<String>,
}

impl<'a, R: Fn(&str) -> Option<String>> Flat<'a, R> {
    fn new(target_dir: &'a Path, rename: R) -> Self {
        Self { target_dir, rename, written: std::collections::HashSet::new() }
    }

    fn entry(&mut self, name: &str, entry: &mut impl Read) -> Result<(), String> {
        let Some(target_filename) = (self.rename)(name) else { return Ok(()) };
        if self.written.insert(target_filename.clone()) {
            write_out(entry, &self.target_dir.join(&target_filename))?;
        }
        Ok(())
    }
}

/// `Flat` over a wheel or zip; how many files it wrote.
fn extract_flat(archive: impl Read + Seek, target_dir: &Path, rename: impl Fn(&str) -> Option<String>) -> Result<usize, String> {
    let mut archive = zip::ZipArchive::new(archive).map_err(|e| format!("zip parse: {e}"))?;
    let mut flat = Flat::new(target_dir, rename);
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("zip entry: {e}"))?;
        let name = entry.name().to_string();
        flat.entry(&name, &mut entry)?;
    }
    Ok(flat.written.len())
}

/// `Flat` over a gzip-compressed tar's regular files; how many it wrote.
fn extract_flat_tar_gz(archive: impl Read, target_dir: &Path, rename: impl Fn(&str) -> Option<String>) -> Result<usize, String> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(archive));
    let mut flat = Flat::new(target_dir, rename);
    for entry in archive.entries().map_err(|e| format!("tar parse: {e}"))? {
        let mut entry = entry.map_err(|e| format!("tar entry: {e}"))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path().map_err(|e| format!("tar entry name: {e}"))?.into_owned();
        let Some(name) = path.to_str() else { continue };
        flat.entry(name.trim_start_matches("./"), &mut entry)?;
    }
    Ok(flat.written.len())
}

fn write_out(entry: &mut impl Read, dest: &Path) -> Result<(), String> {
    let file = std::fs::File::create(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    std::io::copy(entry, &mut out)
        .and_then(|_| out.flush())
        .map_err(|e| format!("extract {}: {e}", dest.display()))
}

/// Extract an `onnxruntime-*` wheel into `target_dir`, applying
/// `repackage_target_filename`; the runtime library it holds.
pub fn extract_wheel(archive: impl Read + Seek, target_dir: &Path) -> Result<PathBuf, String> {
    extract_library(archive, target_dir, repackage_target_filename, DYLIB_NAME)
}

/// Extract `addon`'s library from its release `archive` into
/// `target_dir`; the library's path.
pub fn extract_addon(archive: impl Read + Seek, target_dir: &Path, addon: &AddOn) -> Result<PathBuf, String> {
    let keep = |name: &str| addon_file(addon, name);
    match addon.packing {
        Packing::Zip => extract_library(archive, target_dir, keep, addon.library),
        Packing::TarGz => extract_flat_tar_gz(archive, target_dir, keep).and_then(|_| library_in(target_dir, addon.library)),
    }
}

/// What `name` in `addon`'s archive is written as, when it is kept: the
/// library, and with a `folder` each file in it, except a script, the
/// library's copies under its versioned names and the parts left out.
fn addon_file(addon: &AddOn, name: &str) -> Option<String> {
    let Some(folder) = addon.folder else {
        return (name == addon.library).then(|| name.to_string());
    };
    let file = name.strip_prefix(folder)?.strip_prefix('/')?;
    let library_copy = file.strip_prefix(addon.library).is_some_and(|rest| rest.starts_with('.'));
    let left_out = addon.left_out.iter().any(|part| file.starts_with(part));
    (one_segment(file) && !file.ends_with(".py") && !library_copy && !left_out).then(|| file.to_string())
}

/// `archive`'s files that `rename` keeps, flat in `target_dir`; the path
/// of `library` among them, which an archive must hold.
fn extract_library(archive: impl Read + Seek, target_dir: &Path, rename: impl Fn(&str) -> Option<String>, library: &str) -> Result<PathBuf, String> {
    extract_flat(archive, target_dir, rename)?;
    library_in(target_dir, library)
}

fn library_in(target_dir: &Path, library: &str) -> Result<PathBuf, String> {
    let path = target_dir.join(library);
    if !path.is_file() {
        return Err(format!("{library} missing from the download: its layout may have changed"));
    }
    Ok(path)
}

/// Filter + flatten for files extracted from an NVIDIA library wheel
/// (`nvidia/<package>/lib/libX.so.N` on Linux, `nvidia/<package>/bin/
/// X64_N.dll` on Windows): its shared libraries, by file name.
pub fn nvidia_target_filename(zip_name: &str) -> Option<String> {
    let rest = zip_name.strip_prefix("nvidia/")?;
    let (_package, file) = rest.split_once("/lib/").or_else(|| rest.split_once("/bin/"))?;
    let library = file.contains(".so") || file.ends_with(".dll");
    if !one_segment(file) || !library {
        return None;
    }
    // Shipped beside what ORT links but never loaded by it.
    let name = file.trim_start_matches("lib");
    if name.starts_with("nvblas") || name.starts_with("cufftw") || file.contains(".alt.") {
        return None;
    }
    Some(file.to_string())
}

/// Extract an NVIDIA library wheel's shared libraries into `target_dir`,
/// flat; how many were written. Errors when the wheel holds none.
pub fn extract_nvidia_wheel(archive: impl Read + Seek, target_dir: &Path) -> Result<usize, String> {
    match extract_flat(archive, target_dir, nvidia_target_filename)? {
        0 => Err("no libraries in the NVIDIA wheel — its layout may have changed".to_string()),
        written => Ok(written),
    }
}

pub const DYLIB_NAME: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

#[cfg(test)]
mod tests {
    /// A request that got no answer says why in plain words, and only a
    /// network that is down waits longer before the next try.
    #[test]
    fn a_dropped_connection_says_so_and_waits_longer() {
        use super::{backoff, DlError, Retryable};
        let unreachable = DlError::unanswered("github.com", "error sending request: client error (Connect): tcp connect error: Network is unreachable (os error 101)", false, true);
        assert_eq!(unreachable.message, "no internet connection (github.com could not be reached)");
        let refused = DlError::unanswered("github.com", "error sending request: client error (Connect): tcp connect error: Connection refused (os error 111)", false, true);
        assert_eq!(refused.message, "could not reach github.com (Connection refused (os error 111))");
        let timed_out = DlError::unanswered("pypi.org", "operation timed out", true, false);
        assert_eq!(timed_out.message, "pypi.org did not answer in time");
        assert!(unreachable.is_retryable() && refused.is_retryable() && timed_out.is_retryable());
        let waits = |e: &DlError| [1, 2].map(|tries| backoff(500, tries, e.patience()).as_secs());
        assert_eq!(waits(&unreachable), [8, 16]);
        assert_eq!(waits(&refused), [1, 2], "a refusal will not clear by waiting");
        assert_eq!(waits(&timed_out), [1, 2], "a timeout has waited already");
        assert_eq!(waits(&DlError::transient("HTTP 503")), [1, 2]);
    }

    /// A name no resolver knows, asked for for real: the wording and the
    /// wait come from what reqwest says, on any platform.
    #[test]
    fn a_name_that_cannot_be_found_reads_as_no_internet() {
        use super::{DlError, Retryable};
        let client = super::http_client("prunr-test").unwrap();
        let error = client.get("http://prunr-test.invalid/").send().expect_err(".invalid never resolves");
        let error = DlError::request(&error);
        assert_eq!(error.message, "no internet connection (prunr-test.invalid could not be found)");
        assert_eq!(error.patience(), super::OFFLINE_PATIENCE);
    }

    /// An error reads with its causes, each once: a wrapper that repeats
    /// its cause's text does not repeat it again.
    #[test]
    fn an_error_reads_with_its_causes_each_once() {
        #[derive(Debug)]
        struct Failed(&'static str, Option<std::io::Error>);
        impl std::fmt::Display for Failed {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.0) }
        }
        impl std::error::Error for Failed {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { self.1.as_ref().map(|e| e as _) }
        }
        let refused = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection refused");
        assert_eq!(super::with_causes(&Failed("error sending request", Some(refused))), "error sending request: connection refused");
        let repeated = std::io::Error::other("tls: certificate not trusted");
        assert_eq!(super::with_causes(&Failed("tls: certificate not trusted", Some(repeated))), "tls: certificate not trusted");
    }


    use super::*;

    /// `pick_wheel_on` for a host with glibc 2.28, the oldest that runs
    /// ORT's own Linux wheels.
    fn pick_wheel(urls: &[serde_json::Value], rid: &str) -> Result<WheelInfo, String> {
        pick_wheel_on(urls, rid, (2, 28))
    }

    /// Both `release.yml` (the runtime each artifact ships) and `ci.yml`
    /// (Linux install for the e2e test, macOS smoke test) hard-code the same
    /// version this const carries. Cargo doesn't link CI YAML, so drift
    /// would silently ship mismatched bundled-CPU vs runtime-store
    /// runtimes (release) or run e2e tests against a stale runtime
    /// (ci). This test reads both YAMLs and asserts every
    /// `install-runtime onnxruntime[-windowsml] <ver>` invocation matches
    /// the const.
    /// Counts each file's occurrences too — drift in just one of two
    /// release.yml call sites would be silently accepted by a plain
    /// `contains` check.
    #[test]
    fn yaml_pins_match_const() {
        // (path, package, version, expected occurrence count)
        let cases: &[(&str, &str, &str, usize)] = &[
            ("/../../.github/workflows/release.yml", "onnxruntime", PLAIN_ORT_VERSION, 1),           // Linux + macOS staging
            ("/../../.github/workflows/release.yml", "onnxruntime-windowsml", WINDOWS_ML_VERSION, 1), // Windows staging
            ("/../../.github/workflows/ci.yml", "onnxruntime", PLAIN_ORT_VERSION, 2),                // Linux install before tests, macOS smoke test
        ];
        for (rel, package, version, expected) in cases {
            let needle = format!("install-runtime {package} {version}");
            let abs = format!("{}{rel}", env!("CARGO_MANIFEST_DIR"));
            let yml = std::fs::read_to_string(&abs)
                .unwrap_or_else(|e| panic!("read {abs}: {e}"));
            let count = yml.matches(&needle).count();
            assert_eq!(
                count, *expected,
                "{rel}: expected {expected} occurrence(s) of `{needle}`, found {count}. \
                 Did a version bump miss a call site?",
            );
        }
    }
    use serde_json::json;

    #[test]
    fn host_rid_is_supported_or_unknown() {
        let r = host_rid();
        assert!(matches!(r,
            "linux-x64" | "linux-arm64" | "windows-x64" | "windows-arm64"
            | "macos-arm64" | "macos-x64" | "unknown"
        ), "unexpected rid: {r}");
    }

    /// Every 1.24.1 wheel on PyPI, as its JSON lists them.
    const ORT_1_24_1_WHEELS: &[&str] = &[
        "onnxruntime-1.24.1-cp311-cp311-macosx_14_0_arm64.whl",
        "onnxruntime-1.24.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp311-cp311-win_amd64.whl",
        "onnxruntime-1.24.1-cp312-cp312-macosx_14_0_arm64.whl",
        "onnxruntime-1.24.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp312-cp312-win_amd64.whl",
        "onnxruntime-1.24.1-cp313-cp313-macosx_14_0_arm64.whl",
        "onnxruntime-1.24.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp313-cp313t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp313-cp313t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp313-cp313-win_amd64.whl",
        "onnxruntime-1.24.1-cp314-cp314-macosx_14_0_arm64.whl",
        "onnxruntime-1.24.1-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp314-cp314t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "onnxruntime-1.24.1-cp314-cp314t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "onnxruntime-1.24.1-cp314-cp314-win_amd64.whl",
    ];

    fn urls_for(filenames: &[&str]) -> Vec<serde_json::Value> {
        filenames.iter().map(|f| json!({
            "filename": f, "url": format!("https://example.com/{f}"),
            "digests": {"sha256": "aa".repeat(32)}, "size": 0,
        })).collect()
    }

    #[test]
    fn picks_each_platforms_cp313_wheel_from_the_pinned_release() {
        let urls = urls_for(ORT_1_24_1_WHEELS);
        for (rid, expected) in [
            ("linux-x64", "cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
            ("linux-arm64", "cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"),
            ("windows-x64", "cp313-cp313-win_amd64.whl"),
            ("macos-arm64", "cp313-cp313-macosx_14_0_arm64.whl"),
        ] {
            let pick = pick_wheel(&urls, rid).unwrap_or_else(|e| panic!("{rid}: {e}"));
            assert!(pick.url.ends_with(expected), "{rid} picked {}", pick.url);
        }
        // 1.24 dropped Intel macOS and has no Windows-on-Arm wheel.
        assert!(pick_wheel(&urls, "macos-x64").is_err());
        assert!(pick_wheel(&urls, "windows-arm64").is_err());
    }

    /// NVIDIA's wheels carry older manylinux tags than ORT's; any glibc up
    /// to the host's fits (2.28 here), a newer one or another arch does not.
    #[test]
    fn linux_wheels_fit_up_to_the_glibc_ort_needs() {
        for filename in [
            "nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
            "onnxruntime_gpu-1.24.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        ] {
            assert!(pick_wheel(&urls_for(&[filename]), "linux-x64").is_ok(), "rejected {filename}");
        }
        for filename in [
            "x-1.0-py3-none-manylinux_2_34_x86_64.whl",
            "x-1.0-py3-none-manylinux_2_27_aarch64.whl",
            "x-1.0-py3-none-musllinux_1_2_x86_64.whl",
            "x-1.0-py3-none-manylinux_2_x_x86_64.whl",
        ] {
            assert!(pick_wheel(&urls_for(&[filename]), "linux-x64").is_err(), "took {filename}");
        }
        let migraphx = urls_for(&["onnxruntime_migraphx-1.24.2-cp312-cp312-manylinux_2_34_x86_64.whl"]);
        assert!(pick_wheel_on(&migraphx, "linux-x64", (2, 35)).is_ok(), "a 2.35 host runs a 2.34 wheel");
        assert!(pick_wheel_on(&migraphx, "linux-x64", (2, 31)).is_err(), "a 2.31 host does not");
    }

    /// The host's glibc reads as a version on a glibc Linux.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn the_host_glibc_reads_back() {
        assert!(host_glibc() >= (2, 17), "{:?}", host_glibc());
    }

    #[test]
    fn macos_wheels_match_whatever_minimum_os_they_carry() {
        for (rid, filename) in [
            ("macos-arm64", "onnxruntime-1.19.2-cp312-cp312-macosx_11_0_arm64.whl"),
            ("macos-arm64", "onnxruntime-1.31.0-cp313-cp313-macosx_15_0_arm64.whl"),
            ("macos-x64", "onnxruntime-1.23.2-cp313-cp313-macosx_13_0_x86_64.whl"),
            ("macos-x64", "onnxruntime-1.19.2-cp312-cp312-macosx_10_15_x86_64.whl"),
        ] {
            assert!(pick_wheel(&urls_for(&[filename]), rid).is_ok(), "{rid} rejected {filename}");
        }
        for (rid, filename) in [
            ("macos-arm64", "onnxruntime-1.23.2-cp313-cp313-macosx_13_0_x86_64.whl"),
            ("macos-arm64", "onnxruntime-1.24.1-cp313-cp313-win_arm64.whl"),
            ("macos-arm64", "onnxruntime-1.24.1-cp313-cp313-macosx_14_x_arm64.whl"),
            ("macos-x64", "onnxruntime-1.24.1-cp313-cp313-macosx_14_0_arm64.whl"),
        ] {
            assert!(pick_wheel(&urls_for(&[filename]), rid).is_err(), "{rid} took {filename}");
        }
    }

    /// Every rid `host_rid` names has wheels it accepts; without one,
    /// that platform's install fails with "unsupported host platform".
    #[test]
    fn every_supported_rid_accepts_its_own_wheels() {
        for (rid, tag) in [
            ("linux-x64", "manylinux_2_28_x86_64"),
            ("linux-arm64", "manylinux_2_28_aarch64"),
            ("windows-x64", "win_amd64"),
            ("windows-arm64", "win_arm64"),
            ("macos-arm64", "macosx_14_0_arm64"),
            ("macos-x64", "macosx_13_0_x86_64"),
        ] {
            let urls = urls_for(&[&format!("onnxruntime-1.0.0-cp313-cp313-{tag}.whl")]);
            assert!(pick_wheel(&urls, rid).is_ok(), "{rid} accepts no {tag} wheel");
        }
        assert!(pick_wheel(&urls_for(ORT_1_24_1_WHEELS), "unknown").is_err());
    }

    #[test]
    fn install_subdir_format_is_stable() {
        let s = install_subdir("openvino", "1.24.1");
        assert!(s.starts_with("openvino-1.24.1-"));
        assert!(s.ends_with(host_rid()));
    }

    #[test]
    fn validate_subdir_accepts_real_install_subdir_outputs() {
        assert!(validate_subdir("onnxruntime-1.24.1-linux-x64-gnu").is_ok());
        assert!(validate_subdir("openvino-2024.6-osx-arm64").is_ok());
    }

    #[test]
    fn validate_subdir_rejects_traversal_and_separators() {
        assert!(validate_subdir("").is_err(), "empty");
        assert!(validate_subdir(".").is_err(), "dot");
        assert!(validate_subdir("..").is_err(), "double dot");
        assert!(validate_subdir("../etc").is_err(), "leading traversal");
        assert!(validate_subdir("foo/bar").is_err(), "forward slash");
        assert!(validate_subdir("foo\\bar").is_err(), "backslash");
        assert!(validate_subdir("/abs/path").is_err(), "absolute");
    }

    #[test]
    fn pick_wheel_falls_back_to_any_cp_tag() {
        let urls = vec![json!({
            "filename": "ort-cp311-cp311-win_amd64.whl",
            "url": "https://example.com/cp311.whl",
            "digests": {"sha256": "cc".repeat(32)}, "size": 0,
        })];
        let pick = pick_wheel(&urls, "windows-x64").expect("fallback");
        assert_eq!(pick.url, "https://example.com/cp311.whl");
    }

    #[test]
    fn pick_wheel_rejects_when_no_match() {
        let urls = vec![json!({
            "filename": "ort-cp313-cp313-ANDROID.whl",
            "url": "https://example.com/wrong.whl",
            "digests": {"sha256": "dd".repeat(32)}, "size": 0,
        })];
        assert!(pick_wheel(&urls, "windows-x64").is_err());
    }

    /// PyPI's URL list also includes the source distribution
    /// (`onnxruntime-X.Y.Z.tar.gz`). The picker must skip non-`.whl`
    /// entries even if the host_token happens to appear in the
    /// filename — without this guard, `extract_wheel` would later try
    /// to parse a tarball as a zip and fail with a confusing error.
    #[test]
    fn pick_wheel_skips_sdist_even_with_matching_token() {
        let urls = vec![
            json!({"filename": "ort-win_amd64.tar.gz",
                   "url": "https://example.com/sdist.tar.gz",
                   "digests": {"sha256": "ee".repeat(32)}, "size": 0}),
            json!({"filename": "ort-cp313-cp313-win_amd64.whl",
                   "url": "https://example.com/cp313.whl",
                   "digests": {"sha256": "ff".repeat(32)}, "size": 0}),
        ];
        let pick = pick_wheel(&urls, "windows-x64").expect("wheel pick should succeed");
        assert_eq!(pick.url, "https://example.com/cp313.whl",
            "must skip sdist and pick the .whl");
    }

    /// What a wheel's entry is kept as, if at all.
    #[test]
    fn repackage_keeps_the_runtime_libraries_and_nothing_else() {
        for (entry, kept) in [
            // Versioned forms strip to the canonical name; canonical ones pass.
            ("onnxruntime/capi/libonnxruntime.so.1.24.1", Some("libonnxruntime.so")),
            ("onnxruntime/capi/libonnxruntime.1.24.1.dylib", Some("libonnxruntime.dylib")),
            ("onnxruntime/capi/libonnxruntime.so", Some("libonnxruntime.so")),
            ("onnxruntime/capi/libonnxruntime.dylib", Some("libonnxruntime.dylib")),
            // The Windows ML wheel's onnxruntime.dll delay-loads DirectML.dll
            // from its own folder; without it there the system's older copy loads.
            ("onnxruntime/capi/DirectML.dll", Some("DirectML.dll")),
            ("onnxruntime/capi/onnxruntime.dll", Some("onnxruntime.dll")),
            // The CUDA provider and the shared bridge it loads through stay;
            // the TensorRT provider does not.
            ("onnxruntime/capi/libonnxruntime_providers_cuda.so", Some("libonnxruntime_providers_cuda.so")),
            ("onnxruntime/capi/libonnxruntime_providers_shared.so", Some("libonnxruntime_providers_shared.so")),
            ("onnxruntime/capi/libonnxruntime_providers_tensorrt.so", None),
            ("onnxruntime/capi/onnxruntime_pybind11_state.so", None),
            // The Windows GPU wheel, as on PyPI for 1.24.1.
            ("onnxruntime/capi/onnxruntime_providers_cuda.dll", Some("onnxruntime_providers_cuda.dll")),
            ("onnxruntime/capi/onnxruntime_providers_shared.dll", Some("onnxruntime_providers_shared.dll")),
            ("onnxruntime/capi/onnxruntime_providers_tensorrt.dll", None),
            ("onnxruntime/capi/onnxruntime_pybind11_state.pyd", None),
            ("onnxruntime/capi/__init__.py", None),
            ("onnxruntime/__init__.py", None),
            // A separator after `capi/` could escape the target dir: `/` on
            // Unix, `\` on Windows.
            ("onnxruntime/capi/../etc/passwd", None),
            ("onnxruntime/capi/..\\Windows\\evil.dll", None),
            ("onnxruntime/capi/sub\\file.so", None),
        ] {
            assert_eq!(repackage_target_filename(entry).as_deref(), kept, "{entry}");
        }
    }

    /// The Windows release ships the Windows ML runtime (staged as
    /// `yaml_pins_match_const` checks) as its default: both DLLs side by
    /// side in `<exe>/runtime/` in the zip and in the installer.
    #[test]
    fn release_yml_ships_the_windows_ml_runtime_on_windows() {
        let abs = format!("{}/../../.github/workflows/release.yml", env!("CARGO_MANIFEST_DIR"));
        let yml = std::fs::read_to_string(&abs).unwrap_or_else(|e| panic!("read {abs}: {e}"));
        for line in [
            "Copy-Item \"runtime-stage/onnxruntime.dll\" dist/runtime/",
            "Copy-Item \"runtime-stage/DirectML.dll\" dist/runtime/",
            "Source: \"runtime-stage\\onnxruntime.dll\"; DestDir: \"{app}\\runtime\"",
            "Source: \"runtime-stage\\DirectML.dll\"; DestDir: \"{app}\\runtime\"",
        ] {
            assert!(yml.contains(line), "release.yml no longer has `{line}`");
        }
    }

    /// A zip of `files`, each (name, body).
    fn zip_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, body) in files {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    /// An add-on's archive gives up its library and nothing else.
    #[test]
    fn an_addon_archive_gives_up_only_its_library() {
        let zipped = zip_of(&[("onnxruntime_providers_cuda.dll", "dll"), ("version.json", "{}"), ("sub/onnxruntime_providers_cuda.dll", "no")]);
        let dir = tempfile::tempdir().unwrap();
        let library = extract_addon(std::io::Cursor::new(zipped), dir.path(), &CUDA_ADDON_WINDOWS).unwrap();
        assert_eq!(std::fs::read_to_string(library).unwrap(), "dll");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "only the library");
    }

    /// A gzip-compressed tar of `files`, each (name, body).
    fn tar_gz_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast()));
        for (name, body) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o666);
            header.set_cksum();
            tar.append_data(&mut header, name, body.as_bytes()).unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap()
    }

    /// The Linux add-on, a gzip-compressed tar, gives up its library the
    /// same way; one missing from it is said.
    #[test]
    fn a_tar_gz_addon_gives_up_only_its_library() {
        let packed = tar_gz_of(&[("libonnxruntime_providers_cuda.so", "so"), ("version.json", "{}"), ("sub/libonnxruntime_providers_cuda.so", "no")]);
        let dir = tempfile::tempdir().unwrap();
        let library = extract_addon(std::io::Cursor::new(packed), dir.path(), &CUDA_ADDON_LINUX).unwrap();
        assert_eq!(std::fs::read_to_string(library).unwrap(), "so");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "only the library");
        let dotted = tempfile::tempdir().unwrap();
        let packed = tar_gz_of(&[("./libonnxruntime_providers_cuda.so", "first"), ("libonnxruntime_providers_cuda.so", "second")]);
        let library = extract_addon(std::io::Cursor::new(packed), dotted.path(), &CUDA_ADDON_LINUX).unwrap();
        assert_eq!(std::fs::read_to_string(library).unwrap(), "first", "a ./ name is the same file, and the first write wins");
        let empty = tempfile::tempdir().unwrap();
        let without = tar_gz_of(&[("version.json", "{}")]);
        assert!(extract_addon(std::io::Cursor::new(without), empty.path(), &CUDA_ADDON_LINUX).is_err_and(|e| e.contains("missing")));
    }

    /// The OpenVINO add-on's wheel gives up its folder flat, with one copy
    /// of the plugin: not its scripts, its NPU part, its package record,
    /// another folder, or a name that would leave the install.
    #[test]
    fn a_folder_addon_gives_up_its_folder_flat() {
        let wheel = zip_of(&[
            ("onnxruntime_ep_openvino/__init__.py", "py"),
            ("onnxruntime_ep_openvino/libonnxruntime_providers_openvino_plugin.so", "plugin"),
            ("onnxruntime_ep_openvino/libonnxruntime_providers_openvino_plugin.so.1", "copy"),
            ("onnxruntime_ep_openvino/libonnxruntime_providers_openvino_plugin.so.1.8.1", "copy"),
            ("onnxruntime_ep_openvino/libopenvino.so.2641", "openvino"),
            ("onnxruntime_ep_openvino/libtbb.so.12", "tbb"),
            ("onnxruntime_ep_openvino/libopenvino_intel_npu_plugin.so", "npu"),
            ("onnxruntime_ep_openvino/cache.json", "{}"),
            ("onnxruntime_ep_openvino/third-party-programs.txt", "notices"),
            ("onnxruntime_ep_openvino/sub/libopenvino.so.2641", "no"),
            ("onnxruntime_ep_openvino/..", "no"),
            ("onnxruntime_ep_openvino-1.8.1.dist-info/RECORD", "no"),
            ("libonnxruntime_providers_openvino_plugin.so", "no"),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let library = extract_addon(std::io::Cursor::new(wheel), dir.path(), &OPENVINO_ADDON_LINUX).unwrap();
        assert_eq!(std::fs::read_to_string(library).unwrap(), "plugin");
        let mut kept: Vec<String> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        kept.sort();
        assert_eq!(kept, ["cache.json", "libonnxruntime_providers_openvino_plugin.so", "libopenvino.so.2641", "libtbb.so.12", "third-party-programs.txt"]);
    }

    /// The Windows wheel gives up its DLLs the same way, the folder's own
    /// entry and every NPU part left out.
    #[test]
    fn the_windows_openvino_wheel_gives_up_its_dlls() {
        let wheel = zip_of(&[
            ("onnxruntime_ep_openvino/", ""),
            ("onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll", "plugin"),
            ("onnxruntime_ep_openvino/openvino.dll", "openvino"),
            ("onnxruntime_ep_openvino/tbb12.dll", "tbb"),
            ("onnxruntime_ep_openvino/openvino_runtime.manifest", "manifest"),
            ("onnxruntime_ep_openvino/openvino_intel_npu_compiler.dll", "npu"),
            ("onnxruntime_ep_openvino/openvino_intel_npu_plugin.dll", "npu"),
            ("onnxruntime_ep_openvino/__init__.py", "py"),
            ("onnxruntime_ep_openvino-1.8.1.dist-info/RECORD", "no"),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let library = extract_addon(std::io::Cursor::new(wheel), dir.path(), &OPENVINO_ADDON_WINDOWS).unwrap();
        assert_eq!(std::fs::read_to_string(library).unwrap(), "plugin");
        let mut kept: Vec<String> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        kept.sort();
        assert_eq!(kept, ["onnxruntime_providers_openvino_plugin.dll", "openvino.dll", "openvino_runtime.manifest", "tbb12.dll"]);
    }

    /// An NVIDIA wheel gives up its shared libraries by file name, and
    /// nothing else: headers, Python files and traversal names stay out.
    #[test]
    fn nvidia_wheels_give_up_only_their_libraries() {
        assert_eq!(nvidia_target_filename("nvidia/cudnn/lib/libcudnn.so.9").as_deref(), Some("libcudnn.so.9"));
        assert_eq!(nvidia_target_filename("nvidia/cublas/lib/libcublasLt.so.12").as_deref(), Some("libcublasLt.so.12"));
        assert!(nvidia_target_filename("nvidia/cudnn/include/cudnn.h").is_none());
        assert!(nvidia_target_filename("nvidia/cudnn/lib/__init__.py").is_none());
        assert!(nvidia_target_filename("nvidia/cudnn/lib/../../evil.so").is_none());
        assert!(nvidia_target_filename("nvidia/cudnn/lib/..\\evil.so").is_none());
        assert!(nvidia_target_filename("onnxruntime/capi/libonnxruntime.so").is_none());
        for unused in ["libnvblas.so.12", "libcufftw.so.11", "libnvrtc.alt.so.12", "libnvrtc-builtins.alt.so.12.9"] {
            assert!(nvidia_target_filename(&format!("nvidia/x/lib/{unused}")).is_none(), "{unused}");
        }
    }

    /// The Windows wheels keep their DLLs under `bin/` (as on PyPI for the
    /// pinned versions); the same ones are left out as on Linux.
    #[test]
    fn nvidia_windows_wheels_give_up_their_dlls() {
        for (entry, kept) in [
            ("nvidia/cudnn/bin/cudnn64_9.dll", Some("cudnn64_9.dll")),
            ("nvidia/cudnn/bin/cudnn_engines_precompiled64_9.dll", Some("cudnn_engines_precompiled64_9.dll")),
            ("nvidia/cublas/bin/cublasLt64_12.dll", Some("cublasLt64_12.dll")),
            ("nvidia/cuda_runtime/bin/cudart64_12.dll", Some("cudart64_12.dll")),
            ("nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll", Some("nvrtc64_120_0.dll")),
            ("nvidia/cuda_nvrtc/bin/nvrtc-builtins64_129.dll", Some("nvrtc-builtins64_129.dll")),
            ("nvidia/cublas/bin/nvblas64_12.dll", None),
            ("nvidia/cufft/bin/cufftw64_11.dll", None),
            ("nvidia/cuda_nvrtc/bin/nvrtc64_120_0.alt.dll", None),
            ("nvidia/cudnn/bin/..\\evil.dll", None),
            ("nvidia/cudnn/include/cudnn.h", None),
        ] {
            assert_eq!(nvidia_target_filename(entry).as_deref(), kept, "{entry}");
        }
    }

    /// Extraction writes the libraries flat and refuses a wheel without any.
    #[test]
    fn an_nvidia_wheel_extracts_flat_and_one_without_libraries_fails() {
        let wheel = |files: &[&str]| zip_of(&files.iter().map(|f| (*f, "elf")).collect::<Vec<_>>());
        let dir = tempfile::tempdir().unwrap();
        let n = extract_nvidia_wheel(std::io::Cursor::new(wheel(&["nvidia/curand/lib/libcurand.so.10", "nvidia/curand/include/curand.h"])), dir.path()).unwrap();
        assert_eq!(n, 1);
        assert_eq!(std::fs::read(dir.path().join("libcurand.so.10")).unwrap(), b"elf");
        assert!(extract_nvidia_wheel(std::io::Cursor::new(wheel(&["nvidia/curand/include/curand.h"])), dir.path()).is_err());
    }

    #[test]
    fn verify_digest_accepts_correct_digest() {
        let digest = Sha256::digest(b"hello world");
        assert!(verify_digest(&digest, &hex::encode(digest)).is_ok());
        assert!(verify_digest(&digest, &hex::encode(digest).to_uppercase()).is_ok());
    }

    #[test]
    fn verify_digest_rejects_wrong_digest() {
        assert!(verify_digest(&Sha256::digest(b"hello world"), &"00".repeat(32)).is_err());
    }

    #[test]
    fn retry_succeeds_on_second_attempt() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut calls = 0;
        let r: Result<u32, DlError> = retry_with_backoff(&cancel, 3, 1, |_attempt| {
            calls += 1;
            if calls == 1 { Err(DlError::transient("flaky")) } else { Ok(42) }
        });
        assert_eq!(r.expect("eventual success"), 42);
        assert_eq!(calls, 2);
    }

    #[test]
    fn only_a_server_error_status_is_worth_retrying() {
        use reqwest::StatusCode;
        for (status, retry) in [
            (StatusCode::INTERNAL_SERVER_ERROR, true),
            (StatusCode::SERVICE_UNAVAILABLE, true),
            (StatusCode::NOT_FOUND, false),
            (StatusCode::FORBIDDEN, false),
            (StatusCode::MOVED_PERMANENTLY, false),
        ] {
            let e = DlError::for_status(status, format!("HTTP {status}"));
            assert_eq!(e.is_retryable(), retry, "{status}");
            assert_eq!(e.to_string(), format!("HTTP {status}"));
        }
    }

    #[test]
    fn retry_gives_up_after_max_attempts() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut calls = 0;
        let r: Result<(), DlError> = retry_with_backoff(&cancel, 3, 1, |_attempt| {
            calls += 1;
            Err(DlError::transient("flaky"))
        });
        assert!(r.is_err());
        assert_eq!(calls, 3);
    }

    #[test]
    fn retry_skips_remaining_attempts_on_fatal() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut calls = 0;
        let r: Result<(), DlError> = retry_with_backoff(&cancel, 5, 1, |_attempt| {
            calls += 1;
            Err(DlError::fatal("404"))
        });
        assert!(r.is_err());
        assert_eq!(calls, 1, "fatal errors must not retry");
    }

    #[test]
    fn retry_honours_cancel_during_backoff() {
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_inner = cancel.clone();
        let mut calls = 0;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(80));
            cancel_inner.store(true, Ordering::Release);
        });
        let r: Result<(), DlError> = retry_with_backoff(&cancel, 5, 500, |_attempt| {
            calls += 1;
            Err(DlError::transient("flaky"))
        });
        let err = r.expect_err("cancelled retry should error");
        assert_eq!(
            err.message, "cancelled",
            "cancelled retry must surface the canonical \"cancelled\" message — got {:?}",
            err.message,
        );
        assert!(calls < 5, "cancel during backoff must short-circuit");
    }
}
