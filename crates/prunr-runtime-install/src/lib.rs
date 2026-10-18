//! Shared host-detection + PyPI-wheel install logic for ORT runtimes.
//!
//! Used by `xtask install-runtime` (build-side CLI) and the GUI Settings →
//! Hardware → Install button.

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// Single source of truth for the ONNX Runtime version we ship + offer
/// for runtime-store install. The pykeio/ort `api-23` feature refuses
/// a runtime older than 1.23 at load. The macOS wheel's minimum OS is
/// the Mac app's floor (`LSMinimumSystemVersion`, 14.0 for 1.24), so a
/// bump checks that too. Drift between the bundled runtime and a
/// runtime-store upgrade would surprise a user upgrading via
/// Settings → Hardware.
///
/// **Upper bound:** pykeio/ort tracks ABI v23 across ORT 1.x. ORT 2.0+
/// is expected to bump the ABI; bumping this past the 1.x line without
/// switching the `ort` feature to `api-24` (or whatever the new ABI
/// becomes) would compile but crash at session creation. Run
/// `cargo xtask probe-load-dynamic <new-dylib>` before the bump to
/// confirm — see CLAUDE.md `## Verify before bundling / wiring`.
pub const PINNED_ORT_VERSION: &str = "1.24.1";

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

/// The newest glibc a Linux wheel may need: the one ORT's own wheels
/// need, so anything that runs beside the runtime runs.
const MAX_GLIBC: (u32, u32) = (2, 28);

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
/// only its arch is fixed. A Linux tag may need any glibc up to
/// `MAX_GLIBC` (NVIDIA's wheels carry 2.12 to 2.27).
fn tag_fits(tag: &str, rid: &str) -> bool {
    let number = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let macos = |arch: &str| tag.strip_prefix("macosx_")
        .and_then(|t| t.strip_suffix(arch))
        .and_then(|t| t.strip_suffix('_'))
        .and_then(|version| version.split_once('_'))
        .is_some_and(|(major, minor)| number(major) && number(minor));
    match rid {
        "linux-x64" => manylinux_glibc(tag, "x86_64").is_some_and(|v| v <= MAX_GLIBC),
        "linux-arm64" => manylinux_glibc(tag, "aarch64").is_some_and(|v| v <= MAX_GLIBC),
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
/// as a single path segment. Defense-in-depth: callers should already
/// be passing a value produced by `install_subdir`, but a future API
/// taking a user-supplied `--target` flag could otherwise traverse out.
pub fn validate_subdir(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
    {
        return Err(format!("invalid runtime subdirectory name: {name:?}"));
    }
    Ok(())
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
    pick_wheel(urls, host_rid())
}

fn pick_wheel(urls: &[serde_json::Value], rid: &str) -> Result<WheelInfo, String> {
    let pick = |require_cp313: bool| urls.iter().find_map(|u| {
        let name = u["filename"].as_str()?;
        // Reject sdists / non-wheel artifacts up front so a future PyPI
        // release that includes a wheel-flavour token in a `.tar.gz`
        // filename can't slip through to `extract_wheel` (which would
        // fail at `ZipArchive::new` with a confusing error).
        if !name.ends_with(".whl") { return None; }
        if !platform_tags(name).any(|tag| tag_fits(tag, rid)) { return None; }
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
/// Path-traversal guard: the result is used as a single filename joined
/// onto `target_dir`. Reject anything containing `/` or `\\` so a
/// maliciously-crafted wheel can't escape `target_dir` via
/// `..\Windows\System32\evil.dll` on Windows or its forward-slash
/// equivalent on Unix.
pub fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if stripped.is_empty() || stripped.contains('/') || stripped.contains('\\') {
        return None;
    }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
    // The GPU wheel's TensorRT provider: no session registers it.
    if stripped.starts_with("libonnxruntime_providers_tensorrt") { return None; }
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

pub fn verify_sha256(bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex::encode(Sha256::digest(bytes));
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
/// The SHA verification happens INSIDE the attempt (not after `download_wheel`
/// returns). A transient corruption — CDN edge cache delivering a partial
/// payload, mid-flight bit flip, MITM-injected garbage — would otherwise
/// fail fast with one shot at the network; folding it inside lets the
/// retry loop give it ~3 chances before escalating.
pub fn download_wheel(info: &WheelInfo, hooks: &mut DownloadHooks<'_>) -> Result<bytes::Bytes, String> {
    let cancel = Arc::clone(&hooks.cancel);
    retry_with_backoff(&cancel, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS, |attempt| {
        if attempt > 0 {
            tracing::info!(attempt, url = %info.url, "retrying wheel download");
        }
        let bytes = download_wheel_attempt(info, hooks)?;
        if let Err(msg) = verify_sha256(&bytes, &info.sha256) {
            return Err(DlError::transient(msg));
        }
        Ok(bytes)
    }).map_err(|e| e.message)
}

/// Implemented by error types whose values distinguish transient
/// errors from fatal ones; each implementor decides which is which.
/// `retry_with_backoff` consults `is_retryable` to decide between retry
/// and fail-fast; on cancel during the backoff sleep it constructs a
/// fresh value via `cancelled()` so each error type picks its own
/// user-facing wording.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
    fn cancelled() -> Self;
}

/// A failed download attempt: what to tell the user, and whether another
/// attempt could succeed. Every downloader (runtime wheels, on-demand
/// models, xtask's model fetch) shares it.
#[derive(Debug)]
pub struct DlError {
    pub message: String,
    retryable: bool,
}

impl DlError {
    pub fn fatal(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: false } }
    pub fn transient(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: true } }

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

/// Single download attempt — used by the retry wrapper. Streams chunks,
/// fires progress per percentage point, polls cancel per chunk.
/// `Content-Length` not pre-allocated (untrusted; bogus 9999999999 would
/// OOM). `Vec` grows; resize cost on 80 MB is microseconds.
fn download_wheel_attempt(
    info: &WheelInfo,
    hooks: &mut DownloadHooks<'_>,
) -> Result<bytes::Bytes, DlError> {
    let client = http_client(concat!("prunr-runtime-install/", env!("CARGO_PKG_VERSION")))
        .map_err(|e| DlError::fatal(format!("HTTP client: {e}")))?;
    let response = client.get(&info.url).send()
        .map_err(|e| DlError::transient(format!("connect: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(DlError::for_status(status, format!("HTTP {status}")));
    }
    let total = response.content_length().unwrap_or(info.size_bytes);

    let mut response = response;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut last_pct: u64 = u64::MAX;
    loop {
        if hooks.cancel.load(Ordering::Relaxed) {
            return Err(DlError::cancelled());
        }
        let n = match response.read(&mut chunk) {
            Ok(n) => n,
            // Partial-stream errors (broken pipe, timeout) are transient.
            Err(e) => return Err(DlError::transient(format!("read: {e}"))),
        };
        if n == 0 { break; }
        buf.extend_from_slice(&chunk[..n]);
        let pct = if total > 0 { buf.len() as u64 * 100 / total } else { 0 };
        if pct != last_pct {
            last_pct = pct;
            (hooks.progress)(buf.len() as u64, total);
        }
    }
    Ok(bytes::Bytes::from(buf))
}

/// Run `attempt` up to `max_attempts` times. Returns immediately on
/// non-retryable errors (per `Retryable::is_retryable`). Between
/// attempts, sleeps an exponentially-growing duration starting at
/// `base_ms` (500 → 1000 → 2000…), polling `cancel` every 50 ms so a
/// user-cancel during backoff fires promptly. The closure receives the
/// current attempt index (0-based) for callers that want to log which
/// retry round triggered.
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
            Err(_e) => {
                tries += 1;
                let delay = std::time::Duration::from_millis(base_ms.saturating_mul(1 << tries.min(6)));
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

/// Extract a wheel into `target_dir`, applying `repackage_target_filename`.
/// Caller owns dir creation + safety guards (e.g. refusing
/// non-`runtimes/` paths) — that's install-time policy.
///
/// Skip-on-collision: when two archive entries map to the same
/// canonical target (e.g. an `onnxruntime-openvino` wheel that ships
/// both `libonnxruntime.so` and `libonnxruntime.so.1.24.1` — the
/// latter via the version-strip arm in `repackage_target_filename`),
/// keep the first write and skip subsequent ones. Without this guard,
/// a 0-byte symlink-flattened entry could shadow the real dylib if it
/// happened to come second in archive order.
pub fn extract_wheel(bytes: &[u8], target_dir: &std::path::Path) -> Result<PathBuf, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("zip parse: {e}"))?;
    let mut written: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| format!("zip entry: {e}"))?;
        let name = entry.name().to_string();
        let Some(target_filename) = repackage_target_filename(&name) else {
            continue;
        };
        if !written.insert(target_filename.clone()) {
            // Already wrote this canonical target — keep the first one,
            // skip the duplicate.
            continue;
        }
        let dest = target_dir.join(&target_filename);
        let mut out = std::fs::File::create(&dest)
            .map_err(|e| format!("create {}: {e}", dest.display()))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("extract {}: {e}", dest.display()))?;
    }

    let dylib = target_dir.join(DYLIB_NAME);
    if !dylib.is_file() {
        return Err(format!(
            "{DYLIB_NAME} missing after extract — wheel layout may have changed",
        ));
    }
    Ok(target_dir.to_path_buf())
}

/// Filter + flatten for files extracted from an NVIDIA library wheel
/// (`nvidia/<package>/lib/libX.so.N`): its shared libraries, by file
/// name. The same path-traversal guard as `repackage_target_filename`.
pub fn nvidia_target_filename(zip_name: &str) -> Option<String> {
    let rest = zip_name.strip_prefix("nvidia/")?;
    let (_package, file) = rest.split_once("/lib/")?;
    if file.is_empty() || file.contains('/') || file.contains('\\') || !file.contains(".so") {
        return None;
    }
    // Shipped beside what ORT links but never loaded by it.
    if file.starts_with("libnvblas") || file.starts_with("libcufftw") || file.contains(".alt.") {
        return None;
    }
    Some(file.to_string())
}

/// Extract an NVIDIA library wheel's shared libraries into `target_dir`,
/// flat; how many were written. Errors when the wheel holds none.
pub fn extract_nvidia_wheel(bytes: &[u8], target_dir: &std::path::Path) -> Result<usize, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| format!("zip parse: {e}"))?;
    let mut written = 0;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("zip entry: {e}"))?;
        let Some(file) = nvidia_target_filename(entry.name()) else { continue };
        let dest = target_dir.join(&file);
        let mut out = std::fs::File::create(&dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
        std::io::copy(&mut entry, &mut out).map_err(|e| format!("extract {}: {e}", dest.display()))?;
        written += 1;
    }
    if written == 0 {
        return Err("no libraries in the NVIDIA wheel — its layout may have changed".to_string());
    }
    Ok(written)
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
    use super::*;

    /// Both `release.yml` (the runtime each artifact ships) and `ci.yml`
    /// (Linux install for the e2e test, macOS smoke test) hard-code the same
    /// version this const carries. Cargo doesn't link CI YAML, so drift
    /// would silently ship mismatched bundled-CPU vs runtime-store
    /// runtimes (release) or run e2e tests against a stale runtime
    /// (ci). This test reads both YAMLs and asserts every
    /// `install-runtime onnxruntime[-directml] <ver>` invocation matches
    /// the const.
    /// Counts each file's occurrences too — drift in just one of two
    /// release.yml call sites would be silently accepted by a plain
    /// `contains` check.
    #[test]
    fn yaml_pins_match_const() {
        // (path, package, expected occurrence count)
        let cases: &[(&str, &str, usize)] = &[
            ("/../../.github/workflows/release.yml", "onnxruntime", 1),          // Linux + macOS staging
            ("/../../.github/workflows/release.yml", "onnxruntime-directml", 1), // Windows staging
            ("/../../.github/workflows/ci.yml", "onnxruntime", 2),               // Linux install before tests, macOS smoke test
        ];
        for (rel, package, expected) in cases {
            let needle = format!("install-runtime {package} {PINNED_ORT_VERSION}");
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
    /// to 2.28 fits, a newer one or another arch does not.
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
    fn pick_wheel_prefers_cp313() {
        let urls = vec![
            json!({"filename": "ort-cp310-cp310-win_amd64.whl",
                   "url": "https://example.com/cp310.whl",
                   "digests": {"sha256": "aa".repeat(32)}, "size": 80_000_000}),
            json!({"filename": "ort-cp313-cp313-win_amd64.whl",
                   "url": "https://example.com/cp313.whl",
                   "digests": {"sha256": "bb".repeat(32)}, "size": 80_000_000}),
        ];
        let pick = pick_wheel(&urls, "windows-x64").expect("pick");
        assert_eq!(pick.url, "https://example.com/cp313.whl");
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

    #[test]
    fn repackage_keeps_capi_dylib() {
        // Versioned forms strip to the canonical name.
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.so.1.24.1").as_deref(),
            Some("libonnxruntime.so"),
        );
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.1.24.1.dylib").as_deref(),
            Some("libonnxruntime.dylib"),
        );
        // Canonical forms pass through unchanged.
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.so").as_deref(),
            Some("libonnxruntime.so"),
        );
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.dylib").as_deref(),
            Some("libonnxruntime.dylib"),
        );
    }

    /// The Windows release ships the DirectML runtime as its default:
    /// both DLLs staged from the DirectML wheel, side by side in
    /// `<exe>/runtime/` in the zip and in the installer.
    #[test]
    fn release_yml_ships_the_directml_runtime_on_windows() {
        let abs = format!("{}/../../.github/workflows/release.yml", env!("CARGO_MANIFEST_DIR"));
        let yml = std::fs::read_to_string(&abs).unwrap_or_else(|e| panic!("read {abs}: {e}"));
        for line in [
            format!("cargo xtask install-runtime onnxruntime-directml {PINNED_ORT_VERSION} --stage-to runtime-stage"),
            "Copy-Item \"runtime-stage/onnxruntime.dll\" dist/runtime/".to_string(),
            "Copy-Item \"runtime-stage/DirectML.dll\" dist/runtime/".to_string(),
            "Source: \"runtime-stage\\onnxruntime.dll\"; DestDir: \"{app}\\runtime\"".to_string(),
            "Source: \"runtime-stage\\DirectML.dll\"; DestDir: \"{app}\\runtime\"".to_string(),
        ] {
            assert!(yml.contains(&line), "release.yml no longer has `{line}`");
        }
    }

    /// The DirectML wheel's `onnxruntime.dll` delay-loads `DirectML.dll`
    /// from its own folder; without it there the system's older copy loads.
    #[test]
    fn repackage_keeps_the_directml_dll_beside_the_runtime() {
        for name in ["DirectML.dll", "onnxruntime.dll"] {
            assert_eq!(
                repackage_target_filename(&format!("onnxruntime/capi/{name}")).as_deref(),
                Some(name),
            );
        }
    }

    /// The CUDA wheel keeps its CUDA provider and the shared bridge it
    /// loads through; the TensorRT provider stays behind.
    #[test]
    fn repackage_keeps_the_cuda_provider_and_drops_tensorrt() {
        for name in ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"] {
            assert_eq!(repackage_target_filename(&format!("onnxruntime/capi/{name}")).as_deref(), Some(name));
        }
        assert!(repackage_target_filename("onnxruntime/capi/libonnxruntime_providers_tensorrt.so").is_none());
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

    /// Extraction writes the libraries flat and refuses a wheel without any.
    #[test]
    fn an_nvidia_wheel_extracts_flat_and_one_without_libraries_fails() {
        use std::io::Write;
        let wheel = |files: &[&str]| {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            for f in files {
                zip.start_file(*f, zip::write::SimpleFileOptions::default()).unwrap();
                zip.write_all(b"elf").unwrap();
            }
            zip.finish().unwrap().into_inner()
        };
        let dir = tempfile::tempdir().unwrap();
        let n = extract_nvidia_wheel(&wheel(&["nvidia/curand/lib/libcurand.so.10", "nvidia/curand/include/curand.h"]), dir.path()).unwrap();
        assert_eq!(n, 1);
        assert_eq!(std::fs::read(dir.path().join("libcurand.so.10")).unwrap(), b"elf");
        assert!(extract_nvidia_wheel(&wheel(&["nvidia/curand/include/curand.h"]), dir.path()).is_err());
    }

    #[test]
    fn repackage_drops_python_files_and_pybind() {
        assert!(repackage_target_filename("onnxruntime/capi/onnxruntime_pybind11_state.so").is_none());
        assert!(repackage_target_filename("onnxruntime/capi/__init__.py").is_none());
        assert!(repackage_target_filename("onnxruntime/__init__.py").is_none());
    }

    /// Path-traversal guard: any embedded `/` or `\\` after the
    /// `onnxruntime/capi/` prefix must be rejected so a maliciously
    /// crafted wheel can't escape `target_dir`. The forward-slash check
    /// covers Unix; the backslash check covers Windows.
    #[test]
    fn repackage_rejects_path_traversal_in_entry_name() {
        // forward slash (Unix-style traversal)
        assert!(repackage_target_filename("onnxruntime/capi/../etc/passwd").is_none());
        // backslash (Windows-style traversal)
        assert!(repackage_target_filename("onnxruntime/capi/..\\Windows\\evil.dll").is_none());
        assert!(repackage_target_filename("onnxruntime/capi/sub\\file.so").is_none());
        // sanity: the canonical no-separator forms still pass
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.so"),
            Some("libonnxruntime.so".to_string()),
        );
    }

    #[test]
    fn verify_sha256_accepts_correct_digest() {
        let payload = b"hello world";
        let expected = hex::encode(Sha256::digest(payload));
        assert!(verify_sha256(payload, &expected).is_ok());
    }

    #[test]
    fn verify_sha256_rejects_wrong_digest() {
        assert!(verify_sha256(b"hello world", &"00".repeat(32)).is_err());
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
