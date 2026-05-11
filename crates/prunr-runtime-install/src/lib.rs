//! Shared host-detection + PyPI-wheel install logic for ORT runtimes.
//!
//! Used by `xtask install-runtime` (build-side CLI) and the GUI Settings →
//! Hardware → Install button. Before this crate, both sites independently
//! implemented `host_rid`, wheel-name token mapping, SHA256 verification,
//! and wheel repackaging — divergence on the next OpenVINO release would
//! have silently installed wrong files in one path.

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// Single source of truth for the ONNX Runtime version we ship + offer
/// for runtime-store install. The pykeio/ort `api-23` feature is
/// forward-compatible across ORT 1.17+ so this isn't load-bearing for
/// linkage; it IS load-bearing for "the bundled CPU runtime and the
/// runtime-store OpenVINO upgrade are at matching versions" — drift
/// would surprise a user upgrading via Settings → Hardware.
///
/// **Upper bound:** pykeio/ort tracks ABI v23 across ORT 1.x. ORT 2.0+
/// is expected to bump the ABI; bumping this past the 1.x line without
/// switching the `ort` feature to `api-24` (or whatever the new ABI
/// becomes) would compile but crash at session creation. Run
/// `cargo xtask probe-load-dynamic <new-dylib>` before the bump to
/// confirm — see CLAUDE.md `## Verify before bundling / wiring`.
pub const PINNED_ORT_VERSION: &str = "1.24.1";

/// Version of ONNX Runtime we build from source on macOS to enable the
/// CoreML EP (no PyPI wheel ships this). Diverges from
/// `PINNED_ORT_VERSION` on purpose: the CPU-only runtime ships at a
/// newer release while the CoreML build is held back to the last
/// version we have a known-good CMake recipe for. Mirrored in
/// `.github/workflows/release.yml` as the `MACOS_ORT_VERSION` workflow
/// env var; the `release_yml_macos_ort_matches_const` test asserts they
/// stay in sync.
pub const MACOS_CORE_ML_ORT_VERSION: &str = "1.20.0";

/// Host runtime identifier — `linux-x64`, `macos-arm64`, etc. `unknown`
/// for unsupported platforms; `host_pypi_token` rejects those explicitly.
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

/// PyPI wheel filename token for the host platform.
pub fn host_pypi_token() -> Option<&'static str> {
    Some(match host_rid() {
        "linux-x64" => "manylinux_2_28_x86_64",
        "linux-arm64" => "manylinux_2_28_aarch64",
        "windows-x64" => "win_amd64",
        "windows-arm64" => "win_arm64",
        "macos-arm64" => "macosx_11_0_arm64",
        // Intel Mac. PyPI's onnxruntime wheels use the 10.15 deployment
        // target on x86_64 (the `macosx_10_15_x86_64` filename token);
        // `host_rid` already returns `"macos-x64"` for this case, so the
        // arm has to be present here for the GUI install button + xtask
        // CLI to actually find a wheel instead of bailing with
        // "unsupported host platform".
        "macos-x64" => "macosx_10_15_x86_64",
        _ => return None,
    })
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
    let host_token = host_pypi_token().ok_or_else(||
        format!("unsupported host platform `{}`", host_rid())
    )?;
    let pick = |require_cp313: bool| urls.iter().find_map(|u| {
        let name = u["filename"].as_str()?;
        // Reject sdists / non-wheel artifacts up front so a future PyPI
        // release that includes a wheel-flavour token in a `.tar.gz`
        // filename can't slip through to `extract_wheel` (which would
        // fail at `ZipArchive::new` with a confusing error).
        if !name.ends_with(".whl") { return None; }
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

/// Filter + rename for files extracted from an `onnxruntime-*` wheel.
///
/// Path-traversal guard: the result is used as a single filename joined
/// onto `target_dir`. Reject anything containing `/` or `\\` so a
/// maliciously-crafted wheel can't escape `target_dir` via
/// `..\Windows\System32\evil.dll` on Windows or its forward-slash
/// equivalent on Unix.
pub fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if stripped.contains('/') || stripped.contains('\\') { return None; }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
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

/// Wraps `download_wheel_attempt` in up to 3 tries with exponential
/// backoff. Transient errors (network timeout, 5xx, broken connection,
/// SHA256 mismatch) retry; fatal errors (404, user cancel) fail fast.
/// Mirrors the same policy as the model `download_manager` so users see
/// consistent retry behaviour across runtime + model installs.
///
/// The SHA verification happens INSIDE the attempt (not after `download_wheel`
/// returns). A transient corruption — CDN edge cache delivering a partial
/// payload, mid-flight bit flip, MITM-injected garbage — would otherwise
/// fail fast with one shot at the network; folding it inside lets the
/// retry loop give it ~3 chances before escalating.
pub fn download_wheel(info: &WheelInfo, hooks: &mut DownloadHooks<'_>) -> Result<bytes::Bytes, String> {
    let cancel = Arc::clone(&hooks.cancel);
    retry_with_backoff(&cancel, 3, 500, |attempt| {
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
/// (network blip, 5xx) from fatal (4xx, sha mismatch, user cancel).
/// `retry_with_backoff` consults `is_retryable` to decide between retry
/// and fail-fast; on cancel during the backoff sleep it constructs a
/// fresh value via `cancelled()` so each error type picks its own
/// user-facing wording.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
    fn cancelled() -> Self;
}

#[derive(Debug)]
struct DlError {
    message: String,
    retryable: bool,
}

impl DlError {
    fn fatal(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: false } }
    fn transient(msg: impl Into<String>) -> Self { Self { message: msg.into(), retryable: true } }
}

impl Retryable for DlError {
    fn is_retryable(&self) -> bool { self.retryable }
    fn cancelled() -> Self { DlError::fatal("cancelled") }
}

/// Single download attempt — used by the retry wrapper. Streams chunks,
/// fires progress per percentage point, polls cancel per chunk.
/// `Content-Length` not pre-allocated (untrusted; bogus 9999999999 would
/// OOM). `Vec` grows; resize cost on 80 MB is microseconds.
fn download_wheel_attempt(
    info: &WheelInfo,
    hooks: &mut DownloadHooks<'_>,
) -> Result<bytes::Bytes, DlError> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("prunr-runtime-install/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| DlError::fatal(format!("HTTP client: {e}")))?;
    let response = client.get(&info.url).send()
        .map_err(|e| DlError::transient(format!("connect: {e}")))?;

    // 4xx is the user/server contract being wrong (bad URL, gone wheel) —
    // retrying won't help. 5xx is server transient. Anything not 2xx is
    // an error at this stage.
    let status = response.status();
    if !status.is_success() {
        let msg = format!("HTTP {status}");
        return Err(if status.is_server_error() { DlError::transient(msg) } else { DlError::fatal(msg) });
    }
    let total = response.content_length().unwrap_or(info.size_bytes);

    let mut response = response;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut last_pct: u64 = u64::MAX;
    loop {
        if hooks.cancel.load(Ordering::Relaxed) {
            return Err(DlError::fatal("cancelled"));
        }
        let n = match response.read(&mut chunk) {
            Ok(n) => n,
            // Partial-stream errors (broken pipe, timeout) are transient.
            Err(e) => return Err(DlError::transient(format!("read: {e}"))),
        };
        if n == 0 { break; }
        buf.extend_from_slice(&chunk[..n]);
        let pct = (buf.len() as u64 * 100).checked_div(total).unwrap_or(0);
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

    let dylib = target_dir.join(dylib_name());
    if !dylib.is_file() {
        return Err(format!(
            "{} missing after extract — wheel layout may have changed",
            dylib_name(),
        ));
    }
    Ok(target_dir.to_path_buf())
}

pub fn dylib_name() -> &'static str {
    if cfg!(windows) { "onnxruntime.dll" }
    else if cfg!(target_os = "macos") { "libonnxruntime.dylib" }
    else { "libonnxruntime.so" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both `release.yml` (Linux + Windows staging steps) and `ci.yml`
    /// (Linux ORT runtime install for the e2e test) hard-code the same
    /// version this const carries. Cargo doesn't link CI YAML, so drift
    /// would silently ship mismatched bundled-CPU vs runtime-store
    /// runtimes (release) or run e2e tests against a stale runtime
    /// (ci). This test reads both YAMLs and asserts every
    /// `install-runtime onnxruntime <ver>` invocation matches the const.
    /// Counts each file's occurrences too — drift in just one of two
    /// release.yml call sites would be silently accepted by a plain
    /// `contains` check.
    #[test]
    fn yaml_pins_match_const() {
        // (path, expected occurrence count)
        let cases: &[(&str, usize)] = &[
            ("/../../.github/workflows/release.yml", 2), // Linux + Windows staging
            ("/../../.github/workflows/ci.yml",      1), // Linux ORT install before tests
        ];
        let needle = format!("install-runtime onnxruntime {}", PINNED_ORT_VERSION);
        for (rel, expected) in cases {
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
    /// `release.yml` builds a custom ORT with CoreML on macOS — the
    /// version is the workflow env var `MACOS_ORT_VERSION`, used to
    /// drive the cache key + the `git clone --branch v<ver>` step.
    /// Bumping `MACOS_CORE_ML_ORT_VERSION` here without bumping the
    /// YAML (or vice versa) silently produces release artifacts at the
    /// wrong version. This test also asserts the cache key + git clone
    /// references go through `${{ env.MACOS_ORT_VERSION }}` rather than
    /// hard-coded literals — without that interpolation a future
    /// version bump would only update one of the two call sites and
    /// the test would still pass.
    #[test]
    fn release_yml_macos_ort_matches_const() {
        let abs = format!(
            "{}/../../.github/workflows/release.yml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let yml = std::fs::read_to_string(&abs)
            .unwrap_or_else(|e| panic!("read {abs}: {e}"));
        let expected_decl = format!(
            "MACOS_ORT_VERSION: '{}'",
            MACOS_CORE_ML_ORT_VERSION,
        );
        assert!(
            yml.contains(&expected_decl),
            "release.yml is missing the workflow env declaration `{expected_decl}` — \
             update the workflow to match the const, or vice versa.",
        );
        assert!(
            yml.contains("ort-coreml-${{ env.MACOS_ORT_VERSION }}-macos-aarch64"),
            "release.yml ORT cache key must use ${{{{ env.MACOS_ORT_VERSION }}}} \
             so a version bump only needs to touch the env block + the const.",
        );
        assert!(
            yml.contains("git clone --depth 1 --branch \"v${MACOS_ORT_VERSION}\""),
            "release.yml ORT git clone must reference ${{MACOS_ORT_VERSION}} \
             so the cloned tag matches the cache key automatically.",
        );
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

    /// Contract: every rid that `host_rid` produces (other than the
    /// catch-all `"unknown"`) must have a matching arm in
    /// `host_pypi_token`. Otherwise users on that platform get a
    /// confusing "unsupported host platform" from `pick_wheel_for_host`
    /// despite `host_rid` agreeing the platform is supported.
    #[test]
    fn every_supported_rid_has_a_pypi_token() {
        for rid in [
            "linux-x64", "linux-arm64",
            "windows-x64", "windows-arm64",
            "macos-arm64", "macos-x64",
        ] {
            let token = match rid {
                "linux-x64" => "manylinux_2_28_x86_64",
                "linux-arm64" => "manylinux_2_28_aarch64",
                "windows-x64" => "win_amd64",
                "windows-arm64" => "win_arm64",
                "macos-arm64" => "macosx_11_0_arm64",
                "macos-x64" => "macosx_10_15_x86_64",
                _ => unreachable!(),
            };
            // Compile-time check the arm exists for current host.
            if host_rid() == rid {
                assert_eq!(host_pypi_token(), Some(token), "rid {rid}");
            }
        }
        // Final guard: host's own rid produces Some(token), which is
        // the actual contract `pick_wheel_for_host` depends on.
        if host_rid() != "unknown" {
            assert!(host_pypi_token().is_some(),
                "host_rid()={} but host_pypi_token() is None — pick_wheel_for_host will reject this host",
                host_rid());
        }
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
        let Some(token) = host_pypi_token() else { return; };
        let urls = vec![
            json!({"filename": format!("ort-cp310-cp310-{token}.whl"),
                   "url": "https://example.com/cp310.whl",
                   "digests": {"sha256": "aa".repeat(32)}, "size": 80_000_000}),
            json!({"filename": format!("ort-cp313-cp313-{token}.whl"),
                   "url": "https://example.com/cp313.whl",
                   "digests": {"sha256": "bb".repeat(32)}, "size": 80_000_000}),
        ];
        let pick = pick_wheel_for_host(&urls).expect("pick");
        assert_eq!(pick.url, "https://example.com/cp313.whl");
    }

    #[test]
    fn pick_wheel_falls_back_to_any_cp_tag() {
        let Some(token) = host_pypi_token() else { return; };
        let urls = vec![json!({
            "filename": format!("ort-cp311-cp311-{token}.whl"),
            "url": "https://example.com/cp311.whl",
            "digests": {"sha256": "cc".repeat(32)}, "size": 0,
        })];
        let pick = pick_wheel_for_host(&urls).expect("fallback");
        assert_eq!(pick.url, "https://example.com/cp311.whl");
    }

    #[test]
    fn pick_wheel_rejects_when_no_match() {
        let urls = vec![json!({
            "filename": "ort-cp313-cp313-ANDROID.whl",
            "url": "https://example.com/wrong.whl",
            "digests": {"sha256": "dd".repeat(32)}, "size": 0,
        })];
        assert!(pick_wheel_for_host(&urls).is_err());
    }

    /// PyPI's URL list also includes the source distribution
    /// (`onnxruntime-X.Y.Z.tar.gz`). The picker must skip non-`.whl`
    /// entries even if the host_token happens to appear in the
    /// filename — without this guard, `extract_wheel` would later try
    /// to parse a tarball as a zip and fail with a confusing error.
    #[test]
    fn pick_wheel_skips_sdist_even_with_matching_token() {
        let Some(token) = host_pypi_token() else { return; };
        let urls = vec![
            json!({"filename": format!("ort-{token}.tar.gz"),
                   "url": "https://example.com/sdist.tar.gz",
                   "digests": {"sha256": "ee".repeat(32)}, "size": 0}),
            json!({"filename": format!("ort-cp313-cp313-{token}.whl"),
                   "url": "https://example.com/cp313.whl",
                   "digests": {"sha256": "ff".repeat(32)}, "size": 0}),
        ];
        let pick = pick_wheel_for_host(&urls).expect("wheel pick should succeed");
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
