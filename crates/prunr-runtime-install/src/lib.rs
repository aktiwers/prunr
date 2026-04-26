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
        _ => return None,
    })
}

/// Install-dir name: `<short>-<version>-<rid>`. Locked — existing
/// user installs at `<data>/prunr/runtimes/<this>` must keep working.
pub fn install_subdir(short_name: &str, version: &str) -> String {
    format!("{short_name}-{version}-{}", host_rid())
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
pub fn repackage_target_filename(zip_name: &str) -> Option<String> {
    let stripped = zip_name.strip_prefix("onnxruntime/capi/")?;
    if stripped.contains('/') { return None; }
    if stripped.starts_with("onnxruntime_pybind11_state") { return None; }
    if stripped.ends_with(".py") { return None; }
    if stripped.starts_with("libonnxruntime.so.") {
        return Some("libonnxruntime.so".to_string());
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

/// Stream-download a wheel into memory with progress + cancel.
/// Doesn't pre-allocate from `Content-Length` (untrusted; bogus values
/// would OOM). `Vec` grows; resize cost on 80 MB is microseconds.
pub fn download_wheel(info: &WheelInfo, hooks: &mut DownloadHooks<'_>) -> Result<bytes::Bytes, String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("prunr-runtime-install/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;
    let mut response = client.get(&info.url).send()
        .map_err(|e| format!("download: {e}"))?;
    let total = response.content_length().unwrap_or(info.size_bytes);

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut last_pct: u64 = u64::MAX;
    loop {
        if hooks.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let n = response.read(&mut chunk)
            .map_err(|e| format!("download read: {e}"))?;
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

/// Extract a wheel into `target_dir`, applying `repackage_target_filename`.
/// Caller owns dir creation + safety guards (e.g. refusing
/// non-`runtimes/` paths) — that's install-time policy.
pub fn extract_wheel(bytes: &[u8], target_dir: &std::path::Path) -> Result<PathBuf, String> {
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
    use serde_json::json;

    #[test]
    fn host_rid_is_supported_or_unknown() {
        let r = host_rid();
        assert!(matches!(r,
            "linux-x64" | "linux-arm64" | "windows-x64" | "windows-arm64"
            | "macos-arm64" | "macos-x64" | "unknown"
        ), "unexpected rid: {r}");
    }

    #[test]
    fn install_subdir_format_is_stable() {
        let s = install_subdir("openvino", "1.24.1");
        assert!(s.starts_with("openvino-1.24.1-"));
        assert!(s.ends_with(host_rid()));
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

    #[test]
    fn repackage_keeps_capi_dylib() {
        assert_eq!(
            repackage_target_filename("onnxruntime/capi/libonnxruntime.so.1.24.1").as_deref(),
            Some("libonnxruntime.so"),
        );
    }

    #[test]
    fn repackage_drops_python_files_and_pybind() {
        assert!(repackage_target_filename("onnxruntime/capi/onnxruntime_pybind11_state.so").is_none());
        assert!(repackage_target_filename("onnxruntime/capi/__init__.py").is_none());
        assert!(repackage_target_filename("onnxruntime/__init__.py").is_none());
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
}
