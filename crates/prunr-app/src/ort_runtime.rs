//! ONNX Runtime dylib resolution for Phase 19's `load-dynamic` path.
//!
//! Resolution order, first hit wins:
//!   1. `ORT_DYLIB_PATH` env var (escape hatch + dev override)
//!   2. User Runtime Store install (`<data>/prunr/runtimes/<ep>/<ver>/`)
//!   3. Bundled fallback alongside the executable (`<exe parent>/runtime/`)
//!
//! Called once per process — main GUI/CLI startup and the subprocess
//! worker entry. Must run before any other `ort::*` use.

use std::path::{Path, PathBuf};

pub const DYLIB_NAME: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

#[derive(Debug, Clone, Copy)]
pub enum DylibSource {
    EnvVar,
    RuntimeStore,
    Bundled,
}

impl std::fmt::Display for DylibSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::EnvVar => "ORT_DYLIB_PATH",
            Self::RuntimeStore => "runtime-store",
            Self::Bundled => "bundled",
        })
    }
}

pub fn init() -> Result<DylibSource, String> {
    let (path, source) = resolve_dylib_path().ok_or_else(|| {
        format!(
            "ONNX Runtime not found. Looked for `{DYLIB_NAME}` in (in order): \
             ORT_DYLIB_PATH env, user Runtime Store, bundled location next to executable. \
             Phase 19's Runtime Store will install one on first launch."
        )
    })?;

    let env =
        ort::init_from(&path).map_err(|e| format!("ort::init_from({}): {e}", path.display()))?;
    // `commit()` returns false when an env was already committed (e.g.
    // double-init in tests, or future re-entry). ORT is initialized
    // either way — treat as success.
    let _ = env.commit();
    tracing::info!(path = %path.display(), %source, "ORT runtime loaded");
    Ok(source)
}

pub fn resolve_dylib_path() -> Option<(PathBuf, DylibSource)> {
    if let Some(env_path) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(env_path);
        if path.is_file() {
            return Some((path, DylibSource::EnvVar));
        }
    }
    if let Some(path) = runtime_store_dylib() {
        return Some((path, DylibSource::RuntimeStore));
    }
    if let Some(path) = bundled_dylib() {
        return Some((path, DylibSource::Bundled));
    }
    None
}

fn runtime_store_dylib() -> Option<PathBuf> {
    let root = prunr_models::data_dir()?.join("runtimes");
    if !root.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .collect();
    // Deterministic order so logs reproduce; meaningful EP selection
    // happens at the per-session ladder, not at this layer.
    entries.sort_by_key(|e| e.file_name());
    entries
        .into_iter()
        .map(|e| e.path().join(DYLIB_NAME))
        .find(|p| p.is_file())
}

/// Structured snapshot of runtime resolution state for `prunr doctor`.
/// Each field reflects what the resolver would see on a fresh `init()`.
#[derive(Debug)]
pub struct Diagnostics {
    pub env_path: Option<PathBuf>,
    pub store_root: Option<PathBuf>,
    pub store_entries: Vec<(String, bool)>,
    pub bundled: Option<(PathBuf, bool)>,
    pub resolved: Option<(PathBuf, DylibSource)>,
}

pub fn diagnose() -> Diagnostics {
    let env_path = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from);
    let store_root = prunr_models::data_dir().map(|d| d.join("runtimes"));
    let store_entries = store_root
        .as_ref()
        .filter(|p| p.is_dir())
        .and_then(|root| std::fs::read_dir(root).ok())
        .map(|read| {
            let mut v: Vec<(String, bool)> = read
                .filter_map(Result::ok)
                .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
                .map(|e| {
                    let has_dylib = e.path().join(DYLIB_NAME).is_file();
                    (e.file_name().to_string_lossy().into_owned(), has_dylib)
                })
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        })
        .unwrap_or_default();
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("runtime").join(DYLIB_NAME)))
        .map(|p| {
            let exists = p.is_file();
            (p, exists)
        });
    Diagnostics {
        env_path,
        store_root,
        store_entries,
        bundled,
        resolved: resolve_dylib_path(),
    }
}

/// True iff at least one ORT source other than `excluding` exists. Used
/// by Settings → Hardware → Uninstall to refuse a removal that would
/// leave the app unable to load any runtime — this is the dev-build
/// case where there's no bundled fallback next to the executable, and
/// the user's runtime-store entry is their only ORT.
pub fn has_fallback_excluding(excluding: &Path) -> bool {
    if let Some(env) = std::env::var_os("ORT_DYLIB_PATH") {
        let p = PathBuf::from(env);
        if p.is_file() && p != excluding {
            return true;
        }
    }
    if let Some(root) = prunr_models::data_dir().map(|d| d.join("runtimes")) {
        if let Ok(read) = std::fs::read_dir(&root) {
            for e in read.flatten() {
                let p = e.path().join(DYLIB_NAME);
                if p.is_file() && p != excluding && !p.starts_with(excluding) {
                    return true;
                }
            }
        }
    }
    bundled_dylib().is_some()
}

fn bundled_dylib() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?;
    // Release CI's `cargo xtask install-runtime --stage-to <pkg>/runtime/`
    // puts shared libs in a `runtime/` sibling of the binary. Some
    // packagers flatten this, hence the exe-dir fallback.
    //
    // macOS .app skips both: the binary's rpath
    // (`@executable_path/../Frameworks`, set in `.cargo/config.toml`)
    // resolves the dylib at link time, so `bundled_dylib` returns
    // `None` and the dlopen falls through to rpath. Documented for
    // future readers — there's no Frameworks/ check here on purpose.
    let nested = parent.join("runtime").join(DYLIB_NAME);
    if nested.is_file() {
        return Some(nested);
    }
    let flat = parent.join(DYLIB_NAME);
    flat.is_file().then_some(flat)
}
