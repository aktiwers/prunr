//! ONNX Runtime dylib resolution and the one process-wide `ort` init.
//!
//! Resolution order, first hit wins:
//!   1. `ORT_DYLIB_PATH` env var (escape hatch + dev override). Set but
//!      not a file is an error, never a silent fall-through.
//!   2. User Runtime Store install (`<data>/prunr/runtimes/<ep>-<ver>-<rid>/`):
//!      the one named in [`PREFERRED_FILE`], else the first by name
//!   3. Bundled fallback alongside the executable (`<exe parent>/runtime/`)
//!
//! Every session in the workspace starts from [`session_builder`], which
//! runs [`ensure_initialized`] first. Under `load-dynamic`, building a
//! session before the runtime is loaded does not fail: ort re-enters its
//! own loader lock while formatting the "library not found" error and
//! the thread parks forever. Routing every builder through here turns
//! that hang into an error.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::session::builder::SessionBuilder;
use ort::session::Session;

use crate::types::CoreError;

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

/// Load the runtime once per process. Every later call returns the
/// first outcome, so a missing runtime surfaces as the same error at
/// every session site instead of a hang.
pub fn ensure_initialized() -> Result<DylibSource, String> {
    static RESULT: OnceLock<Result<DylibSource, String>> = OnceLock::new();
    RESULT.get_or_init(init).clone()
}

/// Names the loaded runtime build ("1.24.1-3fa2c901"), once it has
/// loaded: its version where the build info carries one (Microsoft's
/// own builds say `git-branch=HEAD`), and a fingerprint of the library
/// file, which every runtime upgrade changes.
pub fn runtime_tag() -> Option<&'static str> {
    static TAG: OnceLock<Option<String>> = OnceLock::new();
    TAG.get_or_init(|| {
        ensure_initialized().ok()?;
        let meta = std::fs::metadata(LOADED.get()?).ok()?;
        let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
        Some(tag_for(runtime_version_from_info(ort::info()).as_deref(), meta.len(), modified))
    }).as_deref()
}

/// The library loaded by `init`.
static LOADED: OnceLock<PathBuf> = OnceLock::new();

pub(crate) fn tag_for(version: Option<&str>, len: u64, modified: u64) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in len.to_le_bytes().into_iter().chain(modified.to_le_bytes()) {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    format!("{}-{hash:08x}", version.unwrap_or("build"))
}

/// The version in ORT's build info, whose branch is `rel-<version>`.
pub fn runtime_version_from_info(info: &str) -> Option<String> {
    let branch = info.split("git-branch=").nth(1)?.split(',').next()?.trim();
    branch.strip_prefix("rel-").map(str::to_string)
}

/// A session builder on the loaded runtime.
pub fn session_builder() -> Result<SessionBuilder, CoreError> {
    ensure_initialized().map_err(CoreError::Inference)?;
    #[allow(clippy::disallowed_methods)] // the sanctioned wrapper: the runtime is loaded one line above
    Session::builder().map_err(|e| CoreError::Inference(format!("ORT builder init failed: {e}")))
}

fn init() -> Result<DylibSource, String> {
    let (path, source) = resolve_dylib_path()?;
    #[allow(clippy::disallowed_methods)] // the sanctioned init: guarded by the OnceLock in ensure_initialized
    let env = ort::init_from(&path)
        .map_err(|e| format!("ort::init_from({}): {e}", path.display()))?;
    // false means an env was already committed; ORT is initialised either way.
    let _ = env.commit();
    tracing::info!(path = %path.display(), %source, "ORT runtime loaded");
    let _ = LOADED.set(path);
    Ok(source)
}

pub fn resolve_dylib_path() -> Result<(PathBuf, DylibSource), String> {
    if let Some(path) = env_override() {
        if !path.is_file() {
            return Err(format!("ORT_DYLIB_PATH={} is not a file", path.display()));
        }
        return Ok((path, DylibSource::EnvVar));
    }
    if let Some(path) = prunr_models::runtime_store_dir().and_then(|root| store_dylib(&root)) {
        return Ok((path, DylibSource::RuntimeStore));
    }
    if let Some(path) = bundled_dylib() {
        return Ok((path, DylibSource::Bundled));
    }
    Err(format!(
        "ONNX Runtime not found. Looked for `{DYLIB_NAME}` in (in order): \
         ORT_DYLIB_PATH env, user Runtime Store, bundled location next to executable. \
         Set ORT_DYLIB_PATH or run `cargo xtask install-runtime onnxruntime <version>`."
    ))
}

fn env_override() -> Option<PathBuf> {
    std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from)
}

/// File in the runtime store naming the install every process loads.
/// One process loads one runtime, and no wheel carries another's
/// provider, so with two installed the choice is explicit; a file is
/// what the worker subprocess and the CLI see as well as the GUI.
pub const PREFERRED_FILE: &str = "preferred";

/// The install subdir every process loads first, when one is set.
pub fn preferred_runtime() -> Option<String> {
    read_preferred(&prunr_models::runtime_store_dir()?)
}

/// Make the install `subdir` the one every process loads from its next
/// start.
pub fn set_preferred_runtime(subdir: &str) -> Result<(), String> {
    let root = prunr_models::runtime_store_dir().ok_or("could not resolve user data dir")?;
    write_preferred(&root, subdir)
}

/// Drop the preference when it names `subdir`, which is going away.
pub fn forget_preferred_runtime(subdir: &str) {
    let Some(root) = prunr_models::runtime_store_dir() else { return };
    if read_preferred(&root).as_deref() == Some(subdir) {
        if let Err(e) = std::fs::remove_file(root.join(PREFERRED_FILE)) {
            tracing::warn!(%e, subdir, "could not clear the preferred runtime");
        }
    }
}

fn read_preferred(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join(PREFERRED_FILE)).ok()?;
    let name = text.trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn write_preferred(root: &Path, subdir: &str) -> Result<(), String> {
    std::fs::create_dir_all(root).map_err(|e| format!("create {}: {e}", root.display()))?;
    let path = root.join(PREFERRED_FILE);
    std::fs::write(&path, subdir).map_err(|e| format!("write {}: {e}", path.display()))
}

/// The store's dylib: the preferred install's when it has one, else the
/// first install's by name.
fn store_dylib(root: &Path) -> Option<PathBuf> {
    let usable: Vec<(String, PathBuf)> = store_entries_in(root).into_iter().filter(|(_, p)| p.is_file()).collect();
    let preferred = read_preferred(root).and_then(|want| usable.iter().position(|(name, _)| *name == want));
    usable.into_iter().nth(preferred.unwrap_or(0)).map(|(_, p)| p)
}

fn store_entries() -> Vec<(String, PathBuf)> {
    prunr_models::runtime_store_dir().map(|root| store_entries_in(&root)).unwrap_or_default()
}

/// Every runtime-store entry as `(dir name, path the dylib would have)`,
/// sorted by name so logs reproduce; meaningful EP selection happens at
/// the per-session ladder, not at this layer.
fn store_entries_in(root: &Path) -> Vec<(String, PathBuf)> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, PathBuf)> = read
        .filter_map(Result::ok)
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path().join(DYLIB_NAME)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Structured snapshot of runtime resolution state for `prunr doctor`.
/// Each field reflects what the resolver would see on a fresh init.
#[derive(Debug)]
pub struct Diagnostics {
    pub env_path: Option<PathBuf>,
    pub store_root: Option<PathBuf>,
    pub store_entries: Vec<(String, bool)>,
    pub preferred: Option<String>,
    pub bundled: Option<(PathBuf, bool)>,
    pub resolved: Option<(PathBuf, DylibSource)>,
}

pub fn diagnose() -> Diagnostics {
    Diagnostics {
        env_path: env_override(),
        store_root: prunr_models::runtime_store_dir(),
        store_entries: store_entries().into_iter().map(|(name, p)| { let exists = p.is_file(); (name, exists) }).collect(),
        preferred: preferred_runtime(),
        bundled: bundled_nested_path().map(|p| { let exists = p.is_file(); (p, exists) }),
        resolved: resolve_dylib_path().ok(),
    }
}

/// True iff at least one ORT source other than `excluding` exists. Used
/// by Settings → Hardware → Uninstall to refuse a removal that would
/// leave the app unable to load any runtime — this is the dev-build
/// case where there's no bundled fallback next to the executable, and
/// the user's runtime-store entry is their only ORT.
pub fn has_fallback_excluding(excluding: &Path) -> bool {
    env_override().is_some_and(|p| p.is_file() && p != excluding)
        || store_entries().iter().any(|(_, p)| p.is_file() && p != excluding && !p.starts_with(excluding))
        || bundled_dylib().is_some()
}

/// Release CI's `cargo xtask install-runtime --stage-to <pkg>/runtime/`
/// puts shared libs in a `runtime/` sibling of the binary.
fn bundled_nested_path() -> Option<PathBuf> {
    Some(std::env::current_exe().ok()?.parent()?.join("runtime").join(DYLIB_NAME))
}

fn bundled_dylib() -> Option<PathBuf> {
    // Some packagers flatten the `runtime/` dir, hence the exe-dir fallback.
    //
    // macOS .app skips both: the binary's rpath
    // (`@executable_path/../Frameworks`, set in `.cargo/config.toml`)
    // resolves the dylib at link time, so this returns `None` and the
    // dlopen falls through to rpath. There's no Frameworks/ check here
    // on purpose.
    let nested = bundled_nested_path()?;
    if nested.is_file() {
        return Some(nested);
    }
    let flat = nested.parent()?.parent()?.join(DYLIB_NAME);
    flat.is_file().then_some(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install(root: &Path, subdir: &str) {
        let dir = root.join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(DYLIB_NAME), b"").unwrap();
    }

    #[test]
    fn the_first_install_by_name_loads_without_a_preference() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "openvino-1.24.1-windows-x64");
        install(root.path(), "directml-1.24.1-windows-x64");
        assert_eq!(store_dylib(root.path()), Some(root.path().join("directml-1.24.1-windows-x64").join(DYLIB_NAME)));
    }

    #[test]
    fn the_preferred_install_loads_even_when_another_sorts_first() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "directml-1.24.1-windows-x64");
        install(root.path(), "openvino-1.24.1-windows-x64");
        write_preferred(root.path(), "openvino-1.24.1-windows-x64").unwrap();
        assert_eq!(store_dylib(root.path()), Some(root.path().join("openvino-1.24.1-windows-x64").join(DYLIB_NAME)));
    }

    #[test]
    fn a_preference_for_a_missing_or_broken_install_falls_back_to_name_order() {
        let root = tempfile::tempdir().unwrap();
        install(root.path(), "openvino-1.24.1-windows-x64");
        write_preferred(root.path(), "directml-1.24.1-windows-x64").unwrap();
        let fallback = Some(root.path().join("openvino-1.24.1-windows-x64").join(DYLIB_NAME));
        assert_eq!(store_dylib(root.path()), fallback);
        std::fs::create_dir_all(root.path().join("directml-1.24.1-windows-x64")).unwrap();
        assert_eq!(store_dylib(root.path()), fallback, "an install without its library is skipped");
    }

    #[test]
    fn the_preference_file_is_not_an_install_and_reads_back_trimmed() {
        let root = tempfile::tempdir().unwrap();
        assert_eq!(store_dylib(root.path()), None);
        std::fs::write(root.path().join(PREFERRED_FILE), "openvino-1.24.1-linux-x64\n").unwrap();
        assert_eq!(store_entries_in(root.path()), Vec::new());
        assert_eq!(read_preferred(root.path()).as_deref(), Some("openvino-1.24.1-linux-x64"));
        std::fs::write(root.path().join(PREFERRED_FILE), "  ").unwrap();
        assert_eq!(read_preferred(root.path()), None);
    }
}
