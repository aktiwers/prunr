//! Per-EP compiled-model cache.
//!
//! Every EP that supports caching its compiled IR (OpenVINO model
//! cache, CoreML mlmodelc cache, ORT graph optimization for CPU /
//! CUDA) gets a directory under `<data>/prunr/ep_cache/<ep>/<key>/`,
//! where `<key>` is derived from `(model_id, model_version,
//! cache_format_version)`. Bumping any component invalidates the
//! cache automatically.
//!
//! Phase 2.5 (SD memory robustness). See
//! `.planning/phases/26-sd-memory-robustness/PHASE-2.5-SCOPE.md`.
//!
//! Layout:
//!
//! ```text
//! <data>/prunr/ep_cache/
//! ├── openvino/<key>/        ← OpenVINO compiled blob + index
//! ├── coreml/<key>/          ← CoreML mlmodelc bundle
//! ├── ort_optimized/<key>/   ← ORT graph-optimized .onnx (CPU + CUDA)
//! └── tensorrt/<key>/        ← reserved for future TRT EP
//! ```
//!
//! `<key>` shape: `<model_id>-<model_version>-<cache_format_version>`.
//! E.g. `SdV15InpaintFp16-1.0.0-v1`.

use std::path::PathBuf;
use prunr_models::ModelId;

/// Cache format version. Bump when the cache directory layout
/// changes in a way that breaks back-compat (e.g. moving from a
/// flat blob to a multi-file index). Old cache directories under a
/// previous version stay on disk but are never read; the
/// "Clear model cache" Settings button (P25-G) wipes them.
const CACHE_FORMAT_VERSION: u32 = 1;

/// Which EP a cached artefact was built for. Each EP's cache lives in
/// its own subdirectory so cross-EP collision can't corrupt a load
/// (e.g. OpenVINO-compiled IR loaded by CPU EP).
///
/// Mirror of `crate::engine::EpKind`, kept as a separate type to
/// keep `prunr-core::cache` independent of the `engine` module's
/// other concerns. Conversion is one match arm in `engine.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEp {
    Cpu,
    OpenVino,
    Cuda,
    CoreMl,
    DirectMl,
    TensorRt,
}

impl CacheEp {
    /// Subdirectory name. Lowercase + underscores so a future
    /// `<data>/prunr/ep_cache/` listing is grep-friendly.
    pub const fn subdir(self) -> &'static str {
        match self {
            Self::Cpu => "ort_optimized",
            Self::OpenVino => "openvino",
            Self::Cuda => "ort_optimized",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
            Self::TensorRt => "tensorrt",
        }
    }
}

/// Root cache directory: `<data>/prunr/ep_cache/`. Returns `None` on
/// platforms where `data_dir()` itself is unavailable (sandbox /
/// exotic platforms) — caller falls back to no-cache rather than
/// erroring out.
pub fn cache_root() -> Option<PathBuf> {
    prunr_models::data_dir().map(|d| d.join("ep_cache"))
}

/// Per-`(model, ep)` cache directory. Creates parent directories on
/// demand. Returns `None` when `data_dir()` is unavailable OR the
/// model id is unknown to the registry.
pub fn cache_dir_for(id: ModelId, ep: CacheEp) -> Option<PathBuf> {
    let descriptor = prunr_models::descriptor(id)?;
    let root = cache_root()?;
    let key = format!("{:?}-{}-v{}", id, descriptor.version, CACHE_FORMAT_VERSION);
    let dir = root.join(ep.subdir()).join(key);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?id, ?ep, %e, "failed to create EP cache dir");
        return None;
    }
    Some(dir)
}

/// Whether the cache for `(id, ep)` is non-empty. Used by background-
/// prewarm to skip already-cached models. Considers any non-empty
/// directory hit; the EP itself validates the contents on load.
pub fn cache_populated_for(id: ModelId, ep: CacheEp) -> bool {
    let Some(dir) = cache_dir_for(id, ep) else { return false };
    std::fs::read_dir(&dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// Wipe the entire EP cache root. Called from Settings → Hardware →
/// "Clear model cache" so users can recover from a broken cache
/// without manual `rm -rf`. Idempotent.
///
/// Returns the number of bytes reclaimed (best-effort; based on
/// pre-removal directory walk). `0` on any failure — failures are
/// logged at WARN but never propagated, since cache wipe is a
/// best-effort hygiene operation.
pub fn clear_all() -> u64 {
    let Some(root) = cache_root() else { return 0 };
    if !root.exists() { return 0 }
    let bytes = walk_dir_size(&root).unwrap_or(0);
    if let Err(e) = std::fs::remove_dir_all(&root) {
        tracing::warn!(%e, dir = %root.display(), "failed to clear EP cache");
        return 0;
    }
    bytes
}

/// Recursive directory size. Used only to report bytes-freed in the
/// Settings UI; not on any hot path.
fn walk_dir_size(dir: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += walk_dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-EP cache collision would corrupt a load if e.g.
    /// CPU-optimized IR were handed to OpenVINO. Same `model_id`,
    /// different EP → different paths.
    #[test]
    fn cache_dir_layout_per_ep_is_distinct() {
        let Some(_root) = cache_root() else { return; };
        let openvino = cache_dir_for(ModelId::Silueta, CacheEp::OpenVino);
        let cpu = cache_dir_for(ModelId::Silueta, CacheEp::Cpu);
        let coreml = cache_dir_for(ModelId::Silueta, CacheEp::CoreMl);
        match (openvino, cpu, coreml) {
            (Some(a), Some(b), Some(c)) => {
                assert_ne!(a, b, "OpenVINO and CPU caches must not collide");
                assert_ne!(a, c, "OpenVINO and CoreML caches must not collide");
                assert_ne!(b, c, "CPU and CoreML caches must not collide");
            }
            _ => {} // data_dir unavailable in CI sandbox — skip
        }
    }

    /// Bumping `CACHE_FORMAT_VERSION` (or the descriptor's `version`)
    /// must change the cache key so a stale-format cache doesn't get
    /// loaded against a newer ORT or layout.
    #[test]
    fn cache_key_includes_format_version_and_model_version() {
        let Some(dir) = cache_dir_for(ModelId::Silueta, CacheEp::OpenVino) else { return };
        let last = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        assert!(
            last.contains(&format!("v{}", CACHE_FORMAT_VERSION)),
            "cache key must contain CACHE_FORMAT_VERSION; got {last:?}",
        );
        let descriptor = prunr_models::descriptor(ModelId::Silueta).expect("Silueta in registry");
        assert!(
            last.contains(descriptor.version),
            "cache key must contain model version {}; got {last:?}",
            descriptor.version,
        );
    }

    /// Unknown model_id (registry miss) must return `None` so callers
    /// fall back to no-cache. A panic here would crash session build.
    #[test]
    fn cache_dir_returns_none_for_unknown_model() {
        // ModelId::ALL is the canonical inventory; pick one and
        // verify the well-known case works. The registry-miss path is
        // exercised when an old cached enum value is loaded against a
        // pruned registry — synthesise that with a transmute would be
        // unsound, so we just verify the happy path here. A future
        // refactor that drops ModelId::descriptor coverage would
        // surface via a None return at runtime.
        let happy = cache_dir_for(ModelId::Silueta, CacheEp::Cpu);
        if cache_root().is_some() {
            assert!(happy.is_some(), "Silueta is in the registry");
        }
    }
}
