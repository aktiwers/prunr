//! Per-EP compiled-model cache.
//!
//! Every EP that supports caching its compiled IR (OpenVINO model
//! cache, CoreML mlmodelc cache, ORT graph optimization for CPU /
//! CUDA) gets a directory under `<data>/prunr/ep_cache/<ep>/<key>/`,
//! where `<key>` is `<model_stable_name>-<model_version>-v<format>`.
//! Bumping any component invalidates the cache automatically.
//!
//! EP names match `OrtEngine::active_provider()`'s output (`"OpenVINO"`,
//! `"CUDA"`, `"CoreML"`, `"DirectML"`, `"CPU"`) and the existing
//! `ep_compat.json` cache keys — strings rather than an enum because
//! the rest of the codebase already treats EP identity as a string.

use std::path::{Path, PathBuf};
use prunr_models::ModelId;

/// Cache format version. Bump when the cache directory layout
/// changes in a way that breaks back-compat. Old directories under a
/// previous version stay on disk but are never read; the
/// "Clear model cache" Settings button wipes them.
const CACHE_FORMAT_VERSION: u32 = 1;

/// Maximum recursion depth for `walk_dir_size`. Bounds the worst case
/// for an adversarial / corrupted cache directory with deeply nested
/// symlinks; cache layout is two levels deep in normal operation, so
/// 16 has plenty of headroom while still terminating.
const MAX_WALK_DEPTH: u32 = 16;

/// Root cache directory: `<data>/prunr/ep_cache/`. Returns `None` on
/// platforms where `data_dir()` itself is unavailable; callers fall
/// back to no-cache rather than erroring out.
pub fn cache_root() -> Option<PathBuf> {
    prunr_models::data_dir().map(|d| d.join("ep_cache"))
}

/// Pure path builder — no filesystem side effects. Use this from any
/// read-only check (`cache_populated_for`, prewarm decisions). Callers
/// that actually need the directory to exist should use
/// `cache_dir_for` instead, which creates it.
pub fn cache_path_for(id: ModelId, ep_name: &str) -> Option<PathBuf> {
    let root = cache_root()?;
    let descriptor = prunr_models::descriptor(id)?;
    let key = format!(
        "{}-{}-v{}",
        id.stable_name(), descriptor.version, CACHE_FORMAT_VERSION,
    );
    Some(root.join(ep_name.to_ascii_lowercase()).join(key))
}

/// Per-`(model, ep)` cache directory, **created** if missing. Use
/// from session-build sites that are about to write into the cache.
/// Returns `None` when `data_dir()` is unavailable, the model id is
/// unknown, or `create_dir_all` fails.
pub fn cache_dir_for(id: ModelId, ep_name: &str) -> Option<PathBuf> {
    let dir = cache_path_for(id, ep_name)?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?id, ep = %ep_name, %e, "failed to create EP cache dir");
        return None;
    }
    Some(dir)
}

/// Pure path to the ORT-graph-optimized model file for `(id, ep_name)`.
/// **No filesystem side effects** — mirrors the `cache_path_for` /
/// `cache_dir_for` split. Read sites poll this and `fs::read` the path
/// directly; write sites must call `cache_dir_for` first to ensure the
/// parent directory exists.
pub fn optimized_model_path(id: ModelId, ep_name: &str) -> Option<PathBuf> {
    cache_path_for(id, ep_name).map(|d| d.join("optimized.onnx"))
}

/// Per-part variant of `cache_path_for` — isolates each bundle part so
/// optimized blobs from one part can't collide with another's in shared
/// EP cache directories.
pub fn cache_path_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    cache_path_for(id, ep_name).map(|p| p.join(part))
}

/// Per-part variant of `cache_dir_for` — creates the part-scoped dir.
pub fn cache_dir_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    let dir = cache_path_for_part(id, ep_name, part)?;
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?id, ep = %ep_name, part, %e, "failed to create per-part EP cache dir");
        return None;
    }
    Some(dir)
}

/// Per-part path to the ORT-graph-optimized file. No filesystem side
/// effects — same pure/creating split as `cache_path_for`.
pub fn optimized_model_path_for_part(id: ModelId, ep_name: &str, part: &str) -> Option<PathBuf> {
    cache_path_for_part(id, ep_name, part).map(|d| d.join("optimized.onnx"))
}

/// True when the cache for `(id, ep_name)` exists on disk and is
/// non-empty. Read-only — never creates a directory as a side
/// effect. The EP itself validates contents on load; this check is
/// just "should we skip the prewarm work?".
pub fn cache_populated_for(id: ModelId, ep_name: &str) -> bool {
    let Some(dir) = cache_path_for(id, ep_name) else { return false };
    if !dir.is_dir() { return false; }
    std::fs::read_dir(&dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// On session-build entry, sweep stale cache entries for the same
/// `(model, ep)` pair. A stale entry shares this model's `stable_name`
/// but has a different `version` or `CACHE_FORMAT_VERSION` — i.e. an
/// older prunr install or model bump left it behind. Best-effort —
/// any I/O error logs and falls back to leaving entries alone.
///
/// Disambiguation: the registry's other model `stable_name`s are
/// excluded by prefix-match before deletion, so two models whose names
/// share a textual prefix (`lama` vs hypothetical `lama_xl`) never
/// nuke each other's caches.
///
/// Concurrent processes are safe: each computes the same current key
/// and leaves it alone; deletion of stale siblings is idempotent (the
/// loser of a `remove_dir_all` race sees `NotFound`, logged + ignored).
pub fn gc_stale_for_model(id: ModelId, ep_name: &str) {
    let Some(root) = cache_root() else { return };
    let ep_dir = root.join(ep_name.to_ascii_lowercase());
    if !ep_dir.is_dir() { return; }
    let Some(current_name) = cache_path_for(id, ep_name)
        .and_then(|p| p.file_name().map(|n| n.to_owned()))
    else { return };

    let stable = id.stable_name();
    let other_prefixes: Vec<String> = ModelId::ALL.iter()
        .filter(|&&m| m.stable_name() != stable)
        .map(|m| format!("{}-", m.stable_name()))
        .collect();
    let our_prefix = format!("{}-", stable);

    let entries = match std::fs::read_dir(&ep_dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(?id, ep = %ep_name, %e, "gc: read_dir failed");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name() else { continue };
        if name == current_name { continue; }
        let Some(name_str) = name.to_str() else { continue };
        if other_prefixes.iter().any(|p| name_str.starts_with(p)) { continue; }
        if !name_str.starts_with(&our_prefix) { continue; }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => tracing::info!(
                ?id, ep = %ep_name, dir = %path.display(),
                "gc: removed stale cache entry",
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                ?id, ep = %ep_name, dir = %path.display(), %e,
                "gc: failed to remove stale cache entry",
            ),
        }
    }
}

/// Wipe the entire EP cache root. Returns the number of bytes
/// reclaimed (best-effort; based on a pre-removal directory walk).
/// Run on a background thread when called from GUI handlers — the
/// walk + remove can take hundreds of ms on multi-GB caches over
/// slow disks.
pub fn clear_all() -> u64 {
    let Some(root) = cache_root() else { return 0 };
    if !root.exists() { return 0 }
    let bytes = walk_dir_size(&root, 0).unwrap_or(0);
    if let Err(e) = std::fs::remove_dir_all(&root) {
        tracing::warn!(%e, dir = %root.display(), "failed to clear EP cache");
        return 0;
    }
    bytes
}

/// Recursive directory size with depth bound. Used only by `clear_all`
/// for the bytes-reclaimed report. Skips entries beyond
/// `MAX_WALK_DEPTH` so a symlink loop terminates instead of stack-
/// overflowing.
fn walk_dir_size(dir: &Path, depth: u32) -> std::io::Result<u64> {
    if depth > MAX_WALK_DEPTH { return Ok(0); }
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += walk_dir_size(&entry.path(), depth + 1)?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-EP collision would corrupt a load if e.g. CPU-optimized
    /// IR were handed to OpenVINO. Same `model_id`, different EP →
    /// different paths.
    #[test]
    fn cache_path_per_ep_is_distinct() {
        let Some(_) = cache_root() else { return };
        let openvino = cache_path_for(ModelId::Silueta, "OpenVINO");
        let cpu = cache_path_for(ModelId::Silueta, "CPU");
        let coreml = cache_path_for(ModelId::Silueta, "CoreML");
        assert!(openvino.is_some() && cpu.is_some() && coreml.is_some());
        assert_ne!(openvino, cpu);
        assert_ne!(openvino, coreml);
        assert_ne!(cpu, coreml);
    }

    /// EP-name casing must not split the cache (e.g. "openvino" and
    /// "OpenVINO" landing in different subdirs would silently miss the
    /// warm cache on lookups). Lowercased uniformly.
    #[test]
    fn cache_path_normalises_ep_name_case() {
        let Some(_) = cache_root() else { return };
        let lower = cache_path_for(ModelId::Silueta, "openvino").unwrap();
        let upper = cache_path_for(ModelId::Silueta, "OPENVINO").unwrap();
        let mixed = cache_path_for(ModelId::Silueta, "OpenVINO").unwrap();
        assert_eq!(lower, upper);
        assert_eq!(lower, mixed);
    }

    /// Bumping `CACHE_FORMAT_VERSION` or the descriptor's `version`
    /// must change the cache key so a stale-format cache doesn't get
    /// loaded against a newer ORT or layout. Also: stable_name (not
    /// Debug) means a future variant rename doesn't silently
    /// invalidate every user's cache.
    #[test]
    fn cache_key_includes_stable_name_and_versions() {
        let Some(dir) = cache_path_for(ModelId::Silueta, "CPU") else { return };
        let key = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        assert!(key.contains(ModelId::Silueta.stable_name()),
            "cache key must contain stable_name; got {key:?}");
        let descriptor = prunr_models::descriptor(ModelId::Silueta).unwrap();
        assert!(key.contains(descriptor.version),
            "cache key must contain model version {}; got {key:?}", descriptor.version);
        assert!(key.contains(&format!("v{CACHE_FORMAT_VERSION}")),
            "cache key must contain CACHE_FORMAT_VERSION; got {key:?}");
    }

    /// Read-only check must not create the cache directory. Pre-fix
    /// `cache_populated_for` indirected through `cache_dir_for` which
    /// `create_dir_all`'d every poll — every prewarm decision was
    /// silently `mkdir`-ing an empty directory.
    #[test]
    fn cache_populated_does_not_create_directory() {
        let Some(path) = cache_path_for(ModelId::DexiNed, "CPU") else { return };
        let _ = std::fs::remove_dir_all(&path);
        let exists_before = path.exists();
        let populated = cache_populated_for(ModelId::DexiNed, "CPU");
        let exists_after = path.exists();
        assert!(!populated);
        assert_eq!(exists_before, exists_after,
            "cache_populated_for must be read-only");
    }

    /// Stale-entry GC must remove sibling dirs from older versions /
    /// older `CACHE_FORMAT_VERSION`s, leave the current entry alone,
    /// and never touch other models' caches even when their stable
    /// names share a prefix.
    #[test]
    fn gc_stale_for_model_removes_siblings_only() {
        let Some(current) = cache_dir_for(ModelId::Migan, "CPU") else { return };
        let Some(ep_root) = cache_root().map(|r| r.join("cpu")) else { return };
        // Sibling: same stable_name, older version.
        let stale = ep_root.join(format!("{}-0.0.1-v0", ModelId::Migan.stable_name()));
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("dummy"), b"x").unwrap();
        // Other model's cache must survive.
        let other_dir = ep_root.join(format!("{}-1.0.0-v1", ModelId::Silueta.stable_name()));
        std::fs::create_dir_all(&other_dir).unwrap();
        std::fs::write(other_dir.join("dummy"), b"x").unwrap();

        gc_stale_for_model(ModelId::Migan, "CPU");

        assert!(current.is_dir(), "current cache must be preserved");
        assert!(!stale.exists(), "stale sibling must be removed");
        assert!(other_dir.is_dir(), "other model's cache must be untouched");
        let _ = std::fs::remove_dir_all(&current);
        let _ = std::fs::remove_dir_all(&other_dir);
    }

    /// True only when the directory exists AND has at least one entry.
    /// Empty directory still reports false so the prewarm doesn't
    /// skip a missing-IR case (e.g. partial wipe via `rm -rf` on a
    /// subdir of the cache root).
    #[test]
    fn cache_populated_distinguishes_empty_from_nonempty() {
        let Some(dir) = cache_dir_for(ModelId::Migan, "CPU") else { return };
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!cache_populated_for(ModelId::Migan, "CPU"),
            "empty dir must report unpopulated");
        std::fs::write(dir.join("optimized.onnx"), b"placeholder").unwrap();
        assert!(cache_populated_for(ModelId::Migan, "CPU"),
            "dir with content must report populated");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
