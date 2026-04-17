//! Drag-out export: write processed images to temp PNG files for OS-level
//! drag-and-drop into external apps (Finder, Explorer, Word, PowerPoint, ...).
//!
//! Platform support:
//! - Windows / macOS: full drag-out via the `drag` crate (OLE / NSDraggingSession).
//! - Linux: not supported — the `drag` crate needs a GTK window and eframe/winit
//!   can't supply one. Callers should show a fallback toast on Linux.
//!
//! Temp file lifecycle:
//! - Single subdirectory under `std::env::temp_dir()/prunr-drag/`.
//! - Filenames derived from source (e.g. `sunset-nobg.png`, `sunset-lines.png`).
//! - Not deleted at drag end — receiving apps may read async.
//! - Stale cleanup (>10 min old) runs at app start.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::app::BatchItem;
use super::settings::LineMode;

/// Subdirectory name under temp_dir for prunr drag files.
const DRAG_SUBDIR: &str = "prunr-drag";
/// Files older than this are considered stale and get removed at startup.
const STALE_AGE: Duration = Duration::from_secs(10 * 60);

/// Return (and create if needed) the drag temp directory.
fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(DRAG_SUBDIR);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Check whether a path was created by our own drag-out (i.e. lives under prunr-drag/).
/// Used to reject self-originated drops so our own thumbnails can't be re-ingested.
pub(crate) fn is_self_drop(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == DRAG_SUBDIR)
}

/// Remove ALL temp drag files on a background thread. Called on graceful app exit.
pub(crate) fn cleanup_all() {
    std::thread::spawn(|| {
        let dir = temp_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let _ = std::fs::remove_file(&path);
            }
        }
    });
}

/// Remove stale temp files left behind by previous sessions or crashes.
/// Silent on errors — this is best-effort housekeeping.
pub(crate) fn cleanup_stale() {
    let dir = temp_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let Ok(meta) = entry.metadata() else { continue };
        let age = meta
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .unwrap_or(Duration::ZERO);
        if age > STALE_AGE {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Build a human-friendly filename stem for a batch item given its processing mode.
/// Pure function — does not touch the filesystem.
fn make_filename(source_filename: &str, has_result: bool, line_mode: LineMode) -> String {
    let stem = Path::new(source_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");

    if !has_result {
        // No processing yet — drag source as-is with original extension.
        return source_filename.to_string();
    }
    let suffix = match line_mode {
        LineMode::Off => "nobg",
        LineMode::EdgesOnly => "lines",
        LineMode::SubjectOutline => "nobg-lines",
    };
    format!("{stem}-{suffix}.png")
}

/// Prepare a drag temp file for a batch item. Returns the absolute path.
/// If the item has no processed result, writes the raw source bytes as-is
/// (preserving original extension so file managers recognize it).
///
/// Uses the item's own `line_mode` — each image carries its own mode in v2.
pub(crate) fn prepare(item: &BatchItem) -> std::io::Result<PathBuf> {
    let name = make_filename(&item.filename, item.result_rgba.is_some(), item.settings.line_mode);
    let path = temp_dir().join(&name);
    match &item.result_rgba {
        Some(rgba) => {
            let bytes = prunr_core::encode_rgba_png(rgba)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            std::fs::write(&path, &bytes)?;
        }
        None => {
            let bytes = item.source.load_bytes()?;
            std::fs::write(&path, bytes.as_slice())?;
        }
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_filename_uses_source_stem_with_mode_suffix() {
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::Off),
            "sunset-nobg.png"
        );
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::EdgesOnly),
            "sunset-lines.png"
        );
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::SubjectOutline),
            "sunset-nobg-lines.png"
        );
    }

    #[test]
    fn make_filename_unprocessed_preserves_original() {
        assert_eq!(
            make_filename("photo.webp", false, LineMode::Off),
            "photo.webp"
        );
    }

    #[test]
    fn make_filename_handles_weird_names() {
        assert_eq!(
            make_filename("no-extension", true, LineMode::Off),
            "no-extension-nobg.png"
        );
        assert_eq!(
            make_filename("", true, LineMode::Off),
            "image-nobg.png"
        );
    }

    #[test]
    fn cleanup_stale_does_not_panic_on_missing_dir() {
        // Should silently succeed even if temp dir doesn't exist.
        cleanup_stale();
    }
}
