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

use image::DynamicImage;
use prunr_core::{BgEffect, FillStyle, MaskSettings};

use super::item::BatchItem;
use super::settings::LineMode;
use super::worker::SegBundle;

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

fn source_stem(source_filename: &str) -> &str {
    Path::new(source_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
}

/// Build a human-friendly filename stem for a batch item given its processing mode.
fn make_filename(source_filename: &str, has_result: bool, line_mode: LineMode) -> String {
    if !has_result {
        // No processing yet — drag source as-is with original extension.
        return source_filename.to_string();
    }
    let suffix = match line_mode {
        LineMode::Off => "nobg",
        LineMode::EdgesOnly => "lines",
        LineMode::SubjectOutline => "nobg-lines",
    };
    format!("{}-{suffix}.png", source_stem(source_filename))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerKind {
    Subject,
    Lines,
    Mask,
}

impl LayerKind {
    pub const ALL: [Self; 3] = [Self::Subject, Self::Lines, Self::Mask];

    fn suffix(self) -> &'static str {
        match self {
            Self::Subject => "subject",
            Self::Lines => "lines",
            Self::Mask => "mask",
        }
    }
}

fn make_layer_filename(source_filename: &str, kind: LayerKind) -> String {
    format!("{}-{}.png", source_stem(source_filename), kind.suffix())
}

fn decode_source(item: &BatchItem) -> Option<DynamicImage> {
    let bytes = item.source.load_bytes()
        .map_err(|err| tracing::warn!(item_id = item.id, %err, "split drag-out: source read"))
        .ok()?;
    prunr_core::load_image_from_bytes(&bytes)
        .map_err(|err| tracing::warn!(item_id = item.id, %err, "split drag-out: decode"))
        .ok()
}

fn render_layer(
    item: &BatchItem,
    kind: LayerKind,
    original: &DynamicImage,
    seg: Option<&SegBundle>,
) -> Option<Vec<u8>> {
    let item_id = item.id;
    match kind {
        LayerKind::Subject => {
            let seg = seg?;
            // Strip user's fill_style + bg_effect so the receiving app gets
            // a clean transparent subject. Keep every other mask knob so
            // their guided-filter / feather / edge_shift refinements survive.
            let mask = MaskSettings {
                fill_style: FillStyle::None,
                bg_effect: BgEffect::None,
                ..item.settings.mask_settings()
            };
            let rgba = prunr_core::postprocess_from_flat(
                &seg.data, seg.height as usize, seg.width as usize,
                original, &mask, seg.model, None,
            )
                .map_err(|err| tracing::warn!(item_id, %err, "subject layer postprocess"))
                .ok()?;
            prunr_core::encode_rgba_png(&rgba)
                .map_err(|err| tracing::warn!(item_id, %err, "subject layer encode"))
                .ok()
        }
        LayerKind::Lines => {
            let et = item.cached_edge_tensors.as_ref()?;
            let tensor = et.decompress(item.settings.edge_scale)?;
            let rgba = prunr_core::finalize_edges(
                &tensor, et.height, et.width, original, &item.settings.edge_settings(),
            );
            prunr_core::encode_rgba_png(&rgba)
                .map_err(|err| tracing::warn!(item_id, %err, "lines layer encode"))
                .ok()
        }
        LayerKind::Mask => {
            let seg = seg?;
            let gray = prunr_core::tensor_to_mask_from_flat(
                &seg.data, seg.height as usize, seg.width as usize,
                original, &item.settings.mask_settings(), seg.model, None,
            )
                .map_err(|err| tracing::warn!(item_id, %err, "mask layer reshape"))
                .ok()?;
            prunr_core::encode_gray_png(&gray)
                .map_err(|err| tracing::warn!(item_id, %err, "mask layer encode"))
                .ok()
        }
    }
}

fn write_layer(
    item: &BatchItem,
    kind: LayerKind,
    original: &DynamicImage,
    seg: Option<&SegBundle>,
) -> std::io::Result<Option<PathBuf>> {
    let Some(png_bytes) = render_layer(item, kind, original, seg) else { return Ok(None) };
    let path = temp_dir().join(make_layer_filename(&item.filename, kind));
    std::fs::write(&path, &png_bytes)?;
    Ok(Some(path))
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

/// Prepare drag temp files for a batch item. `split=false` yields the single
/// composite PNG (`prepare`); `split=true` yields up to three layer PNGs
/// (subject / lines / mask). Layers whose underlying tensor isn't cached are
/// skipped; if all are missing the path falls back to composite so an
/// unprocessed drag still lands a file.
pub(crate) fn prepare_for_drag(item: &BatchItem, split: bool) -> std::io::Result<Vec<PathBuf>> {
    if !split {
        return Ok(vec![prepare(item)?]);
    }

    // Decode source + decompress seg tensor once — Subject+Mask share seg,
    // Lines uses the edge tensor independently. Avoids a 4K decode 3× on
    // drag completion.
    let Some(original) = decode_source(item) else {
        return Ok(vec![prepare(item)?]);
    };
    let seg = item.cached_tensor.as_ref().and_then(|ct| ct.bundle());

    let paths: Vec<PathBuf> = LayerKind::ALL.iter()
        .copied()
        .filter_map(|k| write_layer(item, k, &original, seg.as_ref()).transpose())
        .collect::<std::io::Result<Vec<_>>>()?;

    if paths.is_empty() {
        return Ok(vec![prepare(item)?]);
    }
    Ok(paths)
}

/// Render the available split layers for `item` as `(filename, PNG bytes)`
/// pairs, without touching the filesystem. Shared by the Save-to-folder path
/// in `app.rs` — drag-out writes these same bytes to the temp dir, Save
/// writes them to the user's chosen folder.
///
/// Empty when no cached tensors are available (item never processed, or
/// caches evicted under memory pressure) — callers should fall back to the
/// composite PNG in that case.
pub(crate) fn render_layer_bytes(item: &BatchItem) -> Vec<(String, Vec<u8>)> {
    let Some(original) = decode_source(item) else { return Vec::new() };
    let seg = item.cached_tensor.as_ref().and_then(|ct| ct.bundle());
    LayerKind::ALL.iter()
        .copied()
        .filter_map(|kind| {
            let bytes = render_layer(item, kind, &original, seg.as_ref())?;
            Some((make_layer_filename(&item.filename, kind), bytes))
        })
        .collect()
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

    #[test]
    fn layer_filename_uses_kind_suffix() {
        assert_eq!(
            make_layer_filename("sunset.jpg", LayerKind::Subject),
            "sunset-subject.png"
        );
        assert_eq!(
            make_layer_filename("sunset.jpg", LayerKind::Lines),
            "sunset-lines.png"
        );
        assert_eq!(
            make_layer_filename("sunset.jpg", LayerKind::Mask),
            "sunset-mask.png"
        );
    }

    #[test]
    fn layer_filename_falls_back_on_empty_source() {
        assert_eq!(
            make_layer_filename("", LayerKind::Subject),
            "image-subject.png"
        );
    }

    #[test]
    fn render_layer_returns_none_when_tensors_missing() {
        use crate::gui::item::{BatchItem, ImageSource};
        use crate::gui::item_settings::ItemSettings;
        use std::sync::Arc;

        let item = BatchItem::new(
            1,
            "x.png".to_string(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (10, 10),
            ItemSettings::default(),
            String::new(),
        );
        let blank = DynamicImage::ImageRgba8(image::RgbaImage::new(10, 10));
        for kind in LayerKind::ALL {
            assert!(render_layer(&item, kind, &blank, None).is_none());
        }
    }
}
