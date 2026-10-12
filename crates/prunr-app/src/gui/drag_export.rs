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
//! - Filenames derived from source (e.g. `sunset.prunr.png`, `sunset-lines.png`).
//! - Not deleted at drag end — receiving apps may read async.
//! - Stale cleanup (>10 min old) runs at app start.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use image::DynamicImage;
use prunr_core::{FillStyle, MaskSettings};

use super::item::{BatchItem, ImageSource};
use super::item_settings::ItemSettings;
use super::settings::LineMode;
use super::worker::{CompressedEdgeTensors, CompressedTensor, SegBundle};

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

/// Remove ALL temp drag files. Called on graceful app exit.
pub(crate) fn cleanup_all() {
    crate::fs_util::sweep_dir_files_async(temp_dir());
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

/// `<source-stem>.prunr.<ext>`, the name a saved or dragged result gets.
pub(crate) fn export_name(source_filename: &str, format: prunr_core::ExportFormat) -> String {
    format!("{}.prunr.{}", source_stem(source_filename), format.extension())
}

/// Build a human-friendly filename stem for a batch item given its processing mode.
///
/// Naming convention: `.prunr` is a tool tag (signals "Prunr touched this") that
/// sits between the source stem and the `.png` extension; the lines-only mode
/// keeps a plain hyphen suffix because it isn't a Prunr-branded output kind.
fn make_filename(source_filename: &str, has_result: bool, line_mode: LineMode) -> String {
    if !has_result {
        // No processing yet — drag source as-is with original extension.
        return source_filename.to_string();
    }
    let stem = source_stem(source_filename);
    match line_mode {
        LineMode::Off => export_name(source_filename, prunr_core::ExportFormat::Png),
        LineMode::EdgesOnly => format!("{stem}-lines.png"),
        LineMode::SubjectOutline => format!("{stem}.prunr-lines.png"),
    }
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

/// What a layer export reads from an item, as shared handles: taken on
/// the UI thread in microseconds, rendered on a worker.
#[derive(Clone)]
pub(crate) struct LayerJob {
    item_id: u64,
    filename: String,
    settings: ItemSettings,
    start: Option<super::timeline::StoredImage>,
    source_rgba: Option<Arc<image::RgbaImage>>,
    source: ImageSource,
    correction: Option<Arc<prunr_core::selection::MaskArtifact>>,
    seg: Option<CompressedTensor>,
    edges: Option<CompressedEdgeTensors>,
}

impl LayerJob {
    pub(crate) fn of(item: &BatchItem, style: super::background_io::SelectionStyle, action: super::brush_state::CutoutAction) -> Self {
        Self {
            item_id: item.id,
            filename: item.filename.clone(),
            settings: item.settings,
            start: item.run_start(super::item::Engine::Segment).map(|s| s.image),
            source_rgba: item.source_rgba.clone(),
            source: item.source.clone(),
            correction: item.correction_plane(style, action),
            seg: item.cached_tensor.clone(),
            edges: item.cached_edge_tensors.clone(),
        }
    }

    pub(crate) fn filename(&self) -> &str {
        &self.filename
    }

    /// The item kept what layers are made from (they were dropped under
    /// memory pressure, or the image was never processed).
    pub(crate) fn has_layers(&self) -> bool {
        self.seg.is_some() || self.edges.is_some()
    }

    /// Every layer it has as `(filename, PNG bytes)`; empty without any.
    pub(crate) fn render_bytes(&self) -> Vec<(String, Vec<u8>)> {
        let Some(src) = LayerSource::of(self) else { return Vec::new() };
        let seg = self.seg.as_ref().and_then(CompressedTensor::bundle);
        LayerKind::ALL.iter()
            .copied()
            .filter_map(|kind| {
                let bytes = match render_layer(self, kind, &src, seg.as_ref())? {
                    LayerOutput::Rgba(rgba) => prunr_core::encode_rgba_png(&rgba).ok()?,
                    LayerOutput::Gray(gray) => prunr_core::encode_gray_png(&gray).ok()?,
                };
                Some((make_layer_filename(&self.filename, kind), bytes))
            })
            .collect()
    }
}

/// What the layers are built from, so they match the canvas: the image
/// the cut-out ran on (an erase or earlier result included) and the
/// selection the re-cut applies.
pub(crate) struct LayerSource {
    original: DynamicImage,
    correction: Option<Arc<prunr_core::selection::MaskArtifact>>,
}

impl LayerSource {
    fn of(job: &LayerJob) -> Option<Self> {
        let start = job.start.as_ref().and_then(|s| s.load()).or_else(|| job.source_rgba.clone());
        let original = match start {
            Some(rgba) => DynamicImage::ImageRgba8((*rgba).clone()),
            None => job.source.decode()
                .map_err(|err| tracing::warn!(item_id = job.item_id, %err, "layer export: source"))
                .ok()?,
        };
        Some(Self { original, correction: job.correction.clone() })
    }
}

fn render_layer(
    job: &LayerJob,
    kind: LayerKind,
    src: &LayerSource,
    seg: Option<&SegBundle>,
) -> Option<LayerOutput> {
    let original = &src.original;
    let correction = src.correction.as_deref();
    let item_id = job.item_id;
    match kind {
        LayerKind::Subject => {
            let seg = seg?;
            // A clean subject for the receiving app: no effect, outline or
            // shadow. The other mask knobs stay, so the guided-filter,
            // feather and edge-shift refinements survive.
            let mask = MaskSettings {
                fill_style: FillStyle::None,
                decoration: prunr_core::Decoration::default(),
                ..job.settings.mask_settings()
            };
            let rgba = prunr_core::postprocess_from_flat(
                &seg.data, seg.height as usize, seg.width as usize,
                original, &prunr_core::PostprocessOpts::new(&mask, seg.model).with_correction(correction),
            )
                .map_err(|err| tracing::warn!(item_id, %err, "subject layer postprocess"))
                .ok()?;
            Some(LayerOutput::Rgba(rgba))
        }
        LayerKind::Lines => {
            let edge = job.settings.edge_settings();
            let tensors = job.edges.as_ref()?.for_compose(&edge)?;
            let rgba = prunr_core::compose_whole_photo(&tensors, original, &edge);
            Some(LayerOutput::Rgba(rgba))
        }
        LayerKind::Mask => {
            let seg = seg?;
            let gray = prunr_core::tensor_to_mask_from_flat(
                &seg.data, seg.height as usize, seg.width as usize,
                original, &prunr_core::PostprocessOpts::new(&job.settings.mask_settings(), seg.model).with_correction(correction),
            )
                .map_err(|err| tracing::warn!(item_id, %err, "mask layer reshape"))
                .ok()?;
            Some(LayerOutput::Gray(gray))
        }
    }
}

/// Owned per-layer pixel buffer, format-tagged so `write_layer` can
/// stream-encode the right PNG variant without a `Vec<u8>` intermediate.
enum LayerOutput {
    Rgba(image::RgbaImage),
    Gray(image::GrayImage),
}

fn write_layer(
    job: &LayerJob,
    kind: LayerKind,
    src: &LayerSource,
    seg: Option<&SegBundle>,
) -> std::io::Result<Option<PathBuf>> {
    let Some(layer) = render_layer(job, kind, src, seg) else { return Ok(None) };
    let path = temp_dir().join(make_layer_filename(&job.filename, kind));
    let file = std::fs::File::create(&path)?;
    let mut w = std::io::BufWriter::new(file);
    let res: Result<(), prunr_core::CoreError> = match layer {
        LayerOutput::Rgba(rgba) => prunr_core::encode_rgba_png_into(&rgba, &mut w),
        // No streaming variant for Gray yet — encode_gray_png allocates
        // a small Vec (mask is single-channel, ~12 MB at 4 K vs ~50 MB
        // RGBA). Acceptable for this layer's traffic.
        LayerOutput::Gray(gray) => prunr_core::encode_gray_png(&gray)
            .and_then(|bytes| {
                std::io::Write::write_all(&mut w, &bytes)
                    .map_err(prunr_core::CoreError::Io)
            }),
    };
    res.map_err(|e| std::io::Error::other(e.to_string()))?;
    std::io::Write::flush(&mut w)?;
    Ok(Some(path))
}

/// Prepare a drag temp file for a batch item. Returns the absolute path.
/// If the item has no processed result, writes the raw source bytes as-is
/// (preserving original extension so file managers recognize it).
///
/// Bakes the per-item bg (color or image) into the PNG so the dropped
/// file matches the canvas — same parity rule as Save and clipboard
/// copy. Split-layer drags don't go through here (`render_layer` keeps
/// each layer clean for downstream compositing).
pub(crate) fn prepare(item: &BatchItem) -> std::io::Result<PathBuf> {
    let name = make_filename(&item.filename, item.result_rgba.is_some(), item.settings.line_mode);
    let path = temp_dir().join(&name);
    match &item.result_rgba {
        Some(rgba) => {
            let baked = item.export_bg().bake(rgba);
            // Stream-encode directly into BufWriter — saves the
            // ~5–30 MB intermediate Vec<u8> at 4 K.
            let file = std::fs::File::create(&path)?;
            let mut w = std::io::BufWriter::new(file);
            prunr_core::encode_rgba_png_into(&baked, &mut w)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            std::io::Write::flush(&mut w)?;
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
pub(crate) fn prepare_for_drag(
    item: &BatchItem,
    split: bool,
    style: super::background_io::SelectionStyle,
    action: super::brush_state::CutoutAction,
) -> std::io::Result<Vec<PathBuf>> {
    if !split {
        return Ok(vec![prepare(item)?]);
    }

    // One source image + one seg decompress — Subject and Mask share the
    // seg, Lines uses the edge tensor independently.
    let job = LayerJob::of(item, style, action);
    let Some(src) = LayerSource::of(&job) else {
        return Ok(vec![prepare(item)?]);
    };
    let seg = job.seg.as_ref().and_then(CompressedTensor::bundle);

    let paths: Vec<PathBuf> = LayerKind::ALL.iter()
        .copied()
        .filter_map(|k| write_layer(&job, k, &src, seg.as_ref()).transpose())
        .collect::<std::io::Result<Vec<_>>>()?;

    if paths.is_empty() {
        return Ok(vec![prepare(item)?]);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_filename_uses_source_stem_with_mode_suffix() {
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::Off),
            "sunset.prunr.png"
        );
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::EdgesOnly),
            "sunset-lines.png"
        );
        assert_eq!(
            make_filename("sunset.jpg", true, LineMode::SubjectOutline),
            "sunset.prunr-lines.png"
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
            "no-extension.prunr.png"
        );
        assert_eq!(
            make_filename("", true, LineMode::Off),
            "image.prunr.png"
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
        let blank = LayerSource { original: DynamicImage::ImageRgba8(image::RgbaImage::new(10, 10)), correction: None };
        let job = LayerJob::of(&item, crate::gui::background_io::SelectionStyle::with_fill(0.0), crate::gui::brush_state::CutoutAction::Erase);
        for kind in LayerKind::ALL {
            assert!(render_layer(&job, kind, &blank, None).is_none());
        }
    }

    /// The exported Lines layer draws what the canvas draws: Dual scale
    /// layers Fine and Bold, whatever single scale is stored.
    #[test]
    fn the_lines_layer_matches_the_whole_photo_compose_for_dual_scale() {
        use crate::gui::item::{BatchItem, ImageSource};
        use crate::gui::item_settings::ItemSettings;
        use crate::gui::worker::{CompressedEdgeTensors, EdgeTensorCache};
        use prunr_core::{EdgeScale, LineStyle};
        use std::sync::Arc;

        let (h, w) = (8u32, 8u32);
        let tensor = |on: bool| vec![if on { 10.0_f32 } else { -10.0 }; (h * w) as usize];
        let raw = EdgeTensorCache { tensors: [tensor(true), tensor(false), tensor(false), tensor(false)], height: h, width: w };
        let all_scales = prunr_core::EdgeInferenceResult { tensors: raw.tensors.clone(), height: h, width: w };
        let settings = ItemSettings {
            line_mode: LineMode::EdgesOnly,
            edge_scale: EdgeScale::Bold,
            line_style: LineStyle::DualScale { fine_color: [255, 0, 0], bold_color: [0, 0, 255] },
            ..ItemSettings::default()
        };
        let mut item = BatchItem::new(1, "x.png".to_string(), ImageSource::Bytes(Arc::new(Vec::new())), (16, 16), settings, String::new());
        item.cached_edge_tensors = CompressedEdgeTensors::from_raw(raw);
        let photo = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(16, 16, image::Rgba([90, 90, 90, 255])));

        let src = LayerSource { original: photo.clone(), correction: None };
        let job = LayerJob::of(&item, crate::gui::background_io::SelectionStyle::with_fill(0.0), crate::gui::brush_state::CutoutAction::Erase);
        let Some(LayerOutput::Rgba(layer)) = render_layer(&job, LayerKind::Lines, &src, None) else {
            panic!("the Lines layer renders from the cached lines");
        };
        let canvas = prunr_core::compose_whole_photo(&all_scales, &photo, &item.settings.edge_settings());
        assert_eq!(layer.as_raw(), canvas.as_raw());
        assert_eq!(layer.get_pixel(8, 8).0[..3], [255, 0, 0], "Fine lines draw in the fine color");
    }

    /// Layers show what the canvas shows: a restore stroke is in the mask
    /// and subject, and the subject's colors come from the image the
    /// cut-out ran on (an erase before it included), not the original file.
    #[test]
    fn layers_carry_the_brush_corrections_and_the_cut_out_s_start_image() {
        use crate::gui::item::{BatchItem, ImageSource};
        use crate::gui::item_settings::ItemSettings;
        use crate::gui::worker::{CompressedTensor, TensorCache};
        use std::sync::Arc;

        let (w, h) = (16u32, 16u32);
        let mut item = BatchItem::new(1, "x.png".to_string(), ImageSource::Bytes(Arc::new(Vec::new())), (w, h), ItemSettings::default(), String::new());
        let all_background = vec![-10.0_f32; 64];
        item.cached_tensor = CompressedTensor::from_raw(TensorCache { data: all_background, height: 8, width: 8, model: prunr_core::ModelKind::Silueta });
        let mut cells = vec![0i8; (w * h) as usize];
        cells[(5 * w + 5) as usize] = prunr_core::selection::FULL;
        let restore = Arc::new(prunr_core::selection::MaskArtifact::from_cells(w, h, cells));
        let erased = Arc::new(image::RgbaImage::from_pixel(w, h, image::Rgba([7, 8, 9, 255])));
        let src = LayerSource { original: DynamicImage::ImageRgba8((*erased).clone()), correction: Some(restore) };
        let seg = item.cached_tensor.as_ref().and_then(|t| t.bundle());

        let job = LayerJob::of(&item, crate::gui::background_io::SelectionStyle::with_fill(0.0), crate::gui::brush_state::CutoutAction::Erase);
        let Some(LayerOutput::Gray(mask)) = render_layer(&job, LayerKind::Mask, &src, seg.as_ref()) else { panic!("mask layer") };
        assert_eq!(mask.get_pixel(5, 5)[0], 255, "the restored pixel is in the mask");
        assert_eq!(mask.get_pixel(9, 9)[0], 0, "the rest stays out");
        let Some(LayerOutput::Rgba(subject)) = render_layer(&job, LayerKind::Subject, &src, seg.as_ref()) else { panic!("subject layer") };
        assert_eq!(subject.get_pixel(5, 5).0, [7, 8, 9, 255], "the subject is cut from the image the cut-out ran on");
    }

    /// A layer export is a snapshot the save thread renders: it moves to
    /// another thread, renders the subject and mask from a cached cut-out,
    /// and says when the image kept nothing to make layers from.
    #[test]
    fn a_layer_job_renders_on_another_thread() {
        use crate::gui::item::{BatchItem, ImageSource};
        use crate::gui::item_settings::ItemSettings;
        use crate::gui::worker::{CompressedTensor, TensorCache};

        let mut item = BatchItem::new(1, "photo.jpg".to_string(), ImageSource::Bytes(Arc::new(Vec::new())), (8, 8), ItemSettings::default(), String::new());
        item.source_rgba = Some(Arc::new(image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]))));
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        assert!(!LayerJob::of(&item, style, crate::gui::brush_state::CutoutAction::Erase).has_layers(), "nothing cached, no layers");

        item.cached_tensor = CompressedTensor::from_raw(TensorCache { data: vec![10.0; 64], height: 8, width: 8, model: prunr_core::ModelKind::Silueta });
        let job = LayerJob::of(&item, style, crate::gui::brush_state::CutoutAction::Erase);
        assert!(job.has_layers());
        let files = std::thread::spawn(move || job.render_bytes()).join().expect("render thread");
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["photo-subject.png", "photo-mask.png"]);
        assert!(files.iter().all(|(_, bytes)| bytes.starts_with(b"\x89PNG")));
    }

    /// The layers carry the edits already made to the cut-out, even while
    /// the brush only selects.
    #[test]
    fn the_layers_carry_the_cutout_edits() {
        use crate::gui::item::{BatchItem, ImageSource};
        use crate::gui::item_settings::ItemSettings;
        use crate::gui::worker::{CompressedTensor, TensorCache};

        let mut item = BatchItem::new(1, "photo.jpg".to_string(), ImageSource::Bytes(Arc::new(Vec::new())), (8, 8), ItemSettings::default(), String::new());
        item.source_rgba = Some(Arc::new(image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]))));
        item.cached_tensor = CompressedTensor::from_raw(TensorCache { data: vec![10.0; 64], height: 8, width: 8, model: prunr_core::ModelKind::Silueta });
        let erased_left: Vec<i8> = (0..64).map(|i| if i % 8 < 4 { -prunr_core::selection::FULL } else { 0 }).collect();
        item.edits = Some(Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, erased_left)));
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);

        let files = LayerJob::of(&item, style, crate::gui::brush_state::CutoutAction::Select).render_bytes();
        let (_, mask) = files.iter().find(|(n, _)| n == "photo-mask.png").expect("a mask layer");
        let mask = image::load_from_memory(mask).expect("png").to_luma8();
        assert_eq!(mask.get_pixel(0, 4)[0], 0, "the erased half is gone");
        assert_eq!(mask.get_pixel(7, 4)[0], 255, "the rest stays");
    }
}
