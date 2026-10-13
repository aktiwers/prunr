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
//! - Not deleted at drag end — receiving apps may read async. Files written
//!   ahead of a drag (on hover) stay too, so hovering again writes nothing.
//! - Stale cleanup (>10 min old) runs at app start.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use image::DynamicImage;
use prunr_core::{FillStyle, MaskSettings};

use super::item::{BatchItem, ExportDepth, ImageSource};
use super::item_settings::ItemSettings;
use super::settings::LineMode;
use super::worker::{CompressedEdgeTensors, CompressedTensor, SegBundle};

/// The `drag` crate needs a GTK window on Linux, which eframe can't give it.
pub(crate) const DRAG_OUT_SUPPORTED: bool = cfg!(any(target_os = "windows", target_os = "macos"));

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

/// The folder one image's drag files go in. Each image has its own, so
/// two images with one file name (IMG_1.jpg from two folders) never write
/// over each other, and the dropped files keep their plain names.
fn item_dir(item_id: u64) -> PathBuf {
    let dir = temp_dir().join(item_id.to_string());
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Check whether a path was created by our own drag-out (i.e. lives under prunr-drag/).
/// Used to reject self-originated drops so our own thumbnails can't be re-ingested.
pub(crate) fn is_self_drop(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == DRAG_SUBDIR)
}

/// Remove ALL temp drag files. Called on graceful app exit, and waited
/// for: the folder holds a handful of files, and a sweep left on a thread
/// can die with the process.
pub(crate) fn cleanup_all() {
    let Ok(entries) = std::fs::read_dir(temp_dir()) else { return };
    for entry in entries.flatten() {
        remove_entry(&entry.path());
    }
}

/// A drag file, or an image's folder of them.
fn remove_entry(path: &Path) {
    let _ = if path.is_dir() { std::fs::remove_dir_all(path) } else { std::fs::remove_file(path) };
}

/// Remove stale temp files left behind by previous sessions or crashes.
/// Silent on errors — this is best-effort housekeeping.
pub(crate) fn cleanup_stale() {
    let dir = temp_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        let age = meta
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .unwrap_or(Duration::ZERO);
        if age > STALE_AGE {
            remove_entry(&entry.path());
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
    /// The faces Restore faces blends into the photo, as on the canvas.
    faces: Option<super::live_preview::FacesInput>,
    /// The photo's depth map, for lines that fade with distance.
    depth: super::item::DepthSource,
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
            faces: item.faces_input(super::item::FaceStage::Photo),
            depth: item.depth_source(),
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
    pub(crate) fn render_bytes(&self, depth: &ExportDepth) -> Vec<(String, Vec<u8>)> {
        let Some(src) = LayerSource::of(self) else { return Vec::new() };
        let seg = self.seg.as_ref().and_then(CompressedTensor::bundle);
        LayerKind::ALL.iter()
            .copied()
            .filter_map(|kind| {
                let bytes = match render_layer(self, kind, &src, seg.as_ref(), depth)? {
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
        let original = match (&job.faces, start) {
            (Some(super::live_preview::FacesInput { faced: Some(faced), .. }), _) => (**faced).clone(),
            (Some(faces), Some(rgba)) => DynamicImage::ImageRgba8(prunr_core::face::compose_faces(&rgba, &faces.key.faces, faces.key.blend)),
            (None, Some(rgba)) => DynamicImage::ImageRgba8((*rgba).clone()),
            (_, None) => job.source.decode()
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
    depth: &ExportDepth,
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
            let amount = job.settings.line_fade();
            let depth = (amount > 0.0)
                .then(|| job.depth.get(depth, job.source_rgba.as_deref())
                    .map_err(|err| tracing::warn!(item_id, %err, "layer export: depth"))
                    .ok())
                .flatten();
            let fade = depth.as_deref().map(|depth| prunr_core::depth::LineFade { depth, amount });
            let rgba = prunr_core::compose_whole_photo(&tensors, original, &edge, fade);
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
    depth: &ExportDepth,
) -> std::io::Result<Option<PathBuf>> {
    let Some(layer) = render_layer(job, kind, src, seg, depth) else { return Ok(None) };
    let path = item_dir(job.item_id).join(make_layer_filename(&job.filename, kind));
    write_atomically(&path, |w| match layer {
        LayerOutput::Rgba(rgba) => prunr_core::encode_rgba_png_into(&rgba, w),
        // No streaming variant for Gray yet — encode_gray_png allocates
        // a small Vec (mask is single-channel, ~12 MB at 4 K vs ~50 MB
        // RGBA). Acceptable for this layer's traffic.
        LayerOutput::Gray(gray) => prunr_core::encode_gray_png(&gray)
            .and_then(|bytes| std::io::Write::write_all(w, &bytes).map_err(prunr_core::CoreError::Io)),
    })?;
    Ok(Some(path))
}

/// Write `path` through a sibling temp file renamed into place, so an app
/// still reading an earlier drop of the same name never sees half a file.
fn write_atomically(
    path: &Path,
    encode: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> Result<(), prunr_core::CoreError>,
) -> std::io::Result<()> {
    let part = path.with_extension("part");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&part)?);
    encode(&mut w).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::io::Write::flush(&mut w)?;
    drop(w);
    std::fs::rename(&part, path)
}

/// One item's composite drag file: the result with its background baked
/// in, like Save and Copy, or the source bytes as-is before processing
/// (keeping the extension so file managers recognize it).
#[derive(Clone)]
struct Composite {
    item_id: u64,
    filename: String,
    result: Option<(super::item::ResultRef, super::item::ExportBg)>,
    source: ImageSource,
}

impl Composite {
    fn of(item: &BatchItem) -> Self {
        Self {
            item_id: item.id,
            filename: make_filename(&item.filename, item.result_ref().is_some(), item.settings.line_mode),
            result: item.result_ref().map(|r| (r, item.export_bg())),
            source: item.source.clone(),
        }
    }

    fn write(&self, depth: &ExportDepth) -> std::io::Result<PathBuf> {
        let path = item_dir(self.item_id).join(&self.filename);
        match &self.result {
            Some((result, bg)) => {
                let rgba = result.load().ok_or_else(|| std::io::Error::other("the result is no longer there"))?;
                let baked = bg.bake(&rgba, depth);
                write_atomically(&path, |w| prunr_core::encode_rgba_png_into(&baked, w))?;
            }
            None => {
                let bytes = self.source.load_bytes()?;
                write_atomically(&path, |w| std::io::Write::write_all(w, &bytes).map_err(prunr_core::CoreError::Io))?;
            }
        }
        Ok(path)
    }
}

/// How a drag writes each item: as layers or one composite, and the
/// correction the layers apply.
#[derive(Clone, Copy)]
pub(crate) struct DragSpec {
    pub split: bool,
    pub style: super::background_io::SelectionStyle,
    pub action: super::brush_state::CutoutAction,
}

/// What dragging one item writes, as shared handles taken on the UI
/// thread: the composite, or its layers (subject / lines / mask) when
/// layers are on. Layers the item no longer has are skipped; with none,
/// the composite goes instead so the drag still lands a file.
#[derive(Clone)]
pub(crate) struct DragFiles {
    composite: Composite,
    layers: Option<LayerJob>,
}

impl DragFiles {
    pub(crate) fn of(item: &BatchItem, spec: DragSpec) -> Self {
        Self { composite: Composite::of(item), layers: spec.split.then(|| LayerJob::of(item, spec.style, spec.action)) }
    }

    /// Render and write the files; slow on a big image, so off the UI
    /// thread where it can be.
    pub(crate) fn write(&self, depth: &ExportDepth) -> std::io::Result<Vec<PathBuf>> {
        if let Some(job) = &self.layers {
            if let Some(src) = LayerSource::of(job) {
                // One seg decompress shared by Subject and Mask.
                let seg = job.seg.as_ref().and_then(CompressedTensor::bundle);
                let paths = LayerKind::ALL.iter()
                    .filter_map(|&k| write_layer(job, k, &src, seg.as_ref(), depth).transpose())
                    .collect::<std::io::Result<Vec<_>>>()?;
                if !paths.is_empty() {
                    return Ok(paths);
                }
            }
        }
        Ok(vec![self.composite.write(depth)?])
    }
}

/// Write each item's files, `writers()` at a time (each write holds a
/// baked copy of its image); the results come back in the order given.
pub(crate) fn write_all(jobs: Vec<DragFiles>) -> Vec<std::io::Result<Vec<PathBuf>>> {
    use rayon::prelude::*;
    let depth = ExportDepth::default();
    match rayon::ThreadPoolBuilder::new().num_threads(super::drag_export_state::writers()).build() {
        Ok(pool) => pool.install(|| jobs.par_iter().map(|job| job.write(&depth)).collect()),
        Err(err) => {
            tracing::warn!(%err, "drag writer pool failed to start; writing one at a time");
            jobs.iter().map(|job| job.write(&depth)).collect()
        }
    }
}

/// What a drag's files were made from, to tell whether files written
/// earlier still show the item. Images are held weakly: the key keeps no
/// pixels alive, and a live weak handle keeps its address from reuse.
pub(crate) struct DragKey {
    settings: ItemSettings,
    result: Option<std::sync::Weak<image::RgbaImage>>,
    bg_image: Option<std::sync::Weak<super::item::BgImage>>,
    backdrop: Option<u64>,
    layers: Option<LayerKey>,
}

/// What layers add to a drag's key: the correction they apply and the
/// tensors they are made from.
struct LayerKey {
    style: super::background_io::SelectionStyle,
    action: super::brush_state::CutoutAction,
    selection: Option<u64>,
    edits: Option<std::sync::Weak<prunr_core::selection::MaskArtifact>>,
    seg: bool,
    edges: bool,
}

impl DragKey {
    pub(crate) fn of(item: &BatchItem, spec: DragSpec) -> Self {
        let DragSpec { split, style, action } = spec;
        Self {
            settings: item.settings,
            result: item.result_rgba.as_ref().map(Arc::downgrade),
            bg_image: item.bg_image.as_ref().map(Arc::downgrade),
            backdrop: item.backdrop_key(),
            layers: split.then(|| LayerKey {
                style,
                action,
                selection: item.selection_hash,
                edits: item.edits.as_ref().map(Arc::downgrade),
                seg: item.cached_tensor.is_some(),
                edges: item.cached_edge_tensors.is_some(),
            }),
        }
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        fn same<T>(a: &Option<std::sync::Weak<T>>, b: &Option<std::sync::Weak<T>>) -> bool {
            match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => a.ptr_eq(b),
                _ => false,
            }
        }
        self.settings == other.settings
            && same(&self.result, &other.result)
            && same(&self.bg_image, &other.bg_image)
            && self.backdrop == other.backdrop
            && match (&self.layers, &other.layers) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    (a.style, a.action, a.selection, a.seg, a.edges) == (b.style, b.action, b.selection, b.seg, b.edges)
                        && same(&a.edits, &b.edits)
                }
                _ => false,
            }
    }
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

    /// A ticked image off screen has its result parked in the history; a
    /// drag still carries that result, not the original photo.
    #[test]
    fn a_parked_result_is_what_a_drag_carries() {
        use crate::gui::brush_state::CutoutAction;
        let spec = DragSpec { split: false, style: crate::gui::background_io::SelectionStyle::with_fill(0.0), action: CutoutAction::Restore };
        let mut item = processed_item(33, "parked.jpg");
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 200, 0, 255]))));
        item.commit(crate::gui::timeline::StepKind::Process);
        assert!(item.park_result());
        assert!(item.result_rgba.is_none());
        let paths = DragFiles::of(&item, spec).write(&Default::default()).unwrap();
        assert!(paths[0].ends_with("parked.prunr.png"), "{:?}", paths[0]);
        assert_eq!(image::open(&paths[0]).unwrap().to_rgba8().get_pixel(0, 0).0, [0, 200, 0, 255]);
        let _ = std::fs::remove_file(&paths[0]);
    }

    /// Two images with one file name get their own drag files.
    #[test]
    fn two_images_with_one_name_keep_their_own_drag_files() {
        use crate::gui::brush_state::CutoutAction;
        let spec = DragSpec { split: false, style: crate::gui::background_io::SelectionStyle::with_fill(0.0), action: CutoutAction::Restore };
        let mut a = processed_item(31, "same-name.jpg");
        a.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]))));
        let mut b = processed_item(32, "same-name.jpg");
        b.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 255, 255]))));
        let (pa, pb) = (DragFiles::of(&a, spec).write(&Default::default()).unwrap(), DragFiles::of(&b, spec).write(&Default::default()).unwrap());
        assert_ne!(pa[0], pb[0]);
        assert_eq!(pa[0].file_name(), pb[0].file_name(), "the dropped files keep the plain name");
        let red = image::open(&pa[0]).unwrap().to_rgba8();
        assert_eq!(red.get_pixel(0, 0).0, [255, 0, 0, 255], "a's file is still a's");
        for p in [&pa[0], &pb[0]] {
            let _ = std::fs::remove_file(p);
        }
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
            assert!(render_layer(&job, kind, &blank, None, &Default::default()).is_none());
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
        let tensor = |scale: EdgeScale| {
            let (sh, sw) = prunr_core::edge_scale_dims(scale, h, w);
            vec![if scale == EdgeScale::Fine { 1.0_f32 } else { 0.0 }; (sh * sw) as usize]
        };
        let raw = EdgeTensorCache { tensors: EdgeScale::ALL.map(tensor), height: h, width: w };
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
        let Some(LayerOutput::Rgba(layer)) = render_layer(&job, LayerKind::Lines, &src, None, &Default::default()) else {
            panic!("the Lines layer renders from the cached lines");
        };
        let canvas = prunr_core::compose_whole_photo(&all_scales, &photo, &item.settings.edge_settings(), None);
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
        let Some(LayerOutput::Gray(mask)) = render_layer(&job, LayerKind::Mask, &src, seg.as_ref(), &Default::default()) else { panic!("mask layer") };
        assert_eq!(mask.get_pixel(5, 5)[0], 255, "the restored pixel is in the mask");
        assert_eq!(mask.get_pixel(9, 9)[0], 0, "the rest stays out");
        let Some(LayerOutput::Rgba(subject)) = render_layer(&job, LayerKind::Subject, &src, seg.as_ref(), &Default::default()) else { panic!("subject layer") };
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
        let files = std::thread::spawn(move || job.render_bytes(&Default::default())).join().expect("render thread");
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

        let files = LayerJob::of(&item, style, crate::gui::brush_state::CutoutAction::Select).render_bytes(&Default::default());
        let (_, mask) = files.iter().find(|(n, _)| n == "photo-mask.png").expect("a mask layer");
        let mask = image::load_from_memory(mask).expect("png").to_luma8();
        assert_eq!(mask.get_pixel(0, 4)[0], 0, "the erased half is gone");
        assert_eq!(mask.get_pixel(7, 4)[0], 255, "the rest stays");
    }

    fn processed_item(id: u64, name: &str) -> crate::gui::item::BatchItem {
        use crate::gui::item::{BatchItem, BatchStatus, ImageSource};
        let mut item = BatchItem::new(id, name.to_string(), ImageSource::Bytes(Arc::new(Vec::new())), (4, 4), ItemSettings::default(), String::new());
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([7, 8, 9, 255]))));
        item.status = BatchStatus::Done;
        item
    }

    /// Files written ahead stand for the item only while it shows what
    /// they were made from: a new result or another background stales them.
    #[test]
    fn a_drag_key_follows_what_the_files_show() {
        use crate::gui::brush_state::CutoutAction;
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        let spec = |split| DragSpec { split, style, action: CutoutAction::Restore };
        let mut item = processed_item(1, "key.jpg");
        let key = DragKey::of(&item, spec(false));
        assert!(key.matches(&DragKey::of(&item, spec(false))));
        assert!(!key.matches(&DragKey::of(&item, spec(true))), "layers write other files");
        item.settings.bg = Some([255, 0, 0, 255]);
        assert!(!key.matches(&DragKey::of(&item, spec(false))), "another background");
        item.settings.bg = None;
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        assert!(!key.matches(&DragKey::of(&item, spec(false))), "a new result");
    }

    /// A drag takes the files written ahead on a worker; once the item
    /// changes, it writes them anew instead.
    #[test]
    fn a_drag_takes_the_files_written_ahead_while_current() {
        use crate::gui::brush_state::CutoutAction;
        use crate::gui::drag_export_state::DragExportState;
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        let spec = |split| DragSpec { split, style, action: CutoutAction::Restore };
        let mut item = processed_item(9, "ahead-test.jpg");
        let mut state = DragExportState::new();
        let key = |item: &crate::gui::item::BatchItem| DragKey::of(item, spec(false));
        state.prewrite(9, key(&item), DragFiles::of(&item, spec(false)));
        assert!(state.has_current(9, &key(&item)));

        let paths = state.take_written(9, &key(&item)).expect("written ahead");
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("ahead-test.prunr.png"));
        assert!(std::fs::read(&paths[0]).expect("file").starts_with(b"\x89PNG"));
        assert!(state.take_written(9, &key(&item)).is_some(), "kept for the next drag");

        let other = processed_item(10, "ahead-other.jpg");
        state.prewrite(10, key(&other), DragFiles::of(&other, spec(false)));
        assert!(state.has_current(9, &key(&item)), "hovering another image keeps these: no rewrite on return");

        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        assert!(state.take_written(9, &key(&item)).is_none(), "stale: the drag writes anew");
        if let Some(other_paths) = state.take_written(10, &key(&other)) {
            let _ = std::fs::remove_file(&other_paths[0]);
        }
        let _ = std::fs::remove_file(&paths[0]);
    }

    /// A write ahead that failed (the photo moved, the disk filled) is
    /// dropped: the next hover tries again and the drag writes fresh.
    #[test]
    fn a_failed_write_ahead_is_tried_again() {
        use crate::gui::brush_state::CutoutAction;
        use crate::gui::drag_export_state::DragExportState;
        use crate::gui::item::{BatchItem, ImageSource};
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        let spec = DragSpec { split: false, style, action: CutoutAction::Restore };
        let gone = ImageSource::Path(PathBuf::from("/nonexistent/prunr-drag-gone.png"));
        let item = BatchItem::new(11, "gone.png".to_string(), gone, (4, 4), ItemSettings::default(), String::new());
        let mut state = DragExportState::new();
        state.prewrite(11, DragKey::of(&item, spec), DragFiles::of(&item, spec));
        while state.is_writing(11) {
            std::thread::yield_now();
        }
        assert!(!state.has_current(11, &DragKey::of(&item, spec)), "a hover writes again");
        state.prewrite(11, DragKey::of(&item, spec), DragFiles::of(&item, spec));
        assert!(state.take_written(11, &DragKey::of(&item, spec)).is_none(), "the drag writes fresh");
    }

    /// A drag of many images writes a few at a time, the hovered image
    /// first: one write per image at once held a baked copy of each.
    #[test]
    fn many_drag_files_are_written_a_few_at_a_time_hovered_first() {
        use crate::gui::brush_state::CutoutAction;
        use crate::gui::drag_export_state::{writers, DragExportState};
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        let spec = DragSpec { split: false, style, action: CutoutAction::Restore };
        let items: Vec<_> = (100..112).map(|id| processed_item(id, &format!("pool{id}.jpg"))).collect();
        let targets: Vec<u64> = items.iter().map(|i| i.id).collect();
        let key_of = |id: u64| items.iter().find(|i| i.id == id).map(|i| DragKey::of(i, spec));
        let files_of = |id: u64| items.iter().find(|i| i.id == id).map(|i| DragFiles::of(i, spec));
        let mut state = DragExportState::new();
        let now = std::time::Instant::now();
        state.note_hover(Some((107, true)), now);
        let more = state.queue_writes(107, &targets, key_of, files_of);
        assert!(more, "the rest wait their turn");
        assert_eq!(state.prewrite_count(), writers(), "only a few at once");
        assert!(state.has_current(107, &DragKey::of(&items[7], spec)), "the hovered image first");
        let mut rounds = 0;
        while state.queue_writes(107, &targets, key_of, files_of) {
            std::thread::yield_now();
            rounds += 1;
            assert!(rounds < 1_000_000, "the queue drains");
        }
        assert_eq!(state.prewrite_count(), targets.len());
        assert_eq!(state.due(now), None, "all written: nothing is checked again every frame");
        state.note_hover(Some((108, true)), now);
        assert_eq!(state.due(now), Some(108), "another thumbnail is checked");
        for (id, item) in targets.iter().zip(&items) {
            for path in state.take_written(*id, &DragKey::of(item, spec)).expect("written") {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// Files the drag still needs are written in parallel, in the order
    /// asked.
    #[test]
    fn files_missing_at_the_drag_are_written_in_order() {
        use crate::gui::brush_state::CutoutAction;
        let style = crate::gui::background_io::SelectionStyle::with_fill(0.0);
        let spec = DragSpec { split: false, style, action: CutoutAction::Restore };
        let items: Vec<_> = (200..206).map(|id| processed_item(id, &format!("order{id}.jpg"))).collect();
        let written = write_all(items.iter().map(|i| DragFiles::of(i, spec)).collect());
        for (result, item) in written.into_iter().zip(&items) {
            let paths = result.expect("written");
            assert!(paths[0].ends_with(format!("order{}.prunr.png", item.id)));
            let _ = std::fs::remove_file(&paths[0]);
        }
    }
}

