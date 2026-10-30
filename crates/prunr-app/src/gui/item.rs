//! `BatchItem`: one image's state (source, result, caches, settings,
//! textures, job) and the per-image logic over it. It holds egui texture
//! handles because their lifecycle is per-item, but no rendering happens
//! here; batch-wide decisions live in `BatchManager` and `Processor`.

use std::path::PathBuf;
use std::sync::Arc;

use super::brush_state::CutoutAction;

/// An upscale run kept for Refine: the model's `raw` output, the `input`
/// it read (after Denoise), and that input enlarged to the output size
/// (`reference`) once AI blend or Color match needed it. The raw stays
/// in memory unless memory runs low while its image is off screen
/// (`BatchManager::move_to_disk_under_pressure`); the next Refine reads it
/// back.
#[derive(Clone)]
pub(crate) struct UpscaleCache {
    pub(crate) raw: super::timeline::StoredImage,
    pub(crate) input: Arc<image::RgbaImage>,
    pub(crate) reference: Option<Arc<prunr_core::upscale::Reference>>,
    /// Which run this is, for the step it lands on to claim.
    run: u64,
}

impl UpscaleCache {
    pub(crate) fn new(raw: Arc<image::RgbaImage>, input: Arc<image::RgbaImage>, reference: Option<Arc<prunr_core::upscale::Reference>>) -> Self {
        Self { raw: super::timeline::Stored::new(raw), input, reference, run: next_run() }
    }

    /// Store the raw compressed on disk and let the reference go: a
    /// Refine reads the raw back and makes the reference again.
    pub(crate) fn park(&mut self) -> Option<super::timeline::CompressJob> {
        super::reclaim::later(self.reference.take());
        self.raw.compress_job()
    }
}

/// A colorize run kept for Strength: the predicted colour and the image
/// it was predicted for.
#[derive(Clone)]
pub(crate) struct ColorizeCache {
    pub(crate) plane: Arc<prunr_core::colorize::ColorPlane>,
    pub(crate) input: Arc<image::RgbaImage>,
    run: u64,
}

impl ColorizeCache {
    pub(crate) fn new(plane: Arc<prunr_core::colorize::ColorPlane>, input: Arc<image::RgbaImage>) -> Self {
        Self { plane, input, run: next_run() }
    }

    /// The image at `strength`.
    pub(crate) fn render(&self, strength: f32) -> image::RgbaImage {
        prunr_core::colorize::colorized(&self.input, &self.plane, strength)
    }
}

/// Both empty, or both the very same allocation.
pub(crate) fn same_arc<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// A fresh run id, for a job's result and a kept run's step to claim;
/// never 0, which a work item without a job carries.
fn next_run() -> u64 {
    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A result's background for export, baked by `bake` (in a save thread
/// where there is one: an effect blurs the full photo).
#[derive(Clone)]
pub(crate) struct ExportBg {
    backdrop: super::item_settings::Backdrop,
    image: Option<Arc<BgImage>>,
    fit: prunr_core::BgImageFit,
    photo: Option<Pixels>,
    /// A lens blur's depth map, and the original an effect is made from
    /// when `photo` has none.
    depth: DepthSource,
}

/// An image's depth map for export: the item's, or measured from its
/// original photo in the export's thread.
#[derive(Clone)]
pub(crate) struct DepthSource {
    depth: Option<Arc<prunr_core::depth::DepthMap>>,
    source: ImageSource,
}

impl DepthSource {
    /// The map; without one, measured from `original` (the photo already
    /// decoded) or from the source decoded now.
    pub(crate) fn get(&self, export: &ExportDepth, original: Option<&image::RgbaImage>) -> Result<Arc<prunr_core::depth::DepthMap>, String> {
        match &self.depth {
            Some(depth) => Ok(Arc::clone(depth)),
            None => export.measure(&self.source, original),
        }
    }
}

/// The depth one export (a save, a copy, a drag) measures: one engine
/// for all its images, and the last map kept, so an image's backdrop,
/// faded lines and Depth map file measure it once between them.
#[derive(Default)]
pub(crate) struct ExportDepth {
    measurer: prunr_core::depth::Measurer,
    last: std::sync::Mutex<Option<(ImageSource, Arc<prunr_core::depth::DepthMap>)>>,
}

impl ExportDepth {
    fn measure(&self, source: &ImageSource, original: Option<&image::RgbaImage>) -> Result<Arc<prunr_core::depth::DepthMap>, String> {
        let last = || self.last.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, map)) = last().as_ref().filter(|(seen, _)| seen.same_as(source)) {
            return Ok(Arc::clone(map));
        }
        let measured = match original {
            Some(photo) => self.measurer.measure(photo),
            None => self.measurer.measure_decoded(&source.decode()?),
        };
        let map = Arc::new(measured.map_err(|e| e.to_string())?);
        *last() = Some((source.clone(), Arc::clone(&map)));
        Ok(map)
    }
}

/// An image's depth map to save beside its result as `<stem>-depth.png`
/// (16-bit grey, white = near) at the photo's size.
#[derive(Clone)]
pub(crate) struct DepthExport {
    name: String,
    dims: (u32, u32),
    depth: DepthSource,
}

/// The depth map's file name for an image named `filename`.
fn depth_file_name(filename: &str) -> String {
    let stem = std::path::Path::new(filename).file_stem().map_or_else(|| filename.into(), |s| s.to_string_lossy());
    prunr_core::depth::map_file_name(&stem)
}

impl DepthExport {
    /// The map written under `filename`'s stem.
    pub(crate) fn named(self, filename: &str) -> Self {
        Self { name: depth_file_name(filename), ..self }
    }

    /// Write the map into `folder`.
    pub(crate) fn write(&self, folder: &std::path::Path, export: &ExportDepth) -> Result<(), String> {
        let depth = self.depth.get(export, None)?;
        let (w, h) = self.dims;
        let bytes = prunr_core::depth::encode_depth_png(&depth, w, h).map_err(|e| e.to_string())?;
        crate::fs_util::write_bytes_replacing(&folder.join(&self.name), &bytes).map_err(|e| e.to_string())
    }
}

impl ExportBg {
    /// Nothing goes behind the result: baking returns it as it is.
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self.backdrop, super::item_settings::Backdrop::Transparent)
    }

    pub(crate) fn bake(&self, rgba: &Arc<image::RgbaImage>, export: &ExportDepth) -> Arc<image::RgbaImage> {
        use super::item_settings::Backdrop;
        let composite = |under: &dyn Fn(&mut image::RgbaImage)| {
            let mut copy = (**rgba).clone();
            under(&mut copy);
            Arc::new(copy)
        };
        let baked = match self.backdrop {
            Backdrop::Transparent => None,
            Backdrop::Color(c) => Some(composite(&|img| prunr_core::apply_background_color(img, c))),
            Backdrop::Image(_) => self.image.as_ref()
                .map(|bg| composite(&|img| prunr_core::apply_background_image(img, &bg.image, self.fit))),
            Backdrop::Effect(effect) => self.photo().map(|(photo, original)| {
                let depth = if effect.needs_depth() {
                    self.depth.get(export, original.then_some(&*photo))
                        .map_err(|err| tracing::warn!(%err, "depth for export"))
                        .ok()
                } else {
                    None
                };
                composite(&|img| { prunr_core::apply_bg_effect(img, &photo, effect, depth.as_deref()); })
            }),
        };
        baked.unwrap_or_else(|| Arc::clone(rgba))
    }

    /// The photo the effect is made from, and whether it is the original.
    fn photo(&self) -> Option<(Arc<image::RgbaImage>, bool)> {
        let chosen = self.photo.as_ref()
            .and_then(|p| Some((p.load()?, matches!(p, Pixels::Resident(_)))));
        chosen.or_else(|| self.depth.source.decode_rgba().ok().map(|photo| (photo, true)))
    }
}

/// `image` is `Arc`-wrapped so cloning across threads (canvas paint and
/// save worker each take a handle) is a refcount bump, not a memcpy of
/// up to ~48 MB.
pub(crate) struct BgImage {
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) image: Arc<image::DynamicImage>,
    pub(crate) hash: u64,
}

/// `DefaultHasher` (SipHasher13) is deterministic across runs within a
/// stdlib version, so a hash persisted in a preset survives reload.
pub(crate) fn bg_image_content_hash(img: &image::DynamicImage) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    img.width().hash(&mut h);
    img.height().hash(&mut h);
    img.as_bytes().hash(&mut h);
    h.finish()
}

/// Mixed into a backdrop's key when it tiles: the tiled picture is
/// uploaded to wrap around, the others clamp at the edge.
const TILED_BACKDROP: u64 = 0x9e37_79b9_7f4a_7c15;

/// Clear a pending texture request if it is still the one for `key`.
fn clear_if<K: PartialEq>(pending: &mut Option<K>, key: K) {
    if pending.as_ref() == Some(&key) {
        *pending = None;
    }
}

/// A step back or forward (`BatchItem::undo_step` / `redo_step`).
pub(crate) type StepFn = fn(&mut BatchItem) -> Result<StepKind, StepRefused>;

/// Why a step back or forward did not happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepRefused {
    /// There is no step that way.
    Nothing,
    /// The step's saved state is gone (its file removed); the image stays
    /// on the step it was.
    Lost,
}

/// The sidebar thumbnail: the one shown, and whether a new one is being
/// made. The one shown stays while a new one is made, so a result never
/// blinks to a spinner.
#[derive(Default)]
pub(crate) struct Thumbnail {
    texture: Option<egui::TextureHandle>,
    build: ThumbBuild,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum ThumbBuild {
    #[default]
    Idle,
    Pending,
    /// It could not be made from what the image has; not asked for again
    /// until the image has something new.
    Failed,
}

impl Thumbnail {
    pub(crate) fn texture(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.build == ThumbBuild::Pending
    }

    /// None shown, none on its way, and none that failed.
    pub(crate) fn wanted(&self) -> bool {
        self.texture.is_none() && self.build == ThumbBuild::Idle
    }

    /// A new one is on its way; the one shown stays until it lands.
    pub(crate) fn asked(&mut self) {
        self.build = ThumbBuild::Pending;
    }

    pub(crate) fn landed(&mut self, texture: egui::TextureHandle) {
        self.texture = Some(texture);
        self.build = ThumbBuild::Idle;
    }

    pub(crate) fn failed(&mut self) {
        self.build = ThumbBuild::Failed;
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

const FILE_GONE: &str = "The file was moved or deleted since it was opened. Open it again.";

/// The file was written since it was opened: it is not the photo the
/// steps were made on.
#[derive(Debug)]
struct FileChanged;

impl std::fmt::Display for FileChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("The file changed since it was opened. Open it again to use the new version.")
    }
}

impl std::error::Error for FileChanged {}

/// The error an image shows when its file cannot be read.
pub(crate) fn source_unreadable(err: std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::NotFound {
        FILE_GONE.to_string()
    } else if err.get_ref().is_some_and(|inner| inner.is::<FileChanged>()) {
        err.to_string()
    } else {
        format!("Could not read the file: {err}")
    }
}

/// A file's length and last write when it was opened; an edit since
/// changes one, whatever size the image now decodes at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FileStamp {
    pub(crate) len: u64,
    modified: Option<std::time::SystemTime>,
}

impl FileStamp {
    pub(crate) fn of(meta: &std::fs::Metadata) -> Self {
        Self { len: meta.len(), modified: meta.modified().ok() }
    }

    /// The file at `path` as it is now; `None` when it cannot be looked at.
    pub(crate) fn at(path: &std::path::Path) -> Option<Self> {
        std::fs::metadata(path).ok().map(|m| Self::of(&m))
    }
}

/// Where an image's raw bytes live — file path (lazy) or in-memory (clipboard/paste).
#[derive(Clone)]
pub(crate) enum ImageSource {
    /// Loaded from a file. Bytes read on demand and dropped after use;
    /// no stamp when the file could not be looked at on open.
    Path(PathBuf, Option<FileStamp>),
    /// From clipboard, drag-drop, or CLI pipe. Bytes kept in memory.
    Bytes(Arc<Vec<u8>>),
}

impl ImageSource {
    /// A file as it is now: later reads refuse it once it is written again.
    pub(crate) fn file(path: PathBuf) -> Self {
        let stamp = FileStamp::at(&path);
        Self::Path(path, stamp)
    }

    /// Decode the source to straight RGBA.
    pub(crate) fn decode_rgba(&self) -> Result<Arc<image::RgbaImage>, String> {
        self.decode().map(|img| Arc::new(img.into_rgba8()))
    }

    pub(crate) fn decode(&self) -> Result<image::DynamicImage, String> {
        let bytes = self.load_bytes().map_err(|e| e.to_string())?;
        prunr_core::load_image_from_bytes(&bytes).map_err(|e| e.to_string())
    }

    /// Whether both name the same image: one path, or one shared buffer.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Path(a, _), Self::Path(b, _)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub(crate) fn load_bytes(&self) -> std::io::Result<Arc<Vec<u8>>> {
        match self {
            Self::Path(path, stamp) => {
                use std::io::Read;
                let mut file = std::fs::File::open(path)?;
                let meta = file.metadata()?;
                if stamp.is_some_and(|stamp| stamp != FileStamp::of(&meta)) {
                    return Err(std::io::Error::other(FileChanged));
                }
                let mut bytes = Vec::with_capacity(meta.len() as usize);
                file.read_to_end(&mut bytes)?;
                Ok(Arc::new(bytes))
            }
            Self::Bytes(bytes) => Ok(bytes.clone()),
        }
    }

}

use super::timeline::{store, DocState, Origin, RunStart, StepKind, Timeline};

/// A transition to a new image: first the image on screen is held while
/// the new one's texture is built, then the new one reveals over it.
pub(crate) enum Reveal {
    Held(super::image_texture::ImageTexture),
    /// The image revealed over, the start, and where the reveal starts
    /// (the subject, when one was found; corner to corner otherwise).
    Running(super::image_texture::ImageTexture, std::time::Instant, Option<Arc<[f32]>>),
}

/// The subject a background-removal run found, outlined on the image
/// while its edges are refined.
pub(crate) struct SubjectOutline {
    pub(crate) texture: egui::TextureHandle,
    pub(crate) shown_at: std::time::Instant,
    /// Per reveal-mesh vertex: 0 on the subject, rising outward.
    pub(crate) order: Arc<[f32]>,
}

/// A run awaited by an item: the timeline version it started at, the
/// step its result records (`None`: it fills in the shown step), the
/// engine running it, and where it started (`None`: the original).
#[derive(Clone)]
pub(crate) struct Job {
    /// This run, among every run of every image: what its result names to
    /// land on it and no other.
    pub(crate) run: u64,
    pub(crate) version: u64,
    pub(crate) step: Option<StepKind>,
    pub(crate) engine: Engine,
    pub(crate) start: Option<RunStart>,
    /// The seed a generated fill runs with.
    pub(crate) seed: Option<u64>,
    /// The region an erase fills.
    pub(crate) region: Option<super::timeline::StoredSelection>,
    /// Who cuts (`None`: another engine).
    pub(crate) cutter: Option<Cutter>,
    /// A cut of another model's cut-out: that cut-out, whose transparency
    /// the run starts from, with `start`'s colours under it.
    pub(crate) keep: Option<super::timeline::StoredImage>,
    /// Under way; a cut waits for a worker slot until it starts.
    pub(crate) running: bool,
    /// When the step it started on was made: an undo or redo to a step
    /// made since, on the same pixels, leaves the run landable.
    pub(crate) since: std::time::Instant,
    /// The strokes landed while an erase fills: those whose step is still
    /// in effect are the next area, which the fill's landing keeps.
    pub(crate) painted: Vec<PaintedStroke>,
}

/// A stroke landed while an erase fills, cropped to where it painted, and
/// the step it made: the step undone, the stroke is no next area.
#[derive(Clone)]
pub(crate) struct PaintedStroke {
    step: u64,
    patch: StrokePatch,
}

/// A stroke's painted cells within their bounds.
#[derive(Clone)]
pub(crate) struct StrokePatch {
    rect: prunr_core::selection::CellRect,
    cells: prunr_core::selection::MaskArtifact,
    /// The stroke's plane, cells across and down.
    plane: (u32, u32),
}

impl StrokePatch {
    /// `None` when the stroke painted nothing.
    pub(crate) fn of(stroke: &prunr_core::selection::MaskArtifact) -> Option<Self> {
        let rect = stroke.bounds()?;
        Some(Self { rect, cells: stroke.crop(rect), plane: (stroke.width, stroke.height) })
    }
}

/// What "Try another" reruns: the shown fill's start (`None`: the
/// original) and the region painted for it.
pub(crate) struct Reroll {
    pub(crate) start: Option<RunStart>,
    pub(crate) region: super::timeline::StoredSelection,
}

/// What runs a job, and so what cancelling it stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Engine {
    /// Background removal, and the filters-only pass.
    Segment,
    Erase,
    Upscale,
    Colorize,
}

impl Engine {
    /// The run's name at the start of a toast ("Upscale cancelled").
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Segment => "Processing",
            Self::Erase => "Erase",
            Self::Upscale => "Upscale",
            Self::Colorize => "Colorize",
        }
    }
}

/// Who cut a step out: a background-removal model, or No model's pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cutter {
    Model(prunr_core::ModelKind),
    NoModel,
}

/// A lens-blur focus and what it was measured from.
struct LensFocus {
    depth: std::sync::Weak<prunr_core::depth::DepthMap>,
    result: Option<std::sync::Weak<image::RgbaImage>>,
    value: f32,
}

impl LensFocus {
    fn is_for(&self, depth: &Arc<prunr_core::depth::DepthMap>, result: Option<&Arc<image::RgbaImage>>) -> bool {
        same_result(self.result.as_ref(), result) && std::sync::Weak::ptr_eq(&self.depth, &Arc::downgrade(depth))
    }
}

/// Whether `seen` is `now`, by allocation: a `Weak` pins it, so a new
/// result can never pass for the old one by address.
fn same_result(seen: Option<&std::sync::Weak<image::RgbaImage>>, now: Option<&Arc<image::RgbaImage>>) -> bool {
    match (seen, now) {
        (None, None) => true,
        (Some(w), Some(r)) => std::sync::Weak::ptr_eq(w, &Arc::downgrade(r)),
        _ => false,
    }
}

/// Which cut-out a backdrop made of it is for: the result it last saw,
/// and a count that moves each time the result is another one.
#[derive(Default)]
struct BackdropCut {
    result: Option<std::sync::Weak<image::RgbaImage>>,
    version: u64,
}

/// How much of the pipeline a Process run needs on an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunTier {
    /// Nothing: its result is what the settings ask for.
    UpToDate,
    /// Everything again.
    Full,
    /// The mask side again, from the cached tensor.
    Mask,
    /// Edge detection onto the cut-out from the cached tensor.
    AddEdge,
}

/// What Process needs on an image under a model that keeps its run (an
/// upscaler, the colorizer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeptRunTier {
    /// Nothing: the shown step is this model's run, as the settings ask.
    UpToDate,
    /// The kept run finished again: only Refine (upscale) or Strength
    /// (colorize) changed.
    Refinish,
    /// A new run.
    Full,
}

/// `selection` applied over `edits`: restored (made positive) or erased
/// (kept negative, as a selection is stored); where it lands it wins.
pub(crate) fn apply_selection(
    edits: Option<&prunr_core::selection::MaskArtifact>,
    selection: &prunr_core::selection::MaskArtifact,
    action: CutoutAction,
) -> prunr_core::selection::MaskArtifact {
    let signed = match action {
        CutoutAction::Restore => selection.as_restore(),
        CutoutAction::Erase | CutoutAction::Select => selection.clone(),
    };
    match edits.map(|e| e.add_mask(&signed)) {
        Some(Ok(merged)) => merged,
        Some(Err(err)) => {
            tracing::warn!(%err, "edits size differs from the selection; keeping the selection alone");
            signed
        }
        None => signed,
    }
}

/// What a correction plane is built from: the edits, and the selection
/// applied over them as `action` says.
#[derive(Clone)]
pub(crate) struct CorrectionInputs {
    edits: Option<Arc<prunr_core::selection::MaskArtifact>>,
    selection: Arc<prunr_core::selection::MaskArtifact>,
    action: CutoutAction,
}

impl CorrectionInputs {
    fn same(&self, other: &Self) -> bool {
        same_arc(self.edits.as_ref(), other.edits.as_ref()) && Arc::ptr_eq(&self.selection, &other.selection) && self.action == other.action
    }
}

/// A correction plane and what it was built from; see
/// `BatchItem::correction_cache`.
#[derive(Clone)]
pub(crate) struct BuiltCorrection {
    inputs: CorrectionInputs,
    plane: Arc<prunr_core::selection::MaskArtifact>,
}

/// What corrects a cut-out: the plane, or what it is built from. A build
/// is a full-resolution pass over the selection, so it runs where the
/// plane is used, off the UI thread.
#[derive(Clone)]
pub(crate) enum Correction {
    Ready(Arc<prunr_core::selection::MaskArtifact>),
    Build(CorrectionInputs),
}

impl Correction {
    /// The plane, and the build that made it here, for the item to keep.
    pub(crate) fn resolve(self) -> (Arc<prunr_core::selection::MaskArtifact>, Option<BuiltCorrection>) {
        match self {
            Self::Ready(plane) => (plane, None),
            Self::Build(inputs) => {
                let plane = Arc::new(apply_selection(inputs.edits.as_deref(), &inputs.selection, inputs.action));
                (Arc::clone(&plane), Some(BuiltCorrection { inputs, plane }))
            }
        }
    }
}

/// The photo faces are found in: a run's start image, the original, or an
/// upscale's output. Compared by identity, never unpacked to compare.
#[derive(Clone)]
pub(crate) enum FacePhoto {
    Start(super::timeline::StoredImage),
    Original(Arc<image::RgbaImage>),
    Upscaled(super::timeline::StoredImage),
}

impl FacePhoto {
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Start(a), Self::Start(b)) | (Self::Upscaled(a), Self::Upscaled(b)) => Arc::ptr_eq(a, b),
            (Self::Original(a), Self::Original(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        match self {
            Self::Start(stored) | Self::Upscaled(stored) => stored.load(),
            Self::Original(rgba) => Some(Arc::clone(rgba)),
        }
    }
}

/// Where an image's pixels are. Loading may unpack a parked step, copy
/// the photo out of its wrapper or decode its file, so it runs where the
/// pixels are used: off the UI thread.
#[derive(Clone)]
pub(crate) enum Pixels {
    Resident(Arc<image::RgbaImage>),
    /// A step's image in the timeline, compressed while off screen.
    Parked(super::timeline::StoredImage),
    /// The photo, not decoded.
    Source(ImageSource),
}

impl Pixels {
    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        self.read().ok()
    }

    /// Width and height without unpacking; `None` for a photo not decoded.
    pub(crate) fn decoded_dims(&self) -> Option<(u32, u32)> {
        match self {
            Self::Resident(rgba) => Some(rgba.dimensions()),
            Self::Parked(stored) => Some(stored.dims()),
            Self::Source(_) => None,
        }
    }

    /// The pixels, or why they could not be read.
    pub(crate) fn read(&self) -> Result<Arc<image::RgbaImage>, String> {
        match self {
            Self::Resident(rgba) => Ok(Arc::clone(rgba)),
            Self::Parked(stored) => stored.load().ok_or_else(|| "a stored image could not be read".to_string()),
            Self::Source(source) => source.decode_rgba(),
        }
    }
}

/// A painted region an erase fills: the one on screen, or a step's,
/// stored compressed until the erase's thread reads it.
#[derive(Clone)]
pub(crate) enum Region {
    Ready(Arc<prunr_core::selection::MaskArtifact>),
    Stored(super::timeline::StoredSelection),
}

impl Region {
    pub(crate) fn load(&self) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        match self {
            Self::Ready(mask) => Some(Arc::clone(mask)),
            Self::Stored(stored) => stored.load(),
        }
    }
}

/// What an export writes: the image as the canvas shows it, the result
/// or the photo with its restored faces.
#[derive(Clone)]
pub(crate) struct ExportImage {
    pub(crate) pixels: Pixels,
    /// The pixels are the photo, not a result.
    pub(crate) photo: bool,
    pub(crate) faces: Option<super::live_preview::FacesInput>,
}

impl ExportImage {
    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        let Some(faces) = &self.faces else { return self.pixels.load() };
        if let Some(faced) = &faces.faced {
            return Some(Arc::clone(faced));
        }
        let photo = self.pixels.load()?;
        let photo = Arc::try_unwrap(photo).unwrap_or_else(|shared| (*shared).clone());
        Some(Arc::new(faces.key.blend_into(photo)))
    }

    /// The pixels as they are, when nothing needs loading or blending.
    pub(crate) fn ready(&self) -> Option<&Arc<image::RgbaImage>> {
        match (&self.pixels, &self.faces) {
            (Pixels::Resident(rgba), None) => Some(rgba),
            _ => None,
        }
    }

    /// The photo as it was opened: an export can copy its file unchanged.
    pub(crate) fn is_untouched_photo(&self) -> bool {
        self.photo && self.faces.is_none()
    }
}

/// Which photo Restore faces works on: the photo a cut-out or No model's
/// preview starts from, or the shown upscale's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaceStage {
    Photo,
    Upscale,
}

impl FaceStage {
    pub(crate) fn of(role: super::knob_catalog::ModelRole) -> Self {
        if role == super::knob_catalog::ModelRole::Upscale { Self::Upscale } else { Self::Photo }
    }

    pub(crate) fn engine(self) -> Engine {
        match self {
            Self::Photo => Engine::Segment,
            Self::Upscale => Engine::Upscale,
        }
    }
}

/// Faces found in one photo with one model.
#[derive(Clone)]
pub(crate) struct FoundFaces {
    pub(crate) photo: FacePhoto,
    /// When the search landed, for the outlines' arrival.
    pub(crate) found_at: std::time::Instant,
    pub(crate) model: prunr_models::ModelId,
    pub(crate) faces: Arc<Vec<prunr_core::face::RestoredFace>>,
    /// The search failed: none are blended, and it is not repeated until
    /// a search setting changes.
    pub(crate) failed: bool,
    /// It failed because it was stopped (Cancel or Esc).
    pub(crate) stopped: bool,
}

/// A face search under way: of which photo, with which model, and the
/// faces it has found while it redraws them, with when it found them.
pub(crate) struct PendingFaces {
    pub(crate) photo: FacePhoto,
    pub(crate) model: prunr_models::ModelId,
    pub(crate) found: Option<(Arc<[prunr_core::face::FacePlace]>, std::time::Instant)>,
}

/// The faces a search has redrawn, each laid over the image as it is
/// done (straight-alpha patches, clear round each face), kept until the
/// image with them blended in is on screen.
pub(crate) struct FacePreview {
    /// The search it shows: of which photo, with which model.
    photo: FacePhoto,
    model: prunr_models::ModelId,
    /// Made on the first face.
    texture: Option<egui::TextureHandle>,
    /// A result with these faces blended in has been applied: the preview
    /// goes when its texture lands.
    blended: bool,
}

/// Paint `tile` into its square of the preview in `slot`, made on the
/// first tile (or anew when the size changes). Only the tile is uploaded
/// (`set_partial`).
fn paint_tile(slot: &mut Option<egui::TextureHandle>, ctx: &egui::Context, name: &str, tile: &prunr_core::TilePreview, image: Arc<egui::ColorImage>) {
    let full = [tile.full_width as usize, tile.full_height as usize];
    let at = [tile.x as usize, tile.y as usize];
    let image = Arc::unwrap_or_clone(image);
    match slot {
        Some(t) if t.size() == full => t.set_partial(at, image, egui::TextureOptions::LINEAR),
        // A tile that is the whole preview (the photo laid under an erase)
        // is the texture: one upload, not a blank one and the tile.
        slot if at == [0, 0] && image.size == full => {
            *slot = Some(ctx.load_texture(name, image, egui::TextureOptions::LINEAR));
        }
        slot => slot
            .insert(ctx.load_texture(
                name,
                egui::ColorImage::new(full, vec![egui::Color32::TRANSPARENT; full[0] * full[1]]),
                egui::TextureOptions::LINEAR,
            ))
            .set_partial(at, image, egui::TextureOptions::LINEAR),
    }
}

/// The faces outlined on an image: those found and redrawn, or those a
/// search under way has found so far.
pub(crate) struct OutlinedFaces<'a> {
    found: &'a [prunr_core::face::FacePlace],
    redrawn: &'a [prunr_core::face::RestoredFace],
    /// When they were found, for their arrival.
    pub(crate) found_at: std::time::Instant,
}

impl<'a> OutlinedFaces<'a> {
    pub(crate) fn places(&self) -> impl Iterator<Item = &'a prunr_core::face::FacePlace> + Clone {
        self.found.iter().chain(self.redrawn.iter().map(prunr_core::face::RestoredFace::place))
    }
}

/// Where `BatchItem::warm_steps` holds the step an undo goes back to and
/// the one a redo goes to.
pub(crate) const BACK: usize = 0;
pub(crate) const AHEAD: usize = 1;

/// A step next to the shown one, and its image and planes once
/// unpacked: holding them keeps the step unpacked for undo to read.
pub(crate) struct WarmStep {
    step: WarmKey,
    pub(crate) image: Option<Arc<image::RgbaImage>>,
    pub(crate) planes: [Option<Arc<prunr_core::selection::MaskArtifact>>; 2],
}

/// What a step stores that undo would unpack: its result, selection and
/// edits.
#[derive(Clone)]
struct WarmKey {
    result: Option<super::timeline::StoredImage>,
    planes: [Option<super::timeline::StoredSelection>; 2],
}

impl WarmKey {
    fn of(step: &DocState) -> Option<Self> {
        let key = Self { result: step.result.clone(), planes: [step.selection.clone(), step.edits.clone()] };
        (key.result.is_some() || key.planes.iter().any(Option::is_some)).then_some(key)
    }

    fn same(&self, other: &Self) -> bool {
        same_arc(self.result.as_ref(), other.result.as_ref()) && self.planes.iter().zip(&other.planes).all(|(a, b)| same_arc(a.as_ref(), b.as_ref()))
    }
}

/// An image as a save wrote it: which one, at which timeline revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SaveMark {
    pub(crate) id: u64,
    revision: u64,
}

pub(crate) struct BatchItem {
    pub(crate) id: u64,
    pub(crate) filename: String,
    pub(crate) source: ImageSource,
    /// The file's size, read where it was imported (for admission's
    /// estimate; 0 when not known).
    pub(crate) source_len: usize,
    pub(crate) dimensions: (u32, u32),
    /// Pre-decoded source RGBA (decoded on background thread for instant switching)
    pub(crate) source_rgba: Option<Arc<image::RgbaImage>>,
    pub(crate) source_texture: Option<super::image_texture::ImageTexture>,
    pub(crate) thumb: Thumbnail,
    pub(crate) result_rgba: Option<Arc<image::RgbaImage>>,
    /// The run's start, read back, held while live previews start from
    /// it so it is not packed away again between drag ticks; cleared when
    /// they start from the photo.
    pub(crate) start_photo: Option<Arc<image::RgbaImage>>,
    pub(crate) result_texture: Option<super::image_texture::ImageTexture>,
    /// A drag's newest quick frame, shown in place of the result until
    /// the full-size render lands; never the result itself.
    pub(crate) quick: super::live_preview::QuickFrames,
    /// The unprocessed photo with restored faces, shown in its place; not
    /// a result. One texture while Restore faces is on before Process.
    pub(crate) photo_look: Option<super::image_texture::ImageTexture>,
    /// True while a background thread is building the source ColorImage.
    pub(crate) source_tex_pending: bool,
    /// True while a background thread is building the result ColorImage.
    pub(crate) result_tex_pending: bool,
    result_tex_request: u64,
    /// A shown step whose result could not be read for its texture: not
    /// asked for again while it shows.
    unreadable_result: Option<super::timeline::StoredImage>,
    /// The transition to the next image after a discrete change (a run,
    /// an edit, an undo). A slider re-cut makes none and swaps at once.
    pub(crate) reveal: Option<Reveal>,
    pub(crate) subject: Option<SubjectOutline>,
    /// The run under way's finished tiles, small, each in its square of a
    /// preview of the whole image: an upscale's, or an erase's over the
    /// photo it started from.
    pub(crate) tile_preview: Option<egui::TextureHandle>,
    /// The images of the steps an undo and a redo go to (`BACK`, `AHEAD`),
    /// unpacked off the UI thread and held so they stay unpacked: either
    /// shows without decoding first. Costs up to two images of RAM.
    pub(crate) warm_steps: Arc<std::sync::Mutex<[Option<WarmStep>; 2]>>,
    /// The image is the one on screen: only it keeps steps warm, so a
    /// batch reprocessing in the background holds no extra images.
    pub(crate) on_screen: bool,
    /// The faces found for Restore faces, and the photo and model they
    /// were found with; a few MB per face.
    pub(crate) found_faces: Option<FoundFaces>,
    pub(crate) faces_pending: Option<PendingFaces>,
    /// The faces a search has redrawn so far, laid over the image.
    pub(crate) face_preview: Option<FacePreview>,
    /// The photo with the found faces blended in, as a preview built it,
    /// so other knobs' previews reuse it. One image of RAM while Restore
    /// faces is on.
    pub(crate) faced_photo: Option<(Arc<image::RgbaImage>, super::live_preview::FacedKey)>,
    /// True while a background thread is decoding source bytes to RGBA.
    pub(crate) decode_pending: bool,
    /// The source could not be decoded; no decode is asked for again.
    pub(crate) decode_failed: bool,
    /// Every state this image has been in, and which one is shown. The
    /// fields above and below (result, recipe, selection, settings) are
    /// the working copy of the shown state.
    pub(crate) timeline: Timeline,
    /// The run (Process, erase, upscale) whose result this item awaits.
    pub(crate) job: Option<Job>,
    /// The timeline revision the last save wrote.
    saved: Option<u64>,
    /// Why the last run or decode failed; cleared by the next run.
    failure: Option<String>,
    pub(crate) ticked: bool,
    /// Per-image processing settings. Edited via the adjustments toolbar.
    pub(crate) settings: super::item_settings::ItemSettings,
    /// The recipe that produced the current result_rgba. None if never processed.
    pub(crate) applied_recipe: Option<prunr_core::ProcessingRecipe>,
    /// Compressed cached tensor from Tier 1 inference (for Tier 2 mask reruns).
    pub(crate) cached_tensor: Option<super::worker::CompressedTensor>,
    /// The last upscale run's model output and input, so Refine re-finishes
    /// it without a model run. Survives `reset_result_caches()`; cleared on a
    /// model swap. RAM: 4K × 4 ≈ 500 MB, double while AI blend or Color match
    /// keep the enlarged reference; outside the tensor budget.
    pub(crate) upscale: Option<UpscaleCache>,
    /// The last colorize run, so Strength re-renders it without a model
    /// run. Not cleared by `reset_result_caches()`; cleared on a model
    /// switch. RAM: the 2 MB colour plane and the input image (shared
    /// with the timeline when it is one).
    pub(crate) colorize: Option<ColorizeCache>,
    /// Every TEED scale (Tier 2 edge reruns read whichever scale the
    /// user has picked without re-inferring).
    pub(crate) cached_edge_tensors: Option<super::worker::CompressedEdgeTensors>,
    /// Decompressed hot copy of the active scale. Lets a slider drag reuse
    /// the same Arc instead of paying zstd per dispatch.
    pub(crate) volatile_edge_tensor: Option<(prunr_core::EdgeScale, Arc<Vec<f32>>)>,
    /// Decompressed segmentation tensor kept across the ticks of one
    /// slider drag; cleared with `cached_tensor` and when the item
    /// leaves the selection.
    pub(crate) volatile_seg_tensor: Option<Arc<super::live_preview::SegTensor>>,
    /// Live-preview cache of the active scale's edge planes (undilated and
    /// dilated, each with its key), and of the Bold scale for dual-scale
    /// styles.
    pub(crate) edge_planes: super::live_preview::EdgePlaneCache,
    /// SubjectOutline live-preview base (`postprocess_from_flat` output) that
    /// edges draw onto, keyed by mask recipe and model: an Edge-only tick
    /// reuses it and skips ~50-100 ms of Lanczos + guided filter.
    pub(crate) cached_masked_base: Option<(Arc<image::RgbaImage>, prunr_core::MaskRecipe, prunr_core::ModelKind)>,
    /// The preset last applied (dropdown or Reset All); the preset button
    /// compares `settings` against it, and it stays across later tweaks.
    pub(crate) applied_preset: String,
    /// Never mutate outside `set_bg_image_hashed` / `clear_bg_image` /
    /// `adopt_settings`: the writers that keep `settings.bg_image_hash` in step,
    /// which the recipe-diff dispatch reads to fire CompositeOnly.
    pub(crate) bg_image: Option<Arc<BgImage>>,
    /// The picture behind the result (a background image or an effect
    /// made from the photo) and its `Backdrop::picture_key`.
    pub(crate) backdrop_texture: Option<(u64, super::image_texture::ImageTexture)>,
    /// The key of the picture a background thread is building, so the
    /// next frame doesn't start the same one again.
    pub(crate) backdrop_tex_pending: Option<u64>,
    /// Where a lens-blur backdrop focuses, measured once per depth map
    /// and result. Weak handles pin the allocations, so a new result can
    /// never pass for the old one by address.
    lens_focus: Option<LensFocus>,
    /// The cut-out Blurred photo and Lens blur fill the subject from.
    backdrop_cut: BackdropCut,

    /// Shared selection mask (Paint and Magic Brush), a signed i8 plane at
    /// source resolution. Survives Process; an erase that lands clears it, as
    /// do Clear and a source change.
    pub(crate) selection_mask: Option<Arc<prunr_core::selection::MaskArtifact>>,

    /// Content hash of `selection_mask`, set in lockstep with the mask by
    /// `commit_selection_mask`. Half of the selection texture's key.
    pub(crate) selection_hash: Option<u64>,
    /// On a cut-out, the restores and erases already made (see `DocState::edits`).
    pub(crate) edits: Option<Arc<prunr_core::selection::MaskArtifact>>,
    /// The last correction built from the selection and the edits, with
    /// the inputs it was built from (held, so a match is never a reused address).
    correction_cache: std::sync::Mutex<Option<BuiltCorrection>>,

    /// Key of the selection texture build in flight, if any; cleared when
    /// that build lands or is dropped as stale.
    pub(crate) selection_tex_pending: Option<super::background_io::SelectionTextureKey>,

    /// Selection overlay texture with the key it was built from. Rebuilt
    /// whenever the mask or the baked style differs from the key.
    pub(crate) selection_texture: Option<SelectionTexture>,

    /// Cached SAM 2 image embedding, filled whenever Magic Brush is active
    /// and this item is selected. Invalidated on source change.
    pub(crate) magic_brush_embedding: Option<std::sync::Arc<prunr_core::sam::SamEmbedding>>,
    pub(crate) adjusting: Option<StrokeAdjustment>,
    /// A background pre-encode was refused (RAM gate or encoder error);
    /// not retried every frame. Cleared with the embedding.
    pub(crate) magic_preencode_refused: bool,
    /// How near each part of the original photo is (~1 MB), measured when
    /// a depth knob first needs it and kept until the source changes.
    pub(crate) depth: Option<Arc<prunr_core::depth::DepthMap>>,
    /// Where the depth map's run stands; see `DepthRun`.
    pub(crate) depth_run: DepthRun,
    /// The previewed photo with the Depth chip's look on it, and what it
    /// was made from, so a line or effect tweak does not blur again. One
    /// image of RAM while the look is on; dropped off screen.
    pub(crate) depth_photo: Option<(Arc<image::RgbaImage>, DepthPhotoKey)>,
    /// The image shrunk for a drag's quick frames (the photo, or a kept
    /// upscale or colorize run's), so each frame does not shrink it
    /// again. About 11 MB; dropped off screen.
    pub(crate) quick_photo: Option<super::live_preview::QuickPhoto>,
}

/// The depth map's run for the photo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DepthRun {
    pub(crate) pending: bool,
    /// The run failed; not retried.
    pub(crate) failed: bool,
}

/// What a depth-looked photo was made from.
#[derive(Clone)]
pub(crate) struct DepthPhotoKey {
    pub(crate) base: Arc<image::RgbaImage>,
    pub(crate) depth: Arc<prunr_core::depth::DepthMap>,
    pub(crate) look: prunr_core::depth::DepthLook,
}

impl DepthPhotoKey {
    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base) && Arc::ptr_eq(&self.depth, &other.depth) && self.look == other.look
    }
}

/// The image's strokes being drawn again as a brush's sliders move: the
/// latest one that brush drew (Last) or every one it drew since the
/// picture changed (All).
pub(crate) struct StrokeAdjustment {
    pub(crate) reach: crate::gui::brush_state::StrokeReach,
    pub(crate) tool: crate::gui::brush_state::Tool,
    /// The slider values where it began.
    pub(crate) from: crate::gui::brush_state::StrokeValues,
    /// The strokes as they stood then.
    pub(crate) strokes: super::timeline::StrokeHistory,
    /// How the shown step draws them.
    pub(crate) shown: Arc<Vec<super::selection_queue::SelectionMade>>,
    /// The step it fills in once landed (the stroke's own, from the start,
    /// when it is the shown step's own stroke); before that, the step it
    /// began on.
    pub(crate) step: u64,
    pub(crate) landed: bool,
}

impl StrokeAdjustment {
    /// The strokes as the sliders at `now` draw them: Last sets each
    /// moved slider's value on the brush's latest stroke, All moves each of
    /// its strokes by as much.
    pub(crate) fn at(&self, now: crate::gui::brush_state::StrokeValues) -> super::timeline::StrokeHistory {
        use crate::gui::brush_state::StrokeReach;
        let (from, tool) = (self.from, self.tool);
        match self.reach {
            StrokeReach::Last => self.strokes.latest_changed(tool, |drawn| drawn.at(drawn.values().moved(from, now))),
            StrokeReach::All => self.strokes.changed(tool, |drawn| drawn.at(drawn.values().adjusted(from, now))),
            // Next starts no adjustment: the strokes stay as drawn.
            StrokeReach::Next => self.strokes.clone(),
        }
    }
}

/// A selection overlay texture and what it was built from. `shown` is
/// the plane on the GPU, so the next change can upload only its patch;
/// it is usually the item's own mask (one Arc), and at most one extra
/// plane per item while a newer mask awaits its patch.
pub(crate) struct SelectionTexture {
    pub(crate) key: super::background_io::SelectionTextureKey,
    pub(crate) texture: super::image_texture::ImageTexture,
    pub(crate) shown: Arc<prunr_core::selection::MaskArtifact>,
}

impl BatchItem {
    /// Clear every edge cache tier together: the mask is built from the
    /// tensor, so any tensor change invalidates all of them.
    pub(crate) fn invalidate_edge_cache(&mut self) {
        self.cached_edge_tensors = None;
        self.volatile_edge_tensor = None;
        self.edge_planes = Default::default();
    }

    /// Drop the seg tensor and every edge cache tier together.
    pub(crate) fn drop_tensor_caches(&mut self) {
        self.set_cached_tensor(None);
        self.invalidate_edge_cache();
    }

    /// The kept lines lack a scale the settings draw.
    pub(crate) fn lacks_drawn_scales(&self) -> bool {
        self.settings.line_mode != prunr_core::LineMode::Off
            && self.cached_edge_tensors.as_ref().is_some_and(|c| !c.has_all(self.settings.edge_settings().drawn_scales()))
    }

    /// The kept lines lack a scale the settings draw and the worker is
    /// not still making it: only a new run of the line detector does.
    pub(crate) fn lines_need_a_run(&self) -> bool {
        self.lacks_drawn_scales() && !self.cached_edge_tensors.as_ref().is_some_and(super::worker::CompressedEdgeTensors::awaits_spare)
    }

    /// The scales the run that made the kept lines left for later join
    /// them, when the kept lines are still that run's (`token`); `None`:
    /// they will not come. Returns whether they were that run's.
    pub(crate) fn land_spare_edge_scales(&mut self, token: u64, spare: Option<super::worker::CompressedEdgeTensors>) -> bool {
        self.cached_edge_tensors.as_mut().is_some_and(|kept| kept.land_spare(token, spare))
    }

    /// Replaces the compressed tensor and the decoded copy tied to it.
    pub(crate) fn set_cached_tensor(&mut self, tensor: Option<super::worker::CompressedTensor>) {
        self.cached_tensor = tensor;
        self.volatile_seg_tensor = None;
    }

    /// Drop what slider drags keep unpacked while the image is shown: the
    /// decompressed tensors and the live-preview caches (edge planes,
    /// outline base, the run start read back). The compressed caches stay,
    /// so the next drag rebuilds them once.
    pub(crate) fn drop_hot_tensors(&mut self) {
        self.volatile_seg_tensor = None;
        self.volatile_edge_tensor = None;
        self.adjusting = None;
        super::reclaim::later((std::mem::take(&mut self.edge_planes), self.cached_masked_base.take(), self.start_photo.take()));
        self.drop_correction_cache();
    }

    fn drop_correction_cache(&mut self) {
        *self.correction_cache.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// What corrects a cut-out: the edits already made, with the
    /// selection on top as `action` says (restored, erased, or nothing
    /// while only selecting), at full resolution. Built once per change
    /// of its inputs.
    #[cfg(test)]
    pub(crate) fn correction_plane(&self, action: CutoutAction) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        self.correction(action).map(|c| {
            let (plane, built) = c.resolve();
            if let Some(built) = built {
                self.keep_built_correction(built);
            }
            plane
        })
    }

    /// What corrects a cut-out, as a handle: built already, or to build
    /// where it is used. Never builds here.
    pub(crate) fn correction(&self, action: CutoutAction) -> Option<Correction> {
        let selection = self.selection_mask.clone()
            .filter(|m| !m.is_blank() && action != CutoutAction::Select);
        match (&self.edits, selection) {
            (None, None) => None,
            (Some(edits), None) => Some(Correction::Ready(Arc::clone(edits))),
            // A selection is stored as an erase: as it is, it erases.
            (None, Some(selection)) if action == CutoutAction::Erase => Some(Correction::Ready(selection)),
            (edits, Some(selection)) => {
                let inputs = CorrectionInputs { edits: edits.clone(), selection, action };
                let cache = self.correction_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                Some(match cache.as_ref().filter(|c| c.inputs.same(&inputs)) {
                    Some(hit) => Correction::Ready(Arc::clone(&hit.plane)),
                    None => Correction::Build(inputs),
                })
            }
        }
    }

    /// Keep a correction plane built elsewhere, so the next use of the
    /// same inputs reads it instead of building again.
    pub(crate) fn keep_built_correction(&self, built: BuiltCorrection) {
        *self.correction_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(built);
    }

    /// Fold the selection into the edits as `action` says and clear it,
    /// here and now, as the selection queue does off the UI thread: the
    /// same edit and the same landing. False when there was no selection.
    #[cfg(test)]
    pub(crate) fn bake_selection(&mut self, action: CutoutAction) -> bool {
        use super::selection_queue::{apply, SelectionOp};
        if self.selection_mask.is_none() {
            return false;
        }
        let op = SelectionOp::Bake { action };
        let made = apply(&op, &self.selection_state());
        let (kind, record) = op.step();
        self.land_selection_state(made, None, Some(kind), record)
    }

    /// The overlay texture, when it was built from the current selection
    /// in `style`.
    pub(crate) fn current_selection_texture(&self, style: super::background_io::SelectionStyle) -> Option<&SelectionTexture> {
        let hash = self.selection_hash?;
        self.selection_texture.as_ref().filter(|t| t.key == (hash, style))
    }

    /// Free the upscale's enlarged input (output-sized; up to ~1 GB for a
    /// 16K result) while the image is not on screen. AI blend and Color
    /// match rebuild it when next used (0.1-0.3 s); the run itself stays.
    pub(crate) fn drop_upscale_reference(&mut self) {
        if let Some(reference) = self.upscale.as_mut().and_then(|run| run.reference.take()) {
            super::reclaim::later(reference);
        }
    }

    /// Clear the kept upscale run, on a model switch and on item error. A knob
    /// that re-runs the model need not: its landing overwrites it, but after a
    /// switch it is the wrong model's and may never be replaced.
    pub(crate) fn invalidate_upscale_cache(&mut self) {
        super::reclaim::later(self.upscale.take());
    }

    /// Clear the kept colorize run, on a model switch and on item error.
    pub(crate) fn invalidate_colorize_cache(&mut self) {
        super::reclaim::later(self.colorize.take());
    }

    /// A depth map is wanted and none is kept, under way or failed.
    pub(crate) fn needs_depth_run(&self) -> bool {
        self.depth.is_none() && !self.depth_run.pending && !self.depth_run.failed
    }

    /// Keep a depth map measured from the photo. True when it was kept.
    pub(crate) fn land_depth(&mut self, result: Result<Arc<prunr_core::depth::DepthMap>, String>) -> bool {
        self.depth_run.pending = false;
        match result {
            Ok(map) => {
                self.depth = Some(map);
                true
            }
            Err(_) => {
                self.depth_run.failed = true;
                false
            }
        }
    }

    /// The strokes' adjustment under way, while nothing else has happened
    /// since: its step is the one shown and the newest.
    pub(crate) fn adjustment(&mut self) -> Option<&mut StrokeAdjustment> {
        if self.adjusting.as_ref().is_some_and(|a| !self.timeline.is_newest_shown(a.step)) {
            self.adjusting = None;
        }
        self.adjusting.as_mut()
    }

    /// Clear every selection-related cache: mask, hash, texture, pending build.
    pub(crate) fn invalidate_selection(&mut self) {
        self.selection_mask = None;
        self.selection_hash = None;
        self.selection_texture = None;
        self.selection_tex_pending = None;
    }

    /// Where the image is, derived: a cut under way, a failure with
    /// nothing to show, a result, or the photo waiting. Never stored, so
    /// undo, cancel and parking cannot leave it stale.
    pub(crate) fn status(&self) -> BatchStatus {
        if self.is_cutting() {
            return BatchStatus::Processing;
        }
        if self.has_result() {
            return BatchStatus::Done;
        }
        match &self.failure {
            Some(why) => BatchStatus::Error(why.clone()),
            None => BatchStatus::Pending,
        }
    }

    pub(crate) fn unreadable(&self) -> Option<&str> {
        self.failure.as_deref().filter(|_| self.decode_failed)
    }

    /// Why the image failed, while the failure is what it shows.
    pub(crate) fn error(&self) -> Option<&str> {
        self.failure.as_deref().filter(|_| !self.is_cutting() && !self.has_result())
    }

    /// A cut is under way: the worker has taken it (`status()` Processing,
    /// without building the status).
    pub(crate) fn is_cutting(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.running && j.engine == Engine::Segment)
    }

    /// No run of any kind is queued or under way on this image: what
    /// `Need::Idle` reads, for gates that need no role.
    pub(crate) fn is_idle(&self) -> bool {
        self.job.is_none()
    }

    /// The shown step draws lines and their tensors are kept: what a lines
    /// layer exports. After an undo to the photo, or with Sketch off,
    /// kept tensors draw nothing on screen.
    pub(crate) fn shows_lines(&self) -> bool {
        self.has_result() && self.settings.line_mode != prunr_core::LineMode::Off && self.cached_edge_tensors.is_some()
    }

    /// A cut is queued for a worker slot or under way: what Cancel stops.
    pub(crate) fn awaits_cut(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.engine == Engine::Segment)
    }

    pub(crate) fn verdict(&self, need: super::capability::Need, role: super::knob_catalog::ModelRole) -> super::capability::Verdict {
        super::capability::check(&super::capability::ItemFacts::of(self), need, role)
    }

    /// The awaited cut has a worker slot and is under way.
    pub(crate) fn mark_running(&mut self) {
        if let Some(job) = self.job.as_mut() {
            job.running = true;
        }
    }

    /// The run or read failed with `why`: nothing is awaited any more.
    pub(crate) fn fail(&mut self, why: String) {
        self.job = None;
        self.failure = Some(prunr_core::plain::reason(why));
    }

    /// True while a model works on this image: a cut-out the worker has
    /// taken, or an upscale, colorize or erase, which start their job only
    /// when they run (a queued cut-out holds its job while it waits).
    pub(crate) fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.running)
    }

    /// True while a run, a decode, a texture or a thumbnail is still to land.
    pub(crate) fn is_busy(&self) -> bool {
        self.job.is_some() || self.has_pending_work()
    }

    /// True while a background thread owes this item a decode, a texture
    /// or a thumbnail.
    pub(crate) fn has_pending_work(&self) -> bool {
        self.result_tex_pending || self.source_tex_pending || self.decode_pending
            || self.thumb.is_pending() || self.backdrop_tex_pending.is_some()
            || self.selection_tex_pending.is_some()
    }

    /// Make `mask` the selection as one undo step of `kind`. Every author
    /// (Paint strokes, Magic Brush candidates, Invert) goes through here.
    /// Returns `false` and changes nothing when `mask` is byte-identical
    /// to the current selection: a no-op stroke must not be a step.
    #[cfg(test)]
    pub(crate) fn commit_selection_mask(&mut self, mask: Arc<prunr_core::selection::MaskArtifact>, kind: StepKind) -> bool {
        let hash = mask.content_hash();
        let made = super::selection_queue::SelState { selection: Some(mask), edits: self.edits.clone() };
        self.land_selection_state(made, Some(hash), Some(kind), None)
    }

    /// The selection and the cut-out's edits, as a selection edit reads
    /// and makes them.
    pub(crate) fn selection_state(&self) -> super::selection_queue::SelState {
        super::selection_queue::SelState { selection: self.selection_mask.clone(), edits: self.edits.clone() }
    }

    /// A selection edit lands: the selection (with its content `hash`)
    /// and edits it made, as one undo step of `kind`, or filling in the
    /// shown step (`None`, a stroke adjustment while the slider moves).
    /// The step records `record`. False when it made what is already
    /// shown; a new step is then not made.
    pub(crate) fn land_selection_state(
        &mut self,
        planes: super::selection_queue::SelState,
        hash: Option<u64>,
        kind: Option<StepKind>,
        record: Option<super::selection_queue::Made>,
    ) -> bool {
        let changed = self.take_selection_state(planes, hash);
        match kind {
            Some(kind) if changed => self.commit(kind),
            Some(_) => return false,
            None if changed => self.amend(),
            None => {}
        }
        self.timeline.current_mut().made = record;
        changed
    }

    /// Take the selection and edits an edit made; false, and nothing
    /// changes, when it made what is already shown.
    fn take_selection_state(&mut self, made: super::selection_queue::SelState, hash: Option<u64>) -> bool {
        let same_edits = same_arc(made.edits.as_ref(), self.edits.as_ref());
        let same_selection = match made.selection {
            Some(_) => hash == self.selection_hash,
            None => self.selection_mask.is_none(),
        };
        if same_selection && same_edits {
            // Nothing new to show: the image takes the planes all the same,
            // so an edit queued on this one finds the base it was built on.
            self.selection_mask = made.selection;
            self.edits = made.edits;
            return false;
        }
        self.selection_mask = made.selection;
        self.selection_hash = hash;
        self.edits = made.edits;
        if self.selection_mask.is_none() {
            self.invalidate_selection();
        }
        self.drop_correction_cache();
        true
    }

    /// The working copy as a timeline state, sharing what the shown state
    /// already stores.
    fn snapshot(&self, kind: StepKind) -> DocState {
        let shown = self.timeline.current();
        // A parked result is the shown step's stored one: the new step
        // shares it, still packed, instead of unpacking it to store again.
        let result = if self.result_parked() { shown.result.clone() } else { store(self.result_rgba.as_ref(), shown.result.as_ref()) };
        DocState {
            kind,
            result,
            recipe: self.applied_recipe.clone(),
            selection: store(self.selection_mask.as_ref(), shown.selection.as_ref()),
            selection_hash: self.selection_hash,
            edits: store(self.edits.as_ref(), shown.edits.as_ref()),
            settings: self.settings,
            applied_preset: self.applied_preset.clone(),
            // A hand edit makes new pixels no run can redo.
            origin: if kind == StepKind::Edit { None } else { shown.origin.clone() },
            faced: shown.faced,
            at: std::time::Instant::now(),
            id: super::timeline::next_step_id(),
            made: None,
        }
    }

    /// Record the working copy as a new step.
    pub(crate) fn commit(&mut self, kind: StepKind) {
        let state = self.snapshot(kind);
        let keeps_run = self.keeps_long_run(&state);
        self.timeline.push(state);
        if keeps_run {
            if let Some(job) = &mut self.job {
                job.version = self.timeline.version();
            }
        }
        self.warm_neighbour_steps();
    }

    /// Whether `next` leaves an erase, upscale or colorize under way
    /// landable: it keeps the pixels the run started from (a knob for the
    /// next run, a preset), so the run's result still lands, on top of
    /// it. A cut keeps the strict rule: its knob changes re-cut.
    fn keeps_long_run(&self, next: &DocState) -> bool {
        let same_pixels = match (&self.timeline.current().result, &next.result) {
            (None, None) => true,
            (Some(shown), Some(kept)) => Arc::ptr_eq(shown, kept),
            _ => false,
        };
        same_pixels && self.job_is_current() && self.job.as_ref().is_some_and(|j| j.engine != Engine::Segment && j.step.is_some())
    }

    /// Whether moving to `target` (an undo or redo) leaves the run under way
    /// landable: a step made since it started, on the pixels it started
    /// from (strokes painted while an erase fills).
    fn move_keeps_run(&self, target: &DocState) -> bool {
        self.keeps_long_run(target) && self.job.as_ref().is_some_and(|j| target.at >= j.since)
    }

    pub(crate) fn undo_keeps_run(&self) -> bool {
        self.timeline.previous().is_some_and(|to| self.move_keeps_run(to))
    }

    pub(crate) fn redo_keeps_run(&self) -> bool {
        self.timeline.next().is_some_and(|to| self.move_keeps_run(to))
    }

    /// Whether a Paint Brush stroke may land now. A stroke is a step, so it
    /// waits for a run on the image, but an erase: a selection painted
    /// while it fills keeps its pixels and is the next area. A cut-out
    /// correction changes pixels, so it waits for any run.
    pub(crate) fn takes_strokes(&self, corrects_cutout: bool) -> bool {
        self.is_idle() || (!corrects_cutout && self.job.as_ref().is_some_and(|j| j.engine == Engine::Erase))
    }

    /// A stroke landed: while an erase fills, where it was painted is the
    /// next area, which the fill's landing keeps, kept with the step it
    /// made so an undo of that step takes it back out.
    pub(crate) fn note_painted_during_erase(&mut self, patch: StrokePatch) {
        let Some(job) = self.job.as_mut().filter(|j| j.engine == Engine::Erase) else { return };
        let step = self.timeline.current().id;
        job.painted.push(PaintedStroke { step, patch });
    }

    /// The fill landed: the marks it was made from go, those painted while
    /// it ran (and not undone since) stay as the next area. Undoing the
    /// fill brings both back.
    pub(crate) fn clear_filled_marks(&mut self) {
        let next = self.selection_mask.as_ref().zip(self.job.as_ref()).and_then(|(now, job)| {
            let mut strokes = job.painted.iter()
                .filter(|s| s.patch.plane == (now.width, now.height) && self.timeline.in_effect(s.step))
                .map(|s| (s.patch.rect, &s.patch.cells))
                .peekable();
            strokes.peek()?;
            now.within(&prunr_core::selection::MaskArtifact::from_patches(now.width, now.height, strokes))
        });
        self.invalidate_selection();
        if let Some(next) = next {
            self.selection_hash = Some(next.content_hash());
            self.selection_mask = Some(Arc::new(next));
        }
    }

    /// The image leaves or comes on screen: off screen it lets go of its
    /// unpacked neighbour steps, on screen it warms them.
    pub(crate) fn set_on_screen(&mut self, on: bool) {
        self.on_screen = on;
        if on {
            self.warm_neighbour_steps();
        } else {
            self.photo_look = None;
            let warm = std::mem::take(&mut *self.warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
            super::reclaim::later(warm);
            self.quick.clear();
        }
    }

    /// Unpack the images and planes of the steps either side of the shown
    /// one on a worker and hold them; drop whatever was held for other
    /// steps.
    fn warm_neighbour_steps(&self) {
        if !self.on_screen {
            return;
        }
        let back = self.timeline.previous().and_then(WarmKey::of);
        let ahead = self.timeline.next().and_then(WarmKey::of);
        for (at, key) in [(BACK, back), (AHEAD, ahead)] {
            let slot = Arc::clone(&self.warm_steps);
            let mut held = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match key {
                Some(key) if held[at].as_ref().is_some_and(|w| w.step.same(&key)) => {}
                Some(key) => {
                    held[at] = Some(WarmStep { step: key.clone(), image: None, planes: [None, None] });
                    drop(held);
                    std::thread::spawn(move || {
                        let image = key.result.as_ref().and_then(|r| r.load());
                        let planes = key.planes.clone().map(|p| p.and_then(|p| p.load()));
                        let mut held = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        // A newer step may have taken the slot meanwhile.
                        if let Some(w) = held[at].as_mut().filter(|w| w.step.same(&key)) {
                            w.image = image;
                            w.planes = planes;
                        }
                    });
                }
                None => held[at] = None,
            }
        }
    }

    /// Fold the working copy into the shown step: a job it started landed.
    pub(crate) fn amend(&mut self) {
        let shown = self.timeline.current();
        let state = DocState { at: shown.at, id: shown.id, made: shown.made.clone(), ..self.snapshot(shown.kind) };
        // The planes it replaces free off the UI thread: a drag amends every few frames.
        super::reclaim::later(std::mem::replace(self.timeline.current_mut(), state));
    }

    /// Keep what is on screen to reveal the next image over, when memory
    /// allows; otherwise the next image replaces it outright.
    pub(crate) fn hold_for_reveal(&mut self) {
        if !matches!(self.reveal, Some(Reveal::Held(_))) {
            self.reveal = self.quick.texture().cloned()
                .or_else(|| self.result_texture.clone())
                .or_else(|| self.source_texture.clone())
                .filter(|t| super::views::effects::reveal_fits(t.byte_size(), crate::hardware::available_ram_bytes_throttled()))
                .map(Reveal::Held);
        }
    }

    /// Paint a finished tile into its square of the run's preview, made
    /// on the run's first tile (or anew when the size changes). Only the
    /// tile is uploaded (`set_partial`).
    pub(crate) fn paint_tile_preview(&mut self, ctx: &egui::Context, tile: &prunr_core::TilePreview, image: Arc<egui::ColorImage>) {
        paint_tile(&mut self.tile_preview, ctx, &format!("tile-preview-{}", self.id), tile, image);
    }

    /// A face search of `photo` with `model` starts; what an earlier one
    /// showed goes.
    pub(crate) fn begin_face_search(&mut self, photo: FacePhoto, model: prunr_models::ModelId) {
        self.faces_pending = Some(PendingFaces { photo, model, found: None });
        self.face_preview = None;
    }

    /// The search under way has found `places`: kept to outline while it
    /// redraws them, when it is a search this image still wants.
    pub(crate) fn keep_found_faces(&mut self, places: Vec<prunr_core::face::FacePlace>, stage: FaceStage) {
        if !self.faces_pending.as_ref().is_some_and(|p| self.wants_faces(&p.photo, p.model, stage)) {
            return;
        }
        if let Some(pending) = self.faces_pending.as_mut() {
            pending.found = Some((places.into(), std::time::Instant::now()));
        }
    }

    /// The search whose redrawn faces the image may show over itself: the
    /// one under way, or else the one that has just found faces for the
    /// photo shown; either while the image still wants it.
    fn previewed_search(&self, stage: FaceStage) -> Option<(FacePhoto, prunr_models::ModelId)> {
        if let Some(p) = &self.faces_pending {
            return self.wants_faces(&p.photo, p.model, stage).then(|| (p.photo.clone(), p.model));
        }
        let (photo, model) = (self.face_photo(stage)?, self.settings.faces.wanted()?);
        self.found_for(&photo, model).filter(|f| !f.failed && !f.faces.is_empty()).map(|_| (photo, model))
    }

    /// A face the search has redrawn, laid in its place over the image,
    /// when the search is one the image still wants. A face that comes
    /// in the frame its search lands counts too.
    pub(crate) fn paint_face_preview(&mut self, ctx: &egui::Context, tile: &prunr_core::TilePreview, image: Arc<egui::ColorImage>, stage: FaceStage) {
        let Some((photo, model)) = self.previewed_search(stage).filter(|_| self.on_screen) else { return };
        let preview = match &mut self.face_preview {
            Some(p) if p.model == model && p.photo.same(&photo) => p,
            slot => slot.insert(FacePreview { photo, model, texture: None, blended: false }),
        };
        paint_tile(&mut preview.texture, ctx, &format!("face-preview-{}", self.id), tile, image);
    }

    /// Hold the start a render read back, while it is still the start a
    /// run redoes from and the image is on screen.
    pub(crate) fn keep_loaded_start(&mut self, start: Arc<image::RgbaImage>) {
        if self.on_screen && self.run_start(Engine::Segment).is_some_and(|s| s.holds(&start)) {
            self.start_photo = Some(start);
        }
    }

    /// The photo's look goes once the image is no longer the unprocessed
    /// photo, or Restore faces is off.
    pub(crate) fn settle_photo_look(&mut self) {
        if self.photo_look.is_some() && (self.has_result() || !self.settings.faces.on) {
            self.photo_look = None;
        }
    }

    /// The faces' preview, while there is one to lay over the image.
    pub(crate) fn face_preview_texture(&self) -> Option<&egui::TextureHandle> {
        self.face_preview.as_ref()?.texture.as_ref()
    }

    /// Drop the faces' preview once the image no longer wants the search
    /// it shows (another photo or model, Restore faces off, a search that
    /// found none, failed or stopped).
    pub(crate) fn settle_face_preview(&mut self, stage: FaceStage) {
        let Some(preview) = &self.face_preview else { return };
        let keep = self.previewed_search(stage).is_some_and(|(photo, model)| preview.model == model && preview.photo.same(&photo));
        if !keep {
            self.face_preview = None;
        }
    }

    /// A result with the found faces blended in has been applied: the
    /// preview goes once that result is on screen.
    fn faces_blended_in(&mut self) {
        if let Some(preview) = self.face_preview.as_mut() {
            preview.blended = true;
        }
    }

    /// A run has landed: its finished-tile preview, the whole image by
    /// now, is shown until the result's texture lands and the result
    /// reveals over it. Without one, the shown image is held as usual.
    pub(crate) fn hold_tile_preview_for_reveal(&mut self) {
        match self.tile_preview.take() {
            Some(preview) => self.reveal = Some(Reveal::Held(super::image_texture::ImageTexture::single(preview))),
            None => self.hold_for_reveal(),
        }
    }

    /// The new result's texture has landed: reveal it over the held image.
    pub(crate) fn start_reveal(&mut self) {
        match self.reveal.take() {
            Some(Reveal::Held(from)) => {
                let order = self.subject.as_ref().map(|s| Arc::clone(&s.order));
                self.reveal = Some(Reveal::Running(from, std::time::Instant::now(), order));
            }
            // Nothing held to reveal over: the outline has no reveal to fade with.
            None => self.subject = None,
            running => self.reveal = running,
        }
    }

    /// `rgba` is the result from now on; its texture and thumbnail are
    /// rebuilt from it.
    fn take_result(&mut self, rgba: Arc<image::RgbaImage>) {
        self.result_rgba = Some(rgba);
        self.result_tex_pending = true;
        self.thumb.asked();
        self.quick.clear();
    }

    /// Show `rgba` as the result: what is on screen, or the run's tile
    /// preview, is held to reveal it over, and a waiting image is done.
    pub(crate) fn show_result(&mut self, rgba: Arc<image::RgbaImage>) {
        self.hold_tile_preview_for_reveal();
        self.take_result(rgba);
        self.failure = None;
    }

    /// A texture of the result now held is to be made: any asked for
    /// earlier no longer lands. `live` swaps it in without a reveal.
    pub(crate) fn request_result_texture(&mut self, live: bool) -> super::background_io::TexTarget {
        use super::background_io::TexTarget;
        self.result_tex_request += 1;
        if live { TexTarget::LiveResult(self.result_tex_request) } else { TexTarget::Result(self.result_tex_request) }
    }

    /// The result texture goes with the image off screen; one still being
    /// made no longer lands.
    pub(crate) fn release_result_texture(&mut self) {
        self.result_texture = None;
        self.result_tex_pending = false;
        self.result_tex_request += 1;
    }

    /// A prepared texture lands on the side it was made for; a result's
    /// reveals over what was held.
    pub(crate) fn land_texture(&mut self, tex: super::image_texture::ImageTexture, target: super::background_io::TexTarget) {
        use super::background_io::TexTarget;
        match target {
            TexTarget::Source => {
                self.source_texture = Some(tex);
                self.source_tex_pending = false;
                tracing::info!(item_id = self.id, kind = "source", "texture uploaded");
            }
            TexTarget::QuickFrame(generation) => self.quick.land(tex, generation),
            TexTarget::PhotoLook => self.photo_look = Some(tex),
            TexTarget::Result(request) | TexTarget::LiveResult(request) if request != self.result_tex_request => {
                tracing::debug!(item_id = self.id, "superseded result texture dropped");
            }
            TexTarget::Result(_) | TexTarget::LiveResult(_) => {
                self.result_texture = Some(tex);
                self.result_tex_pending = false;
                if matches!(target, TexTarget::Result(_)) {
                    self.start_reveal();
                }
                // The faces it laid over the image are in the image now.
                if self.face_preview.as_ref().is_some_and(|p| p.blended) {
                    self.face_preview = None;
                }
                self.quick.result_landed();
                tracing::info!(item_id = self.id, kind = "result", "texture uploaded");
            }
        }
    }

    /// The shown step's result, unpacked here if parked.
    #[cfg(test)]
    pub(crate) fn shown_result(&self) -> Option<Arc<image::RgbaImage>> {
        self.result_pixels().and_then(|p| p.load())
    }

    /// A texture's pixels could not be read: the request ends, and a
    /// result's step is not asked for again while it shows.
    pub(crate) fn texture_unreadable(&mut self, target: super::background_io::TexTarget) {
        use super::background_io::TexTarget;
        match target {
            TexTarget::Result(request) | TexTarget::LiveResult(request) if request == self.result_tex_request => {
                self.result_tex_pending = false;
                self.unreadable_result = self.timeline.current().result.clone();
            }
            TexTarget::Source => self.source_tex_pending = false,
            _ => {}
        }
    }

    /// The shown step's result could not be read for its texture.
    pub(crate) fn result_unreadable(&self) -> bool {
        match (&self.unreadable_result, &self.timeline.current().result) {
            (Some(bad), Some(shown)) => Arc::ptr_eq(bad, shown),
            _ => false,
        }
    }

    /// A parked result a texture build unpacked: held again while the
    /// step still shows it, so its readers find it without a decode.
    pub(crate) fn hold_unpacked_result(&mut self, rgba: Arc<image::RgbaImage>) {
        if self.result_parked() && self.timeline.current().result.as_ref().is_some_and(|r| r.holds(&rgba)) {
            self.result_rgba = Some(rgba);
        }
    }

    /// The decoded source lands: the item holds its pixels (`true`), or
    /// shows why decoding failed.
    pub(crate) fn land_decode(&mut self, result: Result<Arc<image::RgbaImage>, String>) -> bool {
        self.decode_pending = false;
        match result {
            Ok(rgba) => {
                self.decode_failed = false;
                // The image's own file, which never changes under it: what
                // was made from it (depth, selection, embedding) stays.
                self.source_rgba = Some(rgba);
                true
            }
            Err(msg) => {
                self.decode_failed = true;
                self.failure = Some(prunr_core::plain::reason(msg));
                false
            }
        }
    }

    /// A selection overlay built off-thread lands when it is still for
    /// the mask shown: a patch at once, a whole one handed to `upload`
    /// (it stays pending until `install_selection_texture`).
    pub(crate) fn land_selection_texture(
        &mut self,
        result: super::background_io::SelectionTextureResult,
        upload: impl FnOnce(super::image_texture::PreparedImage, super::background_io::TexLanding),
    ) {
        use super::background_io::SelectionImage;
        let current = self.selection_hash == Some(result.key.0) && self.selection_mask.is_some();
        let super::background_io::SelectionTextureResult { item_id, image, key, shown } = result;
        match image {
            SelectionImage::Full { image } if current => {
                upload(image, super::background_io::TexLanding::Selection { item_id, key, shown });
            }
            SelectionImage::Full { .. } => {
                clear_if(&mut self.selection_tex_pending, key);
            }
            // A patch whose base texture has moved on is dropped; the
            // per-frame check asks again against the new base.
            SelectionImage::Patch { base_hash, pos, image } => {
                clear_if(&mut self.selection_tex_pending, key);
                if let Some(tex) = self.selection_texture.as_mut().filter(|t| current && t.key == (base_hash, key.1)) {
                    tex.texture.set_partial(pos, &image, egui::TextureOptions::LINEAR);
                    tex.key = key;
                    tex.shown = shown;
                }
            }
        }
    }

    /// A whole selection overlay uploaded: it shows while still for the
    /// mask shown.
    pub(crate) fn install_selection_texture(
        &mut self,
        texture: super::image_texture::ImageTexture,
        key: super::background_io::SelectionTextureKey,
        shown: Arc<prunr_core::selection::MaskArtifact>,
    ) {
        clear_if(&mut self.selection_tex_pending, key);
        if self.selection_hash == Some(key.0) && self.selection_mask.is_some() {
            self.selection_texture = Some(SelectionTexture { key, texture, shown });
        }
    }

    /// Whether an erase has something to fill.
    pub(crate) fn has_painted_region(&self) -> bool {
        self.selection_mask.as_ref().is_some_and(|m| m.has_selected_region())
    }

    /// A picture built for the backdrop lands when it is still the
    /// item's backdrop (`true`); a stale or empty build frees the slot
    /// for the next frame to build the current one.
    /// The picture built for the backdrop arrives: handed to `upload`
    /// while it is still the backdrop (it stays pending until
    /// `install_backdrop`). False when it was dropped.
    pub(crate) fn land_backdrop(
        &mut self,
        key: u64,
        image: Option<super::image_texture::PreparedImage>,
        upload: impl FnOnce(super::image_texture::PreparedImage, super::background_io::TexLanding, egui::TextureOptions),
    ) -> bool {
        match image.filter(|_| self.backdrop_key() == Some(key)) {
            Some(image) => {
                let options = if self.backdrop_repeats() {
                    egui::TextureOptions { wrap_mode: egui::TextureWrapMode::Repeat, ..egui::TextureOptions::LINEAR }
                } else {
                    egui::TextureOptions::LINEAR
                };
                upload(image, super::background_io::TexLanding::Backdrop { item_id: self.id, key }, options);
                true
            }
            None => {
                clear_if(&mut self.backdrop_tex_pending, key);
                false
            }
        }
    }

    /// An upload on its way is still worth finishing: what it lands on
    /// still waits for it. A parked result is no longer asked for (its
    /// request moved on when it was released).
    pub(crate) fn wants_upload(&self, landing: &super::background_io::TexLanding) -> bool {
        use super::background_io::{TexLanding, TexTarget};
        match landing {
            TexLanding::Image { target, .. } => match *target {
                TexTarget::Source => self.source_tex_pending,
                TexTarget::Result(request) | TexTarget::LiveResult(request) => request == self.result_tex_request,
                TexTarget::QuickFrame(generation) => self.quick.wants(generation),
                TexTarget::PhotoLook => self.on_screen,
            },
            TexLanding::Selection { key, .. } => self.selection_tex_pending == Some(*key),
            TexLanding::Backdrop { key, .. } => self.backdrop_key() == Some(*key),
        }
    }

    /// An upload was dropped part-way: the request it answered ends, so
    /// the next check asks again if the item still needs one. (An image's
    /// request already moved on, or it was never pending.)
    pub(crate) fn upload_dropped(&mut self, landing: &super::background_io::TexLanding) {
        use super::background_io::TexLanding;
        match landing {
            TexLanding::Image { .. } => {}
            TexLanding::Selection { key, .. } => clear_if(&mut self.selection_tex_pending, *key),
            TexLanding::Backdrop { key, .. } => clear_if(&mut self.backdrop_tex_pending, *key),
        }
    }

    /// The backdrop picture uploaded: it shows while it is still the
    /// backdrop. Returns whether it did.
    pub(crate) fn install_backdrop(&mut self, key: u64, texture: super::image_texture::ImageTexture) -> bool {
        clear_if(&mut self.backdrop_tex_pending, key);
        let current = self.backdrop_key() == Some(key);
        if current {
            self.backdrop_texture = Some((key, texture));
        }
        current
    }

    /// Faces found for this image are kept for the blend when they are
    /// for the photo and model it still wants: how many, or why the
    /// search failed. `None`: a search for something else, dropped.
    pub(crate) fn keep_faces(&mut self, landed: super::processor::FacesResult, stage: FaceStage) -> Option<Result<usize, String>> {
        // Outlined since the search found them: they do not arrive again.
        let mut found_at = std::time::Instant::now();
        if self.pending_for(&landed.photo, landed.model) {
            if let Some((_, at)) = self.faces_pending.take().and_then(|p| p.found) {
                found_at = at;
            }
        }
        if !self.wants_faces(&landed.photo, landed.model, stage) {
            return None;
        }
        let (faces, outcome) = match landed.result {
            Ok(faces) => {
                let found = faces.len();
                (faces, Ok(found))
            }
            Err(err) => (Vec::new(), Err(err)),
        };
        self.found_faces = Some(FoundFaces { photo: landed.photo, found_at, model: landed.model, faces: Arc::new(faces), failed: outcome.is_err(), stopped: landed.stopped });
        self.faces_changed();
        Some(outcome)
    }

    /// What the canvas draws for the image: its result's texture, else its
    /// photo's. Both are made off the UI thread, so neither may be up yet.
    pub(crate) fn canvas_texture(&self) -> Option<&super::image_texture::ImageTexture> {
        self.result_texture.as_ref().or(self.source_texture.as_ref())
    }

    /// The size of the frame the image is shown in: the result's, once
    /// there is one, else the source's. Every texture shown for this item
    /// (source, preview, held image, result) is drawn into it, so a
    /// hand-over between them never moves the image.
    pub(crate) fn frame_size(&self) -> egui::Vec2 {
        let (w, h) = self.shown_dims();
        egui::vec2(w as f32, h as f32)
    }

    /// The size of what the canvas shows: the result, parked or not (an
    /// upscale's output size), else the photo.
    pub(crate) fn shown_dims(&self) -> (u32, u32) {
        match (&self.result_rgba, &self.timeline.current().result) {
            (Some(rgba), _) => rgba.dimensions(),
            (None, Some(stored)) => stored.dims(),
            (None, None) => self.dimensions,
        }
    }

    /// The image shown until the new one's texture lands.
    pub(crate) fn held_texture(&self) -> Option<&super::image_texture::ImageTexture> {
        match &self.reveal {
            Some(Reveal::Held(t)) => Some(t),
            _ => None,
        }
    }

    /// A reveal under way: the image it reveals over, and how far it has
    /// come (0..1). A finished one is dropped by `settle_reveal`.
    pub(crate) fn reveal_progress(&self) -> Option<(&super::image_texture::ImageTexture, f32, Option<&[f32]>)> {
        let Some(Reveal::Running(from, at, order)) = &self.reveal else { return None };
        let p = (at.elapsed().as_secs_f64() / super::views::effects::REVEAL_SECS) as f32;
        (p < 1.0).then_some((from, p, order.as_deref()))
    }

    /// Release a finished reveal's textures, and the subject outline it
    /// grew from.
    pub(crate) fn settle_reveal(&mut self) {
        if matches!(self.reveal, Some(Reveal::Running(..))) && self.reveal_progress().is_none() {
            self.reveal = None;
            self.subject = None;
        }
    }

    /// Make the working copy the shown state again.
    pub(crate) fn restore(&mut self) {
        self.hold_for_reveal();
        self.quick.clear();
        // The restored step's face settings may differ from what the faced
        // photo and the cut-out over it were built with.
        self.faces_changed();
        self.subject = None;
        let shown = self.timeline.current().clone();
        // Off screen the restored step's result stays parked in the
        // timeline; on screen the neighbour warm-up has it unpacked, and
        // when it has not yet, the result texture's build unpacks it.
        self.result_rgba = shown.result.as_ref().filter(|_| self.on_screen).and_then(|r| r.loaded());
        self.applied_recipe = shown.recipe;
        self.selection_mask = shown.selection.as_ref().and_then(|s| s.load());
        self.selection_hash = self.selection_mask.as_ref().and(shown.selection_hash);
        self.edits = shown.edits.as_ref().and_then(|e| e.load());
        if self.settings.line_mode != shown.settings.line_mode {
            self.invalidate_edge_cache();
        }
        self.settings = shown.settings;
        self.applied_preset = shown.applied_preset;
        self.job = None;
        self.tile_preview = None;
        self.reset_result_caches();
        if shown.result.is_none() {
            // Back to the original: the source is already on the GPU.
            self.start_reveal();
        }
    }

    /// Step back one state; returns what the undone step did.
    pub(crate) fn undo_step(&mut self) -> Result<StepKind, StepRefused> {
        self.step(Timeline::previous, Timeline::undo)
    }

    /// What a save of the shown state writes: this image as it is now.
    pub(crate) fn save_mark(&self) -> SaveMark {
        SaveMark { id: self.id, revision: self.timeline.revision() }
    }

    /// A save of `mark` was written.
    pub(crate) fn mark_saved(&mut self, mark: SaveMark) {
        if mark.id == self.id {
            self.saved = self.saved.max(Some(mark.revision));
        }
    }

    /// Work that closing the app or removing the image would lose: a state
    /// past the photo as opened that no save wrote as shown, or a run
    /// under way.
    pub(crate) fn has_unsaved_work(&self) -> bool {
        self.job.is_some() || (!self.timeline.shows_opened() && self.saved != Some(self.timeline.revision()))
    }

    /// Step forward one state; returns what the redone step did.
    pub(crate) fn redo_step(&mut self) -> Result<StepKind, StepRefused> {
        self.step(Timeline::next, Timeline::redo)
    }

    /// Move to the step `target` names, unless its saved state is gone.
    fn step(&mut self, target: fn(&Timeline) -> Option<&DocState>, moved: fn(&mut Timeline) -> Option<StepKind>) -> Result<StepKind, StepRefused> {
        let to = target(&self.timeline).ok_or(StepRefused::Nothing)?;
        if !to.readable() {
            return Err(StepRefused::Lost);
        }
        let keeps_run = self.move_keeps_run(to);
        let kind = moved(&mut self.timeline).ok_or(StepRefused::Nothing)?;
        // The run goes on, its preview with it.
        let kept = if keeps_run { (self.job.take(), self.tile_preview.take()) } else { (None, None) };
        self.restore();
        if let (Some(mut job), preview) = kept {
            job.version = self.timeline.version();
            self.job = Some(job);
            self.tile_preview = preview;
        }
        self.warm_neighbour_steps();
        Ok(kind)
    }

    /// Note a run starting now; `step` is the step its result records,
    /// `None` to fill in the shown one (a re-run a knob change asked for).
    pub(crate) fn start_job(&mut self, step: Option<StepKind>, engine: Engine) {
        self.begin_job(step, engine, self.run_start(engine), None, None);
    }

    /// Note a run of `engine` starting now from `start` (`None`: the
    /// original), for a caller that chose where it starts.
    pub(crate) fn start_job_at(&mut self, step: Option<StepKind>, engine: Engine, start: Option<RunStart>) {
        self.begin_job(step, engine, start, None, None);
    }

    /// Note an erase of the shown selection starting now; `seed` is the
    /// one a generated fill runs with.
    pub(crate) fn start_erase(&mut self, seed: Option<u64>) {
        let region = self.timeline.current().selection.clone();
        self.begin_job(Some(StepKind::Erase), Engine::Erase, self.run_start(Engine::Erase), seed, region);
    }

    /// Note "Try another" starting now: the shown fill's region from its
    /// own start, with a new seed, recorded as a new step.
    pub(crate) fn start_reroll(&mut self, reroll: Reroll, seed: u64) {
        self.begin_job(Some(StepKind::Erase), Engine::Erase, reroll.start, Some(seed), Some(reroll.region));
    }

    /// The shown step as "Try another" would rerun it, when it is a
    /// generated fill.
    pub(crate) fn reroll(&self) -> Option<Reroll> {
        let origin = self.timeline.current().origin.as_ref()
            .filter(|o| o.engine == Engine::Erase && o.seed.is_some())?;
        Some(Reroll { start: origin.start.clone(), region: origin.region.clone()? })
    }

    fn begin_job(
        &mut self,
        step: Option<StepKind>,
        engine: Engine,
        start: Option<RunStart>,
        seed: Option<u64>,
        region: Option<super::timeline::StoredSelection>,
    ) {
        // Only a cut queues for a worker slot; other runs start at once.
        let running = engine != Engine::Segment;
        let since = self.timeline.current().at;
        self.job = Some(Job { run: next_run(), version: self.timeline.version(), step, engine, start, seed, region, cutter: None, keep: None, running, since, painted: Vec::new() });
        self.failure = None;
        self.tile_preview = None;
        self.subject = None;
    }

    /// Whether a new step cut by `cutter` builds on the cut-out on screen:
    /// a different model (or No model) made it, and the new one only takes
    /// more away. Undo gets back to before.
    pub(crate) fn cuts_anothers_cutout(&self, cutter: Cutter) -> bool {
        self.timeline.current().result.is_some() && self.shown_cutter().is_some_and(|c| c != cutter)
    }

    /// Who cut the step on screen, when a cut made it and that is known.
    pub(crate) fn shown_cutter(&self) -> Option<Cutter> {
        self.timeline.current().origin.as_ref().filter(|o| o.engine == Engine::Segment).and_then(|o| o.cutter)
    }

    /// Whether a run by `cutter` recording `step` builds on the cut-out on
    /// screen: a new step does; a knob's re-run redoes its step.
    pub(crate) fn chains(&self, step: Option<StepKind>, cutter: Cutter) -> bool {
        step.is_some() && self.cuts_anothers_cutout(cutter)
    }

    /// Note a cut by `cutter` starting now. One that `chains` starts from
    /// the shown cut-out's own colours, so its look is not applied twice,
    /// seen through its transparency; otherwise as `start_job`.
    pub(crate) fn start_cut(&mut self, step: Option<StepKind>, cutter: Cutter) {
        let keep = self.chains(step, cutter).then(|| self.timeline.current().result.clone()).flatten();
        self.start_job(step, Engine::Segment);
        if let Some(job) = &mut self.job {
            job.cutter = Some(cutter);
            job.keep = keep;
        }
    }

    /// A first cut by `cutter` that landed `result`, recorded as Process.
    #[cfg(test)]
    pub(crate) fn land_cut_for_test(&mut self, cutter: Cutter, result: Arc<image::RgbaImage>) {
        self.start_cut(Some(StepKind::Process), cutter);
        self.result_rgba = Some(result);
        self.finish_job();
    }

    /// The image a chained cut starts from, built where the run reads its
    /// file: it becomes the step's start, as an earlier image would be.
    pub(crate) fn chain_start_built(&mut self, run: u64, image: Arc<image::RgbaImage>) {
        if let Some(job) = self.job.as_mut().filter(|j| j.run == run && j.keep.is_some()) {
            job.start = Some(super::timeline::Stored::new(image));
        }
    }

    /// What a run reads besides its file: where it starts and, for a
    /// chained cut, the cut-out it keeps.
    pub(crate) fn job_run_image(&self) -> Option<super::worker::RunImage> {
        let job = self.job.as_ref()?;
        let start = job.start.as_ref().map(|s| Pixels::Parked(Arc::clone(s)));
        match &job.keep {
            // The decoded photo spares the bridge decoding the file again.
            Some(cutout) => Some(super::worker::RunImage::Kept {
                start: start.or_else(|| self.source_rgba.clone().map(Pixels::Resident)),
                cutout: Arc::clone(cutout),
            }),
            None => start.map(super::worker::RunImage::Start),
        }
    }

    /// The seed of the generated fill on screen.
    pub(crate) fn shown_seed(&self) -> Option<u64> {
        self.timeline.current().origin.as_ref().and_then(|o| o.seed)
    }

    /// Whether a landing result belongs to the state this item shows:
    /// an undo, a redo or another step since the run started makes it stale.
    pub(crate) fn job_is_current(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.version == self.timeline.version())
    }

    pub(crate) fn job_run(&self) -> Option<u64> {
        self.job.as_ref().map(|j| j.run)
    }

    /// The awaited run is `run`: its result is this job's, not an earlier
    /// run's that an undo left behind.
    pub(crate) fn awaits_run(&self, run: u64) -> bool {
        self.job_run() == Some(run)
    }

    /// The awaited run will not land (cancelled): the item shows its
    /// current state again.
    pub(crate) fn abandon_job(&mut self) {
        self.job = None;
        self.reveal = None;
        self.subject = None;
        self.tile_preview = None;
    }

    /// Record a landed run's result, now in the working copy.
    pub(crate) fn finish_job(&mut self) {
        let job = self.job.take();
        match job.as_ref().and_then(|j| j.step) {
            Some(kind) => self.commit(kind),
            None => self.amend(),
        }
        if let Some(job) = job {
            let engine = job.engine;
            self.timeline.current_mut().origin = Some(Origin {
                seed: job.seed,
                region: job.region,
                cutter: job.cutter,
                ..Origin::new(job.engine, job.start)
            });
            // Faces blend in later, when their re-cut lands.
            self.settle_faced(engine, false);
        }
    }

    /// Note whether the shown step's image has restored faces: blended by
    /// the run that just landed, or already in the image it started from.
    pub(crate) fn settle_faced(&mut self, engine: Engine, blended: bool) {
        let faced = blended || self.start_faced(engine);
        self.timeline.current_mut().faced = faced;
        if blended {
            self.faces_blended_in();
        }
    }

    fn start_faced(&self, engine: Engine) -> bool {
        self.run_start(engine).is_some_and(|s| self.timeline.image_faced(&s))
    }

    /// The photo at `stage` already has restored faces from an earlier
    /// step: restoring them again would sharpen them twice.
    pub(crate) fn faces_already_restored(&self, stage: FaceStage) -> bool {
        self.start_faced(stage.engine())
    }

    /// A run that keeps its output lands: its step is recorded, claiming
    /// the kept run `run` as its own.
    fn finish_job_keeping(&mut self, run: u64) {
        self.finish_job();
        if let Some(origin) = self.timeline.current_mut().origin.as_mut() {
            origin.kept_run = Some(run);
        }
    }

    /// The erase that made the shown step filled its region at `scale` of
    /// full size (the smallest, when regions differed).
    pub(crate) fn note_fill_scale(&mut self, scale: f32) {
        if let Some(origin) = self.timeline.current_mut().origin.as_mut() {
            origin.fill_scale = Some(scale);
        }
    }

    /// The kept run the shown step claims, if any.
    fn shown_kept_run(&self) -> Option<u64> {
        self.timeline.current().origin.as_ref().and_then(|o| o.kept_run)
    }

    /// An upscale lands: its step is recorded, and the run is kept for
    /// Refine as that step's own.
    pub(crate) fn land_upscale(&mut self, run: UpscaleCache) {
        self.finish_job_keeping(run.run);
        super::reclaim::later(self.upscale.replace(run));
    }

    /// What a run of `model` needs here, measured against the shown step
    /// when this model's run made it. A re-finish needs that step's run
    /// still kept; undo to an earlier one runs it again instead.
    pub(crate) fn kept_run_tier(&self, model: prunr_core::ModelKind) -> KeptRunTier {
        use prunr_core::RequiredTier;
        let (engine, kept) = if model.is_upscale() {
            (Engine::Upscale, self.kept_upscale().is_some())
        } else {
            (Engine::Colorize, self.kept_colorize().is_some())
        };
        let (Some(_), Some(old)) = (self.redo_origin(engine), &self.applied_recipe) else { return KeptRunTier::Full };
        match prunr_core::resolve_tier(old, &self.settings.current_recipe(model)) {
            RequiredTier::Skip | RequiredTier::CompositeOnly => KeptRunTier::UpToDate,
            RequiredTier::UpscaleTier2 | RequiredTier::ColorizeTier2 if kept => KeptRunTier::Refinish,
            _ => KeptRunTier::Full,
        }
    }

    /// Where the next upscale by `model` starts (`None`: the original):
    /// on the shown upscale when it is what the settings ask, so Process
    /// again enlarges it again; else where a run of the upscaler starts,
    /// which redoes a shown upscale under its changed settings.
    pub(crate) fn next_upscale_start(&self, model: prunr_core::ModelKind) -> Option<RunStart> {
        if self.kept_run_tier(model) == KeptRunTier::UpToDate {
            self.shown_start()
        } else {
            self.run_start(Engine::Upscale)
        }
    }

    /// The size of the image the next upscale by `model` starts from.
    pub(crate) fn next_upscale_dims(&self, model: prunr_core::ModelKind) -> (u32, u32) {
        self.next_upscale_start(model).map_or(self.dimensions, |s| s.dims())
    }

    /// A colorize lands: its step is recorded, and the run is kept for
    /// Strength as that step's own.
    pub(crate) fn land_colorize(&mut self, run: ColorizeCache) {
        self.finish_job_keeping(run.run);
        self.colorize = Some(run);
    }

    /// The kept colorize run, when it is the shown step's own.
    pub(crate) fn kept_colorize(&self) -> Option<&ColorizeCache> {
        let shown = self.shown_kept_run();
        self.colorize.as_ref().filter(|run| shown == Some(run.run))
    }


    /// The kept upscale run, when it is the shown step's own.
    pub(crate) fn kept_upscale(&self) -> Option<&UpscaleCache> {
        let shown = self.shown_kept_run();
        self.upscale.as_ref().filter(|run| shown == Some(run.run))
    }

    /// Record that `engine` made the shown step's image, starting from
    /// `start` (`None`: the original). The start is a stored timeline
    /// image, so the history sweep keeps compressing it.
    pub(crate) fn record_origin(&mut self, engine: Engine, start: Option<RunStart>, cutter: Option<Cutter>) {
        self.timeline.current_mut().origin = Some(Origin { cutter, ..Origin::new(engine, start) });
    }

    /// A live render landed on the shown step, from `start`, cut by
    /// `cutter` (`None`: it cut nothing).
    pub(crate) fn record_photo_pass(&mut self, start: Option<RunStart>, cutter: Option<Cutter>) {
        // A render built on another run's result (an erase's fill, an
        // upscale) leaves that run the step's own, with its seed, region
        // and kept run; only where the render began is noted.
        if let Some(start) = start.as_ref() {
            if let Some(origin) = self.timeline.current_mut().origin.as_mut().filter(|o| o.engine != Engine::Segment) {
                origin.rendered_from = Some(start.clone());
                return;
            }
        }
        // A pass that cut nothing keeps who cut the step; on a step nobody
        // cut (an erase, the photo) it records that no model did, since an
        // unrecorded cutter reads as the kept segmentation's cut, which may
        // be an undone one.
        let cutter = cutter.or_else(|| self.shown_cutter()).unwrap_or(Cutter::NoModel);
        self.record_origin(Engine::Segment, start, Some(cutter));
    }

    /// The shown step's origin when a run of `engine` would redo that step.
    /// The eraser never redoes: each erase adds to the last.
    fn redo_origin(&self, engine: Engine) -> Option<&Origin> {
        self.timeline.current().origin.as_ref().filter(|o| o.engine == engine && engine != Engine::Erase)
    }

    /// Where a run of `engine` starts (`None`: the original): the shown
    /// step's own start when the run redoes it, where the last render
    /// began for a render on another run's step, the shown result
    /// otherwise. Nothing is loaded.
    pub(crate) fn run_start(&self, engine: Engine) -> Option<RunStart> {
        if let Some(o) = self.redo_origin(engine) {
            return o.start.clone();
        }
        let rendered = self.timeline.current().origin.as_ref().and_then(|o| o.rendered_from.clone());
        rendered.filter(|_| engine == Engine::Segment).or_else(|| self.shown_start())
    }

    /// The shown result as a run's start (`None`: the original shows).
    fn shown_start(&self) -> Option<RunStart> {
        // The stored step knows its size: a parked result is not the photo's size.
        self.timeline.current().result.clone()
    }

    /// The photo Restore faces works on at `stage`: the shown upscale's
    /// output, or where a photo model's preview starts, as
    /// `build_preview_inputs` takes it.
    pub(crate) fn face_photo(&self, stage: FaceStage) -> Option<FacePhoto> {
        if self.faces_already_restored(stage) {
            return None;
        }
        match stage {
            FaceStage::Upscale => self.kept_upscale().map(|run| FacePhoto::Upscaled(Arc::clone(&run.raw))),
            FaceStage::Photo => self.run_start(Engine::Segment).map(FacePhoto::Start)
                .or_else(|| self.source_rgba.clone().map(FacePhoto::Original)),
        }
    }

    /// The faces found in `photo` with `model`, if those are the ones kept.
    pub(crate) fn found_for(&self, photo: &FacePhoto, model: prunr_models::ModelId) -> Option<&FoundFaces> {
        self.found_faces.as_ref().filter(|f| f.model == model && f.photo.same(photo))
    }

    pub(crate) fn pending_for(&self, photo: &FacePhoto, model: prunr_models::ModelId) -> bool {
        self.faces_pending.as_ref().is_some_and(|p| p.model == model && p.photo.same(photo))
    }

    /// The faces outlined on the image at `stage`: found for its photo, or
    /// found so far by the search of it under way.
    pub(crate) fn outlined_faces(&self, stage: FaceStage) -> Option<OutlinedFaces<'_>> {
        let photo = self.face_photo(stage)?;
        let model = self.settings.faces.wanted()?;
        if let Some(found) = self.found_for(&photo, model) {
            return Some(OutlinedFaces { found: &[], redrawn: &found.faces, found_at: found.found_at });
        }
        let (places, found_at) = self.faces_pending.as_ref().filter(|_| self.pending_for(&photo, model))?.found.as_ref()?;
        Some(OutlinedFaces { found: places, redrawn: &[], found_at: *found_at })
    }

    /// Whether the image wants faces from `photo` with `model`: Restore
    /// faces is on with that model and `photo` is the one shown.
    pub(crate) fn wants_faces(&self, photo: &FacePhoto, model: prunr_models::ModelId, stage: FaceStage) -> bool {
        self.settings.faces.wanted() == Some(model) && self.face_photo(stage).is_some_and(|p| p.same(photo))
    }

    /// The faces a preview at `stage` blends in.
    pub(crate) fn faces_input(&self, stage: FaceStage) -> Option<super::live_preview::FacesInput> {
        self.faces_for(&self.face_photo(stage)?)
    }

    /// The faces a preview of `photo` blends in, if Restore faces is on and
    /// they were found for that photo with the chosen model.
    pub(crate) fn faces_for(&self, photo: &FacePhoto) -> Option<super::live_preview::FacesInput> {
        let found = self.found_for(photo, self.settings.faces.wanted()?)?;
        let key = super::live_preview::FacedKey { faces: Arc::clone(&found.faces), blend: self.settings.faces.blend() };
        let faced = self.faced_photo.as_ref().filter(|(_, k)| k.matches(&key)).map(|(p, _)| Arc::clone(p));
        Some(super::live_preview::FacesInput { key, faced })
    }

    /// The found faces or how they blend changed: the faced photo and the
    /// cut-out built over it are out of date.
    pub(crate) fn faces_changed(&mut self) {
        self.faced_photo = None;
        self.cached_masked_base = None;
    }

    /// The face search `photo` needs, if any: Restore faces is on and
    /// nothing found or under way matches.
    pub(crate) fn faces_to_find(&self, photo: &FacePhoto) -> Option<prunr_models::ModelId> {
        let model = self.settings.faces.wanted()?;
        (self.found_for(photo, model).is_none() && !self.pending_for(photo, model)).then_some(model)
    }

    /// The pixels a run of `engine` starts from (`None`: the original).
    #[cfg(test)]
    pub(crate) fn run_input(&self, engine: Engine) -> Option<Arc<image::RgbaImage>> {
        self.run_start(engine).and_then(|s| s.load())
    }

    /// What an upscale or colorize run of `engine` reads: where its
    /// timeline says it starts, else the photo; loaded on the run's thread.
    pub(crate) fn run_input_or_source(&self, engine: Engine) -> Pixels {
        self.pixels_from(self.run_start(engine).as_ref())
    }

    /// The pixels a run from `start` reads (`None`: the photo).
    pub(crate) fn pixels_from(&self, start: Option<&RunStart>) -> Pixels {
        start.map_or_else(|| self.photo_pixels(), |s| Pixels::Parked(Arc::clone(s)))
    }

    /// Where the faces are in the original photo, when a Faces search has
    /// found them there already, so a depth run need not look again.
    pub(crate) fn faces_found_in_original(&self) -> Option<Vec<prunr_core::face::FacePlace>> {
        let source = self.source_rgba.as_ref()?;
        let original = |photo: &FacePhoto| matches!(photo, FacePhoto::Original(s) if Arc::ptr_eq(s, source));
        if let Some(found) = self.found_faces.as_ref().filter(|f| !f.failed && original(&f.photo)) {
            return Some(found.faces.iter().map(|f| *f.place()).collect());
        }
        let pending = self.faces_pending.as_ref().filter(|p| original(&p.photo))?;
        pending.found.as_ref().map(|(places, _)| places.to_vec())
    }

    /// The size of the image a run of `engine` starts from.
    pub(crate) fn run_input_dims(&self, engine: Engine) -> (u32, u32) {
        self.run_start(engine).map_or(self.dimensions, |s| s.dims())
    }

    /// The pixels the awaited run starts from (`None`: the original),
    /// loaded when the run is sent.
    #[cfg(test)]
    pub(crate) fn job_input(&self) -> Option<Arc<image::RgbaImage>> {
        self.job.as_ref()?.start.as_ref()?.load()
    }

    /// What a full run on this image sends: its file and the pixels the
    /// run starts from, read where the run is sent. The image shows as
    /// processing once it is sent (`mark_running`).
    pub(crate) fn full_run_work(&self) -> super::worker::WorkItem {
        super::worker::WorkItem { id: self.id, run: self.job_run().unwrap_or_default(), source: self.source.clone(), image: self.job_run_image() }
    }

    /// Drop whatever caches a `CacheImpact` says are stale.
    pub(crate) fn apply_cache_impact(
        &mut self,
        impact: crate::gui::knob_catalog::CacheImpact,
    ) {
        use crate::gui::knob_catalog::CacheImpact;
        match impact {
            CacheImpact::Nothing => {}
            CacheImpact::EdgeCache => self.invalidate_edge_cache(),
            CacheImpact::SegCache => self.set_cached_tensor(None),
            CacheImpact::Both => self.drop_tensor_caches(),
        }
    }

    /// How much of the pipeline a Process run needs here, from what last
    /// ran to what the settings ask for now. Caches the run can't reuse
    /// are dropped, and a change the run doesn't touch is noted on the
    /// applied recipe so the image reads as up to date.
    pub(crate) fn run_tier(&mut self, model: prunr_core::ModelKind) -> RunTier {
        use prunr_core::RequiredTier;
        let Some(ref old_recipe) = self.applied_recipe else {
            return RunTier::Full;
        };
        // A run that starts from an earlier result works on pixels the
        // cached tensor never saw, unless it redoes the shown cut-out
        // from that cut-out's own starting image.
        if self.run_start(Engine::Segment).is_some() {
            if self.redo_origin(Engine::Segment).is_none() {
                self.set_cached_tensor(None);
            }
            return RunTier::Full;
        }
        let current = self.settings.current_recipe(model);
        let tier = prunr_core::resolve_tier(old_recipe, &current);
        let impact = crate::gui::knob_catalog::cache_impact_for_recipe_diff(old_recipe, &current);
        self.apply_cache_impact(impact);
        let has_tensor = self.cached_tensor.is_some();
        match tier {
            // A backdrop, Refine or Strength change re-renders at display
            // and export time, not through the dispatcher; the stored
            // composite follows so status reads stay accurate.
            RequiredTier::Skip
            | RequiredTier::CompositeOnly
            | RequiredTier::UpscaleRerun
            | RequiredTier::UpscaleTier2
            | RequiredTier::ColorizeTier2 => {
                if let Some(recipe) = &mut self.applied_recipe {
                    recipe.composite = current.composite.clone();
                }
                RunTier::UpToDate
            }
            // Lines on the subject compose onto the new masked base, which
            // a mask-only rerun would drop.
            RequiredTier::MaskRerun if has_tensor && current.inference.uses_edge_detection => RunTier::AddEdge,
            RequiredTier::MaskRerun if has_tensor => RunTier::Mask,
            RequiredTier::AddEdgeInference if has_tensor => RunTier::AddEdge,
            // Lines on the subject trace the cut-out the cached tensor
            // makes again; over the whole photo there is no cut-out.
            RequiredTier::EdgeRerun if has_tensor && current.inference.uses_segmentation && current.inference.uses_edge_detection => RunTier::AddEdge,
            RequiredTier::MaskRerun
            | RequiredTier::AddEdgeInference
            | RequiredTier::EdgeRerun
            | RequiredTier::FullPipeline => RunTier::Full,
        }
    }

    /// Reset all caches tied to the current result, after it changed, so
    /// textures rebuild and the next reprocess runs from scratch.
    /// `source_texture` stays: the caller decides.
    pub(crate) fn reset_result_caches(&mut self) {
        // `cached_tensor` and the kept upscale run stay: the tensor depends only
        // on source + model, and Refine knobs re-finish from the upscale run.
        // Paths that invalidate them clear them explicitly.
        self.result_texture = None;
        // A result texture still being made is of the result this drops.
        self.result_tex_request += 1;
        self.thumb.clear();
        self.source_tex_pending = false;
        self.result_tex_pending = false;
    }

    /// Apply a finished tier result (success or error). Returns the
    /// `active_provider` for the app's backend label; `None` for reruns that
    /// ran no model, and for errors.
    pub(crate) fn apply_tier_result(
        &mut self,
        result: Result<prunr_core::ProcessResult, String>,
        seg: Option<super::worker::CompressedTensor>,
        edges: Option<super::worker::CompressedEdgeTensors>,
        recipe_snapshot: prunr_core::ProcessingRecipe,
    ) -> Option<String> {
        match result {
            Ok(pr) => {
                self.reset_result_caches();
                self.result_rgba = Some(Arc::new(pr.rgba_image));
                self.failure = None;
                self.applied_recipe = Some(recipe_snapshot);
                // Keep the cache when the worker returned no fresh tensor (a
                // rerun): clobbering it killed live preview after any rerun.
                if let Some(new) = seg {
                    self.set_cached_tensor(Some(new));
                }
                if let Some(new) = edges {
                    self.invalidate_edge_cache();
                    self.cached_edge_tensors = Some(new);
                }
                self.cached_masked_base = None;
                // source_rgba stays on every item: live preview re-cuts from it.
                // Tier 2 reruns report empty active_provider (no inference ran).
                (!pr.active_provider.is_empty()).then_some(pr.active_provider)
            }
            Err(e) => {
                // Clear recipe + tensors so retry runs a fresh Tier 1
                // (otherwise resolve_tier might return Skip for an errored item).
                self.fail(e);
                self.drop_tensor_caches();
                // An image that still shows a result keeps the runs that
                // result is refined from.
                if !self.has_result() {
                    self.invalidate_upscale_cache();
                    self.invalidate_colorize_cache();
                }
                self.applied_recipe = None;
                None
            }
        }
    }

    /// The shown step has a result, on screen or parked in the timeline.
    pub(crate) fn has_result(&self) -> bool {
        self.result_rgba.is_some() || self.timeline.current().result.is_some()
    }

    /// A stroke re-cuts this item from its tensor only while a cut-out is
    /// on screen. After undoing back to the original the tensor stays
    /// cached, and a re-cut there would bring the undone result back.
    pub(crate) fn shows_cutout(&self) -> bool {
        super::capability::ItemFacts::of(self).cutout_mask
    }

    /// The cached segmentation is what made the shown step: after an undo
    /// to a step another model made (or an upscale), a re-cut from it would
    /// rewrite that step with the wrong model's cut-out.
    pub(crate) fn tensor_made_shown_step(&self) -> bool {
        let Some(tensor) = &self.cached_tensor else { return false };
        match self.shown_cutter() {
            Some(cutter) => cutter == Cutter::Model(tensor.model),
            // A cut step from before cutters were recorded: its recipe
            // says who, and with none it was this segmentation's.
            None => self.applied_recipe.as_ref().is_none_or(|r| r.inference.model == tensor.model),
        }
    }

    /// Whether the selection corrects the cut-out on screen under `model`,
    /// rather than acting on the image as a plain selection.
    pub(crate) fn corrects_cutout(&self, model: super::settings::SettingsModel) -> bool {
        model.edits_cutout() && self.shows_cutout()
    }

    /// Release the working result while the item is in the background;
    /// the timeline keeps it and compresses it once nothing else holds it.
    pub(crate) fn park_result(&mut self) -> bool {
        if self.status() != BatchStatus::Done {
            return false;
        }
        let Some(rgba) = self.result_rgba.as_ref() else { return false };
        if !self.timeline.current().result.as_ref().is_some_and(|r| r.holds(rgba)) {
            self.amend();
        }
        self.result_rgba = None;
        true
    }

    /// A result that landed while the image is off screen goes back to its
    /// step, with its texture, once no run is under way.
    pub(crate) fn park_off_screen(&mut self) {
        if self.on_screen || self.job.is_some() || self.result_rgba.is_none() {
            return;
        }
        if self.park_result() {
            self.release_result_texture();
            self.reveal = None;
        }
    }

    /// Bring a parked result back from the timeline. Every result reader
    /// and writer calls this first; a result that landed meanwhile wins.
    #[cfg(test)]
    pub(crate) fn unpark_result(&mut self) {
        if self.result_parked() {
            self.result_rgba = self.timeline.current().result.as_ref().and_then(|r| r.load());
        }
    }

    /// What an export of this image writes: what the canvas shows, the
    /// result (parked or not) or the photo. Nothing is unpacked here.
    pub(crate) fn export_image(&self) -> ExportImage {
        match self.result_pixels() {
            Some(pixels) => ExportImage { pixels, photo: false, faces: None },
            None => ExportImage { pixels: self.photo_pixels(), photo: true, faces: self.faces_input(FaceStage::Photo) },
        }
    }

    /// The shown step's result, on screen or parked.
    pub(crate) fn result_pixels(&self) -> Option<Pixels> {
        self.result_rgba.clone().map(Pixels::Resident)
            .or_else(|| self.timeline.current().result.clone().map(Pixels::Parked))
    }

    /// The photo, however far it is decoded.
    pub(crate) fn photo_pixels(&self) -> Pixels {
        self.source_rgba.clone().map_or_else(|| Pixels::Source(self.source.clone()), Pixels::Resident)
    }

    /// The shown state has a result that the working copy has released.
    fn result_parked(&self) -> bool {
        self.result_rgba.is_none() && self.timeline.current().result.is_some()
    }

    /// What `enforce_tensor_budget` can free: the compressed segmentation
    /// and edge tensors.
    pub(crate) fn evictable_tensor_bytes(&self) -> usize {
        let seg = self.cached_tensor.as_ref().map_or(0, |ct| ct.compressed_size());
        let edge = self.cached_edge_tensors.as_ref().map_or(0, |ct| ct.compressed_size());
        let hot_seg = self.volatile_seg_tensor.as_ref().map_or(0, |t| t.data.len() * 4);
        let hot_edge = self.volatile_edge_tensor.as_ref().map_or(0, |(_, d)| d.len() * 4);
        seg + edge + hot_seg + hot_edge
    }

    /// The segmentation a live render re-cuts: the copy a render of this
    /// drag unpacked, else the packed one for the render's thread.
    pub(crate) fn render_seg(&self) -> (Option<Arc<super::live_preview::SegTensor>>, Option<super::worker::CompressedTensor>) {
        // Only the cut this segmentation made is re-cut with it: not the
        // photo after an undo, an upscale over it, another model's cut, or
        // No model's pass, which carries its cut in its start.
        if !self.shows_cutout() {
            return (None, None);
        }
        match &self.volatile_seg_tensor {
            Some(hot) => (Some(Arc::clone(hot)), None),
            None => (None, self.cached_tensor.clone()),
        }
    }

    /// The segmentation a live render re-cuts, unpacked here.
    #[cfg(test)]
    pub(crate) fn hot_seg_tensor(&self) -> Option<Arc<super::live_preview::SegTensor>> {
        let (hot, packed) = self.render_seg();
        hot.or_else(|| packed.and_then(|ct| super::live_preview::decompress_seg(&ct)).map(Arc::new))
    }

    /// Keep what a render unpacked for the drag's next ticks, while the
    /// item's caches are still the ones it came from; the line scale
    /// only while it is the active one.
    pub(crate) fn keep_unpacked(&mut self, unpacked: super::live_preview::Unpacked) {
        // Off screen the hot copies are let go; one landing late stays out.
        if !self.on_screen {
            return;
        }
        if let Some((from, seg)) = unpacked.seg {
            if self.cached_tensor.as_ref().is_some_and(|c| c.same(&from)) {
                self.volatile_seg_tensor = Some(seg);
            }
        }
        if let Some((from, scale, data)) = unpacked.edge {
            let current = self.cached_edge_tensors.as_ref().is_some_and(|c| c.same_scale(&from, scale));
            if current && scale == self.settings.edge_settings().primary_scale() {
                self.volatile_edge_tensor = Some((scale, data));
            }
        }
    }

    /// Test-only: a BatchItem with every field at its zero/None default.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn new_for_test() -> Self {
        Self::new(
            0,
            "test.png".to_string(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (100, 100),
            super::item_settings::ItemSettings::default(),
            String::new(),
        )
    }

    pub(crate) fn new(
        id: u64,
        filename: String,
        source: ImageSource,
        dimensions: (u32, u32),
        settings: super::item_settings::ItemSettings,
        applied_preset: String,
    ) -> Self {
        let source_len = match &source {
            ImageSource::Bytes(bytes) => bytes.len(),
            ImageSource::Path(..) => 0,
        };
        Self {
            id,
            filename,
            source,
            source_len,
            dimensions,
            source_rgba: None,
            source_texture: None,
            thumb: Thumbnail::default(),
            result_rgba: None,
            start_photo: None,
            result_texture: None,
            quick: Default::default(),
            photo_look: None,
            source_tex_pending: false,
            result_tex_pending: false,
            unreadable_result: None,
            result_tex_request: 0,
            reveal: None,
            subject: None,
            tile_preview: None,
            warm_steps: Arc::default(),
            on_screen: false,
            found_faces: None,
            faces_pending: None,
            faced_photo: None,
            face_preview: None,
            decode_pending: false,
            timeline: Timeline::new(DocState::original(settings, applied_preset.clone())),
            job: None,
            saved: None,
            failure: None,
            decode_failed: false,
            ticked: false,
            settings,
            applied_recipe: None,
            cached_tensor: None,
            upscale: None,
            colorize: None,
            cached_edge_tensors: None,
            volatile_edge_tensor: None,
            volatile_seg_tensor: None,
            edge_planes: Default::default(),
            cached_masked_base: None,
            applied_preset,
            bg_image: None,
            backdrop_texture: None,
            lens_focus: None,
            backdrop_cut: BackdropCut::default(),
            backdrop_tex_pending: None,
            selection_mask: None,
            selection_hash: None,
            edits: None,
            correction_cache: std::sync::Mutex::default(),
            selection_tex_pending: None,
            selection_texture: None,
            magic_brush_embedding: None,
            adjusting: None,
            magic_preencode_refused: false,
            depth: None,
            depth_run: DepthRun::default(),
            depth_photo: None,
            quick_photo: None,
        }
    }

    /// Lockstep writer for `bg_image` + `settings.bg_image_hash`. Going
    /// through this path is the invariant the recipe diff relies on.
    #[cfg(test)]
    pub(crate) fn set_bg_image(&mut self, img: image::DynamicImage, source_path: Option<PathBuf>) {
        let hash = bg_image_content_hash(&img);
        self.set_bg_image_hashed(img, source_path, hash);
    }

    /// `set_bg_image` for a picture hashed already (off the UI thread).
    pub(crate) fn set_bg_image_hashed(&mut self, img: image::DynamicImage, source_path: Option<PathBuf>, hash: u64) {
        let picture = Arc::new(BgImage { source_path, image: Arc::new(img), hash });
        super::reclaim::later(self.bg_image.replace(picture));
        self.settings.bg_image_hash = std::num::NonZeroU64::new(hash);
    }

    /// Take another image's settings and its background picture together,
    /// keeping the picture in step with the settings' reference to it.
    pub(crate) fn adopt_settings(&mut self, settings: super::item_settings::ItemSettings, bg_image: Option<Arc<BgImage>>) {
        self.settings = settings;
        self.bg_image = bg_image;
    }

    pub(crate) fn clear_bg_image(&mut self) {
        self.bg_image = None;
        self.settings.bg_image_hash = None;
    }

    /// What an erase or a selection edit acts on: the result shown (an
    /// erase builds on earlier fills), else the photo.
    pub(crate) fn edit_pixels(&self) -> Pixels {
        self.result_pixels().unwrap_or_else(|| self.photo_pixels())
    }

    /// The depth map to save beside this item's result.
    pub(crate) fn depth_export(&self) -> DepthExport {
        DepthExport { name: depth_file_name(&self.filename), dims: self.dimensions, depth: self.depth_source() }
    }

    pub(crate) fn depth_source(&self) -> DepthSource {
        DepthSource { depth: self.depth.clone(), source: self.source.clone() }
    }

    /// What export puts behind this item's result: only shared handles,
    /// so it can travel to a save thread and bake there.
    pub(crate) fn export_bg(&self) -> ExportBg {
        ExportBg {
            backdrop: self.settings.backdrop(),
            image: self.bg_image.clone(),
            fit: self.settings.bg_image_fit,
            photo: self.backdrop_photo(),
            depth: self.depth_source(),
        }
    }

    /// The photo an effect background is made from: the image the shown
    /// result's run started from (`Parked`: a photo with an object erased
    /// first, unpacked where it is used), or the original (`Resident`).
    pub(crate) fn backdrop_photo(&self) -> Option<Pixels> {
        match self.timeline.latest_run().and_then(|o| o.start.as_ref()) {
            Some(start) => Some(Pixels::Parked(Arc::clone(start))),
            None => self.source_rgba.clone().map(Pixels::Resident),
        }
    }

    /// What the background texture is built from: the picture or effect
    /// and, for an effect, which photo it is made from.
    pub(crate) fn backdrop_key(&self) -> Option<u64> {
        use super::item_settings::Backdrop;
        let backdrop = self.settings.backdrop();
        let key = backdrop.picture_key()?;
        let key = match (backdrop, self.backdrop_photo()) {
            (Backdrop::Effect(_), Some(Pixels::Parked(start))) => key ^ (Arc::as_ptr(&start) as u64).rotate_left(17),
            _ => key,
        };
        // A lens blur is made with the depth map, focused on the result's
        // subject; a blur fills the subject from the background first.
        Some(match backdrop {
            // Clamped, the far edge would bleed into the near one.
            _ if self.backdrop_repeats() => key ^ TILED_BACKDROP,
            Backdrop::Effect(effect) => {
                let lens = effect.needs_depth().then(|| {
                    let depth = self.depth.as_ref().map_or(0, |d| Arc::as_ptr(d) as u64);
                    let focus = self.lens_focus.as_ref().map_or(0, |f| u64::from(f.value.to_bits()));
                    depth.rotate_left(29) ^ focus.rotate_left(43)
                });
                let cut = effect.takes_cutout().then(|| self.backdrop_cut.version.rotate_left(53));
                key ^ lens.unwrap_or(0) ^ cut.unwrap_or(0)
            }
            _ => key,
        })
    }

    /// The backdrop is a picture repeated across the frame.
    pub(crate) fn backdrop_repeats(&self) -> bool {
        self.settings.backdrop().fit(self.settings.bg_image_fit) == prunr_core::BgImageFit::Tile
    }

    /// Note a result other than the one the backdrop's cut-out was, so
    /// Blurred photo and Lens blur are made again.
    pub(crate) fn refresh_backdrop_cut(&mut self) {
        if !same_result(self.backdrop_cut.result.as_ref(), self.result_rgba.as_ref()) {
            self.backdrop_cut = BackdropCut {
                result: self.result_rgba.as_ref().map(Arc::downgrade),
                version: self.backdrop_cut.version + 1,
            };
        }
    }

    /// Measure the lens-blur focus again when the depth map or the result
    /// is another one; a result with the subject where it was gives the
    /// same focus, so `backdrop_key` stays and nothing is rebuilt.
    pub(crate) fn refresh_lens_focus(&mut self) {
        let Some(depth) = &self.depth else {
            self.lens_focus = None;
            return;
        };
        // A parked result is not unpacked to place the focus: it is placed
        // again once the result's texture build hands the pixels back.
        if self.result_parked() || self.lens_focus.as_ref().is_some_and(|f| f.is_for(depth, self.result_rgba.as_ref())) {
            return;
        }
        self.lens_focus = Some(LensFocus {
            depth: Arc::downgrade(depth),
            result: self.result_rgba.as_ref().map(Arc::downgrade),
            value: prunr_core::depth::backdrop_focus(depth, self.result_rgba.as_deref()),
        });
    }


}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BatchStatus {
    Pending,
    Processing,
    Done,
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item_settings::ItemSettings;

    fn fixture_item(id: u64) -> BatchItem {
        BatchItem { id, ..BatchItem::new_for_test() }
    }

    /// Work is unsaved from the first step until a save of the state
    /// shown; a later change, or a run under way, makes it unsaved again.
    /// The photo as it was opened is on disk already.
    #[test]
    fn work_is_unsaved_until_the_state_shown_is_saved() {
        let mut item = fixture_item(1);
        assert!(!item.has_unsaved_work(), "the photo as opened");
        item.commit(StepKind::Adjust);
        assert!(item.has_unsaved_work(), "a step");
        let saved = item.save_mark();
        item.commit(StepKind::Stroke);
        item.mark_saved(saved);
        assert!(item.has_unsaved_work(), "a save of an earlier state");
        item.mark_saved(item.save_mark());
        assert!(!item.has_unsaved_work(), "saved as shown");
        item.undo_step().unwrap();
        assert!(item.has_unsaved_work(), "an undo shows a state not saved");
        item.undo_step().unwrap();
        assert!(!item.has_unsaved_work(), "back to the photo as opened");
        item.start_job(Some(StepKind::Process), Engine::Segment);
        assert!(item.has_unsaved_work(), "a run under way");
    }

    /// Two saves of one image can finish out of order; the older one
    /// landing last does not make the newer one's state unsaved.
    #[test]
    fn an_older_save_landing_last_keeps_the_newer_one() {
        let mut item = fixture_item(1);
        item.commit(StepKind::Adjust);
        let older = item.save_mark();
        item.commit(StepKind::Stroke);
        item.mark_saved(item.save_mark());
        item.mark_saved(older);
        assert!(!item.has_unsaved_work());
    }

    /// Past the undo depth the photo as opened is gone, so the oldest
    /// step kept is work like any other.
    #[test]
    fn the_oldest_step_kept_is_unsaved_once_the_opened_photo_is_dropped() {
        let mut item = fixture_item(1);
        for _ in 0..crate::gui::timeline::TIMELINE_DEPTH + 2 {
            item.commit(StepKind::Stroke);
        }
        while item.undo_step().is_ok() {}
        assert!(item.has_unsaved_work());
    }

    /// A parked result still tells its size, without unpacking it.
    #[test]
    fn a_parked_result_tells_its_size() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(6, 4)));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(StepKind::Process);
        assert!(item.park_result());
        assert_eq!(item.result_pixels().and_then(|p| p.decoded_dims()), Some((6, 4)));
    }

    /// Parking a saved image's result when it leaves the screen keeps it
    /// saved: what it shows did not change.
    #[test]
    fn a_saved_image_stays_saved_when_parked() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(StepKind::Process);
        item.mark_saved(item.save_mark());
        assert!(item.park_result());
        assert!(!item.has_unsaved_work());
    }

    fn selection(cell: usize) -> Arc<prunr_core::selection::MaskArtifact> {
        let mut cells = vec![0i8; 16];
        cells[cell] = prunr_core::selection::FULL;
        Arc::new(prunr_core::selection::MaskArtifact::from_cells(4, 4, cells))
    }

    /// Paint Brush strokes wait for a run on the image, but an erase: a
    /// selection painted while it fills is the next area, and keeps the
    /// fill. A cut-out correction changes pixels, so it waits for any run.
    #[test]
    fn strokes_wait_for_a_run_but_an_erase() {
        let mut item = fixture_item(1);
        assert!(item.takes_strokes(false) && item.takes_strokes(true));
        item.start_erase(None);
        assert!(item.takes_strokes(false), "a selection while an erase fills");
        assert!(!item.takes_strokes(true), "a cut-out correction waits");
        item.abandon_job();
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        assert!(!item.takes_strokes(false), "an upscale waits as before");
    }

    /// What was painted while an erase filled is the next area once the
    /// fill lands, but only strokes still in effect: one undone during the
    /// fill leaves nothing of itself, one redone counts again.
    #[test]
    fn a_stroke_undone_during_an_erase_is_no_next_area() {
        let cells = |on: &[usize]| {
            let mut cells = vec![0i8; 16];
            on.iter().for_each(|&c| cells[c] = prunr_core::selection::FULL);
            Arc::new(prunr_core::selection::MaskArtifact::from_cells(4, 4, cells))
        };
        let next_area = |undo: bool, redo: bool| {
            let mut item = fixture_item(1);
            assert!(item.commit_selection_mask(cells(&[5]), StepKind::Stroke));
            item.start_erase(None);
            let stroke = cells(&[5, 9]);
            assert!(item.commit_selection_mask(Arc::clone(&stroke), StepKind::Stroke));
            item.note_painted_during_erase(StrokePatch::of(&stroke).unwrap());
            if undo {
                item.undo_step().unwrap();
            }
            if redo {
                item.redo_step().unwrap();
            }
            item.clear_filled_marks();
            item.selection_mask.as_ref().map(|m| m.has_selected_region())
        };
        assert_eq!(next_area(false, false), Some(true), "the stroke is the next area");
        assert_eq!(next_area(true, false), None, "undone, it leaves none");
        assert_eq!(next_area(true, true), Some(true), "redone, it counts again");
    }

    /// Undo and redo between strokes painted while an erase fills keep it;
    /// an undo past where it started leaves it behind.
    #[test]
    fn steps_within_an_erase_keep_it_and_one_past_its_start_does_not() {
        let mut item = fixture_item(1);
        assert!(item.commit_selection_mask(selection(5), StepKind::Stroke));
        item.start_erase(None);
        assert!(item.commit_selection_mask(selection(9), StepKind::Stroke));
        assert!(item.job_is_current(), "a stroke keeps the erase");
        assert!(item.undo_keeps_run());
        item.undo_step().unwrap();
        assert!(item.job_is_current(), "undoing it too");
        assert!(item.redo_keeps_run());
        item.redo_step().unwrap();
        assert!(item.job_is_current(), "and redoing it");
        item.undo_step().unwrap();
        assert!(!item.undo_keeps_run(), "the stroke the erase was made from");
    }

    #[test]
    fn an_image_is_running_while_a_model_works_on_it() {
        let mut item = fixture_item(1);
        assert!(!item.is_running());
        item.start_job(Some(StepKind::Process), Engine::Segment);
        assert!(!item.is_running(), "a queued cut-out waits");
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Processing);
        assert!(item.is_running(), "the worker has taken it");
        for engine in [Engine::Upscale, Engine::Colorize, Engine::Erase] {
            let mut item = fixture_item(2);
            crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
            item.start_job(None, engine);
            assert!(item.is_running(), "{engine:?} runs from its start");
        }
    }

    /// A tile that is the whole preview becomes the texture in one upload;
    /// a part lands in a blank texture of the preview's size, and a later
    /// tile updates only its part.
    #[test]
    fn a_whole_preview_tile_is_one_upload() {
        let ctx = egui::Context::default();
        let tile = |x: u32, y: u32, w: u32, h: u32| prunr_core::TilePreview {
            x, y, width: w, height: h, full_width: 4, full_height: 3, rgba: vec![9; (w * h * 4) as usize],
        };
        let image = |w: usize, h: usize| Arc::new(egui::ColorImage::new([w, h], vec![egui::Color32::WHITE; w * h]));
        let uploads = |texture: &Option<egui::TextureHandle>| {
            let id = texture.as_ref().expect("a texture").id();
            ctx.tex_manager().write().take_delta().set.into_iter().filter(|(t, _)| *t == id).map(|(_, d)| d.pos).collect::<Vec<_>>()
        };
        let mut whole = None;
        paint_tile(&mut whole, &ctx, "whole", &tile(0, 0, 4, 3), image(4, 3));
        assert_eq!(uploads(&whole), [None], "the tile is the texture");
        paint_tile(&mut whole, &ctx, "whole", &tile(1, 1, 2, 2), image(2, 2));
        assert_eq!(uploads(&whole), [Some([1, 1])], "a later tile updates its part");
        let mut part = None;
        paint_tile(&mut part, &ctx, "part", &tile(1, 1, 2, 2), image(2, 2));
        assert_eq!(uploads(&part), [None, Some([1, 1])], "a first part lands in a blank preview");
    }

    /// A landed upscale reveals over its finished-tile preview, which
    /// stays on screen until the result's texture lands; a new run never
    /// shows the last run's tiles.
    #[test]
    fn an_upscale_reveals_over_its_tile_preview() {
        let ctx = egui::Context::default();
        let preview = ctx.load_texture("preview", egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let mut item = fixture_item(1);
        item.tile_preview = Some(preview.clone());
        item.hold_tile_preview_for_reveal();
        assert_eq!(item.held_texture().map(|t| t.id()), Some(preview.id()), "the preview is held");
        assert!(item.tile_preview.is_none());
        item.tile_preview = Some(preview);
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        assert!(item.tile_preview.is_none(), "a new run starts clear");
    }

    /// An undo during a run drops the run's tiles with the run: the pill
    /// shows nothing of it while its cancel completes.
    #[test]
    fn an_undo_drops_the_runs_tile_preview() {
        let ctx = egui::Context::default();
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        item.commit(StepKind::Process);
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.tile_preview = Some(ctx.load_texture("preview", egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default()));
        assert!(item.undo_step().is_ok());
        assert!(item.tile_preview.is_none());
    }

    /// The step an undo goes back to is kept unpacked, so undo never waits
    /// on the UI thread for it: once the warm-up lands, reading it back is
    /// the very image held, not a fresh decode.
    #[test]
    fn the_step_undo_returns_to_stays_unpacked() {
        let mut item = fixture_item(1);
        item.on_screen = true;
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]))));
        item.commit(StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]))));
        item.commit(StepKind::Adjust);
        let warm = (0..200).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            item.warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[BACK].as_ref().and_then(|w| w.image.clone())
        });
        let warm = warm.expect("the previous step warms up");
        assert_eq!(warm.get_pixel(0, 0).0, [1, 2, 3, 255]);
        let back = item.timeline.previous().and_then(|s| s.result.as_ref()).and_then(|r| r.load()).expect("previous result");
        assert!(Arc::ptr_eq(&warm, &back), "undo reads the held image");
    }

    /// An image off screen (a batch reprocessing in the background) keeps
    /// no step unpacked: its steps stay free to compress.
    #[test]
    fn an_image_off_screen_keeps_no_step_unpacked() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        item.commit(StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        item.commit(StepKind::Process);
        assert!(item.warm_steps.lock().unwrap().iter().all(Option::is_none));
    }

    /// Nor does it keep the start a late frame read back: nothing drops
    /// it off screen.
    #[test]
    fn an_image_off_screen_keeps_no_loaded_start() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
        item.commit(StepKind::Process);
        let start = |item: &BatchItem| item.timeline.current().result.as_ref().and_then(|r| r.load()).expect("the step's result");
        item.keep_loaded_start(start(&item));
        assert!(item.start_photo.is_none(), "off screen");
        item.on_screen = true;
        item.keep_loaded_start(start(&item));
        assert!(item.start_photo.is_some(), "on screen it is kept");
    }

    /// After an undo, the step a redo goes to is kept unpacked too.
    #[test]
    fn the_step_redo_goes_to_stays_unpacked() {
        let mut item = fixture_item(1);
        item.on_screen = true;
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]))));
        item.commit(StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]))));
        item.commit(StepKind::Adjust);
        let _ = item.undo_step();
        let warm = (0..200).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            item.warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[AHEAD].as_ref().and_then(|w| w.image.clone())
        });
        let warm = warm.expect("the next step warms up");
        assert_eq!(warm.get_pixel(0, 0).0, [9, 9, 9, 255]);
        let ahead = item.timeline.next().and_then(|s| s.result.as_ref()).and_then(|r| r.load()).expect("next result");
        assert!(Arc::ptr_eq(&warm, &ahead), "redo reads the held image");
    }

    /// A cut-out made from an edited photo (an object erased first) gets
    /// its effect background from that edited photo, not the original,
    /// on screen and in export alike.
    #[test]
    fn the_effect_background_follows_the_photo_the_cut_out_was_made_from() {
        use crate::gui::timeline::{Origin, Stored};
        let mut item = fixture_item(1);
        item.source_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 2, image::Rgba([250, 250, 250, 255]))));
        let erased = Stored::new(Arc::new(image::RgbaImage::from_pixel(4, 2, image::Rgba([10, 10, 10, 255]))));
        item.timeline.current_mut().origin = Some(Origin::new(Engine::Segment, Some(erased)));
        item.settings.bg_effect = prunr_core::BgEffect::DesaturatedSource;
        let cut_out = Arc::new(image::RgbaImage::from_pixel(4, 2, image::Rgba([0, 0, 0, 0])));
        assert_eq!(item.export_bg().bake(&cut_out, &ExportDepth::default()).get_pixel(0, 0).0, [10, 10, 10, 255], "export: the edited photo behind");
        assert_eq!(item.backdrop_photo().and_then(|p| p.load()).map(|p| p.get_pixel(0, 0).0), Some([10, 10, 10, 255]), "screen: the same photo");
        let before = item.backdrop_key();
        item.commit(StepKind::Stroke);
        assert_eq!(item.backdrop_key(), before, "a stroke on the cut-out keeps the photo behind");
        item.timeline = crate::gui::timeline::Timeline::new(item.timeline.current().clone());
        item.timeline.current_mut().origin = None;
        assert_ne!(item.backdrop_key(), before, "another photo behind builds another texture");
    }

    /// An effect is a picture of the photo put under whatever the result
    /// leaves transparent: lines over the whole photo keep their pixels
    /// and gain the backdrop around them.
    #[test]
    fn export_puts_the_effect_picture_under_the_result() {
        let mut item = fixture_item(1);
        let photo = image::RgbaImage::from_fn(4, 2, |x, _| image::Rgba([200, 100, x as u8 * 50, 255]));
        item.source_rgba = Some(Arc::new(photo));
        item.settings.bg_effect = prunr_core::BgEffect::DesaturatedSource;
        let line = image::Rgba([255, 0, 0, 255]);
        let result = Arc::new(image::RgbaImage::from_fn(4, 2, |x, _| if x == 0 { line } else { image::Rgba([0, 0, 0, 0]) }));
        let out = item.export_bg().bake(&result, &ExportDepth::default());
        assert_eq!(*out.get_pixel(0, 0), line, "the result's own pixels stay");
        let behind = out.get_pixel(3, 1).0;
        assert_eq!(behind[3], 255, "the transparent part gets the picture");
        assert!(behind[0] == behind[1] && behind[1] == behind[2], "a black-and-white photo: {behind:?}");

        item.settings.bg_effect = prunr_core::BgEffect::None;
        assert!(Arc::ptr_eq(&item.export_bg().bake(&result, &ExportDepth::default()), &result), "nothing behind, nothing baked");
    }

    /// An export measures an image's depth once: a later fetch for the
    /// same source gets the kept map, another source does not.
    #[test]
    fn an_export_reuses_the_map_it_measured_for_the_same_image() {
        let source = ImageSource::Bytes(Arc::new(b"not an image".to_vec()));
        let map = Arc::new(prunr_core::depth::DepthMap::from_raw(1, 2, vec![0.0, 1.0]));
        let export = ExportDepth { last: std::sync::Mutex::new(Some((source.clone(), Arc::clone(&map)))), ..ExportDepth::default() };
        let same = DepthSource { depth: None, source };
        assert!(Arc::ptr_eq(&same.get(&export, None).expect("kept"), &map));
        let other = DepthSource { depth: None, source: ImageSource::Bytes(Arc::new(b"not an image".to_vec())) };
        assert!(other.get(&export, None).is_err(), "another image is decoded, not handed the kept map");
        let own = Arc::new(prunr_core::depth::DepthMap::from_raw(1, 1, vec![0.5]));
        assert!(Arc::ptr_eq(&DepthSource { depth: Some(Arc::clone(&own)), ..other }.get(&export, None).expect("own"), &own));
    }

    /// The upscale run survives a display reset (Refine re-finishes from
    /// it) and goes on a model switch.
    #[test]
    fn the_upscale_run_outlives_a_display_reset_but_not_a_model_switch() {
        let mut item = fixture_item(1);
        let img = Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255])));
        item.upscale = Some(UpscaleCache::new(img.clone(), img.clone(), Some(Arc::new(prunr_core::upscale::Reference::new((*img).clone())))));
        item.reset_result_caches();
        assert!(item.upscale.is_some());
        item.invalidate_upscale_cache();
        assert!(item.upscale.is_none());
    }

    /// The kept colorize run belongs to the step it landed on, and only a
    /// Strength change since that step re-renders it.
    #[test]
    fn the_kept_colorize_run_belongs_to_its_step() {
        let mut item = fixture_item(1);
        let img = || Arc::new(image::RgbaImage::new(4, 4));
        let plane = Arc::new(vec![0.0; 2 * (prunr_core::colorize::COLORIZE_INPUT * prunr_core::colorize::COLORIZE_INPUT) as usize]);
        let model = crate::gui::settings::COLORIZER;
        for _ in 0..2 {
            item.start_job(Some(StepKind::Colorize), Engine::Colorize);
            item.result_rgba = Some(img());
            item.applied_recipe = Some(item.settings.current_recipe(model));
            item.land_colorize(ColorizeCache::new(Arc::clone(&plane), img()));
        }
        assert!(item.kept_colorize().is_some(), "the shown step's own run");
        assert_eq!(item.kept_run_tier(model), KeptRunTier::UpToDate, "nothing changed yet");
        item.settings.color_strength_pct = 50;
        assert_eq!(item.kept_run_tier(model), KeptRunTier::Refinish);
        let _ = item.undo_step();
        assert!(item.kept_colorize().is_none(), "an earlier colorize step: the kept run is not its own");
        let _ = item.redo_step();
        assert!(item.kept_colorize().is_some());
        item.invalidate_colorize_cache();
        assert!(item.kept_colorize().is_none());
    }

    /// Process under an upscaler: nothing for the step's own upscale as
    /// asked, a re-finish for a Refine change, a run for anything else.
    #[test]
    fn an_upscale_needs_a_run_only_when_more_than_refine_changed() {
        let mut item = fixture_item(1);
        let model = prunr_core::ModelKind::RealEsrganX4Plus;
        assert_eq!(item.kept_run_tier(model), KeptRunTier::Full, "never upscaled");
        let img = || Arc::new(image::RgbaImage::new(4, 4));
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.applied_recipe = Some(item.settings.current_recipe(model));
        item.land_upscale(UpscaleCache::new(img(), img(), None));
        assert_eq!(item.kept_run_tier(model), KeptRunTier::UpToDate);
        item.settings.sharpen = 0.4;
        assert_eq!(item.kept_run_tier(model), KeptRunTier::Refinish);
        item.settings.output_scale = prunr_core::OutputScale::X2;
        assert_eq!(item.kept_run_tier(model), KeptRunTier::Full, "a new scale");
        item.settings = Default::default();
        item.start_job(Some(StepKind::Colorize), Engine::Colorize);
        item.result_rgba = Some(img());
        item.finish_job();
        assert_eq!(item.kept_run_tier(model), KeptRunTier::Full, "another model's step on top: upscale that");
    }

    /// Process again on an upscale enlarges what is shown, also after an
    /// undo back to an earlier upscale; a Scale change redoes the shown
    /// upscale from where it started.
    #[test]
    fn the_next_upscale_starts_on_the_upscale_shown_unless_settings_changed() {
        let mut item = fixture_item(1);
        let model = prunr_core::ModelKind::RealEsrganX4Plus;
        let upscale = |item: &mut BatchItem, side: u32| {
            let start = item.next_upscale_start(model);
            item.start_job_at(Some(StepKind::Upscale), Engine::Upscale, start);
            item.applied_recipe = Some(item.settings.current_recipe(model));
            let out = Arc::new(image::RgbaImage::new(side, side));
            item.result_rgba = Some(Arc::clone(&out));
            item.land_upscale(UpscaleCache::new(Arc::clone(&out), out, None));
        };
        assert!(item.next_upscale_start(model).is_none(), "the first from the photo");
        upscale(&mut item, 8);
        assert_eq!(item.next_upscale_dims(model), (8, 8), "Process again enlarges the upscale");
        upscale(&mut item, 16);
        assert_eq!(item.next_upscale_dims(model), (16, 16));
        let _ = item.undo_step();
        assert_eq!(item.next_upscale_dims(model), (8, 8), "after undo: what is shown, with its run no longer kept");
        item.settings.output_scale = prunr_core::OutputScale::X2;
        assert!(item.next_upscale_start(model).is_none(), "a new Scale redoes the shown upscale from the photo");
    }

    /// The kept upscale run belongs to the step it landed on: undo to an
    /// earlier upscale and Refine has no run to re-finish (it would be a
    /// later run's output); redo brings it back.
    #[test]
    fn the_kept_upscale_run_belongs_to_its_step() {
        let mut item = fixture_item(1);
        let img = || Arc::new(image::RgbaImage::new(4, 4));
        for _ in 0..2 {
            item.start_job(Some(StepKind::Upscale), Engine::Upscale);
            item.result_rgba = Some(img());
            item.land_upscale(UpscaleCache::new(img(), img(), None));
        }
        assert!(item.kept_upscale().is_some(), "the shown step's own run");
        let _ = item.undo_step();
        assert!(item.kept_upscale().is_none(), "an earlier upscale step: the kept run is not its own");
        let _ = item.redo_step();
        assert!(item.kept_upscale().is_some());
        item.commit(StepKind::Adjust);
        assert!(item.kept_upscale().is_some(), "a Refine knob step keeps its upscale's run");
    }

    /// Restore faces works on the shown upscale's output; back on the
    /// photo once that run is no longer the shown step's.
    #[test]
    fn faces_are_found_in_the_shown_upscale() {
        let mut item = fixture_item(1);
        let source = Arc::new(image::RgbaImage::new(4, 4));
        item.source_rgba = Some(Arc::clone(&source));
        assert!(item.face_photo(FaceStage::Photo).is_some_and(|p| p.same(&FacePhoto::Original(Arc::clone(&source)))));
        let raw = Arc::new(image::RgbaImage::new(16, 16));
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.result_rgba = Some(Arc::clone(&raw));
        item.land_upscale(UpscaleCache::new(Arc::clone(&raw), Arc::clone(&source), None));
        let kept = item.kept_upscale().map(|run| Arc::clone(&run.raw)).expect("the run landed");
        assert!(item.face_photo(FaceStage::Upscale).is_some_and(|p| p.same(&FacePhoto::Upscaled(kept))));
        let _ = item.undo_step();
        assert!(item.face_photo(FaceStage::Upscale).is_none(), "the earlier step has no upscale of its own");
        let _ = raw;
    }

    /// Pin seed keeps the fill on screen: each fill's seed stays with its
    /// step, so after an undo it is the earlier fill's.
    #[test]
    fn a_fills_seed_belongs_to_its_step() {
        let mut item = fixture_item(1);
        for (seed, v) in [(7, 1), (8, 2)] {
            item.start_erase(Some(seed));
            item.result_rgba = Some(px(v));
            item.finish_job();
        }
        assert_eq!(item.shown_seed(), Some(8));
        let _ = item.undo_step();
        assert_eq!(item.shown_seed(), Some(7));
        let _ = item.undo_step();
        assert_eq!(item.shown_seed(), None, "the original has no fill");
        let _ = item.redo_step();
        assert_eq!(item.shown_seed(), Some(7));
    }

    #[test]
    fn reset_result_caches_clears_display_fields_only() {
        // reset_result_caches owns display/texture cleanup. It must NOT
        // touch cached_tensor — that would kill live preview after any
        // Tier 2 rerun (the tier-2 worker doesn't return a fresh tensor).
        // The selection survives too: it must outlast a Process click.
        let mut item = fixture_item(1);
        item.thumb.asked();
        item.source_tex_pending = true;
        item.result_tex_pending = true;
        item.decode_pending = true;
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        item.selection_hash = Some(mask.content_hash());
        item.selection_mask = Some(Arc::new(mask));
        item.reset_result_caches();
        assert!(item.result_texture.is_none());
        assert!(item.thumb.texture().is_none());
        assert!(!item.thumb.is_pending());
        assert!(!item.source_tex_pending);
        assert!(!item.result_tex_pending);
        assert!(item.decode_pending, "the photo's decode is no result cache: it is still under way");
        assert!(item.selection_mask.is_some() && item.selection_hash.is_some(), "the selection stays");
    }

    /// Each kind of settings change asks for the run it needs: none for a
    /// backdrop or Refine change, the mask side from the cached tensor,
    /// the lines onto it, or everything — and everything when there is no
    /// tensor to start from.
    #[test]
    fn a_process_run_is_as_small_as_the_change_allows() {
        use crate::gui::worker::{CompressedTensor, RawSegTensor};
        use prunr_core::{LineMode, ModelKind::Silueta};
        fn processed(with_tensor: bool, before: fn(&mut ItemSettings)) -> BatchItem {
            let mut item = fixture_item(1);
            before(&mut item.settings);
            item.applied_recipe = Some(item.settings.current_recipe(Silueta));
            if with_tensor {
                item.set_cached_tensor(CompressedTensor::from_raw(RawSegTensor { data: vec![0.5; 16], height: 4, width: 4, model: Silueta }));
            }
            item
        }
        let tier = |with_tensor: bool, before: fn(&mut ItemSettings), change: fn(&mut ItemSettings)| {
            let mut item = processed(with_tensor, before);
            change(&mut item.settings);
            item.run_tier(Silueta)
        };
        let lines_on = |s: &mut ItemSettings| s.line_mode = LineMode::SubjectOutline;

        assert_eq!(fixture_item(1).run_tier(Silueta), RunTier::Full, "never processed");
        assert_eq!(tier(true, |_| {}, |_| {}), RunTier::UpToDate);
        assert_eq!(tier(true, |_| {}, |s| s.sharpen = 0.5), RunTier::UpToDate, "Refine re-finishes the kept run");
        assert_eq!(tier(true, |_| {}, |s| s.feather = 3.0), RunTier::Mask, "a mask knob reruns from the tensor");
        assert_eq!(tier(false, |_| {}, |s| s.feather = 3.0), RunTier::Full, "without a tensor, everything");
        assert_eq!(tier(true, |_| {}, lines_on), RunTier::AddEdge, "lines onto the cut-out");
        assert_eq!(tier(false, |_| {}, lines_on), RunTier::Full);
        assert_eq!(tier(true, lines_on, |s| s.feather = 3.0), RunTier::AddEdge, "a mask change under lines composes them again");
        assert_eq!(tier(true, lines_on, |s| s.edge_thickness = 3), RunTier::AddEdge, "an edge change traces the cached cut-out again");
        assert_eq!(tier(false, lines_on, |s| s.edge_thickness = 3), RunTier::Full, "an edge change without a tensor runs whole");
        let whole_photo = |s: &mut ItemSettings| s.line_mode = LineMode::EdgesOnly;
        assert_eq!(tier(true, whole_photo, |s| s.edge_thickness = 3), RunTier::Full, "lines over the whole photo trace no cut-out");
        assert_eq!(tier(true, |_| {}, |s| s.line_mode = LineMode::EdgesOnly), RunTier::Full, "lines over the whole photo");

        let mut item = processed(true, |_| {});
        item.settings.bg = Some([1, 2, 3, 255]);
        assert_eq!(item.run_tier(Silueta), RunTier::UpToDate, "a backdrop change renders at display time");
        let composite = item.settings.current_recipe(Silueta).composite;
        assert_eq!(item.applied_recipe.as_ref().map(|r| &r.composite), Some(&composite), "and the applied recipe follows it");
    }

    /// A result keeps the texture on screen until its own is uploaded,
    /// then reveals over it; a waiting image is done with it.
    #[test]
    fn a_result_keeps_the_old_texture_until_its_own_lands() {
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let mut item = fixture_item(1);
        let old = tex("old");
        item.result_texture = Some(old.clone().into());
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Pending);
        item.show_result(Arc::new(image::RgbaImage::new(2, 2)));
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(old.id()), "the old texture stays");
        assert!(item.result_tex_pending && item.thumb.is_pending(), "and a new one and thumbnail are on their way");
        assert_eq!(item.status(), BatchStatus::Done);
        let new = tex("new");
        let target = item.request_result_texture(false);
        item.land_texture(new.clone().into(), target);
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(new.id()));
        assert!(!item.result_tex_pending);
        assert!(matches!(item.reveal, Some(Reveal::Running(..))), "the new result reveals over the old");
    }

    /// Result textures are made on threads and finish in any order: one
    /// asked for before the result changed never lands over the newer,
    /// nor over the original an undo went back to.
    #[test]
    fn a_result_texture_lands_only_for_the_result_held() {
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let mut item = fixture_item(1);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        item.commit(StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
        item.commit(StepKind::Adjust);

        let dragged = item.request_result_texture(true);
        assert!(item.undo_step().is_ok());
        let restored = item.request_result_texture(false);
        let shown = tex("restored");
        item.land_texture(shown.clone().into(), restored);
        item.land_texture(tex("dragged").into(), dragged);
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(shown.id()), "the undone render lands no more");

        let stale = item.request_result_texture(false);
        assert!(item.undo_step().is_ok());
        item.land_texture(tex("stale").into(), stale);
        assert!(item.result_texture.is_none(), "the original has no result to show");
    }

    /// A drag's quick frame is what is on screen until the state it was
    /// made for goes: an undo reveals from it and takes it down, as do
    /// leaving the screen and a run's result.
    #[test]
    fn a_quick_frame_goes_with_the_state_it_was_made_for() {
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let frame = tex("frame");
        let put_up = |item: &mut BatchItem| {
            item.quick.expect(4);
            item.quick.land(frame.clone().into(), 4);
            assert!(item.quick.texture().is_some());
        };
        let mut item = fixture_item(1);
        item.result_texture = Some(tex("result").into());
        item.commit(StepKind::Adjust);
        put_up(&mut item);
        assert!(item.undo_step().is_ok());
        assert!(item.quick.texture().is_none(), "an undo takes it down");
        let revealed_from = match &item.reveal {
            Some(Reveal::Held(t) | Reveal::Running(t, ..)) => Some(t.id()),
            None => None,
        };
        assert_eq!(revealed_from, Some(frame.id()), "and reveals from it");

        put_up(&mut item);
        item.set_on_screen(false);
        assert!(item.quick.texture().is_none(), "off screen");

        put_up(&mut item);
        item.show_result(Arc::new(image::RgbaImage::new(2, 2)));
        assert!(item.quick.texture().is_none(), "a run's result replaces it");
    }

    /// The decoded source lands on the image with what was derived from
    /// the old bytes dropped; a failed decode marks the image.
    #[test]
    fn a_decoded_source_replaces_what_was_derived_from_the_old_one() {
        let mut item = fixture_item(1);
        item.dimensions = (2, 2);
        item.decode_pending = true;
        assert!(item.land_decode(Ok(Arc::new(image::RgbaImage::new(2, 2)))));
        assert!(item.source_rgba.is_some() && !item.decode_pending);
        assert!(!item.land_decode(Err("bad file".into())));
        assert_eq!(item.status(), BatchStatus::Error("bad file".into()));
    }

    /// A selection overlay built for a mask the image has since replaced
    /// is dropped; the one for the shown mask is kept.
    #[test]
    fn a_selection_overlay_lands_only_for_the_mask_shown() {
        use crate::gui::background_io::{SelectionImage, SelectionStyle, SelectionTextureResult, TexLanding};
        use crate::gui::image_texture::PreparedImage;
        let ctx = egui::Context::default();
        let image = || PreparedImage::from_color_image(egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]));
        let style = SelectionStyle::with_fill(0.5);
        let full = |hash: u64| SelectionTextureResult {
            item_id: 1,
            image: SelectionImage::Full { image: image() },
            key: (hash, style),
            shown: make_mask(hash, 2, 2),
        };
        // An upload as the frame makes it, installed once its tiles are up.
        let land = |item: &mut BatchItem, result| {
            let mut queued = None;
            item.land_selection_texture(result, |image, landing| queued = Some((image, landing)));
            queued.map(|(image, landing)| {
                let mut uploads = crate::gui::image_texture::Uploads::default();
                uploads.push("selection".into(), image, landing);
                let (texture, landing) = uploads.pump(&ctx, |_| true).landed.pop().expect("uploaded");
                let TexLanding::Selection { key, shown, .. } = landing else { panic!("a selection") };
                item.install_selection_texture(texture, key, shown);
            })
        };
        let mut item = fixture_item(1);
        item.selection_mask = Some(make_mask(7, 2, 2));
        item.selection_hash = Some(7);
        item.selection_tex_pending = Some((3, style));
        assert!(land(&mut item, full(3)).is_none(), "a stale build is not uploaded");
        assert!(item.selection_texture.is_none() && item.selection_tex_pending.is_none(), "and frees the slot");
        item.selection_tex_pending = Some((7, style));
        assert!(land(&mut item, full(7)).is_some(), "the shown mask's overlay is uploaded");
        assert!(item.selection_tex_pending.is_none(), "pending until installed");
        assert_eq!(item.selection_texture.as_ref().map(|t| t.key), Some((7, style)), "and installed");
    }

    /// A lens-blur backdrop keys on where it focuses and on the cut-out
    /// its subject is filled from: the same result keeps the key, a new
    /// one changes it, and the cache holds no result alive.
    #[test]
    fn a_lens_backdrop_keys_on_its_focus_and_its_cut_out() {
        use prunr_core::depth::DepthMap;
        let mut item = fixture_item(1);
        item.settings.bg_effect = prunr_core::BgEffect::LensBlur { amount_pct: 50 };
        item.depth = Some(Arc::new(DepthMap::from_raw(2, 2, vec![0.1, 0.1, 0.9, 0.9])));
        let cutout = |rows: [u8; 2]| {
            let mut img = image::RgbaImage::new(2, 2);
            for (y, a) in rows.into_iter().enumerate() {
                for x in 0..2 { img.put_pixel(x, y as u32, image::Rgba([0, 0, 0, a])); }
            }
            Arc::new(img)
        };
        let refresh = |item: &mut BatchItem| {
            item.refresh_lens_focus();
            item.refresh_backdrop_cut();
        };
        item.result_rgba = Some(cutout([255, 0]));
        refresh(&mut item);
        let near = item.backdrop_key();
        refresh(&mut item);
        assert_eq!(item.backdrop_key(), near, "the same result keeps the key");
        item.result_rgba = Some(cutout([255, 0]));
        refresh(&mut item);
        let again = item.backdrop_key();
        assert_ne!(again, near, "a new cut-out fills the subject anew");
        let weak = item.result_rgba.as_ref().map(Arc::downgrade).expect("a result");
        item.result_rgba = Some(cutout([0, 255]));
        assert!(weak.upgrade().is_none(), "the cache does not keep the old result alive");
        refresh(&mut item);
        assert_ne!(item.backdrop_key(), again, "the subject at another depth is another focus");
    }

    /// Blurred photo is made again for each cut-out; Black & white photo
    /// does not use it and keeps its backdrop.
    #[test]
    fn only_a_blurred_backdrop_follows_the_cut_out() {
        for (effect, follows) in [
            (prunr_core::BgEffect::BlurredSource { radius: 12 }, true),
            (prunr_core::BgEffect::DesaturatedSource, false),
        ] {
            let mut item = fixture_item(1);
            item.settings.bg_effect = effect;
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
            item.refresh_backdrop_cut();
            let before = item.backdrop_key();
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(2, 2)));
            item.refresh_backdrop_cut();
            assert_eq!(item.backdrop_key() != before, follows, "{effect:?}");
        }
    }

    /// The lens backdrop's focus is measured with the faces the depth map
    /// knows: the same map with a face found focuses (and keys) anew.
    #[test]
    fn a_lens_backdrop_focuses_on_the_faces_its_depth_map_knows() {
        use prunr_core::depth::DepthMap;
        let mut item = fixture_item(1);
        item.settings.bg_effect = prunr_core::BgEffect::LensBlur { amount_pct: 50 };
        let raw: Vec<f32> = (0..100).map(|y| y as f32).collect();
        item.depth = Some(Arc::new(DepthMap::from_raw(1, 100, raw.clone())));
        item.refresh_lens_focus();
        let (faceless, key) = (item.lens_focus.as_ref().map(|f| f.value), item.backdrop_key());
        let face = prunr_core::face::Face {
            score: 0.9,
            bbox: [0.0, 20.0, 100.0, 40.0],
            landmarks: [[30.0, 25.0], [70.0, 25.0], [50.0, 30.0], [35.0, 36.0], [65.0, 36.0]],
        };
        let faced = DepthMap::from_raw(1, 100, raw).with_faces(&[prunr_core::face::FacePlace::of(&face, [100, 100])]);
        item.depth = Some(Arc::new(faced));
        item.refresh_lens_focus();
        let focus = item.lens_focus.as_ref().map(|f| f.value);
        assert!(focus.is_some_and(|f| (f - 0.3).abs() < 0.02) && focus != faceless, "the face, not the nearest tenth: {focus:?} {faceless:?}");
        assert_ne!(item.backdrop_key(), key);
    }

    /// A depth run takes the faces a Faces search found in the original
    /// photo instead of looking again; none from another photo, or from
    /// a search that failed.
    #[test]
    fn a_depth_run_reuses_the_faces_found_in_the_original() {
        let mut item = fixture_item(1);
        let source = Arc::new(image::RgbaImage::new(100, 100));
        item.source_rgba = Some(Arc::clone(&source));
        let face = prunr_core::face::Face {
            score: 0.9,
            bbox: [10.0, 10.0, 60.0, 70.0],
            landmarks: [[25.0, 30.0], [45.0, 30.0], [35.0, 40.0], [27.0, 52.0], [43.0, 52.0]],
        };
        let place = prunr_core::face::FacePlace::of(&face, [100, 100]);
        assert_eq!(item.faces_found_in_original(), None, "nothing searched");

        item.begin_face_search(FacePhoto::Original(Arc::clone(&source)), prunr_models::ModelId::FaceGfpgan);
        assert_eq!(item.faces_found_in_original(), None, "searching, nothing found yet");
        if let Some(pending) = item.faces_pending.as_mut() {
            pending.found = Some((vec![place].into(), std::time::Instant::now()));
        }
        assert_eq!(item.faces_found_in_original(), Some(vec![place]), "found, still redrawing");

        let restored = Arc::new(vec![prunr_core::face::RestoredFace::unchanged_for_test(place)]);
        let found = |photo, failed| FoundFaces { photo, found_at: std::time::Instant::now(), model: prunr_models::ModelId::FaceGfpgan, faces: Arc::clone(&restored), failed, stopped: false };
        item.faces_pending = None;
        item.found_faces = Some(found(FacePhoto::Original(Arc::clone(&source)), false));
        assert_eq!(item.faces_found_in_original(), Some(vec![place]));
        item.found_faces = Some(found(FacePhoto::Original(Arc::clone(&source)), true));
        assert_eq!(item.faces_found_in_original(), None, "a failed search");
        item.found_faces = Some(found(FacePhoto::Upscaled(crate::gui::timeline::Stored::new(Arc::new(image::RgbaImage::new(200, 200)))), false));
        assert_eq!(item.faces_found_in_original(), None, "another photo");
    }

    /// The size the canvas shows reads a parked result's stored size, not
    /// the photo's: an upscale off screen still frames at its output size.
    #[test]
    fn a_parked_result_keeps_its_shown_size() {
        let mut item = fixture_item(1);
        item.dimensions = (2, 2);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(StepKind::Upscale);
        assert!(item.park_result());
        assert!(item.result_rgba.is_none());
        assert_eq!(item.shown_dims(), (8, 8));
        assert_eq!(item.frame_size(), egui::vec2(8.0, 8.0));
    }

    /// The first decode of a photo keeps what was made from it before:
    /// its depth map and selection; a different photo landing drops them.
    /// A photo that decodes at another size than it was opened at is not
    /// the file that was opened, even at the same size: it is refused with
    /// why, not shown in its place. A file that is gone says so.
    #[test]
    fn a_changed_or_missing_file_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.png");
        std::fs::write(&path, b"first").unwrap();
        let source = ImageSource::file(path.clone());
        assert_eq!(&**source.load_bytes().unwrap(), b"first");
        std::fs::write(&path, b"other").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(10);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later).unwrap();
        let changed = source.load_bytes().map_err(super::source_unreadable).unwrap_err();
        assert!(changed.contains("changed since it was opened"), "{changed}");
        std::fs::remove_file(&path).unwrap();
        let gone = source.load_bytes().map_err(super::source_unreadable).unwrap_err();
        assert!(gone.contains("moved or deleted"), "{gone}");
    }

    /// An undo onto a step whose saved selection was removed from the disk
    /// cache does not move, and says why, instead of showing that step
    /// with its selection gone.
    #[test]
    fn an_undo_onto_a_lost_step_does_not_move() {
        let mut item = fixture_item(1);
        let plane = |v: i8| Arc::new(prunr_core::selection::MaskArtifact::from_cells(2, 2, vec![v; 4]));
        item.commit_selection_mask(plane(1), StepKind::Stroke);
        let earlier = item.timeline.current().selection.clone().expect("a selection step");
        item.commit_selection_mask(plane(2), StepKind::Stroke);
        earlier.compress_job().expect("only the timeline holds it")();
        assert!(earlier.demote_to_disk());
        std::fs::remove_file(earlier.disk_path().expect("on disk")).unwrap();
        let shown = item.selection_mask.clone();
        assert_eq!(item.undo_step(), Err(StepRefused::Lost));
        assert!(Arc::ptr_eq(item.selection_mask.as_ref().unwrap(), shown.as_ref().unwrap()), "nothing changed");
        assert_eq!(item.redo_step(), Err(StepRefused::Nothing), "still on the newest step");
    }

    /// A decode is of the image's own file, which never changes under it
    /// (a changed file is refused): what was made from the photo (depth,
    /// selection, Magic Brush embedding) stays through every decode.
    #[test]
    fn a_decode_keeps_what_was_made_from_the_photo() {
        let mut item = fixture_item(1);
        item.dimensions = (1, 1);
        item.source_rgba = None;
        item.depth = Some(Arc::new(prunr_core::depth::DepthMap::from_raw(1, 1, vec![0.5])));
        item.selection_mask = Some(Arc::new(prunr_core::selection::MaskArtifact::from_cells(1, 1, vec![1])));
        assert!(item.land_decode(Ok(Arc::new(image::RgbaImage::new(1, 1)))));
        assert!(item.depth.is_some() && item.selection_mask.is_some(), "the first decode");
        assert!(item.land_decode(Ok(Arc::new(image::RgbaImage::new(1, 1)))));
        assert!(item.depth.is_some() && item.selection_mask.is_some(), "a second decode of the same photo");
    }

    /// A run's work is handles: its file is read where the run is sent,
    /// never on the UI thread, so a missing file does not fail it here;
    /// and building it starts nothing (the image runs once it is sent).
    #[test]
    fn a_runs_work_leaves_its_file_unread() {
        let mut item = fixture_item(1);
        item.source = ImageSource::file(PathBuf::from("/nonexistent/photo.png"));
        item.start_job(Some(StepKind::Process), Engine::Segment);
        let work = item.full_run_work();
        assert_eq!(work.id, 1);
        assert!(item.awaits_cut() && !item.is_cutting() && item.failure.is_none(), "{:?}", item.status());
    }

    fn fixture_recipe() -> prunr_core::ProcessingRecipe {
        ItemSettings::default().current_recipe(prunr_core::ModelKind::Silueta)
    }

    fn fixture_process_result(provider: &str) -> prunr_core::ProcessResult {
        prunr_core::ProcessResult {
            rgba_image: image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 255])),
            active_provider: provider.to_string(),
        }
    }

    /// A landed result is Done with its recipe and keeps the source (live
    /// preview re-cuts from `source_rgba` on any item, shown or not). A
    /// Tier 2 rerun reports no provider, so the backend label shown in
    /// the UI is not overwritten.
    #[test]
    fn apply_tier_result_success_sets_done_and_returns_provider() {
        for (provider, want) in [("CUDA", Some("CUDA")), ("", None)] {
            let mut item = fixture_item(1);
            crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Processing);
            item.source_rgba = Some(Arc::new(image::RgbaImage::new(1, 1)));
            let got = item.apply_tier_result(Ok(fixture_process_result(provider)), None, None, fixture_recipe());
            item.finish_job();
            assert_eq!(got.as_deref(), want, "{provider:?}");
            assert_eq!(item.status(), BatchStatus::Done);
            assert!(item.result_rgba.is_some() && item.applied_recipe.is_some());
            assert!(item.source_rgba.is_some(), "source_rgba must stay for live preview");
        }
    }
    #[test]
    fn apply_tier_result_error_clears_recipe_and_tensors_for_fresh_retry() {
        let mut item = fixture_item(1);
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Processing);
        item.applied_recipe = Some(fixture_recipe());

        let provider = item.apply_tier_result(
            Err("boom".to_string()),
            None,
            None,
            fixture_recipe(),
        );

        assert!(provider.is_none());
        assert!(matches!(item.status(), BatchStatus::Error(ref e) if e == "boom"));
        assert!(item.cached_tensor.is_none());
        assert!(item.cached_edge_tensors.is_none());
        assert!(item.applied_recipe.is_none(),
            "applied_recipe must be cleared so resolve_tier picks FullPipeline on retry");
    }

    #[test]
    fn image_source_load_bytes_for_bytes_variant_returns_same_arc() {
        let bytes = Arc::new(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let source = ImageSource::Bytes(bytes.clone());
        let loaded = source.load_bytes().expect("Bytes variant must succeed");
        assert!(Arc::ptr_eq(&loaded, &bytes), "Bytes load must return the same Arc, no realloc");
    }

    #[test]
    fn an_in_memory_image_knows_its_size() {
        let item = BatchItem::new(1, "x".into(), ImageSource::Bytes(Arc::new(vec![0u8; 1234])), (1, 1), ItemSettings::default(), String::new());
        assert_eq!(item.source_len, 1234);
    }

    /// Build a distinct MaskArtifact for testing; the tag sets the first cell.
    fn make_mask(tag: u64, w: u32, h: u32) -> Arc<prunr_core::selection::MaskArtifact> {
        let mut data = vec![0i8; (w * h) as usize];
        data[0] = tag as i8;
        Arc::new(prunr_core::selection::MaskArtifact::from_cells(w, h, data))
    }

    fn px(v: u8) -> Arc<image::RgbaImage> {
        Arc::new(image::RgbaImage::from_pixel(2, 2, image::Rgba([v, v, v, 255])))
    }

    /// What the image shows, on screen or parked off it.
    fn shown(item: &BatchItem) -> Option<u8> {
        item.result_pixels().and_then(|p| p.load()).map(|r| r.get_pixel(0, 0)[0])
    }

    #[test]
    fn identical_commit_is_a_no_op_without_an_undo_step() {
        let mut item = fixture_item(1);
        assert!(item.commit_selection_mask(make_mask(1, 4, 4), StepKind::Stroke));
        assert!(!item.commit_selection_mask(make_mask(1, 4, 4), StepKind::Stroke));
        assert_eq!(item.timeline.undo_steps(), 1);
    }

    /// Undo walks every kind of step on one list and shows the stored
    /// state: the image, the selection with its hash, the settings.
    #[test]
    fn undo_and_redo_restore_the_whole_state_of_each_step() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(px(1));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(StepKind::Process);
        item.commit_selection_mask(make_mask(5, 4, 4), StepKind::Stroke);
        item.result_rgba = Some(px(2)); // the stroke's re-cut landed
        item.amend();
        item.settings.gamma = 1.7;
        item.commit(StepKind::Adjust);

        assert_eq!(item.undo_step(), Ok(StepKind::Adjust));
        assert_eq!((shown(&item), item.settings.gamma), (Some(2), ItemSettings::default().gamma));
        assert_eq!(item.undo_step(), Ok(StepKind::Stroke));
        assert_eq!(shown(&item), Some(1), "the image before the stroke, stored, not re-cut");
        assert!(item.selection_mask.is_none() && item.selection_hash.is_none());
        assert_eq!(item.undo_step(), Ok(StepKind::Process));
        assert_eq!((shown(&item), &item.status()), (None, &BatchStatus::Pending));
        assert_eq!(item.undo_step(), Err(StepRefused::Nothing));

        assert_eq!(item.redo_step(), Ok(StepKind::Process));
        assert_eq!(item.redo_step(), Ok(StepKind::Stroke));
        assert_eq!(item.selection_hash, Some(make_mask(5, 4, 4).content_hash()));
        assert_eq!(shown(&item), Some(2));
        assert_eq!(item.redo_step(), Ok(StepKind::Adjust));
        assert_eq!(item.settings.gamma, 1.7);
    }

    /// A run lands only on the state it started from; landing records its
    /// step, or fills in the shown one.
    #[test]
    fn a_run_lands_only_on_the_state_it_started_from() {
        let mut item = fixture_item(1);
        item.start_job(Some(StepKind::Process), Engine::Segment);
        assert!(item.job_is_current());
        item.result_rgba = Some(px(1));
        item.finish_job();
        assert_eq!(item.timeline.undo_steps(), 1);

        item.start_job(None, Engine::Segment);
        item.result_rgba = Some(px(2));
        item.finish_job();
        assert_eq!(item.timeline.undo_steps(), 1, "a re-run fills in the shown step");

        item.start_job(Some(StepKind::Process), Engine::Segment);
        let _ = item.undo_step();
        assert!(!item.job_is_current(), "an undo since the start makes the result stale");
    }

    /// An erase, upscale or colorize still lands after a step that keeps
    /// the pixels it started from: a knob, a preset. New pixels or an
    /// undo leave it stale, and a cut keeps the strict rule.
    #[test]
    fn a_step_that_keeps_the_pixels_keeps_a_long_run() {
        for (engine, step) in [(Engine::Erase, StepKind::Erase), (Engine::Upscale, StepKind::Upscale), (Engine::Colorize, StepKind::Colorize)] {
            let mut item = fixture_item(1);
            item.result_rgba = Some(px(1));
            item.commit(StepKind::Process);
            item.start_job(Some(step), engine);
            item.settings.gamma = 1.4;
            item.commit(StepKind::Adjust);
            assert!(item.job_is_current(), "{engine:?}: a knob change");
            item.commit(StepKind::Preset);
            assert!(item.job_is_current(), "{engine:?}: a preset");
            item.result_rgba = Some(px(2));
            item.commit(StepKind::Edit);
            assert!(!item.job_is_current(), "{engine:?}: new pixels");
        }

        let mut item = fixture_item(1);
        item.commit(StepKind::Preset);
        item.start_job(Some(StepKind::Erase), Engine::Erase);
        item.commit(StepKind::Adjust);
        let _ = item.undo_step();
        assert!(item.job_is_current(), "an undo back to where the run started");
        let _ = item.undo_step();
        assert!(!item.job_is_current(), "an undo past it");

        let mut item = fixture_item(1);
        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.commit(StepKind::Adjust);
        assert!(!item.job_is_current(), "a cut");
    }

    /// Lands a run of `engine` that produced `v`, recording `step`.
    fn land(item: &mut BatchItem, engine: Engine, step: Option<StepKind>, v: u8) -> Option<u8> {
        item.start_job(step, engine);
        let base = item.job_input().map(|b| b.get_pixel(0, 0).0[0]);
        item.result_rgba = Some(px(v));
        item.finish_job();
        base
    }

    /// Every run starts from the image a non-destructive editor would use:
    /// redoing the shown step starts from that step's own starting image,
    /// a different engine builds on the shown result, the eraser always
    /// builds on it, and a hand edit is never undone by a rerun.
    #[test]
    fn a_run_starts_from_the_image_before_the_step_it_redoes() {
        let mut item = fixture_item(1);
        assert_eq!(land(&mut item, Engine::Segment, Some(StepKind::Process), 1), None, "first cut-out: the original");
        assert_eq!(land(&mut item, Engine::Segment, None, 2), None, "a re-cut redoes it from the original, never the cut-out");

        assert_eq!(land(&mut item, Engine::Upscale, Some(StepKind::Upscale), 3), Some(2), "upscaling builds on the cut-out");
        assert_eq!(land(&mut item, Engine::Upscale, None, 4), Some(2), "upscaling again redoes it from the cut-out, not 16x");

        assert_eq!(land(&mut item, Engine::Erase, Some(StepKind::Erase), 5), Some(4), "an erase builds on the upscale");
        assert_eq!(land(&mut item, Engine::Erase, Some(StepKind::Erase), 6), Some(5), "each erase adds to the last");

        assert_eq!(land(&mut item, Engine::Segment, Some(StepKind::Process), 7), Some(6), "a cut after erasing keeps the erase");
        assert_eq!(land(&mut item, Engine::Segment, None, 8), Some(6), "and a re-cut starts from the erased image");

        item.result_rgba = Some(px(9));
        item.commit(StepKind::Edit);
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(9), "a hand edit is kept by the next run");

        let _ = item.undo_step();
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(6), "undo brings back the step's own start");
        assert!(item.run_start(Engine::Segment).is_some());
        assert_eq!(item.run_input_dims(Engine::Upscale), (2, 2));
    }

    /// Cut out, undo, erase, then a live render with no model (Faces, an
    /// effect) lands on the erase: the segmentation kept from the undone
    /// cut must not take the erase for its cut-out, or the next render
    /// re-cuts the background away.
    #[test]
    fn a_render_on_an_erase_after_an_undone_cut_does_not_bring_the_cut_back() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(px(1));
        item.commit(StepKind::Process);
        crate::gui::tests::fixtures::cut_out(&mut item, prunr_core::ModelKind::Silueta);
        assert!(item.shows_cutout());
        let _ = item.undo_step();
        assert!(!item.shows_cutout(), "undone: the photo shows");

        item.result_rgba = Some(px(3));
        item.commit(StepKind::Erase);
        item.record_origin(Engine::Erase, None, None);
        assert!(!item.shows_cutout());

        item.record_photo_pass(None, None);
        assert!(!item.shows_cutout(), "a render that cut nothing is no cut-out");
        assert!(matches!(item.render_seg(), (None, None)), "so no render re-cuts it");
    }

    /// A render on an upscale or colorize step keeps the run's kept run,
    /// so its Refine or Strength stays on it, and the next render redoes
    /// from the run's result.
    #[test]
    fn a_render_on_a_kept_run_keeps_it() {
        for engine in [Engine::Upscale, Engine::Colorize] {
            let mut item = fixture_item(1);
            item.result_rgba = Some(px(1));
            item.commit(StepKind::Process);
            item.record_origin(engine, None, None);
            item.finish_job_keeping(5);
            let render_start = item.run_start(Engine::Segment).expect("a render starts from the run's result");
            item.result_rgba = Some(px(2));
            item.amend();
            item.record_photo_pass(Some(render_start.clone()), None);
            assert_eq!(item.shown_kept_run(), Some(5), "{engine:?}: the kept run stays");
            let next = item.run_start(Engine::Segment).expect("the next render's start");
            assert!(Arc::ptr_eq(&next, &render_start), "{engine:?}: the next render redoes from the run's result");
        }
    }

    /// A render that cut nothing on a model's cut keeps that cut: a
    /// stroke still re-cuts it from the kept segmentation.
    #[test]
    fn a_render_that_cut_nothing_on_a_cut_keeps_the_cut() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(px(1));
        item.commit(StepKind::Process);
        crate::gui::tests::fixtures::cut_out(&mut item, prunr_core::ModelKind::Silueta);
        item.record_photo_pass(None, None);
        assert!(item.shows_cutout());
    }

    /// A chained cut's start, built where the run reads its file, becomes
    /// the step's start: a knob's re-cut of that step redoes it from there,
    /// and the next model's Process builds on its cut-out in turn.
    #[test]
    fn a_chained_cut_starts_from_the_cutout_it_was_built_from() {
        use prunr_core::ModelKind::{BiRefNetLite, Silueta};
        let (silueta, birefnet) = (Cutter::Model(Silueta), Cutter::Model(BiRefNetLite));
        let mut item = fixture_item(1);
        item.land_cut_for_test(silueta, px(1));
        assert!(!item.cuts_anothers_cutout(silueta));
        assert!(item.cuts_anothers_cutout(birefnet) && item.cuts_anothers_cutout(Cutter::NoModel));

        item.start_cut(Some(StepKind::Process), birefnet);
        assert!(matches!(item.job_run_image(), Some(crate::gui::worker::RunImage::Kept { start: None, .. })), "colours from the photo");
        item.chain_start_built(item.job_run().expect("a run"), px(7));
        item.result_rgba = Some(px(2));
        item.finish_job();
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(7), "a re-cut redoes it from the built start");
        assert!(!item.cuts_anothers_cutout(birefnet));
        assert!(item.cuts_anothers_cutout(silueta), "and Silueta's Process would build on BiRefNet's");
        let _ = item.undo_step();
        assert_eq!(item.run_input(Engine::Segment), None, "undo is back on Silueta's cut from the photo");
    }

    /// A live tweak after another engine's step records where it started,
    /// so the next tweak redoes it instead of stacking on its own output.
    #[test]
    fn a_live_tweak_records_its_start_so_the_next_one_does_not_stack() {
        let mut item = fixture_item(1);
        land(&mut item, Engine::Upscale, Some(StepKind::Upscale), 3);
        let start = item.run_start(Engine::Segment);
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(3), "an effect starts from the upscale");

        item.result_rgba = Some(px(4)); // the first tweak's preview landed
        item.amend();
        item.record_origin(Engine::Segment, start, None);
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(3), "the next tweak starts there too");
        assert_eq!(item.run_input(Engine::Upscale).map(|b| b.get_pixel(0, 0).0[0]), Some(4), "a new upscale builds on the effect");
    }

    /// An upscale or colorize hands its thread the stored start, not the
    /// unpacked pixels: the UI thread never decompresses a step.
    #[test]
    fn an_upscale_reads_its_start_on_its_own_thread() {
        let mut item = fixture_item(1);
        land(&mut item, Engine::Segment, Some(StepKind::Process), 3);
        let input = item.run_input_or_source(Engine::Upscale);
        assert!(matches!(input, Pixels::Parked(_)), "a handle to the stored step");
        assert_eq!(input.read().map(|b| b.get_pixel(0, 0).0[0]), Ok(3));
    }

    /// Parking releases the working image without a step; the timeline
    /// brings it back.
    #[test]
    fn a_parked_result_comes_back_from_the_timeline() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(px(3));
        crate::gui::tests::fixtures::put_in(&mut item, BatchStatus::Done);
        item.commit(StepKind::Process);
        assert!(item.park_result());
        assert!(item.result_rgba.is_none());
        item.start_job(None, Engine::Segment);
        item.abandon_job();
        assert_eq!(item.status(), BatchStatus::Done, "a parked result still counts as shown");
        item.unpark_result();
        assert_eq!((shown(&item), item.timeline.undo_steps()), (Some(3), 1));

        let _ = item.undo_step();
        item.unpark_result();
        assert_eq!(shown(&item), None, "the original has no result to bring back");
    }

    #[test]
    fn evictable_bytes_count_only_the_tensor_caches() {
        let mut item = fixture_item(1);
        assert_eq!(item.evictable_tensor_bytes(), 0);
        item.selection_mask = Some(Arc::new(prunr_core::selection::MaskArtifact::new_empty(16, 16)));
        let img = Arc::new(image::RgbaImage::new(4, 4));
        item.upscale = Some(UpscaleCache::new(img.clone(), img, None));
        assert_eq!(item.evictable_tensor_bytes(), 0, "planes and upscale buffers are not evictable");
    }

    // ── selection_* fields ───────────────────────────────────────────────

    #[test]
    fn invalidate_selection_clears_all_four_fields() {
        let mut item = fixture_item(1);
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        let hash = mask.content_hash();
        item.selection_mask = Some(Arc::new(mask));
        item.selection_hash = Some(hash);
        item.selection_tex_pending = Some((hash, crate::gui::background_io::SelectionStyle::with_fill(0.15)));
        item.invalidate_selection();
        assert!(item.selection_mask.is_none(), "selection_mask must be cleared");
        assert!(item.selection_hash.is_none(), "selection_hash must be cleared");
        assert!(item.selection_texture.is_none(), "selection_texture must be cleared");
        assert!(item.selection_tex_pending.is_none(), "pending build key must be cleared");
    }

}
