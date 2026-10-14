//! Pure data types for batch items and the undo timeline each carries.
//!
//! These types are GUI-agnostic in spirit (they hold egui texture handles
//! because the texture lifecycle is per-item, but no rendering happens here).
//! Logic that mutates these types lives in the coordinators
//! (`BatchManager`, `Processor`); undo lives in `timeline`.

use std::path::PathBuf;
use std::sync::Arc;

use super::brush_state::CutoutAction;

/// An upscale run kept for Refine: the model's `raw` output, the `input`
/// it read (after Denoise), and that input enlarged to the output size
/// (`reference`) once AI blend or Color match needed it.
#[derive(Clone)]
pub(crate) struct UpscaleCache {
    pub(crate) raw: Arc<image::RgbaImage>,
    pub(crate) input: Arc<image::RgbaImage>,
    pub(crate) reference: Option<Arc<image::RgbaImage>>,
    /// Which run this is, for the step it lands on to claim.
    run: u64,
}

impl UpscaleCache {
    pub(crate) fn new(raw: Arc<image::RgbaImage>, input: Arc<image::RgbaImage>, reference: Option<Arc<image::RgbaImage>>) -> Self {
        Self { raw, input, reference, run: next_run() }
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

/// A fresh id for a kept run, for the step it lands on to claim.
fn next_run() -> u64 {
    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A result's background for export, baked by `bake` (in a save thread
/// where there is one: an effect blurs the full photo).
#[derive(Clone)]
pub(crate) struct ExportBg {
    backdrop: super::item_settings::Backdrop,
    image: Option<Arc<BgImage>>,
    fit: prunr_core::BgImageFit,
    photo: Option<BackdropPhoto>,
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

impl DepthExport {
    /// Write the map into `folder`.
    pub(crate) fn write(&self, folder: &std::path::Path, export: &ExportDepth) -> Result<(), String> {
        let depth = self.depth.get(export, None)?;
        let (w, h) = self.dims;
        let bytes = prunr_core::depth::encode_depth_png(&depth, w, h).map_err(|e| e.to_string())?;
        std::fs::write(folder.join(&self.name), bytes).map_err(|e| e.to_string())
    }
}

/// `<stem>-depth.png` for an image named `filename`.
pub(crate) fn depth_map_name(filename: &str) -> String {
    let stem = std::path::Path::new(filename).file_stem().map_or_else(|| filename.into(), |s| s.to_string_lossy());
    prunr_core::depth::map_file_name(&stem)
}

/// The photo an effect background is made from: the image the shown
/// result's run started from (a photo with an object erased first), or
/// the original. An edited one is unpacked where it is used, off the UI
/// thread.
#[derive(Clone)]
pub(crate) enum BackdropPhoto {
    Edited(super::timeline::StoredImage),
    Original(Arc<image::RgbaImage>),
}

impl BackdropPhoto {
    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        match self {
            BackdropPhoto::Edited(stored) => stored.load(),
            BackdropPhoto::Original(photo) => Some(Arc::clone(photo)),
        }
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
        let chosen = match &self.photo {
            Some(BackdropPhoto::Edited(stored)) => stored.load().map(|photo| (photo, false)),
            Some(BackdropPhoto::Original(photo)) => Some((Arc::clone(photo), true)),
            None => None,
        };
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

/// Where an image's raw bytes live — file path (lazy) or in-memory (clipboard/paste).
#[derive(Clone)]
pub(crate) enum ImageSource {
    /// Loaded from a file. Bytes read on demand and dropped after use.
    Path(PathBuf),
    /// From clipboard, drag-drop, or CLI pipe. Bytes kept in memory.
    Bytes(Arc<Vec<u8>>),
}

impl ImageSource {
    /// Read the image bytes. For Path, reads from disk. For Bytes, clones the Arc.
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
            (Self::Path(a), Self::Path(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub(crate) fn load_bytes(&self) -> std::io::Result<Arc<Vec<u8>>> {
        match self {
            Self::Path(path) => Ok(Arc::new(std::fs::read(path)?)),
            Self::Bytes(bytes) => Ok(bytes.clone()),
        }
    }

    /// Estimated compressed file size (for admission cost estimation).
    pub(crate) fn estimated_size(&self) -> usize {
        match self {
            Self::Path(path) => std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0),
            Self::Bytes(bytes) => bytes.len(),
        }
    }
}

use super::timeline::{store, DocState, Origin, RunStart, StepKind, Timeline};

/// A transition to a new image: first the image on screen is held while
/// the new one's texture is built, then the new one reveals over it.
pub(crate) enum Reveal {
    Held(egui::TextureHandle),
    /// The image revealed over, the start, and where the reveal starts
    /// (the subject, when one was found; corner to corner otherwise).
    Running(egui::TextureHandle, std::time::Instant, Option<Arc<[f32]>>),
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

/// Who cut a step out: a background-removal model, or No model's pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cutter {
    Model(prunr_core::ModelKind),
    NoModel,
}

impl Cutter {
    /// The active model's: No model when it has no segmentation model.
    pub(crate) fn of(model: Option<prunr_core::ModelKind>) -> Self {
        model.map_or(Cutter::NoModel, Cutter::Model)
    }
}

/// A lens-blur focus and what it was measured from.
struct LensFocus {
    depth: std::sync::Weak<prunr_core::depth::DepthMap>,
    result: Option<std::sync::Weak<image::RgbaImage>>,
    value: f32,
}

impl LensFocus {
    fn is_for(&self, depth: &Arc<prunr_core::depth::DepthMap>, result: Option<&Arc<image::RgbaImage>>) -> bool {
        let same_result = match (&self.result, result) {
            (None, None) => true,
            (Some(w), Some(r)) => std::sync::Weak::ptr_eq(w, &Arc::downgrade(r)),
            _ => false,
        };
        same_result && std::sync::Weak::ptr_eq(&self.depth, &Arc::downgrade(depth))
    }
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

/// See `BatchItem::correction_cache`.
struct CorrectionCache {
    edits: Option<Arc<prunr_core::selection::MaskArtifact>>,
    selection: Arc<prunr_core::selection::MaskArtifact>,
    action: CutoutAction,
    plane: Arc<prunr_core::selection::MaskArtifact>,
}

impl CorrectionCache {
    fn matches(&self, edits: Option<&Arc<prunr_core::selection::MaskArtifact>>, selection: &Arc<prunr_core::selection::MaskArtifact>, action: CutoutAction) -> bool {
        let same_edits = match (&self.edits, edits) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        same_edits && Arc::ptr_eq(&self.selection, selection) && self.action == action
    }
}

/// The photo faces are found in: a run's start image, the source, or an
/// upscale's output. Compared by identity, never unpacked to compare.
#[derive(Clone)]
pub(crate) enum FacePhoto {
    Start(super::timeline::StoredImage),
    Source(Arc<image::RgbaImage>),
    Upscaled(Arc<image::RgbaImage>),
}

impl FacePhoto {
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Start(a), Self::Start(b)) => Arc::ptr_eq(a, b),
            (Self::Source(a), Self::Source(b)) | (Self::Upscaled(a), Self::Upscaled(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        match self {
            Self::Start(stored) => stored.load(),
            Self::Source(rgba) | Self::Upscaled(rgba) => Some(Arc::clone(rgba)),
        }
    }
}

/// A result for an export off the UI thread: the one on screen, or one
/// parked in the timeline while its image is off screen, unpacked where
/// it is used.
#[derive(Clone)]
pub(crate) enum ResultRef {
    Shown(Arc<image::RgbaImage>),
    Parked(super::timeline::StoredImage),
}

impl ResultRef {
    pub(crate) fn load(&self) -> Option<Arc<image::RgbaImage>> {
        match self {
            Self::Shown(rgba) => Some(Arc::clone(rgba)),
            Self::Parked(stored) => stored.load(),
        }
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

/// Where `BatchItem::warm_steps` holds the step an undo goes back to and
/// the one a redo goes to.
pub(crate) const BACK: usize = 0;
pub(crate) const AHEAD: usize = 1;

/// A step next to the shown one, and its image once unpacked.
pub(crate) struct WarmStep {
    step: super::timeline::StoredImage,
    pub(crate) image: Option<Arc<image::RgbaImage>>,
}

pub(crate) struct BatchItem {
    pub(crate) id: u64,
    pub(crate) filename: String,
    pub(crate) source: ImageSource,
    pub(crate) dimensions: (u32, u32),
    /// Pre-decoded source RGBA (decoded on background thread for instant switching)
    pub(crate) source_rgba: Option<Arc<image::RgbaImage>>,
    /// Same pixels as `source_rgba`, pre-wrapped in `DynamicImage::ImageRgba8`
    /// and shared via `Arc`. Built lazily by `build_preview_inputs` on the
    /// first live-preview dispatch for this item and reused for every
    /// subsequent dispatch in the session, so each drag-tweak avoids a ~15ms,
    /// ~48MB memcpy clone. Cleared whenever `source_rgba` is re-populated
    /// (re-decode) so the cache can't go stale.
    pub(crate) source_dyn: Option<Arc<image::DynamicImage>>,
    pub(crate) source_texture: Option<egui::TextureHandle>,
    pub(crate) thumb_texture: Option<egui::TextureHandle>,
    pub(crate) thumb_pending: bool,
    pub(crate) result_rgba: Option<Arc<image::RgbaImage>>,
    /// `source_dyn` for a live preview that starts from an earlier result:
    /// a `DynamicImage` wrapping that start, so drag ticks don't re-clone
    /// the full RGBA buffer. Keyed by the start's Arc (`Arc::ptr_eq`), so a
    /// new start misses; cleared when previews start from the source.
    pub(crate) start_dyn_cache: Option<(Arc<image::RgbaImage>, Arc<image::DynamicImage>)>,
    pub(crate) result_texture: Option<egui::TextureHandle>,
    /// True while a background thread is building the source ColorImage.
    pub(crate) source_tex_pending: bool,
    /// True while a background thread is building the result ColorImage.
    pub(crate) result_tex_pending: bool,
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
    /// A face search under way: for which photo, with which model.
    pub(crate) faces_pending: Option<(FacePhoto, prunr_models::ModelId)>,
    /// The photo with the found faces blended in, as a preview built it,
    /// so other knobs' previews reuse it. One image of RAM while Restore
    /// faces is on.
    pub(crate) faced_photo: Option<(Arc<image::DynamicImage>, super::live_preview::FacedKey)>,
    /// True while a background thread is decoding source bytes to RGBA.
    pub(crate) decode_pending: bool,
    /// Every state this image has been in, and which one is shown. The
    /// fields above and below (result, recipe, selection, settings) are
    /// the working copy of the shown state.
    pub(crate) timeline: Timeline,
    /// The run (Process, erase, upscale) whose result this item awaits.
    pub(crate) job: Option<Job>,
    pub(crate) status: BatchStatus,
    pub(crate) selected: bool,
    /// Per-image processing settings. Edited via the adjustments toolbar.
    pub(crate) settings: super::item_settings::ItemSettings,
    /// The recipe that produced the current result_rgba. None if never processed.
    pub(crate) applied_recipe: Option<prunr_core::ProcessingRecipe>,
    /// Compressed cached tensor from Tier 1 inference (for Tier 2 mask reruns).
    pub(crate) cached_tensor: Option<super::worker::CompressedTensor>,
    /// The last upscale run's model output and input, so Refine's
    /// after-the-model knobs re-finish it without a model run. Mirrors
    /// `cached_tensor`'s contract: NOT cleared by `reset_result_caches()`;
    /// only cleared explicitly on Tier-1 invalidation (model swap).
    ///
    /// RAM impact: at 4K source × 4× upscale = 15360×8640 RGBA ≈ 500 MB,
    /// plus as much again while AI blend or Color match keep the enlarged
    /// reference. Not under the tensor budget (`evictable_tensor_bytes`).
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
    /// SubjectOutline live-preview cache: the "masked subject" base
    /// (`postprocess_from_flat` output) that edge composition draws onto.
    /// Keyed by `(MaskRecipe, ModelKind)` — when mask settings change, the
    /// base is rebuilt; when only edge settings change, the base is reused
    /// and we skip ~50-100 ms of Lanczos + guided filter per Edge tick.
    pub(crate) cached_masked_base: Option<(Arc<image::RgbaImage>, prunr_core::MaskRecipe, prunr_core::ModelKind)>,
    /// Which preset was last APPLIED to this image (via the dropdown's row
    /// click or via Reset All). The preset button compares current `settings`
    /// against this preset's values to show a modified/clean icon. Stays set
    /// across unrelated tweaks — so "Portrait ✎" keeps saying Portrait even
    /// after the user modifies something.
    pub(crate) applied_preset: String,
    /// Never mutate outside `set_bg_image_hashed` / `clear_bg_image` /
    /// `adopt_settings`: the writers that keep `settings.bg_image_hash` in step,
    /// which the recipe-diff dispatch reads to fire CompositeOnly.
    pub(crate) bg_image: Option<Arc<BgImage>>,
    /// The picture behind the result (a background image or an effect
    /// made from the photo) and its `Backdrop::picture_key`.
    pub(crate) backdrop_texture: Option<(u64, egui::TextureHandle)>,
    /// The key of the picture a background thread is building, so the
    /// next frame doesn't start the same one again.
    pub(crate) backdrop_tex_pending: Option<u64>,
    /// Where a lens-blur backdrop focuses, measured once per depth map
    /// and result: a preview tick that leaves the subject where it was
    /// keeps the backdrop it has. Weak handles pin the allocations, so a
    /// new result can never pass for the old one by address.
    lens_focus: Option<LensFocus>,

    /// Shared selection mask. Both Paint Brush and Magic Brush write
    /// here. Signed i8 plane at source image resolution.
    ///
    /// Survives Process; an erase that lands clears it, as do Clear and a
    /// source change.
    pub(crate) selection_mask: Option<Arc<prunr_core::selection::MaskArtifact>>,

    /// Content hash of `selection_mask`, set in lockstep with the mask by
    /// `commit_selection_mask`. Half of the selection texture's key.
    pub(crate) selection_hash: Option<u64>,
    /// On a cut-out, the restores and erases already made (see `DocState::edits`).
    pub(crate) edits: Option<Arc<prunr_core::selection::MaskArtifact>>,
    /// The last correction built from the selection and the edits, with
    /// the inputs it was built from (held, so a match is never a reused address).
    correction_cache: std::sync::Mutex<Option<CorrectionCache>>,

    /// Key of the selection texture build in flight, if any; cleared when
    /// that build lands or is dropped as stale.
    pub(crate) selection_tex_pending: Option<super::background_io::SelectionTextureKey>,

    /// Selection overlay texture with the key it was built from. Rebuilt
    /// whenever the mask or the baked style differs from the key.
    pub(crate) selection_texture: Option<SelectionTexture>,

    /// Cached SAM 2 image embedding, filled whenever Magic Brush is active
    /// and this item is selected. Invalidated on source change.
    pub(crate) magic_brush_embedding: Option<std::sync::Arc<prunr_core::sam::SamEmbedding>>,
    /// The last Magic Brush stroke decoded on this image, kept so the
    /// Confidence knob re-thresholds it in place instead of needing
    /// another click.
    pub(crate) last_decode: Option<LastDecode>,
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
    pub(crate) depth_photo: Option<(Arc<image::DynamicImage>, DepthPhotoKey)>,
}

/// The depth map's run for one photo: `epoch` counts the photos the item
/// has had, so a map measured from a replaced one is dropped on landing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DepthRun {
    pub(crate) epoch: u64,
    pub(crate) pending: bool,
    /// The run failed; not retried until the photo changes.
    pub(crate) failed: bool,
}

/// What a depth-looked photo was made from.
#[derive(Clone)]
pub(crate) struct DepthPhotoKey {
    pub(crate) base: Arc<image::DynamicImage>,
    pub(crate) depth: Arc<prunr_core::depth::DepthMap>,
    pub(crate) look: prunr_core::depth::DepthLook,
}

impl DepthPhotoKey {
    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base) && Arc::ptr_eq(&self.depth, &other.depth) && self.look == other.look
    }
}

/// The logits of a Magic Brush stroke and how they were applied.
pub(crate) struct LastDecode {
    pub(crate) output: Arc<prunr_core::sam::SamDecoderOutput>,
    pub(crate) mode: prunr_core::selection::BrushMode,
    /// The selection hash the stroke committed.
    pub(crate) committed_hash: u64,
    pub(crate) reading: prunr_core::MaskReading,
}

/// A selection overlay texture and what it was built from. `shown` is
/// the plane on the GPU, so the next change can upload only its patch;
/// it is usually the item's own mask (one Arc), and at most one extra
/// plane per item while a newer mask awaits its patch.
pub(crate) struct SelectionTexture {
    pub(crate) key: super::background_io::SelectionTextureKey,
    pub(crate) handle: egui::TextureHandle,
    pub(crate) shown: Arc<prunr_core::selection::MaskArtifact>,
    /// With Feather on, the feathered plane the texture was built from, so
    /// Delete / Copy / Cut act on it without filtering again.
    pub(crate) feathered: Option<Arc<prunr_core::selection::MaskArtifact>>,
}

impl BatchItem {
    /// Clear every edge cache tier together — compressed multi-scale set,
    /// the hot decompressed tensor, and the derived pre-dilation mask. The
    /// mask is always built from the tensor, so any tensor change invalidates
    /// all of them.
    pub(crate) fn invalidate_edge_cache(&mut self) {
        self.cached_edge_tensors = None;
        self.volatile_edge_tensor = None;
        self.edge_planes = Default::default();
    }

    /// Replaces the compressed tensor and the decoded copy tied to it.
    pub(crate) fn set_cached_tensor(&mut self, tensor: Option<super::worker::CompressedTensor>) {
        self.cached_tensor = tensor;
        self.volatile_seg_tensor = None;
    }

    /// Drop the decompressed tensors kept for slider drags; the
    /// compressed caches stay, so the next drag rebuilds them once.
    pub(crate) fn drop_hot_tensors(&mut self) {
        self.volatile_seg_tensor = None;
        self.volatile_edge_tensor = None;
        self.last_decode = None;
        if let Some(t) = &mut self.selection_texture {
            t.feathered = None;
        }
        self.drop_correction_cache();
    }

    fn drop_correction_cache(&mut self) {
        *self.correction_cache.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// What corrects a cut-out: the edits already made, with the
    /// selection on top as `action` says (restored, erased, or nothing
    /// while only selecting), at full resolution. The selection is the
    /// feathered plane once the overlay has built it (its landing re-cuts
    /// again). Built once per change of its inputs.
    pub(crate) fn correction_plane(&self, style: super::background_io::SelectionStyle, action: CutoutAction) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        let selection = self.selection_mask.as_ref()
            .filter(|m| !m.is_blank() && action != CutoutAction::Select)
            .map(|m| self.feathered_selection(style).unwrap_or_else(|| Arc::clone(m)));
        match (&self.edits, selection) {
            (None, None) => None,
            (Some(edits), None) => Some(Arc::clone(edits)),
            // A selection is stored as an erase: as it is, it erases.
            (None, Some(selection)) if action == CutoutAction::Erase => Some(selection),
            (edits, Some(selection)) => {
                let mut cache = self.correction_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(hit) = cache.as_ref().filter(|c| c.matches(edits.as_ref(), &selection, action)) {
                    return Some(Arc::clone(&hit.plane));
                }
                let plane = Arc::new(apply_selection(edits.as_deref(), &selection, action));
                *cache = Some(CorrectionCache { edits: edits.clone(), selection, action, plane: Arc::clone(&plane) });
                Some(plane)
            }
        }
    }

    /// Fold the selection into the edits as `action` says and clear it:
    /// what was restored or erased stays, and the next strokes start a new
    /// area. False when there was no selection.
    pub(crate) fn bake_selection(&mut self, style: super::background_io::SelectionStyle, action: CutoutAction) -> bool {
        if self.selection_mask.is_none() {
            return false;
        }
        if action != CutoutAction::Select {
            self.edits = self.correction_plane(style, action);
        }
        self.selection_mask = None;
        self.selection_hash = None;
        self.selection_texture = None;
        self.drop_correction_cache();
        true
    }

    /// The current selection feathered as `style` says, once the overlay
    /// has built it. Only the feather matters: a change of tint or fill
    /// keeps the plane.
    fn feathered_selection(&self, style: super::background_io::SelectionStyle) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        let (hash, texture) = (self.selection_hash?, self.selection_texture.as_ref()?);
        let (built_hash, built_style) = texture.key;
        (built_hash == hash && built_style.edge_feather_px == style.edge_feather_px)
            .then(|| texture.feathered.clone())
            .flatten()
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
        if let Some(run) = self.upscale.as_mut() {
            run.reference = None;
        }
    }

    /// Clear the cached upscale run.
    /// Called on model switch and on item error. Tier-1 knob changes do NOT
    /// call this — the next dispatch's pump_upscale_results overwrites the
    /// buffer in-place, so explicit invalidation would just drop work the
    /// next dispatch is about to redo. Model switch is special because the
    /// next dispatch may never happen (user may drag a Tier-2 knob first),
    /// and the stale buffer is from the wrong model.
    pub(crate) fn invalidate_upscale_cache(&mut self) {
        self.upscale = None;
    }

    /// Clear the kept colorize run, on a model switch and on item error.
    pub(crate) fn invalidate_colorize_cache(&mut self) {
        self.colorize = None;
    }

    /// Clear the cached SAM encoder embedding. Called on source image
    /// change and on model switch to a non-Selection category.
    pub(crate) fn invalidate_magic_brush_embedding(&mut self) {
        self.magic_brush_embedding = None;
        self.magic_preencode_refused = false;
        self.last_decode = None;
    }

    /// The photo changed: its depth map, any run measuring the old one and
    /// the photo looked from it are out of date.
    pub(crate) fn invalidate_depth(&mut self) {
        self.depth = None;
        self.depth_photo = None;
        self.depth_run = DepthRun { epoch: self.depth_run.epoch.wrapping_add(1), pending: false, failed: false };
    }

    /// A depth map is wanted and none is kept, under way or failed.
    pub(crate) fn needs_depth_run(&self) -> bool {
        self.depth.is_none() && !self.depth_run.pending && !self.depth_run.failed
    }

    /// Keep a depth map measured at `epoch`, unless the photo changed
    /// since. True when it was kept.
    pub(crate) fn land_depth(&mut self, epoch: u64, result: Result<Arc<prunr_core::depth::DepthMap>, String>) -> bool {
        if epoch != self.depth_run.epoch {
            return false;
        }
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

    /// The last decode while its stroke still owns the selection; any
    /// other author changing the selection ends the retune.
    pub(crate) fn retunable_decode(&mut self) -> Option<&mut LastDecode> {
        let hash = self.selection_hash;
        if self.last_decode.as_ref().is_some_and(|ld| hash != Some(ld.committed_hash)) {
            self.last_decode = None;
        }
        self.last_decode.as_mut()
    }

    /// What was made from the source pixels goes with them: the
    /// selection, the Magic Brush embedding and the depth map. Returns
    /// whether a selection or embedding was there.
    pub(crate) fn invalidate_source_derived(&mut self) -> bool {
        let any = self.selection_mask.is_some() || self.magic_brush_embedding.is_some();
        self.invalidate_selection();
        self.invalidate_magic_brush_embedding();
        self.invalidate_depth();
        any
    }

    /// Clear every selection-related cache: mask, hash, texture, pending build.
    /// Called by `BatchManager::clear_selection` (Esc / Clear) and
    /// `invalidate_source_derived`.
    pub(crate) fn invalidate_selection(&mut self) {
        self.selection_mask = None;
        self.selection_hash = None;
        self.selection_texture = None;
        self.selection_tex_pending = None;
        self.last_decode = None;
    }

    /// True while a model works on this image: a cut-out the worker has
    /// taken, or an upscale, colorize or erase, which start their job only
    /// when they run (a queued cut-out holds its job while it waits).
    pub(crate) fn is_running(&self) -> bool {
        self.status == BatchStatus::Processing || self.job.as_ref().is_some_and(|job| job.engine != Engine::Segment)
    }

    /// True while a run, a decode, a texture or a thumbnail is still to land.
    pub(crate) fn is_busy(&self) -> bool {
        self.status == BatchStatus::Processing || self.job.is_some() || self.has_pending_work()
    }

    /// True while a background thread owes this item a decode, a texture
    /// or a thumbnail.
    pub(crate) fn has_pending_work(&self) -> bool {
        self.result_tex_pending || self.source_tex_pending || self.decode_pending
            || self.thumb_pending || self.backdrop_tex_pending.is_some()
            || self.selection_tex_pending.is_some()
    }

    /// Make `mask` the selection as one undo step of `kind`. Every author
    /// (Paint strokes, Magic Brush candidates, Invert) goes through here.
    /// Returns `false` and changes nothing when `mask` is byte-identical
    /// to the current selection: a no-op stroke must not be a step.
    pub(crate) fn commit_selection_mask(&mut self, mask: Arc<prunr_core::selection::MaskArtifact>, kind: StepKind) -> bool {
        let hash = mask.content_hash();
        if self.selection_hash == Some(hash) {
            return false;
        }
        self.selection_mask = Some(mask);
        self.selection_hash = Some(hash);
        self.commit(kind);
        true
    }

    /// The selection the shown step replaced, the base a Confidence
    /// retune of the last stroke merges onto.
    pub(crate) fn pre_stroke_selection(&self) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        self.timeline.previous()?.selection.as_ref()?.load()
    }

    /// Replace the selection within the shown step: a retune of the last
    /// commit, which stays one step. `hash` is the mask's content hash,
    /// computed where the mask was built. Returns false when unchanged.
    pub(crate) fn replace_selection_in_place(&mut self, mask: Arc<prunr_core::selection::MaskArtifact>, hash: u64) -> bool {
        if self.selection_hash == Some(hash) {
            return false;
        }
        self.selection_mask = Some(mask);
        self.selection_hash = Some(hash);
        self.amend();
        true
    }

    /// The working copy as a timeline state, sharing what the shown state
    /// already stores.
    fn snapshot(&self, kind: StepKind) -> DocState {
        let shown = self.timeline.current();
        DocState {
            kind,
            result: store(self.result_rgba.as_ref(), shown.result.as_ref()),
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
        }
    }

    /// Record the working copy as a new step.
    pub(crate) fn commit(&mut self, kind: StepKind) {
        self.unpark_result();
        let state = self.snapshot(kind);
        self.timeline.push(state);
        self.warm_neighbour_steps();
    }

    /// The image leaves or comes on screen: off screen it lets go of its
    /// unpacked neighbour steps, on screen it warms them.
    pub(crate) fn set_on_screen(&mut self, on: bool) {
        self.on_screen = on;
        if on {
            self.warm_neighbour_steps();
        } else {
            *self.warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = [None, None];
        }
    }

    /// Unpack the images of the steps either side of the shown one on a
    /// worker and hold them; drop whatever was held for other steps.
    fn warm_neighbour_steps(&self) {
        if !self.on_screen {
            return;
        }
        let back = self.timeline.previous().and_then(|s| s.result.clone());
        let ahead = self.timeline.next().and_then(|s| s.result.clone());
        for (at, stored) in [(BACK, back), (AHEAD, ahead)] {
            let slot = Arc::clone(&self.warm_steps);
            let mut held = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match stored {
                Some(stored) if held[at].as_ref().is_some_and(|w| Arc::ptr_eq(&w.step, &stored)) => {}
                Some(stored) => {
                    held[at] = Some(WarmStep { step: Arc::clone(&stored), image: None });
                    drop(held);
                    std::thread::spawn(move || {
                        let image = stored.load();
                        let mut held = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        // A newer step may have taken the slot meanwhile.
                        if let Some(w) = held[at].as_mut().filter(|w| Arc::ptr_eq(&w.step, &stored)) {
                            w.image = image;
                        }
                    });
                }
                None => held[at] = None,
            }
        }
    }

    /// Fold the working copy into the shown step: a job it started landed.
    pub(crate) fn amend(&mut self) {
        self.unpark_result();
        let shown = self.timeline.current();
        let state = DocState { at: shown.at, ..self.snapshot(shown.kind) };
        *self.timeline.current_mut() = state;
    }

    /// Keep what is on screen to reveal the next image over, when memory
    /// allows; otherwise the next image replaces it outright.
    pub(crate) fn hold_for_reveal(&mut self) {
        if !matches!(self.reveal, Some(Reveal::Held(_))) {
            self.reveal = self.result_texture.clone()
                .or_else(|| self.source_texture.clone())
                .filter(|t| super::views::effects::reveal_fits(t.byte_size(), crate::hardware::available_ram_bytes_throttled()))
                .map(Reveal::Held);
        }
    }

    /// Paint a finished tile into its square of the run's preview, made
    /// on the run's first tile (or anew when the size changes). Only the
    /// tile is uploaded (`set_partial`).
    pub(crate) fn paint_tile_preview(&mut self, ctx: &egui::Context, tile: &prunr_core::TilePreview) {
        let full = [tile.full_width as usize, tile.full_height as usize];
        let texture = match &mut self.tile_preview {
            Some(t) if t.size() == full => t,
            slot => slot.insert(ctx.load_texture(
                format!("tile-preview-{}", self.id),
                egui::ColorImage::new(full, vec![egui::Color32::TRANSPARENT; full[0] * full[1]]),
                egui::TextureOptions::LINEAR,
            )),
        };
        let image = egui::ColorImage::from_rgba_unmultiplied([tile.width as usize, tile.height as usize], &tile.rgba);
        texture.set_partial([tile.x as usize, tile.y as usize], image, egui::TextureOptions::LINEAR);
    }

    /// A run has landed: its finished-tile preview, the whole image by
    /// now, is shown until the result's texture lands and the result
    /// reveals over it. Without one, the shown image is held as usual.
    pub(crate) fn hold_tile_preview_for_reveal(&mut self) {
        match self.tile_preview.take() {
            Some(preview) => self.reveal = Some(Reveal::Held(preview)),
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
        self.thumb_pending = true;
    }

    /// Show `rgba` as the result: what is on screen, or the run's tile
    /// preview, is held to reveal it over, and a waiting image is done.
    pub(crate) fn show_result(&mut self, rgba: Arc<image::RgbaImage>) {
        self.hold_tile_preview_for_reveal();
        self.unpark_result();
        self.take_result(rgba);
        if self.status == BatchStatus::Pending {
            self.status = BatchStatus::Done;
        }
    }

    /// Show a finished upscale or colorize with the recipe it ran: over
    /// its tile preview when there is one.
    pub(crate) fn show_run_result(&mut self, shown: Arc<image::RgbaImage>, recipe: prunr_core::ProcessingRecipe) {
        self.applied_recipe = Some(recipe);
        self.status = BatchStatus::Done;
        self.hold_tile_preview_for_reveal();
        self.take_result(shown);
    }

    /// A prepared texture lands on the side it was made for; a result's
    /// reveals over what was held.
    pub(crate) fn land_texture(&mut self, tex: egui::TextureHandle, target: super::background_io::TexTarget) {
        use super::background_io::TexTarget;
        if target == TexTarget::Source {
            self.source_texture = Some(tex);
            self.source_tex_pending = false;
            tracing::info!(item_id = self.id, kind = "source", "texture uploaded");
        } else {
            self.result_texture = Some(tex);
            self.result_tex_pending = false;
            if target == TexTarget::Result {
                self.start_reveal();
            }
            tracing::info!(item_id = self.id, kind = "result", "texture uploaded");
        }
    }

    /// The decoded source lands: the item holds its pixels (`true`), or
    /// shows why decoding failed.
    pub(crate) fn land_decode(&mut self, result: Result<Arc<image::RgbaImage>, String>) -> bool {
        self.decode_pending = false;
        match result {
            Ok(rgba) => {
                self.source_rgba = Some(rgba);
                // Live-preview DynamicImage cache is built from source_rgba;
                // invalidate so the next dispatch rebuilds against the new one.
                self.source_dyn = None;
                self.invalidate_source_derived();
                true
            }
            Err(msg) => {
                self.status = BatchStatus::Error(msg);
                false
            }
        }
    }

    /// A selection overlay built off-thread lands when it is still for
    /// the mask shown; `load` uploads a whole one. Returns whether a
    /// feathered plane the cut-out has not had yet came with it.
    pub(crate) fn land_selection_texture(
        &mut self,
        result: super::background_io::SelectionTextureResult,
        load: impl FnOnce(egui::ColorImage) -> egui::TextureHandle,
    ) -> bool {
        use super::background_io::SelectionImage;
        if self.selection_tex_pending == Some(result.key) {
            self.selection_tex_pending = None;
        }
        if self.selection_hash != Some(result.key.0) {
            return false;
        }
        if self.selection_mask.is_none() {
            return false;
        }
        let super::background_io::SelectionTextureResult { image, key, shown, .. } = result;
        match image {
            SelectionImage::Full { image, feathered } => {
                // Re-cut only for a feathered plane the cut-out has not
                // had yet; a rebuild for zoom keeps the same one.
                let had_it = self.selection_texture.as_ref().is_some_and(|t| {
                    t.feathered.is_some() && t.key.0 == key.0 && t.key.1.edge_feather_px == key.1.edge_feather_px
                });
                let new_plane = feathered.is_some() && !had_it;
                self.selection_texture = Some(SelectionTexture { key, handle: load(image), shown, feathered });
                new_plane
            }
            // A patch whose base texture has moved on is dropped; the
            // per-frame check asks again against the new base.
            SelectionImage::Patch { base_hash, pos, image } => {
                if let Some(tex) = self.selection_texture.as_mut().filter(|t| t.key == (base_hash, key.1)) {
                    tex.handle.set_partial(pos, image, egui::TextureOptions::LINEAR);
                    tex.key = key;
                    tex.shown = shown;
                }
                false
            }
        }
    }

    /// Whether an erase has something to fill.
    pub(crate) fn has_painted_region(&self) -> bool {
        self.selection_mask.as_ref().is_some_and(|m| m.has_selected_region())
    }

    /// A picture built for the backdrop lands when it is still the
    /// item's backdrop (`true`); a stale or empty build frees the slot
    /// for the next frame to build the current one.
    pub(crate) fn land_backdrop(&mut self, key: u64, ci: Option<egui::ColorImage>, ctx: &egui::Context) -> bool {
        if self.backdrop_tex_pending == Some(key) {
            self.backdrop_tex_pending = None;
        }
        let Some(ci) = ci.filter(|_| self.backdrop_key() == Some(key)) else { return false };
        // WrapMode::Repeat enables BgImageFit::Tile (UV > 1.0 wraps);
        // other fits keep UVs in [0, 1] so the wrap is a no-op.
        let opts = egui::TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            wrap_mode: egui::TextureWrapMode::Repeat,
            mipmap_mode: None,
        };
        self.backdrop_texture = Some((key, ctx.load_texture(format!("backdrop_{key:x}"), ci, opts)));
        true
    }

    /// Faces found for this image are kept for the blend when they are
    /// for the photo and model it still wants: how many, or why the
    /// search failed. `None`: a search for something else, dropped.
    pub(crate) fn keep_faces(&mut self, landed: super::processor::FacesResult, stage: FaceStage) -> Option<Result<usize, String>> {
        if self.pending_for(&landed.photo, landed.model) {
            self.faces_pending = None;
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
        self.found_faces = Some(FoundFaces { photo: landed.photo, found_at: std::time::Instant::now(), model: landed.model, faces: Arc::new(faces), failed: outcome.is_err(), stopped: landed.stopped });
        self.faces_changed();
        Some(outcome)
    }

    /// The size of the frame the image is shown in: the result's, once
    /// there is one, else the source's. Every texture shown for this item
    /// (source, preview, held image, result) is drawn into it, so a
    /// hand-over between them never moves the image.
    pub(crate) fn frame_size(&self) -> egui::Vec2 {
        let (w, h) = self.result_rgba.as_ref().map_or(self.dimensions, |r| r.dimensions());
        egui::vec2(w as f32, h as f32)
    }

    /// The image shown until the new one's texture lands.
    pub(crate) fn held_texture(&self) -> Option<&egui::TextureHandle> {
        match &self.reveal {
            Some(Reveal::Held(t)) => Some(t),
            _ => None,
        }
    }

    /// A reveal under way: the image it reveals over, and how far it has
    /// come (0..1). A finished one is dropped by `settle_reveal`.
    pub(crate) fn reveal_progress(&self) -> Option<(&egui::TextureHandle, f32, Option<&[f32]>)> {
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
        // The restored step's face settings may differ from what the faced
        // photo and the cut-out over it were built with.
        self.faces_changed();
        self.subject = None;
        let shown = self.timeline.current().clone();
        self.result_rgba = shown.result.as_ref().and_then(|r| r.load());
        self.applied_recipe = shown.recipe;
        self.selection_mask = shown.selection.as_ref().and_then(|s| s.load());
        self.selection_hash = self.selection_mask.as_ref().and(shown.selection_hash);
        self.edits = shown.edits.as_ref().and_then(|e| e.load());
        if self.settings.line_mode != shown.settings.line_mode {
            self.invalidate_edge_cache();
        }
        self.settings = shown.settings;
        self.applied_preset = shown.applied_preset;
        self.status = if self.result_rgba.is_some() { BatchStatus::Done } else { BatchStatus::Pending };
        self.job = None;
        self.reset_result_caches();
        if self.result_rgba.is_none() {
            // Back to the original: the source is already on the GPU.
            self.start_reveal();
        }
    }

    /// Step back one state; returns what the undone step did.
    pub(crate) fn undo_step(&mut self) -> Option<StepKind> {
        self.unpark_result();
        let undone = self.timeline.undo()?;
        self.restore();
        self.warm_neighbour_steps();
        Some(undone)
    }

    /// Step forward one state; returns what the redone step did.
    pub(crate) fn redo_step(&mut self) -> Option<StepKind> {
        self.unpark_result();
        let redone = self.timeline.redo()?;
        self.restore();
        self.warm_neighbour_steps();
        Some(redone)
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
        self.job = Some(Job { version: self.timeline.version(), step, engine, start, seed, region, cutter: None, keep: None });
        self.tile_preview = None;
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
        self.status = BatchStatus::Done;
        self.finish_job();
    }

    /// The image a chained cut starts from, built where the run reads its
    /// file: it becomes the step's start, as an earlier image would be.
    pub(crate) fn chain_start_built(&mut self, image: Arc<image::RgbaImage>) {
        if let Some(job) = self.job.as_mut().filter(|j| j.keep.is_some()) {
            let dims = image.dimensions();
            job.start = Some(RunStart { image: super::timeline::Stored::new(image), dims });
        }
    }

    /// What a run reads besides its file: where it starts and, for a
    /// chained cut, the cut-out it keeps.
    pub(crate) fn job_run_image(&self) -> Option<super::worker::RunImage> {
        let job = self.job.as_ref()?;
        let start = self.job_input();
        match &job.keep {
            // The decoded photo spares the bridge decoding the file again.
            Some(cutout) => Some(super::worker::RunImage::Kept {
                start: start.or_else(|| self.source_rgba.clone()),
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

    /// The awaited run will not land (cancelled): the item shows its
    /// current state again.
    pub(crate) fn abandon_job(&mut self) {
        self.job = None;
        self.reveal = None;
        self.subject = None;
        self.tile_preview = None;
        let has_result = self.result_rgba.is_some() || self.result_parked();
        self.status = if has_result { BatchStatus::Done } else { BatchStatus::Pending };
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
    }

    fn start_faced(&self, engine: Engine) -> bool {
        self.run_start(engine).is_some_and(|s| self.timeline.image_faced(&s.image))
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
        self.upscale = Some(run);
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
        self.next_upscale_start(model).map_or(self.dimensions, |s| s.dims)
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

    /// The shown step's origin when a run of `engine` would redo that step.
    /// The eraser never redoes: each erase adds to the last.
    fn redo_origin(&self, engine: Engine) -> Option<&Origin> {
        self.timeline.current().origin.as_ref().filter(|o| o.engine == engine && engine != Engine::Erase)
    }

    /// Whether a run of `engine` redoes the shown step rather than
    /// building on it.
    pub(crate) fn redoes_shown_step(&self, engine: Engine) -> bool {
        self.redo_origin(engine).is_some()
    }

    /// Where a run of `engine` starts (`None`: the original): the shown
    /// step's own start when the run redoes it, the shown result otherwise.
    /// Nothing is loaded.
    pub(crate) fn run_start(&self, engine: Engine) -> Option<RunStart> {
        match self.redo_origin(engine) {
            Some(o) => o.start.clone(),
            None => self.shown_start(),
        }
    }

    /// The shown result as a run's start (`None`: the original shows).
    fn shown_start(&self) -> Option<RunStart> {
        self.timeline.current().result.clone().map(|image| RunStart {
            image,
            dims: self.result_rgba.as_ref().map_or(self.dimensions, |r| r.dimensions()),
        })
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
            FaceStage::Photo => self.run_start(Engine::Segment).map(|s| FacePhoto::Start(s.image))
                .or_else(|| self.source_rgba.clone().map(FacePhoto::Source)),
        }
    }

    /// The faces found in `photo` with `model`, if those are the ones kept.
    pub(crate) fn found_for(&self, photo: &FacePhoto, model: prunr_models::ModelId) -> Option<&FoundFaces> {
        self.found_faces.as_ref().filter(|f| f.model == model && f.photo.same(photo))
    }

    /// Whether a search of `photo` with `model` is under way.
    pub(crate) fn pending_for(&self, photo: &FacePhoto, model: prunr_models::ModelId) -> bool {
        self.faces_pending.as_ref().is_some_and(|(p, m)| *m == model && p.same(photo))
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
    pub(crate) fn run_input(&self, engine: Engine) -> Option<Arc<image::RgbaImage>> {
        self.run_start(engine).and_then(|s| s.image.load())
    }

    /// What an upscale or colorize run of `engine` reads: where its
    /// timeline says it starts, else the decoded original, else the
    /// source for the run's thread to decode. Shared via `Arc`: the full
    /// RGBA would otherwise be deep-cloned (tens of MB at 4K) per run.
    pub(crate) fn run_input_or_source(&self, engine: Engine) -> super::processor::RunInput {
        self.run_input(engine).map_or_else(|| self.original_run_input(), super::processor::RunInput::Decoded)
    }

    /// The decoded original, else the source for the run's thread to
    /// decode.
    pub(crate) fn original_run_input(&self) -> super::processor::RunInput {
        self.source_rgba.clone()
            .map_or_else(|| super::processor::RunInput::Source(self.source.clone()), super::processor::RunInput::Decoded)
    }

    /// The size of the image a run of `engine` starts from.
    pub(crate) fn run_input_dims(&self, engine: Engine) -> (u32, u32) {
        self.run_start(engine).map_or(self.dimensions, |s| s.dims)
    }

    /// Whether a run of `engine` starts from an earlier result rather than
    /// the original; cheap, for recipes and labels.
    pub(crate) fn starts_from_result(&self, engine: Engine) -> bool {
        self.run_start(engine).is_some()
    }

    /// The pixels the awaited run starts from (`None`: the original),
    /// loaded when the run is sent.
    pub(crate) fn job_input(&self) -> Option<Arc<image::RgbaImage>> {
        self.job.as_ref()?.start.as_ref()?.image.load()
    }

    /// What a full run on this image sends: its file and the pixels the
    /// run starts from; the image shows as processing. `None` when the
    /// file can't be read: the image fails now rather than wait on a
    /// run that never starts.
    pub(crate) fn full_run_work(&mut self) -> Option<super::worker::WorkItem> {
        let bytes = match self.source.load_bytes() {
            Ok(bytes) => bytes,
            Err(e) => {
                self.job = None;
                self.status = BatchStatus::Error(format!("Failed to load: {e}"));
                return None;
            }
        };
        let input = self.job_run_image();
        self.status = BatchStatus::Processing;
        Some((self.id, bytes, input))
    }

    /// Drop whatever caches a `CacheImpact` says are stale. Single entry
    /// point used by both the toolbar dispatcher and batch classification.
    pub(crate) fn apply_cache_impact(
        &mut self,
        impact: crate::gui::knob_catalog::CacheImpact,
    ) {
        use crate::gui::knob_catalog::CacheImpact;
        match impact {
            CacheImpact::Nothing => {}
            CacheImpact::EdgeCache => self.invalidate_edge_cache(),
            CacheImpact::SegCache => self.set_cached_tensor(None),
            CacheImpact::Both => {
                self.set_cached_tensor(None);
                self.invalidate_edge_cache();
            }
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
        if self.starts_from_result(Engine::Segment) {
            if !self.redoes_shown_step(Engine::Segment) {
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
            // An edge rerun through the subprocess isn't wired for batches;
            // the live preview handles it in-process.
            RequiredTier::MaskRerun
            | RequiredTier::AddEdgeInference
            | RequiredTier::EdgeRerun
            | RequiredTier::FullPipeline => RunTier::Full,
        }
    }

    /// Reset all caches tied to the current result. Call after the result
    /// has changed (history walk, fresh process, etc.) so the next paint
    /// rebuilds textures and the next reprocess re-runs from scratch.
    /// Note: `source_texture` is NOT cleared — callers decide whether the
    /// source view also needs rebuilding (undo: yes; redo: no).
    pub(crate) fn reset_result_caches(&mut self) {
        // Do NOT clear `cached_tensor` here. The seg tensor is a function of
        // the source image + model, so it stays valid across undo/redo and
        // across Tier 2 / AddEdge reruns that don't return a fresh tensor.
        // Callers that actually invalidate the tensor (model swap, crash
        // retry) set `cached_tensor = None` explicitly.
        //
        // Same rule for the kept upscale run: Tier-2 knobs (sharpen /
        // ai_blend / color_match) re-finish FROM it, so clearing it here
        // would force a full re-upscale on every Tier-2 tweak. Tier-1
        // invalidation paths call `invalidate_upscale_cache()` explicitly.
        self.result_texture = None;
        self.thumb_texture = None;
        self.thumb_pending = false;
        self.source_tex_pending = false;
        self.result_tex_pending = false;
        self.decode_pending = false;
    }

    /// Apply a finished tier result to this item (success or error branch).
    /// Returns the `active_provider` string when the caller should update the
    /// app-level backend label; `None` for Tier 2 reruns (empty provider) or
    /// error results.
    ///
    /// Keeps the BatchItem mutation isolated here; the backend update on
    /// `Settings` is applied in the caller via the returned `Option<String>`.
    pub(crate) fn apply_tier_result(
        &mut self,
        result: Result<prunr_core::ProcessResult, String>,
        tensor_cache: Option<super::worker::TensorCache>,
        edge_cache: Option<super::worker::EdgeTensorCache>,
        recipe_snapshot: prunr_core::ProcessingRecipe,
        _is_selected: bool,
    ) -> Option<String> {
        match result {
            Ok(pr) => {
                self.reset_result_caches();
                self.unpark_result();
                self.result_rgba = Some(Arc::new(pr.rgba_image));
                self.status = BatchStatus::Done;
                self.applied_recipe = Some(recipe_snapshot);
                // Preserve existing cache when the worker returned without a
                // fresh tensor (Tier 2 RePostProcess / AddEdgeInference for
                // the seg side). Clobbering it here silently killed live
                // preview after any tier-2 result — the next gamma tweak had
                // nothing to postprocess from.
                if let Some(new) = tensor_cache.and_then(super::worker::CompressedTensor::from_raw) {
                    self.set_cached_tensor(Some(new));
                }
                if let Some(new) = edge_cache.and_then(super::worker::CompressedEdgeTensors::from_raw) {
                    self.cached_edge_tensors = Some(new);
                    self.volatile_edge_tensor = None;
                    self.edge_planes = Default::default();
                }
                self.cached_masked_base = None;
                // Note: we used to null `source_rgba` / `source_texture` on
                // non-selected items here to save ~48 MB per 4K image, but
                // that broke live preview on any item that was NOT the
                // viewed item when its batch result landed. `source_rgba` is
                // required for the in-process Tier 2 rerun (see
                // `build_preview_inputs` → `rgba = item.source_rgba.as_ref()?`)
                // and without it, tweaking a slider on a previously-processed-
                // but-not-yet-reviewed image would silently drop the tweak
                // until the async re-decode from disk landed. Memory-pressure
                // eviction is handled separately by `evict_all_tensors` /
                // `enforce_tensor_budget`, which do preserve the selected
                // item's cache.
                // Tier 2 reruns report empty active_provider (no inference ran).
                (!pr.active_provider.is_empty()).then_some(pr.active_provider)
            }
            Err(e) => {
                // Clear recipe + tensors so retry runs a fresh Tier 1
                // (otherwise resolve_tier might return Skip for an errored item).
                self.status = BatchStatus::Error(e);
                self.set_cached_tensor(None);
                self.invalidate_edge_cache();
                self.invalidate_upscale_cache();
                self.invalidate_colorize_cache();
                self.applied_recipe = None;
                None
            }
        }
    }

    /// True when the item carries a prior processing result.
    pub(crate) fn has_result(&self) -> bool {
        self.result_rgba.is_some()
    }

    /// A stroke re-cuts this item from its tensor only while a cut-out is
    /// on screen. After undoing back to the original the tensor stays
    /// cached, and a re-cut there would bring the undone result back.
    pub(crate) fn shows_cutout(&self) -> bool {
        self.status == BatchStatus::Done && self.cached_tensor.is_some()
    }

    /// The cached segmentation is what made the shown step: after an undo
    /// to a step another model made (or an upscale), a re-cut from it would
    /// rewrite that step with the wrong model's cut-out.
    pub(crate) fn tensor_made_shown_step(&self) -> bool {
        let Some(tensor) = &self.cached_tensor else { return false };
        match self.shown_cutter() {
            Some(cutter) => cutter == Cutter::Model(tensor.model),
            None => self.applied_recipe.as_ref().is_some_and(|r| r.inference.model == tensor.model),
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
        if self.status != BatchStatus::Done {
            return false;
        }
        let Some(rgba) = self.result_rgba.as_ref() else { return false };
        if !self.timeline.current().result.as_ref().is_some_and(|r| r.holds(rgba)) {
            self.amend();
        }
        self.result_rgba = None;
        true
    }

    /// Bring a parked result back from the timeline. Every result reader
    /// and writer calls this first; a result that landed meanwhile wins.
    pub(crate) fn unpark_result(&mut self) {
        if self.result_parked() {
            self.result_rgba = self.timeline.current().result.as_ref().and_then(|r| r.load());
        }
    }

    /// The result to export, parked or not; nothing is unpacked here.
    pub(crate) fn result_ref(&self) -> Option<ResultRef> {
        self.result_rgba.clone().map(ResultRef::Shown)
            .or_else(|| self.timeline.current().result.clone().map(ResultRef::Parked))
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

    /// The decoded segmentation tensor for live preview, decoded on the
    /// first call of a drag and shared by the following ticks.
    pub(crate) fn hot_seg_tensor(&mut self) -> Option<Arc<super::live_preview::SegTensor>> {
        let ct = self.cached_tensor.as_ref()?;
        // Only the step this segmentation cut is re-cut with it: another
        // model's cut is that model's, and No model's pass already carries
        // its cut in its start.
        if self.shown_cutter().is_some_and(|c| c != Cutter::Model(ct.model)) {
            return None;
        }
        if self.volatile_seg_tensor.is_none() {
            self.volatile_seg_tensor = super::live_preview::decompress_seg(ct).map(Arc::new);
        }
        self.volatile_seg_tensor.clone()
    }

    /// Test-only constructor: returns a BatchItem with all fields at their
    /// zero/None default. Use in tests that need a real BatchItem without
    /// building a full batch pipeline. Not compiled into release binaries.
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
        Self {
            id,
            filename,
            source,
            dimensions,
            source_rgba: None,
            source_dyn: None,
            source_texture: None,
            thumb_texture: None,
            thumb_pending: false,
            result_rgba: None,
            start_dyn_cache: None,
            result_texture: None,
            source_tex_pending: false,
            result_tex_pending: false,
            reveal: None,
            subject: None,
            tile_preview: None,
            warm_steps: Arc::default(),
            on_screen: false,
            found_faces: None,
            faces_pending: None,
            faced_photo: None,
            decode_pending: false,
            timeline: Timeline::new(DocState::original(settings, applied_preset.clone())),
            job: None,
            status: BatchStatus::Pending,
            selected: false,
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
            backdrop_tex_pending: None,
            selection_mask: None,
            selection_hash: None,
            edits: None,
            correction_cache: std::sync::Mutex::default(),
            selection_tex_pending: None,
            selection_texture: None,
            magic_brush_embedding: None,
            last_decode: None,
            magic_preencode_refused: false,
            depth: None,
            depth_run: DepthRun::default(),
            depth_photo: None,
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
        self.bg_image = Some(Arc::new(BgImage {
            source_path,
            image: Arc::new(img),
            hash,
        }));
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

    /// Resolve the RGBA source for an inpaint stroke. Stack-based inpaint
    /// runs each stroke against the previous result (if any), so:
    ///   1. result_rgba (most-recent processed result)
    ///   2. source_rgba (original decoded image)
    ///   3. source_dyn  (lazy decode fallback under memory pressure)
    ///
    /// Returns None when no decoded source is available — caller should warn
    /// and skip the dispatch.
    pub(crate) fn source_for_inpaint(&self) -> Option<Arc<image::RgbaImage>> {
        self.result_rgba.as_ref().cloned()
            .or_else(|| self.source_rgba.as_ref().cloned())
            .or_else(|| self.source_dyn.as_ref().map(|d| Arc::new(d.to_rgba8())))
    }

    /// The depth map to save beside this item's result.
    pub(crate) fn depth_export(&self) -> DepthExport {
        DepthExport { name: depth_map_name(&self.filename), dims: self.dimensions, depth: self.depth_source() }
    }

    /// See `DepthSource`.
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

    /// See `BackdropPhoto`.
    pub(crate) fn backdrop_photo(&self) -> Option<BackdropPhoto> {
        let edited = self.timeline.latest_run().and_then(|o| o.start.as_ref());
        match edited {
            Some(start) => Some(BackdropPhoto::Edited(Arc::clone(&start.image))),
            None => self.source_rgba.clone().map(BackdropPhoto::Original),
        }
    }

    /// What the background texture is built from: the picture or effect
    /// and, for an effect, which photo it is made from.
    pub(crate) fn backdrop_key(&self) -> Option<u64> {
        use super::item_settings::Backdrop;
        let backdrop = self.settings.backdrop();
        let key = backdrop.picture_key()?;
        let key = match (backdrop, self.backdrop_photo()) {
            (Backdrop::Effect(_), Some(BackdropPhoto::Edited(start))) => key ^ (Arc::as_ptr(&start) as u64).rotate_left(17),
            _ => key,
        };
        // A lens blur is made with the depth map, focused on the result's subject.
        Some(match backdrop {
            Backdrop::Effect(effect) if effect.needs_depth() => {
                let depth = self.depth.as_ref().map_or(0, |d| Arc::as_ptr(d) as u64);
                let focus = self.lens_focus.as_ref().map_or(0, |f| u64::from(f.value.to_bits()));
                key ^ depth.rotate_left(29) ^ focus.rotate_left(43)
            }
            _ => key,
        })
    }

    /// Measure the lens-blur focus again when the depth map or the result
    /// is another one; a result with the subject where it was gives the
    /// same focus, so `backdrop_key` stays and nothing is rebuilt.
    pub(crate) fn refresh_lens_focus(&mut self) {
        let Some(depth) = &self.depth else {
            self.lens_focus = None;
            return;
        };
        if self.lens_focus.as_ref().is_some_and(|f| f.is_for(depth, self.result_rgba.as_ref())) {
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
        BatchItem::new(
            id,
            "test.png".to_string(),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (100, 100),
            ItemSettings::default(),
            String::new(),
        )
    }

    /// A landed upscale reveals over its finished-tile preview, which
    /// stays on screen until the result's texture lands; a new run never
    /// shows the last run's tiles.
    #[test]
    fn an_image_is_running_while_a_model_works_on_it() {
        let mut item = fixture_item(1);
        assert!(!item.is_running());
        item.start_job(Some(StepKind::Process), Engine::Segment);
        assert!(!item.is_running(), "a queued cut-out waits");
        item.status = BatchStatus::Processing;
        assert!(item.is_running(), "the worker has taken it");
        for engine in [Engine::Upscale, Engine::Colorize, Engine::Erase] {
            let mut item = fixture_item(2);
            item.status = BatchStatus::Done;
            item.start_job(None, engine);
            assert!(item.is_running(), "{engine:?} runs from its start");
        }
    }

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

    /// After an undo, the step a redo goes to is kept unpacked too.
    #[test]
    fn the_step_redo_goes_to_stays_unpacked() {
        let mut item = fixture_item(1);
        item.on_screen = true;
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]))));
        item.commit(StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]))));
        item.commit(StepKind::Adjust);
        item.undo_step();
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
        use crate::gui::timeline::{Origin, RunStart, Stored};
        let mut item = fixture_item(1);
        item.source_rgba = Some(Arc::new(image::RgbaImage::from_pixel(4, 2, image::Rgba([250, 250, 250, 255]))));
        let erased = Stored::new(Arc::new(image::RgbaImage::from_pixel(4, 2, image::Rgba([10, 10, 10, 255]))));
        item.timeline.current_mut().origin = Some(Origin::new(Engine::Segment, Some(RunStart { image: erased, dims: (4, 2) })));
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
        item.upscale = Some(UpscaleCache::new(img.clone(), img.clone(), Some(img)));
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
        item.undo_step();
        assert!(item.kept_colorize().is_none(), "an earlier colorize step: the kept run is not its own");
        item.redo_step();
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
        item.undo_step();
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
        item.undo_step();
        assert!(item.kept_upscale().is_none(), "an earlier upscale step: the kept run is not its own");
        item.redo_step();
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
        assert!(item.face_photo(FaceStage::Photo).is_some_and(|p| p.same(&FacePhoto::Source(Arc::clone(&source)))));
        let raw = Arc::new(image::RgbaImage::new(16, 16));
        item.start_job(Some(StepKind::Upscale), Engine::Upscale);
        item.result_rgba = Some(Arc::clone(&raw));
        item.land_upscale(UpscaleCache::new(Arc::clone(&raw), Arc::clone(&source), None));
        assert!(item.face_photo(FaceStage::Upscale).is_some_and(|p| p.same(&FacePhoto::Upscaled(Arc::clone(&raw)))));
        item.undo_step();
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
        item.undo_step();
        assert_eq!(item.shown_seed(), Some(7));
        item.undo_step();
        assert_eq!(item.shown_seed(), None, "the original has no fill");
        item.redo_step();
        assert_eq!(item.shown_seed(), Some(7));
    }

    #[test]
    fn invalidate_edge_cache_clears_both_atomically() {
        let mut item = fixture_item(1);
        // Simulate populated edge caches (minimal placeholder structs).
        item.edge_planes.store_placeholder();
        // (cached_edge_tensors would need a real CompressedEdgeTensors — leave None
        // here; the method should still run cleanly and clear the planes.)
        assert!(!item.edge_planes.is_empty());
        item.invalidate_edge_cache();
        assert!(item.cached_edge_tensors.is_none());
        assert!(item.edge_planes.is_empty());
    }

    #[test]
    fn reset_result_caches_clears_display_fields_only() {
        // reset_result_caches owns display/texture cleanup. It must NOT
        // touch cached_tensor — that would kill live preview after any
        // Tier 2 rerun (the tier-2 worker doesn't return a fresh tensor).
        let mut item = fixture_item(1);
        item.thumb_pending = true;
        item.source_tex_pending = true;
        item.result_tex_pending = true;
        item.decode_pending = true;
        item.reset_result_caches();
        assert!(item.result_texture.is_none());
        assert!(item.thumb_texture.is_none());
        assert!(!item.thumb_pending);
        assert!(!item.source_tex_pending);
        assert!(!item.result_tex_pending);
        assert!(!item.decode_pending);
    }

    /// Each kind of settings change asks for the run it needs: none for a
    /// backdrop or Refine change, the mask side from the cached tensor,
    /// the lines onto it, or everything — and everything when there is no
    /// tensor to start from.
    #[test]
    fn a_process_run_is_as_small_as_the_change_allows() {
        use crate::gui::worker::{CompressedTensor, TensorCache};
        use prunr_core::{LineMode, ModelKind::Silueta};
        fn processed(with_tensor: bool, before: fn(&mut ItemSettings)) -> BatchItem {
            let mut item = fixture_item(1);
            before(&mut item.settings);
            item.applied_recipe = Some(item.settings.current_recipe(Silueta));
            if with_tensor {
                item.set_cached_tensor(CompressedTensor::from_raw(TensorCache { data: vec![0.5; 16], height: 4, width: 4, model: Silueta }));
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
        assert_eq!(tier(true, lines_on, |s| s.edge_thickness = 3), RunTier::Full, "an edge change runs whole in a batch");
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
        use crate::gui::background_io::TexTarget;
        let ctx = egui::Context::default();
        let tex = |name: &str| ctx.load_texture(name, egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]), Default::default());
        let mut item = fixture_item(1);
        let old = tex("old");
        item.result_texture = Some(old.clone());
        item.status = BatchStatus::Pending;
        item.show_result(Arc::new(image::RgbaImage::new(2, 2)));
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(old.id()), "the old texture stays");
        assert!(item.result_tex_pending && item.thumb_pending, "and a new one and thumbnail are on their way");
        assert_eq!(item.status, BatchStatus::Done);
        let new = tex("new");
        item.land_texture(new.clone(), TexTarget::Result);
        assert_eq!(item.result_texture.as_ref().map(|t| t.id()), Some(new.id()));
        assert!(!item.result_tex_pending);
        assert!(matches!(item.reveal, Some(Reveal::Running(..))), "the new result reveals over the old");
    }

    /// The decoded source lands on the image with what was derived from
    /// the old bytes dropped; a failed decode marks the image.
    #[test]
    fn a_decoded_source_replaces_what_was_derived_from_the_old_one() {
        let mut item = fixture_item(1);
        item.decode_pending = true;
        item.source_dyn = Some(Arc::new(image::DynamicImage::new_rgba8(1, 1)));
        assert!(item.land_decode(Ok(Arc::new(image::RgbaImage::new(2, 2)))));
        assert!(item.source_rgba.is_some() && item.source_dyn.is_none() && !item.decode_pending);
        assert!(!item.land_decode(Err("bad file".into())));
        assert_eq!(item.status, BatchStatus::Error("bad file".into()));
    }

    /// A selection overlay built for a mask the image has since replaced
    /// is dropped; the one for the shown mask is kept, and a feathered
    /// plane that comes with it is reported once.
    #[test]
    fn a_selection_overlay_lands_only_for_the_mask_shown() {
        use crate::gui::background_io::{SelectionImage, SelectionStyle, SelectionTextureResult};
        let ctx = egui::Context::default();
        let load = |image| ctx.load_texture("selection", image, egui::TextureOptions::LINEAR);
        let image = || egui::ColorImage::new([2, 2], vec![egui::Color32::BLACK; 4]);
        let style = SelectionStyle { fill_opacity: 0.5, erase: false, outline_thickness: 1.0, edge_feather_px: 2, fill: None };
        let full = |hash: u64, feathered: bool| SelectionTextureResult {
            item_id: 1,
            image: SelectionImage::Full { image: image(), feathered: feathered.then(|| make_mask(hash, 2, 2)) },
            key: (hash, style),
            shown: make_mask(hash, 2, 2),
        };
        let mut item = fixture_item(1);
        item.selection_mask = Some(make_mask(7, 2, 2));
        item.selection_hash = Some(7);
        item.selection_tex_pending = Some((3, style));
        assert!(!item.land_selection_texture(full(3, true), load));
        assert!(item.selection_texture.is_none() && item.selection_tex_pending.is_none(), "a stale build is dropped and frees the slot");
        assert!(item.land_selection_texture(full(7, true), load), "the shown mask's overlay lands with a new feathered plane");
        assert!(!item.land_selection_texture(full(7, true), load), "a rebuild keeps the plane it had");
    }

    /// A lens-blur backdrop keys on where it focuses: a new result with
    /// the subject where it was keeps the key, one with the subject at
    /// another depth changes it, and the cache holds no result alive.
    #[test]
    fn a_lens_backdrop_keys_on_its_focus_not_the_result_handle() {
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
        item.result_rgba = Some(cutout([255, 0]));
        item.refresh_lens_focus();
        let near = item.backdrop_key();
        item.result_rgba = Some(cutout([255, 0]));
        item.refresh_lens_focus();
        assert_eq!(item.backdrop_key(), near, "a new result with the subject where it was keeps the key");
        let weak = item.result_rgba.as_ref().map(Arc::downgrade).expect("a result");
        item.result_rgba = Some(cutout([0, 255]));
        assert!(weak.upgrade().is_none(), "the cache does not keep the old result alive");
        item.refresh_lens_focus();
        assert_ne!(item.backdrop_key(), near, "the subject at another depth is another focus");
    }

    /// An image whose file is gone when its run would start fails there,
    /// with nothing left waiting on the run.
    #[test]
    fn a_run_on_a_missing_file_fails_instead_of_waiting() {
        let mut item = fixture_item(1);
        item.source = ImageSource::Path(PathBuf::from("/nonexistent/photo.png"));
        item.start_job(Some(StepKind::Process), Engine::Segment);
        assert!(item.full_run_work().is_none());
        assert!(matches!(&item.status, BatchStatus::Error(why) if why.starts_with("Failed to load")), "{:?}", item.status);
        assert!(item.job.is_none(), "nothing is awaited");
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

    #[test]
    fn apply_tier_result_success_sets_done_and_returns_provider() {
        let mut item = fixture_item(1);
        item.status = BatchStatus::Processing;

        let provider = item.apply_tier_result(
            Ok(fixture_process_result("CUDA")),
            None,
            None,
            fixture_recipe(),
            true,
        );

        assert_eq!(provider.as_deref(), Some("CUDA"));
        assert_eq!(item.status, BatchStatus::Done);
        assert!(item.result_rgba.is_some());
        assert!(item.applied_recipe.is_some());
    }

    #[test]
    fn apply_tier_result_tier2_rerun_returns_none_for_empty_provider() {
        // Tier 2 reruns omit active_provider so the caller knows not to
        // overwrite the backend label shown in the UI.
        let mut item = fixture_item(1);
        item.status = BatchStatus::Processing;

        let provider = item.apply_tier_result(
            Ok(fixture_process_result("")),
            None,
            None,
            fixture_recipe(),
            true,
        );

        assert!(provider.is_none());
        assert_eq!(item.status, BatchStatus::Done);
    }

    #[test]
    fn apply_tier_result_success_keeps_source_when_not_selected() {
        // `source_rgba` is required for in-process live preview on any item
        // — including ones that happened to be non-selected at the moment
        // their batch result landed. The caller may not know in advance
        // which items the user will tweak next. Memory-pressure eviction is
        // handled separately via `evict_all_tensors` / `enforce_tensor_budget`
        // on the batch manager.
        let mut item = fixture_item(1);
        item.status = BatchStatus::Processing;
        item.source_rgba = Some(Arc::new(image::RgbaImage::new(1, 1)));

        let _ = item.apply_tier_result(
            Ok(fixture_process_result("CUDA")),
            None,
            None,
            fixture_recipe(),
            false, // not selected — but source_rgba should still be kept
        );

        assert!(item.source_rgba.is_some(), "source_rgba must stay for live preview");
    }

    #[test]
    fn apply_tier_result_success_keeps_source_when_selected() {
        let mut item = fixture_item(1);
        item.status = BatchStatus::Processing;
        item.source_rgba = Some(Arc::new(image::RgbaImage::new(1, 1)));

        let _ = item.apply_tier_result(
            Ok(fixture_process_result("CUDA")),
            None,
            None,
            fixture_recipe(),
            true, // selected — user is looking at it
        );

        assert!(item.source_rgba.is_some(), "source_rgba must stay populated for the selected item");
    }

    #[test]
    fn apply_tier_result_error_clears_recipe_and_tensors_for_fresh_retry() {
        let mut item = fixture_item(1);
        item.status = BatchStatus::Processing;
        item.applied_recipe = Some(fixture_recipe());

        let provider = item.apply_tier_result(
            Err("boom".to_string()),
            None,
            None,
            fixture_recipe(),
            true,
        );

        assert!(provider.is_none());
        assert!(matches!(item.status, BatchStatus::Error(ref e) if e == "boom"));
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
    fn image_source_estimated_size_for_bytes_returns_len() {
        let source = ImageSource::Bytes(Arc::new(vec![0u8; 1234]));
        assert_eq!(source.estimated_size(), 1234);
    }

    #[test]
    fn image_source_path_load_and_size_round_trip() {
        // Path is the 99%-case variant (file open / drag-drop). Write a temp
        // file, read it back via load_bytes, and verify estimated_size matches
        // file metadata.
        let payload: &[u8] = b"PRUNR-TEST-FIXTURE-CONTENTS-1234567890";
        let mut path = std::env::temp_dir();
        path.push(format!("prunr-item-test-{}.bin", std::process::id()));
        std::fs::write(&path, payload).expect("write tempfile");

        let source = ImageSource::Path(path.clone());
        let loaded = source.load_bytes().expect("Path variant must read the file");
        assert_eq!(&**loaded, payload);
        assert_eq!(source.estimated_size(), payload.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn image_source_path_estimated_size_zero_on_missing_file() {
        // Defensive: estimated_size returns 0 (not panic) when the file is
        // missing — used by AdmissionController; must never fail.
        let mut path = std::env::temp_dir();
        path.push(format!("prunr-item-missing-{}-DOES-NOT-EXIST.bin", std::process::id()));
        let source = ImageSource::Path(path);
        assert_eq!(source.estimated_size(), 0);
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

    fn shown(item: &BatchItem) -> Option<u8> {
        item.result_rgba.as_ref().map(|r| r.get_pixel(0, 0)[0])
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
        item.status = BatchStatus::Done;
        item.commit(StepKind::Process);
        item.commit_selection_mask(make_mask(5, 4, 4), StepKind::Stroke);
        item.result_rgba = Some(px(2)); // the stroke's re-cut landed
        item.amend();
        item.settings.gamma = 1.7;
        item.commit(StepKind::Adjust);

        assert_eq!(item.undo_step(), Some(StepKind::Adjust));
        assert_eq!((shown(&item), item.settings.gamma), (Some(2), ItemSettings::default().gamma));
        assert_eq!(item.undo_step(), Some(StepKind::Stroke));
        assert_eq!(shown(&item), Some(1), "the image before the stroke, stored, not re-cut");
        assert!(item.selection_mask.is_none() && item.selection_hash.is_none());
        assert_eq!(item.undo_step(), Some(StepKind::Process));
        assert_eq!((shown(&item), &item.status), (None, &BatchStatus::Pending));
        assert_eq!(item.undo_step(), None);

        assert_eq!(item.redo_step(), Some(StepKind::Process));
        assert_eq!(item.redo_step(), Some(StepKind::Stroke));
        assert_eq!(item.selection_hash, Some(make_mask(5, 4, 4).content_hash()));
        assert_eq!(shown(&item), Some(2));
        assert_eq!(item.redo_step(), Some(StepKind::Adjust));
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
        item.status = BatchStatus::Done;
        item.finish_job();
        assert_eq!(item.timeline.undo_steps(), 1);

        item.start_job(None, Engine::Segment);
        item.result_rgba = Some(px(2));
        item.finish_job();
        assert_eq!(item.timeline.undo_steps(), 1, "a re-run fills in the shown step");

        item.start_job(Some(StepKind::Process), Engine::Segment);
        item.undo_step();
        assert!(!item.job_is_current(), "an undo since the start makes the result stale");
    }

    /// Lands a run of `engine` that produced `v`, recording `step`.
    fn land(item: &mut BatchItem, engine: Engine, step: Option<StepKind>, v: u8) -> Option<u8> {
        item.start_job(step, engine);
        let base = item.job_input().map(|b| b.get_pixel(0, 0).0[0]);
        item.result_rgba = Some(px(v));
        item.status = BatchStatus::Done;
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

        item.undo_step();
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(6), "undo brings back the step's own start");
        assert!(item.starts_from_result(Engine::Segment));
        assert_eq!(item.run_input_dims(Engine::Upscale), (2, 2));
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
        item.chain_start_built(px(7));
        item.result_rgba = Some(px(2));
        item.finish_job();
        assert_eq!(item.run_input(Engine::Segment).map(|b| b.get_pixel(0, 0).0[0]), Some(7), "a re-cut redoes it from the built start");
        assert!(!item.cuts_anothers_cutout(birefnet));
        assert!(item.cuts_anothers_cutout(silueta), "and Silueta's Process would build on BiRefNet's");
        item.undo_step();
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

    /// Parking releases the working image without a step; the timeline
    /// brings it back.
    #[test]
    fn a_parked_result_comes_back_from_the_timeline() {
        let mut item = fixture_item(1);
        item.result_rgba = Some(px(3));
        item.status = BatchStatus::Done;
        item.commit(StepKind::Process);
        assert!(item.park_result());
        assert!(item.result_rgba.is_none());
        item.start_job(None, Engine::Segment);
        item.abandon_job();
        assert_eq!(item.status, BatchStatus::Done, "a parked result still counts as shown");
        item.unpark_result();
        assert_eq!((shown(&item), item.timeline.undo_steps()), (Some(3), 1));

        item.undo_step();
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
    fn default_batch_item_has_no_selection() {
        let item = fixture_item(1);
        assert!(item.selection_mask.is_none());
        assert!(item.selection_hash.is_none());
        assert!(item.selection_texture.is_none());
        assert!(item.magic_brush_embedding.is_none());
    }

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
