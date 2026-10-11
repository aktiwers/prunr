//! Pure data types for batch items and the undo timeline each carries.
//!
//! These types are GUI-agnostic in spirit (they hold egui texture handles
//! because the texture lifecycle is per-item, but no rendering happens here).
//! Logic that mutates these types lives in the coordinators
//! (`BatchManager`, `Processor`); undo lives in `timeline`.

use std::path::PathBuf;
use std::sync::Arc;

/// A result's background for export, baked by `bake` (in a save thread
/// where there is one: an effect blurs the full photo).
#[derive(Clone)]
pub(crate) struct ExportBg {
    backdrop: super::item_settings::Backdrop,
    image: Option<Arc<BgImage>>,
    fit: prunr_core::BgImageFit,
    photo: Option<Arc<image::RgbaImage>>,
    source: ImageSource,
}

impl ExportBg {
    pub(crate) fn bake(&self, rgba: &Arc<image::RgbaImage>) -> Arc<image::RgbaImage> {
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
            Backdrop::Effect(effect) => self.photo()
                .map(|photo| composite(&|img| { prunr_core::apply_bg_effect(img, &photo, effect); })),
        };
        baked.unwrap_or_else(|| Arc::clone(rgba))
    }

    fn photo(&self) -> Option<Arc<image::RgbaImage>> {
        self.photo.clone().or_else(|| self.source.decode().ok().map(|img| Arc::new(img.into_rgba8())))
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
    pub(crate) fn decode(&self) -> Result<image::DynamicImage, String> {
        let bytes = self.load_bytes().map_err(|e| e.to_string())?;
        prunr_core::load_image_from_bytes(&bytes).map_err(|e| e.to_string())
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
}

/// What runs a job, and so what cancelling it stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Engine {
    /// Background removal, and the filters-only pass.
    Segment,
    Erase,
    Upscale,
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
    /// Tier-1 upscale output cached for Tier-2 reprocess. Set by
    /// pump_upscale_results after a successful upscale dispatch.
    /// Mirrors `cached_tensor`'s contract: NOT cleared by
    /// `reset_result_caches()`; only cleared explicitly on Tier-1
    /// invalidation (model swap, output_scale / pre_denoise /
    /// brightness_lift change). Holds the raw model output BEFORE
    /// any postprocess (sharpen, ai_blend, saturation, color_match) —
    /// the postprocess functions are re-run on every Tier-2 dispatch.
    ///
    /// RAM impact: at 4K source × 4× upscale = 15360×8640 RGBA ≈ 500 MB.
    /// Not under the tensor budget (`evictable_tensor_bytes`). The Arc
    /// wraps the buffer so cheap clones in live-preview snapshots
    /// don't duplicate.
    pub(crate) upscale_raw: Option<Arc<image::RgbaImage>>,
    /// Bicubic resize of the original source at upscale_raw's dimensions.
    /// Used by apply_ai_blend and apply_color_match. Built lazily by the
    /// first Tier-2 dispatch that needs it; invalidated alongside upscale_raw.
    pub(crate) bicubic_source: Option<Arc<image::RgbaImage>>,
    /// All 4 DexiNed scales from one inference (Tier 2 edge reruns read
    /// whichever scale the user has picked without re-inferring).
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
    /// Never mutate outside `set_bg_image` / `clear_bg_image` — those are
    /// the only writers that keep `settings.bg_image_hash` in lockstep,
    /// which the recipe-diff dispatch reads to fire CompositeOnly.
    pub(crate) bg_image: Option<Arc<BgImage>>,
    /// The picture behind the result (a background image or an effect
    /// made from the photo) and its `Backdrop::picture_key`.
    pub(crate) backdrop_texture: Option<(u64, egui::TextureHandle)>,
    /// The key of the picture a background thread is building, so the
    /// next frame doesn't start the same one again.
    pub(crate) backdrop_tex_pending: Option<u64>,

    /// Shared selection mask. Both Paint Brush and Magic Brush write
    /// here. Signed i8 plane at source image resolution.
    ///
    /// Survives Process; an erase that lands clears it, as do Clear and a
    /// source change.
    pub(crate) selection_mask: Option<Arc<prunr_core::selection::MaskArtifact>>,

    /// Content hash of `selection_mask`, set in lockstep with the mask by
    /// `commit_selection_mask`. Half of the selection texture's key.
    pub(crate) selection_hash: Option<u64>,

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
    }

    /// The selection a re-cut applies, at full resolution: feathered when
    /// Feather is on and the overlay has built the feathered plane (off
    /// the UI thread; its landing re-cuts again). `None` without a
    /// selection or with a blank one.
    pub(crate) fn correction_plane(&self, style: super::background_io::SelectionStyle) -> Option<Arc<prunr_core::selection::MaskArtifact>> {
        let selection = self.selection_mask.as_ref().filter(|m| !m.is_blank())?;
        let feathered = self.current_selection_texture(style).and_then(|t| t.feathered.clone());
        Some(feathered.unwrap_or_else(|| Arc::clone(selection)))
    }

    /// The overlay texture, when it was built from the current selection
    /// in `style`.
    pub(crate) fn current_selection_texture(&self, style: super::background_io::SelectionStyle) -> Option<&SelectionTexture> {
        let hash = self.selection_hash?;
        self.selection_texture.as_ref().filter(|t| t.key == (hash, style))
    }

    /// Clear the cached upscale_raw buffer and the derived bicubic_source.
    /// Called on model switch and on item error. Tier-1 knob changes do NOT
    /// call this — the next dispatch's pump_upscale_results overwrites the
    /// buffer in-place, so explicit invalidation would just drop work the
    /// next dispatch is about to redo. Model switch is special because the
    /// next dispatch may never happen (user may drag a Tier-2 knob first),
    /// and the stale buffer is from the wrong model.
    pub(crate) fn invalidate_upscale_cache(&mut self) {
        self.upscale_raw = None;
        self.bicubic_source = None;
    }

    /// Clear the cached SAM encoder embedding. Called on source image
    /// change and on model switch to a non-Selection category.
    pub(crate) fn invalidate_magic_brush_embedding(&mut self) {
        self.magic_brush_embedding = None;
        self.magic_preencode_refused = false;
        self.last_decode = None;
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

    /// Clear every selection-related cache: mask, hash, texture, pending build.
    /// Called by `BatchManager::clear_selection` (Esc / Clear) and
    /// `invalidate_selection_on_source_change`.
    pub(crate) fn invalidate_selection(&mut self) {
        self.selection_mask = None;
        self.selection_hash = None;
        self.selection_texture = None;
        self.selection_tex_pending = None;
        self.last_decode = None;
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
            settings: self.settings,
            applied_preset: self.applied_preset.clone(),
            // A hand edit makes new pixels no run can redo.
            origin: if kind == StepKind::Edit { None } else { shown.origin.clone() },
            at: std::time::Instant::now(),
        }
    }

    /// Record the working copy as a new step.
    pub(crate) fn commit(&mut self, kind: StepKind) {
        self.unpark_result();
        let state = self.snapshot(kind);
        self.timeline.push(state);
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
        self.subject = None;
        let shown = self.timeline.current().clone();
        self.result_rgba = shown.result.as_ref().and_then(|r| r.load());
        self.applied_recipe = shown.recipe;
        self.selection_mask = shown.selection.as_ref().and_then(|s| s.load());
        self.selection_hash = self.selection_mask.as_ref().and(shown.selection_hash);
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
        Some(undone)
    }

    /// Step forward one state; returns what the redone step did.
    pub(crate) fn redo_step(&mut self) -> Option<StepKind> {
        self.unpark_result();
        let redone = self.timeline.redo()?;
        self.restore();
        Some(redone)
    }

    /// Note a run starting now; `step` is the step its result records,
    /// `None` to fill in the shown one (a re-run a knob change asked for).
    pub(crate) fn start_job(&mut self, step: Option<StepKind>, engine: Engine) {
        let start = self.run_start(engine);
        self.job = Some(Job { version: self.timeline.version(), step, engine, start });
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
            self.record_origin(job.engine, job.start);
        }
    }

    /// Record that `engine` made the shown step's image, starting from
    /// `start` (`None`: the original). The start is a stored timeline
    /// image, so the history sweep keeps compressing it.
    pub(crate) fn record_origin(&mut self, engine: Engine, start: Option<RunStart>) {
        self.timeline.current_mut().origin = Some(Origin { engine, start });
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
            None => self.timeline.current().result.clone().map(|image| RunStart {
                image,
                dims: self.result_rgba.as_ref().map_or(self.dimensions, |r| r.dimensions()),
            }),
        }
    }

    /// The pixels a run of `engine` starts from (`None`: the original).
    pub(crate) fn run_input(&self, engine: Engine) -> Option<Arc<image::RgbaImage>> {
        self.run_start(engine).and_then(|s| s.image.load())
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

    /// Whether the awaited run works on the result on screen, so the
    /// canvas shows that result while it runs.
    pub(crate) fn job_builds_on_shown_result(&self) -> bool {
        let shown = self.timeline.current().result.as_ref();
        self.job.as_ref().and_then(|j| j.start.as_ref())
            .zip(shown)
            .is_some_and(|(start, shown)| Arc::ptr_eq(&start.image, shown))
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
        // Same rule for `upscale_raw` and `bicubic_source`: the upscale
        // buffer is a function of (input × upscale_model × output_scale ×
        // pre_denoise × brightness_lift). Tier-2 knobs (sharpen / ai_blend /
        // saturation / color_match) reprocess FROM this cached buffer;
        // clearing it here would force a full re-upscale on every Tier-2
        // tweak — exactly the cost the cache exists to avoid. Tier-1
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
            decode_pending: false,
            timeline: Timeline::new(DocState::original(settings, applied_preset.clone())),
            job: None,
            status: BatchStatus::Pending,
            selected: false,
            settings,
            applied_recipe: None,
            cached_tensor: None,
            upscale_raw: None,
            bicubic_source: None,
            cached_edge_tensors: None,
            volatile_edge_tensor: None,
            volatile_seg_tensor: None,
            edge_planes: Default::default(),
            cached_masked_base: None,
            applied_preset,
            bg_image: None,
            backdrop_texture: None,
            backdrop_tex_pending: None,
            selection_mask: None,
            selection_hash: None,
            selection_tex_pending: None,
            selection_texture: None,
            magic_brush_embedding: None,
            last_decode: None,
            magic_preencode_refused: false,
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

    /// What export puts behind this item's result: only shared handles,
    /// so it can travel to a save thread and bake there.
    pub(crate) fn export_bg(&self) -> ExportBg {
        ExportBg {
            backdrop: self.settings.backdrop(),
            image: self.bg_image.clone(),
            fit: self.settings.bg_image_fit,
            photo: self.source_rgba.clone(),
            source: self.source.clone(),
        }
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
        let out = item.export_bg().bake(&result);
        assert_eq!(*out.get_pixel(0, 0), line, "the result's own pixels stay");
        let behind = out.get_pixel(3, 1).0;
        assert_eq!(behind[3], 255, "the transparent part gets the picture");
        assert!(behind[0] == behind[1] && behind[1] == behind[2], "a black-and-white photo: {behind:?}");

        item.settings.bg_effect = prunr_core::BgEffect::None;
        assert!(Arc::ptr_eq(&item.export_bg().bake(&result), &result), "nothing behind, nothing baked");
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
        item.record_origin(Engine::Segment, start);
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
        item.upscale_raw = Some(Arc::new(image::RgbaImage::new(4, 4)));
        assert_eq!(item.evictable_tensor_bytes(), 0, "planes and upscale buffers are not evictable");
    }

    #[test]
    fn upscale_raw_default_is_none() {
        let item = fixture_item(1);
        assert!(item.upscale_raw.is_none(), "new item must have no cached upscale buffer");
        assert!(item.bicubic_source.is_none(), "new item must have no bicubic_source");
    }

    #[test]
    fn reset_result_caches_preserves_upscale_raw() {
        // reset_result_caches clears display fields only. It must NOT clear
        // upscale_raw — the Tier-2 postprocess pipeline reads it.
        let mut item = fixture_item(1);
        let img = Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255])));
        item.upscale_raw = Some(img.clone());
        item.bicubic_source = Some(img.clone());
        item.reset_result_caches();
        assert!(item.upscale_raw.is_some(), "reset_result_caches must NOT clear upscale_raw");
        assert!(item.bicubic_source.is_some(), "reset_result_caches must NOT clear bicubic_source");
    }

    #[test]
    fn invalidate_upscale_cache_clears_field() {
        let mut item = fixture_item(1);
        let img = Arc::new(image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255])));
        item.upscale_raw = Some(img.clone());
        item.bicubic_source = Some(img.clone());
        item.invalidate_upscale_cache();
        assert!(item.upscale_raw.is_none(), "invalidate_upscale_cache must clear upscale_raw");
        assert!(item.bicubic_source.is_none(), "invalidate_upscale_cache must clear bicubic_source");
    }

    #[test]
    fn upscale_raw_arc_clone_zero_cost() {
        let mut item = fixture_item(1);
        let img = Arc::new(image::RgbaImage::new(4, 4));
        item.upscale_raw = Some(img.clone());
        // Taking a second reference — clone is free (refcount bump, no memcpy).
        let _second_ref = item.upscale_raw.as_ref().unwrap().clone();
        assert!(
            Arc::strong_count(item.upscale_raw.as_ref().unwrap()) >= 2,
            "Arc::clone of upscale_raw must be free (refcount bump only)"
        );
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
