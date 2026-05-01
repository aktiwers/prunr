//! Pure data types for batch items and the per-image history they carry.
//!
//! These types are GUI-agnostic in spirit (they hold egui texture handles
//! because the texture lifecycle is per-item, but no rendering happens here).
//! Logic that mutates these types lives in the coordinators
//! (`HistoryManager`, `BatchManager`, `Processor`).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

/// Cap on `stroke_undo_stack` / `stroke_redo_stack` length. Each entry
/// is `Option<Arc<MaskCorrection>>` (one refcount bump), so the cost is
/// negligible — the cap exists to bound worst-case memory if a session
/// runs into the thousands of strokes.
const STROKE_HISTORY_DEPTH: usize = 50;

fn push_stroke_bounded(
    stack: &mut VecDeque<Option<Arc<prunr_core::brush::MaskCorrection>>>,
    snap: Option<Arc<prunr_core::brush::MaskCorrection>>,
) {
    stack.push_back(snap);
    while stack.len() > STROKE_HISTORY_DEPTH {
        stack.pop_front();
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

/// Three-tiered history entry:
/// - Tier 1 (Hot): Raw `Arc<RgbaImage>` — instant access, full RAM cost.
/// - Tier 2 (Warm): Zstd-compressed bytes in RAM — ~3-4x smaller, ~8ms decompress.
/// - Tier 3 (Cold): Zstd file on disk — zero RAM cost, ~50-100ms read.
pub(crate) enum HistorySlot {
    /// Tier 1: uncompressed RGBA in RAM.
    InMemory(Arc<image::RgbaImage>),
    /// Tier 2: zstd-compressed in RAM (~3-4x smaller).
    Compressed(super::history_disk::CompressedEntry),
    /// Tier 3: zstd file on disk.
    OnDisk(super::history_disk::DiskHistoryEntry),
}

impl HistorySlot {
    /// Compress an RGBA image to RAM (Tier 2), falling back to uncompressed (Tier 1).
    pub(crate) fn compress(rgba: Arc<image::RgbaImage>) -> Self {
        super::history_disk::compress_to_ram(&rgba)
            .map(Self::Compressed)
            .unwrap_or(Self::InMemory(rgba))
    }

    /// Demote this slot to disk (Tier 3). Only affects Tier 1/2; Tier 3 is a no-op.
    pub(crate) fn demote_to_disk(self, item_id: u64, seq: usize) -> Self {
        match self {
            Self::InMemory(rgba) => {
                super::history_disk::write_history(item_id, seq, &rgba)
                    .map(Self::OnDisk)
                    .unwrap_or(Self::InMemory(rgba))
            }
            Self::Compressed(entry) => {
                super::history_disk::demote_to_disk(&entry, item_id, seq)
                    .map(Self::OnDisk)
                    .unwrap_or(Self::Compressed(entry))
            }
            Self::OnDisk(_) => self,
        }
    }

    /// Materialise the RGBA image from any tier.
    /// Deletes the backing file only on successful disk read.
    pub(crate) fn into_rgba(self) -> Option<Arc<image::RgbaImage>> {
        match self {
            Self::InMemory(rgba) => Some(rgba),
            Self::Compressed(entry) => {
                super::history_disk::decompress_from_ram(&entry)
                    .ok()
                    .map(Arc::new)
            }
            Self::OnDisk(entry) => match super::history_disk::read_history(&entry) {
                Ok(img) => {
                    super::history_disk::delete_entry(&entry);
                    Some(Arc::new(img))
                }
                Err(_) => None,
            },
        }
    }

    /// Delete the backing disk file if Tier 3 (no-op for Tier 1/2).
    pub(crate) fn cleanup(&self) {
        if let Self::OnDisk(entry) = self {
            super::history_disk::delete_entry(entry);
        }
    }
}

impl Default for HistorySlot {
    fn default() -> Self {
        Self::InMemory(Arc::new(image::RgbaImage::new(1, 1)))
    }
}

/// A history entry: image data + the recipe that produced it.
#[derive(Default)]
pub(crate) struct HistoryEntry {
    pub(crate) slot: HistorySlot,
    pub(crate) recipe: Option<prunr_core::ProcessingRecipe>,
}

impl HistoryEntry {
    pub(crate) fn new(rgba: Arc<image::RgbaImage>, recipe: Option<prunr_core::ProcessingRecipe>) -> Self {
        Self { slot: HistorySlot::compress(rgba), recipe }
    }

    pub(crate) fn cleanup(&self) {
        self.slot.cleanup();
    }

    pub(crate) fn demote_to_disk(self, item_id: u64, seq: usize) -> Self {
        Self { slot: self.slot.demote_to_disk(item_id, seq), recipe: self.recipe }
    }

    pub(crate) fn into_parts(self) -> (HistorySlot, Option<prunr_core::ProcessingRecipe>) {
        (self.slot, self.recipe)
    }
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

/// Snapshot of everything a preset apply (or Reset All Knobs) replaces. Used
/// by Ctrl+Shift+Z / Ctrl+Shift+Y to undo an accidental preset swap without
/// touching the image-result history.
#[derive(Clone)]
pub(crate) struct PresetSnapshot {
    pub(crate) settings: super::item_settings::ItemSettings,
    pub(crate) applied_preset: String,
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
    pub(crate) result_texture: Option<egui::TextureHandle>,
    /// True while a background thread is building the source ColorImage.
    pub(crate) source_tex_pending: bool,
    /// True while a background thread is building the result ColorImage.
    pub(crate) result_tex_pending: bool,
    /// True while a background thread is decoding source bytes to RGBA.
    pub(crate) decode_pending: bool,
    /// History stack for undo: previous results + their recipes, newest last.
    pub(crate) history: VecDeque<HistoryEntry>,
    /// Redo stack: results undone, newest last. Cleared on new processing.
    pub(crate) redo_stack: VecDeque<HistoryEntry>,
    pub(crate) status: BatchStatus,
    pub(crate) selected: bool,
    /// Per-image processing settings. Edited via the adjustments toolbar.
    pub(crate) settings: super::item_settings::ItemSettings,
    /// The recipe that produced the current result_rgba. None if never processed.
    pub(crate) applied_recipe: Option<prunr_core::ProcessingRecipe>,
    /// Compressed cached tensor from Tier 1 inference (for Tier 2 mask reruns).
    pub(crate) cached_tensor: Option<super::worker::CompressedTensor>,
    /// All 4 DexiNed scales from one inference (Tier 2 edge reruns read
    /// whichever scale the user has picked without re-inferring).
    pub(crate) cached_edge_tensors: Option<super::worker::CompressedEdgeTensors>,
    /// Decompressed hot copy of the active scale. Lets a slider drag reuse
    /// the same Arc instead of paying zstd per dispatch.
    pub(crate) volatile_edge_tensor: Option<(prunr_core::EdgeScale, Arc<Vec<f32>>)>,
    /// Post-resize, pre-dilation edge mask for the (line_strength, scale) that
    /// produced it. Lets `edge_thickness` / `solid_line_color` tweaks skip the
    /// expensive tensor→mask resize. Keyed by BOTH dimensions because scale
    /// picks a different upstream tensor — a mask built from the Fine tensor
    /// must not be reused after the user switches to Bold.
    pub(crate) cached_edge_mask: Option<(Arc<image::GrayImage>, u32 /* line_strength bits */, prunr_core::EdgeScale)>,
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
    /// Preset-apply undo stack — snapshots of (settings, applied_preset)
    /// taken BEFORE each preset apply / Reset All on this image. Separate
    /// from `history` (which is the image-result stack) so Ctrl+Shift+Z
    /// rolls back an accidental preset swap without touching the pixels.
    pub(crate) preset_undo_stack: VecDeque<PresetSnapshot>,
    /// Redo counterpart — cleared on a fresh preset apply, fed by undos.
    pub(crate) preset_redo_stack: VecDeque<PresetSnapshot>,
    /// Per-item brush correction. The hash for the recipe diff lives on
    /// `settings.correction_hash` (set in lockstep by `commit_correction`
    /// / `clear_correction`); never mutate the `Arc<MaskCorrection>` from
    /// outside those mutators or the recipe-diff dispatch breaks.
    pub(crate) mask_correction: Option<Arc<prunr_core::brush::MaskCorrection>>,
    /// Snapshots of `mask_correction` taken BEFORE each stroke commit.
    /// `Ctrl+Z` while brush mode is active pops the top entry and
    /// restores the snapshot — the user undoes one stroke per press.
    /// Bounded; oldest dropped at depth limit.
    pub(crate) stroke_undo_stack: VecDeque<Option<Arc<prunr_core::brush::MaskCorrection>>>,
    pub(crate) stroke_redo_stack: VecDeque<Option<Arc<prunr_core::brush::MaskCorrection>>>,
    /// Never mutate outside `set_bg_image` / `clear_bg_image` — those are
    /// the only writers that keep `settings.bg_image_hash` in lockstep,
    /// which the recipe-diff dispatch reads to fire CompositeOnly.
    pub(crate) bg_image: Option<Arc<BgImage>>,
    pub(crate) bg_image_texture: Option<egui::TextureHandle>,
}

impl BatchItem {
    /// Clear every edge cache tier together — compressed multi-scale set,
    /// the hot decompressed tensor, and the derived pre-dilation mask. The
    /// mask is always built from the tensor, so any tensor change invalidates
    /// all of them.
    pub(crate) fn invalidate_edge_cache(&mut self) {
        self.cached_edge_tensors = None;
        self.volatile_edge_tensor = None;
        self.cached_edge_mask = None;
    }

    /// Merge brush strokes into the existing correction. Uses
    /// `Arc::make_mut` so the common single-owner case skips the full
    /// clone. The new hash lands on `settings.correction_hash` —
    /// `mask_settings()` reads it for the recipe diff.
    pub(crate) fn commit_correction(
        &mut self,
        strokes: prunr_core::brush::MaskCorrection,
    ) {
        let pre = self.mask_correction.clone();
        push_stroke_bounded(&mut self.stroke_undo_stack, pre);
        self.stroke_redo_stack.clear();

        // Discard a stale correction whose dimensions no longer match the
        // active model (or a previous code-path bug). Without this, `merge`
        // silently drops the new strokes and the user sees the brush "do
        // nothing" — the undo stack still captures the pre-state, so the
        // reset is reversible.
        if self.mask_correction.as_ref().is_some_and(|c|
            c.width != strokes.width || c.height != strokes.height
        ) {
            tracing::info!(
                old = ?self.mask_correction.as_ref().map(|c| (c.width, c.height)),
                new = ?(strokes.width, strokes.height),
                "discarding stale correction with mismatched dims",
            );
            self.mask_correction = None;
        }

        let arc = self.mask_correction.get_or_insert_with(|| {
            Arc::new(prunr_core::brush::MaskCorrection::empty(strokes.width, strokes.height))
        });
        let current = Arc::make_mut(arc);
        prunr_core::brush::merge(current, &strokes);
        self.settings.correction_hash = Some(prunr_core::brush::content_hash(current));
    }

    /// Drop the brush correction. Reversible via `undo_stroke`.
    pub(crate) fn clear_correction(&mut self) {
        if self.mask_correction.is_some() {
            let pre = self.mask_correction.clone();
            push_stroke_bounded(&mut self.stroke_undo_stack, pre);
            self.stroke_redo_stack.clear();
        }
        self.mask_correction = None;
        self.settings.correction_hash = None;
    }

    /// Pop the last stroke snapshot, push the current state onto the
    /// redo stack, and apply the snapshot. Returns `true` if anything
    /// changed (caller invalidates result caches and re-dispatches).
    pub(crate) fn undo_stroke(&mut self) -> bool {
        let Some(prev) = self.stroke_undo_stack.pop_back() else { return false };
        let current = self.mask_correction.clone();
        push_stroke_bounded(&mut self.stroke_redo_stack, current);
        self.set_correction(prev);
        true
    }

    /// Inverse of `undo_stroke`.
    pub(crate) fn redo_stroke(&mut self) -> bool {
        let Some(next) = self.stroke_redo_stack.pop_back() else { return false };
        let current = self.mask_correction.clone();
        push_stroke_bounded(&mut self.stroke_undo_stack, current);
        self.set_correction(next);
        true
    }

    pub(crate) fn has_stroke_undo(&self) -> bool {
        !self.stroke_undo_stack.is_empty()
    }

    pub(crate) fn has_stroke_redo(&self) -> bool {
        !self.stroke_redo_stack.is_empty()
    }

    fn set_correction(&mut self, c: Option<Arc<prunr_core::brush::MaskCorrection>>) {
        self.settings.correction_hash = c.as_deref().map(prunr_core::brush::content_hash);
        self.mask_correction = c;
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
            CacheImpact::SegCache => self.cached_tensor = None,
            CacheImpact::Both => {
                self.cached_tensor = None;
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
                self.result_rgba = Some(Arc::new(pr.rgba_image));
                self.status = BatchStatus::Done;
                self.applied_recipe = Some(recipe_snapshot);
                // Preserve existing cache when the worker returned without a
                // fresh tensor (Tier 2 RePostProcess / AddEdgeInference for
                // the seg side). Clobbering it here silently killed live
                // preview after any tier-2 result — the next gamma tweak had
                // nothing to postprocess from.
                if let Some(new) = tensor_cache.and_then(super::worker::CompressedTensor::from_raw) {
                    self.cached_tensor = Some(new);
                }
                if let Some(new) = edge_cache.and_then(super::worker::CompressedEdgeTensors::from_raw) {
                    self.cached_edge_tensors = Some(new);
                    self.volatile_edge_tensor = None;
                    self.cached_edge_mask = None;
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
                self.cached_tensor = None;
                self.invalidate_edge_cache();
                self.applied_recipe = None;
                None
            }
        }
    }

    /// Combined compressed size of segmentation + edge tensor caches.
    /// Used by memory governance (`BatchManager::enforce_tensor_budget`)
    /// and any future telemetry / HUD readout.
    pub(crate) fn cache_size(&self) -> usize {
        let seg = self.cached_tensor.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
        let edge = self.cached_edge_tensors.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
        seg + edge
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
            result_texture: None,
            source_tex_pending: false,
            result_tex_pending: false,
            decode_pending: false,
            history: VecDeque::new(),
            redo_stack: VecDeque::new(),
            status: BatchStatus::Pending,
            selected: false,
            settings,
            applied_recipe: None,
            cached_tensor: None,
            cached_edge_tensors: None,
            volatile_edge_tensor: None,
            cached_edge_mask: None,
            cached_masked_base: None,
            applied_preset,
            preset_undo_stack: VecDeque::new(),
            preset_redo_stack: VecDeque::new(),
            mask_correction: None,
            stroke_undo_stack: VecDeque::new(),
            stroke_redo_stack: VecDeque::new(),
            bg_image: None,
            bg_image_texture: None,
        }
    }

    /// Lockstep writer for `bg_image` + `settings.bg_image_hash`. Going
    /// through this path is the invariant the recipe diff relies on.
    pub(crate) fn set_bg_image(&mut self, img: image::DynamicImage, source_path: Option<PathBuf>) {
        let hash = bg_image_content_hash(&img);
        self.bg_image = Some(Arc::new(BgImage {
            source_path,
            image: Arc::new(img),
            hash,
        }));
        self.bg_image_texture = None;
        self.settings.bg_image_hash = Some(hash);
    }

    pub(crate) fn clear_bg_image(&mut self) {
        self.bg_image = None;
        self.bg_image_texture = None;
        self.settings.bg_image_hash = None;
    }

    /// Bake the per-item background into a result image for save / clipboard /
    /// drag-out. Image bg wins over color bg (matches the canvas-paint rule).
    /// Returns the cloned Arc unchanged when neither is set.
    pub(crate) fn bake_export_bg(
        &self,
        rgba: &Arc<image::RgbaImage>,
    ) -> Arc<image::RgbaImage> {
        if let Some(bg) = self.bg_image.as_ref() {
            let mut copy = (**rgba).clone();
            prunr_core::apply_background_image(&mut copy, &bg.image, self.settings.bg_image_fit);
            Arc::new(copy)
        } else if let Some(c) = self.settings.bg_rgb() {
            let mut copy = (**rgba).clone();
            prunr_core::apply_background_color(&mut copy, c);
            Arc::new(copy)
        } else {
            rgba.clone()
        }
    }

    /// Build the egui texture for `bg_image` on demand. Cheap on the
    /// frames after the first — `Option::is_some` short-circuit only.
    pub(crate) fn ensure_bg_image_texture(&mut self, ctx: &egui::Context) {
        if self.bg_image_texture.is_some() {
            return;
        }
        let Some(bg) = self.bg_image.as_ref() else { return };
        let rgba = bg.image.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [w as usize, h as usize],
            rgba.as_raw(),
        );
        // WrapMode::Repeat enables BgImageFit::Tile (UV > 1.0 wraps around).
        // For other fits the UVs stay within [0, 1], so the wrap mode is a
        // no-op. LINEAR filtering for the smooth scale modes (Cover/Contain
        // /Stretch); Tile + Center read 1:1 so filtering doesn't matter.
        let opts = egui::TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            wrap_mode: egui::TextureWrapMode::Repeat,
            mipmap_mode: None,
        };
        let tex = ctx.load_texture(
            format!("bg_image_{:x}", bg.hash),
            color_image,
            opts,
        );
        self.bg_image_texture = Some(tex);
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

    #[test]
    fn invalidate_edge_cache_clears_both_atomically() {
        let mut item = fixture_item(1);
        // Simulate populated edge caches (minimal placeholder structs).
        item.cached_edge_mask = Some((Arc::new(image::GrayImage::new(1, 1)), 0, prunr_core::EdgeScale::Fused));
        // (cached_edge_tensors would need a real CompressedEdgeTensors — leave None
        // here; the method should still run cleanly and clear cached_edge_mask.)
        assert!(item.cached_edge_mask.is_some());
        item.invalidate_edge_cache();
        assert!(item.cached_edge_tensors.is_none());
        assert!(item.cached_edge_mask.is_none());
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

    #[test]
    fn cache_size_zero_when_caches_empty() {
        let item = fixture_item(1);
        assert_eq!(item.cache_size(), 0);
    }

    fn fixture_recipe() -> prunr_core::ProcessingRecipe {
        ItemSettings::default().current_recipe(prunr_core::ModelKind::Silueta, false)
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

    #[test]
    fn history_entry_into_parts_round_trips_construction() {
        let rgba = Arc::new(image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255])));
        let entry = HistoryEntry::new(rgba.clone(), None);
        let (slot, recipe) = entry.into_parts();
        assert!(recipe.is_none());
        // Slot was compressed; rehydrate and check pixel equality.
        let recovered = slot.into_rgba().expect("compressed slot must rehydrate");
        assert_eq!(recovered.dimensions(), (2, 2));
        assert_eq!(recovered.as_raw(), rgba.as_raw());
    }

    #[test]
    fn history_slot_default_is_inmemory_one_by_one_placeholder() {
        let slot = HistorySlot::default();
        match slot {
            HistorySlot::InMemory(ref rgba) => assert_eq!(rgba.dimensions(), (1, 1)),
            HistorySlot::Compressed(_) => panic!("expected InMemory, got Compressed"),
            HistorySlot::OnDisk(_) => panic!("expected InMemory, got OnDisk"),
        }
    }

    fn stamp(width: u16, height: u16, idx: usize, _value: i8) -> prunr_core::brush::MaskCorrection {
        // Stamp a real circle near the requested cell so the correction
        // is non-empty without poking private fields. The exact magnitude
        // doesn't matter — these tests only check undo/redo bookkeeping.
        let mut c = prunr_core::brush::MaskCorrection::empty(width, height);
        let cx = (idx as u16 % width) as f32 + 0.5;
        let cy = (idx as u16 / width) as f32 + 0.5;
        prunr_core::brush::paint_circle(
            &mut c, cx, cy, 1.0,
            prunr_core::brush::Stamp { hardness: 1.0, strength: 1.0, mode: prunr_core::brush::BrushMode::Add },
        );
        c
    }

    #[test]
    fn commit_correction_pushes_pre_state_onto_undo_stack() {
        let mut item = fixture_item(1);
        assert!(!item.has_stroke_undo());
        item.commit_correction(stamp(8, 8, 5, 50));
        assert!(item.has_stroke_undo(), "first stroke must register an undo entry (pre = None)");
        assert!(!item.has_stroke_redo());
    }

    #[test]
    fn undo_stroke_restores_pre_state() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        let after_first = item.mask_correction.clone();
        item.commit_correction(stamp(8, 8, 10, 70));
        assert_ne!(item.mask_correction, after_first, "second stroke changed the grid");

        assert!(item.undo_stroke(), "stroke 2 must be undoable");
        assert_eq!(item.mask_correction, after_first, "undo restored the post-stroke-1 state");
        assert!(item.has_stroke_redo(), "undone stroke goes onto the redo stack");
    }

    #[test]
    fn redo_stroke_inverts_undo() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        item.commit_correction(stamp(8, 8, 10, 70));
        let after_two = item.mask_correction.clone();

        item.undo_stroke();
        assert!(item.redo_stroke(), "redo available after undo");
        assert_eq!(item.mask_correction, after_two);
    }

    #[test]
    fn fresh_commit_after_undo_clears_redo() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        item.commit_correction(stamp(8, 8, 10, 70));
        item.undo_stroke();
        assert!(item.has_stroke_redo());

        item.commit_correction(stamp(8, 8, 12, 30));
        assert!(!item.has_stroke_redo(), "fresh stroke after undo must wipe the redo stack");
    }

    #[test]
    fn stroke_history_caps_at_depth() {
        let mut item = fixture_item(1);
        for i in 0..(STROKE_HISTORY_DEPTH + 5) {
            item.commit_correction(stamp(8, 8, i % 64, 1 + (i % 100) as i8));
        }
        assert_eq!(
            item.stroke_undo_stack.len(),
            STROKE_HISTORY_DEPTH,
            "undo stack must cap at STROKE_HISTORY_DEPTH"
        );
    }

    #[test]
    fn clear_correction_pushes_undo_when_correction_existed() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        let undo_before_clear = item.stroke_undo_stack.len();
        item.clear_correction();
        assert!(item.mask_correction.is_none());
        assert_eq!(
            item.stroke_undo_stack.len(),
            undo_before_clear + 1,
            "clear must record an undoable snapshot of the correction it dropped"
        );
        assert!(item.undo_stroke(), "the user can undo a clear");
        assert!(item.mask_correction.is_some());
    }

    #[test]
    fn clear_correction_no_op_when_already_empty() {
        let mut item = fixture_item(1);
        item.clear_correction();
        assert!(!item.has_stroke_undo(), "clearing an already-empty correction must not push undo");
    }

    #[test]
    fn commit_correction_writes_hash_to_settings() {
        let mut item = fixture_item(1);
        assert!(item.settings.correction_hash.is_none());
        item.commit_correction(stamp(8, 8, 5, 50));
        assert!(
            item.settings.correction_hash.is_some(),
            "settings.correction_hash must mirror the new correction's hash"
        );
    }

    #[test]
    fn clear_correction_clears_settings_hash() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        assert!(item.settings.correction_hash.is_some());
        item.clear_correction();
        assert!(item.settings.correction_hash.is_none(), "clear must wipe the hash too");
    }

    #[test]
    fn undo_redo_keeps_settings_hash_in_sync() {
        let mut item = fixture_item(1);
        item.commit_correction(stamp(8, 8, 5, 50));
        let after_commit = item.settings.correction_hash;
        item.commit_correction(stamp(8, 8, 12, 30));
        let after_two = item.settings.correction_hash;
        assert_ne!(after_commit, after_two);
        item.undo_stroke();
        assert_eq!(item.settings.correction_hash, after_commit, "undo restores the prior hash");
        item.redo_stroke();
        assert_eq!(item.settings.correction_hash, after_two, "redo restores the next hash");
    }
}
