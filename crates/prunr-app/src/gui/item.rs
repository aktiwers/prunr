//! Pure data types for batch items and the per-image history they carry.
//!
//! These types are GUI-agnostic in spirit (they hold egui texture handles
//! because the texture lifecycle is per-item, but no rendering happens here).
//! Logic that mutates these types lives in the coordinators
//! (`HistoryManager`, `BatchManager`, `Processor`).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

/// Unified cap for the action-ordering layer (`actions_undo` / `actions_redo`),
/// the stroke stacks, and the preset stacks. Each ordering-layer entry is a
/// 1-byte enum tag; stroke entries are `Option<Arc<MaskArtifact>>` (one
/// refcount bump); preset entries are ~100-byte `PresetSnapshot` structs.
/// 100 is generous enough for any realistic session while bounding worst-case
/// memory to negligible amounts.
pub(crate) const ACTION_HIST_DEPTH: usize = 100;

/// Tag identifying which per-type stack holds the pre-state for one commit in
/// the ordering layer. `handle_undo` pops from `actions_undo` and dispatches
/// to the matching stack's pop method; `handle_redo` mirrors via `actions_redo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActionType {
    /// Brush stroke commit — pre-state lives in `stroke_undo_stack`.
    Stroke,
    /// Process / inpaint dispatch result — pre-state in `history` / `redo_stack`
    /// (the result archive). Bounded by `MAX_HISTORY_DEPTH`, not
    /// `ACTION_HIST_DEPTH`; result entries are large.
    Result,
    /// Preset apply — pre-state in `preset_undo_stack` / `preset_redo_stack`.
    PresetApply,
}

/// Bounded push onto a `VecDeque<ActionType>` without clearing the opposite
/// stack. Used by the undo/redo dispatchers when an action moves between
/// stacks (same timeline, different end — no divergence).
pub(crate) fn push_action_bounded(stack: &mut VecDeque<ActionType>, kind: ActionType) {
    stack.push_back(kind);
    while stack.len() > ACTION_HIST_DEPTH {
        stack.pop_front();
    }
}


/// Per-item brush stroke history depth. Each entry is a full source-
/// resolution plane (1 B/px: ~4 MB at 2048², ~8 MB at 4K), so 32 strokes
/// cap the stack at ~0.26 GB on 4K content. Shallower than
/// `ACTION_HIST_DEPTH`: the ordering log tolerates stroke entries that
/// have already been dropped (`try_undo_one_action` pops orphans).
const STROKE_HISTORY_DEPTH: usize = 32;

/// Push a snapshot; returns `true` when the oldest one was dropped to
/// stay within `STROKE_HISTORY_DEPTH`.
fn push_stroke_bounded(
    stack: &mut VecDeque<Option<Arc<prunr_core::selection::MaskArtifact>>>,
    snap: Option<Arc<prunr_core::selection::MaskArtifact>>,
) -> bool {
    stack.push_back(snap);
    let mut dropped = false;
    while stack.len() > STROKE_HISTORY_DEPTH {
        stack.pop_front();
        dropped = true;
    }
    dropped
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
    /// Chain-mode mirror of `source_dyn`: caches a `DynamicImage` wrapping
    /// the current `result_rgba` so chain-mode live-preview dispatches
    /// don't re-clone the full RGBA buffer every tick. The tuple stores
    /// the `result_rgba` Arc that produced the cached `DynamicImage`;
    /// `Arc::ptr_eq` against the current `result_rgba` is the staleness
    /// check, so no manual invalidation is needed — every site that
    /// replaces `result_rgba` with a fresh Arc automatically misses.
    pub(crate) chain_dyn_cache: Option<(Arc<image::RgbaImage>, Arc<image::DynamicImage>)>,
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
    /// Snapshots of `selection_mask` taken BEFORE each stroke commit.
    /// `Ctrl+Z` while brush mode is active pops the top entry and
    /// restores the snapshot — the user undoes one stroke per press.
    /// Bounded to STROKE_HISTORY_DEPTH; oldest entries dropped when full.
    /// Snapshots live at source resolution so Paint Brush and Magic Brush
    /// share one undo stack.
    pub(crate) stroke_undo_stack: VecDeque<Option<Arc<prunr_core::selection::MaskArtifact>>>,
    pub(crate) stroke_redo_stack: VecDeque<Option<Arc<prunr_core::selection::MaskArtifact>>>,
    /// Ordering layer: commit-order sequence of action types. Each entry is a
    /// tag pointing at the per-type stack that holds the corresponding pre-state.
    /// `handle_undo` pops from the back (most-recent) and dispatches; new commits
    /// push here in lockstep with pushing to the per-type stack.
    pub(crate) actions_undo: VecDeque<ActionType>,
    pub(crate) actions_redo: VecDeque<ActionType>,
    /// Never mutate outside `set_bg_image` / `clear_bg_image` — those are
    /// the only writers that keep `settings.bg_image_hash` in lockstep,
    /// which the recipe-diff dispatch reads to fire CompositeOnly.
    pub(crate) bg_image: Option<Arc<BgImage>>,
    pub(crate) bg_image_texture: Option<egui::TextureHandle>,
    /// True while a background thread is building the bg-image
    /// `ColorImage` so the UI thread doesn't kick a duplicate spawn
    /// the next frame. Cleared by the drain in
    /// `drain_background_channels`.
    pub(crate) bg_image_tex_pending: bool,

    /// Shared selection mask. Both Paint Brush and Magic Brush write
    /// here. Signed i8 plane at source image resolution.
    ///
    /// LIFECYCLE: NOT cleared by `reset_result_caches()` — selection
    /// survives Process clicks intentionally. Cleared by:
    /// - `BatchManager::clear_selection` (Esc / Clear button)
    /// - `invalidate_selection_on_source_change` (source replaced)
    /// - Item destruction (image switch — per-BatchItem naturally clears)
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
}

/// A selection overlay texture and what it was built from.
pub(crate) struct SelectionTexture {
    pub(crate) key: super::background_io::SelectionTextureKey,
    pub(crate) handle: egui::TextureHandle,
}

impl BatchItem {
    /// Push a marker onto `actions_undo` and clear `actions_redo` — any new
    /// edit branches the redo timeline.
    pub(crate) fn push_action_marker(&mut self, kind: ActionType) {
        push_action_bounded(&mut self.actions_undo, kind);
        self.actions_redo.clear();
    }

    /// Clear every edge cache tier together — compressed multi-scale set,
    /// the hot decompressed tensor, and the derived pre-dilation mask. The
    /// mask is always built from the tensor, so any tensor change invalidates
    /// all of them.
    pub(crate) fn invalidate_edge_cache(&mut self) {
        self.cached_edge_tensors = None;
        self.volatile_edge_tensor = None;
        self.cached_edge_mask = None;
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
    }

    /// Clear every selection-related cache: mask, hash, outline, texture.
    /// Called by `BatchManager::clear_selection` (Esc / Clear) and
    /// `invalidate_selection_on_source_change`.
    pub(crate) fn invalidate_selection(&mut self) {
        self.selection_mask = None;
        self.selection_hash = None;
        self.selection_texture = None;
        self.selection_tex_pending = None;
    }

    /// Make `mask` the selection and register the change for undo: the
    /// previous mask is snapshotted onto the stroke stack with a Stroke
    /// marker on the ordering layer, redo is cleared. Every author — Paint
    /// strokes, Magic Brush candidates, Invert — goes through here, so all
    /// of them are undoable. Returns `false` and changes nothing when
    /// `mask` is byte-identical to the current selection: a no-op stroke
    /// must not create an undo step.
    pub(crate) fn commit_selection_mask(&mut self, mask: Arc<prunr_core::selection::MaskArtifact>) -> bool {
        let hash = mask.content_hash();
        if self.selection_hash == Some(hash) {
            return false;
        }
        let pre = self.selection_mask.replace(mask);
        self.selection_hash = Some(hash);
        if push_stroke_bounded(&mut self.stroke_undo_stack, pre) {
            // The dropped snapshot's marker would otherwise undo nothing.
            if let Some(pos) = self.actions_undo.iter().position(|a| matches!(a, ActionType::Stroke)) {
                self.actions_undo.remove(pos);
            }
        }
        self.stroke_redo_stack.clear();
        self.push_action_marker(ActionType::Stroke);
        true
    }

    /// Roll back the most-recent stroke commit as if it never happened.
    /// Used when an inpaint dispatch is cancelled before its result lands —
    /// the committed `selection_mask` was the input that would have driven
    /// the dispatch, so reverting both the state AND the matching
    /// `ActionType::Stroke` marker prevents the cancelled stroke from:
    ///   - rendering as a ghost brush overlay on canvas, OR
    ///   - accumulating into the next stroke commit, OR
    ///   - showing up as a phantom Cmd+Z entry in the action timeline.
    ///
    /// Differs from `undo_stroke`: no redo push (cancel is final, not
    /// reversible) and explicitly drops the action marker.
    pub(crate) fn revert_last_stroke_commit(&mut self) {
        if let Some(prev) = self.stroke_undo_stack.pop_back() {
            self.selection_mask = prev;
            self.selection_hash = self.selection_mask.as_ref().map(|m| m.content_hash());
        }
        // Pop the matching marker. rposition handles edge cases where
        // a non-Stroke action was pushed between the commit and the
        // cancel — unlikely with current code paths but defensive.
        if let Some(idx) = self.actions_undo.iter().rposition(
            |a| matches!(a, ActionType::Stroke),
        ) {
            self.actions_undo.remove(idx);
        }
    }

    /// Pop the last stroke snapshot, push the current state onto the
    /// redo stack, and restore the snapshot. Returns `true` if anything
    /// changed (caller invalidates caches and re-dispatches).
    pub(crate) fn undo_stroke(&mut self) -> bool {
        let Some(prev) = self.stroke_undo_stack.pop_back() else { return false };
        let current = self.selection_mask.clone();
        push_stroke_bounded(&mut self.stroke_redo_stack, current);
        self.selection_mask = prev;
        self.selection_hash = self.selection_mask.as_ref().map(|m| m.content_hash());
        true
    }

    /// Inverse of `undo_stroke`.
    pub(crate) fn redo_stroke(&mut self) -> bool {
        let Some(next) = self.stroke_redo_stack.pop_back() else { return false };
        let current = self.selection_mask.clone();
        push_stroke_bounded(&mut self.stroke_undo_stack, current);
        self.selection_mask = next;
        self.selection_hash = self.selection_mask.as_ref().map(|m| m.content_hash());
        true
    }

    // Test-only — non-test paths use the unified actions_undo/redo log
    // via try_undo_one_action / try_redo_one_action.
    #[cfg(test)]
    pub(crate) fn has_stroke_undo(&self) -> bool {
        !self.stroke_undo_stack.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn has_stroke_redo(&self) -> bool {
        !self.stroke_redo_stack.is_empty()
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

    /// What `enforce_tensor_budget` can free: the compressed segmentation
    /// and edge tensors.
    pub(crate) fn evictable_tensor_bytes(&self) -> usize {
        let seg = self.cached_tensor.as_ref().map_or(0, |ct| ct.compressed_size());
        let edge = self.cached_edge_tensors.as_ref().map_or(0, |ct| ct.compressed_size());
        seg + edge
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
            chain_dyn_cache: None,
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
            upscale_raw: None,
            bicubic_source: None,
            cached_edge_tensors: None,
            volatile_edge_tensor: None,
            cached_edge_mask: None,
            cached_masked_base: None,
            applied_preset,
            preset_undo_stack: VecDeque::new(),
            preset_redo_stack: VecDeque::new(),
            stroke_undo_stack: VecDeque::new(),
            stroke_redo_stack: VecDeque::new(),
            actions_undo: VecDeque::new(),
            actions_redo: VecDeque::new(),
            bg_image: None,
            bg_image_texture: None,
            bg_image_tex_pending: false,
            selection_mask: None,
            selection_hash: None,
            selection_tex_pending: None,
            selection_texture: None,
            magic_brush_embedding: None,
        }
    }

    /// Lockstep writer for `bg_image` + `settings.bg_image_hash`. Going
    /// through this path is the invariant the recipe diff relies on.
    /// Callers must follow up with
    /// `PrunrApp::kick_bg_image_tex_prep(item_id, ctx)` so the bg-image
    /// texture is built off-thread; the on-thread render path reads
    /// `bg_image_texture` directly and renders the placeholder bg color
    /// until the texture lands.
    pub(crate) fn set_bg_image(&mut self, img: image::DynamicImage, source_path: Option<PathBuf>) {
        let hash = bg_image_content_hash(&img);
        self.bg_image = Some(Arc::new(BgImage {
            source_path,
            image: Arc::new(img),
            hash,
        }));
        self.bg_image_texture = None;
        self.bg_image_tex_pending = false;
        self.settings.bg_image_hash = std::num::NonZeroU64::new(hash);
    }

    pub(crate) fn clear_bg_image(&mut self) {
        self.bg_image = None;
        self.bg_image_texture = None;
        self.bg_image_tex_pending = false;
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

    /// Build a distinct MaskArtifact for testing. Uses hash to create
    /// distinguishable masks without needing actual painted pixels.
    fn make_mask(tag: u64, w: u32, h: u32) -> Arc<prunr_core::selection::MaskArtifact> {
        let mut data = vec![0i8; (w * h) as usize];
        // Tag the first cell so masks with different tags hash differently.
        data[0] = tag as i8;
        Arc::new(prunr_core::selection::MaskArtifact {
            width: w,
            height: h,
            data: Arc::new(data),
        })
    }

    #[test]
    fn identical_commit_is_a_no_op_without_an_undo_step() {
        let mut item = fixture_item(1);
        assert!(item.commit_selection_mask(make_mask(1, 8, 8)));
        assert!(!item.commit_selection_mask(make_mask(1, 8, 8)), "same bytes: no commit");
        assert_eq!(item.stroke_undo_stack.len(), 1);
        assert_eq!(item.actions_undo.iter().filter(|a| matches!(a, ActionType::Stroke)).count(), 1);
    }

    #[test]
    fn dropping_the_oldest_snapshot_also_drops_its_marker() {
        let mut item = fixture_item(1);
        for tag in 0..(STROKE_HISTORY_DEPTH as u64 + 3) {
            assert!(item.commit_selection_mask(make_mask(tag, 8, 8)));
        }
        assert_eq!(item.stroke_undo_stack.len(), STROKE_HISTORY_DEPTH);
        let markers = item.actions_undo.iter().filter(|a| matches!(a, ActionType::Stroke)).count();
        assert_eq!(markers, STROKE_HISTORY_DEPTH, "one marker per surviving snapshot");
    }

    #[test]
    fn stroke_commit_pushes_pre_state_onto_undo_stack() {
        let mut item = fixture_item(1);
        assert!(!item.has_stroke_undo());
        item.commit_selection_mask(make_mask(1, 8, 8));
        assert!(item.has_stroke_undo(), "first stroke must register an undo entry (pre = None)");
        assert!(!item.has_stroke_redo());
    }

    /// Cancel-mid-process cleanup. After commit the item holds the new
    /// selection_mask + a Stroke marker on the action timeline.
    /// If the dispatch is cancelled, revert_last_stroke_commit must
    /// remove BOTH so the next stroke starts fresh and Cmd+Z doesn't
    /// see a phantom action.
    #[test]
    fn revert_last_stroke_commit_clears_state_and_marker() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        assert!(item.selection_mask.is_some(), "post-commit selection_mask is set");
        assert_eq!(item.actions_undo.len(), 1, "post-commit Stroke marker pushed");
        assert!(matches!(item.actions_undo.back(), Some(ActionType::Stroke)));

        item.revert_last_stroke_commit();
        assert!(item.selection_mask.is_none(), "selection_mask reverted to pre-stroke state (None)");
        assert!(
            !item.actions_undo.iter().any(|a| matches!(a, ActionType::Stroke)),
            "Stroke marker dropped — cancelled stroke must not appear in the undo timeline",
        );
        assert!(item.stroke_redo_stack.is_empty(), "no redo entry — cancel is final, not reversible");
    }

    /// Two committed strokes, the second one cancelled. Revert removes
    /// only the most-recent stroke's state + marker; the first survives.
    #[test]
    fn revert_last_stroke_commit_only_pops_the_most_recent() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        let after_first_hash = item.selection_hash;
        item.commit_selection_mask(make_mask(2, 8, 8));
        assert_ne!(item.selection_hash, after_first_hash);
        assert_eq!(item.actions_undo.len(), 2);

        item.revert_last_stroke_commit();
        assert_eq!(
            item.selection_hash, after_first_hash,
            "revert restored the post-stroke-1 state — stroke 1 still in effect",
        );
        assert_eq!(item.actions_undo.len(), 1, "only the second Stroke marker dropped");
    }

    #[test]
    fn undo_stroke_restores_previous_selection_mask() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        let after_first_hash = item.selection_hash;
        item.commit_selection_mask(make_mask(2, 8, 8));
        assert_ne!(item.selection_hash, after_first_hash, "second stroke changed hash");

        assert!(item.undo_stroke(), "stroke 2 must be undoable");
        assert_eq!(item.selection_hash, after_first_hash, "undo restored the post-stroke-1 hash");
        assert!(item.has_stroke_redo(), "undone stroke goes onto the redo stack");
    }

    #[test]
    fn redo_stroke_replays_selection_mask() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        item.commit_selection_mask(make_mask(2, 8, 8));
        let after_two_hash = item.selection_hash;

        item.undo_stroke();
        assert!(item.redo_stroke(), "redo available after undo");
        assert_eq!(item.selection_hash, after_two_hash, "redo replays the second selection");
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
    fn stroke_history_caps_at_depth() {
        let mut item = fixture_item(1);
        for i in 0..(STROKE_HISTORY_DEPTH + 5) {
            item.commit_selection_mask(make_mask(i as u64, 8, 8));
        }
        assert_eq!(
            item.stroke_undo_stack.len(),
            STROKE_HISTORY_DEPTH,
            "undo stack must cap at STROKE_HISTORY_DEPTH"
        );
    }

    #[test]
    fn undo_stroke_rederives_selection_hash() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        let expected_hash = item.selection_hash;
        item.commit_selection_mask(make_mask(2, 8, 8));

        item.undo_stroke();
        assert_eq!(
            item.selection_hash, expected_hash,
            "undo must re-derive selection_hash from the restored mask",
        );
    }

    // ── Ordering layer (actions_undo / actions_redo) ─────────────────────────

    #[test]
    fn stroke_commit_pushes_marker_and_clears_actions_redo() {
        let mut item = fixture_item(1);
        item.actions_redo.push_back(ActionType::Stroke);
        item.commit_selection_mask(make_mask(1, 8, 8));
        assert_eq!(item.actions_undo.back(), Some(&ActionType::Stroke),
            "stroke commit must push a Stroke marker onto actions_undo");
        assert!(item.actions_redo.is_empty(),
            "stroke commit must clear actions_redo — new edit branches the timeline");
    }

    #[test]
    fn action_markers_ordered_across_action_types() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        item.commit_selection_mask(make_mask(2, 8, 8));
        let order: Vec<ActionType> = item.actions_undo.iter().copied().collect();
        assert_eq!(order, vec![ActionType::Stroke, ActionType::Stroke],
            "two strokes produce two Stroke markers in order");
    }

    #[test]
    fn divergence_clears_redo_in_actions_layer() {
        let mut item = fixture_item(1);
        item.commit_selection_mask(make_mask(1, 8, 8));
        // Simulate an undo (normally done via try_undo_one_action, here manually).
        let kind = item.actions_undo.pop_back().unwrap();
        item.actions_redo.push_back(kind);
        assert!(!item.actions_redo.is_empty());

        // New commit branches the timeline.
        item.commit_selection_mask(make_mask(3, 8, 8));
        assert!(item.actions_redo.is_empty(),
            "fresh commit after undo must wipe actions_redo");
    }

    #[test]
    fn push_action_bounded_does_not_clear_opposite_stack() {
        // push_action_bounded is the move-between-stacks helper; it must not
        // clear the opposite stack (that would break the undo↔redo round-trip).
        let mut undo_stack: std::collections::VecDeque<ActionType> = Default::default();
        let mut redo_stack: std::collections::VecDeque<ActionType> = Default::default();
        redo_stack.push_back(ActionType::Stroke);
        push_action_bounded(&mut undo_stack, ActionType::Result);
        assert!(!redo_stack.is_empty(), "push_action_bounded must not clear the opposite stack");
        assert_eq!(undo_stack.back(), Some(&ActionType::Result));
    }

    #[test]
    fn action_hist_depth_caps_actions_undo() {
        let mut item = fixture_item(1);
        for _ in 0..(ACTION_HIST_DEPTH + 5) {
            item.push_action_marker(ActionType::Stroke);
            // Prevent stroke_undo_stack from overflow (not the subject here).
            item.stroke_undo_stack.clear();
        }
        assert_eq!(item.actions_undo.len(), ACTION_HIST_DEPTH,
            "actions_undo must be capped at ACTION_HIST_DEPTH");
    }

    // ── upscale_raw cache contracts ──────────────────────────────────────────

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
        item.selection_tex_pending = Some((hash, crate::gui::background_io::SelectionStyle {
            fill_opacity: 0.15, outline_opacity: 1.0, outline_thickness: 2.0, edge_feather_px: 0,
        }));
        item.invalidate_selection();
        assert!(item.selection_mask.is_none(), "selection_mask must be cleared");
        assert!(item.selection_hash.is_none(), "selection_hash must be cleared");
        assert!(item.selection_texture.is_none(), "selection_texture must be cleared");
        assert!(item.selection_tex_pending.is_none(), "pending build key must be cleared");
    }

}
