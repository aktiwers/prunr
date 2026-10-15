//! Batch state, lifecycle, memory governance, and texture coordination.
//!
//! Owns:
//! - The `Vec<BatchItem>` and selection state
//! - The next-id counter for new items
//! - The `BackgroundIO` channels (thumbnail, decode, save-done, tex-prep,
//!   file-load)
//!
//! Does NOT own:
//! - Inference dispatch (worker channels, admission, live-preview) — `Processor`
//! - Undo — each item's own `timeline`
//! - Drag-out lifecycle — `DragExportState`

use std::collections::HashSet;
use std::sync::Arc;

use super::background_io::BackgroundIO;
use super::item::{BatchItem, BatchStatus, ImageSource, RunTier};
use super::memory::{AdmissionController, ImageMemCost, RESIDENT_HISTORY_STEPS};
use super::state::AppState;
use super::worker::{CachedTensorWork, WorkItem};

/// Per-status counts across the batch. Produced by `status_counts` for
/// callers that need to render progress strings, decide "all done" vs
/// "still processing", etc. without triple-scanning `items` inline.
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct StatusCounts {
    pub done: usize,
    pub processing: usize,
    pub errored: usize,
}

impl StatusCounts {
    pub(crate) fn batch_total(&self) -> usize {
        self.done + self.processing + self.errored
    }
}

/// Derived progress summary for the status bar. Owns its `stage` string so
/// callers don't divide done/total or format text inline.
#[derive(Default, Clone, Debug)]
pub(crate) struct StatusReport {
    pub stage: String,
    pub pct: f32,
}

/// Which shape the Process button should take for the current selection state.
/// Drives both the button label and the icon the toolbar renders.
// `Process*` prefix matches the user-visible button labels and the
// dispatcher arms in `process_items` — renaming would split that pairing.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessButtonLabel {
    /// No checkboxes set — click targets the currently-viewed item only.
    ProcessViewed,
    /// N checkboxes set, 1 ≤ N < total (or 1 ≤ N and total == 1).
    ProcessSelected(usize),
    /// All N ≥ 2 items checked — click targets the whole batch.
    ProcessAll(usize),
}

/// Combined budget for compressed segmentation (`cached_tensor`) + edge
/// (`cached_edge_tensors`) caches across all batch items, in bytes.
const TENSOR_BUDGET: usize = 512 * 1024 * 1024;

/// Maximum thumbnail edge length in pixels. Used by `request_thumbnail`
/// for both source-decode and result-rgba paths.
const THUMB_MAX_PX: u32 = 160;

pub(crate) struct BatchManager {
    pub(crate) items: Vec<BatchItem>,
    pub(crate) selected_index: usize,
    pub(crate) next_id: u64,
    pub(crate) bg_io: BackgroundIO,
    last_history_sweep: std::time::Instant,
}

const HISTORY_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// What landing a background picture did, for the app's part: the saved
/// path map and the message.
pub(crate) enum BgLanded {
    Picked { hash: u64, path: std::path::PathBuf },
    Filled,
    Failed { picked: bool, error: String },
    Gone,
}

impl BatchManager {
    /// Compress every stored undo state nothing else holds, off-thread,
    /// behind the decode slots so a burst can't fan out N zstd encoders.
    /// Doing it inline cost ~0.1-0.6 s per switch, undo or redo.
    pub(crate) fn compress_history_off_thread(&mut self) {
        // A few times a second is plenty; every frame walked every step.
        let now = std::time::Instant::now();
        if now.saturating_duration_since(self.last_history_sweep) < HISTORY_SWEEP_INTERVAL {
            return;
        }
        self.last_history_sweep = now;
        let slots = &self.bg_io.decode_slots;
        for item in &self.items {
            item.timeline.stored_images().for_each(|c| c.compress_when_idle(slots));
            item.timeline.stored_selections().for_each(|c| c.compress_when_idle(slots));
        }
    }

    /// Under memory pressure: move compressed undo states to disk.
    pub(crate) fn demote_history_to_disk(&self) {
        for item in &self.items {
            for cell in item.timeline.stored_images() {
                cell.demote_to_disk();
            }
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            selected_index: 0,
            next_id: 0,
            bg_io: BackgroundIO::new(),
            last_history_sweep: std::time::Instant::now(),
        }
    }

    pub(crate) fn selected_item(&self) -> Option<&BatchItem> {
        self.items.get(self.selected_index)
    }

    /// Look up an item by id. Returns `None` if no batch item has that id —
    /// common when a worker event arrives after the item was removed.
    pub(crate) fn find_by_id(&self, id: u64) -> Option<&BatchItem> {
        self.items.iter().find(|b| b.id == id)
    }

    /// The position of the item with `id`, for callers that index.
    pub(crate) fn index_of(&self, id: u64) -> Option<usize> {
        self.items.iter().position(|b| b.id == id)
    }

    /// Mutable counterpart of `find_by_id`. Used by worker-event handlers
    /// that write results / status into the matching item.
    pub(crate) fn find_by_id_mut(&mut self, id: u64) -> Option<&mut BatchItem> {
        self.items.iter_mut().find(|b| b.id == id)
    }

    /// Derives app-level state from the selected item's status.
    /// Empty when no item is selected; Done/Processing pass through;
    /// all other statuses collapse to Loaded.
    pub(crate) fn app_state(&self) -> AppState {
        match self.selected_item() {
            None => AppState::Empty,
            Some(item) => match item.status {
                BatchStatus::Done => AppState::Done,
                BatchStatus::Processing => AppState::Processing,
                _ => AppState::Loaded,
            },
        }
    }

    /// True when the currently-selected item has the given id. Callers use
    /// this to decide whether a worker message / background-io event affects
    /// what the user is looking at right now (textures, progress, status).
    pub(crate) fn is_selected(&self, id: u64) -> bool {
        self.selected_item().is_some_and(|b| b.id == id)
    }

    /// IDs of ticked items currently in a given status. Used by
    /// the partial-cancel path to decide which in-flight items to stop.
    pub(crate) fn ticked_ids_with_status(&self, status: BatchStatus) -> Vec<u64> {
        self.ticked_with_status(status).map(|i| i.id).collect()
    }

    /// Whether any ticked item is in `status`: the same set
    /// `ticked_ids_with_status` returns, without collecting it.
    pub(crate) fn any_ticked_with_status(&self, status: BatchStatus) -> bool {
        self.ticked_with_status(status).next().is_some()
    }

    fn ticked_with_status(&self, status: BatchStatus) -> impl Iterator<Item = &BatchItem> {
        self.items.iter().filter(move |i| i.ticked && i.status == status)
    }

    /// Tick or untick every item's checkbox.
    pub(crate) fn set_all_ticked(&mut self, ticked: bool) {
        for item in &mut self.items {
            item.ticked = ticked;
        }
    }

    /// Selected-item index clamped to the current batch size. `None` when
    /// the batch is empty — callers early-return on that shape instead of
    /// dealing with saturating arithmetic.
    pub(crate) fn selected_idx_clamped(&self) -> Option<usize> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.selected_index.min(self.items.len() - 1))
        }
    }

    /// Clamp `selected_index` to the current batch size. Returns `true` when
    /// the index moved (i.e. the batch shrank below the old selection). Callers
    /// use the return value to trigger a pending batch sync.
    /// No-op on an empty batch — resets to 0 instead, which is the canonical
    /// "nothing selected" sentinel.
    pub(crate) fn clamp_selected_index(&mut self) -> bool {
        if self.items.is_empty() {
            let moved = self.selected_index != 0;
            self.selected_index = 0;
            moved
        } else {
            let clamped = self.selected_index.min(self.items.len() - 1);
            let moved = clamped != self.selected_index;
            self.selected_index = clamped;
            moved
        }
    }

    /// Switch the selected item to `idx`. Returns `true` when the index actually
    /// changed; `false` when it was already selected. Callers use the return value
    /// to decide whether to reset the canvas (zoom/pan/texture sync).
    pub(crate) fn select_item(&mut self, idx: usize) -> bool {
        debug_assert!(idx < self.items.len(), "select_item: idx {idx} out of bounds (len={})", self.items.len());
        if self.selected_index == idx {
            return false;
        }
        self.selected_index = idx;
        true
    }

    /// Move the item at `from` to position `to`, adjusting `selected_index`
    /// so the same logical item remains selected after the move.
    /// `to == items.len()` is valid and appends the item to the end.
    pub(crate) fn reorder(&mut self, from: usize, to: usize) {
        if from == to || from >= self.items.len() {
            return;
        }
        let item = self.items.remove(from);
        let dst = if from < to { to - 1 } else { to };
        self.items.insert(dst, item);
        if self.selected_index == from {
            self.selected_index = dst;
        } else if from < self.selected_index && self.selected_index <= to {
            self.selected_index -= 1;
        } else if to <= self.selected_index && self.selected_index < from {
            self.selected_index += 1;
        }
    }

    /// Derive the Process button's label shape from current selection state.
    /// Single-item batches where the one item is checked render as
    /// `ProcessSelected(1)` (not `ProcessAll(1)`) — "Process All [1]" reads
    /// as a glitch.
    pub(crate) fn process_button_label(&self) -> ProcessButtonLabel {
        let total = self.items.len();
        let ticked = self.ticked_count();
        match (ticked, total) {
            (0, _) => ProcessButtonLabel::ProcessViewed,
            (n, t) if n == t && t >= 2 => ProcessButtonLabel::ProcessAll(t),
            (n, _) => ProcessButtonLabel::ProcessSelected(n),
        }
    }

    /// Item ids the Process button should dispatch: the checkbox-checked set
    /// if any are checked, else the currently-viewed item. Empty when the
    /// batch is empty.
    pub(crate) fn items_to_process(&self) -> Vec<u64> {
        self.targets(self.selected_idx_clamped()).map(|(_, i)| i.id).collect()
    }

    /// Any item in the action target set satisfies `f`, without
    /// allocating (toolbar Undo/Redo enablement).
    pub(crate) fn any_target_can<F: FnMut(&BatchItem) -> bool>(&self, mut f: F) -> bool {
        self.targets(self.viewed_unclamped()).any(|(_, item)| f(item))
    }

    /// Indices of the items in the action target set.
    pub(crate) fn targeted_indices(&self) -> Vec<usize> {
        self.targets(self.viewed_unclamped()).map(|(i, _)| i).collect()
    }

    /// The action target set, with indices: the ticked items when any are
    /// ticked, else the one at `viewed`. Callers resolve `viewed`
    /// themselves: Process clamps a stale index, Undo/Redo and the
    /// targeted indices skip it.
    fn targets(&self, viewed: Option<usize>) -> impl Iterator<Item = (usize, &BatchItem)> {
        let ticked = self.any_ticked();
        self.items.iter().enumerate()
            .filter(move |&(i, item)| if ticked { item.ticked } else { Some(i) == viewed })
    }

    /// `selected_index` when it names an item.
    fn viewed_unclamped(&self) -> Option<usize> {
        self.selected_item().map(|_| self.selected_index)
    }

    /// Remove the item at `idx`; its undo files go with its timeline.
    /// Returns `false` when `idx` is out of bounds. Caller is responsible
    /// for clamping `selected_index` and any downstream sync.
    pub(crate) fn remove(&mut self, idx: usize) -> bool {
        if idx >= self.items.len() { return false; }
        self.items.remove(idx);
        true
    }

    /// Return every `Processing` item to the state it shows, after a
    /// Cancel-All. Late arrivals from the worker are dropped by the job
    /// check in `on_batch_item_done`.
    pub(crate) fn reset_processing_to_pending(&mut self) {
        for item in &mut self.items {
            if item.status == BatchStatus::Processing {
                item.abandon_job();
            }
        }
    }

    /// Number of items with the checkbox set. Single source of truth so
    /// statusbar / sidebar don't re-roll `iter().filter(...).count()`.
    pub(crate) fn ticked_count(&self) -> usize {
        self.items.iter().filter(|i| i.ticked).count()
    }

    /// True when at least one item has the checkbox set.
    pub(crate) fn any_ticked(&self) -> bool {
        self.items.iter().any(|i| i.ticked)
    }

    /// True when every item has the checkbox set (and the batch is
    /// non-empty — an empty batch is not "all ticked").
    pub(crate) fn all_ticked(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.ticked)
    }

    /// Single pass over `items` producing the three counts that callers
    /// (`poll_worker_results`, statusbar) otherwise compute with three
    /// separate filter-count passes.
    pub(crate) fn status_counts(&self) -> StatusCounts {
        let mut c = StatusCounts::default();
        for item in &self.items {
            match item.status {
                BatchStatus::Done => c.done += 1,
                BatchStatus::Processing => c.processing += 1,
                BatchStatus::Error(_) => c.errored += 1,
                _ => {}
            }
        }
        c
    }

    /// Status-bar progress derived from `status_counts`. Single source of
    /// truth so callers don't divide done/total or format text inline.
    pub(crate) fn progress(&self) -> StatusReport {
        let counts = self.status_counts();
        let total = counts.batch_total();
        let stage = if counts.processing > 0 {
            format!("Processing {}/{total}", counts.done)
        } else {
            "Finishing up".to_string()
        };
        StatusReport {
            stage,
            pct: counts.done as f32 / total.max(1) as f32,
        }
    }

    /// First error message across errored items, or `None` if none errored.
    /// Used for batch-failure toasts so the user sees the *reason* instead of
    /// just the count.
    pub(crate) fn first_error_message(&self) -> Option<&str> {
        self.items.iter().find_map(|item| match &item.status {
            BatchStatus::Error(msg) => Some(msg.as_str()),
            _ => None,
        })
    }

    /// Pre-decode source bytes to RgbaImage on a background thread; the
    /// result lands on `bg_io.decode_rx` for the main thread to attach.
    /// Decode failures travel as `Err(msg)` so the UI clears
    /// `decode_pending` and surfaces the error instead of leaving the
    /// item stuck in a "still loading" state forever.
    pub(crate) fn request_decode_bytes(&self, item_id: u64, bytes: Arc<Vec<u8>>) {
        let tx = self.bg_io.decode_tx.clone();
        let slots = self.bg_io.decode_slots.clone();
        std::thread::spawn(move || {
            // Park here past `available_parallelism()` so a 50-image
            // Process All doesn't hold 50 × ~80 MB transient peaks.
            let _slot = slots.acquire();
            // Inner block scope: the `DynamicImage` (~50 MB at 4 K RGBA)
            // and the encoded `bytes` Arc both drop before the channel
            // send, so concurrent decodes don't pile up DynamicImage +
            // RGBA simultaneously per thread.
            let result = match image::load_from_memory(&bytes) {
                Ok(img) => Ok(Arc::new(img.to_rgba8())),
                Err(e) => Err(format!("decode failed: {e}")),
            };
            drop(bytes);
            let _ = tx.send((item_id, result));
        });
    }

    /// Pre-decode from an ImageSource (reads file if needed).
    pub(crate) fn request_decode_source(&self, item_id: u64, source: &ImageSource) {
        if let Ok(bytes) = source.load_bytes() {
            self.request_decode_bytes(item_id, bytes);
        }
    }

    /// A background picture finished loading. A picked one replaces the
    /// background as one undo step; a preset's fills in the picture its
    /// settings name, if they still do; a preset's that failed is dropped.
    pub(crate) fn land_bg_image(&mut self, load: super::background_io::BgImageLoad) -> BgLanded {
        let super::background_io::BgImageLoad { item_id, path, picked, result } = load;
        let Some(item) = self.find_by_id_mut(item_id) else { return BgLanded::Gone };
        match result {
            Ok((img, hash)) if picked => {
                item.settings.bg = None;
                item.settings.bg_effect = prunr_core::BgEffect::None;
                item.set_bg_image_hashed(img, Some(path.clone()), hash);
                item.commit(super::timeline::StepKind::Adjust);
                BgLanded::Picked { hash, path }
            }
            Ok((img, hash)) => {
                if item.settings.bg_image_hash.map(|h| h.get()) == Some(hash) {
                    item.set_bg_image_hashed(img, Some(path), hash);
                }
                BgLanded::Filled
            }
            Err(error) => {
                if !picked {
                    item.clear_bg_image();
                }
                BgLanded::Failed { picked, error }
            }
        }
    }

    /// Load and hash a background picture off the UI thread; it lands on
    /// `bg_io.bg_image_rx`.
    pub(crate) fn request_bg_image(&self, item_id: u64, path: std::path::PathBuf, picked: bool) {
        let tx = self.bg_io.bg_image_tx.clone();
        let slots = self.bg_io.decode_slots.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            let result = prunr_core::load_image_from_path(&path)
                .map(|img| {
                    let hash = super::item::bg_image_content_hash(&img);
                    (img, hash)
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(super::background_io::BgImageLoad { item_id, path, picked, result });
        });
    }

    /// Filter-only Process: load + decode + `apply_fill_style` on a background
    /// thread, deliver the result via `bg_io.filter_only_rx`. Keeps the UI
    /// thread free on large batches that the inline path used to freeze.
    pub(crate) fn request_filter_only(
        &self,
        item_id: u64,
        source: &ImageSource,
        base: Option<super::worker::RunImage>,
        fill_style: prunr_core::FillStyle,
    ) {
        let tx = self.bg_io.filter_only_tx.clone();
        let source = source.clone();
        let slots = self.bg_io.decode_slots.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            let load = || source.load_bytes().map_err(super::item::load_failed);
            let (start, chain_start) = base.map_or((None, None), |run| run.resolve(item_id, load));
            // Drop bytes / DynamicImage / RGBA before send so concurrent
            // threads don't pile per-image peaks.
            let result: Result<Arc<image::RgbaImage>, String> = (|| {
                let mut rgba = match start {
                    // No effect: the pass is its start, without a copy.
                    Some(start) if fill_style == prunr_core::FillStyle::None => return Ok(start),
                    Some(start) => Arc::unwrap_or_clone(start),
                    None => {
                        let bytes = load()?;
                        let original = prunr_core::load_image_from_bytes(&bytes)
                            .map_err(|_| "Decode failed".to_string())?;
                        drop(bytes);
                        original.into_rgba8()
                    }
                };
                prunr_core::apply_fill_style(&mut rgba, fill_style);
                Ok(Arc::new(rgba))
            })();
            let _ = tx.send((item_id, result, chain_start));
        });
    }

    /// Request thumbnail generation on a background thread for a batch item.
    /// If `result_rgba` is `Some`, thumbnails from the result; otherwise decodes
    /// source bytes.
    ///
    /// Result images stay transparent in storage; the thumb texture needs its
    /// own composite so the sidebar matches what's drawn on the canvas. Solid
    /// line color is already baked into `result_rgba` by the pipeline, so we
    /// don't need a separate parameter for it.
    pub(crate) fn request_thumbnail(
        &self,
        item_id: u64,
        source: &ImageSource,
        result_rgba: Option<&Arc<image::RgbaImage>>,
    ) {
        let tx = self.bg_io.thumb_tx.clone();
        let slots = self.bg_io.decode_slots.clone();
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone();
            std::thread::spawn(move || {
                let _slot = slots.acquire();
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                let thumb = prunr_core::formats::resize_rgba(&rgba, w, h, prunr_core::formats::ResizeFilter::Bilinear);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let source = source.clone();
            std::thread::spawn(move || {
                let _slot = slots.acquire();
                // Build the thumb in an inner scope so the source bytes,
                // DynamicImage, and full-resolution RGBA all drop before
                // we send. On a 50-image batch this is the dominant
                // transient peak during sidebar ingestion (was N ×
                // ~50 MB co-resident).
                let thumb = (|| -> Option<image::RgbaImage> {
                    let bytes = source.load_bytes().ok()?;
                    let img = image::load_from_memory(&bytes).ok()?;
                    drop(bytes);
                    let (w, h) = fit_dimensions(img.width(), img.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                    Some(prunr_core::formats::resize_to_rgba(&img, w, h, prunr_core::formats::ResizeFilter::Bilinear))
                })();
                if let Some(thumb) = thumb {
                    let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
                }
            });
        }
    }

    /// Evict tensor caches from oldest-loaded items until under budget.
    /// Iterates front-to-back (oldest first) to preserve recently-processed items.
    /// Drops BOTH caches on eviction — partial eviction would leave a partially-stale
    /// item (segmentation cached but edges gone, or vice versa) which is useless.
    pub(crate) fn enforce_tensor_budget(&mut self) {
        // Budget only what eviction can free — selection planes and
        // embeddings would otherwise force tensor evictions they never
        // relieve.
        let total: usize = self.items.iter().map(BatchItem::evictable_tensor_bytes).sum();
        if total <= TENSOR_BUDGET { return; }
        let selected_id = self.selected_item().map(|b| b.id);
        let mut remaining = total;
        for item in &mut self.items {
            if remaining <= TENSOR_BUDGET { break; }
            // Preserve the selected item's tensors (most likely to be reused).
            if Some(item.id) == selected_id { continue; }
            remaining -= item.evictable_tensor_bytes();
            item.drop_tensor_caches();
        }
    }

    /// Keep the selected item's selection texture current: request a
    /// rebuild whenever the mask or the baked style differs from what the
    /// texture (or the build in flight) was made from. One path for every
    /// producer — commits, undo/redo, image switch, knob changes.
    pub(crate) fn ensure_selection_texture(
        &mut self,
        style: super::background_io::SelectionStyle,
        ctx: &egui::Context,
    ) {
        let Some(idx) = self.selected_idx_clamped() else { return };
        let item = &mut self.items[idx];
        // No selection left: nothing will replace the old texture, which
        // the overlay would keep drawing.
        let Some(hash) = item.selection_hash else {
            item.selection_texture = None;
            item.selection_tex_pending = None;
            return;
        };
        let key = (hash, style);
        if item.current_selection_texture(style).is_some() || item.selection_tex_pending == Some(key) {
            return;
        }
        let Some(mask) = item.selection_mask.clone() else { return };
        item.selection_tex_pending = Some(key);
        let (item_id, source) = (item.id, item.source_rgba.clone());
        // A patch applies only to a texture of the same style, and only
        // without feather: feathering reshapes the plane beyond the stroke.
        let base = item.selection_texture.as_ref()
            .filter(|t| t.key.1 == style && style.edge_feather_px == 0)
            .map(|t| (t.key.0, Arc::clone(&t.shown)));
        self.bg_io.request_selection_visualization(item_id, mask, key, source, base, ctx.clone());
    }

    /// The items a drag from row `idx` carries: every ticked one when that
    /// row is ticked, else that row; never one still processing.
    pub(crate) fn drag_targets(&self, idx: usize) -> Vec<u64> {
        let free = |b: &&BatchItem| b.status != BatchStatus::Processing;
        match self.items.get(idx) {
            Some(row) if row.ticked => self.items.iter().filter(|b| b.ticked).filter(free).map(|b| b.id).collect(),
            Some(row) => std::iter::once(row).filter(free).map(|b| b.id).collect(),
            None => Vec::new(),
        }
    }

    /// Evict all tensor caches except the selected item (called under memory pressure).
    pub(crate) fn evict_all_tensors(&mut self) {
        let selected_id = self.selected_item().map(|b| b.id);
        for item in &mut self.items {
            if Some(item.id) != selected_id {
                item.drop_tensor_caches();
            }
        }
    }

    /// Commit a new selection mask to the named item, registering it for
    /// undo (`BatchItem::commit_selection_mask`). Does NOT fire any model
    /// dispatch; `apply_selection_to_active_model` owns that decision.
    ///
    /// Returns `false` if the item was not found or the mask is
    /// byte-identical to the current selection (nothing mutated).
    pub(crate) fn commit_selection(
        &mut self,
        item_id: u64,
        mask: prunr_core::selection::MaskArtifact,
        kind: super::timeline::StepKind,
    ) -> bool {
        self.find_by_id_mut(item_id)
            .is_some_and(|item| item.commit_selection_mask(Arc::new(mask), kind))
    }

    /// Drop the selection on the named item (Esc, Clear button) as an
    /// undo step. Returns true if there was one.
    pub(crate) fn clear_selection(&mut self, item_id: u64) -> bool {
        let Some(item) = self.find_by_id_mut(item_id) else { return false };
        let was_present = item.selection_mask.is_some();
        item.invalidate_selection();
        if was_present {
            item.commit(super::timeline::StepKind::Selection);
        }
        was_present
    }

    /// On a cut-out: fold the selection into the edits as `action` says
    /// (see `BatchItem::bake_selection`), one undo step. True when there
    /// was a selection.
    pub(crate) fn bake_selection(
        &mut self,
        item_id: u64,
        style: super::background_io::SelectionStyle,
        action: super::brush_state::CutoutAction,
    ) -> bool {
        let Some(item) = self.find_by_id_mut(item_id) else { return false };
        let baked = item.bake_selection(style, action);
        if baked {
            item.commit(super::timeline::StepKind::Selection);
        }
        baked
    }

    /// Give the images in `ids` the settings of the image at `from`, with
    /// its background picture: a run applies what is shown to the batch.
    pub fn share_settings(&mut self, from: usize, ids: &HashSet<u64>) {
        let Some(source) = self.items.get(from) else { return };
        let (settings, bg_image) = (source.settings, source.bg_image.clone());
        for item in self.in_set(ids) {
            item.adopt_settings(settings, bg_image.clone());
        }
    }

    /// Drop every image's Magic Brush embedding (16 MB each).
    pub fn clear_magic_embeddings(&mut self) {
        for item in &mut self.items {
            item.magic_brush_embedding = None;
        }
    }

    /// Without a background-removal model there is no subject to outline,
    /// so Subject lines draw over the whole photo, the nearest mode that
    /// still has lines.
    pub fn fit_line_modes_to(&mut self, model: super::settings::SettingsModel) {
        if model.uses_segmentation() {
            return;
        }
        for item in self.items.iter_mut().filter(|i| i.settings.line_mode == prunr_core::LineMode::SubjectOutline) {
            item.settings.line_mode = prunr_core::LineMode::EdgesOnly;
        }
    }
}

/// The images a Process run sends, by how much of the pipeline each
/// needs, and how many were already up to date (for the toast).
#[derive(Default)]
pub(crate) struct ClassifiedTiers {
    pub(crate) tier1: HashSet<u64>,
    pub(crate) tier2: HashSet<u64>,
    pub(crate) tier_add_edge: HashSet<u64>,
    pub(crate) skip_count: usize,
}

impl ClassifiedTiers {
    pub(crate) fn process_count(&self) -> usize {
        self.tier1.len() + self.tier2.len() + self.tier_add_edge.len()
    }

    /// Every image the run sends.
    pub(crate) fn all_process_ids(&self) -> HashSet<u64> {
        self.tier1.iter()
            .chain(self.tier2.iter())
            .chain(self.tier_add_edge.iter())
            .copied().collect()
    }
}

impl BatchManager {
    /// The items whose id is in `ids`, in batch order.
    pub(crate) fn in_set<'a>(&'a mut self, ids: &'a HashSet<u64>) -> impl Iterator<Item = &'a mut BatchItem> + 'a {
        self.items.iter_mut().filter(move |i| ids.contains(&i.id))
    }

    /// Whether any image has a decode, thumbnail or texture on its way.
    pub(crate) fn has_pending_work(&self) -> bool {
        self.items.iter().any(BatchItem::has_pending_work)
    }

    /// Sort the images in `ids` by the run a `step` by `cutter` needs
    /// (`BatchItem::run_tier`). One that chains onto another model's
    /// cut-out is a full run, whatever its recipe says.
    pub(crate) fn classify(&mut self, ids: &HashSet<u64>, model: prunr_core::ModelKind, step: Option<super::timeline::StepKind>, cutter: super::item::Cutter) -> ClassifiedTiers {
        let mut tiers = ClassifiedTiers::default();
        for item in self.in_set(ids) {
            if item.chains(step, cutter) {
                tiers.tier1.insert(item.id);
                continue;
            }
            match item.run_tier(model) {
                RunTier::UpToDate => tiers.skip_count += 1,
                RunTier::Full => { tiers.tier1.insert(item.id); }
                RunTier::Mask => { tiers.tier2.insert(item.id); }
                RunTier::AddEdge => { tiers.tier_add_edge.insert(item.id); }
            }
        }
        tiers
    }

    /// The work for the images in `ids` that run again from their cached
    /// tensor. One whose cache or file can't be read fails, and starts
    /// over next time.
    pub(crate) fn cached_tensor_work(&mut self, ids: &HashSet<u64>) -> Vec<CachedTensorWork> {
        fn start_over(item: &mut BatchItem, why: String) {
            item.status = BatchStatus::Error(why);
            item.set_cached_tensor(None);
            item.applied_recipe = None;
        }
        let mut out = Vec::new();
        for item in self.in_set(ids) {
            let Some(ct) = &item.cached_tensor else { continue };
            match (ct.decompress(), item.source.load_bytes()) {
                (Some(data), Ok(bytes)) => {
                    out.push(CachedTensorWork {
                        item_id: item.id,
                        tensor_data: data,
                        tensor_height: ct.height,
                        tensor_width: ct.width,
                        model: ct.model,
                        original_bytes: bytes,
                        mask: item.settings.mask_settings(),
                    });
                    item.status = BatchStatus::Processing;
                }
                (None, _) => start_over(item, "Tensor cache corrupt".into()),
                (_, Err(e)) => start_over(item, super::item::load_failed(e)),
            }
        }
        out
    }

    /// What a full run on each image in `ids` is expected to hold in RAM.
    pub(crate) fn full_run_costs(&self, ids: &HashSet<u64>) -> Vec<ImageMemCost> {
        self.items.iter()
            .filter(|i| ids.contains(&i.id))
            .map(|i| AdmissionController::estimate_cost(i.id, i.dimensions, i.source.estimated_size(), RESIDENT_HISTORY_STEPS))
            .collect()
    }

    /// The work for full runs on the images in `ids`, in batch order;
    /// one whose file can't be read is left out.
    pub(crate) fn full_run_work(&mut self, ids: &HashSet<u64>) -> Vec<WorkItem> {
        self.in_set(ids).filter_map(BatchItem::full_run_work).collect()
    }

    /// The images in `ids` not under way wait their turn; one that has
    /// just failed keeps its error.
    pub(crate) fn mark_waiting(&mut self, ids: &HashSet<u64>) {
        for item in self.in_set(ids).filter(|i| matches!(i.status, BatchStatus::Done | BatchStatus::Pending)) {
            item.status = BatchStatus::Pending;
        }
    }
}

/// Compute dimensions that fit within `max_w` × `max_h` preserving aspect ratio.
fn fit_dimensions(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32).min(1.0);
    ((src_w as f32 * scale).round().max(1.0) as u32,
     (src_h as f32 * scale).round().max(1.0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::item_settings::ItemSettings;
    use crate::gui::worker::{CompressedTensor, TensorCache};
    use prunr_core::ModelKind;
    use std::time::Duration;

    fn fixture() -> BatchManager {
        BatchManager::new()
    }

    /// Recording a step (a switch, an undo, a new result) must cost the UI
    /// thread nothing: a stored state is compressed off-thread once nothing
    /// else holds it, and reads back unchanged. The step an undo returns to
    /// is held unpacked instead.
    #[test]
    fn undo_states_are_compressed_off_the_ui_thread() {
        let mut batch = fixture();
        let mut item = item_with_cache(1, 0);
        item.on_screen = true;
        let rgba = Arc::new(image::RgbaImage::from_pixel(64, 64, image::Rgba([9, 8, 7, 255])));
        item.result_rgba = Some(Arc::clone(&rgba));
        item.status = BatchStatus::Done;
        item.commit(crate::gui::timeline::StepKind::Process);
        item.result_rgba = Some(Arc::new(image::RgbaImage::from_pixel(64, 64, image::Rgba([1, 1, 1, 255]))));
        item.commit(crate::gui::timeline::StepKind::Edit);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(64, 64)));
        item.commit(crate::gui::timeline::StepKind::Edit);
        batch.items.push(item);
        drop(rgba);

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let first_compressed = |b: &BatchManager| b.items[0].timeline.stored_images().next().is_some_and(|c| c.is_compressed());
        while !first_compressed(&batch) && std::time::Instant::now() < deadline {
            batch.compress_history_off_thread();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(first_compressed(&batch), "an older result, held by nothing else, is compressed");
        let warm = |b: &BatchManager| b.items[0].warm_steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[crate::gui::item::BACK].as_ref().is_some_and(|w| w.image.is_some());
        while !warm(&batch) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let previous_compressed = batch.items[0].timeline.stored_images().nth(1).is_some_and(|c| c.is_compressed());
        assert!(warm(&batch) && !previous_compressed, "the step undo returns to stays unpacked");
        let item = &mut batch.items[0];
        item.undo_step();
        item.undo_step();
        assert_eq!(item.result_rgba.as_ref().unwrap().get_pixel(5, 5), &image::Rgba([9, 8, 7, 255]));
    }

    /// A BatchItem with `tensor_floats` f32s of cached tensor (0 = none).
    fn item_with_cache(id: u64, tensor_floats: usize) -> BatchItem {
        let mut item = BatchItem::new(
            id,
            format!("item_{id}.png"),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (10, 10),
            ItemSettings::default(),
            String::new(),
        );
        if tensor_floats > 0 {
            // Real CompressedTensor: zstd-compress f32 data via from_raw.
            let data: Vec<f32> = vec![0.5; tensor_floats];
            let cache = TensorCache { data, height: 10, width: 10, model: ModelKind::Silueta };
            item.set_cached_tensor(CompressedTensor::from_raw(cache));
        }
        item
    }

    /// Every image in the batch, not just the one on screen: an image left
    /// on Subject would otherwise run a cut-out under No model.
    #[test]
    fn without_a_cutout_model_every_subject_line_mode_becomes_whole_photo() {
        use crate::gui::settings::SettingsModel;
        use prunr_core::LineMode;
        let mut batch = fixture();
        for (id, mode) in [(1, LineMode::SubjectOutline), (2, LineMode::Off), (3, LineMode::SubjectOutline)] {
            let mut item = item_with_cache(id, 0);
            item.settings.line_mode = mode;
            batch.items.push(item);
        }
        batch.fit_line_modes_to(SettingsModel::Silueta);
        assert_eq!(batch.items[0].settings.line_mode, LineMode::SubjectOutline, "a cut-out model keeps Subject");
        batch.fit_line_modes_to(SettingsModel::None);
        let modes: Vec<_> = batch.items.iter().map(|i| i.settings.line_mode).collect();
        assert_eq!(modes, [LineMode::EdgesOnly, LineMode::Off, LineMode::EdgesOnly]);
    }

    /// A run gives every image the shown image's settings, and with them
    /// its background image: the bytes, not just the settings' reference
    /// to them, or the others would show and save no background.
    #[test]
    fn a_run_shares_the_shown_images_background_picture() {
        let mut batch = fixture();
        for id in 1..=3 {
            batch.items.push(item_with_cache(id, 0));
        }
        batch.items[0].set_bg_image(image::DynamicImage::new_rgba8(4, 4), None);
        batch.items[0].settings.gamma = 1.7;
        batch.share_settings(0, &std::collections::HashSet::from([1, 2]));
        let shared = &batch.items[1];
        assert_eq!(shared.settings.gamma, 1.7);
        assert!(shared.bg_image.is_some(), "the picture came along");
        assert_eq!(shared.bg_image.as_ref().map(|b| b.hash), shared.settings.bg_image_hash.map(|h| h.get()));
        assert_eq!(batch.items[2].settings.gamma, ItemSettings::default().gamma, "an image outside the run is untouched");
        assert!(batch.items[2].bg_image.is_none());
    }

    // ── new + selected_item ─────────────────────────────────────────────

    #[test]
    fn selected_item_returns_none_when_batch_empty() {
        let bm = fixture();
        assert!(bm.selected_item().is_none());
    }

    #[test]
    fn selected_item_returns_indexed_item() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(2, 0));
        bm.selected_index = 1;
        assert_eq!(bm.selected_item().unwrap().id, 2);
    }

    #[test]
    fn selected_item_returns_none_when_index_out_of_bounds() {
        // Defensive: callers should clamp, but the accessor must not panic.
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.selected_index = 99;
        assert!(bm.selected_item().is_none());
    }

    // ── is_selected ─────────────────────────────────────────────────────

    #[test]
    fn is_selected_matches_selected_item_id() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(42, 0));
        bm.selected_index = 1;
        assert!(bm.is_selected(42));
        assert!(!bm.is_selected(1));
    }

    #[test]
    fn is_selected_false_when_batch_empty() {
        let bm = fixture();
        assert!(!bm.is_selected(0));
        assert!(!bm.is_selected(99));
    }

    // ── selected_idx_clamped ────────────────────────────────────────────

    #[test]
    fn selected_idx_clamped_is_none_when_empty() {
        let bm = fixture();
        assert_eq!(bm.selected_idx_clamped(), None);
    }

    #[test]
    fn selected_idx_clamped_returns_index_when_in_bounds() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(2, 0));
        bm.selected_index = 1;
        assert_eq!(bm.selected_idx_clamped(), Some(1));
    }

    #[test]
    fn selected_idx_clamped_clamps_when_out_of_bounds() {
        // Guards against stale selected_index after items shrink (remove
        // operation) before selected_index has been updated.
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.selected_index = 99;
        assert_eq!(bm.selected_idx_clamped(), Some(0));
    }

    // ── status_counts ───────────────────────────────────────────────────

    #[test]
    fn status_counts_totals_by_status_excluding_pending() {
        let mut bm = fixture();
        let mut push = |id, status| {
            let mut item = item_with_cache(id, 0);
            item.status = status;
            bm.items.push(item);
        };
        push(1, BatchStatus::Done);
        push(2, BatchStatus::Done);
        push(3, BatchStatus::Processing);
        push(4, BatchStatus::Error("oops".into()));
        push(5, BatchStatus::Pending);

        let c = bm.status_counts();
        assert_eq!(c.done, 2);
        assert_eq!(c.processing, 1);
        assert_eq!(c.errored, 1);
        assert_eq!(c.batch_total(), 4, "Pending items are excluded from batch_total");
    }

    #[test]
    fn status_counts_empty_batch_zero() {
        let bm = fixture();
        let c = bm.status_counts();
        assert_eq!(c, StatusCounts::default());
        assert_eq!(c.batch_total(), 0);
    }

    #[test]
    fn progress_stage_says_processing_while_in_flight() {
        let mut bm = fixture();
        let mut push = |id, status| {
            let mut item = item_with_cache(id, 0);
            item.status = status;
            bm.items.push(item);
        };
        push(1, BatchStatus::Done);
        push(2, BatchStatus::Processing);
        push(3, BatchStatus::Pending);

        let report = bm.progress();
        assert_eq!(report.stage, "Processing 1/2");
        // pct = done(1) / batch_total(2) = 0.5; pending excluded from total.
        assert!((report.pct - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn progress_stage_says_finishing_when_no_in_flight() {
        let mut bm = fixture();
        let mut item = item_with_cache(1, 0);
        item.status = BatchStatus::Done;
        bm.items.push(item);

        let report = bm.progress();
        assert_eq!(report.stage, "Finishing up");
        assert!((report.pct - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn progress_empty_batch_no_division_by_zero() {
        let bm = fixture();
        let report = bm.progress();
        assert_eq!(report.stage, "Finishing up");
        assert_eq!(report.pct, 0.0);
    }

    #[test]
    fn first_error_message_returns_first_errored_item() {
        let mut bm = fixture();
        let mut push = |id, status| {
            let mut item = item_with_cache(id, 0);
            item.status = status;
            bm.items.push(item);
        };
        push(1, BatchStatus::Done);
        push(2, BatchStatus::Error("first problem".into()));
        push(3, BatchStatus::Error("second problem".into()));

        assert_eq!(bm.first_error_message(), Some("first problem"));
    }

    #[test]
    fn first_error_message_none_when_no_errors() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        assert_eq!(bm.first_error_message(), None);
    }

    // ── enforce_tensor_budget / evict_all_tensors ───────────────────────

    #[test]
    fn enforce_tensor_budget_no_op_below_budget() {
        let mut bm = fixture();
        // Tiny tensors well under 512 MB.
        bm.items.push(item_with_cache(1, 100));
        bm.items.push(item_with_cache(2, 100));
        let before_1 = bm.items[0].cached_tensor.is_some();
        let before_2 = bm.items[1].cached_tensor.is_some();
        bm.enforce_tensor_budget();
        assert_eq!(bm.items[0].cached_tensor.is_some(), before_1);
        assert_eq!(bm.items[1].cached_tensor.is_some(), before_2);
    }

    #[test]
    fn evict_all_tensors_clears_all_except_selected() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 100));
        bm.items.push(item_with_cache(2, 100));
        bm.items.push(item_with_cache(3, 100));
        bm.selected_index = 1; // item id=2
        // Sanity: all three start with caches.
        assert!(bm.items.iter().all(|i| i.cached_tensor.is_some()));

        bm.evict_all_tensors();

        assert!(bm.items[0].cached_tensor.is_none(), "non-selected evicted");
        assert!(bm.items[1].cached_tensor.is_some(), "selected preserved");
        assert!(bm.items[2].cached_tensor.is_none(), "non-selected evicted");
    }

    #[test]
    fn evict_all_tensors_clears_edge_cache_too() {
        let mut bm = fixture();
        let mut item = item_with_cache(1, 100);
        item.edge_planes.store_placeholder();
        bm.items.push(item);
        bm.items.push(item_with_cache(2, 100));
        bm.selected_index = 1; // selected = id=2

        bm.evict_all_tensors();

        // item id=1 (non-selected) gets BOTH caches cleared.
        assert!(bm.items[0].cached_tensor.is_none());
        assert!(bm.items[0].edge_planes.is_empty());
        assert!(bm.items[0].cached_edge_tensors.is_none());
    }

    // ── request_decode_bytes / request_thumbnail (thread-spawning) ──────

    /// 1x1 PNG byte sequence — minimum valid PNG.
    fn one_pixel_png() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 100, 50, 255]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn request_decode_bytes_emits_decoded_rgba_to_decode_rx() {
        let bm = fixture();
        let png = Arc::new(one_pixel_png());
        bm.request_decode_bytes(42, png);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 42);
        let rgba = result.expect("valid PNG must decode");
        assert_eq!(rgba.dimensions(), (1, 1));
        assert_eq!(rgba.get_pixel(0, 0).0, [200, 100, 50, 255]);
    }

    #[test]
    fn request_decode_source_for_bytes_variant_routes_through() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_decode_source(99, &source);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 99);
        assert!(result.is_ok());
    }

    /// A background picture loads off the UI thread with its content hash.
    #[test]
    fn a_background_picture_loads_with_its_hash() {
        let bm = fixture();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("bg.png");
        std::fs::write(&path, one_pixel_png()).unwrap();
        bm.request_bg_image(5, path.clone(), true);
        let load = bm.bg_io.bg_image_rx.recv_timeout(Duration::from_secs(2)).expect("lands within 2 s");
        assert_eq!((load.item_id, load.picked, load.path), (5, true, path));
        let (img, hash) = load.result.expect("decodes");
        assert_eq!(hash, crate::gui::item::bg_image_content_hash(&img));
    }

    /// A malformed image must surface a recognisable error via `Err` so
    /// the UI clears `decode_pending` and flags the item instead of
    /// leaving it stuck "still loading."
    #[test]
    fn request_decode_bytes_emits_err_on_malformed_input() {
        let bm = fixture();
        let garbage = Arc::new(b"not an image".to_vec());
        bm.request_decode_bytes(7, garbage);

        let (id, result) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 7);
        let err = result.expect_err("garbage bytes must produce Err");
        assert!(err.contains("decode failed"), "expected 'decode failed' prefix, got: {err}");
    }

    #[test]
    fn request_thumbnail_with_result_rgba_emits_to_thumb_rx() {
        let bm = fixture();
        let result = Arc::new(image::RgbaImage::from_pixel(200, 200, image::Rgba([10, 20, 30, 255])));
        let source = ImageSource::Bytes(Arc::new(Vec::new())); // unused when result_rgba is Some
        bm.request_thumbnail(7, &source, Some(&result));

        let (id, w, h, pixels) = bm.bg_io.thumb_rx.recv_timeout(Duration::from_secs(2))
            .expect("thumb_tx must produce a result within 2s");
        assert_eq!(id, 7);
        // 200x200 fits within 160x160 → scaled to 160x160.
        assert_eq!((w, h), (160, 160));
        assert_eq!(pixels.len(), (w * h * 4) as usize);
    }

    #[test]
    fn request_thumbnail_without_result_decodes_source() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_thumbnail(8, &source, None);

        let (id, w, h, pixels) = bm.bg_io.thumb_rx.recv_timeout(Duration::from_secs(2))
            .expect("thumb_tx must produce a result within 2s");
        assert_eq!(id, 8);
        // 1x1 fits trivially → stays 1x1.
        assert_eq!((w, h), (1, 1));
        assert_eq!(pixels.len(), 4);
    }

    // ── fit_dimensions (private helper) ─────────────────────────────────

    #[test]
    fn fit_dimensions_scales_down_oversized() {
        // 4000x2000 → fits inside 1000x1000 with aspect preserved.
        let (w, h) = fit_dimensions(4000, 2000, 1000, 1000);
        assert_eq!((w, h), (1000, 500));
    }

    #[test]
    fn fit_dimensions_does_not_upscale() {
        // Small image should NOT be upscaled to fill the box.
        let (w, h) = fit_dimensions(10, 10, 1000, 1000);
        assert_eq!((w, h), (10, 10));
    }

    #[test]
    fn fit_dimensions_minimum_size_is_one() {
        // Pathological case: extreme aspect ratio shouldn't produce zero dims.
        let (w, h) = fit_dimensions(1000, 1, 50, 50);
        assert!(w >= 1 && h >= 1);
    }

    // ── process_button_label ─────────────────────────────────────────────

    fn checked(id: u64) -> BatchItem {
        let mut i = item_with_cache(id, 0);
        i.ticked = true;
        i
    }

    #[test]
    fn process_button_label_empty_batch_is_viewed() {
        let bm = fixture();
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessViewed);
    }

    #[test]
    fn process_button_label_no_checkboxes_is_viewed() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(2, 0));
        bm.items.push(item_with_cache(3, 0));
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessViewed);
    }

    #[test]
    fn process_button_label_one_checkbox_of_three_is_selected_1() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(item_with_cache(2, 0));
        bm.items.push(item_with_cache(3, 0));
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessSelected(1));
    }

    #[test]
    fn process_button_label_two_checkboxes_of_three_is_selected_2() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(checked(2));
        bm.items.push(item_with_cache(3, 0));
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessSelected(2));
    }

    #[test]
    fn process_button_label_all_checked_multi_is_all() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(checked(2));
        bm.items.push(checked(3));
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessAll(3));
    }

    #[test]
    fn process_button_label_single_item_checked_is_selected_not_all() {
        // "Process All [1]" reads as a glitch — the t>=2 guard keeps it Selected.
        let mut bm = fixture();
        bm.items.push(checked(1));
        assert_eq!(bm.process_button_label(), ProcessButtonLabel::ProcessSelected(1));
    }

    // ── items_to_process ─────────────────────────────────────────────────

    #[test]
    fn items_to_process_empty_batch_is_empty() {
        let bm = fixture();
        assert!(bm.items_to_process().is_empty());
    }

    #[test]
    fn items_to_process_no_checkboxes_returns_viewed() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(10, 0));
        bm.items.push(item_with_cache(20, 0));
        bm.items.push(item_with_cache(30, 0));
        bm.selected_index = 1;
        assert_eq!(bm.items_to_process(), vec![20]);
    }

    #[test]
    fn items_to_process_no_checkboxes_clamps_stale_index() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(42, 0));
        bm.selected_index = 99; // stale; clamped to 0
        assert_eq!(bm.items_to_process(), vec![42]);
    }

    /// The three target-set readers agree on ticked items and on the item
    /// on screen; a stale index clamps for Process only, the other two
    /// target nothing.
    #[test]
    fn every_target_reader_picks_the_same_items() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(10, 0));
        bm.items.push(item_with_cache(20, 0));
        bm.items.push(item_with_cache(30, 0));
        bm.selected_index = 1;
        assert_eq!(bm.items_to_process(), vec![20]);
        assert_eq!(bm.targeted_indices(), vec![1]);
        assert!(bm.any_target_can(|i| i.id == 20));
        assert!(!bm.any_target_can(|i| i.id != 20));

        bm.items[0].ticked = true;
        bm.items[2].ticked = true;
        assert_eq!(bm.items_to_process(), vec![10, 30]);
        assert_eq!(bm.targeted_indices(), vec![0, 2]);
        assert!(bm.any_target_can(|i| i.id == 30));
        assert!(!bm.any_target_can(|i| i.id == 20));

        bm.set_all_ticked(false);
        bm.selected_index = 99;
        assert_eq!(bm.items_to_process(), vec![30]);
        assert!(bm.targeted_indices().is_empty());
        assert!(!bm.any_target_can(|_| true));

        let empty = fixture();
        assert!(empty.items_to_process().is_empty());
        assert!(empty.targeted_indices().is_empty());
        assert!(!empty.any_target_can(|_| true));
    }

    #[test]
    fn items_to_process_some_checkboxes_returns_only_checked() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(item_with_cache(2, 0));
        bm.items.push(checked(3));
        bm.selected_index = 1; // viewed item is NOT checked — still excluded
        let got = bm.items_to_process();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&1));
        assert!(got.contains(&3));
        assert!(!got.contains(&2));
    }

    #[test]
    fn items_to_process_all_checked_returns_all() {
        let mut bm = fixture();
        bm.items.push(checked(1));
        bm.items.push(checked(2));
        bm.items.push(checked(3));
        let got = bm.items_to_process();
        assert_eq!(got.len(), 3);
    }

    /// Filter-only Process must be off the UI thread — decodes + applies
    /// the fill style on a background thread; UI learns via `filter_only_rx`.
    #[test]
    fn request_filter_only_emits_processed_rgba_to_filter_only_rx() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_filter_only(7, &source, None, prunr_core::FillStyle::default());

        let (id, result, _) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
            .expect("filter_only_tx must produce a result within 2s");
        assert_eq!(id, 7);
        let rgba = result.expect("default FillStyle on a 1x1 source must succeed");
        assert_eq!(rgba.dimensions(), (1, 1));
    }

    /// Load failures must travel the same channel — the UI's drain maps
    /// `Err` to `BatchStatus::Error`. Without this the drain branch would
    /// be unreachable and bad sources would silently stall in `Processing`.
    #[test]
    fn request_filter_only_emits_load_failure_via_err_variant() {
        let bm = fixture();
        let source = ImageSource::Path(std::path::PathBuf::from(
            "/nonexistent/prunr-test-missing.png"
        ));
        bm.request_filter_only(11, &source, None, prunr_core::FillStyle::default());

        let (id, result, _) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
            .expect("filter_only_tx must produce a result within 2s");
        assert_eq!(id, 11);
        assert!(result.is_err(), "missing file must surface as Err, not stall");
    }

    // ── select_item ─────────────────────────────────────────────────────

    #[test]
    fn select_item_returns_true_when_index_changes() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.items.push(item_with_cache(2, 0));
        bm.selected_index = 0;
        assert!(bm.select_item(1));
        assert_eq!(bm.selected_index, 1);
    }

    #[test]
    fn select_item_returns_false_when_already_selected() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        bm.selected_index = 0;
        assert!(!bm.select_item(0));
        assert_eq!(bm.selected_index, 0);
    }

    // ── reorder ─────────────────────────────────────────────────────────

    #[test]
    fn reorder_moves_item_forward_and_adjusts_selected() {
        // [A B C D] selected=1(B), move from=0(A) to=3 → [B C A D] selected=0(B)
        let mut bm = fixture();
        for id in [10, 20, 30, 40] { bm.items.push(item_with_cache(id, 0)); }
        bm.selected_index = 1;
        bm.reorder(0, 3);
        assert_eq!(bm.items.iter().map(|i| i.id).collect::<Vec<_>>(), [20, 30, 10, 40]);
        assert_eq!(bm.selected_index, 0, "selected item B must follow its new position");
    }

    #[test]
    fn reorder_moves_item_backward_and_adjusts_selected() {
        // [A B C D] selected=2(C), move from=3(D) to=1 → [A D B C] selected=3(C)
        let mut bm = fixture();
        for id in [10, 20, 30, 40] { bm.items.push(item_with_cache(id, 0)); }
        bm.selected_index = 2;
        bm.reorder(3, 1);
        assert_eq!(bm.items.iter().map(|i| i.id).collect::<Vec<_>>(), [10, 40, 20, 30]);
        assert_eq!(bm.selected_index, 3, "selected item C must follow its new position");
    }

    #[test]
    fn reorder_noop_when_from_equals_to() {
        let mut bm = fixture();
        for id in [10, 20, 30] { bm.items.push(item_with_cache(id, 0)); }
        bm.selected_index = 1;
        bm.reorder(1, 1);
        assert_eq!(bm.items.iter().map(|i| i.id).collect::<Vec<_>>(), [10, 20, 30]);
        assert_eq!(bm.selected_index, 1);
    }

    #[test]
    fn reorder_noop_when_from_out_of_bounds() {
        let mut bm = fixture();
        for id in [10, 20] { bm.items.push(item_with_cache(id, 0)); }
        bm.reorder(5, 0);
        assert_eq!(bm.items.iter().map(|i| i.id).collect::<Vec<_>>(), [10, 20]);
    }

    // ── selection lifecycle ─────────────────────────────────────────────

    fn style(fill: f32) -> crate::gui::background_io::SelectionStyle {
        crate::gui::background_io::SelectionStyle::with_fill(fill)
    }

    #[test]
    fn ensure_selection_texture_requests_once_per_key() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert!(bm.items[0].selection_tex_pending.is_none(), "no selection: nothing to build");

        let mask = prunr_core::selection::MaskArtifact::new_empty(8, 8);
        let hash = mask.content_hash();
        bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke);
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.15))));
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.15))), "same key: single-flight");

        // A style change while the first build is in flight re-keys the request.
        bm.ensure_selection_texture(style(0.5), &ctx);
        assert_eq!(bm.items[0].selection_tex_pending, Some((hash, style(0.5))));
    }

    /// The overlay keeps its last texture while a new one builds, so it
    /// never flickers. With no selection left (undo of the first stroke)
    /// nothing replaces it: the texture must go, or the stroke stays drawn.
    #[test]
    fn the_overlay_texture_goes_with_the_last_selection() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        let mask = std::sync::Arc::new(prunr_core::selection::MaskArtifact::new_empty(8, 8));
        let handle = ctx.load_texture("sel", egui::ColorImage::new([8, 8], vec![egui::Color32::WHITE; 64]), Default::default());
        let item = &mut bm.items[0];
        item.selection_texture = Some(crate::gui::item::SelectionTexture {
            key: (1, style(0.15)), handle, shown: mask, feathered: None,
        });
        assert!(item.selection_hash.is_none(), "the selection was undone");
        bm.ensure_selection_texture(style(0.15), &ctx);
        assert!(bm.items[0].selection_texture.is_none());
    }

    /// A cut-out's re-cut applies the selection at the photo's own size,
    /// feathered once the overlay has built the feathered plane; never a
    /// copy shrunk to the model's working size.
    #[test]
    fn a_recut_applies_the_full_size_selection_and_its_feather() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let ctx = egui::Context::default();
        let cells: Vec<i8> = (0..64).map(|i| if i % 8 < 4 { -127 } else { 0 }).collect();
        let selection = std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, cells));
        let feathered = std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-60; 64]));
        let item = &mut bm.items[0];
        item.commit_selection_mask(std::sync::Arc::clone(&selection), crate::gui::timeline::StepKind::Stroke);
        let hash = item.selection_hash.expect("committed");
        let plane = item.correction_plane(style(0.15), crate::gui::brush_state::CutoutAction::Erase).expect("a selection");
        assert_eq!((plane.width, plane.height), (8, 8), "the photo's size, not the model's");
        assert!(std::sync::Arc::ptr_eq(&plane, &selection));

        let handle = ctx.load_texture("sel", egui::ColorImage::new([8, 8], vec![egui::Color32::WHITE; 64]), Default::default());
        item.selection_texture = Some(crate::gui::item::SelectionTexture {
            key: (hash, style(0.15)), handle, shown: selection, feathered: Some(std::sync::Arc::clone(&feathered)),
        });
        assert!(std::sync::Arc::ptr_eq(&item.correction_plane(style(0.15), crate::gui::brush_state::CutoutAction::Erase).unwrap(), &feathered), "the feathered plane once built");
    }

    /// A drag carries the ticked rows when it starts on one, else its own
    /// row; never a row still processing.
    #[test]
    fn a_drag_carries_the_ticked_rows_or_its_own() {
        let mut bm = fixture();
        for id in 0..3 {
            bm.items.push(item_with_cache(id, 0));
        }
        bm.items[0].ticked = true;
        bm.items[2].ticked = true;
        assert_eq!(bm.drag_targets(0), [0, 2]);
        assert_eq!(bm.drag_targets(1), [1]);
        bm.items[2].status = BatchStatus::Processing;
        assert_eq!(bm.drag_targets(0), [0]);
        bm.items[1].status = BatchStatus::Processing;
        assert!(bm.drag_targets(1).is_empty());
    }

    /// The Cancel button's "Selected" label and the partial-cancel gate
    /// read the same set.
    #[test]
    fn any_selected_with_status_agrees_with_the_ids() {
        let mut bm = fixture();
        for id in 0..3 {
            bm.items.push(item_with_cache(id, 0));
        }
        bm.items[1].status = BatchStatus::Processing;
        let agree = |bm: &BatchManager| {
            bm.any_ticked_with_status(BatchStatus::Processing)
                != bm.ticked_ids_with_status(BatchStatus::Processing).is_empty()
        };
        assert!(!bm.any_ticked_with_status(BatchStatus::Processing) && agree(&bm));
        bm.items[0].ticked = true;
        assert!(!bm.any_ticked_with_status(BatchStatus::Processing) && agree(&bm), "ticked but not processing");
        bm.set_all_ticked(true);
        assert!(bm.any_ticked_with_status(BatchStatus::Processing) && agree(&bm));
        bm.set_all_ticked(false);
        assert!(bm.items.iter().all(|i| !i.ticked));
    }

    #[test]
    fn commit_selection_sets_hash_in_lockstep() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        let expected_hash = mask.content_hash();
        assert!(bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke));
        let item = bm.find_by_id(1).unwrap();
        assert_eq!(
            item.selection_hash,
            Some(expected_hash),
            "selection_hash must mirror content_hash() of the committed mask",
        );
        assert!(item.selection_mask.is_some());
    }

    #[test]
    fn commit_selection_skips_byte_identical_mask() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        assert!(bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke));
        let identical = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        assert!(
            !bm.commit_selection(1, identical, crate::gui::timeline::StepKind::Stroke),
            "byte-identical mask must early-return without mutating caches",
        );
    }

    #[test]
    fn clear_selection_is_idempotent() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let mask = prunr_core::selection::MaskArtifact::new_empty(16, 16);
        bm.commit_selection(1, mask, crate::gui::timeline::StepKind::Stroke);
        assert!(bm.clear_selection(1), "first clear must return true (was present)");
        assert!(!bm.clear_selection(1), "second clear must return false (nothing to clear)");
    }

    /// Selection survives Process clicks: `reset_result_caches()` must
    /// NOT touch selection_mask / selection_hash.
    #[test]
    fn reset_result_caches_does_not_clear_selection() {
        let mut item = super::super::item::BatchItem::new_for_test();
        let mask = prunr_core::selection::MaskArtifact::new_empty(32, 32);
        let hash = mask.content_hash();
        item.selection_mask = Some(Arc::new(mask));
        item.selection_hash = Some(hash);
        item.reset_result_caches();
        assert!(
            item.selection_mask.is_some(),
            "selection_mask must survive reset_result_caches",
        );
        assert!(
            item.selection_hash.is_some(),
            "selection_hash must survive reset_result_caches",
        );
    }

    #[test]
    fn a_source_change_drops_the_selection_and_embedding() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        {
            let item = bm.find_by_id_mut(1).unwrap();
            let mask = prunr_core::selection::MaskArtifact::new_empty(8, 8);
            item.selection_mask = Some(Arc::new(mask));
            item.magic_brush_embedding = Some(Arc::new(prunr_core::sam::SamEmbedding {
                image_embed: vec![0.0; 1_048_576],
                high_res_feats_0: vec![0.0; 2_097_152],
                high_res_feats_1: vec![0.0; 1_048_576],
            }));
        }
        bm.find_by_id_mut(1).unwrap().invalidate_source_derived();
        let item = bm.find_by_id(1).unwrap();
        assert!(item.selection_mask.is_none(), "selection_mask must be cleared on source change");
        assert!(item.magic_brush_embedding.is_none(), "embedding must be cleared on source change");
    }

    /// A depth map belongs to the photo it was measured from: a source
    /// change drops it, and a run still measuring the old photo lands on
    /// nothing; one measuring the new photo lands.
    #[test]
    fn a_source_change_drops_the_depth_map_and_its_run() {
        let mut bm = fixture();
        bm.items.push(item_with_cache(1, 0));
        let map = || Ok(Arc::new(prunr_core::depth::DepthMap::from_raw(1, 1, vec![0.5])));
        let item = bm.find_by_id_mut(1).unwrap();
        item.depth_run.pending = true;
        let old_epoch = item.depth_run.epoch;
        assert!(item.land_depth(old_epoch, map()));
        item.depth_run.pending = true;
        bm.find_by_id_mut(1).unwrap().invalidate_source_derived();
        let item = bm.find_by_id_mut(1).unwrap();
        assert!(item.depth.is_none() && !item.depth_run.pending, "the map and its run are dropped");
        assert!(item.needs_depth_run());
        assert!(!item.land_depth(old_epoch, map()), "a run on the old photo lands on nothing");
        assert!(item.depth.is_none());
        let epoch = item.depth_run.epoch;
        assert!(item.land_depth(epoch, map()));
        assert!(item.depth.is_some());
    }
}
