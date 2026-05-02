//! Batch state, lifecycle, memory governance, and texture coordination.
//!
//! Owns:
//! - The `Vec<BatchItem>` and selection state
//! - The next-id counter for new items
//! - The `BackgroundIO` channels (thumbnail, decode, save-done, tex-prep,
//!   file-load) — historically mixed but the majority are batch-driven.
//!   Future `Saver` / `ImageOpener` extractions may carve out the file-load
//!   and save-done channels.
//!
//! Does NOT own:
//! - Inference dispatch (worker channels, admission, live-preview) — `Processor`
//! - History mutation logic — `HistoryManager` (no own state) handles that
//! - Drag-out lifecycle — `DragExportState`

use std::sync::Arc;

use super::background_io::BackgroundIO;
use super::item::{BatchItem, BatchStatus, ImageSource};
use super::state::AppState;

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
}

impl BatchManager {
    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            selected_index: 0,
            next_id: 0,
            bg_io: BackgroundIO::new(),
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

    /// IDs of checkbox-selected items currently in a given status. Used by
    /// the partial-cancel path to decide which in-flight items to stop.
    pub(crate) fn selected_ids_with_status(&self, status: BatchStatus) -> Vec<u64> {
        self.items.iter()
            .filter(|i| i.selected && i.status == status)
            .map(|i| i.id)
            .collect()
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
        let selected = self.selected_count();
        match (selected, total) {
            (0, _) => ProcessButtonLabel::ProcessViewed,
            (n, t) if n == t && t >= 2 => ProcessButtonLabel::ProcessAll(t),
            (n, _) => ProcessButtonLabel::ProcessSelected(n),
        }
    }

    /// Item ids the Process button should dispatch: the checkbox-checked set
    /// if any are checked, else the currently-viewed item. Empty when the
    /// batch is empty.
    pub(crate) fn items_to_process(&self) -> Vec<u64> {
        let checked: Vec<u64> = self.items.iter()
            .filter(|i| i.selected)
            .map(|i| i.id)
            .collect();
        if !checked.is_empty() {
            return checked;
        }
        self.selected_idx_clamped()
            .and_then(|idx| self.items.get(idx).map(|i| i.id))
            .into_iter()
            .collect()
    }

    /// Any item in the action target set satisfies `f`. Target set = checked
    /// items if any, else the currently-viewed one — same rule as
    /// `items_to_process`, expressed as a fold so callers (toolbar
    /// Undo/Redo enablement) don't allocate an id vec just to test.
    pub(crate) fn any_target_can<F: FnMut(&BatchItem) -> bool>(&self, mut f: F) -> bool {
        let has_selected = self.has_any_selected();
        let current_id = self.selected_item().map(|b| b.id);
        self.items.iter().any(|item| {
            let is_target = if has_selected { item.selected } else { Some(item.id) == current_id };
            is_target && f(item)
        })
    }

    /// First (in batch order) item in the action target set, by reference —
    /// no Vec<u64> allocation. Per-frame toolbar tooltip code uses this
    /// instead of `items_to_process().first()` so the Process button label
    /// path doesn't churn ~400 B per frame. Same target-set rule as
    /// `items_to_process` and `any_target_can`.
    pub(crate) fn first_target_item(&self) -> Option<&BatchItem> {
        let has_selected = self.has_any_selected();
        let current_id = self.selected_item().map(|b| b.id);
        self.items.iter().find(|item| {
            if has_selected { item.selected } else { Some(item.id) == current_id }
        })
    }

    /// Remove the item at `idx`, cleaning up any on-disk history /
    /// redo entries it owned. Returns `false` (without doing anything)
    /// when `idx` is out of bounds. Caller is responsible for clamping
    /// `selected_index` and triggering any downstream sync — this helper
    /// only mutates `items` and the removed entries' on-disk state.
    pub(crate) fn remove(&mut self, idx: usize) -> bool {
        if idx >= self.items.len() { return false; }
        let item = self.items.remove(idx);
        for entry in item.history { entry.cleanup(); }
        for entry in item.redo_stack { entry.cleanup(); }
        true
    }

    /// Flip every `Processing` item back to `Pending`. Used after a
    /// Cancel-All so spinners stop and items can be re-Processed without
    /// going through the error path. Late `ImageDone` arrivals from the
    /// worker / subprocess are silently dropped by the Processing-only
    /// guard in `on_batch_item_done`.
    pub(crate) fn reset_processing_to_pending(&mut self) {
        for item in &mut self.items {
            if item.status == BatchStatus::Processing {
                item.status = BatchStatus::Pending;
            }
        }
    }

    /// Number of items with the checkbox set. Single source of truth so
    /// statusbar / sidebar don't re-roll `iter().filter(...).count()`.
    pub(crate) fn selected_count(&self) -> usize {
        self.items.iter().filter(|i| i.selected).count()
    }

    /// True when at least one item has the checkbox set.
    pub(crate) fn has_any_selected(&self) -> bool {
        self.items.iter().any(|i| i.selected)
    }

    /// True when every item has the checkbox set (and the batch is
    /// non-empty — an empty batch is not "all selected").
    pub(crate) fn all_selected(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|i| i.selected)
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

    /// Filter-only Process: load + decode + `apply_fill_style` on a background
    /// thread, deliver the result via `bg_io.filter_only_rx`. Keeps the UI
    /// thread free on large batches that the inline path used to freeze.
    pub(crate) fn request_filter_only(
        &self,
        item_id: u64,
        source: &ImageSource,
        fill_style: prunr_core::FillStyle,
    ) {
        let tx = self.bg_io.filter_only_tx.clone();
        let source = source.clone();
        let slots = self.bg_io.decode_slots.clone();
        std::thread::spawn(move || {
            let _slot = slots.acquire();
            // Drop bytes / DynamicImage / RGBA before send so concurrent
            // threads don't pile per-image peaks.
            let result: Result<Arc<image::RgbaImage>, String> = (|| {
                let bytes = source.load_bytes()
                    .map_err(|e| format!("Failed to load: {e}"))?;
                let original = prunr_core::load_image_from_bytes(&bytes)
                    .map_err(|_| "Decode failed".to_string())?;
                drop(bytes);
                let mut rgba = original.to_rgba8();
                drop(original);
                prunr_core::apply_fill_style(&mut rgba, fill_style);
                Ok(Arc::new(rgba))
            })();
            let _ = tx.send((item_id, result));
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
                let thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
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
                    let rgba = img.to_rgba8();
                    drop(img);
                    let (w, h) = fit_dimensions(rgba.width(), rgba.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                    Some(image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Triangle))
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
        let total: usize = self.items.iter().map(BatchItem::cache_size).sum();
        if total <= TENSOR_BUDGET { return; }
        let selected_id = self.selected_item().map(|b| b.id);
        let mut remaining = total;
        for item in &mut self.items {
            if remaining <= TENSOR_BUDGET { break; }
            // Preserve the selected item's tensors (most likely to be reused).
            if Some(item.id) == selected_id { continue; }
            remaining -= item.cache_size();
            item.cached_tensor = None;
            item.invalidate_edge_cache();
        }
    }

    /// Evict all tensor caches except the selected item (called under memory pressure).
    pub(crate) fn evict_all_tensors(&mut self) {
        let selected_id = self.selected_item().map(|b| b.id);
        for item in &mut self.items {
            if Some(item.id) != selected_id {
                item.cached_tensor = None;
                item.invalidate_edge_cache();
            }
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

    /// Construct a BatchItem with optional cached tensor of `bytes_uncompressed`
    /// raw size. Used by enforce_tensor_budget / cache_size tests.
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
            item.cached_tensor = CompressedTensor::from_raw(cache);
        }
        item
    }

    // ── new + selected_item ─────────────────────────────────────────────

    // (BatchManager::new defaults are covered indirectly by tests/batch_tests.rs
    //  which assert the same fields via PrunrApp::new_for_test → BatchManager::new.)

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
        item.cached_edge_mask = Some((Arc::new(image::GrayImage::new(1, 1)), 0, prunr_core::EdgeScale::Fused));
        bm.items.push(item);
        bm.items.push(item_with_cache(2, 100));
        bm.selected_index = 1; // selected = id=2

        bm.evict_all_tensors();

        // item id=1 (non-selected) gets BOTH caches cleared.
        assert!(bm.items[0].cached_tensor.is_none());
        assert!(bm.items[0].cached_edge_mask.is_none());
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
        i.selected = true;
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
        bm.request_filter_only(7, &source, prunr_core::FillStyle::default());

        let (id, result) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
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
        bm.request_filter_only(11, &source, prunr_core::FillStyle::default());

        let (id, result) = bm.bg_io.filter_only_rx.recv_timeout(Duration::from_secs(2))
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
}
