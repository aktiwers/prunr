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

/// Which shape the Process button should take for the current selection state.
/// Drives both the button label and the icon the toolbar renders.
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

    /// True when the currently-selected item has the given id. Callers use
    /// this to decide whether a worker message / background-io event affects
    /// what the user is looking at right now (textures, progress, status).
    pub(crate) fn is_selected(&self, id: u64) -> bool {
        self.selected_item().map_or(false, |b| b.id == id)
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

    /// Derive the Process button's label shape from current selection state.
    /// Single-item batches where the one item is checked render as
    /// `ProcessSelected(1)` (not `ProcessAll(1)`) — "Process All [1]" reads
    /// as a glitch.
    pub(crate) fn process_button_label(&self) -> ProcessButtonLabel {
        let total = self.items.len();
        let selected = self.items.iter().filter(|i| i.selected).count();
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
        let has_selected = self.items.iter().any(|i| i.selected);
        let current_id = self.selected_item().map(|b| b.id);
        self.items.iter().any(|item| {
            let is_target = if has_selected { item.selected } else { Some(item.id) == current_id };
            is_target && f(item)
        })
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

    /// Pre-decode source bytes to RgbaImage on a background thread; the
    /// result lands on `bg_io.decode_rx` for the main thread to attach.
    pub(crate) fn request_decode_bytes(&self, item_id: u64, bytes: Arc<Vec<u8>>) {
        let tx = self.bg_io.decode_tx.clone();
        std::thread::spawn(move || {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let _ = tx.send((item_id, Arc::new(img.to_rgba8())));
            }
        });
    }

    /// Pre-decode from an ImageSource (reads file if needed).
    pub(crate) fn request_decode_source(&self, item_id: u64, source: &ImageSource) {
        if let Ok(bytes) = source.load_bytes() {
            self.request_decode_bytes(item_id, bytes);
        }
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
        if let Some(rgba) = result_rgba {
            let rgba = rgba.clone();
            std::thread::spawn(move || {
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                let thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let source = source.clone();
            std::thread::spawn(move || {
                if let Ok(bytes) = source.load_bytes() {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let rgba = img.to_rgba8();
                        let (w, h) = fit_dimensions(rgba.width(), rgba.height(), THUMB_MAX_PX, THUMB_MAX_PX);
                        let thumb = image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Triangle);
                        let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
                    }
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
        item.cached_edge_mask = Some((Arc::new(image::GrayImage::new(1, 1)), 0));
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

        let (id, rgba) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 42);
        assert_eq!(rgba.dimensions(), (1, 1));
        assert_eq!(rgba.get_pixel(0, 0).0, [200, 100, 50, 255]);
    }

    #[test]
    fn request_decode_source_for_bytes_variant_routes_through() {
        let bm = fixture();
        let source = ImageSource::Bytes(Arc::new(one_pixel_png()));
        bm.request_decode_source(99, &source);

        let (id, _rgba) = bm.bg_io.decode_rx.recv_timeout(Duration::from_secs(2))
            .expect("decode_tx must produce a result within 2s");
        assert_eq!(id, 99);
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
}
