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
//! - Inference dispatch (worker channels, admission, live-preview) — those
//!   move to `Processor` in 10-05e
//! - History mutation logic — `HistoryManager` (no own state) handles that
//! - Drag-out lifecycle — `DragExportState` owns that

use std::sync::Arc;

use super::background_io::BackgroundIO;
use super::item::{BatchItem, ImageSource};

/// Combined budget for compressed segmentation (`cached_tensor`) + edge
/// (`cached_edge_tensor`) caches across all batch items, in bytes.
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
        assert!(bm.items[0].cached_edge_tensor.is_none());
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
}
