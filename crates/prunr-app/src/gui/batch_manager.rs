//! Batch state, lifecycle, memory governance, and texture coordination.
//!
//! Owns:
//! - The `Vec<BatchItem>` and selection state
//! - The next-id counter for new items
//! - The cloned `egui::Context` for `request_repaint` from background paths
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

pub(crate) struct BatchManager {
    pub(crate) items: Vec<BatchItem>,
    pub(crate) selected_index: usize,
    pub(crate) next_id: u64,
    pub(crate) bg_io: BackgroundIO,
    #[allow(dead_code)] // used by methods that move in 10-05d's later sub-stages / 10-05e
    pub(crate) egui_ctx: egui::Context,
}

impl BatchManager {
    pub(crate) fn new(egui_ctx: egui::Context) -> Self {
        Self {
            items: Vec::new(),
            selected_index: 0,
            next_id: 0,
            bg_io: BackgroundIO::new(),
            egui_ctx,
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
                let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
                let thumb = image::imageops::resize(rgba.as_ref(), w, h, image::imageops::FilterType::Triangle);
                let _ = tx.send((item_id, thumb.width(), thumb.height(), thumb.into_raw()));
            });
        } else {
            let source = source.clone();
            std::thread::spawn(move || {
                if let Ok(bytes) = source.load_bytes() {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let rgba = img.to_rgba8();
                        let (w, h) = fit_dimensions(rgba.width(), rgba.height(), 160, 160);
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
