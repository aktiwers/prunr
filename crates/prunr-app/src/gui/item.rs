//! Pure data types for batch items and the per-image history they carry.
//!
//! These types are GUI-agnostic in spirit (they hold egui texture handles
//! because the texture lifecycle is per-item, but no rendering happens here).
//! Logic that mutates these types lives in the coordinators
//! (`HistoryManager`, `BatchManager`, `Processor`).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

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
                    .map(|img| Arc::new(img))
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

impl Default for HistoryEntry {
    fn default() -> Self {
        Self { slot: HistorySlot::default(), recipe: None }
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
    /// Compressed cached DexiNed output (for Tier 2 edge reruns on line_strength tweaks).
    pub(crate) cached_edge_tensor: Option<super::worker::CompressedTensor>,
    /// Post-resize, pre-dilation edge mask for the line_strength that produced
    /// it. Lets `edge_thickness` / `solid_line_color` tweaks skip the expensive
    /// tensor→mask resize. Invalidated alongside `cached_edge_tensor`.
    pub(crate) cached_edge_mask: Option<(Arc<image::GrayImage>, u32 /* line_strength bits */)>,
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
}

impl BatchItem {
    /// Clear the edge tensor + derived mask cache together — the mask is
    /// always built from the tensor, so any tensor change invalidates both.
    pub(crate) fn invalidate_edge_cache(&mut self) {
        self.cached_edge_tensor = None;
        self.cached_edge_mask = None;
    }

    /// Reset all caches tied to the current result. Call after the result
    /// has changed (history walk, fresh process, etc.) so the next paint
    /// rebuilds textures and the next reprocess re-runs from scratch.
    /// Note: `source_texture` is NOT cleared — callers decide whether the
    /// source view also needs rebuilding (undo: yes; redo: no).
    pub(crate) fn reset_result_caches(&mut self) {
        self.cached_tensor = None;
        self.result_texture = None;
        self.thumb_texture = None;
        self.thumb_pending = false;
        self.source_tex_pending = false;
        self.result_tex_pending = false;
        self.decode_pending = false;
    }

    /// Combined compressed size of segmentation + edge tensor caches.
    /// Used by memory governance (`BatchManager::enforce_tensor_budget`)
    /// and any future telemetry / HUD readout.
    pub(crate) fn cache_size(&self) -> usize {
        let seg = self.cached_tensor.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
        let edge = self.cached_edge_tensor.as_ref().map(|ct| ct.compressed_size()).unwrap_or(0);
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
            cached_edge_tensor: None,
            cached_edge_mask: None,
            applied_preset,
            preset_undo_stack: VecDeque::new(),
            preset_redo_stack: VecDeque::new(),
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
        item.cached_edge_mask = Some((Arc::new(image::GrayImage::new(1, 1)), 0));
        // (cached_edge_tensor would need a real CompressedTensor — leave None
        // here; the method should still run cleanly and clear cached_edge_mask.)
        assert!(item.cached_edge_mask.is_some());
        item.invalidate_edge_cache();
        assert!(item.cached_edge_tensor.is_none());
        assert!(item.cached_edge_mask.is_none());
    }

    #[test]
    fn reset_result_caches_clears_expected_fields() {
        let mut item = fixture_item(1);
        item.thumb_pending = true;
        item.source_tex_pending = true;
        item.result_tex_pending = true;
        item.decode_pending = true;
        item.reset_result_caches();
        assert!(item.cached_tensor.is_none());
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
}
