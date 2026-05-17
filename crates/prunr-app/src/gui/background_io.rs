use std::sync::{mpsc, Arc, Condvar, Mutex};

/// Filter-only Process result: `Ok(rgba)` on success, `Err(msg)` so load /
/// decode failures take the same channel path as successful results
/// (drained in `drain_background_channels`, mapped to `BatchStatus::Error`).
pub type FilterOnlyResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// Pre-decode result: `Ok(rgba)` on success, `Err(msg)` so a malformed
/// image clears `decode_pending` instead of leaving the item stuck.
pub type DecodeResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// History-eviction → off-thread compress → drain payload. The bg
/// thread sends `(item_id, compressed_entry, recipe)` once
/// `compress_to_ram` returns; the drain swaps the in-memory placeholder
/// at `history.back()` for the compressed slot. Recipe is matched at
/// drain time so a freshly-processed result can't be clobbered by a
/// stale eviction.
pub type HistoryDemoteResult = (
    u64,
    super::history_disk::CompressedEntry,
    Option<prunr_core::ProcessingRecipe>,
);

/// Plan 05: selection outline built off-thread after every commit_selection.
/// Drain consumes via `drain_background_channels` with hash-stale-discard guard.
pub(crate) struct SelectionOutlineResult {
    pub(crate) item_id: u64,
    pub(crate) outline: Vec<(u32, u32)>,
    pub(crate) hash: u64,
}

/// Plan 05: selection fill texture (ACCENT-tinted ColorImage) built off-thread.
/// Peak RAM on a 4K image: 8.3M pixels × 4 bytes = ~33 MB briefly on the rayon
/// worker; drops after `ctx.load_texture` in `drain_background_channels`.
pub(crate) struct SelectionTextureResult {
    pub(crate) item_id: u64,
    pub(crate) color_image: egui::ColorImage,
    pub(crate) hash: u64,
}

/// Counting semaphore used to bound the number of simultaneously-decoding
/// background threads. Without this, a 50-image Process All fans out 50
/// threads each holding `compressed bytes + DynamicImage + RgbaImage`
/// (~50–80 MB at 4 K) → multi-GB transient before any thread releases.
/// Threads still spawn immediately; they park on `acquire` until a slot
/// opens. Cap is `available_parallelism()` so cold-cache disk paths
/// still saturate cores.
pub struct DecodeSlots {
    state: Mutex<usize>,
    cv: Condvar,
}

impl DecodeSlots {
    pub fn new(slots: usize) -> Self {
        Self { state: Mutex::new(slots.max(1)), cv: Condvar::new() }
    }
    pub fn acquire(self: &Arc<Self>) -> DecodeSlotGuard {
        let mut s = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while *s == 0 {
            s = self.cv.wait(s).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *s -= 1;
        DecodeSlotGuard { sem: self.clone() }
    }
}

pub struct DecodeSlotGuard {
    sem: Arc<DecodeSlots>,
}

impl Drop for DecodeSlotGuard {
    fn drop(&mut self) {
        let mut s = self.sem.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *s += 1;
        self.sem.cv.notify_one();
    }
}

/// Owned handles to the texture-prep channel + decode slots, bundled so
/// `PrunrApp::spawn_tex_prep` takes one parameter instead of two. The
/// `Clone` derive is shallow (Sender is Clone, Arc is Clone) and cheap.
#[derive(Clone)]
pub struct TexPrepHandles {
    pub tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
    pub slots: Arc<DecodeSlots>,
}

/// Bundles all background thread communication channels.
pub struct BackgroundIO {
    /// File paths from file dialog / drag-and-drop (loaded lazily on demand)
    pub file_load_tx: mpsc::Sender<(std::path::PathBuf, String)>,
    pub file_load_rx: mpsc::Receiver<(std::path::PathBuf, String)>,
    /// Thumbnail generation results
    pub thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
    pub thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    /// Pre-decoded source images for instant canvas switching
    pub decode_tx: mpsc::Sender<DecodeResult>,
    pub decode_rx: mpsc::Receiver<DecodeResult>,
    /// Save completion notifications
    pub save_done_tx: mpsc::Sender<String>,
    pub save_done_rx: mpsc::Receiver<String>,
    /// Pre-built ColorImages ready for GPU upload (item_id, texture_name, image, is_result)
    pub tex_prep_tx: mpsc::Sender<(u64, String, egui::ColorImage, bool)>,
    pub tex_prep_rx: mpsc::Receiver<(u64, String, egui::ColorImage, bool)>,
    /// Filter-only (model=None) Process results — keeps the UI thread free
    /// while load + decode + `apply_fill_style` runs per item.
    pub filter_only_tx: mpsc::Sender<FilterOnlyResult>,
    pub filter_only_rx: mpsc::Receiver<FilterOnlyResult>,
    /// Off-thread zstd compression of evicted background-item RGBAs.
    /// Eviction places an in-memory placeholder at `history.back()` and
    /// kicks a worker that compresses + posts here; drain swaps the
    /// placeholder for `HistorySlot::Compressed`. Without this the
    /// zstd encode (~10–50 ms per 4K image × N items) ran inline on
    /// every selection change.
    pub history_demote_tx: mpsc::Sender<HistoryDemoteResult>,
    pub history_demote_rx: mpsc::Receiver<HistoryDemoteResult>,
    /// Off-thread RGBA decode + ColorImage build for a per-item bg
    /// image. Drain calls `ctx.load_texture` with the bg-specific
    /// options (Linear filter + Repeat wrap for `BgImageFit::Tile`).
    /// Without this the `to_rgba8()` clone (~32 MB on a 4K bg) +
    /// `ctx.load_texture` ran inline in `views/canvas.rs::render` on
    /// the first frame after every bg-image pick.
    pub bg_tex_prep_tx: mpsc::Sender<(u64, u64, egui::ColorImage)>,
    pub bg_tex_prep_rx: mpsc::Receiver<(u64, u64, egui::ColorImage)>,
    /// Bounds simultaneous decode/thumbnail/filter threads to
    /// `available_parallelism()`. Threads spawn immediately but park here
    /// until a slot opens, capping transient RAM at N × per-thread peak.
    pub decode_slots: Arc<DecodeSlots>,
    /// Plan 05: selection outline (Vec<(u32,u32)> boundary pixels) built
    /// off-thread after every commit_selection. Hash-stale-discard guard
    /// in drain_background_channels. `(crate)` — only the drain + request paths touch this.
    pub(crate) selection_outline_tx: mpsc::Sender<SelectionOutlineResult>,
    pub(crate) selection_outline_rx: mpsc::Receiver<SelectionOutlineResult>,
    /// Plan 05: selection fill texture (ACCENT-tinted egui::ColorImage) built
    /// off-thread after every commit_selection. Drained via ctx.load_texture
    /// (allowed — drain_background_channels runs in logic(), not a render closure).
    pub(crate) selection_texture_tx: mpsc::Sender<SelectionTextureResult>,
    pub(crate) selection_texture_rx: mpsc::Receiver<SelectionTextureResult>,
}

impl BackgroundIO {
    /// Clone the (tx, slots) pair as one handle bundle for spawn paths.
    pub fn tex_prep_handles(&self) -> TexPrepHandles {
        TexPrepHandles {
            tx: self.tex_prep_tx.clone(),
            slots: self.decode_slots.clone(),
        }
    }
}

impl Default for BackgroundIO {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundIO {
    pub fn new() -> Self {
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();
        let (decode_tx, decode_rx) = mpsc::channel();
        let (save_done_tx, save_done_rx) = mpsc::channel();
        let (tex_prep_tx, tex_prep_rx) = mpsc::channel();
        let (filter_only_tx, filter_only_rx) = mpsc::channel();
        let (history_demote_tx, history_demote_rx) = mpsc::channel();
        let (bg_tex_prep_tx, bg_tex_prep_rx) = mpsc::channel();
        let (selection_outline_tx, selection_outline_rx) = mpsc::channel();
        let (selection_texture_tx, selection_texture_rx) = mpsc::channel();
        let cap = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            file_load_tx, file_load_rx,
            thumb_tx, thumb_rx,
            decode_tx, decode_rx,
            save_done_tx, save_done_rx,
            tex_prep_tx, tex_prep_rx,
            filter_only_tx, filter_only_rx,
            history_demote_tx, history_demote_rx,
            bg_tex_prep_tx, bg_tex_prep_rx,
            decode_slots: Arc::new(DecodeSlots::new(cap)),
            selection_outline_tx, selection_outline_rx,
            selection_texture_tx, selection_texture_rx,
        }
    }

    /// Spawn an off-thread job to build the selection outline polyline and
    /// ACCENT-tinted ColorImage for a newly-committed MaskArtifact.
    ///
    /// Peak RAM on the worker: outline Vec<(u32,u32)> + ColorImage pixels.
    /// For a 4K image (8.3M pixels): outline ≈ sparse Vec (boundary only),
    /// ColorImage = 8.3M × 4 bytes ≈ 33 MB briefly, dropped after drain.
    ///
    /// Results carry the mask's `content_hash`. The drain path discards any
    /// result whose hash no longer matches `item.selection_hash` — guards
    /// against a newer commit racing a still-running worker.
    pub(crate) fn request_selection_visualization(
        &self,
        item_id: u64,
        mask: std::sync::Arc<prunr_core::selection::MaskArtifact>,
        hash: u64,
        _edge_feather_px: u32,
    ) {
        // _edge_feather_px is reserved for the source-RGBA-guided feather
        // wiring (see DEFERRED.md). Until that lands, feather has no
        // effect on the visualization — the parameter stays in the
        // signature so callers can be wired straight through later.
        let outline_tx = self.selection_outline_tx.clone();
        let texture_tx = self.selection_texture_tx.clone();
        rayon::spawn(move || {
            let mut outline = prunr_core::selection::refine::outline_polyline(&mask);
            decimate_outline_in_place(&mut outline, OUTLINE_MAX_POINTS);
            let _ = outline_tx.send(SelectionOutlineResult { item_id, outline, hash });

            // ACCENT-tinted ColorImage: pixels where mask >= 0.5 carry
            // full-opacity ACCENT; render-time fill_opacity scales the
            // alpha so we ship pre-tinted at 255 here.
            let w = mask.width as usize;
            let h = mask.height as usize;
            let mut pixels = Vec::with_capacity(w * h);
            for &v in mask.data.iter() {
                if v >= 0.5 {
                    pixels.push(egui::Color32::from_rgba_unmultiplied(0x7b, 0x2d, 0x8e, 255));
                } else {
                    pixels.push(egui::Color32::TRANSPARENT);
                }
            }
            let color_image = egui::ColorImage::new([w, h], pixels);
            let _ = texture_tx.send(SelectionTextureResult { item_id, color_image, hash });
        });
    }
}

/// Cap on the outline polyline length sent to the render closure. Above
/// this length the renderer allocates a Vec<Pos2> per frame whose size
/// dominates the per-frame heap churn — and a polyline with > 50k vertices
/// is well past the screen-pixel density at any reasonable canvas size,
/// so uniform-stride decimation costs nothing visible.
const OUTLINE_MAX_POINTS: usize = 50_000;

fn decimate_outline_in_place(outline: &mut Vec<(u32, u32)>, cap: usize) {
    if outline.len() <= cap {
        return;
    }
    let stride = outline.len() as f32 / cap as f32;
    let original = std::mem::take(outline);
    outline.reserve(cap);
    for i in 0..cap {
        let idx = ((i as f32) * stride) as usize;
        outline.push(original[idx.min(original.len() - 1)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_under_cap_is_noop() {
        let mut v = vec![(0u32, 0u32), (1, 1), (2, 2)];
        decimate_outline_in_place(&mut v, 10);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn decimate_over_cap_yields_exactly_cap_entries() {
        let mut v: Vec<(u32, u32)> = (0..200_000u32).map(|i| (i, i)).collect();
        decimate_outline_in_place(&mut v, OUTLINE_MAX_POINTS);
        assert_eq!(v.len(), OUTLINE_MAX_POINTS);
    }
}
