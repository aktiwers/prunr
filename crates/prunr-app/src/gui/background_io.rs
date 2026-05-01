use std::sync::{mpsc, Arc, Condvar, Mutex};

/// Filter-only Process result: `Ok(rgba)` on success, `Err(msg)` so load /
/// decode failures take the same channel path as successful results
/// (drained in `drain_background_channels`, mapped to `BatchStatus::Error`).
pub type FilterOnlyResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// Pre-decode result: `Ok(rgba)` on success, `Err(msg)` so a malformed
/// image clears `decode_pending` instead of leaving the item stuck.
pub type DecodeResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

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
    /// Bounds simultaneous decode/thumbnail/filter threads to
    /// `available_parallelism()`. Threads spawn immediately but park here
    /// until a slot opens, capping transient RAM at N × per-thread peak.
    pub decode_slots: Arc<DecodeSlots>,
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
            decode_slots: Arc::new(DecodeSlots::new(cap)),
        }
    }
}
