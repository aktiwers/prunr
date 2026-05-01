use std::sync::mpsc;

/// Filter-only Process result: `Ok(rgba)` on success, `Err(msg)` so load /
/// decode failures take the same channel path as successful results
/// (drained in `drain_background_channels`, mapped to `BatchStatus::Error`).
pub type FilterOnlyResult = (u64, Result<std::sync::Arc<image::RgbaImage>, String>);

/// Bundles all background thread communication channels.
pub struct BackgroundIO {
    /// File paths from file dialog / drag-and-drop (loaded lazily on demand)
    pub file_load_tx: mpsc::Sender<(std::path::PathBuf, String)>,
    pub file_load_rx: mpsc::Receiver<(std::path::PathBuf, String)>,
    /// Thumbnail generation results
    pub thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
    pub thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    /// Pre-decoded source images for instant canvas switching
    pub decode_tx: mpsc::Sender<(u64, std::sync::Arc<image::RgbaImage>)>,
    pub decode_rx: mpsc::Receiver<(u64, std::sync::Arc<image::RgbaImage>)>,
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
        Self {
            file_load_tx, file_load_rx,
            thumb_tx, thumb_rx,
            decode_tx, decode_rx,
            save_done_tx, save_done_rx,
            tex_prep_tx, tex_prep_rx,
            filter_only_tx, filter_only_rx,
        }
    }
}
