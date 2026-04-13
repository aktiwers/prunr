use std::sync::mpsc;

/// Bundles all background thread communication channels.
pub struct BackgroundIO {
    /// Files loaded by background thread (file dialog / drag-and-drop)
    pub file_load_tx: mpsc::Sender<(Vec<u8>, String)>,
    pub file_load_rx: mpsc::Receiver<(Vec<u8>, String)>,
    /// Thumbnail generation results
    pub thumb_tx: mpsc::Sender<(u64, u32, u32, Vec<u8>)>,
    pub thumb_rx: mpsc::Receiver<(u64, u32, u32, Vec<u8>)>,
    /// Pre-decoded source images for instant canvas switching
    pub decode_tx: mpsc::Sender<(u64, std::sync::Arc<image::RgbaImage>)>,
    pub decode_rx: mpsc::Receiver<(u64, std::sync::Arc<image::RgbaImage>)>,
    /// Save completion notifications
    pub save_done_tx: mpsc::Sender<String>,
    pub save_done_rx: mpsc::Receiver<String>,
}

impl BackgroundIO {
    pub fn new() -> Self {
        let (file_load_tx, file_load_rx) = mpsc::channel();
        let (thumb_tx, thumb_rx) = mpsc::channel();
        let (decode_tx, decode_rx) = mpsc::channel();
        let (save_done_tx, save_done_rx) = mpsc::channel();
        Self {
            file_load_tx, file_load_rx,
            thumb_tx, thumb_rx,
            decode_tx, decode_rx,
            save_done_tx, save_done_rx,
        }
    }
}
