/// App-level state derived from the selected item's `BatchStatus`.
/// Computed on demand via `BatchManager::app_state()` — not stored.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AppState {
    /// No image loaded. Show drop zone.
    #[default]
    Empty,
    /// Image loaded, ready for processing. Stores original image bytes.
    Loaded,
    /// Inference running on worker thread.
    Processing,
    /// Inference complete. Result available.
    Done,
}
