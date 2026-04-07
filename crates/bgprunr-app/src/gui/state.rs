/// Application state machine: Empty -> Loaded -> Processing -> Animating -> Done
/// Can also transition Done -> Loaded (load new image) or Processing -> Loaded (cancel)
#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    /// No image loaded. Show drop zone.
    Empty,
    /// Image loaded, ready for processing. Stores original image bytes.
    Loaded,
    /// Inference running on worker thread.
    Processing,
    /// Reveal animation playing after inference completes.
    Animating,
    /// Inference complete. Result available.
    Done,
}

impl Default for AppState {
    fn default() -> Self {
        Self::Empty
    }
}
