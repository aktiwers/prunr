/// App-level state derived from the selected item's `BatchStatus`.
/// Computed on demand via `BatchManager::app_state()` — not stored.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AppState {
    #[default]
    Empty,
    Loaded,
    Processing,
    Done,
}

