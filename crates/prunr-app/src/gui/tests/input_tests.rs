use super::super::app::PrunrApp;
use super::super::state::AppState;

// NOTE: These tests validate the state-machine logic of the helpers,
// not the egui input system. Full keyboard testing requires the running app (Plan 03).

#[test]
fn handle_cancel_raises_global_flag_during_processing() {
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Processing;
    assert!(!app.processor.cancels.is_cancelled(0), "cancel registry should start clean");

    app.handle_cancel();

    assert!(app.processor.cancels.is_cancelled(0),
        "handle_cancel must raise the global flag (every id reports cancelled)");
}

#[test]
fn handle_cancel_selected_raises_only_selected_item_flags() {
    use super::super::item::{BatchItem, BatchStatus, ImageSource};
    use super::super::item_settings::ItemSettings;
    use std::sync::Arc;

    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Processing;
    // Seed batch: id 1 selected+processing, id 2 selected+done, id 3 unselected+processing.
    for (id, selected, status) in [(1u64, true, BatchStatus::Processing), (2, true, BatchStatus::Done), (3, false, BatchStatus::Processing)] {
        let mut item = BatchItem::new(
            id, format!("x{id}.png"),
            ImageSource::Bytes(Arc::new(Vec::new())),
            (1, 1),
            ItemSettings::default(),
            String::new(),
        );
        item.selected = selected;
        item.status = status;
        app.batch.items.push(item);
    }

    app.handle_cancel_selected();

    // Only id 1 matches selected + Processing.
    assert!(app.processor.cancels.is_cancelled(1), "selected processing item must be cancelled");
    assert!(!app.processor.cancels.is_cancelled(2), "selected-but-done item must not be cancelled");
    assert!(!app.processor.cancels.is_cancelled(3), "unselected item must not be cancelled");
}

#[test]
fn handle_open_path_nonexistent_does_not_panic() {
    let mut app = PrunrApp::new_for_test();
    // Opening a non-existent path should set error status, not panic
    app.handle_open_path(std::path::PathBuf::from("/nonexistent/path/image.png"));
    // State should remain Empty (no transition on error)
    assert_eq!(app.state, AppState::Empty);
}

#[test]
fn show_shortcuts_toggle_starts_false() {
    let app = PrunrApp::new_for_test();
    assert!(!app.show_shortcuts);
}
