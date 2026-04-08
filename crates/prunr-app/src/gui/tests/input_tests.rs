use std::sync::atomic::Ordering;

use super::super::app::PrunrApp;
use super::super::state::AppState;

// NOTE: These tests validate the state-machine logic of the helpers,
// not the egui input system. Full keyboard testing requires the running app (Plan 03).

#[test]
fn handle_cancel_sets_flag_during_processing() {
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Processing;

    // Before cancel, flag should be false
    let flag_before = app.cancel_flag.load(Ordering::Relaxed);
    assert!(!flag_before, "cancel_flag should start false");

    app.handle_cancel();

    let flag_after = app.cancel_flag.load(Ordering::Relaxed);
    assert!(flag_after, "handle_cancel should set cancel_flag to true");
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
