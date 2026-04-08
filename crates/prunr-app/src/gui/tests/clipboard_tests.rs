use super::super::app::PrunrApp;
use super::super::state::AppState;

#[test]
fn handle_copy_without_clipboard_does_not_panic() {
    // PrunrApp::new_for_test() sets clipboard = None
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Done;

    // set a dummy result_rgba so handle_copy reaches the clipboard check
    let rgba = image::RgbaImage::new(2, 2);
    app.result_rgba = Some(rgba);

    // Should not panic, should set error status_text
    app.handle_copy();
    assert!(
        app.status_text.contains("clipboard") || app.status_text.contains("saving"),
        "Expected clipboard error message, got: {}",
        app.status_text
    );
}

#[test]
fn handle_copy_without_result_does_not_panic() {
    // result_rgba = None -- handle_copy should be a no-op or set error status
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Done;
    app.result_rgba = None;

    // Should not panic
    app.handle_copy();
    // Status text stays as-is or gets set to clipboard error; either is fine
}
