use super::super::app::PrunrApp;
use super::super::state::AppState;
use super::fixtures::push_test_item;

#[test]
fn handle_copy_without_clipboard_does_not_panic() {
    // PrunrApp::new_for_test() sets clipboard = None
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Done;
    push_test_item(&mut app, 1).result_rgba =
        Some(std::sync::Arc::new(image::RgbaImage::new(2, 2)));
    app.batch.selected_index = 0;

    // Should not panic, should set error status_text
    app.handle_copy();
    assert!(
        app.status.text.contains("clipboard") || app.status.text.contains("saving"),
        "Expected clipboard error message, got: {}",
        app.status.text
    );
}

#[test]
fn handle_copy_without_result_does_not_panic() {
    // No result on the selected item — handle_copy should be a no-op.
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Done;
    push_test_item(&mut app, 1);
    app.batch.selected_index = 0;

    // Should not panic
    app.handle_copy();
    // Status text stays as-is or gets set to clipboard error; either is fine
}
