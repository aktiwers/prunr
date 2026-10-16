use super::super::app::PrunrApp;
use super::fixtures::push_test_item;

#[test]
fn handle_copy_never_touches_the_system_clipboard_in_tests() {
    // A test bridge has no platform clipboard: a copy reports failure
    // instead of writing to the clipboard of the display the tests run on.
    let mut app = PrunrApp::new_for_test();
    push_test_item(&mut app, 1).result_rgba =
        Some(std::sync::Arc::new(image::RgbaImage::new(2, 2)));
    app.batch.selected_index = 0;
    app.handle_copy();
    assert_eq!(app.system.take_copy_outcomes().len(), 1, "one copy, reported back");
}

#[test]
fn handle_copy_without_result_does_not_panic() {
    // No result on the selected item — handle_copy should be a no-op.
    let mut app = PrunrApp::new_for_test();
    push_test_item(&mut app, 1);
    app.batch.selected_index = 0;

    // Should not panic
    app.handle_copy();
    // Status text stays as-is or gets set to clipboard error; either is fine
}
