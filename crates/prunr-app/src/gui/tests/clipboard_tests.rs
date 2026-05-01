use super::super::app::PrunrApp;
use super::super::state::AppState;
use crate::gui::item::{BatchItem, ImageSource};
use crate::gui::item_settings::ItemSettings;

fn push_test_item(app: &mut PrunrApp, with_result: bool) {
    let mut item = BatchItem::new(
        1,
        "test.png".to_string(),
        ImageSource::Bytes(std::sync::Arc::new(Vec::new())),
        (2, 2),
        ItemSettings::default(),
        String::new(),
    );
    if with_result {
        item.result_rgba = Some(std::sync::Arc::new(image::RgbaImage::new(2, 2)));
    }
    app.batch.items.push(item);
    app.batch.selected_index = 0;
}

#[test]
fn handle_copy_without_clipboard_does_not_panic() {
    // PrunrApp::new_for_test() sets clipboard = None
    let mut app = PrunrApp::new_for_test();
    app.state = AppState::Done;
    push_test_item(&mut app, true);

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
    push_test_item(&mut app, false);

    // Should not panic
    app.handle_copy();
    // Status text stays as-is or gets set to clipboard error; either is fine
}
