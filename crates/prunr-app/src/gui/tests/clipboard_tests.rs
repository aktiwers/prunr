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

/// A copy whose image cannot be read (the photo's bytes do not decode)
/// still answers, with an error: it said nothing and never ended.
#[test]
fn a_copy_of_an_unreadable_image_says_it_failed() {
    let mut app = PrunrApp::new_for_test();
    push_test_item(&mut app, 1);
    app.batch.selected_index = 0;
    app.handle_copy();
    let ctx = egui::Context::default();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !app.status.is_temporary() && std::time::Instant::now() < deadline {
        app.drain_background_channels(&ctx);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(app.status.text.starts_with("Could not copy"), "{:?}", app.status.text);
}
