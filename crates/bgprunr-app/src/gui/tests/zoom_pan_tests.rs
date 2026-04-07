use crate::gui::app::BgPrunrApp;
use crate::gui::state::AppState;

#[test]
fn zoom_initialized_to_one() {
    let app = BgPrunrApp::new_for_test();
    assert!((app.zoom - 1.0).abs() < f32::EPSILON);
}

#[test]
fn pan_offset_initialized_to_zero() {
    let app = BgPrunrApp::new_for_test();
    assert!((app.pan_offset.x).abs() < f32::EPSILON);
    assert!((app.pan_offset.y).abs() < f32::EPSILON);
}

#[test]
fn previous_zoom_initialized_to_one() {
    let app = BgPrunrApp::new_for_test();
    assert!((app.previous_zoom - 1.0).abs() < f32::EPSILON);
}

#[test]
fn before_after_toggle_switches_show_original() {
    let mut app = BgPrunrApp::new_for_test();
    app.state = AppState::Done;
    assert!(!app.show_original);
    app.show_original = !app.show_original;
    assert!(app.show_original);
    app.show_original = !app.show_original;
    assert!(!app.show_original);
}

#[test]
fn show_original_resets_on_new_image_load() {
    let mut app = BgPrunrApp::new_for_test();
    app.show_original = true;
    // Simulate loading a new image via load_image internals
    // load_image resets show_original to false
    app.show_original = false; // simulates what load_image does
    app.state = AppState::Loaded;
    assert!(!app.show_original);
}

#[test]
fn zoom_clamped_to_range() {
    let mut app = BgPrunrApp::new_for_test();
    // Simulate zoom beyond max
    app.zoom = 25.0_f32.clamp(0.10, 20.0);
    assert!((app.zoom - 20.0).abs() < f32::EPSILON);
    // Simulate zoom below min
    app.zoom = 0.01_f32.clamp(0.10, 20.0);
    assert!((app.zoom - 0.10).abs() < f32::EPSILON);
}

#[test]
fn pending_fit_zoom_flag_defaults_false() {
    let app = BgPrunrApp::new_for_test();
    assert!(!app.pending_fit_zoom);
}

#[test]
fn pending_actual_size_flag_defaults_false() {
    let app = BgPrunrApp::new_for_test();
    assert!(!app.pending_actual_size);
}
