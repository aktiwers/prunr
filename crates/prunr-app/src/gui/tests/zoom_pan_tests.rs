use crate::gui::app::PrunrApp;

#[test]
fn zoom_initialized_to_one() {
    let app = PrunrApp::new_for_test();
    assert!((app.zoom_state.zoom - 1.0).abs() < f32::EPSILON);
}

#[test]
fn pan_offset_initialized_to_zero() {
    let app = PrunrApp::new_for_test();
    assert!((app.zoom_state.pan_offset.x).abs() < f32::EPSILON);
    assert!((app.zoom_state.pan_offset.y).abs() < f32::EPSILON);
}

#[test]
fn previous_zoom_initialized_to_one() {
    let app = PrunrApp::new_for_test();
    assert!((app.zoom_state.previous_zoom - 1.0).abs() < f32::EPSILON);
}

#[test]
fn pending_fit_zoom_flag_defaults_false() {
    let app = PrunrApp::new_for_test();
    assert!(!app.zoom_state.pending_fit_zoom);
}

#[test]
fn pending_actual_size_flag_defaults_false() {
    let app = PrunrApp::new_for_test();
    assert!(!app.zoom_state.pending_actual_size);
}
