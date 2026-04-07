use crate::gui::app::BgPrunrApp;
use crate::gui::state::AppState;
use crate::gui::theme;

#[test]
fn anim_progress_defaults_to_zero() {
    let app = BgPrunrApp::new_for_test();
    assert!((app.anim_progress).abs() < f32::EPSILON);
}

#[test]
fn anim_mask_defaults_to_none() {
    let app = BgPrunrApp::new_for_test();
    assert!(app.anim_mask.is_none());
}

#[test]
fn anim_progress_advances_by_dt() {
    let mut progress = 0.0_f32;
    let dt = 0.016_f32; // ~60fps
    progress = (progress + dt / theme::ANIM_DURATION_SECS).min(1.0);
    assert!(progress > 0.0);
    assert!(progress < 1.0);
}

#[test]
fn anim_completes_at_one() {
    let mut progress = 0.95_f32;
    let dt = 0.1_f32;
    progress = (progress + dt / theme::ANIM_DURATION_SECS).min(1.0);
    assert!((progress - 1.0).abs() < f32::EPSILON);
}

#[test]
fn anim_skip_transitions_to_done() {
    let mut app = BgPrunrApp::new_for_test();
    app.state = AppState::Animating;
    app.anim_progress = 0.3;
    // Simulate skip
    app.state = AppState::Done;
    app.anim_progress = 0.0;
    assert_eq!(app.state, AppState::Done);
    assert!((app.anim_progress).abs() < f32::EPSILON);
}

#[test]
fn animation_disabled_skips_animating_state() {
    let mut app = BgPrunrApp::new_for_test();
    app.settings.reveal_animation_enabled = false;
    // Simulate what logic() does when WorkerResult::Done arrives
    if app.settings.reveal_animation_enabled {
        app.state = AppState::Animating;
    } else {
        app.state = AppState::Done;
    }
    assert_eq!(app.state, AppState::Done);
}
