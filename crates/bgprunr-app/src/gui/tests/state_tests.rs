use super::super::state::AppState;

#[test]
fn app_state_default_is_empty() {
    assert_eq!(AppState::default(), AppState::Empty);
}

#[test]
fn app_state_has_four_variants() {
    // Verify all four variants exist and are distinct
    let states = [AppState::Empty, AppState::Loaded, AppState::Processing, AppState::Done];
    for (i, a) in states.iter().enumerate() {
        for (j, b) in states.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn animating_state_exists_and_is_distinct() {
    let state = AppState::Animating;
    assert_ne!(state, AppState::Processing);
    assert_ne!(state, AppState::Done);
    assert_eq!(state, AppState::Animating);
}
