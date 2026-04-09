use super::super::state::AppState;

#[test]
fn app_state_default_is_empty() {
    assert_eq!(AppState::default(), AppState::Empty);
}

#[test]
fn app_state_has_four_variants() {
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
