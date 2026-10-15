use super::super::app::PrunrApp;
use super::super::state::AppState;

#[test]
fn handle_cancel_raises_global_flag_during_processing() {
    let mut app = PrunrApp::new_for_test();
    assert!(!app.processor.cancels.is_cancelled(0), "cancel registry should start clean");

    app.handle_cancel();

    assert!(app.processor.cancels.is_cancelled(0),
        "handle_cancel must raise the global flag (every id reports cancelled)");
}

#[test]
fn handle_cancel_selected_raises_only_selected_item_flags() {
    use super::super::item::BatchStatus;
    use super::fixtures::push_test_item;

    let mut app = PrunrApp::new_for_test();
    // Seed batch: id 1 selected+processing, id 2 selected+done, id 3 unselected+processing.
    for (id, selected, status) in [(1u64, true, BatchStatus::Processing), (2, true, BatchStatus::Done), (3, false, BatchStatus::Processing)] {
        let item = push_test_item(&mut app, id);
        item.selected = selected;
        item.status = status;
    }

    app.handle_cancel_selected();

    // Only id 1 matches selected + Processing.
    assert!(app.processor.cancels.is_cancelled(1), "selected processing item must be cancelled");
    assert!(!app.processor.cancels.is_cancelled(2), "selected-but-done item must not be cancelled");
    assert!(!app.processor.cancels.is_cancelled(3), "unselected item must not be cancelled");

    // Cancelled item flipped immediately → thumbnail / canvas spinners stop
    // before the subprocess acknowledges the cancel.
    let item1 = app.batch.find_by_id(1).expect("id 1 still present");
    assert_eq!(item1.status, BatchStatus::Pending);
    let item3 = app.batch.find_by_id(3).expect("id 3 still present");
    assert_eq!(item3.status, BatchStatus::Processing, "unselected item keeps running");
}

#[test]
fn handle_open_path_nonexistent_does_not_panic() {
    let mut app = PrunrApp::new_for_test();
    // Opening a non-existent path should set error status, not panic
    app.handle_open_path(std::path::PathBuf::from("/nonexistent/path/image.png"));
    // State should remain Empty (no transition on error)
    assert_eq!(app.batch.app_state(), AppState::Empty);
}

/// F1 shows the shortcuts where they can also be changed: Settings on
/// the Hotkeys tab. A second press closes it.
#[test]
fn f1_opens_settings_on_hotkeys_and_closes_it_again() {
    use crate::gui::views::settings::SettingsTab;
    use crate::gui::views::shortcuts::Action;
    let ctx = egui::Context::default();
    let mut app = PrunrApp::new_for_test();
    app.perform(Action::Shortcuts, &ctx);
    assert!(app.show_settings);
    assert_eq!(app.settings_tab, SettingsTab::Hotkeys);
    app.perform(Action::Shortcuts, &ctx);
    assert!(!app.show_settings);
    app.settings_tab = SettingsTab::Performance;
    app.show_settings = true;
    app.perform(Action::Shortcuts, &ctx);
    assert!(app.show_settings && app.settings_tab == SettingsTab::Hotkeys, "from another tab it switches, not closes");
}
