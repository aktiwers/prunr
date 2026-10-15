use crate::gui::app::PrunrApp;
use crate::gui::item::BatchStatus;

#[test]
fn batch_items_starts_empty() {
    let app = PrunrApp::new_for_test();
    assert!(app.batch.items.is_empty());
}

#[test]
fn selected_batch_index_starts_at_zero() {
    let app = PrunrApp::new_for_test();
    assert_eq!(app.batch.selected_index, 0);
}

#[test]
fn sidebar_hidden_defaults_false() {
    let app = PrunrApp::new_for_test();
    assert!(!app.sidebar_hidden);
}

#[test]
fn batch_status_done_is_distinct() {
    assert_ne!(BatchStatus::Done, BatchStatus::Pending);
    assert_ne!(BatchStatus::Done, BatchStatus::Processing);
}

#[test]
fn auto_process_setting_defaults_false() {
    let app = PrunrApp::new_for_test();
    assert!(!app.settings.auto_process_on_import);
}

#[test]
fn next_batch_id_starts_at_zero() {
    let app = PrunrApp::new_for_test();
    assert_eq!(app.batch.next_id, 0);
}
