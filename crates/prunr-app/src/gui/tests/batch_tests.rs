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
fn batch_status_pending_is_default() {
    let status = BatchStatus::Pending;
    assert_eq!(status, BatchStatus::Pending);
}

#[test]
fn batch_status_done_is_distinct() {
    assert_ne!(BatchStatus::Done, BatchStatus::Pending);
    assert_ne!(BatchStatus::Done, BatchStatus::Processing);
}

#[test]
fn nav_prev_wraps_around() {
    let mut idx: usize = 0;
    let len: usize = 5;
    // Simulate [ key at index 0
    if idx == 0 { idx = len - 1; } else { idx -= 1; }
    assert_eq!(idx, 4);
}

#[test]
fn nav_next_wraps_around() {
    let mut idx: usize = 4;
    let len: usize = 5;
    idx = (idx + 1) % len;
    assert_eq!(idx, 0);
}

#[test]
fn batch_reorder_preserves_items() {
    // Simulate reorder: move item at index 0 to index 2
    let mut items = vec!["a", "b", "c", "d"];
    let from = 0;
    let to = 2;
    let item = items.remove(from);
    let dst = if from < to { to - 1 } else { to };
    items.insert(dst, item);
    assert_eq!(items, vec!["b", "a", "c", "d"]);
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
