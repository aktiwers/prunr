//! Shared test fixtures for `gui::tests::*` modules.

use std::sync::Arc;

use crate::gui::app::PrunrApp;
use crate::gui::item::{BatchItem, ImageSource};
use crate::gui::item_settings::ItemSettings;

/// Construct a placeholder `BatchItem` and append it to the batch. Returns
/// a mutable ref so callers can tweak `selected` / `status` / `result_rgba`
/// before running the action under test.
pub(super) fn push_test_item(app: &mut PrunrApp, id: u64) -> &mut BatchItem {
    let item = BatchItem::new(
        id,
        format!("test{id}.png"),
        ImageSource::Bytes(Arc::new(Vec::new())),
        (1, 1),
        ItemSettings::default(),
        String::new(),
    );
    app.batch.items.push(item);
    app.batch.items.last_mut().unwrap()
}
