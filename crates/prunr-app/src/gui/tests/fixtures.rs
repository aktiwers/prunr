//! Shared test fixtures for `gui::tests::*` modules.

use std::sync::Arc;

use crate::gui::app::PrunrApp;
use crate::gui::item::{BatchItem, ImageSource};
use crate::gui::item_settings::ItemSettings;

/// Construct a placeholder `BatchItem` and append it to the batch. Returns
/// a mutable ref so callers can tweak `selected` / `status` / `result_rgba`
/// before running the action under test.
pub(crate) fn push_test_item(app: &mut PrunrApp, id: u64) -> &mut BatchItem {
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

/// Make `item` a cut-out by `model` as a run leaves it: its segmentation
/// kept, a result on the shown step, and the step recorded as that cut.
/// Adds no undo step.
pub(crate) fn cut_out(item: &mut BatchItem, model: prunr_core::ModelKind) {
    let (w, h) = if item.dimensions.0 > 1 { item.dimensions } else { (8, 8) };
    let raw = crate::gui::worker::RawSegTensor { data: vec![0.5; (w * h) as usize], height: h, width: w, model };
    item.set_cached_tensor(crate::gui::worker::CompressedTensor::from_raw(raw));
    if item.result_rgba.is_none() {
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(w, h)));
    }
    item.set_status_for_test(crate::gui::item::BatchStatus::Done);
    item.amend();
    item.record_origin(crate::gui::item::Engine::Segment, None, Some(crate::gui::item::Cutter::Model(model)));
}
