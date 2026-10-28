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
    item.amend();
    item.record_origin(crate::gui::item::Engine::Segment, None, Some(crate::gui::item::Cutter::Model(model)));
}

/// Put `item` in `status` through the transitions the app makes: Done
/// lands the run under way on the result it holds, or makes a cut-out
/// (`cut_out`); Processing is a cut the worker has taken; Error a failed
/// run; Pending ends any run, leaving what the image shows.
pub(crate) fn put_in(item: &mut BatchItem, status: crate::gui::item::BatchStatus) {
    use crate::gui::item::{BatchStatus, Cutter, Engine};
    match status {
        BatchStatus::Pending => item.abandon_job(),
        BatchStatus::Processing => {
            if !item.job.as_ref().is_some_and(|j| j.engine == Engine::Segment) {
                item.start_cut(Some(crate::gui::timeline::StepKind::Process), Cutter::Model(prunr_core::ModelKind::Silueta));
            }
            item.mark_running();
        }
        BatchStatus::Done if item.has_result() => {
            if item.job.is_some() {
                item.finish_job();
            }
        }
        BatchStatus::Done => cut_out(item, prunr_core::ModelKind::Silueta),
        BatchStatus::Error(why) => item.fail(why),
    }
}

/// Land every selection edit queued so far, as frames would.
pub(crate) fn settle_selection_edits(app: &mut crate::gui::app::PrunrApp) {
    let ctx = egui::Context::default();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while app.batch.selection_queue.any_pending() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
        app.land_selection_edits(&ctx);
    }
    assert!(!app.batch.selection_queue.any_pending(), "the selection edits land");
}

/// A process that idles for half a minute: a stand-in for a test's or
/// another instance's process.
pub(crate) fn idle_child() -> std::process::Child {
    #[cfg(unix)]
    let child = std::process::Command::new("sleep").arg("30").spawn();
    #[cfg(windows)]
    let child = std::process::Command::new("ping").args(["-n", "30", "127.0.0.1"]).stdout(std::process::Stdio::null()).spawn();
    child.expect("a process to stand in")
}

/// A process that ends at once.
pub(crate) fn quick_child() -> std::process::Child {
    #[cfg(unix)]
    let child = std::process::Command::new("true").spawn();
    #[cfg(windows)]
    let child = std::process::Command::new("cmd").args(["/C", "exit", "0"]).spawn();
    child.expect("a process that ends at once")
}
