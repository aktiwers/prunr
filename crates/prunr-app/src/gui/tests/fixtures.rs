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

/// Line planes at 2×2 for `scales` (the others not made), as a run
/// sends them; `spare` is the pass still to make the rest.
pub(crate) fn edge_lines(scales: &[prunr_core::EdgeScale], spare: Option<u64>) -> Option<crate::gui::worker::CompressedEdgeTensors> {
    let tensors = prunr_core::EdgeScale::ALL.map(|s| {
        let (h, w) = prunr_core::edge_scale_dims(s, 2, 2);
        if scales.contains(&s) { vec![0.5; (h * w) as usize] } else { Vec::new() }
    });
    crate::gui::worker::CompressedEdgeTensors::from_raw(crate::gui::worker::RawEdgeTensors { tensors, height: 2, width: 2, spare })
}

/// Land a cut by Silueta on `item` with every line scale, as a run under
/// its settings leaves it (lines traced on the cut-out under Subject
/// outline).
pub(crate) fn cut_with_lines(item: &mut BatchItem) {
    use prunr_core::ModelKind::Silueta;
    item.start_cut(Some(crate::gui::timeline::StepKind::Process), crate::gui::item::Cutter::Model(Silueta));
    let seg = crate::gui::worker::CompressedTensor::from_raw(crate::gui::worker::RawSegTensor { data: vec![0.5; 4], height: 2, width: 2, model: Silueta });
    let result = prunr_core::ProcessResult { rgba_image: image::RgbaImage::new(2, 2), active_provider: "CPU".into() };
    let recipe = item.settings.current_recipe(Silueta);
    item.apply_tier_result(Ok(result), seg, edge_lines(&prunr_core::EdgeScale::ALL, None), recipe);
    item.finish_job();
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

/// A hard, full-strength Add stamp: a stroke covers whole cells.
pub(crate) const HARD: prunr_core::brush::Stamp =
    prunr_core::brush::Stamp { hardness: 1.0, strength: 1.0, mode: prunr_core::selection::BrushMode::Add };

/// A Paint Brush stroke of `shape` at Size `size` through `samples`
/// (x, y, radius in image pixels), drawn with `stamp` and a hard edge.
pub(crate) fn painted(
    shape: prunr_core::brush::BrushShape,
    size: f32,
    samples: &[(f32, f32, f32)],
    stamp: prunr_core::brush::Stamp,
) -> crate::gui::brush_state::StrokeDrawn {
    painted_with_edge(shape, size, samples, stamp, crate::gui::brush_state::StrokeEdge::HARD)
}

/// `painted`, its edge drawn as `edge`.
pub(crate) fn painted_with_edge(
    shape: prunr_core::brush::BrushShape,
    size: f32,
    samples: &[(f32, f32, f32)],
    stamp: prunr_core::brush::Stamp,
    edge: crate::gui::brush_state::StrokeEdge,
) -> crate::gui::brush_state::StrokeDrawn {
    let mut brush = crate::gui::brush_state::BrushState::default();
    brush.begin_stroke(shape, size);
    for &(x, y, radius) in samples {
        brush.extend_stroke_with_radius(x, y, radius);
    }
    crate::gui::brush_state::StrokeDrawn::painted(brush.commit_stroke().expect("a stroke"), stamp, edge)
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
