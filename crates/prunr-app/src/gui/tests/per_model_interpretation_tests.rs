//! Brushes and selections against the app: which active model re-cuts on a
//! committed selection (`LivePreview::is_pending_for`), how strokes sit on
//! the undo timeline, the live preview's tensor reuse, strokes drawn
//! again as the sliders move, the Magic Brush embedding cache, the cut-out switch (Restore / Erase /
//! Select) and its correction cache, and what a model switch leaves
//! correctable.
//!
//! The Magic Brush's model (Sam21HieraSmall) is not a `SettingsModel`: it
//! is never the active model, so it has no row in the per-model table.


use crate::gui::app::PrunrApp;
use crate::gui::settings::SettingsModel;
use crate::gui::timeline::StepKind;

use super::fixtures::push_test_item;
use crate::gui::item::BatchItem;
use crate::gui::live_preview::{DispatchInputs, PreviewKind};
use crate::gui::worker::{CompressedTensor, RawSegTensor};

/// A segmentation result exists for the item: the correction has a tensor
/// to apply to.
fn give_tensor(item: &mut BatchItem) {
    give_tensor_of(item, prunr_core::ModelKind::Silueta);
}

/// An 8×8 segmentation tensor made by `model`.
fn give_tensor_of(item: &mut BatchItem, model: prunr_core::ModelKind) {
    let cache = RawSegTensor { data: vec![0.5; 64], height: 8, width: 8, model };
    item.set_cached_tensor(CompressedTensor::from_raw(cache));
}

/// An 8×8 opaque grey image of value `v`.
fn px(v: u8) -> image::RgbaImage {
    image::RgbaImage::from_pixel(8, 8, image::Rgba([v, v, v, 255]))
}

/// The live preview's mask inputs for `id`, at the app's current cut-out
/// action.
fn preview_inputs(app: &mut PrunrApp, id: u64, filter_only: bool) -> Option<DispatchInputs> {
    let action = app.settings.brush.cutout_action;
    let role = if filter_only { crate::gui::knob_catalog::ModelRole::Filters } else { crate::gui::knob_catalog::ModelRole::Cutout };
    PrunrApp::build_preview_inputs(app.batch.find_by_id_mut(id).expect("the item"), PreviewKind::Mask, role, action).ok()
}

/// A second stroke on the first item: its bottom-right cell joins the
/// selection.
fn add_corner_stroke(app: &mut PrunrApp) {
    let corner = prunr_core::selection::MaskArtifact::from_cells(8, 8, (0..64).map(|i| if i == 63 { prunr_core::selection::FULL } else { 0 }).collect());
    let more = crate::gui::processor::merge_stroke(app.batch.items[0].selection_mask.clone(), corner);
    app.batch.items[0].commit_selection_mask(std::sync::Arc::new(more), StepKind::Stroke);
}

/// A processed item: the cut-out is on screen and its tensor is cached.
fn give_cutout(item: &mut BatchItem) {
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
}

/// The live preview unpacks the segmentation tensor once per drag, on the
/// render's thread: the first tick sends it packed, the render's unpacked
/// copy is kept, and the next ticks share that one plane until the cached
/// tensor changes.
#[test]
fn preview_inputs_reuse_the_decompressed_tensor_until_it_changes() {
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 3);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.on_screen = true;
    give_cutout(item);
    let first = preview_inputs(&mut app, 3, false).expect("inputs");
    assert!(first.seg_tensor.is_none(), "nothing unpacked on the UI thread");
    let packed = first.packed.seg.expect("sent packed");
    let unpacked = Arc::new(crate::gui::live_preview::decompress_seg(&packed).expect("unpacks"));
    let render = |seg: &Arc<_>| crate::gui::live_preview::Unpacked { seg: Some((packed.clone(), Arc::clone(seg))), edge: None };
    app.batch.items[0].keep_unpacked(render(&unpacked));
    let second = preview_inputs(&mut app, 3, false).expect("inputs");
    assert!(second.packed.seg.is_none());
    assert!(second.seg_tensor.is_some_and(|s| Arc::ptr_eq(&s, &unpacked)), "the next tick reuses the render's copy");
    let item = app.batch.find_by_id_mut(3).unwrap();
    item.set_cached_tensor(None);
    assert!(item.volatile_seg_tensor.is_none());
    item.keep_unpacked(render(&unpacked));
    assert!(item.volatile_seg_tensor.is_none(), "a copy of a tensor no longer cached is not kept");
}

fn make_mask(w: u32, h: u32) -> prunr_core::selection::MaskArtifact {
    let mut data = vec![0i8; (w * h) as usize];
    // Paint a 4x4 square at (2, 2) — non-trivial but small.
    for y in 2..6_u32.min(h) {
        for x in 2..6_u32.min(w) {
            data[(y * w + x) as usize] = prunr_core::selection::FULL;
        }
    }
    prunr_core::selection::MaskArtifact::from_cells(w, h, data)
}

fn app_with_model(model: SettingsModel) -> PrunrApp {
    let mut app = PrunrApp::new_for_test();
    app.settings.model = model;
    app
}

/// A stroke is one step that shares the shown result with the step
/// before it: recording it copies no image.
#[test]
fn a_stroke_is_one_step_sharing_the_result_before_it() {
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 7);
    item.dimensions = (8, 8);
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.commit(StepKind::Process);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);

    app.commit_selection_and_dispatch(7, make_mask(8, 8), StepKind::Stroke);

    let item = &app.batch.items[0];
    assert_eq!(item.timeline.undo_steps(), 2);
    let stored: Vec<_> = item.timeline.stored_images().collect();
    assert!(Arc::ptr_eq(stored[0], stored[1]), "both steps point at one stored image");
    assert!(app.processor.live_preview.is_pending_for(7), "the re-cut is queued");
}

/// Queue `drawn` on image `item_id` as its brush does, and land it.
fn land(app: &mut PrunrApp, item_id: u64, drawn: crate::gui::brush_state::StrokeDrawn) {
    let dims = app.batch.find_by_id(item_id).expect("the image").dimensions;
    app.queue_selection_op(item_id, crate::gui::selection_queue::SelectionOp::Stroke { drawn, dims, source: None });
    super::fixtures::settle_selection_edits(app);
}

/// Image 9 (`dims`, on Silueta) with `drawn` landed: the latest stroke.
fn app_with_landed_stroke(drawn: crate::gui::brush_state::StrokeDrawn, dims: (u32, u32)) -> PrunrApp {
    let mut app = app_with_model(SettingsModel::Silueta);
    push_test_item(&mut app, 9).dimensions = dims;
    land(&mut app, 9, drawn);
    app
}

/// An image (id 9) whose latest stroke is a Magic Brush pick whose
/// logits ramp left to right, so a higher confidence keeps fewer
/// columns. Returns the pick's selection hash.
fn app_with_a_magic_pick() -> (PrunrApp, u64) {
    use crate::gui::brush_state::StrokeDrawn;
    let m = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
    let mut masks = vec![-9.0f32; 3 * m * m];
    masks.iter_mut().take(m * m).enumerate().for_each(|(i, v)| *v = -4.0 + 8.0 * (i % m) as f32 / (m - 1) as f32);
    let decoded = prunr_core::sam::SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] };
    let output = std::sync::Arc::new(decoded.best_only());
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.brush.magic_confidence_threshold = 0.5;
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    give_tensor(item);
    let reading = app.settings.brush.magic_reading();
    land(&mut app, 9, StrokeDrawn::Magic { output, mode: prunr_core::selection::BrushMode::Add, reading, feather: 0.0 });
    assert_eq!(app.batch.items[0].timeline.undo_steps(), 1);
    let hash = app.batch.items[0].selection_hash.expect("the pick");
    (app, hash)
}

/// Move the brush sliders by `change`, a frame seeing them before and after.
fn slide(app: &mut PrunrApp, change: impl FnOnce(&mut crate::gui::brush_state::BrushSettings)) {
    app.adjust_strokes();
    change(&mut app.settings.brush);
    app.adjust_strokes();
}

/// A redraw was asked for: let it land.
fn redraw_lands(app: &mut PrunrApp) {
    assert!(app.batch.selection_queue.any_pending(), "a slider moved: the strokes are drawn again");
    super::fixtures::settle_selection_edits(app);
}

/// The shown step's stroke, as it records it.
fn latest(app: &PrunrApp) -> Option<crate::gui::brush_state::StrokeDrawn> {
    use crate::gui::selection_queue::{Made, SelectionMade};
    match &app.batch.items[0].timeline.current().made {
        Some(Made::One(SelectionMade::Stroke(drawn))) => Some(drawn.clone()),
        _ => None,
    }
}

fn selected_cells(app: &PrunrApp) -> usize {
    app.batch.items[0].selection_mask.as_ref().map_or(0, |m| m.cells().iter().filter(|&&c| c != 0).count())
}

/// Moving Confidence draws the latest Magic Brush pick again from its
/// logits: the selection changes, the pick keeps its single undo step
/// and records the new value. Remove specks does the same.
#[test]
fn confidence_redraws_the_latest_pick_in_place() {
    use crate::gui::brush_state::StrokeDrawn;
    let (mut app, first_hash) = app_with_a_magic_pick();
    slide(&mut app, |b| b.magic_confidence_threshold = 0.9);
    redraw_lands(&mut app);
    let item = &app.batch.items[0];
    assert_ne!(item.selection_hash, Some(first_hash), "the selection follows the slider");
    assert_eq!(item.timeline.undo_steps(), 1, "still one undo step for the pick");
    assert!(matches!(latest(&app), Some(StrokeDrawn::Magic { reading, .. }) if (reading.confidence - 0.9).abs() < 1e-6));
    let sel = item.selection_mask.as_ref().unwrap();
    assert!(sel.cells()[7] != 0 && sel.cells()[0] == 0, "only the right-hand columns");

    app.adjust_strokes();
    assert!(!app.batch.selection_queue.any_pending(), "nothing moved, nothing redrawn");
    slide(&mut app, |b| b.magic_remove_specks = !b.magic_remove_specks);
    assert!(app.batch.selection_queue.any_pending(), "flipping Remove specks redraws the pick");
}

/// A slider moved while a run holds the image redraws nothing then, and
/// draws the move once the image is free.
#[test]
fn a_slider_moved_on_a_busy_image_redraws_once_it_is_free() {
    use crate::gui::item::Engine;
    let (mut app, first_hash) = app_with_a_magic_pick();
    app.adjust_strokes();
    app.batch.items[0].start_job(Some(StepKind::Process), Engine::Segment);
    app.settings.brush.magic_confidence_threshold = 0.9;
    app.adjust_strokes();
    assert!(!app.batch.selection_queue.any_pending(), "none starts while the run holds it");
    app.batch.items[0].abandon_job();
    app.adjust_strokes();
    redraw_lands(&mut app);
    assert_ne!(app.batch.items[0].selection_hash, Some(first_hash));
}

/// A Paint Brush stroke and a Magic Brush depth pick are each drawn
/// again when a slider of their brush moves (Size, Hardness, Opacity;
/// Range): in place, as the one undo step the stroke made, and only
/// for the brush that drew it.
#[test]
fn each_brush_redraws_its_latest_stroke_in_place() {
    use crate::gui::brush_state::StrokeDrawn;
    let settings = crate::gui::brush_state::BrushSettings { radius: 2.0, hardness: 1.0, strength: 1.0, ..Default::default() };
    let drawn = super::fixtures::painted(prunr_core::brush::BrushShape::Circle, settings.radius, &[(2.0, 8.0, 1.0), (14.0, 8.0, 1.0)], settings.stamp());
    let mut app = app_with_landed_stroke(drawn, (16, 16));
    app.settings.brush = settings;
    let (steps, before) = (app.batch.items[0].timeline.undo_steps(), selected_cells(&app));
    assert!(matches!(latest(&app), Some(StrokeDrawn::Paint { .. })));
    app.adjust_strokes();
    assert!(!app.batch.selection_queue.any_pending(), "the values it was drawn with: nothing to redraw");
    slide(&mut app, |b| b.magic_confidence_threshold = 0.9);
    assert!(!app.batch.selection_queue.any_pending(), "another brush's slider leaves it");
    slide(&mut app, |b| b.radius = 4.0);
    redraw_lands(&mut app);
    assert!(selected_cells(&app) > before, "Size doubled the stroke: {} cells, {before} before", selected_cells(&app));
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps, "still the stroke's one step");
    slide(&mut app, |b| b.strength = 0.5);
    redraw_lands(&mut app);
    let opacity = app.batch.items[0].selection_mask.as_ref().unwrap().cells()[8 * 16 + 8].unsigned_abs();
    assert!(opacity > 0 && opacity < prunr_core::selection::FULL.unsigned_abs(), "Opacity halves it: {opacity}");

    let depth = std::sync::Arc::new(prunr_core::depth::DepthMap::from_raw(16, 16, (0..256).map(|i| (i % 16) as f32 / 15.0).collect()));
    let drawn = StrokeDrawn::Depth { around: depth.at(0.5, 0.5), depth, mode: prunr_core::selection::BrushMode::Add, range: 0.05, feather: 0.0 };
    let mut app = app_with_landed_stroke(drawn, (16, 16));
    app.settings.brush.magic_depth_range = 0.05;
    let before = selected_cells(&app);
    assert!(matches!(latest(&app), Some(StrokeDrawn::Depth { .. })));
    slide(&mut app, |b| b.magic_depth_range = 0.3);
    redraw_lands(&mut app);
    assert!(selected_cells(&app) > before, "a wider Range picks more: {} cells, {before} before", selected_cells(&app));
}

/// An eraser's Paint Brush stroke is drawn again by its Size as on any
/// other model: its area is painted before Process fills it.
#[test]
fn an_eraser_stroke_redraws_with_its_size() {
    let mut app = app_with_model(SettingsModel::Inpaint);
    push_test_item(&mut app, 9).dimensions = (32, 32);
    app.settings.brush.radius = 4.0;
    let stamp = app.settings.brush.region_stamp(false);
    land(&mut app, 9, super::fixtures::painted(prunr_core::brush::BrushShape::Circle, 4.0, &[(8.0, 16.0, 2.0), (24.0, 16.0, 2.0)], stamp));
    let (steps, before) = (app.batch.items[0].timeline.undo_steps(), selected_cells(&app));
    slide(&mut app, |b| b.radius = 8.0);
    redraw_lands(&mut app);
    assert!(selected_cells(&app) > before, "a bigger Size paints more: {} cells, {before} before", selected_cells(&app));
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps, "still the stroke's one step");
}

/// A stroke stays the one its sliders redraw only while its step is the
/// one shown and the newest: another step after it, or an undo, ends it,
/// so a redraw never writes into a step the stroke did not make.
#[test]
fn the_latest_stroke_ends_with_the_next_step() {
    let land_one = || {
        let settings = crate::gui::brush_state::BrushSettings { radius: 2.0, ..Default::default() };
        let drawn = super::fixtures::painted(prunr_core::brush::BrushShape::Circle, settings.radius, &[(8.0, 8.0, 1.0)], settings.stamp());
        let mut app = app_with_landed_stroke(drawn, (16, 16));
        app.settings.brush = settings;
        app
    };
    let mut after_a_step = land_one();
    after_a_step.batch.items[0].commit(StepKind::Adjust);
    let mut undone = land_one();
    undone.handle_undo(&egui::Context::default());
    for (case, mut app) in [("another step", after_a_step), ("an undo", undone)] {
        assert!(app.batch.items[0].timeline.latest_stroke().is_none(), "{case}");
        slide(&mut app, |b| b.radius = 5.0);
        assert!(!app.batch.selection_queue.any_pending(), "{case}: nothing redrawn");
    }
}

/// Image 9, a 32 × 32 Silueta cut-out to paint on.
fn app_for_strokes() -> PrunrApp {
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 9);
    item.dimensions = (32, 32);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
    app
}

/// A Paint Brush stroke down column `x` at Size `size` on image 9,
/// landed as the brush lands it.
fn paint_down(app: &mut PrunrApp, x: f32, size: f32) {
    paint_down_on(app, 9, x, size);
}

fn paint_down_on(app: &mut PrunrApp, item_id: u64, x: f32, size: f32) {
    app.settings.brush.radius = size;
    let samples = [(x, 4.0, size / 4.0), (x, 28.0, size / 4.0)];
    let drawn = super::fixtures::painted(prunr_core::brush::BrushShape::Circle, size, &samples, app.settings.brush.stamp());
    land(app, item_id, drawn);
}

/// With "all strokes" on, move the sliders by `change` and let the
/// redraw land.
fn adjust_all(app: &mut PrunrApp, change: impl FnOnce(&mut crate::gui::brush_state::BrushSettings)) {
    app.brush_state.reach = crate::gui::brush_state::StrokeReach::All;
    slide(app, change);
    super::fixtures::settle_selection_edits(app);
}

/// Each Paint Brush stroke's Size, as a redraw of every stroke reads them.
fn stroke_sizes(app: &PrunrApp) -> Vec<f32> {
    use crate::gui::brush_state::StrokeDrawn;
    use crate::gui::selection_queue::SelectionMade;
    app.batch.items[0].timeline.strokes_since_picture().map_or_else(Vec::new, |h| {
        h.ops.iter().filter_map(|op| match op {
            SelectionMade::Stroke(StrokeDrawn::Paint { size, .. }) => Some(*size),
            _ => None,
        }).collect()
    })
}

/// "All strokes": a slider moves every stroke by as much as it moved,
/// so they keep their differences; the drag fills in one undo step, and
/// one undo takes it back.
#[test]
fn all_strokes_move_together_in_one_step() {
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    paint_down(&mut app, 24.0, 16.0);
    let (steps, before, cells) = (app.batch.items[0].timeline.undo_steps(), app.batch.items[0].selection_hash, selected_cells(&app));
    adjust_all(&mut app, |b| b.radius = 32.0);
    assert_eq!(stroke_sizes(&app), [16.0, 32.0], "twice the size each");
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps + 1);
    app.settings.brush.radius = 48.0;
    app.adjust_strokes();
    super::fixtures::settle_selection_edits(&mut app);
    assert_eq!(stroke_sizes(&app), [24.0, 48.0], "measured from where the drag began");
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps + 1, "the drag fills in its one step");
    assert!(selected_cells(&app) > cells);
    app.handle_undo(&egui::Context::default());
    assert_eq!(app.batch.items[0].selection_hash, before, "one undo, the strokes as they were");
}

/// A slider that draws none of the strokes redraws nothing: Confidence
/// over Paint Brush strokes makes no step.
#[test]
fn a_slider_none_of_the_strokes_have_redraws_nothing() {
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    let steps = app.batch.items[0].timeline.undo_steps();
    app.brush_state.reach = crate::gui::brush_state::StrokeReach::All;
    slide(&mut app, |b| b.magic_confidence_threshold = 0.9);
    assert!(!app.batch.selection_queue.any_pending(), "nothing to draw again");
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps);
}

/// A redraw made for a selection the image has since left is dropped,
/// and the next frame draws the sliders' values again from what it shows.
#[test]
fn a_redraw_dropped_for_a_state_left_is_drawn_again() {
    use crate::gui::brush_state::StrokeDrawn;
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    slide(&mut app, |b| b.radius = 16.0);
    assert!(app.batch.selection_queue.any_pending());
    let item = &mut app.batch.items[0];
    item.selection_mask = item.selection_mask.as_deref().cloned().map(std::sync::Arc::new);
    super::fixtures::settle_selection_edits(&mut app);
    assert!(matches!(latest(&app), Some(StrokeDrawn::Paint { size, .. }) if size == 8.0), "dropped");
    app.adjust_strokes();
    redraw_lands(&mut app);
    assert!(matches!(latest(&app), Some(StrokeDrawn::Paint { size, .. }) if size == 16.0), "drawn again");
}

/// A redraw of every stroke, with the values they have, gives back the
/// very selection and cut-out edits shown: strokes, a fold into the
/// cut-out, Invert and Clear between them.
#[test]
fn redrawing_every_stroke_as_it_is_changes_nothing() {
    use crate::gui::brush_state::CutoutAction;
    use crate::gui::selection_queue::{apply, Replay, SelState, SelectionMade, SelectionOp};
    let mut app = app_for_strokes();
    paint_down(&mut app, 6.0, 12.0);
    app.queue_selection_op(9, SelectionOp::Bake { action: CutoutAction::Erase });
    super::fixtures::settle_selection_edits(&mut app);
    paint_down(&mut app, 16.0, 8.0);
    app.queue_selection_op(9, SelectionOp::Invert);
    super::fixtures::settle_selection_edits(&mut app);
    // Into the one column Invert left out, so it adds something.
    paint_down(&mut app, 16.0, 20.0);
    assert!(app.batch.clear_selection(9));
    paint_down(&mut app, 12.0, 4.0);
    let item = &app.batch.items[0];
    let strokes = item.timeline.strokes_since_picture().expect("strokes to redraw");
    let kinds: Vec<&str> = strokes.ops.iter().map(|op| match op {
        SelectionMade::Stroke(_) => "stroke",
        SelectionMade::Invert => "invert",
        SelectionMade::Fold { .. } => "fold",
        SelectionMade::Clear => "clear",
    }).collect();
    assert_eq!(kinds, ["stroke", "fold", "stroke", "invert", "stroke", "clear", "stroke"]);
    let replay = Replay { strokes, alone: false, dims: (32, 32), source: None };
    let made = apply(&SelectionOp::Replay(std::sync::Arc::new(replay)), &SelState::default());
    assert_eq!(made.selection.as_ref().map(|s| s.content_hash()), item.selection_hash, "the same selection");
    assert_eq!(made.edits.as_ref().map(|e| e.cells().to_vec()), item.edits.as_ref().map(|e| e.cells().to_vec()), "the same cut-out edits");
}

/// Each slider moves only the strokes it draws: Range moves a depth
/// pick and leaves a Paint Brush stroke as it was.
#[test]
fn a_slider_moves_only_its_own_strokes() {
    use crate::gui::brush_state::StrokeDrawn;
    use crate::gui::selection_queue::SelectionMade;
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    let depth = std::sync::Arc::new(prunr_core::depth::DepthMap::from_raw(32, 32, (0..1024).map(|i| (i % 32) as f32 / 31.0).collect()));
    land(&mut app, 9, StrokeDrawn::Depth { around: depth.at(0.75, 0.5), depth, mode: prunr_core::selection::BrushMode::Add, range: 0.05, feather: 0.0 });
    app.settings.brush.magic_depth_range = 0.05;
    adjust_all(&mut app, |b| b.magic_depth_range = 0.15);
    let ops = app.batch.items[0].timeline.strokes_since_picture().expect("strokes").ops;
    let range = ops.iter().find_map(|op| match op { SelectionMade::Stroke(StrokeDrawn::Depth { range, .. }) => Some(*range), _ => None });
    assert!(range.is_some_and(|r| (r - 0.15).abs() < 1e-6), "{range:?}");
    assert_eq!(stroke_sizes(&app), [8.0], "the paint stroke as it was");
}

/// Strokes made before the picture changed are not redrawn: a Process
/// after them starts the strokes afresh.
#[test]
fn strokes_before_the_picture_changed_stay_as_they_are() {
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    app.batch.items[0].commit(StepKind::Process);
    paint_down(&mut app, 24.0, 16.0);
    assert_eq!(stroke_sizes(&app), [16.0]);
}

/// The latest stroke adjusted alone, then every stroke: the one adjusted
/// moves on from its new value, the others from theirs.
#[test]
fn every_stroke_includes_the_latest_one_adjusted_alone() {
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    paint_down(&mut app, 24.0, 8.0);
    slide(&mut app, |b| b.radius = 16.0);
    redraw_lands(&mut app);
    assert_eq!(stroke_sizes(&app), [8.0, 16.0]);
    adjust_all(&mut app, |b| b.radius = 32.0);
    assert_eq!(stroke_sizes(&app), [16.0, 32.0]);
}

/// "Next stroke": a slider sets the strokes to come and leaves the latest
/// one. Back on "Last stroke", only a slider moved after that redraws it,
/// and only that slider's value changes.
#[test]
fn next_leaves_the_latest_stroke_to_the_sliders_moved_after() {
    use crate::gui::brush_state::{StrokeDrawn, StrokeReach};
    let mut app = app_for_strokes();
    paint_down(&mut app, 8.0, 8.0);
    app.brush_state.reach = StrokeReach::Next;
    slide(&mut app, |b| b.radius = 16.0);
    assert!(!app.batch.selection_queue.any_pending(), "the next stroke only");
    app.brush_state.reach = StrokeReach::Last;
    app.adjust_strokes();
    assert!(!app.batch.selection_queue.any_pending(), "switching redraws nothing");
    slide(&mut app, |b| b.hardness = 0.5);
    redraw_lands(&mut app);
    assert!(
        matches!(latest(&app), Some(StrokeDrawn::Paint { size, stamp, .. }) if size == 8.0 && stamp.hardness == 0.5),
        "the Hardness moved, the Size as drawn",
    );
    slide(&mut app, |b| b.radius = 24.0);
    redraw_lands(&mut app);
    assert_eq!(stroke_sizes(&app), [24.0], "a Size moved now takes the stroke with it");
}

/// Feather is each stroke's own, like its other sliders: on "Next
/// stroke" it redraws nothing, on "Last stroke" it softens the latest
/// stroke against the photo and leaves the earlier one as drawn, on "All
/// strokes" it moves both by as much.
#[test]
fn feather_softens_only_the_strokes_its_reach_takes() {
    use crate::gui::brush_state::{StrokeDrawn, StrokeReach};
    use crate::gui::selection_queue::SelectionMade;
    let mut app = app_for_strokes();
    app.batch.items[0].source_rgba = Some(std::sync::Arc::new(image::RgbaImage::from_pixel(32, 32, image::Rgba([128, 128, 128, 255]))));
    paint_down(&mut app, 8.0, 8.0);
    paint_down(&mut app, 24.0, 8.0);
    let left = |app: &PrunrApp| -> Vec<i8> {
        let cells = app.batch.items[0].selection_mask.as_ref().expect("a selection").cells();
        cells.chunks(32).flat_map(|row| row[..10].to_vec()).collect()
    };
    let feathers = |app: &PrunrApp| -> Vec<f32> {
        app.batch.items[0].timeline.strokes_since_picture().map_or_else(Vec::new, |h| {
            h.ops.iter().filter_map(|op| match op {
                SelectionMade::Stroke(StrokeDrawn::Paint { feather, .. }) => Some(*feather),
                _ => None,
            }).collect()
        })
    };
    let (hard_left, hard) = (left(&app), app.batch.items[0].selection_hash);

    app.brush_state.reach = StrokeReach::Next;
    slide(&mut app, |b| b.edge_feather = 2.0);
    assert!(!app.batch.selection_queue.any_pending(), "Next: the strokes to come only");

    app.brush_state.reach = StrokeReach::Last;
    app.adjust_strokes();
    slide(&mut app, |b| b.edge_feather = 3.0);
    redraw_lands(&mut app);
    assert_eq!(feathers(&app), [0.0, 3.0]);
    assert_ne!(app.batch.items[0].selection_hash, hard, "the latest stroke softened");
    assert_eq!(left(&app), hard_left, "the earlier stroke as drawn");

    adjust_all(&mut app, |b| b.edge_feather = 5.0);
    assert_eq!(feathers(&app), [2.0, 5.0]);
    assert_ne!(left(&app), hard_left, "the earlier stroke softened too");
}

/// A slider moved on one image leaves another image's latest stroke as
/// it was: selecting that image redraws nothing.
#[test]
fn another_image_keeps_its_latest_stroke_as_drawn() {
    let mut app = app_for_strokes();
    let item = push_test_item(&mut app, 10);
    item.dimensions = (32, 32);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
    paint_down_on(&mut app, 10, 8.0, 8.0);
    paint_down(&mut app, 8.0, 8.0);
    app.batch.select_item(0);
    slide(&mut app, |b| b.radius = 16.0);
    redraw_lands(&mut app);
    app.batch.select_item(1);
    app.adjust_strokes();
    assert!(!app.batch.selection_queue.any_pending(), "the other image's stroke is not redrawn");
}

/// A stroke committed under each active model: segmentation re-cuts at
/// once; the eraser, upscale and colorize wait for the user's Process, and
/// with no model nothing runs.
#[test]
fn a_selection_commit_recuts_only_under_a_segmentation_model() {
    for (model, has_cutout, recuts) in [
        (SettingsModel::BiRefNetLite, true, true),
        (SettingsModel::Inpaint, false, false),
        (SettingsModel::RealEsrganUpscale, false, false),
        (SettingsModel::ColorizeDdcolor, false, false),
        (SettingsModel::None, false, false),
    ] {
        let mut app = app_with_model(model);
        let item = push_test_item(&mut app, 1);
        item.dimensions = (64, 64);
        if has_cutout {
            give_cutout(item);
        }
        app.batch.commit_selection(1, make_mask(64, 64), StepKind::Stroke);
        app.apply_selection_to_active_model(1);
        assert_eq!(app.processor.live_preview.is_pending_for(1), recuts, "{model:?}");
    }
}

// ── Regression: Paint Brush BG-removal immediate-feedback ────────────────────
//
// A Paint stroke on a BG-removal model must reach the live-preview rerun
// exactly as painted: direction (Subtract is the default), hardness
// falloff and strength. The core suite pins the lossless round trip; this
// pins the app wiring that stores and dispatches it.
#[test]
fn paint_brush_bg_removal_keeps_stroke_direction_and_softness() {
    use prunr_core::brush::{paint_circle, Stamp};
    use prunr_core::selection::BrushMode;

    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 7);
    item.dimensions = (32, 32);
    give_cutout(item);
    let item_id = 7u64;

    let mut committed = prunr_core::selection::MaskArtifact::new_empty(32, 32);
    let stamp = Stamp { hardness: 0.3, strength: 0.8, mode: BrushMode::Subtract };
    paint_circle(&mut committed, 16.0, 16.0, 8.0, stamp);

    app.commit_selection_and_dispatch(item_id, committed.clone(), StepKind::Stroke);
    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "a Paint stroke on a BG-removal model must queue the immediate rerun"
    );
    let stored = app.batch.find_by_id(item_id).unwrap().selection_mask.clone().unwrap();
    assert_eq!(stored.cells(), committed.cells(), "the stored selection is the stroke, cell for cell");
    assert!(stored.cells().iter().any(|&v| v < 0), "Subtract stroke must stay negative");
    assert!(
        stored.cells().iter().any(|&v| v < 0 && v > -prunr_core::selection::FULL),
        "hardness falloff must survive as intermediate magnitudes"
    );
}

// ── Segmentation without a tensor → the selection waits for the result ───────
//
// A brush is usable as soon as an image is loaded. With no tensor there is
// nothing to correct, so the commit must not queue a rerun (the no-tensor
// live-preview path would paint the raw source as a "result"); once a result
// lands, the same selection is applied.
#[test]
fn bg_removal_without_tensor_waits_until_a_result_lands() {
    let mut app = app_with_model(SettingsModel::BiRefNetLite);

    let item = push_test_item(&mut app, 9);
    item.dimensions = (64, 64);
    let item_id = 9u64;

    app.commit_selection_and_dispatch(item_id, make_mask(64, 64), StepKind::Stroke);
    assert!(
        !app.processor.live_preview.is_pending_for(item_id),
        "no tensor: a stroke must author the selection without dispatching"
    );
    assert!(app.batch.find_by_id(item_id).unwrap().selection_mask.is_some());

    give_cutout(app.batch.find_by_id_mut(item_id).unwrap());
    app.apply_selection_to_active_model(item_id);
    assert!(
        app.processor.live_preview.is_pending_for(item_id),
        "once a tensor exists the waiting selection must be applied"
    );
}

// ── Strokes undo and redo on their own ──────────────────────────────────────
//
// A stroke is one step on the timeline whatever the model and whether or
// not a result exists yet; undoing it shows the state before it.

#[test]
fn a_stroke_before_the_first_result_undoes_and_redoes() {
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 3);
    item.dimensions = (8, 8);
    app.commit_selection_and_dispatch(3, make_mask(8, 8), StepKind::Stroke);
    assert!(app.batch.items[0].selection_mask.is_some());

    app.handle_undo(&ctx);
    assert!(app.batch.items[0].selection_mask.is_none(), "undo reverts the stroke");
    assert_eq!(app.batch.items[0].status(), crate::gui::item::BatchStatus::Pending, "and touches no result");

    app.handle_redo(&ctx);
    assert!(app.batch.items[0].selection_mask.is_some(), "redo paints it again");
}

#[test]
fn an_eraser_stroke_undo_keeps_the_result() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Inpaint);
    let item = push_test_item(&mut app, 4);
    item.dimensions = (8, 8);
    crate::gui::tests::fixtures::put_in(item, crate::gui::item::BatchStatus::Done);
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.commit(StepKind::Erase);
    app.commit_selection_and_dispatch(4, make_mask(8, 8), StepKind::Stroke);

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none(), "the region stroke is undone");
    assert!(item.has_result() && item.status() == crate::gui::item::BatchStatus::Done, "the result stays");
}

#[test]
fn a_stroke_undo_brings_the_stored_result_back() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 5);
    item.dimensions = (8, 8);
    crate::gui::tests::fixtures::put_in(item, crate::gui::item::BatchStatus::Done);
    // History entries may be stored compressed, so compare pixels, not Arcs.
    let before = image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
    let recut = image::RgbaImage::from_pixel(8, 8, image::Rgba([9, 9, 9, 255]));
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.result_rgba = Some(Arc::new(before.clone()));
    give_tensor(item);
    app.batch.items[0].commit(StepKind::Process);
    app.commit_selection_and_dispatch(5, make_mask(8, 8), StepKind::Stroke);
    // The re-cut landed.
    app.batch.items[0].result_rgba = Some(Arc::new(recut.clone()));
    app.batch.items[0].amend();

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none());
    assert!(item.shown_result().is_some_and(|r| *r == before), "the pre-stroke image is back");
    assert_eq!(item.status(), crate::gui::item::BatchStatus::Done);

    app.handle_redo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_some());
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == recut), "redo restores the re-cut");
}

/// A Process run that lands with the selection kept applies it within the
/// run's own step. Undo walks back exactly: first the run, to the corrected
/// image it replaced (selection still on), then the stroke, to the image
/// before it.
#[test]
fn a_process_run_with_a_kept_selection_undoes_in_order() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(px(0)));
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);

    app.commit_selection_and_dispatch(9, make_mask(8, 8), StepKind::Stroke);
    app.batch.items[0].result_rgba = Some(Arc::new(px(2))); // the stroke's re-cut
    app.batch.items[0].amend();

    // Process: the run lands raw, then the kept selection re-cuts it.
    let item = &mut app.batch.items[0];
    item.start_job(Some(StepKind::Process), crate::gui::item::Engine::Segment);
    item.result_rgba = Some(Arc::new(px(3)));
    item.finish_job();
    app.recut_with_selection(9);
    assert!(app.processor.live_preview.is_pending_for(9), "the kept selection re-cuts the new result");
    app.batch.items[0].result_rgba = Some(Arc::new(px(4)));
    app.batch.items[0].amend();

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.shown_result().is_some_and(|r| *r == px(2)), "undoing the run brings back the corrected image it replaced");
    assert!(item.selection_mask.is_some(), "the selection belongs to the stroke, not the run");

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "undoing the stroke brings back the image before it");
    assert!(item.selection_mask.is_none());
}

/// Undoing back to the original keeps the model output cached; a stroke
/// there edits the selection only and must not re-cut, which would bring
/// the undone result back.
#[test]
fn a_stroke_after_undoing_the_result_does_not_bring_it_back() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 6);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_tensor(item);
    // The state after undoing a Process: the tensor stays, the result is gone.
    crate::gui::tests::fixtures::put_in(item, crate::gui::item::BatchStatus::Pending);

    app.commit_selection_and_dispatch(6, make_mask(8, 8), StepKind::Stroke);
    assert!(!app.processor.live_preview.is_pending_for(6), "a stroke on the original must not re-cut");

    app.handle_undo(&ctx);
    app.handle_redo(&ctx);
    assert!(!app.processor.live_preview.is_pending_for(6), "nor may stepping that stroke");
    assert_eq!(app.batch.items[0].status(), crate::gui::item::BatchStatus::Pending);
}

/// A re-cut computed for a state the image has left (by undo, redo or
/// another step) never lands: its result is for a state no longer shown.
#[test]
fn a_recut_for_a_state_left_behind_never_lands() {
    use crate::gui::live_preview::PreviewResult;
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 8);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(px(0)));
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
    app.commit_selection_and_dispatch(8, make_mask(8, 8), StepKind::Stroke);
    let inputs = preview_inputs(&mut app, 8, false).expect("the stroke's re-cut starts");

    app.handle_undo(&ctx);
    let landed = PreviewResult { version: inputs.version, ..PreviewResult::mask_for_test(8, std::sync::Arc::new(px(9))) };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &app.batch.items[0];
    assert!(item.shown_result().is_some_and(|r| *r == px(1)), "the undone stroke's re-cut does not land");
    assert!(app.batch.items[0].timeline.can_redo(), "and the redo step is intact");
}

/// Switching away releases an image's result and switching back restores
/// it from the timeline; neither costs or moves an undo step.
#[test]
fn switching_images_keeps_every_undo_step() {
    use crate::gui::item::BatchStatus;
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(px(0)));
    item.result_rgba = Some(Arc::new(px(1)));
    item.commit(StepKind::Process);
    item.result_rgba = Some(Arc::new(px(2)));
    item.commit(StepKind::Process);
    push_test_item(&mut app, 2).dimensions = (8, 8);

    app.batch.select_item(0);
    app.sync_selected_batch_textures(&ctx);
    app.batch.select_item(1);
    app.sync_selected_batch_textures(&ctx);
    assert!(app.batch.items[0].result_rgba.is_none(), "the background result is released");
    app.batch.select_item(0);
    app.sync_selected_batch_textures(&ctx);
    assert!(app.batch.items[0].shown_result().is_some_and(|r| *r == px(2)), "and back on return");

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "undo shows the first result");
    app.handle_undo(&ctx);
    assert_eq!(app.batch.items[0].status(), BatchStatus::Pending, "and the next undo the original");
}

/// With Magic Brush on, the shown image and the one before it keep their
/// embeddings, so flipping between two is instant; paging further drops
/// the rest instead of holding 16 MB per visited image.
#[test]
fn magic_brush_keeps_the_shown_and_the_previous_embedding_only() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::BiRefNetLite);
    let embedding = Arc::new(prunr_core::sam::SamEmbedding {
        image_embed: Vec::new(), high_res_feats_0: Vec::new(), high_res_feats_1: Vec::new(),
    });
    for id in 1..=3 {
        push_test_item(&mut app, id).dimensions = (8, 8);
    }
    app.magic_brush_state.activate();
    // Each image is encoded while it is shown.
    for idx in 0..3 {
        app.batch.select_item(idx);
        app.sync_selected_batch_textures(&ctx);
        app.batch.items[idx].magic_brush_embedding = Some(Arc::clone(&embedding));
    }
    let kept: Vec<bool> = app.batch.items.iter().map(|i| i.magic_brush_embedding.is_some()).collect();
    assert_eq!(kept, [false, true, true], "A dropped; B (previous) and C (shown) kept");
}

/// The drift check re-applies the mask only where a mask was applied:
/// a Full-lines result, or an eraser or upscale result, with non-default
/// mask settings must stay as it is.
#[test]
fn the_drift_check_never_recuts_a_result_that_used_no_mask() {
    use std::sync::Arc;
    for (model, line_mode) in [
        (SettingsModel::Silueta, prunr_core::LineMode::EdgesOnly),
        (SettingsModel::RealEsrganUpscale, prunr_core::LineMode::Off),
        (SettingsModel::ColorizeDdcolor, prunr_core::LineMode::Off),
    ] {
        let mut app = app_with_model(model);
        let item = push_test_item(&mut app, 7);
        give_cutout(item);
        item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
        item.settings.line_mode = line_mode;
        item.settings.gamma = 1.5;
        let kind = model.to_model_kind().unwrap_or(prunr_core::ModelKind::BiRefNetLite);
        item.applied_recipe = Some(item.settings.current_recipe(kind));
        app.batch.select_item(0);
        app.recipe_drift_tripwire();
        assert!(!app.processor.live_preview.is_pending_for(7), "{model:?} {line_mode:?}: no mask was applied, nothing to re-apply");
    }
}

/// A cut-out with a selection on it, nothing queued yet.
fn cutout_with_selection(app: &mut PrunrApp, id: u64) {
    use std::sync::Arc;
    let item = push_test_item(app, id);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_cutout(item);
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.applied_recipe = Some(item.settings.current_recipe(prunr_core::ModelKind::Silueta));
    item.commit(StepKind::Process);
    let selection = crate::gui::processor::merge_stroke(None, make_mask(8, 8));
    item.commit_selection_mask(Arc::new(selection), StepKind::Stroke);
}

/// The switch says what the selection does to the cut-out: Restore
/// brings it back, Erase removes it, Select leaves the cut-out alone.
#[test]
fn the_switch_says_what_the_selection_does() {
    use crate::gui::brush_state::CutoutAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 4);
    let item = &app.batch.items[0];
    let at = |action| item.correction_plane(action).map(|p| p.cells()[2 * 8 + 2]);
    assert!(at(CutoutAction::Restore).is_some_and(|v| v > 0), "restored");
    assert!(at(CutoutAction::Erase).is_some_and(|v| v < 0), "erased");
    assert_eq!(at(CutoutAction::Select), None, "only selected");
}

/// Switching Restore / Erase applies the current selection anew, on
/// every image with one: an image off screen is dragged, copied and
/// saved with its result too.
#[test]
fn switching_the_action_recuts_every_selection() {
    use crate::gui::brush_state::CutoutAction;
    use crate::gui::views::brush_chip::BrushChipOutcome;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 5);
    cutout_with_selection(&mut app, 6);
    cutout_with_selection(&mut app, 7);
    app.batch.items[2].bake_selection(CutoutAction::Restore);
    app.batch.select_item(0);
    assert!(!app.processor.live_preview.is_pending_for(5));
    app.settings.brush.cutout_action = CutoutAction::Erase;
    app.apply_brush_change(BrushChipOutcome { action_changed: true, ..Default::default() });
    assert!(app.processor.live_preview.is_pending_for(5), "the re-cut is queued");
    assert!(app.processor.live_preview.is_pending_for(6), "off screen too");
    assert!(!app.processor.live_preview.is_pending_for(7), "edits alone do not follow the switch");
}

/// A re-cut landing on an image off screen leaves it as it was left:
/// result parked in the timeline, no texture, no decompressed tensor.
#[test]
fn a_recut_landing_off_screen_keeps_the_image_parked() {
    use crate::gui::live_preview::PreviewResult;
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 3);
    let inputs = preview_inputs(&mut app, 3, false).expect("the re-cut starts");
    let packed = inputs.packed.seg.clone().expect("sent packed");
    app.batch.items[0].set_on_screen(false);
    let unpacked = crate::gui::live_preview::Unpacked {
        seg: Some((packed.clone(), Arc::new(crate::gui::live_preview::decompress_seg(&packed).expect("unpacks")))),
        edge: None,
    };
    let landed = PreviewResult { version: inputs.version, unpacked, ..PreviewResult::mask_for_test(3, Arc::new(px(9))) };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &mut app.batch.items[0];
    assert!(item.result_rgba.is_none(), "parked");
    assert!(!item.result_tex_pending, "no texture for an image off screen");
    assert!(item.volatile_seg_tensor.is_none() && item.start_photo.is_none(), "working copies released");
    item.unpark_result();
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(9)), "the re-cut is what comes back");
}

/// Delete folds the selection into the cut-out's edits as one step and
/// starts a new area; Undo brings the selection back and the edits away.
#[test]
fn delete_on_a_cutout_keeps_the_erase_and_undo_returns_the_selection() {
    use crate::gui::views::selection_action_bar::SelectionAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 6);
    let selection = app.batch.items[0].selection_mask.clone();

    app.handle_selection_action(0, SelectionAction::Delete, &egui::Context::default());
    assert!(app.batch.items[0].selection_mask.is_some(), "folded off the UI thread");
    super::fixtures::settle_selection_edits(&mut app);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none(), "a new area starts");
    let edits = item.edits.as_ref().expect("the erase is kept");
    assert!(edits.cells()[2 * 8 + 2] < 0 && edits.cells()[0] == 0);
    assert_eq!(item.timeline.undo_steps(), 3, "the run, the stroke and the fold");

    let item = &mut app.batch.items[0];
    let _ = item.undo_step();
    assert!(item.edits.is_none(), "undo takes the erase away");
    assert_eq!(item.selection_mask.as_ref().map(|m| m.content_hash()), selection.map(|m| m.content_hash()));
}

/// Restore on the selection brings the area back on top of an earlier
/// erase: the newer edit wins where it lands.
#[test]
fn restore_wins_over_an_earlier_erase() {
    use crate::gui::views::selection_action_bar::SelectionAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 8);
    let ctx = egui::Context::default();
    app.handle_selection_action(0, SelectionAction::Delete, &ctx);
    super::fixtures::settle_selection_edits(&mut app);
    let selection = crate::gui::processor::merge_stroke(None, make_mask(8, 8));
    app.batch.items[0].commit_selection_mask(std::sync::Arc::new(selection), StepKind::Stroke);
    app.handle_selection_action(0, SelectionAction::Restore, &ctx);
    super::fixtures::settle_selection_edits(&mut app);
    let edits = app.batch.items[0].edits.as_ref().expect("edits");
    assert!(edits.cells()[2 * 8 + 2] > 0);
}

/// The correction is built once per change of its inputs: a second
/// dispatch with the same edits and selection reuses the plane.
#[test]
fn the_correction_is_built_once() {
    use crate::gui::brush_state::CutoutAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 10);
    let item = &mut app.batch.items[0];
    item.edits = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-50; 64])));
    let first = item.correction_plane(CutoutAction::Restore).unwrap();
    let second = item.correction_plane(CutoutAction::Restore).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    let erase = item.correction_plane(CutoutAction::Erase).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first, &erase), "another action builds anew");
}

/// A live render on a cut-out with edits and a selection hands its thread
/// the correction's inputs, not a plane built on the UI thread; the
/// plane the render built is kept, so the next render reads it.
#[test]
fn a_render_builds_the_correction_on_its_own_thread() {
    use crate::gui::brush_state::CutoutAction;
    use crate::gui::item::Correction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 10);
    app.settings.brush.cutout_action = CutoutAction::Restore;
    app.batch.items[0].edits = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-50; 64])));
    let dispatched = preview_inputs(&mut app, 10, false).and_then(|i| i.correction);
    assert!(matches!(dispatched, Some(Correction::Build(_))), "the render gets the inputs");
    let item = &app.batch.items[0];
    let Some(Correction::Build(inputs)) = item.correction(CutoutAction::Restore) else { panic!("built off the UI thread") };
    let (plane, built) = Correction::Build(inputs).resolve();
    item.keep_built_correction(built.expect("a build"));
    let Some(Correction::Ready(kept)) = item.correction(CutoutAction::Restore) else { panic!("the build is kept") };
    assert!(std::sync::Arc::ptr_eq(&plane, &kept));
}

/// The cached correction (up to three full-size planes) goes once it
/// cannot be hit again: the selection baked, or the image off screen.
#[test]
fn the_correction_cache_lets_go_after_a_bake_and_off_screen() {
    use crate::gui::brush_state::CutoutAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 10);
    let item = &mut app.batch.items[0];
    item.edits = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-50; 64])));
    let plane = item.correction_plane(CutoutAction::Restore).unwrap();
    let held_edits = std::sync::Arc::clone(item.edits.as_ref().unwrap());
    item.drop_hot_tensors();
    assert_eq!(std::sync::Arc::strong_count(&held_edits), 2, "off screen: the cache holds no edits");
    assert_eq!(std::sync::Arc::strong_count(&plane), 1, "off screen: the cache holds no plane");
    item.correction_plane(CutoutAction::Restore).unwrap();
    assert!(item.bake_selection(CutoutAction::Restore));
    let baked = item.edits.as_ref().unwrap();
    assert_eq!(std::sync::Arc::strong_count(baked), 2, "baked: the working copy and its step, the cache none");
}

/// A step's image shows its selection as the switch stood then; an undo
/// back onto a live selection re-applies it as the switch stands now.
#[test]
fn undo_onto_a_selection_reapplies_the_current_action() {
    use crate::gui::brush_state::CutoutAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.brush.cutout_action = CutoutAction::Erase;
    cutout_with_selection(&mut app, 11);
    add_corner_stroke(&mut app);
    app.settings.brush.cutout_action = CutoutAction::Restore;

    app.handle_undo(&egui::Context::default());
    assert!(app.batch.items[0].selection_mask.is_some());
    assert!(app.processor.live_preview.is_pending_for(11), "the selection is re-applied as Restore");
}

/// An undo to a step another model made does not re-cut it from the
/// cached segmentation, which would rewrite that step with this model's
/// cut-out.
#[test]
fn undo_onto_another_models_step_does_not_recut_it() {
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 12);
    add_corner_stroke(&mut app);
    app.batch.items[0].applied_recipe = Some(app.batch.items[0].settings.current_recipe(prunr_core::ModelKind::U2net));
    app.batch.items[0].commit(StepKind::Process);
    app.batch.items[0].applied_recipe = Some(app.batch.items[0].settings.current_recipe(prunr_core::ModelKind::Silueta));
    // The tensor stays U2net's, as after reprocessing with another model.
    give_tensor_of(&mut app.batch.items[0], prunr_core::ModelKind::U2net);
    app.handle_undo(&egui::Context::default());
    assert!(!app.processor.live_preview.is_pending_for(12));
}

/// Switching model leaves the cut-out on screen, so the brushes still
/// correct it: Escape folds a live Erase into the edits, and the next
/// run re-applies them, whichever model makes it.
#[test]
fn a_model_switch_keeps_the_shown_cutout_correctable() {
    use crate::gui::brush_state::CutoutAction;
    use crate::gui::tests::harness::{click, harness, settle};
    let mut h = harness();
    {
        let app = h.state_mut();
        app.settings.model = SettingsModel::Silueta;
        app.settings.brush.cutout_action = CutoutAction::Erase;
        cutout_with_selection(app, 13);
        app.batch.select_item(0);
    }
    settle(&mut h);
    click(&mut h, "Model");
    let row = format!("BiRefNet\n{}", crate::gui::views::model_blurb(SettingsModel::BiRefNetLite));
    click(&mut h, &row);
    assert_eq!(h.state().settings.model, SettingsModel::BiRefNetLite);
    let item = &h.state().batch.items[0];
    assert!(item.corrects_cutout(SettingsModel::BiRefNetLite), "the Silueta cut-out is still on screen");
    h.key_press(egui::Key::Escape);
    crate::gui::tests::harness::settle_until_selection_lands(&mut h);
    let item = &h.state().batch.items[0];
    assert!(item.selection_mask.is_none(), "Escape folded the selection");
    assert!(item.edits.is_some(), "into the edits the next run re-applies");
}

/// A cached segmentation re-cuts only the step it cut. No model's pass on
/// a cut-out already holds the cut in its start, and re-cutting it squared
/// its edges' transparency; a tweak of that pass keeps it a pass. Another
/// model's cut is never re-cut with it either.
#[test]
fn the_cached_segmentation_recuts_only_the_step_it_cut() {
    use crate::gui::item::{Cutter, Engine};
    use crate::gui::live_preview::PreviewResult;
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let blank = || Arc::new(image::RgbaImage::new(8, 8));
    let mut app = app_with_model(SettingsModel::None);
    let item = push_test_item(&mut app, 4);
    item.dimensions = (8, 8);
    item.source_rgba = Some(blank());
    give_tensor(item);
    item.land_cut_for_test(Cutter::Model(prunr_core::ModelKind::Silueta), blank());
    assert!(item.hot_seg_tensor().is_some(), "Silueta's own cut");

    item.start_cut(Some(StepKind::Process), Cutter::NoModel);
    item.chain_start_built(item.job_run().expect("a run"), blank());
    item.result_rgba = Some(blank());
    item.finish_job();
    assert!(item.hot_seg_tensor().is_none(), "No model's pass is not re-cut");

    let inputs = preview_inputs(&mut app, 4, true).expect("the pass re-renders");
    assert!(inputs.seg_tensor.is_none());
    let landed = PreviewResult { version: inputs.version, start: inputs.start.clone(), ..PreviewResult::mask_for_test(4, blank()) };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &mut app.batch.items[0];
    assert!(item.hot_seg_tensor().is_none(), "still a pass after its tweak");

    item.record_origin(Engine::Segment, None, Some(Cutter::Model(prunr_core::ModelKind::BiRefNetLite)));
    assert!(item.hot_seg_tensor().is_none(), "another model's cut");
    item.record_origin(Engine::Segment, None, None);
    assert!(item.hot_seg_tensor().is_some(), "a step whose cutter is not known is re-cut as before");
}

/// A Paint Brush dot on cell `i` of an 8 × 8 image.
fn dot(i: usize) -> crate::gui::selection_queue::SelectionOp {
    let (x, y) = ((i % 8) as f32 + 0.5, (i / 8) as f32 + 0.5);
    let drawn = super::fixtures::painted(prunr_core::brush::BrushShape::Square, 1.0, &[(x, y, 0.5)], super::fixtures::HARD);
    crate::gui::selection_queue::SelectionOp::Stroke { drawn, dims: (8, 8), source: None }
}

/// Selection edits run off the UI thread in order: two strokes queued
/// before either lands build on each other and land as two steps; a run
/// waits for them; an undo pressed meanwhile undoes the last stroke once
/// it has landed.
#[test]
fn queued_strokes_land_in_order_and_an_undo_waits_for_them() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::None);
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    app.batch.select_item(0);
    let steps = app.batch.items[0].timeline.undo_steps();
    app.queue_selection_op(1, dot(0));
    app.queue_selection_op(1, dot(9));
    assert_eq!(app.run_refusal(&app.batch.items[0]), Some(crate::gui::capability::Reason::FinishingSelection), "a run waits");
    app.handle_undo(&ctx);
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps, "the undo waits too");
    super::fixtures::settle_selection_edits(&mut app);
    app.land_selection_edits(&ctx);
    let item = &app.batch.items[0];
    assert_eq!(item.timeline.undo_steps(), steps + 1, "two strokes landed, then the undo took the second back");
    let cells = item.selection_mask.as_ref().expect("the first stroke").cells();
    assert!(cells[0] != 0 && cells[9] == 0);
}

/// An edit made on a selection the image has since left goes back in
/// the queue against what it shows now: the stroke is not lost.
#[test]
fn an_edit_on_a_changed_selection_is_made_again() {
    let mut app = app_with_model(SettingsModel::None);
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    app.batch.select_item(0);
    app.queue_selection_op(1, dot(3));
    // The selection changes while the stroke is being merged.
    app.batch.items[0].commit_selection_mask(make_mask(8, 8).into(), StepKind::Selection);
    super::fixtures::settle_selection_edits(&mut app);
    let merged = app.batch.items[0].selection_mask.as_ref().expect("a selection").cells();
    assert!(merged[3] != 0 && merged[2 * 8 + 2] != 0, "the stroke landed on the new selection");
}

/// A stroke that adds nothing still hands the image its planes, so the
/// next stroke queued on it lands without being made again; strokes
/// queued on a selection changed under them all land, the queue settles.
#[test]
fn strokes_settle_on_no_op_and_changed_selections() {
    let mut app = app_with_model(SettingsModel::None);
    let item = push_test_item(&mut app, 1);
    item.dimensions = (8, 8);
    app.batch.select_item(0);
    app.queue_selection_op(1, dot(5));
    super::fixtures::settle_selection_edits(&mut app);
    let steps = app.batch.items[0].timeline.undo_steps();
    // The same stroke again adds nothing; one queued after it builds on it.
    app.queue_selection_op(1, dot(5));
    app.queue_selection_op(1, dot(6));
    super::fixtures::settle_selection_edits(&mut app);
    assert_eq!(app.batch.items[0].timeline.undo_steps(), steps + 1, "one step: the second stroke");
    // Three strokes, and the selection changes under them.
    for x in [10, 11, 12] {
        app.queue_selection_op(1, dot(x));
    }
    let mut cells = vec![0i8; 64];
    cells[40] = prunr_core::selection::FULL;
    app.batch.items[0].commit_selection_mask(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, cells)), StepKind::Selection);
    super::fixtures::settle_selection_edits(&mut app);
    let cells = app.batch.items[0].selection_mask.as_ref().expect("a selection").cells();
    assert!([10, 11, 12, 40].iter().all(|&x| cells[x] != 0), "every stroke landed on the new selection");
}
