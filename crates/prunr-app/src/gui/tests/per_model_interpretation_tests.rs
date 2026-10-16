//! Brushes and selections against the app: which active model re-cuts on a
//! committed selection (`LivePreview::is_pending_for`), how strokes sit on
//! the undo timeline, the live preview's tensor reuse, the Magic Brush
//! retune and embedding cache, the cut-out switch (Restore / Erase /
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

/// The live preview's mask inputs for `id`, at the app's current selection
/// style and cut-out action.
fn preview_inputs(app: &mut PrunrApp, id: u64, filter_only: bool) -> Option<DispatchInputs> {
    let style = app.selection_style();
    let action = app.settings.brush.cutout_action;
    let role = if filter_only { crate::gui::knob_catalog::ModelRole::Filters } else { crate::gui::knob_catalog::ModelRole::Cutout };
    PrunrApp::build_preview_inputs(&mut app.batch.items, id, PreviewKind::Mask, role, style, action).ok()
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

/// The live preview decompresses the segmentation tensor once per drag:
/// consecutive dispatch inputs share one plane, and replacing the cached
/// tensor drops it.
#[test]
fn preview_inputs_reuse_the_decompressed_tensor_until_it_changes() {
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 3);
    item.dimensions = (8, 8);
    item.source_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    give_cutout(item);
    let first = preview_inputs(&mut app, 3, false).expect("inputs").seg_tensor.expect("tensor");
    let second = preview_inputs(&mut app, 3, false).expect("inputs").seg_tensor.expect("tensor");
    assert!(Arc::ptr_eq(&first, &second), "second tick must reuse the hot tensor");
    let item = app.batch.find_by_id_mut(3).unwrap();
    item.set_cached_tensor(None);
    assert!(item.volatile_seg_tensor.is_none());
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

/// Moving the Confidence knob re-thresholds the last Magic Brush stroke
/// from its kept logits: the selection changes, the stroke keeps its single
/// undo entry, and the retune follows the selection it committed.
#[test]
fn confidence_change_retunes_the_last_stroke_in_place() {
    use crate::gui::item::LastDecode;
    use crate::gui::processor::DecodedSelection;
    use prunr_core::selection::BrushMode;
    use std::sync::Arc;
    let mut app = app_with_model(SettingsModel::Silueta);
    app.settings.brush.magic_confidence_threshold = 0.5;
    let item = push_test_item(&mut app, 9);
    item.dimensions = (8, 8);
    give_tensor(item);
    app.commit_selection_and_dispatch(9, make_mask(8, 8), StepKind::Stroke);
    let first_hash = app.batch.items[0].selection_hash.unwrap();
    assert_eq!(app.batch.items[0].timeline.undo_steps(), 1);

    // Logits ramp left→right: a higher confidence keeps fewer columns.
    let m = prunr_core::sam::SAM_MASK_RESOLUTION as usize;
    let mut masks = vec![-9.0f32; 3 * m * m];
    for y in 0..m {
        for x in 0..m {
            masks[y * m + x] = -4.0 + 8.0 * x as f32 / (m - 1) as f32;
        }
    }
    let output = Arc::new(prunr_core::sam::SamDecoderOutput { masks, iou_predictions: [0.9, 0.1, 0.1] });
    let committed_reading = app.settings.brush.magic_reading();
    app.batch.items[0].last_decode = Some(LastDecode {
        output, mode: BrushMode::Add,
        committed_hash: first_hash, reading: committed_reading,
    });

    app.settings.brush.magic_confidence_threshold = 0.9;
    app.maybe_rethreshold_last_stroke();
    assert!(app.magic_brush_state.rethreshold_in_flight);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let result = loop {
        if let Some(r) = app.processor.pump_sam_decoder_results().pop() { break r; }
        assert!(std::time::Instant::now() < deadline, "re-threshold never landed");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    let Ok(DecodedSelection::Retuned { merged, hash }) = result.result else {
        panic!("re-threshold decodes without a session");
    };
    app.apply_rethreshold(9, merged, hash, result.reading);

    let item = &app.batch.items[0];
    let new_hash = item.selection_hash.unwrap();
    assert_ne!(new_hash, first_hash, "the selection follows the knob");
    assert_eq!(item.timeline.undo_steps(), 1, "still one undo step for the stroke");
    let ld = item.last_decode.as_ref().unwrap();
    assert_eq!(ld.committed_hash, new_hash);
    assert!((ld.reading.confidence - 0.9).abs() < 1e-6);
    // The retuned selection keeps only the right-hand columns.
    let sel = item.selection_mask.as_ref().unwrap();
    assert!(sel.cells()[7] != 0 && sel.cells()[0] == 0);

    // The speck switch retunes the last stroke the same way.
    app.magic_brush_state.rethreshold_in_flight = false;
    app.maybe_rethreshold_last_stroke();
    assert!(!app.magic_brush_state.rethreshold_in_flight, "nothing changed, nothing to retune");
    app.settings.brush.magic_remove_specks = !app.settings.brush.magic_remove_specks;
    app.maybe_rethreshold_last_stroke();
    assert!(app.magic_brush_state.rethreshold_in_flight, "flipping Remove specks retunes the last stroke");
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
    item.set_status_for_test(crate::gui::item::BatchStatus::Done);
    item.result_rgba = Some(Arc::new(image::RgbaImage::new(8, 8)));
    item.commit(StepKind::Erase);
    app.commit_selection_and_dispatch(4, make_mask(8, 8), StepKind::Stroke);

    app.handle_undo(&ctx);
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none(), "the region stroke is undone");
    assert!(item.result_rgba.is_some() && item.status() == crate::gui::item::BatchStatus::Done, "the result stays");
}

#[test]
fn a_stroke_undo_brings_the_stored_result_back() {
    use std::sync::Arc;
    let ctx = egui::Context::default();
    let mut app = app_with_model(SettingsModel::Silueta);
    let item = push_test_item(&mut app, 5);
    item.dimensions = (8, 8);
    item.set_status_for_test(crate::gui::item::BatchStatus::Done);
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
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == before), "the pre-stroke image is back");
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
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(2)), "undoing the run brings back the corrected image it replaced");
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
    item.set_status_for_test(crate::gui::item::BatchStatus::Pending);

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
    give_tensor(item);
    super::fixtures::cut_out(item, prunr_core::ModelKind::Silueta);
    app.commit_selection_and_dispatch(8, make_mask(8, 8), StepKind::Stroke);
    let inputs = preview_inputs(&mut app, 8, false).expect("the stroke's re-cut starts");

    app.handle_undo(&ctx);
    let landed = PreviewResult { version: inputs.version, ..PreviewResult::mask_for_test(8, std::sync::Arc::new(px(9))) };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &app.batch.items[0];
    assert!(item.result_rgba.as_ref().is_some_and(|r| **r == px(1)), "the undone stroke's re-cut does not land");
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
    item.set_status_for_test(BatchStatus::Done);
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
    assert!(app.batch.items[0].result_rgba.as_ref().is_some_and(|r| **r == px(2)), "and back on return");

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
    let style = app.selection_style();
    let item = &app.batch.items[0];
    let at = |action| item.correction_plane(style, action).map(|p| p.cells()[2 * 8 + 2]);
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
    let style = app.selection_style();
    app.batch.items[2].bake_selection(style, CutoutAction::Restore);
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
    assert!(app.batch.items[0].volatile_seg_tensor.is_some());
    app.batch.items[0].set_on_screen(false);
    let landed = PreviewResult { version: inputs.version, ..PreviewResult::mask_for_test(3, Arc::new(px(9))) };
    app.apply_completed_previews(&ctx, vec![landed]);
    let item = &mut app.batch.items[0];
    assert!(item.result_rgba.is_none(), "parked");
    assert!(!item.result_tex_pending, "no texture for an image off screen");
    assert!(item.volatile_seg_tensor.is_none() && item.source_dyn.is_none(), "working copies released");
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
    let item = &app.batch.items[0];
    assert!(item.selection_mask.is_none(), "a new area starts");
    let edits = item.edits.as_ref().expect("the erase is kept");
    assert!(edits.cells()[2 * 8 + 2] < 0 && edits.cells()[0] == 0);
    assert_eq!(item.timeline.undo_steps(), 3, "the run, the stroke and the fold");

    let item = &mut app.batch.items[0];
    item.undo_step();
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
    let selection = crate::gui::processor::merge_stroke(None, make_mask(8, 8));
    app.batch.items[0].commit_selection_mask(std::sync::Arc::new(selection), StepKind::Stroke);
    app.handle_selection_action(0, SelectionAction::Restore, &ctx);
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
    let style = app.selection_style();
    let item = &mut app.batch.items[0];
    item.edits = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-50; 64])));
    let first = item.correction_plane(style, CutoutAction::Restore).unwrap();
    let second = item.correction_plane(style, CutoutAction::Restore).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    let erase = item.correction_plane(style, CutoutAction::Erase).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first, &erase), "another action builds anew");
}

/// The cached correction (up to three full-size planes) goes once it
/// cannot be hit again: the selection baked, or the image off screen.
#[test]
fn the_correction_cache_lets_go_after_a_bake_and_off_screen() {
    use crate::gui::brush_state::CutoutAction;
    let mut app = app_with_model(SettingsModel::Silueta);
    cutout_with_selection(&mut app, 10);
    let style = app.selection_style();
    let item = &mut app.batch.items[0];
    item.edits = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(8, 8, vec![-50; 64])));
    let plane = item.correction_plane(style, CutoutAction::Restore).unwrap();
    let held_edits = std::sync::Arc::clone(item.edits.as_ref().unwrap());
    item.drop_hot_tensors();
    assert_eq!(std::sync::Arc::strong_count(&held_edits), 2, "off screen: the cache holds no edits");
    assert_eq!(std::sync::Arc::strong_count(&plane), 1, "off screen: the cache holds no plane");
    item.correction_plane(style, CutoutAction::Restore).unwrap();
    assert!(item.bake_selection(style, CutoutAction::Restore));
    let baked = item.edits.as_ref().unwrap();
    assert_eq!(std::sync::Arc::strong_count(baked), 1, "baked: the cache holds no plane");
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
    settle(&mut h);
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
    item.chain_start_built(blank());
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
