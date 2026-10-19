//! Placement contracts for popovers, flyouts and overlays: what the canvas
//! does underneath them, and how each kind dismisses.

use egui_kittest::Harness;

use super::fixtures::push_test_item;
use super::harness::{
    assert_all_controls_named, click, harness, harness_builder, loaded, names, new_app, settle, tree,
};
use crate::gui::app::PrunrApp;
use crate::gui::item::BatchStatus;

/// A point on the canvas, below any popover and left of the sidebar.
const CANVAS_POINT: egui::Pos2 = egui::pos2(600.0, 820.0);

fn pointer_to(h: &mut Harness<'_, PrunrApp>, pos: egui::Pos2) {
    h.input_mut().events.push(egui::Event::PointerMoved(pos));
    h.step();
}

fn press(h: &mut Harness<'_, PrunrApp>, pos: egui::Pos2, pressed: bool) {
    h.input_mut().events.push(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
}

fn scroll(h: &mut Harness<'_, PrunrApp>, lines: f32) {
    h.input_mut().events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: egui::vec2(0.0, lines),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
}

#[test]
fn canvas_zooms_while_a_popover_is_open() {
    let mut h = loaded();
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Mask popover opened");
    let before = h.state().zoom_state.zoom;
    pointer_to(&mut h, CANVAS_POINT);
    scroll(&mut h, 1.0);
    settle(&mut h);
    assert_ne!(h.state().zoom_state.zoom, before, "scroll over the canvas zoomed");
    assert!(egui::Popup::is_any_open(&h.ctx), "scrolling did not close the popover");
}

#[test]
fn the_press_that_dismisses_a_popover_does_not_pan() {
    let mut h = loaded();
    click(&mut h, "Preset");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Preset popover opened");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(!h.state().zoom_state.is_panning, "the press under an open popover does not pan");
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(!egui::Popup::is_any_open(&h.ctx), "the click closed it");
    assert!(!h.state().zoom_state.is_panning);
    press(&mut h, CANVAS_POINT, true);
    assert!(h.state().zoom_state.is_panning, "the next press pans");
    press(&mut h, CANVAS_POINT, false);
}

#[test]
fn a_flyout_stays_open_while_the_canvas_is_used() {
    let mut h = loaded();
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Mask flyout opened");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(h.state().zoom_state.is_panning, "the canvas pans under the flyout");
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(egui::Popup::is_any_open(&h.ctx), "the canvas click did not close it");
    click(&mut h, "Mask");
    assert!(!egui::Popup::is_any_open(&h.ctx), "its chip closes it");
}

#[test]
fn escape_closes_the_flyout_and_keeps_the_selection() {
    let mut h = loaded();
    {
        let app = h.state_mut();
        let item = &mut app.batch.items[0];
        item.selection_mask = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(1, 1, vec![100])));
    }
    settle(&mut h);
    click(&mut h, "Mask");
    assert!(egui::Popup::is_any_open(&h.ctx));
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(!egui::Popup::is_any_open(&h.ctx), "Escape closed the flyout");
    assert!(h.state().batch.items[0].selection_mask.is_some(), "and did not clear the selection");
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(h.state().batch.items[0].selection_mask.is_none(), "the next Escape clears it");
}

#[test]
fn the_tool_strip_follows_the_paint_brush() {
    let mut h = loaded();
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "no strip without a tool");
    h.state_mut().brush_state.enable();
    settle(&mut h);
    let present = names(&h);
    for knob in ["Restore", "Erase", "Size", "Hardness", "Opacity", "More Paint Brush settings"] {
        assert!(present.iter().any(|n| n == knob), "{knob} missing from the strip: {present:?}");
    }
    assert_all_controls_named(&h, "paint brush strip");
    click(&mut h, "More Paint Brush settings");
    assert!(egui::Popup::is_any_open(&h.ctx), "the dots open the full panel");
    assert!(names(&h).iter().any(|n| n == "Feather"), "{:?}", names(&h));
    assert_all_controls_named(&h, "paint brush panel");
    h.state_mut().brush_state.disable();
    settle(&mut h);
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "the strip leaves with the tool");
}

/// The strip draws on the foreground layer: under an open dialog it
/// would sit on top of the dialog, so it leaves while one is up.
#[test]
fn the_tool_strip_leaves_while_a_dialog_is_open() {
    let mut h = loaded();
    h.state_mut().brush_state.enable();
    settle(&mut h);
    assert!(names(&h).iter().any(|n| n == "Hardness"), "the strip is up");
    h.state_mut().show_cli_help = true;
    settle(&mut h);
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "no strip over the command-line reference");
    h.state_mut().show_cli_help = false;
    settle(&mut h);
    assert!(names(&h).iter().any(|n| n == "Hardness"), "the strip is back once the dialog closes");
}

/// A native file dialog is a separate window the app keeps drawing
/// under; the canvas holds still until the user answers it.
#[test]
fn the_canvas_holds_still_while_a_file_dialog_is_up() {
    let mut h = loaded();
    click(&mut h, "Open");
    assert!(h.state().system.dialog_open(), "Open asked for a dialog");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(!h.state().zoom_state.is_panning, "no pan under the dialog");
    press(&mut h, CANVAS_POINT, false);
    h.state_mut().system.answer_for_test(None);
    settle(&mut h);
    assert!(!h.state().system.dialog_open(), "a cancel closes it");
    press(&mut h, CANVAS_POINT, true);
    assert!(h.state().zoom_state.is_panning, "the canvas pans again");
    press(&mut h, CANVAS_POINT, false);
}

/// Under an eraser both brushes carry the region knobs, since either
/// paints the region the erase fills.
#[test]
fn an_erasers_magic_brush_strip_has_the_region_knobs() {
    let mut h = loaded();
    h.state_mut().settings.model = crate::gui::settings::SettingsModel::Inpaint;
    h.state_mut().magic_brush_state.activate();
    settle(&mut h);
    let present = names(&h);
    for knob in ["Expand region", "Feather"] {
        assert!(present.iter().any(|n| n == knob), "{knob} missing from the strip: {present:?}");
    }
}

#[test]
fn the_tool_strip_follows_the_magic_brush() {
    let mut h = loaded();
    h.state_mut().magic_brush_state.activate();
    settle(&mut h);
    let present = names(&h);
    for knob in ["Confidence", "More Magic Brush settings"] {
        assert!(present.iter().any(|n| n == knob), "{knob} missing from the strip: {present:?}");
    }
    // Magic Brush picks objects; size, hardness and shape would change nothing.
    for knob in ["Size", "Hardness", "Circle", "Square", "Line"] {
        assert!(!present.iter().any(|n| n == knob), "{knob} on the Magic Brush strip: {present:?}");
    }
    assert_all_controls_named(&h, "magic brush strip");
    click(&mut h, "More Magic Brush settings");
    assert!(egui::Popup::is_any_open(&h.ctx));
    let panel = names(&h);
    assert!(!panel.iter().any(|n| n == "Size" || n == "Hardness"), "{panel:?}");
    assert!(panel.iter().any(|n| n == "Remove specks"), "{panel:?}");
    assert_all_controls_named(&h, "magic brush panel");
}

/// Magic Brush's Depth switch trades Confidence for Range on the strip.
#[test]
fn the_magic_brush_strip_selects_by_depth() {
    let mut h = loaded();
    h.state_mut().hardware_install_cache.depth = true;
    h.state_mut().magic_brush_state.activate();
    settle(&mut h);
    click(&mut h, "By depth");
    assert!(h.state().settings.brush.magic_depth);
    let present = names(&h);
    assert!(present.iter().any(|n| n == "Range") && !present.iter().any(|n| n == "Confidence"), "{present:?}");
    assert_all_controls_named(&h, "magic brush strip with depth");
    assert!(h.state().model_store.is_none(), "the model is installed: no store");
}

/// Turning on a depth knob without the depth model opens the Model Store
/// on Depth, from the Depth strip's own button too.
#[test]
fn a_depth_knob_without_the_model_offers_the_store() {
    let mut h = loaded();
    h.state_mut().hardware_install_cache = Default::default();
    h.state_mut().magic_brush_state.activate();
    settle(&mut h);
    click(&mut h, "By depth");
    assert_eq!(h.state().model_store.map(|r| r.filter), Some(Some(prunr_models::ModelCategory::Depth)));

    let mut h = loaded();
    {
        let app = h.state_mut();
        app.hardware_install_cache = Default::default();
        app.settings.model = crate::gui::settings::SettingsModel::None;
    }
    settle(&mut h);
    click(&mut h, "Depth");
    click(&mut h, crate::gui::views::get_model::label(prunr_core::depth::MODEL.into()));
    settle(&mut h);
    assert_eq!(h.state().model_store.map(|r| r.filter), Some(Some(prunr_models::ModelCategory::Depth)));
    assert!(!egui::Popup::is_any_open(&h.ctx), "the Depth strip closed");
}

/// While Set focus waits, a click on the canvas belongs to it: no pan,
/// and the wait ends whether or not the click hit the image.
#[test]
fn picking_the_focus_takes_the_canvas_click() {
    let mut h = loaded();
    {
        let app = h.state_mut();
        app.settings.model = crate::gui::settings::SettingsModel::None;
        app.picking_depth_focus = true;
    }
    settle(&mut h);
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    assert!(!h.state().zoom_state.is_panning, "the press does not pan");
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(!h.state().picking_depth_focus, "the click ended the wait");
}

#[test]
fn bracket_keys_step_the_brush_size() {
    let mut h = loaded();
    h.state_mut().brush_state.enable();
    settle(&mut h);
    let start = h.state().settings.brush.radius;
    h.key_press(egui::Key::CloseBracket);
    settle(&mut h);
    let larger = h.state().settings.brush.radius;
    assert!(larger > start, "] grows the brush: {start} -> {larger}");
    h.key_press(egui::Key::OpenBracket);
    settle(&mut h);
    assert!(h.state().settings.brush.radius < larger, "[ shrinks it");
}

#[test]
fn applying_a_preset_closes_the_popover() {
    let mut h = loaded();
    click(&mut h, "Preset");
    assert!(egui::Popup::is_any_open(&h.ctx));
    click(&mut h, "Prunr");
    assert!(!egui::Popup::is_any_open(&h.ctx), "a pick closes a set-and-go popover");
}

/// The first-launch runtime prompt is a modal for each runtime it can
/// offer: its buttons are named, and Escape closes it as Not now.
#[test]
fn the_runtime_prompt_is_a_modal_that_escape_closes() {
    use crate::runtime_install::RuntimeId;
    let mut h = loaded();
    for &rt in RuntimeId::OFFERED {
        h.state_mut().runtime_prompt = Some(rt);
        settle(&mut h);
        let present = names(&h);
        for n in [rt.install_button_label(), "Not now", "Open Settings"] {
            assert!(present.iter().any(|x| x == n), "{n} missing: {present:?}");
        }
        assert_all_controls_named(&h, rt.display_name());
        h.key_press(egui::Key::Escape);
        settle(&mut h);
        assert!(h.state().runtime_prompt.is_none(), "Escape closed the {rt:?} prompt");
        assert!(h.state().settings.is_runtime_prompt_snoozed(rt), "as Not now");
    }
}

/// The save dialog takes the popover's place and opens ready for a name:
/// text typed at once goes into it. (Saving writes the presets folder, so
/// Enter is checked live.)
#[test]
fn the_save_preset_dialog_takes_a_name_typed_at_once() {
    let mut h = loaded();
    click(&mut h, "Preset");
    click(&mut h, "Save current as…");
    assert!(!egui::Popup::is_any_open(&h.ctx), "opening the dialog closed the popover");
    h.event(egui::Event::Text("Mine".into()));
    settle(&mut h);
    assert_eq!(h.state().save_preset.as_deref(), Some("Mine"), "the typing went into the name");
}

#[test]
fn escape_closes_a_modal_before_touching_the_selection() {
    let mut h = loaded();
    {
        let app = h.state_mut();
        app.batch.items[0].selection_mask = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(1, 1, vec![100])));
        app.model_store = Some(crate::gui::views::adjustments_toolbar::ModelStoreRequest { filter: None });
    }
    settle(&mut h);
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(h.state().model_store.is_none(), "Escape closed the Model Store");
    assert!(h.state().batch.items[0].selection_mask.is_some(), "the selection survived");
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(h.state().batch.items[0].selection_mask.is_none(), "the next Escape clears it");
}

/// The Sketch flyout names every control in each mode, for each line
/// style (its color pickers) and each pre-filter (its steps).
#[test]
fn the_sketch_flyout_names_every_control_in_every_mode() {
    use prunr_core::{InputTransform, LineMode, LineStyle};
    for mode in [LineMode::SubjectOutline, LineMode::EdgesOnly] {
        for (i, style) in LineStyle::ALL.iter().enumerate() {
            let mut h = loaded();
            {
                let s = &mut h.state_mut().batch.items[0].settings;
                s.line_mode = mode;
                s.line_style = *style;
                s.input_transform = InputTransform::ALL[i % InputTransform::ALL.len()];
            }
            settle(&mut h);
            click(&mut h, "Sketch");
            assert!(egui::Popup::is_any_open(&h.ctx), "the Sketch flyout opened");
            let present = names(&h);
            assert!(present.iter().any(|n| n == "Line width (px)"), "{mode:?}: {present:?}");
            assert_all_controls_named(&h, &format!("sketch flyout {mode:?} {style:?}"));
        }
    }
}

/// The Background flyout names every control with a background image:
/// picking one takes a file dialog, so the catalog's clicks never reach it.
#[test]
fn the_background_flyout_names_every_control_with_an_image() {
    let mut h = loaded();
    h.state_mut().batch.items[0].set_bg_image(image::DynamicImage::new_rgba8(4, 4), None);
    settle(&mut h);
    click(&mut h, "Background");
    assert!(egui::Popup::is_any_open(&h.ctx), "the Background flyout opened");
    assert_all_controls_named(&h, "background strip, image");
    click(&mut h, "All background settings");
    assert_all_controls_named(&h, "background panel, image");
}

/// Colorize's Strength is a flyout: it names its slider and stays open
/// while the canvas is used.
#[test]
fn the_colorize_strength_flyout_names_its_control_and_stays_open() {
    let mut h = loaded();
    h.state_mut().settings.model = crate::gui::settings::SettingsModel::ColorizeDdcolor;
    settle(&mut h);
    click(&mut h, "Strength");
    assert!(egui::Popup::is_any_open(&h.ctx), "Strength opened");
    assert_all_controls_named(&h, "Strength on the colorizer");
    pointer_to(&mut h, CANVAS_POINT);
    press(&mut h, CANVAS_POINT, true);
    press(&mut h, CANVAS_POINT, false);
    settle(&mut h);
    assert!(egui::Popup::is_any_open(&h.ctx), "the canvas click did not close it");
    click(&mut h, "Strength");
    assert!(!egui::Popup::is_any_open(&h.ctx), "its chip closes it");
}

/// Quality is a set-and-go popover: picking one closes it.
#[test]
fn picking_a_quality_closes_its_popover() {
    let mut h = loaded();
    h.state_mut().settings.model = crate::gui::settings::SettingsModel::SdInpaint;
    h.state_mut().settings.brush.sd_scheduler = crate::gui::brush_state::SdScheduler::DpmPlusPlus2MKarras;
    settle(&mut h);
    click(&mut h, "Quality");
    let quality = names(&h).into_iter().find(|n| n.starts_with("Quality") && n != "Quality").expect("the Quality row");
    click(&mut h, &quality);
    assert!(!egui::Popup::is_any_open(&h.ctx), "a pick closes it");
}

/// Save options is a set-and-go popover by the Save button: it names its
/// controls, offers Separate layers only for a cut-out model, and a format
/// pick sticks as the setting.
#[test]
fn save_options_name_their_controls_and_follow_the_model() {
    use crate::gui::settings::SettingsModel;
    use std::sync::Arc;
    for (model, layers) in [(SettingsModel::BiRefNetLite, true), (SettingsModel::Inpaint, false)] {
        let mut h = harness();
        {
            let app = h.state_mut();
            app.settings.model = model;
            let item = push_test_item(app, 1);
            item.dimensions = (4, 4);
            item.source_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            item.result_rgba = Some(Arc::new(image::RgbaImage::new(4, 4)));
            crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
            item.commit(crate::gui::timeline::StepKind::Process);
            app.batch.select_item(0);
        }
        settle(&mut h);
        click(&mut h, "Save options");
        assert_all_controls_named(&h, "save options");
        let names = names(&h);
        assert_eq!(names.iter().any(|n| n == "Separate layers"), layers, "{model:?}");
        click(&mut h, "WebP");
        assert_eq!(h.state().settings.export_format, prunr_core::ExportFormat::Webp);
    }
}

/// The Outline and shadow flyout names every control, with the outline on
/// and a Drop shadow (every row showing).
#[test]
fn the_outline_and_shadow_flyout_names_every_control() {
    let mut h = loaded();
    {
        let d = &mut h.state_mut().batch.items[0].settings.decoration;
        d.outline_width = 6.0;
        d.shadow = prunr_core::ShadowKind::Drop;
    }
    settle(&mut h);
    click(&mut h, "Outline and shadow");
    click(&mut h, "All outline and shadow settings");
    assert_all_controls_named(&h, "outline and shadow");
    let names = names(&h);
    for expected in ["Width", "Color", "No shadow", "Soft", "Drop", "Opacity", "Blur", "Distance"] {
        assert!(names.iter().any(|n| n.starts_with(expected)), "{expected} missing from {names:?}");
    }
}

/// With the SD eraser's brush on, the strip offers "Try another", greyed
/// until a fill exists, and names it.
#[test]
fn the_sd_eraser_strip_offers_try_another() {
    let mut h = loaded();
    h.state_mut().settings.model = crate::gui::settings::SettingsModel::SdInpaint;
    settle(&mut h);
    h.state_mut().brush_state.enable();
    settle(&mut h);
    let tree = tree(&h);
    let node = tree.iter().find(|n| n.name.as_deref() == Some("Try another")).expect("Try another on the strip");
    assert!(node.disabled, "greyed before any fill");
    assert_all_controls_named(&h, "SD eraser strip");
}

/// The Model picker lists a model that is not downloaded with "Get" and
/// its size; picking it opens the Model Store on its section instead of
/// switching to a model that cannot run.
#[test]
fn the_picker_offers_a_model_that_is_not_downloaded() {
    use crate::gui::views::get_model;
    let mut h = loaded();
    {
        let app = h.state_mut();
        app.hardware_install_cache = Default::default();
        app.settings.model = crate::gui::settings::SettingsModel::None;
    }
    settle(&mut h);
    click(&mut h, "Model");
    let row = format!("BiRefNet\n{}", get_model::badge_and_size(prunr_models::ModelId::BiRefNetLite));
    click(&mut h, &row);
    settle(&mut h);
    assert_eq!(h.state().model_store.map(|r| r.filter), Some(Some(prunr_models::ModelCategory::Segmentation)));
    assert_eq!(h.state().settings.model, crate::gui::settings::SettingsModel::None, "the model did not switch");
}

/// Without a face model the Faces strip offers the Model Store on its
/// Faces tab, and closes as the store opens rather than staying over it.
#[test]
fn the_faces_strip_opens_the_store_and_steps_aside() {
    let mut h = loaded();
    h.state_mut().hardware_install_cache = Default::default();
    settle(&mut h);
    click(&mut h, "Faces");
    click(&mut h, "Get a face model");
    settle(&mut h);
    let store = h.state().model_store;
    assert_eq!(store.map(|r| r.filter), Some(Some(prunr_models::ModelCategory::Face)));
    assert!(!egui::Popup::is_any_open(&h.ctx), "the Faces strip closed");
}

/// A dialog closed by a shortcut rather than its own backdrop or button
/// still ignores the click that opens it again: Help › Keyboard
/// shortcuts used to flash Settings open and shut after an earlier
/// Settings had been closed with its shortcut.
#[test]
fn a_dialog_reopened_from_a_menu_stays_open() {
    use crate::gui::views::shortcuts::Action;
    let mut h = loaded();
    let ctx = h.ctx.clone();
    h.state_mut().perform(Action::Settings, &ctx);
    settle(&mut h);
    assert!(h.state().show_settings);
    h.state_mut().perform(Action::Settings, &ctx);
    h.run_steps(30);
    assert!(!h.state().show_settings, "closed by its shortcut");
    click(&mut h, "Help");
    click(&mut h, "Keyboard shortcuts");
    h.run_steps(10);
    assert!(h.state().show_settings, "Settings opened on Shortcuts stays open");
}

/// The capsule and the canvas animate while a run works, for minutes at
/// a time: thirty frames a second keeps them smooth at half the drawing.
#[test]
fn a_run_in_flight_draws_thirty_frames_a_second() {
    use crate::gui::dispatch_progress::{DispatchProgress, ProgressKind};
    for status in [BatchStatus::Done, BatchStatus::Processing] {
        // egui takes the frame time off every wait it is asked for: at
        // the harness's default quarter second, each would read as none.
        let mut h = harness_builder().with_step_dt(1.0 / 60.0).build_eframe(|cc| new_app(cc));
        let photo = h.ctx.load_texture("photo", egui::ColorImage::filled([4, 4], egui::Color32::GRAY), Default::default());
        {
            let app = h.state_mut();
            let item = push_test_item(app, 1);
            crate::gui::tests::fixtures::put_in(item, status.clone());
            item.source_texture = Some(photo.clone().into());
            item.thumb.landed(photo);
            app.batch.select_item(0);
            app.processor.show_progress_for_test(DispatchProgress::new(ProgressKind::Upscale, prunr_core::Step::Upscaling.label()));
        }
        // Work landing off the UI thread (a texture, a startup probe on a
        // slow runner) asks for one frame at once; the pace is what the
        // frames settle to, and then holds.
        let delay = |h: &Harness<'_, PrunrApp>| h.output().viewport_output[&egui::ViewportId::ROOT].repaint_delay;
        let paced = |d: std::time::Duration| d >= std::time::Duration::from_millis(15);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        h.run_steps(3);
        while !paced(delay(&h)) && std::time::Instant::now() < deadline {
            h.step();
        }
        for _ in 0..3 {
            assert!(paced(delay(&h)), "{status:?}: the next frame is asked for {:?} ahead", delay(&h));
            h.step();
        }
    }
}

/// A strip's full panel keeps its width while its values change: Mask's
/// ends in a right-aligned row, which takes all the room it is given,
/// and the panel only had a minimum width, so every wider value pushed
/// it further right and it never came back.
#[test]
fn a_strip_s_full_panel_keeps_its_width() {
    use egui_kittest::kittest::Queryable;
    let mut h = loaded();
    click(&mut h, "Mask");
    click(&mut h, "All mask settings");
    let right = |h: &Harness<'_, PrunrApp>| h.get_by_label("Contrast").rect().right();
    let first = right(&h);
    for shift in [-20.0, 0.0, -20.0, 0.0] {
        h.state_mut().batch.items[0].settings.edge_shift = prunr_core::MaskSettings::edge_shift_for_grow(shift);
        h.run_steps(5);
    }
    assert_eq!(right(&h), first, "the panel's sliders end where they did");
    let strip_right = h.get_by_label("All mask settings").rect().right();
    assert!((first - strip_right).abs() < 100.0, "the panel spans the strip above it: {first} vs {strip_right}");
}
