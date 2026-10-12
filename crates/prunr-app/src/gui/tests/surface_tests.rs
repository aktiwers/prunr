//! Placement contracts for popovers, flyouts and overlays: what the canvas
//! does underneath them, and how each kind dismisses.

use egui_kittest::Harness;

use super::fixtures::push_test_item;
use super::tree_tests::{click, harness, settle};
use crate::gui::app::PrunrApp;
use crate::gui::item::BatchStatus;

/// A point on the canvas, below any popover and left of the sidebar.
const CANVAS_POINT: egui::Pos2 = egui::pos2(600.0, 820.0);

pub(super) fn loaded() -> Harness<'static, PrunrApp> {
    let mut h = harness();
    {
        let app = h.state_mut();
        push_test_item(app, 1).status = BatchStatus::Done;
        app.batch.select_item(0);
    }
    settle(&mut h);
    h
}

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
    use super::tree_tests::{assert_all_controls_named, names};
    let mut h = loaded();
    assert!(!names(&h).iter().any(|n| n == "Hardness"), "no strip without a tool");
    h.state_mut().brush_state.toggle();
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

#[test]
fn the_tool_strip_follows_the_magic_brush() {
    use super::tree_tests::{assert_all_controls_named, names};
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

#[test]
fn bracket_keys_step_the_brush_size() {
    let mut h = loaded();
    h.state_mut().brush_state.toggle();
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

#[test]
fn the_save_preset_dialog_is_a_modal_that_escape_closes() {
    use super::tree_tests::{assert_all_controls_named, names};
    let mut h = loaded();
    {
        let item = &mut h.state_mut().batch.items[0];
        item.selection_mask = Some(std::sync::Arc::new(prunr_core::selection::MaskArtifact::from_cells(1, 1, vec![100])));
    }
    click(&mut h, "Preset");
    click(&mut h, "Save current as…");
    assert!(!egui::Popup::is_any_open(&h.ctx), "opening the dialog closed the popover");
    let present = names(&h);
    for n in ["Save", "Cancel", "Close window"] {
        assert!(present.iter().any(|x| x == n), "{n} missing: {present:?}");
    }
    assert_all_controls_named(&h, "save preset dialog");
    h.key_press(egui::Key::Escape);
    settle(&mut h);
    assert!(!names(&h).iter().any(|x| x == "Close window"), "Escape closed the dialog");
    assert!(h.state().batch.items[0].selection_mask.is_some(), "and kept the selection");
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
    use super::tree_tests::{assert_all_controls_named, names};
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

/// The Background flyout names every control for each choice it can show.
#[test]
fn the_background_flyout_names_every_control_for_each_choice() {
    use super::tree_tests::assert_all_controls_named;
    use prunr_core::BgEffect;
    type Choose = fn(&mut crate::gui::item::BatchItem);
    let set: [(&str, Choose); 5] = [
        ("transparent", |_| {}),
        ("solid", |i| i.settings.bg = Some([10, 20, 30, 255])),
        ("image", |i| i.set_bg_image(image::DynamicImage::new_rgba8(4, 4), None)),
        ("blurred", |i| i.settings.bg_effect = BgEffect::BlurredSource { radius: 5 }),
        ("black and white", |i| i.settings.bg_effect = BgEffect::DesaturatedSource),
    ];
    for (name, apply) in set {
        let mut h = loaded();
        apply(&mut h.state_mut().batch.items[0]);
        settle(&mut h);
        click(&mut h, "Background");
        assert!(egui::Popup::is_any_open(&h.ctx), "{name}: the Background flyout opened");
        assert_all_controls_named(&h, &format!("background flyout, {name}"));
    }
}

/// The Effect flyout names every control for each effect, "None" first.
#[test]
fn the_effect_flyout_names_every_control_for_each_effect() {
    use super::tree_tests::{assert_all_controls_named, names};
    for style in prunr_core::FillStyle::ALL {
        let mut h = loaded();
        h.state_mut().batch.items[0].settings.fill_style = *style;
        settle(&mut h);
        click(&mut h, "Effect");
        assert!(egui::Popup::is_any_open(&h.ctx), "{style:?}: the Effect flyout opened");
        assert!(names(&h).iter().any(|n| n == "None"), "{style:?}: None is in the list");
        assert_all_controls_named(&h, &format!("effect flyout, {style:?}"));
    }
}

/// The upscale row's Scale popover and Refine flyout name every control.
#[test]
fn the_upscale_surfaces_name_every_control() {
    use super::tree_tests::{assert_all_controls_named, names};
    for (chip, expect) in [("Scale", "2\u{00d7}"), ("Refine", "Sharpen")] {
        let mut h = loaded();
        h.state_mut().settings.model = crate::gui::settings::SettingsModel::RealEsrganUpscale;
        settle(&mut h);
        click(&mut h, chip);
        assert!(egui::Popup::is_any_open(&h.ctx), "{chip} opened");
        assert!(names(&h).iter().any(|n| n == expect), "{chip}: {expect} shown");
        assert_all_controls_named(&h, &format!("{chip} on an upscaler"));
    }
}

/// The SD eraser's Quality and Prompt popovers and Advanced flyout name
/// every control, the prompts included.
#[test]
fn the_sd_eraser_surfaces_name_every_control() {
    use super::tree_tests::{assert_all_controls_named, names};
    for (chip, expect) in [("Quality", "Fast"), ("Prompt", "Negative prompt"), ("Advanced", "Guidance")] {
        let mut h = loaded();
        h.state_mut().settings.model = crate::gui::settings::SettingsModel::SdInpaint;
        settle(&mut h);
        click(&mut h, chip);
        assert!(egui::Popup::is_any_open(&h.ctx), "{chip} opened");
        let names = names(&h);
        assert!(names.iter().any(|n| n.starts_with(expect)), "{chip}: {expect} shown in {names:?}");
        assert_all_controls_named(&h, &format!("{chip} on the SD eraser"));
    }
}

/// Quality is a set-and-go popover: picking one closes it.
#[test]
fn picking_a_quality_closes_its_popover() {
    let mut h = loaded();
    h.state_mut().settings.model = crate::gui::settings::SettingsModel::SdInpaint;
    h.state_mut().settings.brush.sd_scheduler = crate::gui::brush_state::SdScheduler::DpmPlusPlus2MKarras;
    settle(&mut h);
    click(&mut h, "Quality");
    let quality = super::tree_tests::names(&h).into_iter().find(|n| n.starts_with("Quality") && n != "Quality").expect("the Quality row");
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
            item.status = BatchStatus::Done;
            item.commit(crate::gui::timeline::StepKind::Process);
            app.batch.select_item(0);
        }
        settle(&mut h);
        click(&mut h, "Save options");
        super::tree_tests::assert_all_controls_named(&h, "save options");
        let names = super::tree_tests::names(&h);
        assert_eq!(names.iter().any(|n| n == "Separate layers"), layers, "{model:?}");
        click(&mut h, "WebP");
        assert_eq!(h.state().settings.export_format, prunr_core::ExportFormat::Webp);
    }
}

/// The Outline & Shadow flyout names every control, with the outline on
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
    click(&mut h, "Outline & Shadow");
    click(&mut h, "All outline and shadow settings");
    super::tree_tests::assert_all_controls_named(&h, "outline and shadow");
    let names = super::tree_tests::names(&h);
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
    h.state_mut().brush_state.toggle();
    settle(&mut h);
    let tree = super::tree_tests::tree(&h);
    let node = tree.iter().find(|n| n.name.as_deref() == Some("Try another")).expect("Try another on the strip");
    assert!(node.disabled, "greyed before any fill");
    super::tree_tests::assert_all_controls_named(&h, "SD eraser strip");
}
