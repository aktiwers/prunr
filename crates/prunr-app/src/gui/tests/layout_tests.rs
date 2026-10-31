//! The toolbar fits the minimum window: in every model's configuration,
//! with an image loaded, a selection and a brush on, no two toolbar
//! controls overlap and none runs past the window's right edge.

use super::fixtures::push_test_item;
use crate::gui::automation::checks::first_overlap;
use super::harness::{click, harness_sized, names, settle, tree};
use egui_kittest::Harness;

use crate::gui::app::PrunrApp;
use crate::gui::brush_state::Tool;
use crate::gui::item::BatchStatus;
use crate::gui::settings::SettingsModel;
use crate::gui::theme::MIN_WINDOW_SIZE;

/// The two toolbar rows end above this y; the canvas strip starts below.
const TOOLBAR_BOTTOM: f32 = 100.0;

const MODELS: [SettingsModel; 4] = [
    SettingsModel::BiRefNetLite,
    SettingsModel::SdInpaint,
    SettingsModel::Inpaint,
    SettingsModel::RealEsrganUpscale,
];

/// The window at `width` with one processed 64×64 image, selected.
fn with_a_done_image(width: f32, model: SettingsModel) -> Harness<'static, PrunrApp> {
    let mut harness = harness_sized(egui::vec2(width, MIN_WINDOW_SIZE[1]));
    {
        let app = harness.state_mut();
        app.settings.model = model;
        let item = push_test_item(app, 1);
        item.dimensions = (64, 64);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        app.batch.select_item(0);
    }
    harness
}

/// The named controls `keep` picks, with their rects.
fn controls(harness: &Harness<'_, PrunrApp>, keep: impl Fn(&[f32; 4]) -> bool) -> Vec<(String, [f32; 4])> {
    tree(harness)
        .iter()
        .filter(|n| n.is_control())
        .filter_map(|n| Some((n.name.clone().unwrap_or_default(), n.rect?)))
        .filter(|(_, r)| r[2] > r[0] && keep(r))
        .collect()
}

/// What is wrong with the toolbar at `width` for `model`, if anything.
fn toolbar_clash(width: f32, model: SettingsModel) -> Option<String> {
    let mut harness = with_a_done_image(width, model);
    let mut cells = vec![0i8; 64 * 64];
    cells[64 * 10 + 10] = prunr_core::selection::FULL;
    harness.state_mut().batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(64, 64, cells), crate::gui::timeline::StepKind::Stroke);
    settle(&mut harness);
    // Upscale has no brush.
    if names(&harness).iter().any(|n| n == "Paint Brush") {
        click(&mut harness, "Paint Brush");
    }
    let bars = controls(&harness, |r| r[3] <= TOOLBAR_BOTTOM);
    if let Some((name, r)) = bars.iter().find(|(_, r)| r[2] > width) {
        return Some(format!("{name:?} ends at {} past {width}", r[2]));
    }
    first_overlap(&bars).map(|v| v.detail)
}

#[test]
fn the_toolbar_fits_the_minimum_window_in_every_mode() {
    for model in MODELS {
        if let Some(clash) = toolbar_clash(MIN_WINDOW_SIZE[0], model) {
            panic!("{model:?} at {}: {clash}", MIN_WINDOW_SIZE[0]);
        }
    }
}

/// Finds the narrowest width each mode lays out at; run to re-measure
/// `MIN_WINDOW_SIZE` after a toolbar change:
///   cargo test -p prunr-app --lib narrowest_toolbar -- --ignored --nocapture
#[test]
#[ignore = "measurement"]
fn narrowest_toolbar() {
    for model in MODELS {
        let mut width = 1400.0;
        while width > 600.0 && toolbar_clash(width - 10.0, model).is_none() {
            width -= 10.0;
        }
        eprintln!("{model:?}: fits down to {width} ({:?} below)", toolbar_clash(width - 10.0, model));
    }
}

/// A window with no room (the screen asleep, a minimised compositor
/// surface) still renders with a brush on: nothing asks egui for a
/// negative width.
#[test]
fn a_window_with_no_room_renders_with_a_brush_on() {
    let mut harness = harness_sized(egui::vec2(MIN_WINDOW_SIZE[0], MIN_WINDOW_SIZE[1]));
    {
        let app = harness.state_mut();
        app.settings.model = SettingsModel::BiRefNetLite;
        let item = push_test_item(app, 1);
        item.dimensions = (64, 64);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        app.batch.select_item(0);
        app.brush_state.enable();
    }
    settle(&mut harness);
    harness.set_size(egui::vec2(1.0, 1.0));
    settle(&mut harness);
}

/// The toolbar's layout settles: at any width, with a selection and the
/// brush on, the row decides once which chips drop their words and stays
/// so, frame after frame. A layout that measures itself into flipping
/// redraws the toolbar every frame, which flickers.
#[test]
fn the_toolbar_layout_settles_at_every_width() {
    let mut flips = Vec::new();
    for width in (MIN_WINDOW_SIZE[0] as u32..=1800).step_by(4) {
        let mut harness = harness_sized(egui::vec2(width as f32, MIN_WINDOW_SIZE[1]));
        {
            let app = harness.state_mut();
            app.settings.model = SettingsModel::BiRefNetLite;
            let item = push_test_item(app, 1);
            item.dimensions = (64, 64);
            crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
            // The preset chip shows the applied preset's name, wider in
            // full mode than in compact.
            item.applied_preset = "Prunr".to_string();
            let mut cells = vec![0i8; 64 * 64];
            cells[64 * 10 + 10] = prunr_core::selection::FULL;
            app.batch.select_item(0);
            app.batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(64, 64, cells), crate::gui::timeline::StepKind::Stroke);
        }
        settle(&mut harness);
        click(&mut harness, "Paint Brush");
        harness.run_steps(6);
        let fit = |h: &egui_kittest::Harness<'_, crate::gui::app::PrunrApp>| controls(h, |r| r[3] <= TOOLBAR_BOTTOM);
        let settled = fit(&harness);
        for _ in 0..6 {
            harness.run_steps(1);
            if fit(&harness) != settled {
                flips.push(width);
                break;
            }
        }
    }
    assert!(flips.is_empty(), "the toolbar keeps re-laying out at widths {flips:?}");
}


/// What is wrong with the brush strip at `width` for `model` and `tool`:
/// a control past the canvas (under the sidebar), two overlapping, or a
/// layout that does not settle.
fn strip_clash(width: f32, model: SettingsModel, tool: Tool) -> Option<String> {
    crate::gui::automation::record::turn_on();
    let mut harness = with_a_done_image(width, model);
    settle(&mut harness);
    let title = match tool {
        Tool::Paint => "Paint Brush",
        Tool::Magic => "Magic Brush",
    };
    if !names(&harness).iter().any(|n| n == title) {
        return None;
    }
    click(&mut harness, title);
    harness.run_steps(4);
    use crate::gui::views::tool_strip::{fit_id, Fit};
    let fit = |h: &Harness<'_, PrunrApp>| h.ctx.data(|d| d.get_temp::<Fit>(fit_id(tool, model)));
    let settled = fit(&harness);
    if settled.is_none() {
        return Some("the strip measured nothing".to_string());
    }
    for _ in 0..4 {
        harness.run_steps(1);
        if fit(&harness) != settled {
            return Some(format!("the strip keeps re-laying out: {settled:?} then {:?}", fit(&harness)));
        }
    }
    if !names(&harness).iter().any(|n| n == "Last") {
        return Some("no Next / Last / All saying what the sliders adjust".to_string());
    }
    let all = controls(&harness, |_| true);
    let rect_of = |name: &str| all.iter().find(|(n, _)| n == name).map(|(_, r)| *r);
    let canvas_right = rect_of("Select all")?[0];
    let reset = rect_of("Reset the brush")?;
    // The strip's last controls: one starting past the sidebar's edge is
    // left out of the scan below, so they are checked on their own.
    for (name, r) in [("Reset", Some(reset)), ("Try another", rect_of("Try another"))] {
        if let Some(r) = r.filter(|r| r[2] > canvas_right) {
            return Some(format!("its {name} ends at {} under the sidebar at {canvas_right}", r[2]));
        }
    }
    let strip_top = reset[1] - 60.0;
    let strip = controls(&harness, |r| r[1] >= TOOLBAR_BOTTOM && r[1] >= strip_top && r[3] <= strip_top + 200.0 && r[0] < canvas_right);
    if let Some((name, r)) = strip.iter().find(|(_, r)| r[2] > canvas_right) {
        return Some(format!("{name:?} ends at {} under the sidebar at {canvas_right}", r[2]));
    }
    // A label wider than the room its row was given: the row overflowed
    // even where no control's rect shows it.
    if let Some(clip) = crate::gui::automation::record::last(&harness.ctx).1.first() {
        return Some(format!("clipped: {clip:?}"));
    }
    first_overlap(&strip).map(|v| v.detail)
}

/// The brush strip stays on the canvas at every width, and its layout
/// settles: at ~1000 px its last knob ran under the sidebar.
#[test]
fn the_brush_strip_stays_on_the_canvas_at_every_width() {
    let mut clashes = Vec::new();
    for width in (MIN_WINDOW_SIZE[0] as u32..=1400).step_by(50) {
        for model in MODELS {
            for tool in [Tool::Paint, Tool::Magic] {
                if let Some(clash) = strip_clash(width as f32, model, tool) {
                    clashes.push(format!("{model:?} {tool:?} at {width}: {clash}"));
                }
            }
        }
    }
    assert!(clashes.is_empty(), "{clashes:#?}");
}

/// The batch list's rows keep their places while their thumbnails load:
/// a row's loading spinner once moved the list's layout to just under
/// itself, so the next row overlapped it and jumped down when the
/// thumbnail landed; a click aimed between the two missed.
#[test]
fn list_rows_hold_their_places_while_thumbnails_load() {
    let mut harness = super::harness::harness();
    {
        let app = harness.state_mut();
        push_test_item(app, 1);
        push_test_item(app, 2);
    }
    // The frame that asks for the thumbnails draws both spinners.
    harness.step();
    assert!(harness.state().batch.items.iter().all(|i| i.thumb.is_pending()), "the thumbnails are still on their way");
    let rows = controls(&harness, |_| true);
    let rect = |name: &str| rows.iter().find(|(n, _)| n == name).map(|(_, r)| *r).unwrap_or_else(|| panic!("{name} in the list: {rows:?}"));
    let (first, second) = (rect("test1.png"), rect("test2.png"));
    assert!(second[1] >= first[3], "the second row starts below the first: {first:?} then {second:?}");
}
