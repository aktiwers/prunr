//! The toolbar fits the minimum window: in every model's configuration,
//! with an image loaded, a selection and a brush on, no two toolbar
//! controls overlap and none runs past the window's right edge.

use super::fixtures::push_test_item;
use super::harness::{click, harness_sized, names, settle, tree};
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

/// What is wrong with the toolbar at `width` for `model`, if anything.
fn toolbar_clash(width: f32, model: SettingsModel) -> Option<String> {
    let mut harness = harness_sized(egui::vec2(width, MIN_WINDOW_SIZE[1]));
    {
        let app = harness.state_mut();
        app.settings.model = model;
        let item = push_test_item(app, 1);
        item.dimensions = (64, 64);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        let mut cells = vec![0i8; 64 * 64];
        cells[64 * 10 + 10] = prunr_core::selection::FULL;
        app.batch.select_item(0);
        app.batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(64, 64, cells), crate::gui::timeline::StepKind::Stroke);
    }
    settle(&mut harness);
    // Upscale has no brush.
    if names(&harness).iter().any(|n| n == "Paint Brush") {
        click(&mut harness, "Paint Brush");
    }
    let root = tree(&harness);
    let bars: Vec<(String, [f32; 4])> = root
        .iter()
        .filter(|n| n.is_control())
        .filter_map(|n| Some((n.name.clone().unwrap_or_default(), n.rect?)))
        .filter(|(_, r)| r[3] <= TOOLBAR_BOTTOM && r[2] > r[0])
        .collect();
    for (name, r) in &bars {
        if r[2] > width {
            return Some(format!("{name:?} ends at {} past {width}", r[2]));
        }
    }
    for (i, (a, ra)) in bars.iter().enumerate() {
        for (b, rb) in &bars[i + 1..] {
            let w = ra[2].min(rb[2]) - ra[0].max(rb[0]);
            let h = ra[3].min(rb[3]) - ra[1].max(rb[1]);
            if w > 1.0 && h > 1.0 {
                return Some(format!("{a:?} overlaps {b:?}"));
            }
        }
    }
    None
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
/// brush on, the row decides once whether it is compact and stays so,
/// frame after frame. A layout that measures itself into flipping
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
        let fit = |h: &egui_kittest::Harness<'_, crate::gui::app::PrunrApp>| h.ctx.data(|d| d.get_temp::<(f32, f32)>(egui::Id::new("toolbar_fit")));
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
/// a control past the canvas (under the sidebar) or two overlapping.
fn strip_clash(width: f32, model: SettingsModel, tool: &str) -> Option<String> {
    let mut harness = harness_sized(egui::vec2(width, MIN_WINDOW_SIZE[1]));
    {
        let app = harness.state_mut();
        app.settings.model = model;
        let item = push_test_item(app, 1);
        item.dimensions = (64, 64);
        crate::gui::tests::fixtures::put_in(item, BatchStatus::Done);
        app.batch.select_item(0);
    }
    settle(&mut harness);
    if !names(&harness).iter().any(|n| n == tool) {
        return None;
    }
    click(&mut harness, tool);
    harness.run_steps(4);
    let fit_id = egui::Id::new(("tool_strip_fit", tool));
    let fit = |h: &egui_kittest::Harness<'_, crate::gui::app::PrunrApp>| h.ctx.data(|d| d.get_temp::<f32>(fit_id));
    let settled = fit(&harness);
    for _ in 0..4 {
        harness.run_steps(1);
        if fit(&harness) != settled {
            return Some(format!("the strip keeps re-laying out: {settled:?} then {:?}", fit(&harness)));
        }
    }
    let root = tree(&harness);
    let rect_of = |name: &str| root.iter().find(|n| n.name.as_deref() == Some(name)).and_then(|n| n.rect);
    let canvas_right = rect_of("Select All")?[0];
    let more = format!("More {tool} settings");
    let strip_top = rect_of(&more)?[1] - 60.0;
    let strip: Vec<(String, [f32; 4])> = root
        .iter()
        .filter(|n| n.is_control())
        .filter_map(|n| Some((n.name.clone().unwrap_or_default(), n.rect?)))
        .filter(|(_, r)| r[1] >= TOOLBAR_BOTTOM && r[1] >= strip_top && r[3] <= strip_top + 200.0 && r[0] < canvas_right && r[2] > r[0])
        .collect();
    for (name, r) in &strip {
        if r[2] > canvas_right {
            return Some(format!("{name:?} ends at {} under the sidebar at {canvas_right}", r[2]));
        }
    }
    for (i, (a, ra)) in strip.iter().enumerate() {
        for (b, rb) in &strip[i + 1..] {
            let w = ra[2].min(rb[2]) - ra[0].max(rb[0]);
            let h = ra[3].min(rb[3]) - ra[1].max(rb[1]);
            if w > 1.0 && h > 1.0 {
                return Some(format!("{a:?} overlaps {b:?}"));
            }
        }
    }
    None
}

/// The brush strip stays on the canvas at every width, and its layout
/// settles: at ~1000 px its last knob ran under the sidebar.
#[test]
fn the_brush_strip_stays_on_the_canvas_at_every_width() {
    let mut clashes = Vec::new();
    for width in (MIN_WINDOW_SIZE[0] as u32..=1400).step_by(50) {
        for model in MODELS {
            for tool in ["Paint Brush", "Magic Brush"] {
                if let Some(clash) = strip_clash(width as f32, model, tool) {
                    clashes.push(format!("{model:?} {tool} at {width}: {clash}"));
                }
            }
        }
    }
    assert!(clashes.is_empty(), "{clashes:#?}");
}
