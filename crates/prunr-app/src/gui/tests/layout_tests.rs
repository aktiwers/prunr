//! The toolbar fits the minimum window: in every model's configuration,
//! with an image loaded, a selection and a brush on, no two toolbar
//! controls overlap and none runs past the window's right edge.

use egui_kittest::Harness;

use super::fixtures::push_test_item;
use super::tree_tests::{click, names, settle, tree};
use crate::gui::app::PrunrApp;
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
    let mut harness = Harness::builder()
        .with_size(egui::vec2(width, MIN_WINDOW_SIZE[1]))
        .build_eframe(|cc| {
            egui_material_icons::initialize(&cc.egui_ctx);
            PrunrApp::new_for_test()
        });
    {
        let app = harness.state_mut();
        app.settings.model = model;
        let item = push_test_item(app, 1);
        item.dimensions = (64, 64);
        item.status = BatchStatus::Done;
        let mut cells = vec![0i8; 64 * 64];
        cells[64 * 10 + 10] = prunr_core::selection::FULL;
        app.batch.select_item(0);
        app.batch.commit_selection(1, prunr_core::selection::MaskArtifact::from_cells(64, 64, cells));
    }
    settle(&mut harness);
    // Upscale has no brush; every other mode shows the tool's chip too.
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
