//! The Faces chip: Restore faces on or off and its strength on the strip;
//! the panel picks the model, which faces, and whether hair goes too.
//! Without a face model installed the strip offers the Model Store.

use egui::Ui;
use egui_material_icons::icons::ICON_FACE;

use crate::gui::hardware_cache::HardwareInstallCache;
use crate::gui::item_settings::{FaceModel, FaceSettings};

use super::{chip, get_model};
use super::{fmt, hint};

const PANEL_WIDTH: f32 = 360.0;

/// What the chip changed this frame.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FacesChange {
    /// On/off or the model: the faces have to be found again.
    pub search: bool,
    /// Strength, which faces or hair: only the blend.
    pub blend: bool,
    /// "Get a face model" was clicked.
    pub open_store: bool,
    /// The Outline switch moved (an app setting, saved).
    pub outline: bool,
}

/// What the search of the shown photo came to, when one ran.
pub(crate) type FoundCount = Option<FaceSearch>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaceSearch {
    Found(usize),
    Failed,
    /// Stopped by Cancel or Esc.
    Stopped,
    /// The photo is made from an image whose faces were already restored.
    RestoredEarlier,
}

const ON_OFF: [chip::Choice<bool>; 2] = [
    chip::Choice { value: false, name: "Off", description: "", enabled: true },
    chip::Choice { value: true, name: "On", description: "", enabled: true },
];

const WHICH: [chip::Choice<bool>; 2] = [
    chip::Choice { value: false, name: "All", description: "Every face in the photo", enabled: true },
    chip::Choice { value: true, name: "Largest", description: "Only the biggest face: a portrait with people behind", enabled: true },
];

/// The Faces chip's strip and panel, which the canvas outlines the
/// found faces under.
pub(super) const ID_SALT: &str = "faces";

pub(crate) fn render(ui: &mut Ui, s: &mut FaceSettings, outlines: &mut bool, installed: HardwareInstallCache, found: FoundCount) -> FacesChange {
    let mut change = FacesChange::default();
    let any_installed = FaceModel::ALL.iter().any(|&m| installed.has_face_model(m));
    let summary: std::borrow::Cow<'static, str> = if !any_installed {
        get_model::BADGE.into()
    } else if s.on {
        fmt::percent(f32::from(s.strength_pct) / 100.0).into()
    } else {
        "Off".into()
    };
    let group = chip::GroupChip {
        id_salt: ID_SALT,
        icon: ICON_FACE.codepoint,
        label: "Faces",
        summary: &summary,
        tooltip: "Redraw soft, small or compressed faces.",
        tuned: s.on,
        width: PANEL_WIDTH,
        live: true,
    };
    let header = chip::PanelHeader { note: "", reset: "Reset" };
    chip::strip_chip(ui, group, "More face settings", header, |ui, part| match part {
        chip::Part::Quick => {
            if !any_installed {
                ui.label(egui::RichText::new(needs_text()).color(crate::gui::theme::TEXT_SECONDARY));
                change.open_store |= chip::button(ui, chip::ButtonKind::Secondary, "Get a face model").clicked();
                return;
            }
            change.search |= chip::choice_buttons(ui, &ON_OFF, &mut s.on);
            change.blend |= strength_row(ui, s);
            change.outline |= chip::inline_switch(ui, "Outline", outlines).commit;
        }
        chip::Part::Full { reset } => {
            if reset {
                let model = s.model;
                *s = FaceSettings { model, ..FaceSettings::default() };
                change.search = true;
            }
            hint(ui, &found_text(s.on, found));
            let models = FaceModel::ALL.map(|m| chip::Choice {
                value: m,
                name: model_name(m),
                description: model_description(m),
                enabled: installed.has_face_model(m),
            });
            change.search |= chip::choice_row(ui, "Model", &models, &mut s.model);
            for m in FaceModel::ALL {
                if !installed.has_face_model(m) {
                    change.open_store |= get_model::button(ui, m.id());
                }
            }
            change.blend |= strength_row(ui, s);
            hint(ui, "Up to 50% adds detail and keeps the photo's own shading; above, the redrawn shading too.");
            change.blend |= chip::choice_row(ui, "Faces", &WHICH, &mut s.largest_only);
            change.blend |= chip::toggle_row(ui, "Include hair", &mut s.hair).commit;
        }
    });
    change
}

/// "Faces needs a face model, 295 to 340 MB.", built once.
fn needs_text() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(|| get_model::needs_one_of("Faces", "a face model", &FaceModel::ALL.map(FaceModel::id)))
}

fn strength_row(ui: &mut Ui, s: &mut FaceSettings) -> bool {
    chip::slider_row_pct(ui, "Strength", &mut s.strength_pct, fmt::percent).changed
}

fn model_name(m: FaceModel) -> &'static str {
    match m {
        FaceModel::Gfpgan => "GFPGAN",
        FaceModel::RestoreFormerPp => "RestoreFormer++",
    }
}

fn model_description(m: FaceModel) -> &'static str {
    match m {
        FaceModel::Gfpgan => "The crispest",
        FaceModel::RestoreFormerPp => "Closest to the person's own features",
    }
}

fn found_text(on: bool, found: FoundCount) -> std::borrow::Cow<'static, str> {
    match (on, found) {
        (false, _) => "These apply while Faces is on.".into(),
        (true, None) => "Faces are found once the image is processed.".into(),
        (true, Some(FaceSearch::RestoredEarlier)) => "Faces were restored in an earlier step; they are not restored twice.".into(),
        (true, Some(FaceSearch::Failed)) => "The face search failed. Turn Faces off and on to try again.".into(),
        (true, Some(FaceSearch::Stopped)) => "Stopped. Turn Faces off and on to search again.".into(),
        (true, Some(FaceSearch::Found(0))) => "No faces found.".into(),
        (true, Some(FaceSearch::Found(1))) => "1 face found.".into(),
        (true, Some(FaceSearch::Found(n))) => format!("{n} faces found.").into(),
    }
}
