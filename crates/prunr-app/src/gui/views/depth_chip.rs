//! The Depth chip (No model): lens blur, fog and aerial perspective on
//! the photo, read from its depth map. Without the depth model the strip
//! offers the Model Store.

use egui::Ui;
use egui_material_icons::icons::ICON_LANDSCAPE;
use prunr_core::depth::{DepthLook, LensBlurMode};

use super::{chip, get_model};
use super::{fmt, hint};

const PANEL_WIDTH: f32 = 380.0;
pub(crate) const DEPTH_MODEL: prunr_models::ModelId = prunr_core::depth::MODEL.model_id();

/// "Depth needs Depth Anything V2 Small · 94 MB.", built once.
fn needs_text() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(|| get_model::needs("Depth", DEPTH_MODEL))
}

/// What the chip changed this frame.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthChange {
    pub look: chip::ChipChange,
    /// "Set focus" was clicked: the next click on the image focuses.
    pub pick_focus: bool,
    /// "Get the depth model" was clicked.
    pub open_store: bool,
}

impl DepthChange {
    fn merge(&mut self, c: chip::ChipChange) {
        self.look.changed |= c.changed;
        self.look.commit |= c.commit;
    }

    fn picked(&mut self, changed: bool) {
        self.merge(chip::ChipChange { changed, commit: changed });
    }
}

const BLUR_MODES: [chip::Choice<LensBlurMode>; 3] = [
    chip::Choice { value: LensBlurMode::Off, name: "Off", description: "", enabled: true },
    chip::Choice { value: LensBlurMode::Point, name: "Point", description: "Sharp at the focus's depth; softer the farther in front or behind", enabled: true },
    chip::Choice { value: LensBlurMode::TiltShift, name: "Tilt-shift", description: "Sharp in a band across the photo; softer above and below, like a miniature", enabled: true },
];

/// `picking` is true while the next click on the image sets the focus.
pub(crate) fn render(ui: &mut Ui, look: &mut DepthLook, installed: bool, picking: bool) -> DepthChange {
    let mut change = DepthChange::default();
    let summary: std::borrow::Cow<'static, str> = if installed { summary(look).into() } else { get_model::BADGE.into() };
    let group = chip::GroupChip {
        id_salt: "depth",
        icon: ICON_LANDSCAPE.codepoint,
        label: "Depth",
        summary: &summary,
        tooltip: "Blur, haze and grade the photo by how far each part is.",
        tuned: look.is_active(),
        width: PANEL_WIDTH,
        live: true,
    };
    chip::strip_chip(ui, group, "All depth settings", chip::PanelHeader::RESET, |ui, part| match part {
        chip::Part::Quick => {
            if !installed {
                ui.label(egui::RichText::new(needs_text()).color(crate::gui::theme::TEXT_SECONDARY));
                change.open_store |= get_model::button(ui, DEPTH_MODEL);
                return;
            }
            change.picked(chip::choice_buttons(ui, &BLUR_MODES, &mut look.blur));
            change.merge(chip::slider_row_pct(ui, "Amount", &mut look.blur_pct, fmt::percent));
            change.merge(chip::slider_row_pct(ui, "Fog", &mut look.fog_pct, fmt::percent));
        }
        chip::Part::Full { reset } => {
            if reset {
                *look = DepthLook::default();
                change.picked(true);
            }
            if !installed {
                change.open_store |= get_model::button(ui, DEPTH_MODEL);
            }
            change.picked(chip::choice_row(ui, "Lens blur", &BLUR_MODES, &mut look.blur));
            change.merge(chip::slider_row_pct(ui, "Amount", &mut look.blur_pct, fmt::percent));
            hint(ui, "Lens blur. Blurs what is far from the focus.");
            ui.horizontal(|ui| {
                let text = if picking { "Click the image\u{2026}" } else { "Set focus" };
                let resp = chip::named(chip::button(ui, chip::ButtonKind::Secondary, text), "Set focus");
                change.pick_focus |= chip::tooltip(resp, "Set focus", "Click the image where it should be sharp.", None).clicked();
                if look.focus.is_some() && chip::button(ui, chip::ButtonKind::Secondary, "Auto focus").clicked() {
                    look.focus = None;
                    change.picked(true);
                }
            });
            hint(ui, focus_text(look));
            change.merge(chip::slider_row_pct(ui, "Fog", &mut look.fog_pct, fmt::percent));
            hint(ui, "Haze that thickens with distance.");
            change.merge(chip::slider_row_pct(ui, "Aerial perspective", &mut look.aerial_pct, fmt::percent));
            hint(ui, "Far things turn cooler and paler, like distant hills.");
        }
    });
    change
}


/// The chip face: what the look does, or Off.
fn summary(look: &DepthLook) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if look.blur != LensBlurMode::Off && look.blur_pct > 0 {
        parts.push(if look.blur == LensBlurMode::TiltShift { "Tilt-shift" } else { "Lens blur" });
    }
    if look.fog_pct > 0 {
        parts.push("Fog");
    }
    if look.aerial_pct > 0 {
        parts.push("Aerial");
    }
    if parts.is_empty() { "Off".to_string() } else { parts.join(" + ") }
}

fn focus_text(look: &DepthLook) -> &'static str {
    match (look.blur, look.focus.is_some()) {
        (LensBlurMode::TiltShift, false) => "The band runs across the middle until you set the focus.",
        (LensBlurMode::TiltShift, true) => "The band runs through the point you clicked.",
        (_, false) => "Focused on the nearest part of the photo until you set the focus.",
        (_, true) => "Focused at the depth of the point you clicked.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_face_names_what_is_on() {
        assert_eq!(summary(&DepthLook::default()), "Off");
        let look = DepthLook { blur: LensBlurMode::Point, blur_pct: 40, fog_pct: 10, aerial_pct: 5, ..DepthLook::default() };
        assert_eq!(summary(&look), "Lens blur + Fog + Aerial");
        let idle_blur = DepthLook { blur: LensBlurMode::TiltShift, ..DepthLook::default() };
        assert_eq!(summary(&idle_blur), "Off", "a blur at no amount does nothing");
    }
}
