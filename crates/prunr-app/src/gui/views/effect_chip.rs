//! The Effect chip: a quick strip of favourite looks and the main knob,
//! and a panel with every look as a tile and the picked one's knobs.

use egui::{Color32, Ui};
use egui_material_icons::icons::ICON_FORMAT_PAINT;
use prunr_core::SubjectEffect;

use super::{chip, variant_index};
use super::effects::paint_rounded_mesh;

const PANEL_WIDTH: f32 = 460.0;
const TILE: egui::Vec2 = egui::vec2(58.0, 42.0);
const QUICK_TILE: egui::Vec2 = egui::vec2(34.0, 34.0);
const TILE_COLUMNS: usize = 6;
/// The looks on the strip, by their place in `SubjectEffect::ALL`: None,
/// Desaturate, Sepia, Duotone and Posterize.
const QUICK: [usize; 5] = [0, 1, 3, 11, 5];

/// Returns true when the effect or one of its knobs changed.
pub(super) fn render(ui: &mut Ui, style: &mut SubjectEffect) -> bool {
    let picked = variant_index(SubjectEffect::ALL, style);
    let group = chip::GroupChip {
        id_salt: "fill_style",
        icon: ICON_FORMAT_PAINT.codepoint,
        label: "Effect",
        summary: if matches!(style, SubjectEffect::None) { "" } else { style.name() },
        tooltip: "Restyle the subject's colors, or the whole photo with No model.",
        tuned: !matches!(style, SubjectEffect::None),
        width: PANEL_WIDTH,
        live: true,
    };
    let header = chip::PanelHeader { note: style.name(), reset: "Back to own colors" };
    let mut changed = false;
    chip::strip_chip(ui, group, "All effects", header, |ui, part| match part {
        chip::Part::Quick => {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let extra = (!QUICK.contains(&picked)).then_some(picked);
                for i in QUICK.into_iter().chain(extra) {
                    changed |= look_tile(ui, style, i, QUICK_TILE, false);
                }
            });
            changed |= params(ui, style, true);
        }
        chip::Part::Full { reset } => {
            if reset {
                *style = SubjectEffect::None;
                changed = true;
            }
            egui::Grid::new("effect_tiles").spacing([6.0, 12.0]).show(ui, |ui| {
                for (i, _) in SubjectEffect::ALL.iter().enumerate() {
                    changed |= look_tile(ui, style, i, TILE, true);
                    if (i + 1) % TILE_COLUMNS == 0 {
                        ui.end_row();
                    }
                }
            });
            if has_knobs(style) {
                chip::panel_section(ui, style.name(), None);
                changed |= params(ui, style, false);
            }
        }
    });
    changed
}

/// One look's tile; picking it sets the look with its default knobs.
fn look_tile(ui: &mut Ui, style: &mut SubjectEffect, index: usize, size: egui::Vec2, caption: bool) -> bool {
    let option = &SubjectEffect::ALL[index];
    let selected = variant_index(SubjectEffect::ALL, style) == index;
    // The picked look shows its own colours; the rest show their defaults.
    let shown = if selected { *style } else { *option };
    let resp = chip::swatch_tile(ui, option.name(), selected, size, caption, |painter, rect| {
        let look = Look::of(&shown);
        paint_rounded_mesh(painter, rect, 10.0, |u, v| look.at(u, v));
    });
    let resp = if caption { resp } else { resp.on_hover_text(option.name()) };
    if resp.clicked() && !selected {
        *style = *option;
        return true;
    }
    false
}

fn has_knobs(style: &SubjectEffect) -> bool {
    !matches!(style, SubjectEffect::None | SubjectEffect::Desaturate | SubjectEffect::Invert | SubjectEffect::Sepia)
}

/// The picked look's knobs. On the strip only the main one, and none for
/// a look set by colours.
fn params(ui: &mut Ui, style: &mut SubjectEffect, quick: bool) -> bool {
    match style {
        SubjectEffect::None | SubjectEffect::Desaturate | SubjectEffect::Invert | SubjectEffect::Sepia => false,
        SubjectEffect::Threshold { level } => chip::slider_row(ui, "Level", level, 0..=255).changed,
        SubjectEffect::Posterize { levels } => chip::slider_row(ui, "Levels", levels, 2..=8).changed,
        SubjectEffect::Solarize { pivot } => chip::slider_row(ui, "Pivot", pivot, 0..=255).changed,
        SubjectEffect::HueShift { degrees } => chip::slider_row(ui, "Degrees", degrees, -180..=180).changed,
        SubjectEffect::Saturate { percent } => chip::slider_row(ui, "Percent", percent, 0..=300).changed,
        SubjectEffect::ColorSplash { keep_hue, tolerance } => {
            let mut changed = chip::slider_row(ui, "Hue (degrees)", keep_hue, 0..=359).changed;
            if !quick {
                changed |= chip::slider_row(ui, "Tolerance (\u{00b0})", tolerance, 0..=180).changed;
                chip::kept_hues(ui, *keep_hue, *tolerance);
            }
            changed
        }
        SubjectEffect::Pixelate { block_size } => chip::slider_row(ui, "Block size (px)", block_size, 2..=64).changed,
        SubjectEffect::Halftone { dot_spacing } => chip::slider_row(ui, "Dot spacing (px)", dot_spacing, 2..=32).changed,
        _ if quick => false,
        SubjectEffect::Duotone { dark: shadows, light: highlights }
        | SubjectEffect::CrossProcess { shadow: shadows, highlight: highlights } => {
            chip::color_row(ui, "Shadows", shadows) | chip::color_row(ui, "Highlights", highlights)
        }
        SubjectEffect::GradientMap { stops } => ["Shadow", "Dark mid", "Light mid", "Highlight"]
            .into_iter()
            .zip(stops.iter_mut())
            .fold(false, |changed, (name, stop)| chip::color_row(ui, name, stop) | changed),
    }
}

/// How a look's tile is drawn: a gradient that suggests it, in the
/// look's own colours where it has them.
enum Look {
    /// A diagonal ramp through two to four colours.
    Ramp([Color32; 4], usize),
    /// Two colours split at `at` across (`diagonal`) or down the tile.
    Split(Color32, Color32, f32, bool),
    /// Vertical stripes, `count` pairs.
    Stripes(Color32, Color32, f32),
    /// A checkerboard of `count` columns.
    Checker(Color32, Color32, f32),
    /// Dots on a grid of `count` columns.
    Dots(Color32, Color32, f32),
}

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

fn from(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

const WARM: Color32 = rgb(0xc98a5a);
const COOL: Color32 = rgb(0x5a7bb0);

impl Look {
    fn of(style: &SubjectEffect) -> Self {
        let two = |a, b| Look::Ramp([a, b, b, b], 2);
        match style {
            SubjectEffect::None => two(WARM, COOL),
            SubjectEffect::Desaturate => two(rgb(0xbbbbbb), rgb(0x444444)),
            SubjectEffect::Invert => two(rgb(0x3a75a5), rgb(0xa5843a)),
            SubjectEffect::Duotone { dark, light } => two(from(*light), from(*dark)),
            SubjectEffect::Sepia => two(rgb(0xd8b98a), rgb(0x6b4a2a)),
            SubjectEffect::Threshold { .. } => Look::Split(Color32::WHITE, Color32::BLACK, 0.5, false),
            SubjectEffect::Posterize { .. } => Look::Stripes(WARM, COOL, 3.0),
            SubjectEffect::Solarize { .. } => Look::Ramp([WARM, rgb(0x2a4a7a), rgb(0xc9a05a), rgb(0xc9a05a)], 3),
            SubjectEffect::HueShift { .. } => two(rgb(0x5ac98a), rgb(0xb05a7b)),
            SubjectEffect::Saturate { .. } => two(rgb(0xff8a3a), rgb(0x2a7bff)),
            SubjectEffect::ColorSplash { .. } => Look::Split(rgb(0x888888), rgb(0xd0453a), 0.55, true),
            SubjectEffect::Pixelate { .. } => Look::Checker(WARM, COOL, 4.0),
            SubjectEffect::CrossProcess { shadow, highlight } => two(from(*highlight), from(*shadow)),
            SubjectEffect::Halftone { .. } => Look::Dots(rgb(0x222222), rgb(0xe8e2d6), 5.0),
            SubjectEffect::GradientMap { stops } => Look::Ramp(stops.map(from), 4),
        }
    }

    fn at(&self, u: f32, v: f32) -> Color32 {
        match *self {
            Look::Ramp(stops, n) => {
                let t = (u + v) / 2.0 * (n - 1) as f32;
                let i = (t.floor() as usize).min(n - 2);
                stops[i].lerp_to_gamma(stops[i + 1], t - i as f32)
            }
            Look::Split(a, b, at, diagonal) => {
                let t = if diagonal { (u + v) / 2.0 } else { u };
                if t < at { a } else { b }
            }
            Look::Stripes(a, b, count) => if (u * count * 2.0).floor() as i32 % 2 == 0 { a } else { b },
            Look::Checker(a, b, count) => {
                let rows = count * TILE.y / TILE.x;
                if ((u * count).floor() as i32 + (v * rows).floor() as i32) % 2 == 0 { a } else { b }
            }
            Look::Dots(dot, paper, count) => {
                let rows = count * TILE.y / TILE.x;
                let du = (u * count).fract() - 0.5;
                let dv = (v * rows).fract() - 0.5;
                if du * du + dv * dv < 0.09 { dot } else { paper }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_look_has_a_tile() {
        for style in SubjectEffect::ALL {
            let look = Look::of(style);
            for (u, v) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
                let _ = look.at(u, v);
            }
        }
    }

    #[test]
    fn a_ramp_runs_from_its_first_colour_to_its_last() {
        let look = Look::Ramp([Color32::RED, Color32::GREEN, Color32::BLUE, Color32::BLUE], 3);
        assert_eq!(look.at(0.0, 0.0), Color32::RED);
        assert_eq!(look.at(1.0, 1.0), Color32::BLUE);
    }

    #[test]
    fn the_strip_holds_the_favourite_looks() {
        let names = QUICK.map(|i| SubjectEffect::ALL[i].name());
        assert_eq!(names, ["None", "Desaturate", "Sepia", "Duotone", "Posterize"]);
    }
}
