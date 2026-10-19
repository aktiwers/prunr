//! Every surface the app shows, how a user reaches it with clicks and
//! keys from a fresh window, and what it must show: one list, walked by
//! the test harness on every surface and model, so a new surface needs
//! one entry here and nowhere else.

use crate::gui::knob_catalog::Surface;

/// One user action, as the control socket can replay it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    /// Click the control by its name. A name a quick strip and its open
    /// panel share picks the panel's: the later one in the tree.
    Click(&'static str),
    /// Open a strip's full panel by its toggle, unless it is open: the
    /// panel stays open across closing the strip, so a second click would
    /// close it.
    Expand(&'static str),
    /// A chord in the settings form: "Escape", "Mod+Z".
    Key(&'static str),
}

/// What a visit opens, and so how its opening and closing are told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The window itself, nothing opened.
    Window,
    /// A set-and-go choice: a pick closes it.
    Popover,
    /// Knobs that change the image live; it stays open while they do.
    Flyout,
    /// The tool strip a brush brings up.
    Strip,
    /// A dialog, by its `PrunrApp::MODALS` name.
    Modal(&'static str),
}

/// What the window holds before the visit's steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Start {
    Empty,
    /// One processed image, selected.
    Image,
}

/// One more state of an open surface: its steps run after the visit's.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Variant {
    pub name: &'static str,
    pub steps: Vec<Step>,
    pub expect: &'static [&'static str],
}

pub struct FixedVariant {
    pub name: &'static str,
    pub steps: &'static [Step],
    pub expect: &'static [&'static str],
}

pub enum Variants {
    None,
    Fixed(&'static [FixedVariant]),
    /// Built from a list the app already has (each effect, each tab).
    Each(fn() -> Vec<Variant>),
}

pub struct Visit {
    pub id: &'static str,
    pub kind: Kind,
    /// Shown only for the models `knob_catalog::shows` gives it; `None`
    /// shows for every model and is visited once.
    pub surface: Option<Surface>,
    pub start: Start,
    pub open: &'static [Step],
    /// Names a control must start with once it is open.
    pub expect: &'static [&'static str],
    pub variants: Variants,
    pub dismiss: Option<Step>,
}

impl Visit {
    pub fn variants(&self) -> Vec<Variant> {
        match &self.variants {
            Variants::None => Vec::new(),
            Variants::Fixed(list) => list.iter().map(|v| Variant { name: v.name, steps: v.steps.to_vec(), expect: v.expect }).collect(),
            Variants::Each(build) => build(),
        }
    }
}

use Step::{Click, Expand, Key};

const ESCAPE: Option<Step> = Some(Key("Escape"));

const fn visit(id: &'static str, kind: Kind, open: &'static [Step], expect: &'static [&'static str]) -> Visit {
    let dismiss = match kind {
        Kind::Window | Kind::Strip => None,
        Kind::Popover | Kind::Flyout | Kind::Modal(_) => ESCAPE,
    };
    Visit { id, kind, surface: None, start: Start::Image, open, expect, variants: Variants::None, dismiss }
}

const fn on(surface: Surface, v: Visit) -> Visit {
    Visit { surface: Some(surface), ..v }
}

const fn with(variants: Variants, v: Visit) -> Visit {
    Visit { variants, ..v }
}

const fn panel(open: &'static [Step], expect: &'static [&'static str]) -> FixedVariant {
    FixedVariant { name: "full panel", steps: open, expect }
}

pub const VISITS: &[Visit] = &[
    Visit { start: Start::Empty, ..visit("empty window", Kind::Window, &[], &["Open", "Settings", "Model"]) },
    // A screen reader, and the control socket, pick an image by its name.
    visit("loaded window", Kind::Window, &[], &["test1.png", "Process"]),
    visit("model picker", Kind::Popover, &[Click("Model")], &["Silueta", "Get more models"]),
    visit("help menu", Kind::Popover, &[Click("Help")], &["Keyboard shortcuts", "Command-line reference", "Pipelines"]),
    visit("preset", Kind::Popover, &[Click("Preset")], &["Prunr", "Save current as…"]),
    visit("save options", Kind::Popover, &[Click("Save options")], &["PNG", "WebP"]),
    with(Variants::Each(settings_tabs), visit("settings", Kind::Modal("settings"), &[Click("Settings")], &["Close settings"])),
    visit("reset confirm", Kind::Modal("reset_confirm"), &[Click("Settings"), Click("Reset to defaults")], &["Cancel"]),
    with(
        Variants::Fixed(&[FixedVariant { name: "a card open", steps: &[Click("Details Silueta")], expect: &[] }]),
        visit("model store", Kind::Modal("model_store"), &[Click("Model Store")], &["Details Silueta"]),
    ),
    visit("command-line reference", Kind::Modal("cli_help"), &[Click("Help"), Click("Command-line reference")], &[]),
    visit("pipelines", Kind::Modal("pipeline_flow"), &[Click("Help"), Click("Pipelines")], &[]),
    visit("save preset", Kind::Modal("save_preset"), &[Click("Preset"), Click("Save current as…")], &["Save", "Cancel", "Close window"]),
    on(Surface::Mask, with(
        Variants::Fixed(&[panel(&[Expand("All mask settings")], &["Contrast"])]),
        visit("mask", Kind::Flyout, &[Click("Mask")], &["All mask settings"]),
    )),
    on(Surface::Lines, with(
        Variants::Fixed(&[
            FixedVariant { name: "whole photo panel", steps: &[Click("Whole photo"), Expand("All line settings")], expect: &["Style", "Fade with distance"] },
        ]),
        visit("sketch", Kind::Flyout, &[Click("Sketch")], &["Off", "Whole photo", "All line settings"]),
    )),
    on(Surface::Effect, with(Variants::Each(effects), visit("effect", Kind::Flyout, &[Click("Effect")], &["All effects"]))),
    on(Surface::Decoration, with(
        Variants::Fixed(&[FixedVariant {
            name: "panel with a drop shadow",
            steps: &[Click("Drop"), Expand("All outline and shadow settings")],
            expect: &["Width", "No shadow", "Soft", "Drop", "Opacity", "Blur", "Distance"],
        }]),
        visit("outline and shadow", Kind::Flyout, &[Click("Outline and shadow")], &["Outline", "All outline and shadow settings"]),
    )),
    on(Surface::Background, with(
        Variants::Fixed(&[
            FixedVariant { name: "cream", steps: &[Click("Cream"), Expand("All background settings")], expect: &["Custom color"] },
            FixedVariant { name: "transparent", steps: &[Expand("All background settings"), Click("Transparent")], expect: &[] },
            FixedVariant { name: "solid color", steps: &[Expand("All background settings"), Click("Solid color")], expect: &["Custom color"] },
            FixedVariant { name: "blurred", steps: &[Expand("All background settings"), Click("Blurred")], expect: &[] },
            FixedVariant { name: "lens blur", steps: &[Expand("All background settings"), Click("Lens blur")], expect: &[] },
            FixedVariant { name: "black and white", steps: &[Expand("All background settings"), Click("Black and white")], expect: &[] },
        ]),
        visit("background", Kind::Flyout, &[Click("Background")], &["Transparent", "Cream", "All background settings"]),
    )),
    on(Surface::Faces, with(
        Variants::Fixed(&[panel(&[Expand("More face settings")], &["Include hair"])]),
        visit("faces", Kind::Flyout, &[Click("Faces")], &["More face settings"]),
    )),
    on(Surface::Depth, with(
        Variants::Fixed(&[panel(&[Expand("All depth settings")], &["Set focus", "Aerial perspective"])]),
        visit("depth", Kind::Flyout, &[Click("Depth")], &["Off", "Point", "Tilt-shift", "Amount", "Fog"]),
    )),
    on(Surface::Upscale, visit("scale", Kind::Popover, &[Click("Scale")], &["2\u{00d7}"])),
    on(Surface::Upscale, visit("refine", Kind::Flyout, &[Click("Refine")], &["Sharpen"])),
    on(Surface::Colorize, visit("strength", Kind::Flyout, &[Click("Strength")], &["Strength"])),
    on(Surface::SdChips, visit("quality", Kind::Popover, &[Click("Quality")], &["Fast"])),
    on(Surface::SdChips, visit("prompt", Kind::Popover, &[Click("Prompt")], &["Negative prompt"])),
    on(Surface::SdChips, visit("stable diffusion advanced", Kind::Flyout, &[Click("Advanced")], &["Guidance"])),
    on(Surface::RoremChips, visit("rorem advanced", Kind::Flyout, &[Click("Advanced")], &["Steps", "Pin seed"])),
    on(Surface::Brushes, with(
        Variants::Fixed(&[panel(&[Click("More Paint Brush settings")], &["Feather"])]),
        visit("paint brush", Kind::Strip, &[Click("Paint Brush")], &["Size", "More Paint Brush settings"]),
    )),
    on(Surface::SelectionStrokes, visit("paint brush strokes", Kind::Strip, &[Click("Paint Brush")], &["Hardness", "Opacity"])),
    on(Surface::FillRegionKnobs, visit("paint brush region", Kind::Strip, &[Click("Paint Brush")], &["Expand region", "Feather"])),
    on(Surface::Brushes, with(
        Variants::Fixed(&[panel(&[Click("More Magic Brush settings")], &[])]),
        visit("magic brush", Kind::Strip, &[Click("Magic Brush")], &["More Magic Brush settings"]),
    )),
];

/// Modals no click from a fresh window reaches, each with why; their own
/// tests cover them.
#[cfg(test)]
pub const UNVISITED_MODALS: &[(&str, &str)] = &[
    ("license", "opens on downloading a model under a non-commercial license; the store offers no download while every model is installed"),
    ("runtime_prompt", "opens once at first launch, from the hardware probe"),
    ("file_dialog", "a native window, not drawn by the app"),
];

fn effects() -> Vec<Variant> {
    prunr_core::SubjectEffect::ALL
        .iter()
        .map(|e| Variant { name: e.name(), steps: vec![Expand("All effects"), Click(e.name())], expect: &["Gradient map"] })
        .collect()
}

fn settings_tabs() -> Vec<Variant> {
    crate::gui::views::settings::SettingsTab::ALL
        .iter()
        .map(|t| Variant { name: t.label(), steps: vec![Click(t.label())], expect: &[] })
        .collect()
}

/// One visit as the control socket hands it to a script: the models
/// are named as the model picker shows them.
#[derive(Debug, serde::Serialize)]
pub struct Entry {
    pub id: &'static str,
    pub kind: Kind,
    pub surface: Option<String>,
    /// The models it shows for; empty when it shows for every one.
    pub models: Vec<&'static str>,
    pub start: Start,
    pub open: &'static [Step],
    pub expect: &'static [&'static str],
    pub variants: Vec<Variant>,
    pub dismiss: Option<Step>,
}

/// A model as a script picks it (`name`, the picker's) and checks it
/// took (`id`, what the socket's `state` reports).
#[derive(Debug, serde::Serialize)]
pub struct Model {
    pub name: &'static str,
    pub id: String,
}

/// The catalog for a script: every visit, and every model.
#[derive(Debug, serde::Serialize)]
pub struct Catalog {
    pub models: Vec<Model>,
    pub visits: Vec<Entry>,
}

pub fn catalog() -> Catalog {
    use crate::gui::settings::SettingsModel;
    let name = |m: SettingsModel| crate::gui::views::model_info(m).name;
    Catalog {
        models: SettingsModel::ALL.into_iter().map(|m| Model { name: name(m), id: format!("{m:?}") }).collect(),
        visits: VISITS
            .iter()
            .map(|v| Entry {
                id: v.id,
                kind: v.kind,
                surface: v.surface.map(|s| format!("{s:?}")),
                models: v.surface.map_or_else(Vec::new, |s| {
                    SettingsModel::ALL.into_iter().filter(|&m| crate::gui::knob_catalog::shows(m, s)).map(name).collect()
                }),
                start: v.start,
                open: v.open,
                expect: v.expect,
                variants: v.variants(),
                dismiss: v.dismiss,
            })
            .collect(),
    }
}
