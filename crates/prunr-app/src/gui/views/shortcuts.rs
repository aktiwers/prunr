//! The shipped keyboard bindings. One table feeds the key handler, the F1
//! overlay, the Settings › Hotkeys tab, button tooltips, the selection
//! action-bar hints and the empty-canvas tips, so none of them can drift
//! from the keys that actually work.

use std::sync::OnceLock;

use egui::{Event, InputState, Key};

use super::kv_row;
use crate::gui::theme;

const MOD_NAME: &str = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Open,
    Process,
    Save,
    Copy,
    Cut,
    Delete,
    Invert,
    Cancel,
    Undo,
    Redo,
    Shortcuts,
    CliHelp,
    PipelineFlow,
    Screenshot,
    FitToWindow,
    ActualSize,
    Settings,
    BeforeAfter,
    PrevImage,
    NextImage,
    ToggleQueue,
    ToggleAdjustments,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mods {
    None,
    Command,
    Shift,
    CommandShift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chord {
    pub mods: Mods,
    pub key: Key,
}

const fn chord(mods: Mods, key: Key) -> Chord {
    Chord { mods, key }
}

/// How a row's chords reach the app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// Key press, including OS key repeat (held navigation and undo keys).
    Press,
    /// Once per physical press — toggles must not flicker while held.
    FreshPress,
    /// egui turns the chord into its own event (`Event::Copy`), which the
    /// app intercepts in `raw_input_hook`; the row exists for display.
    Event,
}

pub struct Shortcut {
    pub action: Action,
    pub chords: &'static [Chord],
    /// Verb phrase for the shortcut lists and tips ("Open images").
    pub label: &'static str,
    pub delivery: Delivery,
}

impl Action {
    /// Discriminant order; `keys` indexes by it.
    pub const ALL: [Action; 22] = [
        Action::Open, Action::Process, Action::Save, Action::Copy, Action::Cut,
        Action::Delete, Action::Invert, Action::Cancel, Action::Undo, Action::Redo,
        Action::Shortcuts, Action::CliHelp, Action::PipelineFlow, Action::Screenshot,
        Action::FitToWindow, Action::ActualSize, Action::Settings, Action::BeforeAfter,
        Action::PrevImage, Action::NextImage, Action::ToggleQueue, Action::ToggleAdjustments,
    ];
}

use Delivery::{Event as ByEvent, FreshPress, Press};
use Mods::{Command, CommandShift, None as NoMods, Shift};

pub const SHORTCUTS: &[Shortcut] = &[
    Shortcut { action: Action::Open, chords: &[chord(Command, Key::O)], label: "Open images", delivery: Press },
    Shortcut { action: Action::Process, chords: &[chord(Command, Key::R)], label: "Process the image", delivery: Press },
    Shortcut { action: Action::Save, chords: &[chord(Command, Key::S)], label: "Save the result", delivery: Press },
    Shortcut { action: Action::Copy, chords: &[chord(Command, Key::C)], label: "Copy the result or selection", delivery: ByEvent },
    Shortcut { action: Action::Cut, chords: &[chord(Command, Key::X)], label: "Cut the selection", delivery: Press },
    Shortcut { action: Action::Delete, chords: &[chord(NoMods, Key::Delete)], label: "Delete the selection", delivery: Press },
    Shortcut { action: Action::Invert, chords: &[chord(NoMods, Key::Enter)], label: "Invert the selection", delivery: Press },
    Shortcut { action: Action::Cancel, chords: &[chord(NoMods, Key::Escape)], label: "Cancel, clear or close", delivery: Press },
    Shortcut { action: Action::Undo, chords: &[chord(Command, Key::Z)], label: "Undo", delivery: Press },
    Shortcut { action: Action::Redo, chords: &[chord(CommandShift, Key::Z), chord(Command, Key::Y)], label: "Redo", delivery: Press },
    Shortcut { action: Action::BeforeAfter, chords: &[chord(NoMods, Key::B)], label: "Compare with the original", delivery: FreshPress },
    Shortcut { action: Action::PrevImage, chords: &[chord(NoMods, Key::ArrowLeft), chord(NoMods, Key::A)], label: "Previous image", delivery: Press },
    Shortcut { action: Action::NextImage, chords: &[chord(NoMods, Key::ArrowRight), chord(NoMods, Key::D)], label: "Next image", delivery: Press },
    Shortcut { action: Action::FitToWindow, chords: &[chord(Command, Key::Num0)], label: "Fit to window", delivery: Press },
    Shortcut { action: Action::ActualSize, chords: &[chord(Command, Key::Num1)], label: "Actual size", delivery: Press },
    Shortcut { action: Action::ToggleQueue, chords: &[chord(NoMods, Key::H), chord(NoMods, Key::Tab)], label: "Show or hide the queue", delivery: FreshPress },
    Shortcut { action: Action::ToggleAdjustments, chords: &[chord(Shift, Key::H)], label: "Show or hide the adjustments", delivery: FreshPress },
    Shortcut { action: Action::Settings, chords: &[chord(Command, Key::Space)], label: "Open settings", delivery: FreshPress },
    Shortcut { action: Action::Shortcuts, chords: &[chord(NoMods, Key::F1)], label: "Show keyboard shortcuts", delivery: FreshPress },
    Shortcut { action: Action::CliHelp, chords: &[chord(NoMods, Key::F2)], label: "Show the command-line reference", delivery: FreshPress },
    Shortcut { action: Action::PipelineFlow, chords: &[chord(NoMods, Key::F3)], label: "Show the pipelines", delivery: FreshPress },
    Shortcut { action: Action::Screenshot, chords: &[chord(Shift, Key::F12)], label: "Save a window screenshot", delivery: FreshPress },
];

/// Pointer gestures listed with the shortcuts. Not key chords, so they
/// live outside the table but render in the same grid.
const GESTURES: &[(&str, &str)] = &[
    ("Drag", "Pan the image (right-drag while a brush is on)"),
    ("Scroll", "Zoom in or out"),
];

/// Which shortcuts fired this frame. One bit per `Action`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Pressed(u32);

impl Pressed {
    fn set(&mut self, action: Action) {
        self.0 |= 1 << action as u32;
    }

    pub fn is(self, action: Action) -> bool {
        self.0 & (1 << action as u32) != 0
    }
}

/// Collect the shortcuts pressed this frame.
pub fn pressed(ctx: &egui::Context) -> Pressed {
    let mut out = Pressed::default();
    // Read before `ctx.input`: egui's context lock is not re-entrant.
    let text_focused = ctx.memory(|m| m.focused().is_some());
    ctx.input(|i| {
        if i.events.is_empty() {
            return;
        }
        for s in SHORTCUTS.iter().filter(|s| s.delivery != ByEvent) {
            let fresh = s.delivery == FreshPress;
            if s.chords.iter().any(|c| chord_pressed(i, *c, fresh, text_focused)) {
                out.set(s.action);
            }
        }
    });
    out
}

/// Bare keys type into a focused text field, so they are suppressed while
/// one has focus. Modifier chords, function keys and Escape always fire.
fn blocked_by_text_focus(c: Chord) -> bool {
    let bare = matches!(c.mods, Mods::None | Mods::Shift);
    bare && !matches!(c.key, Key::Escape | Key::F1 | Key::F2 | Key::F3 | Key::F12)
}

fn chord_pressed(i: &InputState, c: Chord, fresh: bool, text_focused: bool) -> bool {
    if text_focused && blocked_by_text_focus(c) {
        return false;
    }
    let want_command = matches!(c.mods, Mods::Command | Mods::CommandShift);
    let want_shift = matches!(c.mods, Mods::Shift | Mods::CommandShift);
    if i.modifiers.command != want_command || i.modifiers.shift != want_shift {
        return false;
    }
    if fresh {
        i.events.iter().any(|e| matches!(
            e,
            Event::Key { key, pressed: true, repeat: false, .. } if *key == c.key
        ))
    } else {
        i.key_pressed(c.key)
    }
}

fn key_name(key: Key) -> &'static str {
    match key {
        Key::Escape => "Esc",
        Key::ArrowLeft => "\u{2190}",
        Key::ArrowRight => "\u{2192}",
        Key::Enter => "Enter",
        Key::Delete => "Del",
        Key::Tab => "Tab",
        Key::Space => "Space",
        Key::Num0 => "0",
        Key::Num1 => "1",
        other => other.name(),
    }
}

fn chord_display(c: Chord) -> String {
    let mut s = String::new();
    if matches!(c.mods, Mods::Command | Mods::CommandShift) {
        s.push_str(MOD_NAME);
        s.push('+');
    }
    if matches!(c.mods, Mods::Shift | Mods::CommandShift) {
        s.push_str("Shift+");
    }
    s.push_str(key_name(c.key));
    s
}

fn row(action: Action) -> &'static Shortcut {
    // Every action has exactly one row — pinned by `every_action_has_one_row`.
    SHORTCUTS.iter().find(|s| s.action == action).expect("shortcut row")
}

/// Platform-resolved key text for `action` ("Ctrl+O", "\u{2190} / A"),
/// built once and shared so tooltips stay allocation-free per frame.
pub fn keys(action: Action) -> &'static str {
    static DISPLAY: OnceLock<[String; Action::ALL.len()]> = OnceLock::new();
    let table = DISPLAY.get_or_init(|| {
        Action::ALL.map(|a| {
            row(a).chords.iter().map(|c| chord_display(*c)).collect::<Vec<_>>().join(" / ")
        })
    });
    &table[action as usize]
}

/// Label for `action`, as in the shortcut lists.
pub fn label(action: Action) -> &'static str {
    row(action).label
}

/// Returns true if the modal should close.
pub fn render(ctx: &egui::Context) -> bool {
    theme::standard_modal_window(
        ctx, "shortcuts", "Keyboard shortcuts",
        [theme::SHORTCUT_OVERLAY_WIDTH, theme::SHORTCUT_OVERLAY_HEIGHT],
        |ui| {
            ui.vertical(|ui| {
                ui.add_space(theme::SPACE_SM);
                render_shortcut_grid(ui);
            });
        },
    )
}

/// Two-column key/action grid. Used by the F1 overlay and the Settings
/// Hotkeys tab.
pub fn render_shortcut_grid(ui: &mut egui::Ui) {
    egui::Grid::new("shortcuts_grid")
        .num_columns(2)
        .spacing([theme::SPACE_LG, theme::SPACE_SM])
        .show(ui, |ui| {
            for s in SHORTCUTS {
                kv_row(ui, keys(s.action), s.label, theme::TEXT_PRIMARY);
            }
            for (gesture, what) in GESTURES {
                kv_row(ui, gesture, what, theme::TEXT_PRIMARY);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Modifiers, RawInput};

    #[test]
    fn every_action_has_one_row() {
        assert_eq!(Action::ALL.len(), SHORTCUTS.len());
        for (i, a) in Action::ALL.into_iter().enumerate() {
            assert_eq!(a as usize, i, "ALL must follow discriminant order: {a:?}");
            assert_eq!(SHORTCUTS.iter().filter(|s| s.action == a).count(), 1, "{a:?}");
            assert!(!keys(a).is_empty(), "{a:?}");
            assert!(!label(a).is_empty(), "{a:?}");
        }
    }

    #[test]
    fn no_chord_is_bound_twice() {
        let mut seen: Vec<(Chord, Action)> = Vec::new();
        for s in SHORTCUTS {
            for c in s.chords {
                if let Some((_, other)) = seen.iter().find(|(seen_c, _)| seen_c == c) {
                    panic!("{c:?} bound to both {other:?} and {:?}", s.action);
                }
                seen.push((*c, s.action));
            }
        }
    }

    #[test]
    fn display_strings_are_platform_resolved() {
        let expect_mod = if cfg!(target_os = "macos") { "Cmd+O" } else { "Ctrl+O" };
        assert_eq!(keys(Action::Open), expect_mod);
        assert_eq!(keys(Action::Redo), format!("{MOD_NAME}+Shift+Z / {MOD_NAME}+Y"));
        assert_eq!(keys(Action::PrevImage), "\u{2190} / A");
        assert_eq!(keys(Action::Cancel), "Esc");
        assert_eq!(keys(Action::ToggleAdjustments), "Shift+H");
    }

    fn press(key: Key, modifiers: Modifiers) -> RawInput {
        RawInput {
            modifiers,
            events: vec![Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            ..Default::default()
        }
    }

    /// Run `frames` as consecutive passes; return what the last one pressed.
    /// egui sets `repeat` itself when a key was already down last frame.
    fn pressed_after(frames: Vec<RawInput>) -> Pressed {
        let ctx = egui::Context::default();
        let mut last = Pressed::default();
        for raw in frames {
            ctx.begin_pass(raw);
            last = pressed(&ctx);
            let _ = ctx.end_pass();
        }
        last
    }

    fn pressed_for(raw: RawInput) -> Pressed {
        pressed_after(vec![raw])
    }

    #[test]
    fn command_chords_route_to_their_action() {
        let p = pressed_for(press(Key::O, Modifiers::COMMAND));
        assert!(p.is(Action::Open));
        assert!(!p.is(Action::Process));
    }

    #[test]
    fn shift_decides_undo_versus_redo() {
        let undo = pressed_for(press(Key::Z, Modifiers::COMMAND));
        assert!(undo.is(Action::Undo) && !undo.is(Action::Redo));
        let redo = pressed_for(press(Key::Z, Modifiers::COMMAND | Modifiers::SHIFT));
        assert!(redo.is(Action::Redo) && !redo.is(Action::Undo));
    }

    #[test]
    fn bare_keys_ignore_held_modifiers() {
        let p = pressed_for(press(Key::B, Modifiers::COMMAND));
        assert!(!p.is(Action::BeforeAfter));
        let p = pressed_for(press(Key::Tab, Modifiers::SHIFT));
        assert!(!p.is(Action::ToggleQueue));
        let p = pressed_for(press(Key::H, Modifiers::SHIFT));
        assert!(p.is(Action::ToggleAdjustments) && !p.is(Action::ToggleQueue));
    }

    #[test]
    fn toggles_ignore_key_repeat_but_navigation_does_not() {
        let held = |key| vec![press(key, Modifiers::NONE), press(key, Modifiers::NONE)];
        let p = pressed_after(held(Key::F1));
        assert!(!p.is(Action::Shortcuts));
        let p = pressed_after(held(Key::ArrowRight));
        assert!(p.is(Action::NextImage));
    }

    #[test]
    fn copy_is_delivered_by_the_copy_event_not_the_table() {
        assert_eq!(row(Action::Copy).delivery, Delivery::Event);
        let p = pressed_for(press(Key::C, Modifiers::COMMAND));
        assert!(!p.is(Action::Copy));
    }

    #[test]
    fn text_focus_blocks_bare_keys_only() {
        assert!(blocked_by_text_focus(chord(NoMods, Key::B)));
        assert!(blocked_by_text_focus(chord(Shift, Key::H)));
        assert!(!blocked_by_text_focus(chord(Command, Key::O)));
        assert!(!blocked_by_text_focus(chord(NoMods, Key::Escape)));
        assert!(!blocked_by_text_focus(chord(NoMods, Key::F1)));
    }
}
