//! The keyboard bindings. One shipped table plus the user's overrides
//! (`Settings.hotkeys`) resolve into one `Bindings`, which feeds the key
//! handler, the F1 overlay, the Settings › Hotkeys tab, button tooltips,
//! the selection action-bar hints and the empty-canvas tips, so none of
//! them can drift from the keys that actually work.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use egui::{Event, InputState, Key};

use super::kv_row;
use crate::gui::theme;

const MOD_NAME: &str = if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" };

/// Magic Brush subtract modifier. Alt on macOS; Alt or Ctrl elsewhere,
/// because Linux window managers commonly bind Alt+click to window moves
/// and the app never receives the click.
pub(crate) fn is_subtract_modifier(m: &egui::Modifiers) -> bool {
    m.alt || (!cfg!(target_os = "macos") && m.ctrl)
}

pub(crate) const MAGIC_BRUSH_TIP: &str = if cfg!(target_os = "macos") {
    "Click or stroke to select an object; Shift adds, Alt subtracts."
} else {
    "Click or stroke to select an object; Shift adds, Alt or Ctrl subtracts."
};

pub(crate) const SUBTRACT_HINT: &str = if cfg!(target_os = "macos") { "Alt = subtract" } else { "Alt/Ctrl = subtract" };

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

impl Chord {
    /// The platform-neutral settings form: "Mod+Shift+Z", "Delete".
    pub fn to_setting(self) -> String {
        let mut s = String::new();
        if matches!(self.mods, Mods::Command | Mods::CommandShift) {
            s.push_str("Mod+");
        }
        if matches!(self.mods, Mods::Shift | Mods::CommandShift) {
            s.push_str("Shift+");
        }
        s.push_str(self.key.name());
        s
    }

    pub fn parse(s: &str) -> Option<Chord> {
        let mut command = false;
        let mut shift = false;
        let mut key = None;
        for token in s.split('+') {
            match token.trim() {
                "Mod" => command = true,
                "Shift" => shift = true,
                // Exactly one key, and it must be one egui knows.
                name => key = match (key, Key::from_name(name)) {
                    (None, Some(k)) => Some(k),
                    _ => return None,
                },
            }
        }
        Some(Chord { mods: Mods::from_flags(command, shift), key: key? })
    }

    /// The chord a fresh key press carries this frame, for the capture
    /// field. `None` while nothing was pressed and for Alt chords, which
    /// Linux window managers tend to keep.
    pub fn from_input(i: &InputState) -> Option<Chord> {
        i.events.iter().find_map(|e| match e {
            Event::Key { key, pressed: true, repeat: false, modifiers, .. } if !modifiers.alt => {
                Some(Chord { mods: Mods::from_flags(modifiers.command, modifiers.shift), key: *key })
            }
            _ => None,
        })
    }
}

impl Mods {
    fn from_flags(command: bool, shift: bool) -> Self {
        match (command, shift) {
            (false, false) => Mods::None,
            (true, false) => Mods::Command,
            (false, true) => Mods::Shift,
            (true, true) => Mods::CommandShift,
        }
    }
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
    /// The settings key for an action's override.
    pub fn name(self) -> String {
        format!("{self:?}")
    }

    pub fn from_name(name: &str) -> Option<Action> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }

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

/// Every action's chords: the shipped table with the user's overrides
/// applied, plus the display text each tooltip shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Bindings {
    chords: Vec<Vec<Chord>>,
    display: Vec<Arc<str>>,
}

impl Bindings {
    pub fn shipped() -> Self {
        let chords = Action::ALL.iter().map(|a| row(*a).chords.to_vec()).collect();
        let mut b = Self { chords, display: Vec::new() };
        b.display = Action::ALL.iter().map(|a| b.render_display(*a)).collect();
        b
    }

    /// The shipped bindings with `overrides` applied; an override that
    /// parses to nothing leaves the shipped chords, an empty list unbinds.
    pub fn from_overrides(overrides: &BTreeMap<String, Vec<String>>) -> Self {
        let mut b = Self::shipped();
        for (name, list) in overrides {
            let Some(action) = Action::from_name(name) else { continue };
            let parsed: Vec<Chord> = list.iter().filter_map(|s| Chord::parse(s)).collect();
            if parsed.is_empty() && !list.is_empty() {
                continue;
            }
            b.set(action, parsed);
        }
        b
    }

    pub fn chords(&self, action: Action) -> &[Chord] {
        &self.chords[action as usize]
    }

    pub fn is_default(&self, action: Action) -> bool {
        self.chords[action as usize] == row(action).chords
    }

    /// The other action `c` is bound to, if any.
    pub fn conflict(&self, c: Chord, except: Action) -> Option<Action> {
        Action::ALL.into_iter()
            .find(|a| *a != except && self.chords[*a as usize].contains(&c))
    }

    pub fn set(&mut self, action: Action, chords: Vec<Chord>) {
        self.chords[action as usize] = chords;
        self.display[action as usize] = self.render_display(action);
    }

    /// Only the actions that differ from the shipped table.
    pub fn to_overrides(&self) -> BTreeMap<String, Vec<String>> {
        Action::ALL.into_iter()
            .filter(|a| !self.is_default(*a))
            .map(|a| (a.name(), self.chords(a).iter().map(|c| c.to_setting()).collect()))
            .collect()
    }

    fn render_display(&self, action: Action) -> Arc<str> {
        let text = self.chords[action as usize].iter()
            .map(|c| chord_display(*c)).collect::<Vec<_>>().join(" / ");
        Arc::from(if text.is_empty() { "Unbound".to_string() } else { text })
    }
}

fn bindings_slot() -> &'static RwLock<Arc<Bindings>> {
    static SLOT: OnceLock<RwLock<Arc<Bindings>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(Arc::new(Bindings::shipped())))
}

/// Make `bindings` the ones the key handler and every label use.
pub fn install(bindings: Bindings) {
    *bindings_slot().write().unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(bindings);
}

pub fn current() -> Arc<Bindings> {
    Arc::clone(&bindings_slot().read().unwrap_or_else(std::sync::PoisonError::into_inner))
}

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
    pressed_with(ctx, &current())
}

fn pressed_with(ctx: &egui::Context, bindings: &Bindings) -> Pressed {
    let mut out = Pressed::default();
    // Read before `ctx.input`: egui's context lock is not re-entrant.
    let text_focused = ctx.memory(|m| m.focused().is_some());
    ctx.input(|i| {
        if i.events.is_empty() {
            return;
        }
        for s in SHORTCUTS.iter().filter(|s| s.delivery != ByEvent) {
            let fresh = s.delivery == FreshPress;
            if bindings.chords(s.action).iter().any(|c| chord_pressed(i, *c, fresh, text_focused)) {
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

pub fn chord_display(c: Chord) -> String {
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

/// Platform-resolved key text for `action` ("Ctrl+O", "\u{2190} / A")
/// under the current bindings, shared so tooltips allocate nothing.
pub fn keys(action: Action) -> Arc<str> {
    Arc::clone(&current().display[action as usize])
}

/// Whether the user can rebind `action`: Copy arrives as egui's own
/// event and Escape is the universal cancel.
pub fn rebindable(action: Action) -> bool {
    row(action).delivery != ByEvent && action != Action::Cancel
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
                kv_row(ui, &keys(s.action), s.label, theme::TEXT_PRIMARY);
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
        assert_eq!(&*keys(Action::Open), expect_mod);
        assert_eq!(&*keys(Action::Redo), format!("{MOD_NAME}+Shift+Z / {MOD_NAME}+Y"));
        assert_eq!(&*keys(Action::PrevImage), "\u{2190} / A");
        assert_eq!(&*keys(Action::Cancel), "Esc");
        assert_eq!(&*keys(Action::ToggleAdjustments), "Shift+H");
    }

    #[test]
    fn settings_form_round_trips_every_shipped_chord() {
        for s in SHORTCUTS {
            for c in s.chords {
                let text = c.to_setting();
                assert_eq!(Chord::parse(&text), Some(*c), "{text}");
            }
        }
        assert_eq!(Chord::parse("Mod+Shift+Z"), Some(chord(CommandShift, Key::Z)));
        assert_eq!(Chord::parse("Mod+"), None);
        assert_eq!(Chord::parse("Hyper+Q"), None);
    }

    #[test]
    fn overrides_replace_the_shipped_chord_and_store_only_changes() {
        let mut overrides = BTreeMap::new();
        overrides.insert("Undo".to_string(), vec!["Mod+U".to_string()]);
        overrides.insert("Nonsense".to_string(), vec!["Mod+Q".to_string()]);
        overrides.insert("Save".to_string(), vec!["Hyper+S".to_string()]);
        overrides.insert("Screenshot".to_string(), Vec::new());
        let b = Bindings::from_overrides(&overrides);
        assert_eq!(b.chords(Action::Undo), &[chord(Command, Key::U)]);
        assert!(b.is_default(Action::Save), "an unparsable override keeps the shipped chord");
        assert!(b.chords(Action::Screenshot).is_empty(), "an empty list unbinds");
        assert_eq!(&*b.display[Action::Screenshot as usize], "Unbound");
        let back = b.to_overrides();
        assert_eq!(back.len(), 2);
        assert_eq!(back["Undo"], vec!["Mod+U".to_string()]);
        assert!(back["Screenshot"].is_empty());

        let ctx = egui::Context::default();
        ctx.begin_pass(press(Key::U, Modifiers::COMMAND));
        let p = pressed_with(&ctx, &b);
        assert!(p.is(Action::Undo));
        let _ = ctx.end_pass();
        ctx.begin_pass(press(Key::Z, Modifiers::COMMAND));
        let p = pressed_with(&ctx, &b);
        assert!(!p.is(Action::Undo), "the shipped chord no longer fires");
        let _ = ctx.end_pass();
    }

    #[test]
    fn conflict_names_the_other_action() {
        let b = Bindings::shipped();
        assert_eq!(b.conflict(chord(Command, Key::Z), Action::Redo), Some(Action::Undo));
        assert_eq!(b.conflict(chord(Command, Key::Z), Action::Undo), None);
        assert_eq!(b.conflict(chord(Command, Key::U), Action::Undo), None);
        assert!(rebindable(Action::Undo) && !rebindable(Action::Copy) && !rebindable(Action::Cancel));
    }

    #[test]
    fn capture_reads_one_fresh_press_and_skips_alt_chords() {
        let ctx = egui::Context::default();
        ctx.begin_pass(press(Key::K, Modifiers::COMMAND | Modifiers::SHIFT));
        let c = ctx.input(Chord::from_input);
        assert_eq!(c, Some(chord(CommandShift, Key::K)));
        let _ = ctx.end_pass();
        ctx.begin_pass(press(Key::K, Modifiers::ALT));
        assert_eq!(ctx.input(Chord::from_input), None);
        let _ = ctx.end_pass();
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
