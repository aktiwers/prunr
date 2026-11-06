//! What the widgets drew that the control tree cannot tell: each
//! tooltip's text and each text drawn past its room. Recorded only while
//! a script or test reads them; otherwise every hook is one relaxed load.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use crate::gui::views::shortcuts::{self, Action};

static ON: AtomicBool = AtomicBool::new(false);

/// Start recording, for the life of the process.
pub fn turn_on() {
    ON.store(true, Ordering::Relaxed);
}

/// Whether a script or test reads the recordings.
pub fn is_on() -> bool {
    ON.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tooltip {
    pub title: String,
    pub body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shortcut: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Clip {
    pub text: String,
    pub rect: [f32; 4],
    pub room: [f32; 4],
}

/// One pass's recordings, stamped with the pass that made them.
#[derive(Clone, Default)]
struct Pass {
    nr: u64,
    tooltips: Vec<Tooltip>,
    clips: Vec<Clip>,
}

fn id() -> egui::Id {
    egui::Id::new("automation_record")
}

fn add(ctx: &egui::Context, f: impl FnOnce(&mut Pass)) {
    // Read before `data_mut`: a context accessor inside another's
    // closure deadlocks.
    let nr = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        let pass = d.get_temp_mut_or_default::<Pass>(id());
        if pass.nr != nr {
            *pass = Pass { nr, ..Pass::default() };
        }
        f(pass);
    });
}

/// A tooltip a control carries this pass, hovered or not.
pub fn tooltip(ctx: &egui::Context, title: &str, body: &str, shortcut: Option<Action>) {
    if !is_on() {
        return;
    }
    let shortcut = shortcut.map(|a| shortcuts::keys(ctx, a).to_string());
    add(ctx, |p| p.tooltips.push(Tooltip { title: title.to_owned(), body: body.to_owned(), shortcut }));
}

/// `text` drawn over `rect`, which should lie within `room` side to side.
pub fn clip(ctx: &egui::Context, text: &str, rect: egui::Rect, room: egui::Rect) {
    if !is_on() || (rect.min.x >= room.min.x - 0.5 && rect.max.x <= room.max.x + 0.5) {
        return;
    }
    let corners = |r: egui::Rect| [r.min.x, r.min.y, r.max.x, r.max.y];
    add(ctx, |p| p.clips.push(Clip { text: text.to_owned(), rect: corners(rect), room: corners(room) }));
}

/// The room a widget in `ui` has: its row's span, cut to what is shown.
pub fn room(ui: &egui::Ui) -> egui::Rect {
    ui.max_rect().intersect(ui.clip_rect())
}

/// The last finished pass's tooltips and clips. Read between passes
/// (after a test's frame, or in the next pass's `logic`).
pub fn last(ctx: &egui::Context) -> (Vec<Tooltip>, Vec<Clip>) {
    let nr = ctx.cumulative_pass_nr();
    ctx.data(|d| match d.get_temp::<Pass>(id()) {
        Some(pass) if pass.nr + 1 == nr => (pass.tooltips, pass.clips),
        _ => (Vec::new(), Vec::new()),
    })
}
