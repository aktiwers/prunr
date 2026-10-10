//! Driving the app from outside: the control tree read from the widgets'
//! AccessKit nodes, and the commands that act on it.
//!
//! `PRUNR_CONTROL_PORT=<port>` opens a local line-JSON socket (see
//! `server`). Clicks and keys are injected as egui input, so the app
//! reacts exactly as it does to a user; intents go through the same
//! `PrunrApp::perform` the keyboard uses; `state` is a serde dump.

pub mod server;
pub mod state;
pub mod tree;

use std::path::PathBuf;
use std::sync::mpsc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::app::PrunrApp;
use super::views::shortcuts::{Action, Chord};
use tree::ControlNode;

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    Ping,
    /// The controls as a flat list, or the whole tree with `all`.
    Tree {
        #[serde(default)]
        all: bool,
    },
    Find { name: String },
    Click {
        name: String,
        #[serde(default)]
        secondary: bool,
    },
    /// A chord in the settings form: "Mod+Shift+Z", "Escape", "A".
    Key { chord: String },
    Type { text: String },
    /// An action by its name, e.g. "Process", "ToggleQueue".
    Intent { action: String },
    Open { paths: Vec<PathBuf> },
    State,
    Screenshot,
    Wait {
        #[serde(default = "one")]
        frames: u32,
    },
}

fn one() -> u32 {
    1
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    pub fn ok(result: impl Serialize) -> Self {
        match serde_json::to_value(result) {
            Ok(result) => Self { ok: true, result: Some(result), error: None },
            Err(e) => Self::err(format!("serialize: {e}")),
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self { ok: false, result: None, error: Some(msg.into()) }
    }
}

/// A request waiting for the UI thread, with the channel its reply goes back on.
pub struct Pending {
    pub request: Request,
    pub reply: mpsc::Sender<Response>,
}

struct Deferred {
    frames_left: u32,
    reply: mpsc::Sender<Response>,
    response: Option<Response>,
}

enum Outcome {
    Now(Response),
    /// Reply once this many more frames have run, so injected input has
    /// been processed and the UI has reacted.
    After(u32, Response),
}

/// Frame pacing while a reply is pending.
const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
/// How often an idle window checks the inbox. The socket thread's
/// `request_repaint` is not enough: Wayland compositors drop wake-ups
/// from other threads while the window is idle, so the UI thread polls.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

pub struct Automation {
    tree: tree::Mirror,
    inbox: mpsc::Receiver<Pending>,
    /// Input batches, one per frame, handed to `raw_input_hook`.
    frames: std::collections::VecDeque<Vec<egui::Event>>,
    deferred: Vec<Deferred>,
}

impl Automation {
    /// Turns the tree on and listens on 127.0.0.1:`port`. `None` when
    /// the port cannot be bound.
    pub fn start(ctx: &egui::Context, port: u16) -> Option<Self> {
        let inbox = server::spawn(port, ctx.clone())?;
        let tree = tree::Mirror::default();
        ctx.enable_accesskit();
        ctx.add_plugin(tree.clone());
        Some(Self { tree, inbox, frames: Default::default(), deferred: Vec::new() })
    }

    /// The next frame's injected input.
    pub fn take_events(&mut self) -> Vec<egui::Event> {
        self.frames.pop_front().unwrap_or_default()
    }

    fn busy(&self) -> bool {
        !self.frames.is_empty() || !self.deferred.is_empty()
    }

    fn execute(&mut self, app: &mut PrunrApp, ctx: &egui::Context, request: Request) -> Outcome {
        let snapshot = |tree: &tree::Mirror| tree.snapshot().ok_or_else(|| Response::err("no frame rendered yet"));
        match request {
            Request::Ping => Outcome::Now(Response::ok("pong")),
            Request::Tree { all } => Outcome::Now(match snapshot(&self.tree) {
                Ok(root) if all => Response::ok(root),
                Ok(root) => Response::ok(controls(&root)),
                Err(e) => e,
            }),
            Request::Find { name } => Outcome::Now(match snapshot(&self.tree) {
                Ok(root) => Response::ok(find(&root, &name)),
                Err(e) => e,
            }),
            Request::Click { name, secondary } => {
                let root = match snapshot(&self.tree) {
                    Ok(root) => root,
                    Err(e) => return Outcome::Now(e),
                };
                let found = find(&root, &name);
                let [target] = found.as_slice() else {
                    return Outcome::Now(Response::err(format!("{} controls named {name:?}", found.len())));
                };
                if target.disabled {
                    return Outcome::Now(Response::err(format!("{name:?} is disabled")));
                }
                let Some(pos) = target.center() else {
                    return Outcome::Now(Response::err(format!("{name:?} has no rectangle")));
                };
                let button = if secondary { egui::PointerButton::Secondary } else { egui::PointerButton::Primary };
                let modifiers = egui::Modifiers::NONE;
                self.frames.extend([
                    vec![egui::Event::PointerMoved(pos)],
                    vec![egui::Event::PointerButton { pos, button, pressed: true, modifiers }],
                    vec![egui::Event::PointerButton { pos, button, pressed: false, modifiers }],
                ]);
                Outcome::After(4, Response::ok(target.clone()))
            }
            Request::Key { chord } => {
                let Some(chord) = Chord::parse(&chord) else {
                    return Outcome::Now(Response::err(format!("unknown chord {chord:?}")));
                };
                let modifiers = modifiers_for(chord);
                let key = |pressed| egui::Event::Key { key: chord.key, physical_key: None, pressed, repeat: false, modifiers };
                self.frames.extend([vec![key(true)], vec![key(false)]]);
                Outcome::After(3, Response::ok(()))
            }
            Request::Type { text } => {
                self.frames.push_back(vec![egui::Event::Text(text)]);
                Outcome::After(2, Response::ok(()))
            }
            Request::Intent { action } => Outcome::Now(match Action::from_name(&action) {
                Some(action) => {
                    app.perform(action, ctx);
                    Response::ok(())
                }
                None => Response::err(format!(
                    "unknown action {action:?}; one of {}",
                    Action::ALL.map(Action::name).join(", ")
                )),
            }),
            Request::Open { paths } => {
                let count = paths.len();
                for path in paths {
                    app.handle_open_path(path);
                }
                Outcome::Now(Response::ok(count))
            }
            Request::State => Outcome::Now(Response::ok(state::dump(app))),
            Request::Screenshot => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                Outcome::After(3, Response::ok(super::app::screenshot_dir()))
            }
            Request::Wait { frames } => Outcome::After(frames.max(1), Response::ok(())),
        }
    }
}

/// Serves queued requests and releases replies whose frames have run.
/// Called once per frame from `logic`, after the keyboard.
pub fn pump(app: &mut PrunrApp, ctx: &egui::Context) {
    let Some(mut auto) = app.automation.take() else { return };
    auto.deferred.retain_mut(|d| {
        if d.frames_left > 1 {
            d.frames_left -= 1;
            return true;
        }
        if let Some(response) = d.response.take() {
            let _ = d.reply.send(response);
        }
        false
    });
    while let Ok(Pending { request, reply }) = auto.inbox.try_recv() {
        tracing::debug!(?request, "control request");
        match auto.execute(app, ctx, request) {
            Outcome::Now(response) => {
                let _ = reply.send(response);
            }
            Outcome::After(frames_left, response) => {
                auto.deferred.push(Deferred { frames_left, reply, response: Some(response) });
            }
        }
    }
    ctx.request_repaint_after(if auto.busy() { FRAME_INTERVAL } else { POLL_INTERVAL });
    app.automation = Some(auto);
}

fn modifiers_for(chord: Chord) -> egui::Modifiers {
    let command = chord.mods.command();
    let mac = cfg!(target_os = "macos");
    egui::Modifiers {
        alt: false,
        ctrl: command && !mac,
        shift: chord.mods.shift(),
        mac_cmd: command && mac,
        command,
    }
}

/// The controls in the tree, flat, without their children.
pub fn controls(root: &ControlNode) -> Vec<ControlNode> {
    root.iter().filter(|n| n.is_control()).map(ControlNode::leaf).collect()
}

/// Controls whose name is `name`: exact first, then ignoring case, then
/// containing it, so "Open" does not also match "Open recent" unless
/// nothing is named exactly that.
pub fn find(root: &ControlNode, name: &str) -> Vec<ControlNode> {
    let all = controls(root);
    let lower = name.to_lowercase();
    let pick = |f: &dyn Fn(&str) -> bool| -> Vec<ControlNode> {
        all.iter().filter(|n| n.name.as_deref().is_some_and(f)).cloned().collect()
    };
    let exact = pick(&|n| n == name);
    if !exact.is_empty() {
        return exact;
    }
    let ci = pick(&|n| n.to_lowercase() == lower);
    if !ci.is_empty() {
        return ci;
    }
    pick(&|n| n.to_lowercase().contains(&lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(role: &str, name: &str) -> ControlNode {
        ControlNode {
            id: name.len() as u64,
            role: role.into(),
            name: Some(name.into()),
            value: None,
            number: None,
            toggled: None,
            disabled: false,
            rect: Some([0.0, 0.0, 10.0, 10.0]),
            children: Vec::new(),
        }
    }

    fn tree() -> ControlNode {
        let mut root = node("Window", "root");
        root.children = vec![node("Button", "Open"), node("Button", "Open recent"), node("Button", "settings"), node("Label", "Open")];
        root
    }

    #[test]
    fn find_prefers_exact_then_case_insensitive_then_contains() {
        let root = tree();
        assert_eq!(find(&root, "Open").len(), 1, "exact match wins over 'Open recent'");
        assert_eq!(find(&root, "Settings").len(), 1, "case-insensitive fallback");
        assert_eq!(find(&root, "recent").len(), 1, "contains fallback");
        assert!(find(&root, "Save").is_empty());
    }

    #[test]
    fn controls_skips_labels_and_the_root() {
        let names: Vec<String> = controls(&tree()).into_iter().filter_map(|n| n.name).collect();
        assert_eq!(names, ["Open", "Open recent", "settings"]);
    }

    #[test]
    fn requests_parse_from_tagged_json() {
        let req: Request = serde_json::from_str(r#"{"cmd":"click","name":"Settings"}"#).unwrap();
        assert!(matches!(req, Request::Click { ref name, secondary: false } if name == "Settings"));
        let req: Request = serde_json::from_str(r#"{"cmd":"wait"}"#).unwrap();
        assert!(matches!(req, Request::Wait { frames: 1 }));
        assert!(serde_json::from_str::<Request>(r#"{"cmd":"explode"}"#).is_err());
    }

    #[test]
    fn chord_modifiers_follow_the_platform() {
        let m = modifiers_for(Chord::parse("Mod+Shift+Z").unwrap());
        assert!(m.command && m.shift && !m.alt);
        assert_eq!(m.ctrl, !cfg!(target_os = "macos"));
        assert_eq!(m.mac_cmd, cfg!(target_os = "macos"));
        assert_eq!(modifiers_for(Chord::parse("A").unwrap()), egui::Modifiers::NONE);
    }
}
