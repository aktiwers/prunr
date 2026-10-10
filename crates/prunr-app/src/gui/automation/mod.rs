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

pub struct Pending {
    pub request: Request,
    pub reply: mpsc::Sender<Response>,
}

/// A reply held until the frames that carry its input have run and
/// the UI has reacted; zero frames means right away.
struct Deferred {
    frames_left: u32,
    reply: mpsc::Sender<Response>,
    response: Response,
}

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

    pub fn take_events(&mut self) -> Vec<egui::Event> {
        self.frames.pop_front().unwrap_or_default()
    }

    fn busy(&self) -> bool {
        !self.frames.is_empty() || !self.deferred.is_empty()
    }

    fn snapshot(&self) -> Result<ControlNode, String> {
        self.tree.snapshot().ok_or_else(|| "no frame rendered yet".to_string())
    }

    /// The reply and how many frames to wait before sending it.
    fn run(&mut self, app: &mut PrunrApp, ctx: &egui::Context, request: Request) -> Result<(u32, Response), String> {
        Ok(match request {
            Request::Ping => (0, Response::ok("pong")),
            Request::Tree { all: true } => (0, Response::ok(self.snapshot()?)),
            Request::Tree { all: false } => (0, Response::ok(controls(&self.snapshot()?))),
            Request::Find { name } => (0, Response::ok(find(&self.snapshot()?, &name))),
            Request::Click { name, secondary } => {
                let root = self.snapshot()?;
                let found = find(&root, &name);
                let [target] = found.as_slice() else {
                    return Err(format!("{} controls named {name:?}", found.len()));
                };
                if target.disabled {
                    return Err(format!("{name:?} is disabled"));
                }
                let pos = target.center().ok_or_else(|| format!("{name:?} has no rectangle"))?;
                let button = if secondary { egui::PointerButton::Secondary } else { egui::PointerButton::Primary };
                let modifiers = egui::Modifiers::NONE;
                self.frames.extend([
                    vec![egui::Event::PointerMoved(pos)],
                    vec![egui::Event::PointerButton { pos, button, pressed: true, modifiers }],
                    vec![egui::Event::PointerButton { pos, button, pressed: false, modifiers }],
                ]);
                (4, Response::ok(target.clone()))
            }
            Request::Key { chord } => {
                let chord = Chord::parse(&chord).ok_or_else(|| format!("unknown chord {chord:?}"))?;
                let modifiers = chord.modifiers();
                let key = |pressed| egui::Event::Key { key: chord.key, physical_key: None, pressed, repeat: false, modifiers };
                self.frames.extend([vec![key(true)], vec![key(false)]]);
                (3, Response::ok(()))
            }
            Request::Type { text } => {
                self.frames.push_back(vec![egui::Event::Text(text)]);
                (2, Response::ok(()))
            }
            Request::Intent { action } => {
                let action = Action::from_name(&action).ok_or_else(|| {
                    format!("unknown action {action:?}; one of {}", Action::ALL.map(Action::name).join(", "))
                })?;
                app.perform(action, ctx);
                (0, Response::ok(()))
            }
            Request::Open { paths } => {
                let count = paths.len();
                for path in paths {
                    app.handle_open_path(path);
                }
                (0, Response::ok(count))
            }
            Request::State => (0, Response::ok(state::dump(app))),
            Request::Screenshot => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                (3, Response::ok(super::app::screenshot_dir()))
            }
            Request::Wait { frames } => (frames.max(1), Response::ok(())),
        })
    }
}

/// Serves queued requests and releases replies whose frames have run.
/// Called once per frame from `logic`, after the keyboard.
pub fn pump(app: &mut PrunrApp, ctx: &egui::Context) {
    let Some(mut auto) = app.automation.take() else { return };
    let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut auto.deferred).into_iter().partition(|d| d.frames_left <= 1);
    auto.deferred = waiting;
    for d in &mut auto.deferred {
        d.frames_left -= 1;
    }
    for d in ready {
        let _ = d.reply.send(d.response);
    }
    while let Ok(Pending { request, reply }) = auto.inbox.try_recv() {
        tracing::debug!(?request, "control request");
        let (frames_left, response) = auto.run(app, ctx, request).unwrap_or_else(|e| (0, Response::err(e)));
        auto.deferred.push(Deferred { frames_left, reply, response });
    }
    ctx.request_repaint_after(if auto.busy() { FRAME_INTERVAL } else { POLL_INTERVAL });
    app.automation = Some(auto);
}

/// The controls in the tree, flat, without their children.
pub fn controls(root: &ControlNode) -> Vec<ControlNode> {
    root.iter().filter(|n| n.is_control()).map(ControlNode::leaf).collect()
}

/// Controls whose name is `name`: exact first, then ignoring case, then
/// containing it, so "Open" does not also match "Open recent" unless
/// nothing is named exactly that.
pub fn find(root: &ControlNode, name: &str) -> Vec<ControlNode> {
    let named: Vec<(&ControlNode, String)> = root
        .iter()
        .filter(|n| n.is_control())
        .filter_map(|n| n.name.as_ref().map(|name| (n, name.to_lowercase())))
        .collect();
    let lower = name.to_lowercase();
    type Pass<'a> = &'a dyn Fn(&ControlNode, &str) -> bool;
    let passes: [Pass<'_>; 3] = [
        &|n, _| n.name.as_deref() == Some(name),
        &|_, l| l == lower,
        &|_, l| l.contains(&lower),
    ];
    passes
        .iter()
        .map(|pass| named.iter().filter(|(n, l)| pass(n, l)).map(|(n, _)| n.leaf()).collect::<Vec<_>>())
        .find(|found| !found.is_empty())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use accesskit::Role;
    use tree::test_node;

    fn root() -> ControlNode {
        let mut root = test_node(Role::Window, Some("root"));
        root.children = vec![
            test_node(Role::Button, Some("Open")),
            test_node(Role::Button, Some("Open recent")),
            test_node(Role::Button, Some("settings")),
            test_node(Role::Label, Some("Open")),
        ];
        root
    }

    #[test]
    fn find_prefers_exact_then_case_insensitive_then_contains() {
        let root = root();
        assert_eq!(find(&root, "Open").len(), 1, "exact match wins over 'Open recent'");
        assert_eq!(find(&root, "Settings").len(), 1, "case-insensitive fallback");
        assert_eq!(find(&root, "recent").len(), 1, "contains fallback");
        assert!(find(&root, "Save").is_empty());
    }

    #[test]
    fn controls_skips_labels_and_the_root() {
        let names: Vec<String> = controls(&root()).into_iter().filter_map(|n| n.name).collect();
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
}
