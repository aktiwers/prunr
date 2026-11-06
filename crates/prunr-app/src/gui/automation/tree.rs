//! The control tree: egui's AccessKit nodes, read back as plain data.
//! Widgets name themselves (button text, slider text, `chip::named`), so
//! the tree is derived from the real UI every frame. A control a test can
//! find by name, the control socket can find too.

use std::sync::{Arc, Mutex, MutexGuard};

use accesskit::Role;
use accesskit_consumer::{Node, Tree};
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ControlNode {
    pub id: u64,
    #[serde(serialize_with = "role_name")]
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toggled: Option<bool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub disabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ControlNode>,
}

fn role_name<S: serde::Serializer>(role: &Role, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{role:?}"))
}

impl ControlNode {
    /// `node` and its children, their rectangles in points: AccessKit's
    /// are in pixels, `pixels_per_point` to the point.
    pub fn from_node(node: &Node<'_>, pixels_per_point: f32) -> Self {
        let points = |v: f64| v as f32 / pixels_per_point;
        Self {
            // The consumer packs its tree index below the AccessKit id.
            id: (u128::from(node.id()) >> 64) as u64,
            role: node.role(),
            name: node.label().as_deref().and_then(readable_name),
            value: node.value(),
            number: node.numeric_value(),
            toggled: node.toggled().map(|t| t == accesskit::Toggled::True),
            disabled: node.is_disabled(),
            rect: node.bounding_box().map(|r| [points(r.x0), points(r.y0), points(r.x1), points(r.y1)]),
            children: name_slider_value_boxes(node.children().map(|c| Self::from_node(&c, pixels_per_point)).collect()),
        }
    }

    /// The roles egui gives to buttons, toggles, sliders, pickers and text fields.
    pub fn is_control(&self) -> bool {
        matches!(
            self.role,
            Role::Button
                | Role::CheckBox
                | Role::Switch
                | Role::RadioButton
                | Role::Slider
                | Role::SpinButton
                | Role::ComboBox
                | Role::TextInput
                | Role::MultilineTextInput
                | Role::Tab
                | Role::MenuItem
                | Role::Link
                | Role::ListBoxOption
        )
    }

    /// Every node under and including this one, parents first.
    pub fn iter(&self) -> impl Iterator<Item = &ControlNode> {
        let mut stack = vec![self];
        std::iter::from_fn(move || {
            let node = stack.pop()?;
            stack.extend(node.children.iter().rev());
            Some(node)
        })
    }

    pub fn leaf(&self) -> Self {
        Self { children: Vec::new(), ..self.clone() }
    }

    pub fn center(&self) -> Option<egui::Pos2> {
        self.rect.map(|[x0, y0, x1, y1]| egui::pos2((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    }
}

/// egui lays a slider's value box out as the sibling after the slider,
/// with no label of its own and no way to name it from outside; it
/// takes the slider's name plus "value".
fn name_slider_value_boxes(mut children: Vec<ControlNode>) -> Vec<ControlNode> {
    for i in 1..children.len() {
        let (before, after) = children.split_at_mut(i);
        let (slider, value_box) = (&before[i - 1], &mut after[0]);
        if slider.role == Role::Slider && value_box.role == Role::SpinButton && value_box.name.is_none() {
            value_box.name = slider.name.as_ref().map(|n| format!("{n} value"));
        }
    }
    children
}

/// The full tree, kept current from the update egui puts in each frame's
/// output: the same source the test harness reads. An egui plugin, so it
/// sees the output before eframe hands it to the platform.
#[derive(Clone, Default)]
pub struct Mirror(Arc<Mutex<Option<Mirrored>>>);

struct Mirrored {
    tree: Tree,
    /// The scale of the frame the tree came from.
    pixels_per_point: f32,
}

struct Silent;

impl accesskit_consumer::TreeChangeHandler for Silent {
    fn node_added(&mut self, _: &Node<'_>) {}
    fn node_updated(&mut self, _: &Node<'_>, _: &Node<'_>) {}
    fn focus_moved(&mut self, _: Option<&Node<'_>>, _: Option<&Node<'_>>) {}
    fn node_removed(&mut self, _: &Node<'_>) {}
}

impl Mirror {
    fn lock(&self) -> MutexGuard<'_, Option<Mirrored>> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn apply(&self, update: accesskit::TreeUpdate, pixels_per_point: f32) {
        let mut mirrored = self.lock();
        match mirrored.as_mut() {
            Some(m) => {
                m.tree.update_and_process_changes(update, &mut Silent);
                m.pixels_per_point = pixels_per_point;
            }
            None => *mirrored = Some(Mirrored { tree: Tree::new(update, true), pixels_per_point }),
        }
    }

    pub fn snapshot(&self) -> Option<ControlNode> {
        self.lock().as_ref().map(|m| ControlNode::from_node(&m.tree.state().root(), m.pixels_per_point))
    }
}

impl egui::plugin::Plugin for Mirror {
    fn debug_name(&self) -> &'static str {
        "prunr-control-tree"
    }

    fn output_hook(&mut self, output: &mut egui::FullOutput) {
        // Cloned, not taken: a screen reader attached to the window
        // still gets the update.
        if let Some(update) = output.platform_output.accesskit_update.clone() {
            self.apply(update, output.pixels_per_point);
        }
    }
}

/// A label with its icon glyphs (private-use codepoints) and the padding
/// around them removed, or `None` when nothing readable remains.
pub fn readable_name(label: &str) -> Option<String> {
    let text: String = label.chars().filter(|c| !('\u{E000}'..='\u{F8FF}').contains(c)).collect();
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
pub(crate) fn test_node(role: Role, name: Option<&str>) -> ControlNode {
    ControlNode { role, name: name.map(str::to_string), rect: Some([0.0, 0.0, 10.0, 10.0]), ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Controls are found and clicked in points, the unit of every pointer
    /// event; on a 2x screen (Retina) the tree's pixel rectangles put each
    /// click at twice its place, off the window.
    #[test]
    fn a_control_is_placed_in_points_on_a_2x_screen() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(2.0);
        ctx.enable_accesskit();
        let mirror = Mirror::default();
        ctx.add_plugin(mirror.clone());
        let mut placed = egui::Rect::NOTHING;
        for _ in 0..2 {
            let _ = ctx.run_ui(Default::default(), |ui| {
                placed = ui.button("Close without saving").rect;
            });
        }
        let root = mirror.snapshot().expect("a frame ran");
        let button = root.iter().find(|n| n.name.as_deref() == Some("Close without saving")).expect("the button");
        let center = button.center().expect("a rectangle");
        assert!((center - placed.center()).length() < 0.5, "{center:?} is not {:?}", placed.center());
    }

    #[test]
    fn readable_name_strips_icon_glyphs_and_padding() {
        assert_eq!(readable_name("\u{e2c8}  Open"), Some("Open".to_string()));
        assert_eq!(readable_name("Open"), Some("Open".to_string()));
        assert_eq!(readable_name("\u{e5cd}"), None);
        assert_eq!(readable_name("   "), None);
    }

    #[test]
    fn slider_value_box_takes_the_slider_name() {
        let named = name_slider_value_boxes(vec![
            test_node(Role::Slider, Some("Size")),
            test_node(Role::SpinButton, None),
            test_node(Role::SpinButton, None),
        ]);
        assert_eq!(named[1].name.as_deref(), Some("Size value"));
        assert_eq!(named[2].name, None, "only the box right after the slider");
    }

    #[test]
    fn iter_walks_parents_first() {
        let leaf = |id| ControlNode { id, ..Default::default() };
        let mut root = leaf(1);
        let mut mid = leaf(2);
        mid.children.push(leaf(3));
        root.children.push(mid);
        root.children.push(leaf(4));
        let ids: Vec<u64> = root.iter().map(|n| n.id).collect();
        assert_eq!(ids, [1, 2, 3, 4]);
    }
}
