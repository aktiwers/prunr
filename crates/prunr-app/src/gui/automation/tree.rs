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
    pub fn from_node(node: &Node<'_>) -> Self {
        Self {
            // The consumer packs its tree index below the AccessKit id.
            id: (u128::from(node.id()) >> 64) as u64,
            role: node.role(),
            name: node.label().as_deref().and_then(readable_name),
            value: node.value(),
            number: node.numeric_value(),
            toggled: node.toggled().map(|t| t == accesskit::Toggled::True),
            disabled: node.is_disabled(),
            rect: node.bounding_box().map(|r| [r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32]),
            children: name_slider_value_boxes(node.children().map(|c| Self::from_node(&c)).collect()),
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
pub struct Mirror(Arc<Mutex<Option<Tree>>>);

struct Silent;

impl accesskit_consumer::TreeChangeHandler for Silent {
    fn node_added(&mut self, _: &Node<'_>) {}
    fn node_updated(&mut self, _: &Node<'_>, _: &Node<'_>) {}
    fn focus_moved(&mut self, _: Option<&Node<'_>>, _: Option<&Node<'_>>) {}
    fn node_removed(&mut self, _: &Node<'_>) {}
}

impl Mirror {
    fn lock(&self) -> MutexGuard<'_, Option<Tree>> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn apply(&self, update: accesskit::TreeUpdate) {
        let mut tree = self.lock();
        match tree.as_mut() {
            Some(tree) => tree.update_and_process_changes(update, &mut Silent),
            None => *tree = Some(Tree::new(update, true)),
        }
    }

    pub fn snapshot(&self) -> Option<ControlNode> {
        self.lock().as_ref().map(|t| ControlNode::from_node(&t.state().root()))
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
            self.apply(update);
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
