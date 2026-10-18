//! What is wrong with one frame's control tree: the checks the test
//! harness runs on every surface, kept beside the tree so the control
//! socket can run the same ones on the live window.

use serde::Serialize;

use super::tree::ControlNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// A control with no readable name: an icon-only button, a bare slider.
    Nameless,
    /// Two controls drawn over each other.
    Overlap,
    /// A name the surface must show is not there.
    Missing,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Violation {
    pub rule: Rule,
    pub detail: String,
}

impl Violation {
    fn new(rule: Rule, detail: String) -> Self {
        Self { rule, detail }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.rule, self.detail)
    }
}

/// Every control in the tree without a readable name.
pub fn nameless(root: &ControlNode) -> Vec<Violation> {
    root.iter()
        .filter(|n| n.is_control() && n.name.is_none())
        .map(|n| Violation::new(Rule::Nameless, format!("{:?} {:?} at {:?}", n.role, n.id, n.rect)))
        .collect()
}

/// The first two of `controls` (name, rect) that overlap by more than a
/// pixel each way; touching edges are not an overlap.
pub fn first_overlap(controls: &[(String, [f32; 4])]) -> Option<Violation> {
    controls.iter().enumerate().find_map(|(i, (a, ra))| {
        controls[i + 1..].iter().find_map(|(b, rb)| {
            let w = ra[2].min(rb[2]) - ra[0].max(rb[0]);
            let h = ra[3].min(rb[3]) - ra[1].max(rb[1]);
            (w > 1.0 && h > 1.0).then(|| Violation::new(Rule::Overlap, format!("{a:?} overlaps {b:?}")))
        })
    })
}

/// Each of `expect` that no control's name starts with.
pub fn missing(root: &ControlNode, expect: &[&str]) -> Vec<Violation> {
    let names: Vec<&str> = root.iter().filter(|n| n.is_control()).filter_map(|n| n.name.as_deref()).collect();
    expect
        .iter()
        .filter(|e| !names.iter().any(|n| n.starts_with(**e)))
        .map(|e| Violation::new(Rule::Missing, format!("{e:?}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::automation::tree::test_node;
    use accesskit::Role;

    #[test]
    fn nameless_finds_controls_only() {
        let mut root = test_node(Role::Window, None);
        root.children = vec![test_node(Role::Button, None), test_node(Role::Label, None), test_node(Role::Slider, Some("Size"))];
        let found = nameless(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].rule, Rule::Nameless);
    }

    #[test]
    fn overlap_needs_more_than_a_touch() {
        let rects = |b: [f32; 4]| vec![("a".to_string(), [0.0, 0.0, 10.0, 10.0]), ("b".to_string(), b)];
        assert_eq!(first_overlap(&rects([10.0, 0.0, 20.0, 10.0])), None, "touching");
        assert_eq!(first_overlap(&rects([9.5, 0.0, 20.0, 10.0])), None, "half a pixel");
        assert_eq!(first_overlap(&rects([5.0, 5.0, 20.0, 20.0])).map(|v| v.rule), Some(Rule::Overlap));
    }

    #[test]
    fn missing_matches_a_name_s_start() {
        let mut root = test_node(Role::Window, None);
        root.children = vec![test_node(Role::Button, Some("Negative prompt")), test_node(Role::Label, Some("Fast"))];
        assert!(missing(&root, &["Negative"]).is_empty());
        assert_eq!(missing(&root, &["Fast"]).len(), 1, "a label is not a control");
    }
}
