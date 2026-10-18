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
    /// User text that breaks the copy rules: our plumbing words, three
    /// dots, "(s)" plurals, "4px", British spelling.
    Copy,
    /// Text drawn past the room it was given.
    Clipped,
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

/// Words that describe our plumbing, not what a knob does.
pub const ENGINEERING_WORDS: [&str; 11] = [
    "tier-1", "tier-2", "tier 1", "tier 2", "real-time", "updates live", "inference", "postprocess", "debounce", "subprocess", "knob",
];

/// British spellings; user text spells the US way.
pub const BRITISH_SPELLINGS: [&str; 5] = ["grey", "colour", "licence", "centre", "behaviour"];

/// The engineering word `text` uses, ORT included, if any.
pub fn engineering_word(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    let word = ENGINEERING_WORDS.iter().find(|w| lower.contains(*w)).copied();
    word.or_else(|| text.split(|c: char| !c.is_alphanumeric()).any(|w| w == "ORT").then_some("ORT"))
}

/// The British spelling `text` uses, if any.
pub fn british_word(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    BRITISH_SPELLINGS.iter().find(|w| lower.split(|c: char| !c.is_alphabetic()).any(|t| t.starts_with(*w))).copied()
}

/// Three dots where an ellipsis belongs, or a "(s)" plural.
pub fn typography(text: &str) -> Option<&'static str> {
    if text.contains("...") {
        Some("three dots, not an ellipsis")
    } else if text.contains("(s)") {
        Some("a (s) plural")
    } else {
        None
    }
}

/// A number run into its unit, "4px" for "4 px".
fn unit_without_space(text: &str) -> bool {
    text.as_bytes().windows(3).enumerate().any(|(i, w)| {
        w[0].is_ascii_digit() && &w[1..] == b"px" && !text.as_bytes().get(i + 3).is_some_and(u8::is_ascii_alphanumeric)
    })
}

/// Every copy rule `text` breaks, as one line each.
pub fn copy_problems(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    found.extend(engineering_word(text).map(|w| format!("says {w:?}")));
    found.extend(british_word(text).map(|w| format!("spells {w:?}")));
    found.extend(typography(text).map(str::to_string));
    if unit_without_space(text) {
        found.push("a number run into \"px\"".to_string());
    }
    found
}

/// The copy problems in each tooltip's title and body; a tooltip with no
/// title is one too.
pub fn tooltip_copy(tooltips: &[super::record::Tooltip]) -> Vec<Violation> {
    tooltips
        .iter()
        .flat_map(|t| {
            let empty = t.title.trim().is_empty().then(|| Violation::new(Rule::Copy, format!("a tooltip with no title (body {:?})", t.body)));
            let problems = [&t.title, &t.body]
                .into_iter()
                .flat_map(|text| copy_problems(text).into_iter().map(move |p| Violation::new(Rule::Copy, format!("{text:?} {p}"))));
            empty.into_iter().chain(problems).collect::<Vec<_>>()
        })
        .collect()
}

/// Each text drawn past its room.
pub fn clipped(clips: &[super::record::Clip]) -> Vec<Violation> {
    clips
        .iter()
        .map(|c| Violation::new(Rule::Clipped, format!("{:?} spans x {:.0}..{:.0}, its room {:.0}..{:.0}", c.text, c.rect[0], c.rect[2], c.room[0], c.room[2])))
        .collect()
}

/// The first two controls of the open surface whose area is `region`
/// that overlap. The surface is the node egui gives its area, with that
/// rect; the window's controls under it are not its own.
pub fn overlap_within(root: &ControlNode, region: [f32; 4]) -> Option<Violation> {
    let same = |r: &[f32; 4]| r.iter().zip(region).all(|(a, b)| (a - b).abs() <= 1.0);
    let surface = root.iter().find(|n| n.rect.as_ref().is_some_and(same))?;
    let controls: Vec<(String, [f32; 4])> = surface
        .iter()
        .filter(|n| n.is_control())
        .filter_map(|n| Some((n.name.clone().unwrap_or_default(), n.rect?)))
        .filter(|(_, r)| r[2] > r[0])
        .collect();
    first_overlap(&controls)
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
    fn copy_problems_name_each_rule_broken() {
        assert!(copy_problems("Edge thickness, 0-10 pixels.").is_empty());
        assert!(copy_problems("Save current as…").is_empty());
        assert_eq!(copy_problems("Re-runs inference on change (Tier-1)."), ["says \"tier-1\""]);
        assert_eq!(copy_problems("Uses ORT"), ["says \"ORT\""]);
        assert_eq!(copy_problems("Background colour"), ["spells \"colour\""]);
        assert_eq!(copy_problems("Loading..."), ["three dots, not an ellipsis"]);
        assert_eq!(copy_problems("3 file(s)"), ["a (s) plural"]);
        assert_eq!(copy_problems("Width 4px"), ["a number run into \"px\""]);
        assert!(copy_problems("4 px, 4pxx is not a unit").is_empty());
    }

    #[test]
    fn tooltip_copy_checks_title_and_body_and_wants_a_title() {
        use super::super::record::Tooltip;
        let tip = |title: &str, body: &str| Tooltip { title: title.into(), body: body.into(), shortcut: None };
        assert!(tooltip_copy(&[tip("Mask", "Subject mask gamma. Lower = darker subject.")]).is_empty());
        let found = tooltip_copy(&[tip("", "x"), tip("Mask", "Real-time (Tier-2).")]);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found.iter().all(|v| v.rule == Rule::Copy));
    }

    /// Only the open surface's own controls count: the window's, drawn
    /// under it, may cross them.
    #[test]
    fn overlap_within_checks_only_the_surface_s_own_controls() {
        let at = |role: Role, name: Option<&str>, rect: [f32; 4]| ControlNode { rect: Some(rect), ..test_node(role, name) };
        let popup_rect = [100.0, 0.0, 300.0, 200.0];
        let mut window = at(Role::GenericContainer, None, [0.0, 0.0, 400.0, 400.0]);
        window.children = vec![at(Role::Button, Some("under"), [110.0, 10.0, 190.0, 30.0])];
        let mut popup = at(Role::Unknown, None, popup_rect);
        popup.children = vec![at(Role::Button, Some("a"), [110.0, 10.0, 200.0, 30.0]), at(Role::Button, Some("b"), [110.0, 40.0, 200.0, 60.0])];
        let mut root = test_node(Role::Window, None);
        root.children = vec![window, popup];
        assert_eq!(overlap_within(&root, popup_rect), None, "the window's button under a is not the popup's");
        root.children[1].children[1].rect = Some([150.0, 20.0, 250.0, 50.0]);
        assert_eq!(overlap_within(&root, popup_rect).map(|v| v.rule), Some(Rule::Overlap));
        assert_eq!(overlap_within(&root, [0.0, 0.0, 1.0, 1.0]), None, "no surface with that rect");
    }

    #[test]
    fn missing_matches_a_name_s_start() {
        let mut root = test_node(Role::Window, None);
        root.children = vec![test_node(Role::Button, Some("Negative prompt")), test_node(Role::Label, Some("Fast"))];
        assert!(missing(&root, &["Negative"]).is_empty());
        assert_eq!(missing(&root, &["Fast"]).len(), 1, "a label is not a control");
    }
}
