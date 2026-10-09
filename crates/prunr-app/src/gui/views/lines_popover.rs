//! Labels and one-line descriptions for the Lines group: sketch modes,
//! detector scales and compositions.

use prunr_core::{ComposeMode, EdgeScale, LineMode};

pub(super) fn mode_label(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "Off",
        LineMode::EdgesOnly => "Full",
        LineMode::SubjectOutline => "Subject",
    }
}

pub(super) fn mode_description(mode: LineMode) -> &'static str {
    match mode {
        LineMode::Off => "No lines",
        LineMode::EdgesOnly => "Lines of the whole image",
        LineMode::SubjectOutline => "Lines of the subject only, on a transparent background",
    }
}

pub(super) fn scale_label(scale: EdgeScale) -> &'static str {
    match scale {
        EdgeScale::Fine => "Fine",
        EdgeScale::Balanced => "Balanced",
        EdgeScale::Bold => "Bold",
        EdgeScale::Fused => "Fused",
    }
}

pub(super) fn scale_description(scale: EdgeScale) -> &'static str {
    match scale {
        EdgeScale::Fine => "Tiny texture, crispest detail",
        EdgeScale::Balanced => "Mid-size shapes, smooth transitions",
        EdgeScale::Bold => "Big silhouettes only",
        EdgeScale::Fused => "Every scale combined; the most detailed result",
    }
}

pub(super) fn compose_description(mode: ComposeMode) -> &'static str {
    match mode {
        ComposeMode::LinesOnly => "Lines inside the subject, transparent background",
        ComposeMode::SubjectFilled => "The cut-out subject with lines on top",
        ComposeMode::Engraving => "Lines cut through the filled subject",
        ComposeMode::Ghost => "A faded subject with strong lines",
        ComposeMode::InverseMask => "Lines in the background, subject hidden",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_cover_all_variants() {
        for mode in [LineMode::Off, LineMode::EdgesOnly, LineMode::SubjectOutline] {
            assert!(!mode_label(mode).is_empty());
            assert!(!mode_description(mode).is_empty());
        }
        for scale in [EdgeScale::Fine, EdgeScale::Balanced, EdgeScale::Bold, EdgeScale::Fused] {
            assert!(!scale_label(scale).is_empty());
            assert!(!scale_description(scale).is_empty());
        }
        for mode in ComposeMode::ALL {
            assert!(!compose_description(*mode).is_empty());
        }
    }
}
