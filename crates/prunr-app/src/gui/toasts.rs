//! `egui_notify::Toasts` with captions pre-wrapped at word boundaries:
//! egui-notify 0.22 hardcodes infinite-width `TextWrapping::Extend`, so a
//! caption with a path or error text would drift off-screen.

use egui::{Context, Vec2};
use egui_notify::{Anchor, Toast};

/// Readable line length at the default ~12 px font, under ~360 px wide.
const TOAST_LINE_CHARS: usize = 60;

#[derive(Default)]
pub(crate) struct Toasts(
    egui_notify::Toasts,
    /// The latest captions, kept while a script reads them.
    std::collections::VecDeque<String>,
    #[cfg(test)] Vec<(&'static str, String)>,
);

/// How many captions `Toasts` keeps for a script; more never show at once.
const KEPT_CAPTIONS: usize = 8;

impl Toasts {
    pub fn new(anchor: Anchor, margin: Vec2) -> Self {
        Self(
            egui_notify::Toasts::default().with_anchor(anchor).with_margin(margin),
            std::collections::VecDeque::new(),
            #[cfg(test)] Vec::new(),
        )
    }

    pub fn success(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.keep(&caption);
        #[cfg(test)] self.2.push(("success", caption.clone()));
        self.0.success(wrap(&caption))
    }
    pub fn info(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.keep(&caption);
        #[cfg(test)] self.2.push(("info", caption.clone()));
        self.0.info(wrap(&caption))
    }
    pub fn warning(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.keep(&caption);
        #[cfg(test)] self.2.push(("warning", caption.clone()));
        self.0.warning(wrap(&caption))
    }
    pub fn error(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.keep(&caption);
        #[cfg(test)] self.2.push(("error", caption.clone()));
        self.0.error(wrap(&caption))
    }
    /// A success toast for `Ok`, an error toast for `Err`.
    pub fn outcome(&mut self, outcome: Result<String, String>) -> &mut Toast {
        match outcome {
            Ok(caption) => self.success(caption),
            Err(caption) => self.error(caption),
        }
    }

    #[cfg(test)]
    pub fn len_for_test(&self) -> usize {
        self.0.len()
    }

    /// The last toast shown: its level ("success", "error", …) and caption.
    #[cfg(test)]
    pub fn last_for_test(&self) -> Option<(&'static str, &str)> {
        self.2.last().map(|(level, caption)| (*level, caption.as_str()))
    }

    pub fn show(&mut self, ctx: &Context) {
        self.0.show(ctx)
    }

    fn keep(&mut self, caption: &str) {
        if !crate::gui::automation::record::is_on() {
            return;
        }
        if self.1.len() == KEPT_CAPTIONS {
            self.1.pop_front();
        }
        self.1.push_back(caption.to_owned());
    }

    /// The captions on screen, oldest first: the latest as many as are
    /// still shown (they leave in the order they came). Empty unless a
    /// script turned recording on.
    pub fn shown(&self) -> Vec<String> {
        self.1.iter().rev().take(self.0.len()).rev().cloned().collect()
    }
}


/// Greedy word-wrap counted in chars (close enough to display width);
/// a word over the limit hard-breaks rather than overflow the screen.
fn wrap(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + input.len() / TOAST_LINE_CHARS);
    for (i, line) in input.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        wrap_one_line(line, &mut out);
    }
    out
}

fn wrap_one_line(line: &str, out: &mut String) {
    let mut col = 0usize;
    let mut first = true;
    for word in line.split_whitespace() {
        let wlen = word.chars().count();
        if wlen >= TOAST_LINE_CHARS {
            if !first { out.push('\n'); col = 0; }
            for ch in word.chars() {
                if col >= TOAST_LINE_CHARS {
                    out.push('\n');
                    col = 0;
                }
                out.push(ch);
                col += 1;
            }
            first = false;
            continue;
        }
        let need = if first { wlen } else { wlen + 1 };
        if col + need > TOAST_LINE_CHARS {
            out.push('\n');
            col = 0;
            first = true;
        }
        if !first { out.push(' '); col += 1; }
        out.push_str(word);
        col += wlen;
        first = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_messages_pass_through_unchanged() {
        assert_eq!(wrap("Settings saved"), "Settings saved");
        assert_eq!(wrap("OK"), "OK");
    }

    #[test]
    fn wraps_at_word_boundary_around_60_chars() {
        let s = "The brush correction grid was discarded because the model resolution changed";
        let w = wrap(s);
        for line in w.lines() {
            assert!(line.chars().count() <= TOAST_LINE_CHARS,
                "line too long: {line:?}");
        }
        // Original words are all preserved.
        let normalised_in: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
        let normalised_out: String = w.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(normalised_in, normalised_out);
    }

    #[test]
    fn long_word_hard_breaks() {
        let s = format!("path={}", "x".repeat(120));
        let w = wrap(&s);
        for line in w.lines() {
            assert!(line.chars().count() <= TOAST_LINE_CHARS, "line too long: {line:?}");
        }
    }

    #[test]
    fn preserves_user_inserted_newlines() {
        let s = "Header line\nSecond line that is not particularly long";
        let w = wrap(s);
        assert_eq!(w.lines().count(), 2);
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(wrap(""), "");
    }
}
