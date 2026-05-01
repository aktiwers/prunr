//! Newtype around `egui_notify::Toasts` that pre-wraps long captions at
//! word boundaries.
//!
//! `egui-notify` 0.22 hardcodes `TextWrapping::Extend` with infinite
//! width inside its `show()` — there's no caller-side knob to opt into
//! actual wrapping, so the toast widens with the longest caption and
//! drifts off-screen on error messages that interpolate paths or
//! exception text. Wrapping the caption ourselves before handing it off
//! sidesteps the upstream limitation without forking the crate.

use egui::{Context, Vec2};
use egui_notify::{Anchor, Toast};

/// Per-line character cap. Tuned for the default egui_notify font at
/// ~12 px — readable line length without going wider than ~360 px.
/// Words longer than this get hard-broken inside the wrapper.
const TOAST_LINE_CHARS: usize = 60;

#[derive(Default)]
pub(crate) struct Toasts(egui_notify::Toasts);

impl Toasts {
    pub fn new(anchor: Anchor, margin: Vec2) -> Self {
        Self(egui_notify::Toasts::default().with_anchor(anchor).with_margin(margin))
    }

    pub fn success(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.0.success(wrap(&caption.into()))
    }
    pub fn info(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.0.info(wrap(&caption.into()))
    }
    pub fn warning(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.0.warning(wrap(&caption.into()))
    }
    pub fn error(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.0.error(wrap(&caption.into()))
    }
    #[allow(dead_code)] // facade parity with egui_notify
    pub fn basic(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.0.basic(wrap(&caption.into()))
    }

    #[allow(dead_code)] // facade parity with egui_notify
    pub fn dismiss_all_toasts(&mut self) {
        self.0.dismiss_all_toasts()
    }

    pub fn show(&mut self, ctx: &Context) {
        self.0.show(ctx)
    }
}


/// Greedy word-wrap: break on whitespace into lines of at most
/// `TOAST_LINE_CHARS` chars (counted in unicode scalar values, which is
/// close enough to display width for our use cases). Words longer than
/// the limit hard-break — uncommon in practice (URLs, file paths) but
/// we'd rather wrap mid-word than overflow the screen.
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
            // Hard-break very long words so the toast still fits.
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
