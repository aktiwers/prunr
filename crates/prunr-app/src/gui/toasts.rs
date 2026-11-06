//! `egui_notify::Toasts` with captions pre-wrapped at word boundaries:
//! egui-notify 0.22 hardcodes infinite-width `TextWrapping::Extend`, so a
//! caption with a path or error text would drift off-screen.

use std::time::{Duration, Instant};

use egui::{Context, Vec2};
use egui_notify::{Anchor, Toast};

/// Readable line length at the default ~12 px font, under ~360 px wide.
const TOAST_LINE_CHARS: usize = 60;

#[derive(Default)]
pub(crate) struct Toasts(
    egui_notify::Toasts,
    /// The latest captions and when each leaves, kept while a script
    /// reads them.
    std::collections::VecDeque<(String, Instant)>,
    #[cfg(test)] Vec<(&'static str, String)>,
);

/// How many captions `Toasts` keeps for a script; more never show at once.
const KEPT_CAPTIONS: usize = 8;

/// How long a success or a note stays.
const BRIEF: Duration = Duration::from_millis(3500);

/// How long an error or a warning stays: long enough to read at about 15
/// characters a second, 8 to 20 s. Hovering holds a toast, and each can
/// be closed.
fn reading_time(caption: &str) -> Duration {
    Duration::from_secs_f32((4.0 + caption.chars().count() as f32 / 15.0).clamp(8.0, 20.0))
}

impl Toasts {
    pub fn new(anchor: Anchor, margin: Vec2) -> Self {
        Self(
            egui_notify::Toasts::default().with_anchor(anchor).with_margin(margin),
            std::collections::VecDeque::new(),
            #[cfg(test)] Vec::new(),
        )
    }

    pub fn success(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.push("success", caption.into(), BRIEF, |toasts, text| toasts.success(text))
    }
    pub fn info(&mut self, caption: impl Into<String>) -> &mut Toast {
        self.push("info", caption.into(), BRIEF, |toasts, text| toasts.info(text))
    }
    pub fn warning(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.push("warning", caption.clone(), reading_time(&caption), |toasts, text| toasts.warning(text))
    }
    pub fn error(&mut self, caption: impl Into<String>) -> &mut Toast {
        let caption = caption.into();
        self.push("error", caption.clone(), reading_time(&caption), |toasts, text| toasts.error(text))
    }

    #[cfg_attr(not(test), allow(unused_variables))] // the level is kept for tests only
    fn push(&mut self, level: &'static str, caption: String, lasts: Duration, show: fn(&mut egui_notify::Toasts, String) -> &mut Toast) -> &mut Toast {
        self.keep(&caption, lasts);
        let text = wrap(&caption);
        #[cfg(test)] self.2.push((level, caption));
        show(&mut self.0, text).duration(lasts)
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

    fn keep(&mut self, caption: &str, lasts: Duration) {
        if !crate::gui::automation::record::is_on() {
            return;
        }
        if self.1.len() == KEPT_CAPTIONS {
            self.1.pop_front();
        }
        self.1.push_back((caption.to_owned(), Instant::now() + lasts));
    }

    /// The captions on screen, oldest first: those not yet due to leave,
    /// the latest as many as are shown. Empty unless a script turned
    /// recording on.
    pub fn shown(&self) -> Vec<String> {
        let now = Instant::now();
        let mut shown: Vec<String> = self.1.iter().rev().filter(|(_, leaves)| *leaves > now).take(self.0.len()).map(|(caption, _)| caption.clone()).collect();
        shown.reverse();
        shown
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

    /// An error stays long enough to read, a long one longer, within
    /// bounds; a success stays briefly.
    #[test]
    fn an_error_stays_long_enough_to_read() {
        assert_eq!(reading_time("Could not save."), Duration::from_secs(8));
        let long = "Could not install CUDA: the download stopped. Check the connection and try again from Settings.";
        assert!(reading_time(long) > Duration::from_secs(10), "{:?}", reading_time(long));
        assert_eq!(reading_time(&"x".repeat(2000)), Duration::from_secs(20));
        assert!(BRIEF < reading_time(""));
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(wrap(""), "");
    }
}
