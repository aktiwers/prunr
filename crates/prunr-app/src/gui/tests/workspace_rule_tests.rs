//! Rules for the whole workspace that a reader cannot be trusted to
//! remember, checked on the source: each test names the rule, scans the
//! production code of every crate (tests are exempt) and lists each line
//! that breaks it with what to use instead.

use std::path::{Path, PathBuf};

use super::rule_scan_tests::{production_lines, rust_files};

/// Every crate's production sources: `(path from crates/, text)`.
fn workspace_sources() -> Vec<(String, String)> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&crates).expect("crates dir") {
        let src = entry.expect("dir entry").path().join("src");
        if src.is_dir() {
            rust_files(&src, &mut files);
        }
    }
    files.sort();
    files
        .into_iter()
        .map(|file: PathBuf| {
            let rel = file.strip_prefix(&crates).expect("under crates").to_string_lossy().replace('\\', "/");
            let text = std::fs::read_to_string(&file).expect("readable source");
            (rel, text)
        })
        .filter(|(rel, _)| !is_test_file(rel))
        .collect()
}

fn is_test_file(rel: &str) -> bool {
    rel.contains("/tests/") || rel.ends_with("_tests.rs") || rel.ends_with("/tests.rs")
}

/// The production text of `source` as one string, test modules cut.
fn production_text(source: &str) -> String {
    production_lines(source).map(|(_, l)| l).collect::<Vec<_>>().join("\n")
}

/// `line`'s code, before any `//` comment that is not inside a string.
fn code_part(line: &str) -> &str {
    let mut in_str = false;
    let bytes = line.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'"' if i == 0 || bytes[i - 1] != b'\\' => in_str = !in_str,
            b'/' if !in_str && bytes.get(i + 1) == Some(&b'/') => return &line[..i],
            _ => {}
        }
    }
    line
}

/// `line`'s comment text (after a `//` outside any string), if any.
fn comment_part(line: &str) -> Option<&str> {
    let code = code_part(line);
    (code.len() < line.len()).then(|| &line[code.len()..])
}

fn report(found: Vec<String>, what: &str) {
    assert!(found.is_empty(), "{what}:\n{}", found.join("\n"));
}

/// Diagnostics go through `tracing`; stderr is the CLI's user interface
/// and `main` before the subscriber starts.
#[test]
fn only_the_cli_and_main_write_to_stderr() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if rel.ends_with("/main.rs") || rel.ends_with("/cli.rs") || rel.contains("/bin/") {
            continue;
        }
        for (n, line) in production_lines(&text) {
            if code_part(line).contains("eprintln!") || code_part(line).contains("eprint!") {
                found.push(format!("{rel}:{n}: use tracing::{{warn, error, ..}}! with fields"));
            }
        }
    }
    report(found, "stderr written outside the CLI");
}

/// A poisoned lock is taken over, not panicked on: one panicking thread
/// would otherwise take every later user of the lock down with it.
#[test]
fn locks_take_over_a_poisoned_guard() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        let code: String = production_text(&text).lines().map(code_part).collect::<Vec<_>>().join("\n");
        let squeezed: String = code.split_whitespace().collect();
        for lock in [".lock()", ".read()", ".write()"] {
            for take in [".unwrap()", ".expect("] {
                let pattern = format!("{lock}{take}");
                if squeezed.contains(&pattern) {
                    found.push(format!("{rel}: `{pattern}`: use .unwrap_or_else(std::sync::PoisonError::into_inner)"));
                }
            }
        }
    }
    report(found, "locks that panic on poison");
}

/// Dialogs and panels are made with the surface helpers, which give them
/// the backdrop, close and placement every surface shares.
#[test]
fn no_bare_egui_window() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if !rel.starts_with("prunr-app/src/gui/") || rel.ends_with("gui/theme.rs") {
            continue;
        }
        for (n, line) in production_lines(&text) {
            if code_part(line).contains("egui::Window::new") {
                found.push(format!("{rel}:{n}: use theme::standard_modal_window, chip::popup_for or chip::flyout_for"));
            }
        }
    }
    report(found, "bare egui windows");
}

/// Comment lines that name a planning id on purpose, with why.
const PLANNING_ALLOWED: [(&str, &str, &str); 2] = [
    ("prunr-app/src/hardware.rs", "Apple M1/M2/M3", "chip names, not planning ids"),
    ("prunr-app/src/hardware.rs", "Apple M2 Pro", "a chip name"),
];

/// Whether `word` (punctuation stripped) is a planning id: H6, M11, K3,
/// P24-3, #123 (an upstream issue reads owner/repo#123 and passes).
fn is_planning_id(word: &str) -> bool {
    let tracker = |w: &str| {
        let mut chars = w.chars();
        matches!(chars.next(), Some('H' | 'M' | 'K')) && (1..=2).contains(&chars.clone().count()) && chars.all(|c| c.is_ascii_digit())
    };
    let phase_task = |w: &str| {
        w.strip_prefix('P').and_then(|r| r.split_once('-')).is_some_and(|(a, b)| {
            !a.is_empty() && !b.is_empty() && a.chars().all(|c| c.is_ascii_digit()) && b.chars().all(|c| c.is_ascii_digit())
        })
    };
    let issue = |w: &str| w.strip_prefix('#').is_some_and(|r| r.len() >= 2 && r.chars().all(|c| c.is_ascii_digit()));
    tracker(word) || phase_task(word) || issue(word)
}

/// Comments say why, never what is queued: planning ids and future work
/// rot when the plan moves.
#[test]
fn comments_name_no_planning_ids_or_future_work() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        for (n, line) in production_lines(&text) {
            let Some(comment) = comment_part(line) else { continue };
            if PLANNING_ALLOWED.iter().any(|(f, t, _)| rel == *f && comment.contains(t)) {
                continue;
            }
            let words: Vec<&str> = comment
                .split(|c: char| c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '(' | ')' | '/' | '`' | '\'' | '"'))
                .filter(|w| !w.is_empty())
                .collect();
            let phase = words.windows(2).any(|w| w[0] == "Phase" && w[1].chars().all(|c| c.is_ascii_digit()));
            let queued = ["REVIEW-FINDINGS", "DEFERRED", " will wire", " will move", " will refactor", "will be wired", "once #"]
                .iter()
                .any(|p| comment.contains(p));
            let id = words.iter().any(|w| is_planning_id(w));
            if phase || queued || id {
                found.push(format!("{rel}:{n}: {}", comment.trim()));
            }
        }
    }
    report(found, "planning ids or future work in comments (they belong in the planning docs)");
}

/// Words that describe our plumbing, not what a knob does.
const ENGINEERING_WORDS: [&str; 10] = [
    "tier-1", "tier-2", "tier 1", "tier 2", "real-time", "updates live", "inference", "postprocess", "debounce", "subprocess",
];

/// The string literals on `line` (a quote escaped inside one is kept).
fn literals(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut start = None;
    for i in 0..bytes.len() {
        if bytes[i] == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            match start.take() {
                Some(s) => out.push(&line[s..i]),
                None => start = Some(i + 1),
            }
        }
    }
    out
}

/// What the user reads (labels, tooltips, toasts) uses imaging words,
/// not ours: no tiers, inference, debounce, subprocess or ORT.
#[test]
fn user_text_has_no_engineering_words() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if !rel.starts_with("prunr-app/src/gui/views/") {
            continue;
        }
        for (n, line) in production_lines(&text) {
            let code = code_part(line);
            if ["tracing::", "debug!(", "info!(", "warn!(", "error!(", "Id::new("].iter().any(|t| code.contains(t)) {
                continue;
            }
            for literal in literals(code) {
                let lower = literal.to_lowercase();
                let word = ENGINEERING_WORDS.iter().find(|w| lower.contains(*w));
                let ort = literal.split(|c: char| !c.is_alphanumeric()).any(|w| w == "ORT");
                if let Some(word) = word.copied().or(ort.then_some("ORT")) {
                    found.push(format!("{rel}:{n}: \"{literal}\" says {word:?}"));
                }
            }
        }
    }
    report(found, "engineering words in user text");
}

/// The accessors that hold the egui context's lock while their closure runs.
const CTX_LOCKING: [&str; 6] = ["input(", "input_mut(", "memory(", "memory_mut(", "data(", "data_mut("];

/// No egui context accessor inside another's closure: each takes the
/// context's lock, and a nested one deadlocks once another thread (a
/// repaint request) waits to write.
#[test]
fn no_egui_context_accessor_nests_in_another() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if !rel.starts_with("prunr-app/") {
            continue;
        }
        let code: String = production_text(&text).lines().map(code_part).collect::<Vec<_>>().join("\n");
        for receiver in ["ctx.", "ctx()."] {
            for accessor in CTX_LOCKING {
                let opener = format!("{receiver}{accessor}");
                let mut from = 0;
                while let Some(at) = code[from..].find(&opener) {
                    let start = from + at + opener.len();
                    let mut depth = 1usize;
                    let mut end = start;
                    for (i, c) in code[start..].char_indices() {
                        match c {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        if depth == 0 {
                            end = start + i;
                            break;
                        }
                    }
                    let body = &code[start..end];
                    if body.contains("ctx.") || body.contains("ctx().") {
                        let line = code[..start].lines().count();
                        found.push(format!("{rel}:{line}: `{opener}..)` calls the context again inside"));
                    }
                    from = start;
                }
            }
        }
    }
    report(found, "nested egui context accessors");
}

/// Whether the item at `line` (0-based) is under `#[cfg(test)]` or
/// `#[cfg(any(test, ...))]`: on itself, or on the `impl` holding it.
fn is_test_only(lines: &[&str], line: usize) -> bool {
    let gated = |at: usize| {
        lines[..at]
            .iter()
            .rev()
            .map(|l| l.trim())
            .take_while(|l| l.starts_with("#[") || l.starts_with("///") || l.starts_with("//"))
            .any(|l| l.starts_with("#[cfg(test)]") || l.starts_with("#[cfg(any(test"))
    };
    if gated(line) {
        return true;
    }
    let holder = lines[..line].iter().rposition(|l| l.starts_with("impl") || l.starts_with("pub impl"));
    holder.is_some_and(gated)
}

/// Test constructors are compiled into tests only, never the release.
#[test]
fn test_constructors_are_test_only() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        let lines: Vec<&str> = production_lines(&text).map(|(_, l)| l).collect();
        for (i, line) in lines.iter().enumerate() {
            let code = code_part(line);
            if (code.contains("fn new_for_test") || code.contains("fn for_test")) && !is_test_only(&lines, i) {
                found.push(format!("{rel}:{}: gate it with #[cfg(test)] or #[cfg(any(test, feature = \"test-utils\"))]", i + 1));
            }
        }
    }
    report(found, "test constructors in the release build");
}

/// An item is looked up by id with `BatchManager::find_by_id{,_mut}`,
/// the one place that knows how items are stored.
#[test]
fn items_are_found_by_id_through_the_batch() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if !rel.starts_with("prunr-app/src/gui/") || rel.ends_with("gui/batch_manager.rs") {
            continue;
        }
        for (n, line) in production_lines(&text) {
            let squeezed: String = code_part(line).split_whitespace().collect();
            if squeezed.contains("items.iter") && squeezed.contains(".find(|") && squeezed.contains(".id==") {
                found.push(format!("{rel}:{n}: use BatchManager::find_by_id{{,_mut}}"));
            }
        }
    }
    report(found, "hand-rolled item lookups");
}
