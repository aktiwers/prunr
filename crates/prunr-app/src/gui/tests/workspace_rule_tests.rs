//! Rules for the whole workspace that a reader cannot be trusted to
//! remember, checked on the source: each test names the rule, scans the
//! production code of every crate (tests are exempt) and lists each line
//! that breaks it with what to use instead.

use std::path::{Path, PathBuf};

use super::rule_scan_tests::{production_lines, rust_files};

/// Every `.rs` file of every crate under `dir_name` (`src`, `tests`):
/// `(path from crates/, text)`, read once per test run.
fn read_sources(dir_name: &str) -> Vec<(String, String)> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&crates).expect("crates dir") {
        let dir = entry.expect("dir entry").path().join(dir_name);
        if dir.is_dir() {
            rust_files(&dir, &mut files);
        }
    }
    files.sort();
    files
        .into_iter()
        .map(|file| {
            let rel = file.strip_prefix(&crates).expect("under crates").to_string_lossy().replace('\\', "/");
            let text = std::fs::read_to_string(&file).expect("readable source");
            (rel, text)
        })
        .collect()
}

/// Every crate's sources, test files included.
fn all_sources() -> &'static [(String, String)] {
    static SOURCES: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    SOURCES.get_or_init(|| read_sources("src"))
}

/// Every crate's production sources (test files left out).
fn workspace_sources() -> impl Iterator<Item = (&'static str, &'static str)> {
    all_sources().iter().filter(|(rel, _)| !is_test_file(rel)).map(|(rel, text)| (rel.as_str(), text.as_str()))
}

fn is_test_file(rel: &str) -> bool {
    rel.contains("/tests/") || rel.ends_with("_tests.rs") || rel.ends_with("/tests.rs")
}

/// The production code of `source` as one string: test modules and
/// comments cut.
fn production_code(source: &str) -> String {
    let mut code = String::with_capacity(source.len());
    for (_, line) in production_lines(source) {
        code.push_str(code_part(line));
        code.push('\n');
    }
    code
}

/// The byte ranges of `line`'s string literals (a quote escaped inside
/// one is kept), and where a `//` comment outside them starts.
fn scan_line(line: &str) -> (Vec<std::ops::Range<usize>>, Option<usize>) {
    let bytes = line.as_bytes();
    let mut literals = Vec::new();
    let mut open = None;
    for i in 0..bytes.len() {
        match bytes[i] {
            b'"' if i == 0 || bytes[i - 1] != b'\\' => match open.take() {
                Some(start) => literals.push(start..i),
                None => open = Some(i + 1),
            },
            b'/' if open.is_none() && bytes.get(i + 1) == Some(&b'/') => return (literals, Some(i)),
            _ => {}
        }
    }
    (literals, None)
}

/// `line`'s code, before any `//` comment that is not inside a string.
fn code_part(line: &str) -> &str {
    scan_line(line).1.map_or(line, |at| &line[..at])
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
        for (n, line) in production_lines(text) {
            let code = code_part(line);
            if code.contains("eprintln!") || code.contains("eprint!") {
                found.push(format!("{rel}:{n}: use tracing::{{warn, error, ..}}! with fields"));
            }
        }
    }
    report(found, "stderr written outside the CLI");
}

/// A float colour becomes a byte by rounding: `as u8` truncates, and
/// every channel comes out half a step dark.
#[test]
fn colour_bytes_are_rounded_not_truncated() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if rel.ends_with("prunr-core/src/math.rs") {
            continue;
        }
        let core = rel.starts_with("prunr-core/");
        for (n, line) in production_lines(text) {
            let code = code_part(line);
            let float = code.contains("255.0) as u8") || (core && code.contains("as f32") && code.contains(") as u8"));
            let integer = (code.contains("/ 255) as u8") || code.contains("/ 255).clamp(0, 255) as u8")) && !code.contains("+ 127");
            if (float && !code.contains("round") && !code.contains("+ 0.5") && !code.contains("floor")) || integer {
                found.push(format!("{rel}:{n}: use math::round_u8 (floats) or math::div255_round (integers)"));
            }
        }
    }
    report(found, "colour bytes truncated");
}

/// Decodes of the app's own images, which carry no EXIF turn.
const DECODE_ALLOWED: [(&str, &str); 1] = [
    ("prunr-app/src/gui/mod.rs", "the app icon, built in"),
];

/// Images are decoded through prunr-core's loaders, which turn a photo
/// the way its EXIF says: a decode of its own shows a phone's portrait
/// sideways.
#[test]
fn images_decode_through_the_upright_loaders() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        if rel.ends_with("prunr-core/src/formats.rs") || DECODE_ALLOWED.iter().any(|(f, _)| rel == *f) {
            continue;
        }
        for (n, line) in production_lines(text) {
            let code = code_part(line);
            if ["image::open(", "load_from_memory", "ImageReader::"].iter().any(|p| code.contains(p)) {
                found.push(format!("{rel}:{n}: use prunr_core::load_image_from_{{path,bytes}} / image_dimensions_from_{{path,bytes}}"));
            }
        }
    }
    report(found, "images decoded without their EXIF turn");
}

/// A poisoned lock is taken over, not panicked on: one panicking thread
/// would otherwise take every later user of the lock down with it.
#[test]
fn locks_take_over_a_poisoned_guard() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        let squeezed: String = production_code(text).split_whitespace().collect();
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
        for (n, line) in production_lines(text) {
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
        for (n, line) in production_lines(text) {
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

/// The string literals on `line`'s code.
/// The string literals in GUI source `text` that can reach the user, with
/// their line: those in a logging call (spanning every line it does), an
/// `expect` message or an egui id are left out.
fn user_literals(text: &str) -> Vec<(usize, &str)> {
    const LOGGING: [&str; 5] = ["tracing::", "debug!(", "info!(", "warn!(", "error!("];
    let mut out = Vec::new();
    let mut in_log = false;
    for (n, line) in production_lines(text) {
        let code = code_part(line);
        if in_log || LOGGING.iter().any(|t| code.contains(t)) {
            in_log = !code.trim_end().ends_with(';');
            continue;
        }
        if code.contains(".expect(") || code.contains("Id::new(") {
            continue;
        }
        out.extend(literals(code).map(|literal| (n, literal)));
    }
    out
}

/// GUI sources whose literals are no user text, with why.
const USER_TEXT_ALLOWED: [(&str, &str); 1] = [
    ("prunr-app/src/gui/automation/checks.rs", "holds the user-text rules: its literals are the banned words and the reasons, read by a script"),
];

/// The GUI's sources, every user-text rule's one scope.
fn is_gui_source(rel: &str) -> bool {
    rel.starts_with("prunr-app/src/gui/") && !USER_TEXT_ALLOWED.iter().any(|(f, _)| rel == *f)
}

fn literals(line: &str) -> impl Iterator<Item = &str> {
    scan_line(line).0.into_iter().map(move |r| &line[r])
}

/// What the user reads (labels, tooltips, toasts) uses imaging words,
/// not ours: no tiers, inference, debounce, subprocess or ORT.
#[test]
fn user_text_has_no_engineering_words() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources().filter(|(rel, _)| is_gui_source(rel)) {
        for (n, literal) in user_literals(text) {
            if let Some(word) = crate::gui::automation::checks::engineering_word(literal) {
                found.push(format!("{rel}:{n}: \"{literal}\" says {word:?}"));
            }
        }
    }
    report(found, "engineering words in user text");
}

/// Every character user text writes has a glyph in the app's fonts: a
/// missing one draws as an empty box ("Settings □ Hardware").
#[test]
fn user_text_writes_only_glyphs_the_fonts_have() {
    let ctx = egui::Context::default();
    egui_material_icons::initialize(&ctx);
    let _ = ctx.run_ui(Default::default(), |_| {});
    let font = egui::FontId::proportional(14.0);
    let mut found = Vec::new();
    for (rel, text) in workspace_sources().filter(|(rel, _)| is_gui_source(rel)) {
        for (n, literal) in user_literals(text) {
            if let Some(c) = literal.chars().find(|&c| !c.is_ascii() && !ctx.fonts_mut(|f| f.has_glyph(&font, c))) {
                found.push(format!("{rel}:{n}: \"{literal}\" writes {c:?}, which the fonts lack"));
            }
        }
    }
    report(found, "glyphs the fonts lack in user text");
}

/// A key named in user text comes from the binding table
/// (`shortcuts::keys`): Settings › Shortcuts rebinds them, and a key
/// written by hand stays wrong after.
#[test]
fn user_text_names_no_keys_by_hand() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources().filter(|(rel, _)| is_gui_source(rel) && !rel.ends_with("/shortcuts.rs")) {
        for (n, literal) in user_literals(text) {
            let words = literal.split(|c: char| !c.is_alphanumeric() && c != '+');
            let named = words.into_iter().find(|w| {
                let f_key = w.strip_prefix('F').is_some_and(|d| !d.is_empty() && d.len() <= 2 && d.bytes().all(|b| b.is_ascii_digit()));
                f_key || ["Shift+", "Ctrl+", "Cmd+", "Alt+"].iter().any(|m| w.starts_with(m))
            });
            if let Some(key) = named {
                found.push(format!("{rel}:{n}: \"{literal}\" names {key}: use shortcuts::keys"));
            }
        }
    }
    report(found, "keys written by hand in user text");
}

/// User text writes "…" (not three dots) and plurals through
/// `fmt::count` (not "(s)").
#[test]
fn user_text_has_no_three_dots_or_plural_brackets() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources().filter(|(rel, _)| is_gui_source(rel)) {
        for (n, literal) in user_literals(text) {
            if crate::gui::automation::checks::typography(literal).is_some() {
                found.push(format!("{rel}:{n}: \"{literal}\": use an ellipsis character / fmt::count"));
            }
        }
    }
    report(found, "three dots or (s) plurals in user text");
}

/// User text spells the US way, as most of it does: gray, color,
/// license, center.
#[test]
fn user_text_uses_us_spelling() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources().filter(|(rel, _)| is_gui_source(rel)) {
        for (n, literal) in user_literals(text) {
            if let Some(word) = crate::gui::automation::checks::british_word(literal) {
                found.push(format!("{rel}:{n}: \"{literal}\" spells {word:?}"));
            }
        }
    }
    report(found, "British spelling in user text");
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
        let code = production_code(text);
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
    let holder = lines[..line].iter().rposition(|l| l.starts_with("impl"));
    holder.is_some_and(gated)
}

/// Test constructors are compiled into tests only, never the release.
#[test]
fn test_constructors_are_test_only() {
    let mut found = Vec::new();
    for (rel, text) in workspace_sources() {
        let lines: Vec<&str> = production_lines(text).map(|(_, l)| l).collect();
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
        for (n, line) in production_lines(text) {
            let squeezed: String = code_part(line).split_whitespace().collect();
            if squeezed.contains("items.iter") && squeezed.contains(".find(|") && squeezed.contains(".id==") {
                found.push(format!("{rel}:{n}: use BatchManager::find_by_id{{,_mut}}"));
            }
        }
    }
    report(found, "hand-rolled item lookups");
}

/// Core entry points no contract test calls by name, each with the call
/// path to one that does; each link is a call one way or the other.
const EXERCISED_THROUGH: [&[&str]; 8] = [
    &["process_image_with_mask", "process_image"],
    &["process_image_from_decoded", "process_image_with_mask", "process_image"],
    &["process_image_unchecked", "process_image_from_decoded", "process_image_with_mask", "process_image"],
    &["infer_only", "infer_after_first_look"],
    &["first_look", "infer_after_first_look"],
    &["process_lama", "tile_compose"],
    &["upscale_rgba_with_engine", "upscale_tiled"],
    &["upscale_rgba", "upscale_rgba_with_engine", "upscale_tiled"],
];

/// A function in the source: whether it is `pub`, its signature, its body.
struct Function<'a> {
    name: &'a str,
    public: bool,
    signature: &'a str,
    body: &'a str,
}

/// Each `fn` with a body in `code` (comments already cut).
fn functions(code: &str) -> Vec<Function<'_>> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    for (at, _) in code.match_indices("fn ") {
        if code[..at].chars().next_back().is_some_and(ident) {
            continue;
        }
        let rest = &code[at + 3..];
        let name = &rest[..rest.find(|c: char| !ident(c)).unwrap_or(rest.len())];
        let Some(open) = code[at..].find('{').map(|o| at + o) else { break };
        if code[at..open].contains(';') {
            continue;
        }
        let mut depth = 0usize;
        let mut end = code.len();
        for (i, c) in code[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let public = code[..at].trim_end().ends_with("pub") || code[..at].trim_end().ends_with("pub(crate)");
        out.push(Function { name, public, signature: &code[at..open], body: &code[open..end] });
    }
    out
}

/// Whether `text` calls `name` (not a longer name ending in it).
fn calls(text: &str, name: &str) -> bool {
    text.match_indices(&format!("{name}(")).any(|(i, _)| !text[..i].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_'))
}

/// Whether `signature` takes a `&Progress` (however the path is spelled).
fn takes_progress(signature: &str) -> bool {
    signature.match_indices("Progress").any(|(i, _)| {
        let after = signature[i + "Progress".len()..].chars().next();
        let before = signature[..i].trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == ':');
        !after.is_some_and(|c| c.is_alphanumeric() || c == '_') && before.ends_with('&')
    })
}

/// Anything a user waits on reports through `Progress`, and a run that
/// reports is held to `check_contract`: every public core function taking
/// a `Progress` is called by a test that checks its reports, or is listed
/// with the call path to one that is.
#[test]
fn every_core_pipeline_is_held_to_the_progress_contract() {
    let core = |rel: &&(String, String)| rel.0.starts_with("prunr-core/");
    let core_tests = read_sources("tests");
    let contract_tests: Vec<&str> = all_sources()
        .iter()
        .chain(core_tests.iter())
        .filter(|s| core(s))
        .flat_map(|(_, text)| {
            text.match_indices("check_contract(").filter_map(move |(i, _)| {
                let start = text[..i].rfind("#[test]")?;
                let end = text[i..].find("#[test]").map_or(text.len(), |e| i + e);
                Some(&text[start..end])
            })
        })
        .collect();
    let code: Vec<String> = all_sources().iter().filter(|s| core(s) && !is_test_file(&s.0)).map(|(_, t)| production_code(t)).collect();
    let fns: Vec<Function<'_>> = code.iter().flat_map(|c| functions(c)).collect();
    let tested = |name: &str| contract_tests.iter().any(|t| calls(t, name));
    let linked = |a: &str, b: &str| fns.iter().any(|f| (f.name == a && calls(f.body, b)) || (f.name == b && calls(f.body, a)));

    for path in EXERCISED_THROUGH {
        for pair in path.windows(2) {
            assert!(linked(pair[0], pair[1]), "{} and {} no longer call each other: fix the path", pair[0], pair[1]);
        }
        let last = path[path.len() - 1];
        assert!(tested(last), "{last} ends a path but no contract test calls it");
    }
    let mut missing: Vec<&str> = fns
        .iter()
        .filter(|f| f.public && takes_progress(f.signature) && !tested(f.name))
        .filter(|f| !EXERCISED_THROUGH.iter().any(|p| p[0] == f.name))
        .map(|f| f.name)
        .collect();
    missing.sort_unstable();
    missing.dedup();
    report(
        missing.iter().map(|m| format!("{m}: record it with RecordingSink and check_contract, or list its path")).collect(),
        "core pipelines not held to the progress contract",
    );
}
