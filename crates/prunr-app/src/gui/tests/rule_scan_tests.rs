//! Whether an image is processed, and what may run on it, has one
//! definition: `BatchItem::status()` / `has_result()` in item.rs and the
//! capability check. A hand-written test of the working copy or the
//! status elsewhere is how the gates drifted apart before (an erase that
//! missed a parked result, a render that waited for good), so the source
//! may not contain one outside those modules. Tests are exempt: they
//! build states.

use std::path::{Path, PathBuf};

const OWNERS: [&str; 2] = ["gui/item.rs", "gui/capability/"];

const BANNED: [(&str, &str); 12] = [
    ("result_rgba.is_some()", "BatchItem::has_result() (parked results count)"),
    ("result_rgba.is_none()", "BatchItem::has_result() (parked results count)"),
    ("status() == BatchStatus::Done", "has_result(), or the capability check for what may run"),
    ("status() != BatchStatus::Done", "has_result(), or the capability check for what may run"),
    ("status() == BatchStatus::Pending", "has_result(), or the capability check for what may run"),
    ("status() != BatchStatus::Pending", "has_result(), or the capability check for what may run"),
    ("status() == BatchStatus::Processing", "is_cutting(), or is_idle() for any run"),
    ("status() != BatchStatus::Processing", "is_cutting(), or is_idle() for any run"),
    ("cached_tensor.is_some()", "shows_cutout() / Need::Cutout (a kept mask may be from an undone step)"),
    ("AppState::Loaded | AppState::Done", "is_idle() or verdict(): the selected item's display state is not a gate"),
    ("app_state() == AppState::Processing", "is_cutting() / is_idle()"),
    ("status_counts().processing", "any_awaiting_cut() (counts waiting cuts too)"),
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("source dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The lines of `source` before its first inline `#[cfg(test)]` module
/// (`mod name {`); a `mod name;` declaration is not one.
fn production_lines(source: &str) -> impl Iterator<Item = (usize, &str)> {
    let lines: Vec<&str> = source.lines().collect();
    let end = lines.windows(2)
        .position(|w| w[0].trim() == "#[cfg(test)]" && w[1].trim_start().starts_with("mod ") && w[1].trim_end().ends_with('{'))
        .unwrap_or(lines.len());
    lines.into_iter().take(end).enumerate().map(|(i, l)| (i + 1, l))
}

#[test]
fn no_gate_tests_the_working_copy_or_the_status_by_hand() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src.join("gui"), &mut files);
    let mut found = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&src).expect("under src").to_string_lossy().replace('\\', "/");
        if OWNERS.iter().any(|o| rel.starts_with(o)) || rel.starts_with("gui/tests/") || rel.ends_with("_tests.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for (n, line) in production_lines(&source) {
            for (pattern, instead) in BANNED {
                if line.contains(pattern) {
                    found.push(format!("{rel}:{n}: `{pattern}`: use {instead}"));
                }
            }
        }
    }
    assert!(found.is_empty(), "hand-written gates:\n{}", found.join("\n"));
}
