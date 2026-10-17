//! Whether an image is processed, and what may run on it, has one
//! definition: `BatchItem::status()` / `has_result()` in item.rs and the
//! capability check. A hand-written test of the working copy or the
//! status elsewhere is how the gates drifted apart before (an erase that
//! missed a parked result, a render that waited for good), so the source
//! may not contain one outside those modules. Tests are exempt: they
//! build states.

use std::path::{Path, PathBuf};

const OWNERS: [&str; 2] = ["gui/item.rs", "gui/capability/"];

const BANNED: [(&str, &str); 18] = [
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
    ("counts.processing", "any_awaiting_cut() (counts waiting cuts too)"),
    ("status(), BatchStatus::Done", "has_result()"),
    ("status(), BatchStatus::Pending", "has_result()"),
    ("status(), BatchStatus::Processing", "is_cutting(), or is_idle() for any run"),
    ("j.engine == Engine::Segment", "awaits_cut()"),
    ("j.engine == super::item::Engine::Segment", "awaits_cut()"),
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

/// What the render code (`gui/views/`) may not do: it runs every frame
/// on the UI thread, so file I/O, a decode, an unpack, a blocking wait or
/// a texture upload there stalls the window. Work like that goes to a
/// worker and lands through a channel the frame drains.
const RENDER_BANNED: [(&str, &str); 11] = [
    ("std::fs::", "a worker (BatchManager.bg_io, disk_queue) and a channel"),
    ("File::open", "a worker and a channel"),
    ("ImageReader", "a decode on a worker (request_decode_source)"),
    ("to_rgba8()", "pixels prepared on a worker"),
    (".load()", "a handle (Pixels, Region) unpacked where it is used"),
    ("decompress(", "tensors unpacked on the render thread (Packed)"),
    ("load_texture(", "drain_background_channels, from a ColorImage built off-thread"),
    (".recv()", "try_recv()"),
    ("thread::sleep", "a repaint request"),
    ("thread::spawn", "an app method or coordinator: views return an intent"),
    ("is_available(", "HardwareInstallCache (read after installs, not per frame)"),
];

/// Render-code lines that match a banned pattern and are fine, each with
/// why: `(file, the line's text, reason)`.
const RENDER_ALLOWED: [(&str, &str, &str); 2] = [
    ("gui/views/canvas.rs", "ctx.load_texture(\"checker_dark\"", "a 2 KB checker made once (OnceLock)"),
    ("gui/views/canvas.rs", "ctx.load_texture(\"checker_light\"", "a 2 KB checker made once (OnceLock)"),
];

#[test]
fn the_render_code_does_no_blocking_work() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src.join("gui/views"), &mut files);
    let mut found = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&src).expect("under src").to_string_lossy().replace('\\', "/");
        if rel.ends_with("_tests.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for (n, line) in production_lines(&source) {
            if line.trim_start().starts_with("//") {
                continue;
            }
            if RENDER_ALLOWED.iter().any(|(f, text, _)| rel == *f && line.contains(text)) {
                continue;
            }
            for (pattern, instead) in RENDER_BANNED {
                if line.contains(pattern) {
                    found.push(format!("{rel}:{n}: `{pattern}`: use {instead}"));
                }
            }
        }
    }
    assert!(found.is_empty(), "blocking work in the render code:\n{}", found.join("\n"));
}

/// Changing the selection is a full-resolution pass: it goes through the
/// selection queue (`PrunrApp::queue_selection_op`), in order, off the UI
/// thread. The edits themselves live in the queue, the merge in
/// processor.rs (also run by the Magic Brush retune job) and the landing
/// in item.rs.
const SELECTION_OWNERS: [&str; 3] = ["gui/selection_queue.rs", "gui/processor.rs", "gui/item.rs"];

const SELECTION_BANNED: [&str; 4] = ["merge_stroke(", "commit_selection_mask(", "bake_selection(", ".invert()"];

/// Lines outside the owners that may use a banned call, with why.
const SELECTION_ALLOWED: [(&str, &str, &str); 1] = [
    ("gui/batch_manager.rs", ".is_some_and(|item| item.commit_selection_mask(", "BatchManager::commit_selection, a #[cfg(test)] helper"),
];

#[test]
fn selection_edits_go_through_the_queue() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src.join("gui"), &mut files);
    let mut found = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&src).expect("under src").to_string_lossy().replace('\\', "/");
        if SELECTION_OWNERS.contains(&rel.as_str()) || rel.starts_with("gui/tests/") || rel.ends_with("_tests.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for (n, line) in production_lines(&source) {
            if line.trim_start().starts_with("//") || SELECTION_ALLOWED.iter().any(|(f, text, _)| rel == *f && line.contains(text)) {
                continue;
            }
            for pattern in SELECTION_BANNED {
                if line.contains(pattern) {
                    found.push(format!("{rel}:{n}: `{pattern}`: use queue_selection_op"));
                }
            }
        }
    }
    assert!(found.is_empty(), "selection edits outside the queue:\n{}", found.join("\n"));
}
