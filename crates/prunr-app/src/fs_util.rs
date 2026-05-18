//! File-system helpers shared between subprocess IPC and GUI cleanup paths.

use std::path::Path;

/// Remove every regular file directly inside `dir`. Non-recursive — uses
/// `DirEntry::file_type()` (does NOT follow symlinks) so only true regular
/// files at depth 0 are removed; subdirectories, symlinks, and device nodes
/// are all skipped.
///
/// Silent on errors — best-effort housekeeping.
pub(crate) fn sweep_dir_files(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_type().ok().is_some_and(|t| t.is_file()) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// `sweep_dir_files` on a background thread. Used by graceful-exit cleanup
/// paths where blocking the shutdown sequence on file-system I/O would
/// stall the parent process.
pub(crate) fn sweep_dir_files_async(dir: std::path::PathBuf) {
    std::thread::spawn(move || sweep_dir_files(&dir));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweeps_files_leaves_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), b"x").unwrap();
        std::fs::write(tmp.path().join("b.txt"), b"y").unwrap();
        let sub = tmp.path().join("nested");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("c.txt"), b"z").unwrap();

        sweep_dir_files(tmp.path());

        assert!(!tmp.path().join("a.txt").exists());
        assert!(!tmp.path().join("b.txt").exists());
        assert!(sub.is_dir(), "subdir survives non-recursive sweep");
        assert!(sub.join("c.txt").exists(), "subdir contents survive");
    }

    #[test]
    fn missing_dir_is_a_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let nope = tmp.path().join("does-not-exist");
        sweep_dir_files(&nope); // must not panic
    }
}
