//! File-system helpers shared between subprocess IPC, GUI cleanup and
//! every file the app writes over.

use std::path::{Path, PathBuf};

/// Write `path` through a temp file beside it, renamed into place once
/// whole and on disk: a reader never sees half a file, and a write that
/// fails (a full disk, a dropped share) leaves the file that was there
/// untouched. The temp file goes when the write fails. A folder that lets
/// a file be overwritten but not made gets the file written in place.
pub fn write_replacing(
    path: &Path,
    write: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut part = path.as_os_str().to_owned();
    part.push(format!(".{}.part", std::process::id()));
    let part = PathBuf::from(part);
    let file = match std::fs::File::create(&part) {
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied && path.is_file() => {
            return write_synced(std::fs::File::create(path)?, write);
        }
        file => file?,
    };
    let result = write_synced(file, write).and_then(|()| std::fs::rename(&part, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

/// Write into `file` and sync it to disk: a share reports a failed write
/// only when flushed, and the rename must not land before that.
fn write_synced(
    file: std::fs::File,
    write: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(file);
    write(&mut w)?;
    w.into_inner().map_err(std::io::IntoInnerError::into_error)?.sync_all()
}

/// `bytes` as the whole of `path`, through `write_replacing`.
pub fn write_bytes_replacing(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    write_replacing(path, |w| std::io::Write::write_all(w, bytes))
}

/// `img` saved to `path` in the format its extension names, through
/// `write_replacing`.
pub fn save_image_replacing(path: &Path, img: &image::RgbaImage) -> Result<(), String> {
    let format = image::ImageFormat::from_path(path).map_err(|e| e.to_string())?;
    write_replacing(path, |w| img.write_to(w, format).map_err(std::io::Error::other)).map_err(|e| e.to_string())
}

/// Remove every regular file directly inside `dir`. Non-recursive — uses
/// `DirEntry::file_type()` (does NOT follow symlinks) so only true regular
/// files at depth 0 are removed; subdirectories, symlinks, and device nodes
/// are all skipped.
///
/// Silent on errors — best-effort housekeeping.
pub(crate) fn sweep_dir_files(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
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

/// Remove `{prefix}{pid}` dirs directly inside `parent` whose owning
/// process is no longer running, sparing `self_pid`. A crash or OOM kill
/// skips graceful cleanup, so without this such dirs pile up forever.
///
/// Liveness is `/proc/{pid}` on Linux. Other platforms fall back to
/// "untouched for more than 24 h", conservative enough not to clobber a
/// concurrent prunr instance.
pub(crate) fn sweep_dead_pid_dirs(parent: &Path, prefix: &str, self_pid: u32) {
    let Ok(entries) = std::fs::read_dir(parent) else { return };
    for entry in entries.flatten() {
        if !entry.file_type().ok().is_some_and(|t| t.is_dir()) { continue; }
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else { continue };
        let Some(pid_str) = name.strip_prefix(prefix) else { continue };
        let Ok(other_pid) = pid_str.parse::<u32>() else { continue };
        if other_pid == self_pid || !is_pid_dead(other_pid, &entry) { continue; }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

#[cfg(target_os = "linux")]
fn is_pid_dead(pid: u32, _entry: &std::fs::DirEntry) -> bool {
    !Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn is_pid_dead(_pid: u32, entry: &std::fs::DirEntry) -> bool {
    let Ok(meta) = entry.metadata() else { return false };
    let Ok(modified) = meta.modified() else { return false };
    let Ok(age) = modified.elapsed() else { return false };
    age > std::time::Duration::from_secs(86_400)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder that lets its files be overwritten but none be made still
    /// takes the save, written in place.
    #[cfg(unix)]
    #[test]
    fn a_folder_that_refuses_new_files_is_written_in_place() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("photo.png");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        let written = write_bytes_replacing(&path, b"new");
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        written.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    /// A write that fails part-way (a full disk) keeps the file that was
    /// there and leaves no temp file; one that works replaces it whole.
    #[test]
    fn a_failed_write_keeps_the_old_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("photo.png");
        std::fs::write(&path, b"the old file").unwrap();
        let failed = write_replacing(&path, |w| {
            std::io::Write::write_all(w, b"half of the new")?;
            Err(std::io::Error::other("disk full"))
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"the old file");
        let left: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("photo.png")], "no temp file left");

        write_bytes_replacing(&path, b"the new file").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"the new file");
        write_bytes_replacing(&tmp.path().join("fresh.png"), b"made").unwrap();
        assert_eq!(std::fs::read(tmp.path().join("fresh.png")).unwrap(), b"made");
    }

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
