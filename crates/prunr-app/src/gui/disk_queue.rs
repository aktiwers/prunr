//! Files written and removed in the order they were asked for, on one
//! worker, off the UI thread: settings, presets, and the undo states
//! kept on disk once they go.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};

enum Job {
    Write(PathBuf, Vec<u8>),
    Remove(PathBuf),
    Flush(mpsc::Sender<()>),
}

fn jobs() -> &'static mpsc::Sender<Job> {
    static JOBS: OnceLock<mpsc::Sender<Job>> = OnceLock::new();
    JOBS.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("disk-queue".into())
            .spawn(move || rx.into_iter().for_each(run))
            .map_err(|err| tracing::error!(%err, "disk queue did not start"))
            .ok();
        tx
    })
}

fn run(job: Job) {
    match job {
        Job::Write(path, bytes) => {
            if let Err(err) = write_whole(&path, &bytes) {
                tracing::warn!(?path, %err, "file not written");
                note_failure(format!("A settings or preset file could not be saved ({err}). Changes made now may not be kept."));
            }
        }
        Job::Remove(path) => match std::fs::remove_file(&path) {
            Err(err) if err.kind() != std::io::ErrorKind::NotFound => tracing::warn!(?path, %err, "file not removed"),
            _ => {}
        },
        Job::Flush(done) => {
            let _ = done.send(());
        }
    }
}

/// A reader never sees half a file: it lands under its name once whole.
fn write_whole(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::fs_util::write_bytes_replacing(path, bytes)
}

static FAILURE: Mutex<Option<String>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);

/// The first write that failed this session is kept for the person; the
/// rest only go to the log, as one message says what is wrong.
fn note_failure(message: String) {
    if !FAILED.swap(true, Ordering::AcqRel) {
        *FAILURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message);
    }
}

/// The failed write to tell the person about, once a session.
pub(crate) fn take_failure() -> Option<String> {
    if !FAILED.load(Ordering::Acquire) {
        return None;
    }
    FAILURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
}

pub(crate) fn write(path: PathBuf, bytes: Vec<u8>) {
    let _ = jobs().send(Job::Write(path, bytes));
}

pub(crate) fn remove(path: PathBuf) {
    let _ = jobs().send(Job::Remove(path));
}

/// Wait until everything asked for so far is on disk: before the app
/// exits, so the last change is not lost.
pub(crate) fn flush() {
    let (tx, rx) = mpsc::channel();
    if jobs().send(Job::Flush(tx)).is_ok() {
        let _ = rx.recv();
    }
}

#[cfg(test)]
mod tests {
    /// The first failed write is told once; a second is not told again.
    #[test]
    fn a_failed_write_is_told_once() {
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("settings.json");
        std::fs::create_dir(&blocked).unwrap();
        super::write(blocked.clone(), b"x".to_vec());
        super::write(blocked, b"y".to_vec());
        super::flush();
        assert!(super::take_failure().is_some_and(|m| m.contains("could not be saved")));
        assert_eq!(super::take_failure(), None, "told once");
    }

    /// Writes and removals land in the order asked, whole, and a flush
    /// returns once they are on disk.
    #[test]
    fn config_files_land_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("settings.json");
        super::write(path.clone(), b"first".to_vec());
        super::write(path.clone(), b"second".to_vec());
        super::flush();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let left: Vec<_> = std::fs::read_dir(dir.path().join("nested")).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("settings.json")], "no temp file left");
        super::remove(path.clone());
        super::write(dir.path().join("other.json"), b"x".to_vec());
        super::flush();
        assert!(!path.exists());
        super::remove(path.clone());
        super::flush();
    }
}
