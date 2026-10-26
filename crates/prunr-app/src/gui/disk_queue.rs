//! Files written and removed in the order they were asked for, on one
//! worker, off the UI thread: settings, presets, the undo states kept on
//! disk once they go, and what the app changes in the store of what it
//! learned about this machine (`ep_compat`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};

enum Job {
    Write(PathBuf, Vec<u8>),
    Remove(PathBuf),
    /// A change made by another module's own writer, kept in order.
    Run(Box<dyn FnOnce() + Send>),
    Flush(mpsc::Sender<()>),
}

fn jobs() -> &'static mpsc::Sender<Job> {
    static JOBS: OnceLock<mpsc::Sender<Job>> = OnceLock::new();
    JOBS.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("disk-queue".into())
            .spawn(move || rx.into_iter().for_each(|job| run(job, &FAILURES)))
            .map_err(|err| tracing::error!(%err, "disk queue did not start"))
            .ok();
        tx
    })
}

fn run(job: Job, failures: &Failures) {
    match job {
        Job::Write(path, bytes) => {
            if let Err(err) = write_whole(&path, &bytes) {
                tracing::warn!(?path, %err, "file not written");
                failures.note(format!("A settings or preset file could not be saved ({err}). Changes made now may not be kept."));
            }
        }
        Job::Remove(path) => match std::fs::remove_file(&path) {
            Err(err) if err.kind() != std::io::ErrorKind::NotFound => tracing::warn!(?path, %err, "file not removed"),
            _ => {}
        },
        Job::Run(change) => change(),
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

/// The first write that failed, kept for the person; the rest only go to
/// the log, as one message says what is wrong.
struct Failures {
    failed: AtomicBool,
    first: Mutex<Option<String>>,
}

impl Failures {
    const fn new() -> Self {
        Self { failed: AtomicBool::new(false), first: Mutex::new(None) }
    }

    fn note(&self, message: String) {
        if !self.failed.swap(true, Ordering::AcqRel) {
            *self.first.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message);
        }
    }

    fn take(&self) -> Option<String> {
        if !self.failed.load(Ordering::Acquire) {
            return None;
        }
        self.first.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    }
}

static FAILURES: Failures = Failures::new();

/// The failed write to tell the person about, once a session.
pub(crate) fn take_failure() -> Option<String> {
    FAILURES.take()
}

pub(crate) fn write(path: PathBuf, bytes: Vec<u8>) {
    let _ = jobs().send(Job::Write(path, bytes));
}

pub(crate) fn remove(path: PathBuf) {
    let _ = jobs().send(Job::Remove(path));
}

pub(crate) fn call(change: impl FnOnce() + Send + 'static) {
    let _ = jobs().send(Job::Run(Box::new(change)));
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
        let failures = super::Failures::new();
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("settings.json");
        std::fs::create_dir(&blocked).unwrap();
        super::run(super::Job::Write(blocked.clone(), b"x".to_vec()), &failures);
        super::run(super::Job::Write(blocked, b"y".to_vec()), &failures);
        assert!(failures.take().is_some_and(|m| m.contains("could not be saved")));
        assert_eq!(failures.take(), None, "told once");
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
