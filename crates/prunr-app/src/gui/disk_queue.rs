//! Settings and preset files, written in the order they were asked for on
//! one worker, off the UI thread.

use std::path::PathBuf;
use std::sync::{mpsc, OnceLock};

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
            .name("config-writer".into())
            .spawn(move || rx.into_iter().for_each(run))
            .map_err(|err| tracing::error!(%err, "config writer did not start"))
            .ok();
        tx
    })
}

fn run(job: Job) {
    match job {
        Job::Write(path, bytes) => {
            if let Err(err) = write_whole(&path, &bytes) {
                tracing::warn!(?path, %err, "config file not written");
            }
        }
        Job::Remove(path) => match std::fs::remove_file(&path) {
            Err(err) if err.kind() != std::io::ErrorKind::NotFound => tracing::warn!(?path, %err, "config file not removed"),
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
    let mut partial = path.as_os_str().to_owned();
    partial.push(".part");
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, path)
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
        assert!(!dir.path().join("nested").join("settings.json.part").exists());
        super::remove(path.clone());
        super::write(dir.path().join("other.json"), b"x".to_vec());
        super::flush();
        assert!(!path.exists());
        super::remove(path.clone());
        super::flush();
    }
}
