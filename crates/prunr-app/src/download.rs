//! One file downloaded into place: through a partial file of this
//! process's own, checked against its SHA-256, moved in only when it
//! matches.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use prunr_runtime_install::DlError;

/// Download `url` to `dest` via a `.partial` sidecar, renamed only once its
/// SHA256 matches. Transient network errors (timeout, 5xx) get up to 3
/// tries with backoff; fatal ones (404, SHA mismatch, disk full, cancel)
/// fail fast.
pub(crate) fn download_to_file(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    use prunr_runtime_install::{retry_with_backoff, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS};
    retry_with_backoff(&cancel, DOWNLOAD_ATTEMPTS, DOWNLOAD_BACKOFF_BASE_MS, |_attempt| {
        download_attempt(url, dest, expected_sha, &cancel, on_progress, on_verifying)
    })
}

fn download_attempt(
    url: &str,
    dest: &Path,
    expected_sha: &str,
    cancel: &Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    let partial = partial_path(dest);

    // Tear down any leftover partial from a previous aborted attempt so
    // the writer starts fresh.
    let _ = std::fs::remove_file(&partial);
    sweep_stale_partials(dest);

    let client = prunr_runtime_install::http_client(crate::runtime_install::USER_AGENT)
        .map_err(|e| DlError::fatal(format!("Could not start the download: {}", prunr_runtime_install::with_causes(&e))))?;

    let response = client.get(url).send().map_err(|e| DlError::request(&e))?;

    let status = response.status();
    if !status.is_success() {
        let msg = if status.is_server_error() { format!("The server answered {status}; try again later") } else { format!("The download was refused ({status})") };
        return Err(DlError::for_status(status, msg));
    }

    let total = response.content_length().unwrap_or(0);
    finish_download(&partial, dest, fetch_into(response, &partial, total, expected_sha, cancel, on_progress, on_verifying))
}

/// A fetched file moves into place; one that failed (a full disk, a
/// dropped connection, a cancel, a bad checksum) leaves no partial, which
/// on a full disk could be gigabytes keeping it full.
fn finish_download(partial: &Path, dest: &Path, fetched: Result<(), DlError>) -> Result<(), DlError> {
    let result = fetched.and_then(|()| std::fs::rename(partial, dest).map_err(|e| DlError::fatal(format!("Could not finalize file: {e}"))));
    if result.is_err() {
        let _ = std::fs::remove_file(partial);
    }
    result
}

/// Stream `response` into `partial`, hashing as it goes, and check the hash.
fn fetch_into(
    response: impl std::io::Read,
    partial: &Path,
    total: u64,
    expected_sha: &str,
    cancel: &Arc<AtomicBool>,
    on_progress: &dyn Fn(u64, u64),
    on_verifying: &dyn Fn(),
) -> Result<(), DlError> {
    let mut file = std::fs::File::create(partial)
        .map_err(|e| DlError::fatal(format!("Could not create {}: {e}", partial.display())))?;
    let digest = prunr_runtime_install::stream_hashed(
        response,
        &mut file,
        cancel,
        |e| DlError::fatal(format!("Write error: {e}")),
        |written| on_progress(written, total),
    )?;
    std::io::Write::flush(&mut file)
        .map_err(|e| DlError::fatal(format!("Flush error: {e}")))?;
    drop(file);

    on_verifying();
    let actual_sha = hex::encode(digest);
    if !actual_sha.eq_ignore_ascii_case(expected_sha) {
        return Err(DlError::fatal(format!(
            "Checksum mismatch — expected {expected_sha}, got {actual_sha}",
        )));
    }

    Ok(())
}

/// This process's own partial of `dest`: two instances fetching one
/// model would otherwise write into one file, and the one finishing first
/// would keep whatever the other made of it.
fn partial_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(format!("{PARTIAL}.{}", std::process::id()));
    PathBuf::from(s)
}

pub(crate) const PARTIAL: &str = ".partial";

/// Remove the partials of `dest` an earlier run left (quit mid-download
/// keeps gigabytes), sparing one a running instance is writing now: the
/// suffix names the process writing it, and one without (an older
/// version's) is no live download's.
fn sweep_stale_partials(dest: &Path) {
    let (Some(dir), Some(name)) = (dest.parent(), dest.file_name()) else { return };
    crate::fs_util::sweep_left_by_ended_runs(dir, PARTIAL, |of| of == name.as_encoded_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A download that cannot connect says why (the connection refused,
    /// a certificate not trusted), not only that the request failed.
    #[test]
    fn a_failed_download_names_its_cause() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let failed = download_attempt("http://127.0.0.1:1/model.onnx", &dir.path().join("model.onnx"), "00", &cancel, &|_, _| {}, &|| {});
        let why = failed.expect_err("nothing listens on port 1").message.to_lowercase();
        assert!(why.contains("refused"), "{why}");
    }

    /// A download that failed part-way leaves no partial file; one that
    /// finished moves into place.
    #[test]
    fn a_failed_download_leaves_no_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let (partial, dest) = (dir.path().join("model.onnx.partial"), dir.path().join("model.onnx"));
        std::fs::write(&partial, b"half").unwrap();
        assert!(finish_download(&partial, &dest, Err(DlError::fatal("Write error: no space"))).is_err());
        assert!(!partial.exists() && !dest.exists());
        std::fs::write(&partial, b"whole").unwrap();
        finish_download(&partial, &dest, Ok(())).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"whole");
    }

    /// Each process downloads into a partial of its own: two instances
    /// fetching one model wrote into one file, and the one finishing first
    /// kept whatever the other had made of it.
    #[test]
    fn each_process_downloads_into_its_own_partial() {
        let p = partial_path(Path::new("/tmp/big_lama.onnx"));
        assert_eq!(p, PathBuf::from(format!("/tmp/big_lama.onnx.partial.{}", std::process::id())));
    }

    /// A partial an earlier run left (quit mid-download: gigabytes kept)
    /// goes when the model is fetched again, however fresh: a retry right
    /// after a quit must not be refused for the space it holds. One a
    /// running instance writes stays however old (a laptop asleep for
    /// hours), and so do other models' files.
    #[test]
    fn a_partial_left_by_an_earlier_run_is_swept() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("big_lama.onnx");
        let old_time = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        let file = |name: String, old: bool| {
            let path = dir.path().join(name);
            std::fs::write(&path, b"half").unwrap();
            if old {
                std::fs::File::options().write(true).open(&path).unwrap().set_modified(old_time).unwrap();
            }
            path
        };
        let dead = crate::fs_util::a_free_pid();
        let live = std::process::id();
        let left = [file("big_lama.onnx.partial".into(), true), file(format!("big_lama.onnx.partial.{dead}"), false)];
        let kept = [file(format!("big_lama.onnx.partial.{live}"), true), file(format!("migan.onnx.partial.{dead}"), true)];
        sweep_stale_partials(&dest);
        assert!(left.iter().all(|p| !p.exists()), "an earlier run's partials go");
        assert!(kept.iter().all(|p| p.exists()), "a live download and other models' files stay");
    }
}
