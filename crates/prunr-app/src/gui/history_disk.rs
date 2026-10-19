//! Undo payloads at rest: zstd framing for images, and the session's
//! directory where stored states spill under memory pressure. Files live
//! in the platform cache dir (`~/.cache/prunr/...` on Linux, avoiding a
//! tmpfs `/tmp`), one directory per process.
//!
//! Dead sessions' dirs are swept at start, this session's files on
//! graceful exit.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

const HISTORY_SUBDIR: &str = "prunr-history";
/// Age past which a loose file from a pre-per-process build is dropped.
const STALE_AGE: Duration = Duration::from_secs(30 * 60);
pub(super) const ZSTD_LEVEL: i32 = 1; // fastest compression

/// The `w, h` little-endian header every undo payload starts with.
pub(super) fn read_dims(header: &[u8; 8]) -> (u32, u32) {
    let [w0, w1, w2, w3, h0, h1, h2, h3] = *header;
    (u32::from_le_bytes([w0, w1, w2, w3]), u32::from_le_bytes([h0, h1, h2, h3]))
}

pub fn compress_to_ram(rgba: &image::RgbaImage) -> std::io::Result<Vec<u8>> {
    let (w, h) = (rgba.width(), rgba.height());
    // zstd-level-1 on natural-image RGBA hits ~2:1 in practice.
    // Right-sizing avoids 1–2 reallocs mid-encode on 4 K images.
    let mut raw = Vec::with_capacity(8 + rgba.as_raw().len() / 2 + 64);
    let mut encoder = zstd::Encoder::new(&mut raw, ZSTD_LEVEL)?;
    encoder.write_all(&w.to_le_bytes())?;
    encoder.write_all(&h.to_le_bytes())?;
    encoder.write_all(rgba.as_raw())?;
    encoder.finish()?;
    Ok(raw)
}

pub fn decompress_from_bytes(bytes: &[u8]) -> std::io::Result<image::RgbaImage> {
    let mut decoder = zstd::Decoder::new(bytes)?;
    let mut header = [0u8; 8];
    decoder.read_exact(&mut header)?;
    let (w, h) = read_dims(&header);
    // Pre-size: header carries `w, h` so we know the exact decompressed
    // size. Avoids ~25 doublings + a peak ~100 MB transient mid-grow at
    // 4 K (read_to_end starts at 0 and doubles).
    let expected = (w as usize) * (h as usize) * 4;
    let mut pixels = Vec::with_capacity(expected);
    decoder.read_to_end(&mut pixels)?;
    image::RgbaImage::from_raw(w, h, pixels).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pixel data does not match dimensions",
        )
    })
}

/// Write an already-compressed undo payload to its own file in this
/// session's history directory.
pub fn write_blob(bytes: &[u8]) -> std::io::Result<PathBuf> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = cache_dir().join(format!("state_{n}.zst"));
    crate::fs_util::write_bytes_replacing(&path, bytes)?;
    Ok(path)
}

fn history_root() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join(HISTORY_SUBDIR)
}

/// This process's history directory, `prunr-history/{pid}/`, created on
/// first use. Per process, so two running instances never share, sweep
/// or overwrite each other's files (names are only a per-process sequence).
fn cache_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = history_root().join(std::process::id().to_string());
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
}

/// Startup housekeeping: drop the history dirs of instances that are no
/// longer running, and loose files from builds that predate per-process
/// dirs. Never touches a live session's files: they back undo steps for
/// as long as the session runs, however old they are.
pub fn cleanup_stale() {
    let root = history_root();
    crate::fs_util::sweep_dead_pid_dirs(&root, "", std::process::id());
    let Ok(entries) = std::fs::read_dir(&root) else { return };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        if !entry.file_type().ok().is_some_and(|t| t.is_file()) { continue; }
        let age = entry.metadata().ok()
            .and_then(|m| m.modified().ok())
            .and_then(|m| now.duration_since(m).ok())
            .unwrap_or(Duration::ZERO);
        if age > STALE_AGE {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Empty this session's history dir on graceful exit. The dir itself
/// stays (later writes in this process rely on it) and is removed by the
/// next startup's sweep once this process is gone.
pub fn cleanup_all() {
    crate::fs_util::sweep_dir_files_async(cache_dir().clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session's own spilled states are in use for as long as the
    /// session runs, however old the file: the sweep must leave them.
    #[test]
    fn the_sweep_keeps_this_session_s_history_however_old() {
        let path = write_blob(b"state").unwrap();
        let hours_ago = SystemTime::now() - Duration::from_secs(2 * 3600);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(hours_ago).unwrap();
        cleanup_stale();
        assert!(path.exists(), "an undo step still pointing at this file would be lost");
        assert!(path.parent().unwrap().ends_with(std::process::id().to_string()), "files live in a per-process dir");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cleanup_stale_does_not_panic_on_missing_dir() {
        cleanup_stale();
    }

    #[test]
    fn ram_roundtrip_preserves_pixels() {
        // Non-trivial pattern so any off-by-one in the header/pixel split
        // shows up as a mismatch rather than a clean black image.
        let rgba = image::RgbaImage::from_fn(50, 40, |x, y| {
            image::Rgba([x as u8, y as u8, ((x + y) & 0xFF) as u8, 255])
        });
        let recovered = decompress_from_bytes(&compress_to_ram(&rgba).unwrap()).unwrap();
        assert_eq!(rgba.dimensions(), recovered.dimensions());
        assert_eq!(rgba.as_raw(), recovered.as_raw());
    }

    #[test]
    fn compress_decompress_roundtrip_1x1() {
        let rgba = image::RgbaImage::from_pixel(1, 1, image::Rgba([42, 17, 200, 99]));
        let recovered = decompress_from_bytes(&compress_to_ram(&rgba).unwrap()).unwrap();
        assert_eq!(rgba.as_raw(), recovered.as_raw());
    }
}
