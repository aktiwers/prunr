//! Persist undo/redo history entries to disk as zstd-compressed raw RGBA.
//!
//! This keeps RAM usage bounded during large batch processing while
//! preserving Ctrl+Z / Ctrl+Y functionality. Files are written to the
//! platform cache directory (`~/.cache/prunr/history/` on Linux,
//! `%LOCALAPPDATA%/prunr/history/` on Windows, `~/Library/Caches/prunr/history/`
//! on macOS) to avoid tmpfs-backed `/tmp` on Linux.
//!
//! Temp file lifecycle mirrors `drag_export.rs`:
//! - Stale cleanup (>30 min old) at app start.
//! - Full cleanup on graceful exit.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

const HISTORY_SUBDIR: &str = "prunr-history";
const STALE_AGE: Duration = Duration::from_secs(30 * 60);
const ZSTD_LEVEL: i32 = 1; // fastest compression, ~3:1 on RGBA data

/// A reference to a history entry stored on disk.
pub struct DiskHistoryEntry {
    pub path: PathBuf,
}

/// An RGBA image compressed in RAM (Tier 2: warm cache).
/// ~3-4x smaller than raw RGBA, ~8ms to decompress.
pub struct CompressedEntry {
    pub data: Vec<u8>, // zstd frame containing [u32 w][u32 h][pixels]
}

/// Compress an RgbaImage to a zstd buffer in RAM.
pub fn compress_to_ram(rgba: &image::RgbaImage) -> std::io::Result<CompressedEntry> {
    let (w, h) = (rgba.width(), rgba.height());
    let mut raw = Vec::with_capacity(8 + rgba.as_raw().len() / 3);
    let mut encoder = zstd::Encoder::new(&mut raw, ZSTD_LEVEL)?;
    encoder.write_all(&w.to_le_bytes())?;
    encoder.write_all(&h.to_le_bytes())?;
    encoder.write_all(rgba.as_raw())?;
    encoder.finish()?;
    Ok(CompressedEntry { data: raw })
}

/// Decompress an in-RAM compressed entry back to an RgbaImage.
pub fn decompress_from_ram(entry: &CompressedEntry) -> std::io::Result<image::RgbaImage> {
    let mut decoder = zstd::Decoder::new(entry.data.as_slice())?;
    let mut header = [0u8; 8];
    decoder.read_exact(&mut header)?;
    let w = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let h = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let mut pixels = Vec::new();
    decoder.read_to_end(&mut pixels)?;
    image::RgbaImage::from_raw(w, h, pixels).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pixel data does not match dimensions",
        )
    })
}

/// Write a CompressedEntry to disk (demote Tier 2 → Tier 3).
pub fn demote_to_disk(entry: &CompressedEntry, item_id: u64, seq: usize) -> std::io::Result<DiskHistoryEntry> {
    let path = cache_dir().join(format!("{item_id}_{seq}.zraw"));
    std::fs::write(&path, &entry.data)?;
    Ok(DiskHistoryEntry { path })
}

/// Return (and create if needed) the history cache directory.
/// Cached after first call to avoid repeated getenv + stat syscalls.
fn cache_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join(HISTORY_SUBDIR);
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
}

/// Write an RgbaImage to disk as zstd-compressed raw RGBA.
/// Format: [u32 LE width][u32 LE height][zstd-compressed RGBA pixels]
/// Uses streaming compression to avoid an intermediate Vec allocation.
pub fn write_history(
    item_id: u64,
    seq: usize,
    rgba: &image::RgbaImage,
) -> std::io::Result<DiskHistoryEntry> {
    let (w, h) = (rgba.width(), rgba.height());
    let path = cache_dir().join(format!("{item_id}_{seq}.zraw"));

    let file = std::fs::File::create(&path)?;
    let mut encoder = zstd::Encoder::new(file, ZSTD_LEVEL)?;
    // Write dimensions as header before compressed data.
    // The encoder wraps the file, so header goes into the zstd stream.
    // We decode dimensions from the stream on read.
    encoder.write_all(&w.to_le_bytes())?;
    encoder.write_all(&h.to_le_bytes())?;
    encoder.write_all(rgba.as_raw())?;
    encoder.finish()?;

    Ok(DiskHistoryEntry { path })
}

/// Read a history entry back from disk and decompress via streaming decoder.
pub fn read_history(entry: &DiskHistoryEntry) -> std::io::Result<image::RgbaImage> {
    let file = std::fs::File::open(&entry.path)?;
    let mut decoder = zstd::Decoder::new(file)?;

    let mut header = [0u8; 8];
    decoder.read_exact(&mut header)?;
    let w = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let h = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

    let mut pixels = Vec::new();
    decoder.read_to_end(&mut pixels)?;

    image::RgbaImage::from_raw(w, h, pixels).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pixel data does not match dimensions",
        )
    })
}

/// Delete a specific history file. Silent on error (best-effort).
pub fn delete_entry(entry: &DiskHistoryEntry) {
    let _ = std::fs::remove_file(&entry.path);
}

/// Remove stale history files left behind by previous sessions or crashes.
pub fn cleanup_stale() {
    let dir = cache_dir();
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }
        let Ok(meta) = entry.metadata() else { continue };
        let age = meta
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .unwrap_or(Duration::ZERO);
        if age > STALE_AGE {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Remove ALL history files. Called on graceful app exit.
pub fn cleanup_all() {
    std::thread::spawn(|| {
        let dir = cache_dir();
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let _ = std::fs::remove_file(&path);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_write_read() {
        let rgba = image::RgbaImage::from_pixel(100, 80, image::Rgba([255, 0, 128, 255]));
        let entry = write_history(99999, 0, &rgba).unwrap();
        assert!(entry.path.exists());
        let recovered = read_history(&entry).unwrap();
        assert_eq!(rgba.dimensions(), recovered.dimensions());
        assert_eq!(rgba.as_raw(), recovered.as_raw());
        delete_entry(&entry);
        assert!(!entry.path.exists());
    }

    #[test]
    fn cleanup_stale_does_not_panic_on_missing_dir() {
        cleanup_stale();
    }
}
