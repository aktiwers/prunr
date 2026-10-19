//! Length-prefixed binary framing for subprocess IPC.
//!
//! Format: [4 bytes LE length][bincode payload]
//! Max message size: 1 MB. Commands and events are tiny (≤ a few KB); image
//! data travels via temp files. If a frame exceeds this limit it is a bug
//! (e.g. an accidental inline payload), not a legitimate large message.

use std::io::{self, Read, Write, BufReader, BufWriter};

const MAX_MESSAGE_SIZE: u32 = 1024 * 1024; // 1 MB

/// Write a single message: [4-byte LE length][bincode payload].
///
/// Does NOT flush — callers in a write loop let the BufWriter coalesce
/// frames. Call `flush_writer` explicitly before the stream closes or
/// before a blocking read-response cycle where the peer must see the
/// frame before sending a reply.
pub fn write_message<W: Write, T: serde::Serialize>(
    writer: &mut BufWriter<W>,
    msg: &T,
) -> io::Result<()> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    Ok(())
}

/// Flush the writer. Call after the last `write_message` in a session so
/// any coalesced frames reach the peer before the pipe closes.
pub fn flush_writer<W: Write>(writer: &mut BufWriter<W>) -> io::Result<()> {
    writer.flush()
}

/// Read a single message: [4-byte LE length][bincode payload].
/// Returns `None` on clean EOF (subprocess exited).
///
/// Stream-decodes via `Read::take(len)` — no payload Vec, no memcpy.
pub fn read_message<R: Read, T: serde::de::DeserializeOwned>(
    reader: &mut BufReader<R>,
) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("IPC message too large: {len} bytes (max {MAX_MESSAGE_SIZE})"),
        ));
    }
    let mut limited = reader.take(len as u64);
    let msg: T = bincode::serde::decode_from_std_read(
        &mut limited,
        bincode::config::standard(),
    ).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(msg))
}

/// Read a one-shot IPC temp file and delete it in the same breath; a file
/// that cannot be read is left for the prefix-scoped cleanup.
pub fn read_and_delete(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let _ = std::fs::remove_file(path);
    Some(bytes)
}

/// Hand `image` to the other process as a file: its width and height
/// (u32 LE), then its raw pixels. No encode on this side, no decode on
/// the other. A write that fails leaves no partial file behind.
pub fn write_image<P: image::Pixel<Subpixel = u8>>(path: &std::path::Path, image: &image::ImageBuffer<P, Vec<u8>>) -> io::Result<()> {
    let write = || -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&image.width().to_le_bytes())?;
        file.write_all(&image.height().to_le_bytes())?;
        file.write_all(image.as_raw())
    };
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(path);
    })
}

/// An image `write_image` wrote, its file removed; `None` when it is
/// missing or short.
pub fn read_image<P: image::Pixel<Subpixel = u8>>(path: &std::path::Path) -> Option<image::ImageBuffer<P, Vec<u8>>> {
    let read = || -> io::Result<Option<image::ImageBuffer<P, Vec<u8>>>> {
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;
        let [w, h] = [0, 4].map(|at| u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]]));
        let mut pixels = Vec::with_capacity(w as usize * h as usize * usize::from(P::CHANNEL_COUNT));
        file.read_to_end(&mut pixels)?;
        Ok(image::ImageBuffer::from_raw(w, h, pixels))
    };
    let image = read().ok().flatten();
    let _ = std::fs::remove_file(path);
    image
}

/// A worker's raw RGBA result. Unlike `read_and_delete`, the file goes
/// even when it cannot be read or does not match `w`×`h`.
pub fn read_result_rgba(path: &std::path::Path, w: u32, h: u32) -> Option<image::RgbaImage> {
    let image = std::fs::read(path).ok().and_then(|data| image::RgbaImage::from_raw(w, h, data));
    let _ = std::fs::remove_file(path);
    image
}

/// View an f32 slice as raw bytes. Zero-copy on little-endian hosts (every
/// platform we target). The returned slice borrows from `data`; the caller
/// feeds it straight into `fs::write` / `zstd::encode_all` / similar
/// byte-sinks without an intermediate Vec.
pub fn f32s_as_le_bytes(data: &[f32]) -> &[u8] {
    // Safety (documented via bytemuck::Pod): f32 is POD; on little-endian
    // targets (x86_64, aarch64, wasm32) the in-memory layout is already LE.
    // Prunr ships only LE platforms — the conditional makes the big-endian
    // case a compile error rather than a silent wrong-byte bug.
    #[cfg(target_endian = "little")]
    { bytemuck::cast_slice(data) }
    #[cfg(not(target_endian = "little"))]
    { compile_error!("prunr IPC assumes little-endian f32 layout"); }
}

/// Read raw little-endian bytes back to a Vec<f32>. `pod_collect_to_vec`
/// does one memcpy regardless of input alignment — safer than
/// `cast_slice::<_, f32>` which would panic on an unaligned source.
pub fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    #[cfg(target_endian = "little")]
    { bytemuck::pod_collect_to_vec(bytes) }
    #[cfg(not(target_endian = "little"))]
    { compile_error!("prunr IPC assumes little-endian f32 layout"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn a_read_temp_file_is_gone_and_a_missing_one_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("result.raw");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        assert_eq!(read_and_delete(&path), Some(vec![1, 2, 3]));
        assert!(!path.exists());
        assert_eq!(read_and_delete(&path), None);
    }

    /// An image goes over as its size and pixels and comes back the
    /// same, colour or grey; the file is gone after, and a short one
    /// reads as nothing.
    #[test]
    fn an_image_file_round_trips_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("img.raw");
        let rgba = image::RgbaImage::from_fn(3, 2, |x, y| image::Rgba([x as u8, y as u8, 7, 200]));
        write_image(&path, &rgba).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 8 + 3 * 2 * 4);
        assert_eq!(read_image::<image::Rgba<u8>>(&path), Some(rgba));
        assert!(!path.exists());
        let gray = image::GrayImage::from_fn(2, 2, |x, y| image::Luma([(x + 2 * y) as u8 * 60]));
        write_image(&path, &gray).unwrap();
        assert_eq!(read_image::<image::Luma<u8>>(&path), Some(gray));
        std::fs::write(&path, [4u8, 0, 0, 0, 4, 0, 0, 0, 1, 2]).unwrap();
        assert_eq!(read_image::<image::Luma<u8>>(&path), None);
        assert!(!path.exists());
    }

    #[test]
    fn a_result_file_is_gone_whether_or_not_it_fits_its_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("result.raw");
        std::fs::write(&path, [1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let image = read_result_rgba(&path, 2, 1).expect("2x1 RGBA");
        assert_eq!(image.as_raw(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(!path.exists());

        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        assert!(read_result_rgba(&path, 2, 1).is_none());
        assert!(!path.exists(), "a short file is removed too");
        assert!(read_result_rgba(&path, 2, 1).is_none());
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    enum TestMsg {
        Hello { name: String },
        Number(u64),
        Empty,
    }

    #[test]
    fn roundtrip_multiple_messages() {
        let mut buf = Vec::new();
        let msgs = vec![
            TestMsg::Hello { name: "a".into() },
            TestMsg::Number(42),
            TestMsg::Empty,
        ];
        {
            let mut writer = BufWriter::new(&mut buf);
            for m in &msgs {
                write_message(&mut writer, m).unwrap();
            }
        }
        let mut reader = BufReader::new(Cursor::new(&buf));
        for expected in &msgs {
            let got: Option<TestMsg> = read_message(&mut reader).unwrap();
            assert_eq!(got.as_ref(), Some(expected));
        }
        // Next read should be EOF
        let eof: Option<TestMsg> = read_message(&mut reader).unwrap();
        assert!(eof.is_none());
    }

    #[test]
    fn oversize_frame_returns_error_not_panic() {
        // Craft a frame header claiming a payload > MAX_MESSAGE_SIZE. The
        // reader must reject it with InvalidData, not panic or try to allocate
        // a huge buffer.
        let bad_len: u32 = MAX_MESSAGE_SIZE + 1;
        let mut buf = bad_len.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 8]); // truncated payload — never reached
        let mut reader = BufReader::new(Cursor::new(buf));
        let result: io::Result<Option<TestMsg>> = read_message(&mut reader);
        let err = result.expect_err("oversize frame must be an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn f32_byte_roundtrip() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, std::f32::consts::PI, 1e-10, 1e10];
        let bytes = f32s_as_le_bytes(&data);
        assert_eq!(bytes.len(), data.len() * 4);
        let recovered = le_bytes_to_f32s(bytes);
        assert_eq!(recovered, data);
    }
}
