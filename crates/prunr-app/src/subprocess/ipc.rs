//! Length-prefixed binary framing for subprocess IPC.
//!
//! Format: [4 bytes LE length][bincode payload]
//! Max message size: 64 MB (commands/events are small; image data goes via temp files).

use std::io::{self, Read, Write, BufReader, BufWriter};

const MAX_MESSAGE_SIZE: u32 = 64 * 1024 * 1024; // 64 MB

/// Write a single message: [4-byte LE length][bincode payload].
pub fn write_message<W: Write, T: serde::Serialize>(
    writer: &mut BufWriter<W>,
    msg: &T,
) -> io::Result<()> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

/// Read a single message: [4-byte LE length][bincode payload].
/// Returns `None` on clean EOF (subprocess exited).
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
    let mut payload = vec![0u8; len as usize];
    reader.read_exact(&mut payload)?;
    let (msg, _) = bincode::serde::decode_from_slice(&payload, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(msg))
}

/// Serialize f32 slice to raw little-endian bytes.
/// Used for tensor IPC temp files.
pub fn f32s_to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize raw little-endian bytes to f32 vec.
pub fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    enum TestMsg {
        Hello { name: String },
        Number(u64),
        Empty,
    }

    #[test]
    fn roundtrip_single_message() {
        let mut buf = Vec::new();
        let msg = TestMsg::Hello { name: "prunr".to_string() };
        {
            let mut writer = BufWriter::new(&mut buf);
            write_message(&mut writer, &msg).unwrap();
        }
        let mut reader = BufReader::new(Cursor::new(&buf));
        let recovered: Option<TestMsg> = read_message(&mut reader).unwrap();
        assert_eq!(recovered, Some(msg));
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
    fn eof_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut reader = BufReader::new(Cursor::new(&buf));
        let result: Option<TestMsg> = read_message(&mut reader).unwrap();
        assert!(result.is_none());
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
        let bytes = f32s_to_le_bytes(&data);
        assert_eq!(bytes.len(), data.len() * 4);
        let recovered = le_bytes_to_f32s(&bytes);
        assert_eq!(recovered, data);
    }
}
