//! Failures in a person's words. A system, driver or model failure's own
//! text ("Non-zero status code returned while running Conv node", "No
//! space left on device (os error 28)") becomes one plain sentence by its
//! kind, its text kept in the log; Prunr's own messages pass through.

/// What went wrong, as far as a person needs to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    CardMemory,
    Memory,
    CardStopped,
    NoPermission,
    DiskFull,
    FileGone,
    Unreadable,
    Offline,
    ModelFailed,
}

const NO_PERMISSION: &[&str] = &["(os error 13)", "(os error 5)", "permission denied", "access is denied"];
const DISK_FULL: &[&str] = &[
    "(os error 28)", "(os error 39)", "(os error 112)", "(os error 122)", "(os error 27)",
    "no space left on device", "not enough space on the disk", "disk quota exceeded", "file too large",
];
const MEMORY: &[&str] = &["memory limit exceeded"];
const FILE_GONE: &[&str] = &["(os error 2)", "(os error 3)", "no such file or directory", "cannot find the file specified", "cannot find the path specified"];
const UNREADABLE: &[&str] = &[
    crate::types::IMAGE_ENDS_EARLY, "format error decoding", "image format could not be determined", "the image format",
    "does not support the format features", "svg parse error",
];
const OFFLINE: &[&str] = &[
    "error sending request", "dns error", "failed to lookup address", "connection refused", "connection reset",
    "network is unreachable", "operation timed out", "tls handshake", "invalid certificate", "certificate verify failed",
];
const MODEL_FAILED: &[&str] = &[
    "non-zero status code returned", "[onnxruntimeerror]", "ort session creation failed", "status message:",
    "invalid_graph", "invalid_protobuf", "failed to load model",
];

impl Kind {
    /// The kind of failure `raw` reports; `None` for a message Prunr wrote,
    /// one of these sentences included, so a text already plain stays.
    pub fn of(raw: &str) -> Option<Self> {
        if Self::ALL.iter().any(|kind| raw.contains(kind.sentence())) {
            return None;
        }
        let lower = raw.to_ascii_lowercase();
        let has = |marks: &[&str]| marks.iter().any(|mark| lower.contains(mark));
        if crate::ep_compat::is_device_lost(raw) {
            return Some(Kind::CardStopped);
        }
        if crate::ep_compat::is_out_of_memory(raw) {
            return Some(if crate::ep_compat::is_card_out_of_memory(raw) { Kind::CardMemory } else { Kind::Memory });
        }
        const BY_TEXT: [(Kind, &[&str]); 7] = [
            (Kind::Memory, MEMORY),
            (Kind::NoPermission, NO_PERMISSION),
            (Kind::DiskFull, DISK_FULL),
            (Kind::FileGone, FILE_GONE),
            (Kind::Unreadable, UNREADABLE),
            (Kind::Offline, OFFLINE),
            (Kind::ModelFailed, MODEL_FAILED),
        ];
        BY_TEXT.iter().find(|(_, marks)| has(marks)).map(|&(kind, _)| kind)
    }

    const ALL: [Kind; 9] = [
        Kind::CardMemory, Kind::Memory, Kind::CardStopped, Kind::NoPermission, Kind::DiskFull,
        Kind::FileGone, Kind::Unreadable, Kind::Offline, Kind::ModelFailed,
    ];

    pub fn sentence(self) -> &'static str {
        match self {
            Kind::CardMemory => "The graphics card ran out of memory. Close other apps that use it, or try a smaller image.",
            Kind::Memory => "The computer ran out of memory. Close other apps, or try a smaller image.",
            Kind::CardStopped => "The graphics card stopped responding. Try again; if it keeps happening, update its driver or run on the processor (CPU only).",
            Kind::NoPermission => "Prunr is not allowed to open or save that file. Check its permissions, or choose another folder.",
            Kind::DiskFull => "The disk is full. Free some space and try again.",
            Kind::FileGone => "The file is not there any more: it may have been moved or deleted.",
            Kind::Unreadable => "The file is not an image Prunr can read, or it is damaged.",
            Kind::Offline => "Could not reach the server. Check the internet connection and try again.",
            Kind::ModelFailed => "The model stopped with an error. Try again; if it repeats, try another model.",
        }
    }
}

/// `raw` as a person reads it: a system, driver or model failure as the
/// sentence for its kind, its own text logged; anything else as it is.
pub fn error(raw: &str) -> &str {
    match Kind::of(raw) {
        Some(kind) => {
            tracing::warn!(detail = raw, ?kind, "a failure shown in plain words");
            kind.sentence()
        }
        None => raw,
    }
}

/// `failure` as a person reads it (`error`), owned.
pub fn reason(failure: impl std::fmt::Display) -> String {
    let raw = failure.to_string();
    match error(&raw) {
        shown if std::ptr::eq(shown, raw.as_str()) => raw,
        shown => shown.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_of_failure_reads_as_its_own_sentence() {
        let cases = [
            ("CUDA failure 2: out of memory", Kind::CardMemory),
            ("Non-zero status code returned while running Gather node. Status Message: CUDA error: out of memory", Kind::CardMemory),
            ("std::bad_alloc", Kind::Memory),
            ("DXGI_ERROR_DEVICE_REMOVED (887A0005)", Kind::CardStopped),
            ("Permission denied (os error 13)", Kind::NoPermission),
            ("Access is denied. (os error 5)", Kind::NoPermission),
            ("No space left on device (os error 28)", Kind::DiskFull),
            ("There is not enough space on the disk. (os error 112)", Kind::DiskFull),
            ("No such file or directory (os error 2)", Kind::FileGone),
            ("The system cannot find the path specified. (os error 3)", Kind::FileGone),
            ("Format error decoding Png: Invalid PNG signature.", Kind::Unreadable),
            ("The image format could not be determined", Kind::Unreadable),
            ("error sending request for url (https://example.com/model.onnx)", Kind::Offline),
            ("Non-zero status code returned while running Conv node. Name:'/conv1'", Kind::ModelFailed),
            ("ORT session creation failed (CUDA): invalid_graph", Kind::ModelFailed),
        ];
        for (raw, kind) in cases {
            assert_eq!(Kind::of(raw), Some(kind), "{raw}");
            assert_eq!(error(raw), kind.sentence(), "{raw}");
        }
    }

    /// Prunr's own messages already say what happened and what to do.
    #[test]
    fn prunrs_own_messages_pass_through() {
        for own in [
            crate::CoreError::TooLarge { width: 21000, height: 14000, need_mb: 13414, free_mb: 12493 }.message(),
            "This eraser needs 15.6 GB of free memory to load, 2.0 GB of it kept spare for other apps; 13.0 GB is free.".to_string(),
            "The graphics card has no room for this eraser, and on the processor it needs about 900 MB of free memory, 400 MB is free.".to_string(),
            "Eraser (Big-LaMa) is not installed".to_string(),
            "The model stopped unexpectedly. Try again; if it repeats, try another model.".to_string(),
            "HTTP status client error (404 Not Found) for url (https://example.com/a.onnx)".to_string(),
        ] {
            assert_eq!(Kind::of(&own), None, "{own}");
            assert_eq!(error(&own), own);
        }
    }

    #[test]
    fn a_reason_is_owned_and_plain() {
        let io = std::io::Error::from_raw_os_error(13);
        assert_eq!(reason(&io), Kind::NoPermission.sentence());
        assert_eq!(reason("Eraser (Big-LaMa) is not installed"), "Eraser (Big-LaMa) is not installed");
    }

    /// A line that already holds a plain sentence is left whole, so a
    /// second pass keeps what it says around it (a file's name).
    #[test]
    fn a_plain_sentence_is_not_rewritten_again() {
        for kind in Kind::ALL {
            let line = format!("photo.jpg: could not read: {}", kind.sentence());
            assert_eq!(error(&line), line, "{kind:?}");
        }
    }

    /// A short read is not taken for an image's fault: a download or a
    /// stored file can end early too.
    #[test]
    fn a_short_read_alone_is_not_an_unreadable_image() {
        assert_eq!(Kind::of("failed to fill whole buffer"), None);
        let truncated = image::ImageError::IoError(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        let error = crate::CoreError::from(truncated);
        assert_eq!(Kind::of(&error.to_string()), Some(Kind::Unreadable));
        assert_eq!(Kind::of(&error.message()), Some(Kind::Unreadable), "also without the prefix, as the CLI says it");
    }

    /// An image error is not an unreadable image by its prefix alone: a
    /// write that runs out of room or an allocation past the decoder's
    /// limit says what it is.
    #[test]
    fn an_image_error_reads_by_its_cause() {
        assert_eq!(Kind::of("Image format error: Disk quota exceeded (os error 122)"), Some(Kind::DiskFull));
        assert_eq!(Kind::of("Image format error: File too large (os error 27)"), Some(Kind::DiskFull));
        assert_eq!(Kind::of("Image format error: Memory limit exceeded"), Some(Kind::Memory));
        assert_eq!(Kind::of("Image format error: Format error decoding Png: CRC error"), Some(Kind::Unreadable));
    }

    /// A new kind joins `ALL`: this match fails to build until it is
    /// handled, and the count follows.
    #[test]
    fn every_kind_is_in_all() {
        let listed = |kind: Kind| match kind {
            Kind::CardMemory | Kind::Memory | Kind::CardStopped | Kind::NoPermission | Kind::DiskFull
            | Kind::FileGone | Kind::Unreadable | Kind::Offline | Kind::ModelFailed => Kind::ALL.contains(&kind),
        };
        assert!(Kind::ALL.iter().all(|&kind| listed(kind)));
        assert_eq!(Kind::ALL.len(), 9);
    }

    /// Ten-something os errors are not the two they start like.
    #[test]
    fn an_os_error_is_matched_by_its_whole_number() {
        assert_eq!(Kind::of("Read-only file system (os error 30)"), None);
        assert_eq!(Kind::of("Too many open files (os error 24)"), None);
    }
}
