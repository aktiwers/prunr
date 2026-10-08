//! Every integration suite that can create an ORT session must call
//! `test_common::ensure_ort_initialized` (directly or via `skip_if_no_ort`)
//! before doing so — see that helper's doc for the hang it prevents.
//! Pure source scan; needs no runtime.

use std::{fs, path::Path};

const SESSION_CREATORS: &[&str] = &[
    "OrtEngine::",
    "process_inpaint",
    "upscale_rgba",
    "upscale_two_pass",
    "infer_only(",
];
const INIT_MARKERS: &[&str] = &["skip_if_no_ort", "ensure_ort_initialized"];

#[test]
fn every_suite_that_builds_a_session_initialises_ort_first() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let this_file = Path::new(file!()).file_name().expect("file name");
    let mut offenders = Vec::new();
    for entry in fs::read_dir(&dir).expect("tests dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "rs") || path.file_name() == Some(this_file) {
            continue;
        }
        let src = fs::read_to_string(&path).expect("read test source");
        let creates = SESSION_CREATORS.iter().any(|m| src.contains(m));
        let inits = INIT_MARKERS.iter().any(|m| src.contains(m));
        if creates && !inits {
            offenders.push(path.file_name().expect("file name").to_string_lossy().into_owned());
        }
    }
    assert!(
        offenders.is_empty(),
        "suites that create ORT sessions without test_common::skip_if_no_ort / \
         ensure_ort_initialized (would deadlock without a runtime): {offenders:?}"
    );
}
