//! The runtime resolver lives in `prunr_core::ort_runtime` so every
//! session site in the workspace shares one init; re-exported here for
//! the CLI and Settings callers.

pub use prunr_core::ort_runtime::*;
