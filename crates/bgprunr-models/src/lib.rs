// Production: models are zstd-compressed at build time and decompressed at startup.
// Development: --features dev-models loads from the filesystem so model changes
// do not trigger recompilation of the bgprunr-models crate.
//
// IMPORTANT: This crate has NO dependencies on other workspace crates.
// The dependency arrow is bgprunr-app -> bgprunr-core -> bgprunr-models (never in reverse).

#[cfg(not(feature = "dev-models"))]
pub static SILUETA_BYTES: &[u8] =
    include_bytes_zstd::include_bytes_zstd!("../../models/silueta.onnx", 19);

#[cfg(not(feature = "dev-models"))]
pub static U2NET_BYTES: &[u8] =
    include_bytes_zstd::include_bytes_zstd!("../../models/u2net.onnx", 19);

#[cfg(feature = "dev-models")]
pub fn silueta_bytes() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/silueta.onnx"),
    )
    .expect("models/silueta.onnx not found — run `cargo xtask fetch-models`")
}

#[cfg(feature = "dev-models")]
pub fn u2net_bytes() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/u2net.onnx"),
    )
    .expect("models/u2net.onnx not found — run `cargo xtask fetch-models`")
}

#[cfg(test)]
mod tests {
    // Tests run with --features dev-models to avoid embedding 174MB during testing.
    // Compile-time test: verifies the public API surface compiles correctly.

    #[test]
    fn test_model_api_compiles() {
        // When dev-models feature is active, these functions must exist and be callable.
        // Actual file I/O is NOT tested here (models may not be downloaded yet).
        // This test simply verifies the module compiles and the public API is accessible.
        #[cfg(feature = "dev-models")]
        {
            // Verify function signatures are correct (fn() -> Vec<u8>).
            let _: fn() -> Vec<u8> = super::silueta_bytes;
            let _: fn() -> Vec<u8> = super::u2net_bytes;
        }
        #[cfg(not(feature = "dev-models"))]
        {
            // Verify static refs are accessible.
            let _: &[u8] = super::SILUETA_BYTES;
            let _: &[u8] = super::U2NET_BYTES;
        }
    }
}
