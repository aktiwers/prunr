// Production: models are embedded as pre-compressed zstd blobs and decompressed at runtime.
// Development: --features dev-models loads from the filesystem so model changes
// do not trigger recompilation of the prunr-models crate.
//
// IMPORTANT: This crate has NO dependencies on other workspace crates.
// The dependency arrow is prunr-app -> prunr-core -> prunr-models (never in reverse).

/// Embedded pre-compressed model data (plain include_bytes — no compile-time decompression).
#[cfg(not(feature = "dev-models"))]
static SILUETA_ZST: &[u8] = include_bytes!("../../../models/silueta.onnx.zst");
#[cfg(not(feature = "dev-models"))]
static U2NET_ZST: &[u8] = include_bytes!("../../../models/u2net.onnx.zst");

/// Load Silueta model bytes. In dev-models mode reads from filesystem;
/// in release mode decompresses the embedded zstd blob (~200ms).
#[cfg(not(feature = "dev-models"))]
pub fn silueta_bytes() -> Vec<u8> {
    zstd::bulk::decompress(SILUETA_ZST, 50 * 1024 * 1024)
        .expect("failed to decompress embedded silueta model")
}

/// Load U2Net model bytes. In dev-models mode reads from filesystem;
/// in release mode decompresses the embedded zstd blob (~200ms).
#[cfg(not(feature = "dev-models"))]
pub fn u2net_bytes() -> Vec<u8> {
    zstd::bulk::decompress(U2NET_ZST, 200 * 1024 * 1024)
        .expect("failed to decompress embedded u2net model")
}

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
            // Verify function API works in release mode too.
            let _: fn() -> Vec<u8> = super::silueta_bytes;
            let _: fn() -> Vec<u8> = super::u2net_bytes;
        }
    }
}
