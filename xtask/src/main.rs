use sha2::{Digest, Sha256};
use std::io::Write;

struct ModelSpec {
    name: &'static str,
    url: &'static str,
    sha256: &'static str, // Empty string = bootstrap mode (skip verification, print hash)
}

// After first run, replace empty strings with the printed SHA256 values.
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "silueta.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        sha256: "",
    },
    ModelSpec {
        name: "u2net.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        sha256: "",
    },
];

fn main() -> anyhow::Result<()> {
    let task = std::env::args().nth(1).unwrap_or_default();
    match task.as_str() {
        "fetch-models" => fetch_models(),
        _ => {
            eprintln!("Usage: cargo xtask <task>");
            eprintln!("Tasks:");
            eprintln!("  fetch-models   Download and verify ONNX model files to models/");
            std::process::exit(1);
        }
    }
}

fn fetch_models() -> anyhow::Result<()> {
    std::fs::create_dir_all("models")?;
    let client = reqwest::blocking::Client::builder()
        .user_agent("bgprunr-xtask/0.1")
        .build()?;

    for spec in MODELS {
        let dest = std::path::Path::new("models").join(spec.name);

        if dest.exists() {
            println!("{}: exists, verifying checksum...", spec.name);
            let bytes = std::fs::read(&dest)?;
            let hash = hex::encode(Sha256::digest(&bytes));

            if spec.sha256.is_empty() {
                println!("  Computed SHA256: {hash}");
                println!("  IMPORTANT: Hardcode this in xtask/src/main.rs");
                continue;
            }

            if hash == spec.sha256 {
                println!("  OK (cached)");
                continue;
            }
            println!("  Checksum mismatch — re-downloading");
        } else {
            println!("{}: downloading from {}", spec.name, spec.url);
        }

        let response = client.get(spec.url).send()?;
        if !response.status().is_success() {
            anyhow::bail!(
                "HTTP {} downloading {}",
                response.status(),
                spec.name
            );
        }
        let bytes = response.bytes()?;
        let hash = hex::encode(Sha256::digest(&bytes));

        if spec.sha256.is_empty() {
            println!("  Computed SHA256: {hash}");
            println!(
                "  IMPORTANT: Hardcode this in xtask/src/main.rs as {} constant",
                spec.name
            );
        } else if hash != spec.sha256 {
            anyhow::bail!(
                "SHA256 mismatch for {}:\n  expected: {}\n  got:      {}",
                spec.name,
                spec.sha256,
                hash
            );
        }

        let mut file = std::fs::File::create(&dest)?;
        file.write_all(&bytes)?;
        println!("  Saved to {}", dest.display());
    }

    println!("\nDone. If any SHA256 values above say IMPORTANT, update xtask/src/main.rs.");
    Ok(())
}
