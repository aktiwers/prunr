# Phase 1: Workspace Scaffolding - Research

**Researched:** 2026-04-06
**Domain:** Cargo workspace setup, xtask model fetching, CI pipeline, model embedding, cargo-dist release tooling
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Models are NOT committed to git and NOT downloaded at runtime by end users
- Developer runs `cargo xtask fetch-models` once after cloning — downloads silueta + u2net from HuggingFace
- Models cached in `models/` directory (gitignored)
- SHA256 checksums hardcoded in xtask — verified on download
- At compile time, `include-bytes-zstd` embeds models into the binary (level 19 compression)
- `dev-models` feature flag loads from filesystem instead (avoids recompilation during development)
- `prunr-models` is an isolated crate so model embedding only recompiles when model files change
- GitHub Actions with native runners for all platforms
- macOS: macos-14 (arm64) + macos-13 (x86_64) native runners — no cross-compilation
- Linux: ubuntu-latest x86_64
- Windows: windows-latest x86_64
- Models cached via actions/cache keyed by SHA256 — download once, reuse across builds
- `cargo xtask fetch-models` runs in CI before build step
- cargo-dist for generating release workflow and per-platform binary artifacts
- Even Phase 1 should produce a placeholder binary artifact in CI (validates the pipeline)
- Single workspace version via `workspace.package.version` — all crates share one version
- Single binary architecture: `prunr` (no args) = GUI, `prunr remove ...` = CLI
- Workspace has: `prunr` (binary crate, name=`prunr-app`), `prunr-core` (lib), `prunr-models` (lib)
- Key traits defined in prunr-core: `InferenceEngine`, `ImageProcessor`
- Error handling: `thiserror` enums per crate with `#[from]` conversions

### Claude's Discretion
- Exact xtask implementation details (reqwest vs ureq for download)
- CI workflow file structure (single vs matrix)
- cargo-dist configuration specifics
- Placeholder binary content (can be minimal "hello world" that proves the build works)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DIST-01 | Application is distributed as a single self-contained binary per platform | Covered by single binary (prunr-app) + cargo-dist multi-platform artifact generation |
| DIST-02 | Both ONNX models (silueta + u2net) are embedded in the binary | Covered by include-bytes-zstd in isolated prunr-models crate with dev-models feature |
| DIST-03 | Binary runs on Linux x86_64, macOS x86_64 + aarch64, Windows x86_64 | Covered by GitHub Actions native runners on all four targets + cargo-dist target triples |
| DIST-04 | No runtime dependencies — user downloads one file and runs it | Covered by model embedding + ort download-binaries + static/copy-dylibs strategy for ORT DLL |
</phase_requirements>

---

## Summary

Phase 1 establishes the Cargo workspace that all subsequent phases build on. The work divides into four areas: (1) workspace layout and Cargo configuration, (2) the xtask binary for one-time model fetching, (3) GitHub Actions CI producing native builds for all four platform targets, and (4) model embedding via `include-bytes-zstd` in the isolated `prunr-models` crate. The placeholder binary need not do anything meaningful — it just proves `cargo build` works on every target.

The workspace pattern using `[workspace.package]` and `[workspace.dependencies]` for shared version and dependency pinning is stable since Rust 1.64 and is the right approach for a project with three crates sharing the same version number. The `cargo xtask` pattern (with a `.cargo/config.toml` alias) is the idiomatic way to write developer automation in Rust without relying on `build.rs` or Makefiles. cargo-dist 0.31.0 is the current release and handles the four target triples out of the box via `dist init`.

The most critical architectural decision to get right in Phase 1 is isolating `prunr-models` as its own crate with no other workspace crate as a dependency. If the model embedding and any source code are in the same compilation unit, every source edit triggers a ~170MB recompile of the model blob. Getting this isolation wrong costs every subsequent phase significant development-loop time.

**Primary recommendation:** Follow the exact workspace structure from ARCHITECTURE.md. Isolate model embedding in `prunr-models`. Use reqwest (blocking, rustls) for the xtask downloader. Configure cargo-dist via `[workspace.metadata.dist]` with the four standard target triples. Use `Swatinem/rust-cache` for Cargo artifact caching and `actions/cache` with a SHA256-keyed cache for model files.

---

## Standard Stack

### Core (Phase 1 only — this phase's direct dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `thiserror` | `2.0` | Per-crate typed error enums with `derive(Error)` | Stable derive macro; 609M downloads; v2.0 stable as of late 2024; locked decision |
| `include-bytes-zstd` | git (`daac-tools/include-bytes-zstd`) | Proc-macro that compresses a file at build time and decompresses at runtime | Only Rust proc-macro that provides this; avoids 170MB raw blob in the compiler's I/O |
| `cargo-dist` | `0.31.0` | Generates GitHub Actions release workflow + per-platform binary archives | Handles the four target triples and checksums; `dist init` is one command |
| `reqwest` (xtask only) | `0.12` blocking | HTTP download in xtask for model fetching | Simpler API than ureq 3.x for streaming downloads with progress; rustls avoids OpenSSL dep |
| `sha2` (xtask only) | `0.10` | SHA256 verification of downloaded model files | RustCrypto standard; implements SHA-256 correctly |

### Supporting (workspace-level, not yet used in Phase 1 binary but declared for later phases)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ort` | `=2.0.0-rc.12` | ONNX Runtime bindings | Phase 2 (inference engine) |
| `egui` + `eframe` | `0.34.1` | GUI framework | Phase 4 (GUI shell) |
| `clap` | `4.5` | CLI argument parsing | Phase 3 (CLI commands) |
| `rayon` | `1.11` | Batch parallelism | Phase 3+ |
| `image` | `0.25` | Image decode/encode | Phase 2 |

### Alternatives Considered (xtask downloader)

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `reqwest` blocking | `ureq` 3.x | ureq is lighter but has a less ergonomic streaming API for large files; reqwest's blocking feature is cleaner for a build tool context where async overhead is irrelevant |
| `reqwest` blocking | `curl` binding | curl adds a native dependency; reqwest with rustls is fully static |

### Installation

```bash
# Install cargo-dist (developer machine and CI)
cargo install cargo-dist --version 0.31.0

# xtask dependencies (in xtask/Cargo.toml — not workspace-level)
# reqwest = { version = "0.12", features = ["blocking", "rustls-tls"], default-features = false }
# sha2 = "0.10"
# hex = "0.4"
```

---

## Architecture Patterns

### Recommended Project Structure

```
prunr/                          # Git repo root
├── Cargo.toml                    # [workspace] — members, [workspace.package], [workspace.dependencies]
├── Cargo.lock                    # Committed (binary project)
├── .cargo/
│   └── config.toml               # [alias] xtask = "run --package xtask --"
├── crates/
│   ├── prunr-core/             # lib crate — inference pipeline, traits, types
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs         # InferenceEngine trait (stub in Phase 1)
│   │       └── types.rs          # Error enum (thiserror)
│   ├── prunr-models/           # lib crate — model embedding ONLY
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs            # include_bytes_zstd! or dev-models filesystem load
│   └── prunr-app/              # binary crate — placeholder main.rs
│       ├── Cargo.toml            # name = "prunr" (the binary name)
│       └── src/
│           └── main.rs           # Prints version string; proves build works
├── xtask/                        # Developer tooling — NOT a workspace member for distribution
│   ├── Cargo.toml                # [package] name = "xtask"
│   └── src/
│       └── main.rs               # fetch-models subcommand
├── models/                       # .gitignored — populated by `cargo xtask fetch-models`
│   ├── silueta.onnx
│   └── u2net.onnx
├── .gitignore                    # models/ + target/
├── .github/
│   └── workflows/
│       ├── ci.yml                # PR/push CI: build + test on all platforms
│       └── release.yml           # Generated by cargo-dist; triggered on version tag
└── dist.toml                     # OR [workspace.metadata.dist] in root Cargo.toml
```

### Pattern 1: Cargo Workspace with Shared Version and Dependencies

**What:** All crates inherit version, edition, and common dependencies from `[workspace.package]` and `[workspace.dependencies]` in the root Cargo.toml. No version is duplicated across crate manifests.

**When to use:** Always — this is the only correct approach for a workspace where all crates share one version.

**Example:**
```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/prunr-core",
    "crates/prunr-models",
    "crates/prunr-app",
    "xtask",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourname/prunr"

[workspace.dependencies]
# Inference (Phase 2+)
ort = { version = "=2.0.0-rc.12", features = ["cuda", "coreml", "directml", "ndarray", "download-binaries"] }
ndarray = "0.16"
# GUI (Phase 4+)
egui = "0.34"
eframe = { version = "0.34", default-features = true }
# CLI (Phase 3+)
clap = { version = "4.5", features = ["derive"] }
# Error handling
thiserror = "2.0"
# Image (Phase 2+)
image = { version = "0.25", features = ["jpeg", "png", "webp", "bmp"] }
rayon = "1.11"
# Model embedding
include-bytes-zstd = { git = "https://github.com/daac-tools/include-bytes-zstd" }
```

```toml
# crates/prunr-core/Cargo.toml
[package]
name = "prunr-core"
version.workspace = true
edition.workspace = true

[dependencies]
prunr-models = { path = "../prunr-models" }
thiserror = { workspace = true }
```

```toml
# crates/prunr-app/Cargo.toml
[package]
name = "prunr-app"
version.workspace = true
edition.workspace = true

[[bin]]
name = "prunr"     # ← The distributed binary name
path = "src/main.rs"

[dependencies]
prunr-core = { path = "../prunr-core" }
```

### Pattern 2: xtask Alias

**What:** A `.cargo/config.toml` alias makes `cargo xtask <subcommand>` work from any directory in the workspace. The xtask binary is a normal Rust binary that reads `std::env::args()` to dispatch subcommands.

**When to use:** Any developer automation that must not run on every build. Model fetching, code generation, dist publishing.

**Example:**
```toml
# .cargo/config.toml
[alias]
xtask = "run --package xtask --"
```

```rust
// xtask/src/main.rs — minimal dispatch pattern
fn main() -> anyhow::Result<()> {
    let task = std::env::args().nth(1).unwrap_or_default();
    match task.as_str() {
        "fetch-models" => fetch_models(),
        _ => {
            eprintln!("Usage: cargo xtask <task>");
            eprintln!("Tasks: fetch-models");
            std::process::exit(1);
        }
    }
}
```

### Pattern 3: Model Embedding with dev-models Feature

**What:** `prunr-models/src/lib.rs` uses `cfg` guards to switch between compile-time embedding (default/release) and runtime filesystem loading (dev-models feature). The crate has NO other workspace crate as a dependency.

**When to use:** Always — this is the isolation pattern that prevents model recompilation on source edits.

**Example:**
```rust
// crates/prunr-models/src/lib.rs
// Source: ARCHITECTURE.md canonical pattern

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
            .join("../../models/silueta.onnx")
    )
    .expect("models/silueta.onnx not found — run `cargo xtask fetch-models`")
}

#[cfg(feature = "dev-models")]
pub fn u2net_bytes() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/u2net.onnx")
    )
    .expect("models/u2net.onnx not found — run `cargo xtask fetch-models`")
}
```

```toml
# crates/prunr-models/Cargo.toml
[package]
name = "prunr-models"
version.workspace = true
edition.workspace = true

[features]
default = []
dev-models = []

[build-dependencies]
include-bytes-zstd = { workspace = true }
```

### Pattern 4: GitHub Actions CI with Native Runners

**What:** A matrix workflow that builds natively on each target platform. No cross-compilation (avoids CoreML/Metal SDK issues on macOS). Each job installs Rust, runs `cargo xtask fetch-models`, then builds.

**When to use:** All CI runs — both CI (push/PR) and release workflows.

```yaml
# .github/workflows/ci.yml
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        target: x86_64-unknown-linux-gnu
      - os: macos-13
        target: x86_64-apple-darwin
      - os: macos-14
        target: aarch64-apple-darwin
      - os: windows-latest
        target: x86_64-pc-windows-msvc
```

### Pattern 5: cargo-dist Configuration

**What:** `dist init` generates `.github/workflows/release.yml` and adds `[workspace.metadata.dist]` to root Cargo.toml (or creates `dist.toml`). Configure targets to match the four platform targets.

**When to use:** Run once during Phase 1 setup; the generated workflow runs on every version tag push.

**Example configuration in root Cargo.toml:**
```toml
[workspace.metadata.dist]
cargo-dist-version = "0.31.0"
ci = "github"
installers = []
targets = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
]
```

### Anti-Patterns to Avoid

- **Model embedding in prunr-core or prunr-app:** Any source edit in these crates would trigger a 170MB model recompilation. The models crate MUST be isolated.
- **Using `build.rs` for model fetching:** build.rs runs on every build. xtask is explicit and one-time — the correct tool for optional, destructive developer setup.
- **Putting `xtask` in the workspace members list for distribution:** xtask is a developer tool. It should be a workspace member (so it can be run via `cargo xtask`) but it should NOT be a dist target. Mark it with `dist = false` in `[package.metadata.dist]` or simply exclude it from cargo-dist targets.
- **Using `include_bytes!` (without zstd) on u2net:** 170MB raw blob in a compilation unit — the compiler processes the entire blob during every build that touches the crate. The proc-macro approach compresses at build time and decompresses at runtime.
- **Committing Cargo.lock with `git = ...` dependencies unpinned:** The `include-bytes-zstd` git dependency must be pinned by Cargo.lock. Commit Cargo.lock.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-platform binary release pipeline | Custom shell scripts per platform | `cargo-dist` | Handles target triples, archives, checksums, GitHub Release creation, and the CI workflow generation |
| Build-time blob compression | Custom `build.rs` compress step | `include-bytes-zstd` proc-macro | The proc-macro runs at compile time and returns a `Vec<u8>` at runtime; far simpler than managing intermediate compressed files |
| HTTP download with progress + SHA256 | Manual socket code | `reqwest` (blocking) + `sha2` | reqwest handles TLS, redirects, chunked encoding; sha2 gives standard-compliant SHA-256 |
| Cargo workspace version management | Per-crate version fields | `workspace.package.version` inheritance | Single source of truth; `version.workspace = true` in each crate |
| Cargo build caching in CI | Custom caching logic | `Swatinem/rust-cache` action | Knows about workspace layout, Cargo.lock, toolchain hashes; smarter than `actions/cache` with manual key construction |

**Key insight:** The workspace scaffolding phase is primarily configuration, not code. The tools (cargo-dist, workspace inheritance, xtask pattern) are all well-established. The value is in applying them correctly the first time, not in building custom alternatives.

---

## Common Pitfalls

### Pitfall 1: Model Crate Not Truly Isolated
**What goes wrong:** `prunr-models` has `prunr-core` as a dependency, or `prunr-core` has inline `include_bytes_zstd!` calls. Result: every source edit in core triggers model recompilation. A developer changing a type in `types.rs` waits several minutes for Cargo to re-embed and re-link 170MB of compressed model data.

**Why it happens:** Convenience — it seems natural to put model access in the crate that uses the models. The isolation looks like extra indirection.

**How to avoid:** `prunr-models` must have ZERO dependencies on other workspace crates. Its only job is to vend model bytes. The dependency arrow goes: `prunr-app` → `prunr-core` → `prunr-models`. Never in reverse.

**Warning signs:** `cargo build --timings` shows `prunr-models` recompiling after a change to `engine.rs`.

### Pitfall 2: xtask Not Listed as Workspace Member
**What goes wrong:** `cargo xtask fetch-models` fails with "package `xtask` not found in workspace". The alias in `.cargo/config.toml` relies on `--package xtask`, which requires xtask to be in the workspace members list.

**Why it happens:** Assumption that xtask is separate from the workspace. It must be a member to use the cargo alias pattern.

**How to avoid:** Add `"xtask"` to the `members` array in root `Cargo.toml`. Exclude it from cargo-dist targets using `[package.metadata.dist] dist = false` in `xtask/Cargo.toml`.

### Pitfall 3: cargo-dist Releasing the xtask Binary
**What goes wrong:** cargo-dist sees `xtask` as a binary crate and produces a `xtask` artifact in GitHub Releases. End users can download developer tooling.

**Why it happens:** cargo-dist releases all packages with binaries by default.

**How to avoid:** Add to `xtask/Cargo.toml`:
```toml
[package.metadata.dist]
dist = false
```

### Pitfall 4: Model Cache Key in CI Not Tied to SHA256
**What goes wrong:** Model files are re-downloaded on every CI run, adding ~174MB of bandwidth and 2–5 minutes to each build. Or worse, a cached model from a previous SHA256 is used after the xtask checksum list is updated — xtask rejects the cached file and re-downloads it anyway, wasting the cache hit.

**Why it happens:** Using the run number or date as the cache key, or using a fixed key with no expiry.

**How to avoid:** Key the model cache on the hardcoded SHA256 values from the xtask source. When SHA256s change (model update), the cache key changes automatically and a fresh download occurs. When SHA256s are unchanged, the cache is hit.

```yaml
- name: Cache models
  uses: actions/cache@v4
  with:
    path: models/
    key: models-${{ hashFiles('xtask/src/main.rs') }}
```

Hashing the xtask source file is a practical proxy — the SHA256 constants live there, so any change to them changes the cache key.

### Pitfall 5: ORT `download-binaries` Not Working on Windows Due to Antivirus
**What goes wrong:** `cargo build` on Windows CI hangs or fails when `ort` tries to download the ONNX Runtime DLL via the `download-binaries` feature. Windows Defender or other security policies block the in-build download.

**Why it happens:** `download-binaries` fetches binaries at build time from pykeio's CDN. Corporate or restrictive Windows runners may block outbound connections during build.

**How to avoid:** In Phase 1, the placeholder binary does not use `ort` at all (no inference logic yet). Declare `ort` only in `prunr-core`'s `[dependencies]` and do not call any `ort` API from the placeholder binary. The linker will not try to link ORT until Phase 2 wires it up. This defers the ORT distribution problem to Phase 2 where it belongs.

**Warning signs:** CI hangs at `[ort] Downloading ONNX Runtime...` for more than 3 minutes.

### Pitfall 6: Missing `models/` Directory at Build Time
**What goes wrong:** `cargo build` fails with `include_bytes_zstd!("../../models/silueta.onnx", 19)` panicking because the file does not exist. This happens in CI if `cargo xtask fetch-models` was not run before the build step.

**Why it happens:** The proc-macro runs at compile time and requires the model file to exist on disk. CI starts with a clean checkout — models are gitignored.

**How to avoid:**
- In CI, always run `cargo xtask fetch-models` before any `cargo build` step
- For the default (non-dev-models) build in CI, ensure the cache restore step runs before the build step
- During local development with `--features dev-models`, the build succeeds even without model files (because the `cfg` branch does not call `include_bytes_zstd!`)

```yaml
# Correct CI step order:
- name: Restore model cache
  uses: actions/cache@v4
  with:
    path: models/
    key: models-${{ hashFiles('xtask/src/main.rs') }}

- name: Fetch models (if cache miss)
  if: steps.cache-models.outputs.cache-hit != 'true'
  run: cargo xtask fetch-models

- name: Build
  run: cargo build --release
```

---

## Code Examples

### Workspace Root Cargo.toml (complete Phase 1 version)

```toml
# Source: Cargo Book official docs on workspace.package + workspace.dependencies
[workspace]
members = [
    "crates/prunr-core",
    "crates/prunr-models",
    "crates/prunr-app",
    "xtask",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourname/prunr"

[workspace.dependencies]
# Phase 1 — active
thiserror = "2.0"
include-bytes-zstd = { git = "https://github.com/daac-tools/include-bytes-zstd" }

# Phase 2+ — declared now for consistent pinning
ort = { version = "=2.0.0-rc.12", features = ["cuda", "coreml", "directml", "ndarray", "download-binaries"] }
ndarray = "0.16"
image = { version = "0.25", features = ["jpeg", "png", "webp", "bmp"] }
rayon = "1.11"
egui = "0.34"
eframe = { version = "0.34", default-features = true }
egui_extras = { version = "0.34", features = ["all_loaders"] }
clap = { version = "4.5", features = ["derive"] }
arboard = { version = "3.6", features = ["wayland-data-control"] }
resvg = "0.47"
indicatif = "0.17"
rfd = "0.15"
zstd = "0.13"

[profile.release]
strip = true
lto = "thin"
opt-level = 3

[profile.dist]
inherits = "release"

[workspace.metadata.dist]
cargo-dist-version = "0.31.0"
ci = "github"
installers = []
targets = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
]
```

### SHA256-Verified Download in xtask

```rust
// xtask/src/main.rs — fetch-models subcommand
// Source: sha2 docs.rs + reqwest docs

use sha2::{Digest, Sha256};
use std::io::Write;

struct ModelSpec {
    name: &'static str,
    url: &'static str,
    sha256: &'static str,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "silueta.onnx",
        url: "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
        sha256: "<sha256-of-silueta>",   // Replace with actual hash
    },
    ModelSpec {
        name: "u2net.onnx",
        url: "https://huggingface.co/qualcomm/U2Net/resolve/main/u2net.onnx",
        sha256: "<sha256-of-u2net>",     // Replace with actual hash
    },
];

fn fetch_models() -> anyhow::Result<()> {
    std::fs::create_dir_all("models")?;
    let client = reqwest::blocking::Client::new();

    for spec in MODELS {
        let dest = std::path::Path::new("models").join(spec.name);
        if dest.exists() {
            println!("{} already exists, verifying checksum...", spec.name);
            let bytes = std::fs::read(&dest)?;
            let hash = format!("{:x}", Sha256::digest(&bytes));
            if hash == spec.sha256 {
                println!("  OK (cached)");
                continue;
            }
            println!("  Checksum mismatch — re-downloading");
        }

        println!("Downloading {}...", spec.name);
        let bytes = client.get(spec.url).send()?.bytes()?;
        let hash = format!("{:x}", Sha256::digest(&bytes));

        if hash != spec.sha256 {
            anyhow::bail!(
                "SHA256 mismatch for {}:\n  expected: {}\n  got:      {}",
                spec.name, spec.sha256, hash
            );
        }

        let mut file = std::fs::File::create(&dest)?;
        file.write_all(&bytes)?;
        println!("  Saved to {}", dest.display());
    }
    Ok(())
}
```

### Placeholder Binary (prunr-app/src/main.rs)

```rust
// crates/prunr-app/src/main.rs
// Minimal placeholder — proves the build works on all platforms

fn main() {
    println!(
        "prunr v{} — background removal tool (placeholder)",
        env!("CARGO_PKG_VERSION")
    );
    println!("Run `prunr --help` when the CLI is implemented.");
}
```

### Minimal Trait Stubs in prunr-core (Phase 1 skeleton only)

```rust
// crates/prunr-core/src/lib.rs
// Phase 1: define the trait interface; no implementation yet

pub mod engine;
pub mod types;

pub use engine::InferenceEngine;
pub use types::CoreError;
```

```rust
// crates/prunr-core/src/engine.rs
use crate::types::CoreError;

pub trait InferenceEngine: Send + Sync {
    fn active_provider(&self) -> &str;
    // Phase 2 will add: fn process(&self, ...) -> Result<...>
}
```

```rust
// crates/prunr-core/src/types.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Model error: {0}")]
    Model(String),
}
```

### GitHub Actions CI Workflow (ci.yml skeleton)

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    name: Build (${{ matrix.target }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-13
            target: x86_64-apple-darwin
          - os: macos-14
            target: aarch64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust stable
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Cache Cargo artifacts
        uses: Swatinem/rust-cache@v2
        with:
          key: ${{ matrix.target }}

      - name: Restore model cache
        id: cache-models
        uses: actions/cache@v4
        with:
          path: models/
          key: models-${{ hashFiles('xtask/src/main.rs') }}

      - name: Fetch models (cache miss)
        if: steps.cache-models.outputs.cache-hit != 'true'
        run: cargo xtask fetch-models

      - name: Build
        run: cargo build --target ${{ matrix.target }}

      - name: Test
        run: cargo test --target ${{ matrix.target }}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-crate `version = "x.y.z"` in each Cargo.toml | `version.workspace = true` with `[workspace.package]` | Stabilized Rust 1.64 (2022) | One place to bump version before release |
| Makefiles / shell scripts for dev automation | `cargo xtask` pattern | 2019+ (community convention) | Developer tooling is plain Rust, runs on all platforms, can use crates.io |
| Raw `include_bytes!` for binary blobs | `include-bytes-zstd` proc-macro | 2021+ | 170MB blob does not sit raw in the compiler's I/O; significant compile time reduction |
| cargo-release + manual script for releases | `cargo-dist` | 2022+ (v0.31.0 current 2026) | One command generates complete GitHub Actions release workflow |
| Shared dependency versions duplicated per crate | `[workspace.dependencies]` | Stabilized Rust 1.64 | Upgrade once at workspace root; all crates pick it up |
| `actions-rs` GitHub Actions for Rust CI | `dtolnay/rust-toolchain` + `Swatinem/rust-cache` | 2022 (actions-rs unmaintained) | `dtolnay/rust-toolchain` is maintained; `Swatinem/rust-cache` handles workspace layout correctly |

**Deprecated / outdated:**
- `actions-rs/toolchain` and `actions-rs/cargo`: Unmaintained since 2022. Replace with `dtolnay/rust-toolchain@stable`.
- `cargo-make`: Makefiles in TOML. Superseded by `cargo xtask` for this use case.
- `resolver = "1"` in Cargo workspace: Always use `resolver = "2"` for any workspace with platform-specific dependencies.

---

## Open Questions

1. **HuggingFace model URLs and SHA256 checksums**
   - What we know: silueta (~4MB) and u2net (~170MB) are the two target models; they come from HuggingFace
   - What's unclear: The exact download URLs and expected SHA256 hashes are not in any planning document
   - Recommendation: Look up the exact URLs and hash the files before hardcoding in xtask; these are load-bearing constants that must be correct before CI can verify model integrity

2. **include-bytes-zstd crate version vs git reference**
   - What we know: STACK.md recommends the git reference `{ git = "https://github.com/daac-tools/include-bytes-zstd" }`; a published crates.io version also exists
   - What's unclear: Whether the published crates.io version is current enough to use instead of the git reference
   - Recommendation: Use the crates.io published version if it works; the git reference is acceptable for now but pinning a specific git commit is safer for reproducibility

3. **cargo-dist: `dist.toml` vs `[workspace.metadata.dist]`**
   - What we know: cargo-dist 0.31.0 supports both configuration locations
   - What's unclear: The `dist init` command may prefer one over the other depending on the project type
   - Recommendation: Run `cargo dist init` interactively and let it choose; do not pre-create the configuration manually — `dist init` generates the correct format for the installed version

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`cargo test`) |
| Config file | none — standard `cargo test` |
| Quick run command | `cargo test` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIST-01 | Single binary artifact produced per platform | smoke (CI build check) | `cargo build --release --target <triple>` succeeds | ❌ Wave 0: CI workflow |
| DIST-02 | Models embedded in binary (or accessible via dev-models) | unit | `cargo test -p prunr-models -- test_model_bytes_accessible` | ❌ Wave 0 |
| DIST-03 | Binary builds on all 4 platform targets | smoke (CI matrix) | CI matrix job for each target succeeds | ❌ Wave 0: CI workflow |
| DIST-04 | No runtime deps: binary runs standalone | smoke (manual on clean VM) | `cargo build --release` produces binary; runs without installing anything | manual-only (Phase 6 full validation) |

**Note on DIST-04:** Full clean-VM validation is deferred to Phase 6 (distribution verification phase). In Phase 1, the CI smoke test on native runners (no pre-installed ORT, no model files outside the binary) serves as a proxy.

### Sampling Rate

- **Per task commit:** `cargo test --workspace`
- **Per wave merge:** Full CI matrix passes on all 4 targets
- **Phase gate:** CI matrix green + cargo-dist produces artifacts before moving to Phase 2

### Wave 0 Gaps

- [ ] `crates/prunr-models/src/lib.rs` — needs `#[cfg(test)] mod tests` block with `test_model_bytes_accessible` that loads bytes and asserts non-empty (run with `--features dev-models` so it doesn't embed 174MB during test)
- [ ] `.github/workflows/ci.yml` — matrix CI workflow (created during scaffolding tasks)
- [ ] `.github/workflows/release.yml` — generated by `cargo dist init` (created during scaffolding tasks)
- [ ] `xtask/src/main.rs` — `fetch-models` subcommand (created during scaffolding tasks)

---

## Sources

### Primary (HIGH confidence)

- [Cargo Book — Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) — workspace.package, workspace.dependencies, resolver = "2", member inheritance patterns
- [matklad/cargo-xtask GitHub](https://github.com/matklad/cargo-xtask) — xtask pattern setup, `.cargo/config.toml` alias, workspace membership
- [cargo-dist GitHub Releases](https://github.com/axodotdev/cargo-dist/releases) — v0.31.0 confirmed as February 23, 2026 release
- [cargo-dist Simple Guide](https://axodotdev.github.io/cargo-dist/book/workspaces/simple-guide.html) — `dist init` process, `[workspace.metadata.dist]`, target triples
- [cargo-dist More Complex Workspaces](https://axodotdev.github.io/cargo-dist/book/workspaces/workspace-guide.html) — library crate exclusion, singular vs unified announcements
- [daac-tools/include-bytes-zstd GitHub](https://github.com/daac-tools/include-bytes-zstd) — macro usage, compression level parameter, runtime decompression via ruzstd
- ARCHITECTURE.md (this project) — canonical workspace structure, model embedding pattern, crate dependency graph
- STACK.md (this project) — verified crate versions, include-bytes-zstd usage, ort pinning rationale
- PITFALLS.md (this project) — model crate isolation rationale, include_bytes compile time pitfall (Pitfall 7), Windows DLL concerns

### Secondary (MEDIUM confidence)

- [thiserror docs.rs](https://docs.rs/crate/thiserror/latest) — v2.0.18 confirmed as latest (2026-01-18); `derive(Error)` API unchanged
- [Swatinem/rust-cache GitHub](https://github.com/Swatinem/rust-cache) — cache key construction from Cargo.lock/toolchain hashes, workspace support
- [dtolnay/rust-toolchain GitHub](https://github.com/marketplace/actions/rust-toolchain) — replacement for unmaintained actions-rs; stable toolchain installation
- [actions/cache GitHub](https://github.com/actions/cache) — `actions/cache@v4` for model file caching with SHA256-based keys

### Tertiary (LOW confidence — flag for validation)

- [WebSearch: cross-platform Rust CI 2025](https://ahmedjama.com/blog/2025/12/cross-platform-rust-pipeline-github-actions/) — native macOS runner patterns; not verified against official GitHub docs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — thiserror, include-bytes-zstd, cargo-dist versions verified; xtask pattern from official matklad repo
- Architecture: HIGH — workspace structure from ARCHITECTURE.md canonical doc; workspace inheritance from Cargo Book
- Pitfalls: HIGH — model crate isolation from PITFALLS.md (Pitfall 7); xtask vs build.rs from CONTEXT.md decisions; ORT DLL from PITFALLS.md (Pitfall 6)
- CI patterns: MEDIUM — Swatinem/rust-cache and dtolnay/rust-toolchain verified as current best practice; native runner OS names (macos-13, macos-14) from CONTEXT.md decisions

**Research date:** 2026-04-06
**Valid until:** 2026-10-06 (stable domain; workspace and CI patterns change slowly)
