# Stack Research

**Domain:** Pure Rust desktop application — local AI inference + native GUI (Linux/macOS/Windows)
**Researched:** 2026-04-06
**Confidence:** HIGH (core choices verified against crates.io, official docs, and multiple sources)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `ort` | `2.0.0-rc.12` | ONNX Runtime Rust bindings for model inference | Only mature Rust crate giving direct access to ONNX Runtime 1.24 with GPU execution providers (CUDA, CoreML, DirectML). Wraps Microsoft's battle-tested runtime — the same engine rembg uses. rc.12 is production-ready per community usage; stable 2.0.0 final not yet released as of 2026-04-06. |
| `egui` + `eframe` | `0.34.1` | Immediate-mode GUI + native desktop harness | Best fit for image tools: built-in TextureHandle for GPU-uploaded bitmaps, native wgpu renderer, file drag-and-drop via `ViewportBuilder::with_drag_and_drop(true)`, cross-platform windowing via winit. Immediate-mode eliminates manual invalidation complexity for image preview states. |
| `image` | `0.25.8` | PNG/JPEG/WebP/BMP decode and encode | The standard Rust imaging library (82M downloads). Provides `DynamicImage` — the central type flowing through the inference pipeline. Handles all required input formats and PNG+alpha output. |
| `clap` | `4.5.x` | CLI argument parsing | De-facto standard for Rust CLIs. Derive macro makes argument structs self-documenting. v4 is stable and actively maintained; no reason to consider alternatives. |
| `rayon` | `1.11.0` | Data parallelism for batch image processing | Work-stealing thread pool with `.par_iter()` as a drop-in for `.iter()`. Zero configuration, proven in production (266M downloads). Batch processing across images needs no custom executor. |
| `resvg` | `0.47.0` | SVG rasterization | Pure Rust, zero native dependencies in the final binary. Produces a pixmap raster you can hand directly to `image::DynamicImage`. Ships as a sub-library `usvg` for parsing; `resvg` for rendering. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ndarray` | `0.16.1` | N-dimensional arrays for inference preprocessing | Required when constructing the 4D tensor `[1, 3, 320, 320]` fed to ONNX Runtime. ort's `ndarray` feature bridges `Array` types directly into session inputs. |
| `arboard` | `3.6.1` | Cross-platform clipboard: copy result image | Supports image data (`set_image`) as well as text on Linux (X11/Wayland), macOS, and Windows. Enable `wayland-data-control` feature for Wayland correctness in 2025+. |
| `egui_extras` | `0.34.1` (same release as egui) | Image loading loaders for egui | Provides `install_image_loaders()` which registers decoders for PNG/JPEG/WebP into egui's texture system. Use `features = ["all_loaders"]`. |
| `zstd` | `0.13.3` | Runtime zstd decompression for embedded models | Used to decompress the u2net model (~170MB raw → ~130MB zstd) at startup. Pair with `include-bytes-zstd` for the build-time compression macro. |
| `include-bytes-zstd` | latest (`daac-tools`) | Embed zstd-compressed bytes at compile time | Proc-macro replacement for `include_bytes!` that compresses the file at build time and decompresses at runtime via ruzstd. Reduces binary size without a separate asset distribution step. |
| `tokio` | not needed | — | **Do not add.** The inference pipeline is CPU/GPU-bound and fully synchronous. `rayon` covers all parallelism needs. Adding an async runtime bloats the binary and complicates the architecture for zero gain. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo` workspace | Shared core between `bgprunr-cli` and `bgprunr-gui` binaries | Workspace `Cargo.toml` with `[workspace.dependencies]` for shared version pinning (Rust 2024 edition resolver v3 via rustc 1.84+). Structure: `crates/core` (lib), `crates/cli` (bin), `crates/gui` (bin). |
| `cargo-cross` | Cross-compilation via Docker | For building Linux → Windows or Linux → macOS (note: Apple SDK licensing means macOS cross-compilation requires `osxcross` image). Use native CI runners per platform instead for release builds to avoid SDK issues. |
| GitHub Actions (or similar CI) | Per-platform release builds | Run native Linux/macOS/Windows runners in CI. Native builds avoid cross-compilation SDK headaches, especially for macOS. Produce three binaries for release. |
| `cargo-bloat` | Binary size analysis | Diagnose what's contributing to binary size. Useful when fine-tuning the embedded model compression strategy. |
| `cargo-dist` | Release packaging | Automates multi-platform binary archives, checksums, and GitHub Releases. Optional but saves significant release boilerplate. |

---

## Cargo.toml Dependency Block

```toml
[workspace.dependencies]
# Inference
ort = { version = "=2.0.0-rc.12", features = ["cuda", "coreml", "directml", "ndarray", "download-binaries"] }
ndarray = "0.16"

# GUI
egui = "0.34"
eframe = { version = "0.34", default-features = true }
egui_extras = { version = "0.34", features = ["all_loaders"] }

# Image I/O
image = { version = "0.25", features = ["jpeg", "png", "webp", "bmp"] }
resvg = "0.47"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Parallelism
rayon = "1.11"

# Clipboard
arboard = { version = "3.6", features = ["wayland-data-control"] }

# Model embedding / compression
include-bytes-zstd = { git = "https://github.com/daac-tools/include-bytes-zstd" }
zstd = "0.13"
```

**Note on `ort` version pinning:** Use `=2.0.0-rc.12` (exact pin) rather than `"2.0"` because Cargo does not resolve pre-release versions automatically. When 2.0.0 stable releases, change to `"2"`. The `download-binaries` feature fetches prebuilt ONNX Runtime 1.24 shared libraries at build time — no system-level ORT install required.

**Note on GPU features:** `cuda` applies to Linux/Windows builds; `coreml` applies to macOS. Including all three features is harmless — ort only activates what the host platform supports. Disable GPU features in the `core` crate and enable only in the `gui`/`cli` crates via feature flags if you want tighter control.

---

## Alternatives Considered

| Recommended | Alternative | Why Not / When Alternative Is Better |
|-------------|-------------|--------------------------------------|
| `ort` 2.0-rc | `tract` 0.21 | tract is pure Rust (zero native deps) but lacks GPU support. U2-Net/silueta inference is ~5-10x slower on CPU-only vs ORT+GPU. Use tract only if you need a zero-dependency build with no GPU requirement. |
| `ort` 2.0-rc | `candle` (HuggingFace) | candle is pure Rust with CUDA support, but targets HF model formats. ONNX compatibility requires conversion; rembg model parity not guaranteed. Use candle if you want to experiment with HF-native models in future. |
| `egui` 0.34 | `iced` 0.13 | iced uses Elm architecture — well structured but retained-mode means more boilerplate for image tool state (zoom level, mask visibility, before/after toggle). No accessibility support as of April 2025 survey. Use iced if your app grows into a document editor with complex undo/redo. |
| `egui` 0.34 | `Slint` | Slint has a DSL build step and requires a commercial license for closed-source distribution. Accessibility is better than egui, but the DSL adds tooling complexity. Use Slint for commercial apps requiring screen reader support. |
| `egui` 0.34 | `Tauri` | Tauri embeds a web renderer — violates the "pure native Rust" constraint and adds 50-80MB webview dependency. Not suitable. |
| `rayon` | `tokio` + `async` | Async is designed for I/O-bound concurrency. Inference is CPU/GPU-bound. `rayon` parallelises work-stealing across available cores with no runtime overhead. `tokio` adds complexity for no benefit here. |
| `arboard` | `copypasta` | `copypasta` lacks image support and is less actively maintained. `arboard` (1Password-maintained) supports image data on all three platforms. |
| `include-bytes-zstd` | raw `include_bytes!` + runtime decompress | `include_bytes!` on a 170MB file causes extreme compile-time memory pressure (the compiler reads the whole blob into memory). The proc-macro approach compresses at build time and decompresses lazily at runtime. |
| `zstd` | `lz4` / `brotli` | zstd gives the best compression ratio for binary model data (ONNX weights are partially compressible). lz4 is faster to decompress but much larger output; brotli is similar ratio but slower decompression. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `onnxruntime` crate (crates.io) | Unmaintained wrapper targeting ORT 1.8 (2021). No GPU support, no 2.x API. | `ort` 2.0-rc.12 |
| `wonnx` | WebGPU-only ONNX runtime in pure Rust. No CUDA, no CoreML, no CPU fallback for arbitrary models. U2-Net not validated. | `ort` |
| `rembg-rs` crate | Thin wrapper with limited maintenance. Rolling your own inference pipeline via `ort` directly gives full control over preprocessing, model switching, and batch parallelism — which BgPrunR requires. | Direct `ort` + `ndarray` pipeline |
| Global `ORT_DYLIB_PATH` env var at runtime | Breaks the single-binary distribution goal. Users should not need to set environment variables. | `download-binaries` feature at build time; libraries bundled via `copy-dylibs` feature or static linking |
| `glow` backend in eframe | OpenGL/glow is now opt-in in eframe 0.34. wgpu (default) supports Vulkan/Metal/DX12/WebGPU, giving better GPU texture upload performance for displaying inference output. | Default eframe (wgpu) |
| Async everywhere (`tokio`) | Inference is synchronous and CPU/GPU-bound. Async overhead and complexity brings no benefit. Rayon covers batch parallelism. | `rayon` |

---

## Stack Patterns by Variant

**GPU build (default for distribution):**
- Enable `cuda` + `coreml` + `directml` features on `ort`
- ort downloads and links ONNX Runtime 1.24 at build time
- Runtime selects CUDA (Linux/Windows with NVIDIA), CoreML (macOS), DirectML (Windows Intel/AMD) in priority order, falls back to CPU

**CPU-only build (CI testing, minimal binary):**
- Disable GPU features: `ort = { version = "=2.0.0-rc.12", default-features = false, features = ["ndarray", "download-binaries"] }`
- Useful for CI smoke tests where GPU is unavailable

**Development build:**
- Use `ORT_STRATEGY=download` (default) — ort fetches prebuilt ONNX Runtime so developers need zero system-level setup
- Combine with `cargo run --bin bgprunr-gui`

**Release build:**
- Static link where possible, or bundle the ORT dylib alongside the binary via `copy-dylibs`
- Compress u2net model with `include-bytes-zstd`; silueta (~4MB) can use raw `include_bytes!`
- Strip debug symbols: `[profile.release] strip = true`
- Enable LTO for binary size: `lto = "thin"`

---

## Version Compatibility

| Crate A | Compatible With | Notes |
|---------|-----------------|-------|
| `ort =2.0.0-rc.12` | `ndarray 0.16` | ort's `ndarray` feature bridges to ndarray 0.16. Do not use ndarray 0.15 — ort 2.x dropped compatibility. |
| `egui 0.34` | `eframe 0.34` | Always use the same version for egui and eframe; they release in lockstep. `egui_extras` must also match. |
| `eframe 0.34` | `wgpu` (bundled) | eframe manages its own wgpu version internally. Do not add a top-level `wgpu` dependency unless using egui_wgpu custom render passes. |
| `image 0.25` | `resvg 0.47` | resvg does not depend on the `image` crate directly. Convert `resvg::tiny_skia::Pixmap` → `image::RgbaImage` manually via `from_raw()`. |
| `arboard 3.6` | Linux (X11 + Wayland) | Without `wayland-data-control` feature, clipboard silently fails on Wayland compositors. Enable it. |

---

## Sources

- [ort crates.io page](https://crates.io/crates/ort) — version history, rc.12 confirmed as 2026-03-05
- [pykeio/ort GitHub](https://github.com/pykeio/ort) — execution providers, linking strategies
- [deepwiki.com pykeio/ort installation](https://deepwiki.com/pykeio/ort/2.1-installation-and-setup) — MEDIUM confidence, Cargo.toml feature flags
- [egui GitHub releases](https://github.com/emilk/egui/releases) — 0.34.1 confirmed latest as March 2025
- [eframe crates.io](https://crates.io/crates/eframe) — version confirmed
- [image crates.io](https://crates.io/crates/image) — 0.25.8 confirmed latest
- [clap crates.io](https://crates.io/crates/clap) — 4.5.x confirmed latest stable
- [rayon docs.rs](https://docs.rs/crate/rayon/latest) — 1.11.0 confirmed
- [resvg crates.io](https://crates.io/crates/resvg) — 0.47.0 confirmed, released 2026-02-09
- [arboard crates.io](https://crates.io/crates/arboard) — 3.6.1 confirmed, Wayland note verified
- [include-bytes-zstd GitHub](https://github.com/daac-tools/include-bytes-zstd) — proc-macro approach verified
- [zstd crates.io](https://crates.io/crates/zstd) — 0.13.3 confirmed
- [ndarray crates.io](https://crates.io/crates/ndarray/0.16.1) — 0.16.1 confirmed
- [2025 Survey of Rust GUI Libraries](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) — egui vs iced vs Slint comparison, MEDIUM confidence
- [ort execution providers docs](https://ort.pyke.io/perf/execution-providers) — 403 on direct fetch; content verified via WebSearch and DeepWiki

---

*Stack research for: Pure Rust desktop AI background removal (BgPrunR)*
*Researched: 2026-04-06*
