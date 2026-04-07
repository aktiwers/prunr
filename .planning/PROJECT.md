# BgPrunR

## What This Is

BgPrunR is a desktop application that removes backgrounds from images using AI, running entirely on the user's machine. It provides both a CLI for scripting/batch workflows and a GUI for interactive use. Drop a photo in, click a button, get a transparent PNG — no internet, no cloud, no subscriptions.

## Core Value

One-click local background removal that is fast, private, and works offline — your photos never leave your machine.

## Requirements

### Validated

- ✓ Single-binary distribution with embedded models and assets — Validated in Phase 1
- ✓ Cross-platform: Linux, macOS, Windows — Validated in Phase 1 (CI green on all 4 targets)
- ✓ Local AI inference using ONNX Runtime (U2-Net / silueta models) — Validated in Phase 2 (43 tests pass, CORE-05 reference test green)
- ✓ Batch processing with parallel inference across thread pool — Validated in Phase 2
- ✓ Large image warning/downscale prompt (over 8000px) — Validated in Phase 2
- ✓ CLI mode sharing the same core inference engine — Validated in Phase 3 (single, batch, model select, exit codes all working)
- ✓ egui-based GUI with drag-and-drop, zoom, pan, before/after comparison — Validated in Phase 5 (cursor-centered zoom, space+drag pan, B-key toggle)
- ✓ Keyboard shortcuts for all major actions — Validated in Phase 5 (14 shortcuts including Ctrl+0/1, Tab, [/], Ctrl+,)
- ✓ Settings dialog (auto-remove, model selection, parallelism) — Validated in Phase 5 (5-field dialog with Ctrl+, shortcut)
- ✓ Reveal animation on background removal completion — Validated in Phase 5 (ease-out cubic fade, skippable)
- ✓ Batch sidebar for multi-image workflows — Validated in Phase 5 (thumbnails, DnD reorder, Process All, auto-remove)

### Active

- [ ] GPU acceleration (CUDA/Metal/Vulkan) with automatic CPU fallback
- [ ] Support PNG, JPEG, WebP, BMP input; SVG rasterized on load via resvg
- [ ] PNG output with transparency
- [ ] Single-binary distribution with embedded models and assets
- [ ] Cross-platform: Linux, macOS, Windows
- [ ] Copy result to clipboard
- [ ] Two bundled models: silueta (~4MB, fast) and u2net (~170MB, best quality)
- [ ] Progress indication during inference

### Out of Scope

- Cloud/server processing — violates core privacy principle
- Web UI or embedded web server — pure native Rust only
- Video background removal — image-only for v1
- Real-time camera feed — desktop file processing only
- Plugin/extension system — keep it simple and self-contained
- Custom model training — use pre-trained ONNX models only

## Context

**Inspiration:** Python rembg library (github.com/danielgatis/rembg) which wraps ONNX Runtime + U2-Net models. BgPrunR replicates the same inference pipeline in pure Rust using the `ort` crate (Rust ONNX Runtime bindings), eliminating the Python dependency entirely.

**Inference pipeline:** Input image → resize to 320×320 → normalize (mean/std per rembg spec) → ONNX inference → sigmoid → threshold → resize mask to original dimensions → apply alpha channel. This must match rembg's preprocessing exactly for equivalent output quality.

**Model bundling:** Both silueta (~4MB) and u2net (~170MB) ONNX models are embedded in the binary at compile time. The user selects which model to use in settings. Silueta is the default for speed; u2net available for maximum quality.

**GPU strategy:** The `ort` crate supports CUDA (Linux/Windows), CoreML/Metal (macOS), and DirectML (Windows) execution providers. Build with feature flags to enable platform-appropriate GPU backends. CPU is always available as fallback.

## Constraints

- **Tech stack**: Pure Rust — Cargo workspace with `ort` (ONNX), `egui`/`eframe` (GUI), `image` (decode/encode), `clap` (CLI), `rayon` (parallelism), `resvg` (SVG)
- **Distribution**: Single self-contained binary per platform, no runtime dependencies for end users
- **Binary size**: ~180MB acceptable (dominated by u2net model), compress with `include_bytes!` + zstd decompression
- **Performance**: Blazing fast, fully non-blocking architecture — UI thread never waits on inference, all I/O and inference is async/threaded, batch processing parallelized across images
- **Architecture**: SOLID principles — single responsibility crates, trait-based abstractions for inference backends, dependency inversion between core/GUI/CLI
- **Compatibility**: Must build and run on Linux (x86_64), macOS (x86_64 + aarch64), Windows (x86_64)
- **Privacy**: Zero network access — no telemetry, no update checks, no model downloads at runtime

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| egui over iced/Slint | Best for image tools: built-in textures, zoom/pan, immediate mode simplifies state, GPU-rendered | — Pending |
| ort over tract/candle | Direct ONNX Runtime bindings = exact rembg model compatibility, mature GPU support | — Pending |
| Embed models in binary | User downloads nothing extra; single-file distribution; trade binary size for UX | — Pending |
| GPU + CPU fallback | Best performance when GPU available, universal compatibility via CPU fallback | — Pending |
| rayon for batch parallelism | Proven work-stealing thread pool, zero-config parallelism across images | — Pending |
| resvg for SVG | Rasterize SVG on load — simple, reliable, avoids complex vector pipeline | — Pending |
| Cargo workspace | Shared core between CLI and GUI binaries, clean separation of concerns | — Pending |

---
*Last updated: 2026-04-07 after Phase 5 completion*
