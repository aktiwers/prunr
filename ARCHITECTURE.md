# BgPrunR Architecture

> Living document — updated as the codebase evolves. Last updated: 2026-04-06.

## Design Principles

1. **Blazing fast, non-blocking everywhere** — The UI thread never waits on inference. All I/O and inference runs on dedicated worker threads. Batch processing parallelizes across images.
2. **SOLID in Rust** — Single-responsibility crates. Trait-based abstractions for inference backends. Dependency inversion between core/GUI/CLI via trait objects and channels.
3. **Cross-platform by default** — Every architectural decision must work on Linux x86_64, macOS x86_64+aarch64, and Windows x86_64. Platform-specific code is isolated behind feature flags.
4. **Single binary, zero dependencies** — Models, assets, and runtime libraries are embedded. The user downloads one file and runs it.

## Workspace Structure

```
bgprunr/
├── Cargo.toml                    # [workspace] — members below
├── crates/
│   ├── bgprunr-core/             # Library: inference pipeline, image I/O
│   │   └── src/
│   │       ├── lib.rs            # Public API surface + trait definitions
│   │       ├── engine.rs         # InferenceEngine trait + ORT implementation
│   │       ├── pipeline.rs       # Pre-process → infer → post-process → alpha
│   │       ├── batch.rs          # Parallel batch via rayon, progress callbacks
│   │       ├── formats.rs        # Image decode/encode (image crate + resvg)
│   │       └── types.rs          # Shared types: ProcessResult, Progress, Error (thiserror)
│   │
│   ├── bgprunr-models/           # Library: model embedding (isolated for build speed)
│   │   └── src/lib.rs            # include_bytes_zstd! for silueta + u2net
│   │                             # Dev feature: load from filesystem
│   │
│   └── bgprunr-app/              # Binary: single binary for both CLI and GUI
│       └── src/
│           ├── main.rs           # Entry point: no args → GUI, subcommands → CLI
│           ├── cli.rs            # clap subcommands: remove, batch (indicatif progress)
│           ├── gui/
│           │   ├── mod.rs        # eframe::run_native entry point
│           │   ├── app.rs        # App struct, eframe::App impl, message routing
│           │   ├── worker.rs     # Background inference thread + mpsc channels
│           │   ├── state.rs      # Application state machine (Idle → Loading → Processing → Done)
│           │   ├── views/
│           │   │   ├── canvas.rs     # Image viewer: textures, zoom, pan, checkerboard
│           │   │   ├── sidebar.rs    # Batch queue: thumbnails, drag-reorder
│           │   │   ├── toolbar.rs    # Action buttons, progress bar
│           │   │   ├── settings.rs   # Settings dialog
│           │   │   ├── shortcuts.rs  # ? help overlay
│           │   │   └── animation.rs  # Reveal animation: mask dissolve/particle effect
│           │   └── input.rs      # Keyboard/mouse input handling, shortcut dispatch
│           └── shared.rs         # Common utilities between CLI and GUI paths
│
├── xtask/                        # Developer tooling
│   ├── Cargo.toml
│   └── src/main.rs               # cargo xtask fetch-models (SHA256-verified download)
│
├── models/                       # ONNX model files (.gitignored, fetched via xtask)
│   ├── silueta.onnx              # ~4MB — fast model, default
│   └── u2net.onnx                # ~170MB — quality model
│
├── assets/                       # App icon, fonts
├── ARCHITECTURE.md               # This file
└── .planning/                    # GSD planning docs
```

## Crate Dependency Graph

```
bgprunr-models  (no deps on other workspace crates)
      │
      ▼
bgprunr-core    (depends on: bgprunr-models)
      │
      ▼
bgprunr-app     (single binary: CLI + GUI, depends on: bgprunr-core)
```

**Single binary architecture:** `bgprunr` (no args) opens the GUI. `bgprunr remove ...` runs CLI mode. One binary to distribute.

**Why this matters:**
- `bgprunr-models` compiles independently. Its 170MB embed only recompiles when model files change, not on every source edit.
- `bgprunr-core` owns all inference logic. The app binary is a thin presentation layer.
- `bgprunr-app` contains both CLI and GUI code in one binary — `bgprunr` (no args) = GUI, `bgprunr remove ...` = CLI.

## Data Flow

### Single Image (GUI)

```
User drops image
       │
       ▼
  [UI Thread]  ──load + decode──►  DynamicImage in memory
       │
       │  send via mpsc channel
       ▼
  [Worker Thread]
       │
       ├── preprocess(img)     → resize 320×320, normalize (ImageNet mean/std)
       ├── session.run(tensor) → raw ONNX output (ORT handles GPU/CPU)
       ├── postprocess(output) → sigmoid, threshold, resize mask to original dims
       └── apply_alpha(img, mask) → RGBA image with transparent background
       │
       │  send result via mpsc channel
       │  call ctx.request_repaint()
       ▼
  [UI Thread]  ──receive result──►  Play reveal animation → display result
```

### Batch Processing (GUI)

```
User drops N images
       │
       ▼
  [UI Thread]  ──populate sidebar queue──►  Vec<QueueItem>
       │
       │  send batch request via channel
       ▼
  [Worker Thread]  ──rayon::par_iter──►  N images processed in parallel
       │                                   (thread pool sized to avoid
       │                                    oversubscription with ORT)
       │
       │  send per-image results via channel
       │  call ctx.request_repaint() on each completion
       ▼
  [UI Thread]  ──cache results──►  HashMap<ImageId, ProcessResult>
                                   (switching images = cache lookup, no re-inference)
```

### CLI

```
Args parsed (clap)
       │
       ├── single: core::pipeline::process_image()  →  save PNG
       │
       └── batch:  core::batch::batch_process()
                     │
                     ├── rayon thread pool (--jobs N)
                     ├── indicatif progress bar (callback per image)
                     └── exit code: 0 (all ok) / 1 (all fail) / 2 (partial)
```

## Threading Model

| Thread | Responsibility | Never does |
|--------|---------------|------------|
| **UI thread** (egui render loop) | Renders frames, handles input, polls channels via `try_recv()` | Blocks on inference, file I/O, or any operation >1ms |
| **Worker thread** (single, long-lived) | Runs inference, image decode/encode | Touches egui state directly (sends via channel) |
| **Rayon pool** (batch only) | Parallel image processing | Conflicts with ORT intra-op threads (pool sized accordingly) |

### Thread Oversubscription Prevention

```
ORT intra_op_threads = num_cpus / rayon_pool_size
```

If rayon has 4 workers and the machine has 16 cores, each ORT session uses 4 intra-op threads. Total: 4 × 4 = 16 threads = no oversubscription.

## Inference Pipeline Detail

Matching rembg's Python preprocessing exactly:

```
1. Input: DynamicImage (any format, any size)
2. Resize to 320×320 (bilinear interpolation)
3. Convert to f32 tensor [1, 3, 320, 320] (NCHW layout)
   - Divide by 255.0
   - Normalize: (pixel - mean) / std
     mean = [0.485, 0.456, 0.406]  (ImageNet)
     std  = [0.229, 0.224, 0.225]  (ImageNet)
4. Run ONNX session (GPU or CPU)
5. Take first output tensor
6. Apply sigmoid activation
7. Normalize to [0, 1] range: (val - min) / (max - min)
8. Threshold at 0.5 → binary mask
9. Resize mask back to original image dimensions (bilinear)
10. Apply mask as alpha channel to original image → RGBA output
```

## GPU Execution Provider Strategy

```rust
SessionBuilder::new()?
    .with_execution_providers([
        CUDAExecutionProvider::default().build(),        // Linux/Windows NVIDIA
        CoreMLExecutionProvider::default().build(),      // macOS
        DirectMLExecutionProvider::default().build(),    // Windows (AMD/Intel)
        CPUExecutionProvider::default().build(),         // Always available
    ])?
```

- EPs are tried in order; first available wins
- The active EP is logged at session creation and exposed via `Engine::active_provider() -> &str`
- **No silent fallback** — the app always shows which backend is active

## Model Embedding

```rust
// bgprunr-models/src/lib.rs

#[cfg(not(feature = "dev-models"))]
pub static SILUETA_BYTES: &[u8] = include_bytes_zstd!("../../models/silueta.onnx", 19);

#[cfg(not(feature = "dev-models"))]
pub static U2NET_BYTES: &[u8] = include_bytes_zstd!("../../models/u2net.onnx", 19);

#[cfg(feature = "dev-models")]
pub fn load_model(name: &str) -> Vec<u8> {
    std::fs::read(format!("models/{name}.onnx")).expect("model file not found")
}
```

- Production: zstd-compressed at build time (level 19), decompressed once at startup
- Development: `--features dev-models` loads from filesystem (no recompilation on model changes)
- Isolated crate: changing source code in core/gui/cli does not trigger model recompilation

## State Machine (GUI)

```
        ┌───────────┐
        │   Empty   │  (app just launched, no image loaded)
        └─────┬─────┘
              │ load image
              ▼
        ┌───────────┐
        │  Loaded    │  (image displayed, ready to process)
        └─────┬─────┘
              │ user clicks Remove / Ctrl+R
              ▼
        ┌───────────┐
        │ Processing │  (worker thread running, progress shown)
        └─────┬─────┘
              │ inference complete
              ▼
        ┌───────────┐
        │ Animating  │  (reveal animation playing)
        └─────┬─────┘
              │ animation done / skipped
              ▼
        ┌───────────┐
        │   Done     │  (result displayed, can save/copy/compare)
        └─────┴─────┘
              │ load new image → back to Loaded
              │ Escape during Processing → back to Loaded
```

## Key Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ort` | 2.0.0-rc.12 | ONNX Runtime bindings (CUDA, CoreML, DirectML, CPU) |
| `egui` + `eframe` | 0.34.1 | GPU-accelerated immediate-mode GUI |
| `image` | 0.25 | Image decode/encode (PNG, JPEG, WebP, BMP) |
| `ndarray` | 0.16 | Tensor manipulation (ort 2.x compatible) |
| `rayon` | 1.11 | Work-stealing thread pool for batch parallelism |
| `clap` | 4.5 | CLI argument parsing |
| `resvg` | 0.47 | SVG → raster conversion |
| `arboard` | 3.4 | Clipboard (with `wayland-data-control` feature) |
| `indicatif` | 0.17 | CLI progress bars |
| `include-bytes-zstd` | 0.2 | Build-time model compression |
| `rfd` | 0.15 | Native file dialogs (open/save) |

## Keyboard Shortcuts — Platform Modifier

All shortcuts use Ctrl on Linux/Windows and Cmd (⌘) on macOS. In code, use egui's `Modifiers::command` (not `ctrl`) which maps to the correct platform modifier automatically.

```rust
// Correct — platform-aware
if ui.input(|i| i.modifiers.command && i.key_pressed(Key::O)) { /* open */ }

// Wrong — Ctrl on macOS feels alien
if ui.input(|i| i.modifiers.ctrl && i.key_pressed(Key::O)) { /* open */ }
```

## Platform-Specific Notes

| Platform | GPU EP | Clipboard | File dialogs | Notes |
|----------|--------|-----------|-------------|-------|
| Linux x86_64 | CUDA (if NVIDIA) → CPU | arboard + wayland-data-control | rfd (GTK/portal) | Test on both X11 and Wayland |
| macOS x86_64 | CoreML → CPU | arboard (AppKit) | rfd (NSOpenPanel) | Universal binary not required (separate x86/arm builds) |
| macOS aarch64 | CoreML (Neural Engine/GPU) → CPU | arboard (AppKit) | rfd (NSOpenPanel) | Primary Apple Silicon target |
| Windows x86_64 | CUDA → DirectML → CPU | arboard (Win32) | rfd (IFileDialog) | Bundle ORT DLL or static link to avoid system32 conflicts |

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-04-06 | Initial architecture | Project initialization |
| 2026-04-06 | Single binary (CLI+GUI), xtask model fetch, thiserror errors | Phase 1 discussion decisions |
