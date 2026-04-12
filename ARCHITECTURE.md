# Prunr Architecture

> Living document — updated as the codebase evolves. Last updated: 2026-04-12.

## Design Principles

1. **Blazing fast, non-blocking everywhere** — The UI thread never waits on inference. All I/O and inference runs on dedicated worker threads. Batch processing parallelizes across images.
2. **SOLID in Rust** — Single-responsibility crates. Trait-based abstractions for inference backends. Dependency inversion between core/GUI/CLI via trait objects and channels.
3. **Cross-platform by default** — Every architectural decision must work on Linux x86_64, macOS x86_64+aarch64, and Windows x86_64. Platform-specific code is isolated behind feature flags.
4. **Single binary, zero dependencies** — Models, assets, and runtime libraries are embedded. The user downloads one file and runs it.

## Workspace Structure

```
prunr/
├── Cargo.toml                    # [workspace] — members below
├── crates/
│   ├── prunr-core/             # Library: inference pipeline, image I/O
│   │   └── src/
│   │       ├── lib.rs            # Public API surface + trait definitions
│   │       ├── engine.rs         # InferenceEngine trait + ORT implementation
│   │       ├── pipeline.rs       # Pre-process → infer → post-process → alpha
│   │       ├── batch.rs          # Parallel batch via rayon, progress callbacks
│   │       ├── guided_filter.rs  # Guided filter alpha matting for edge refinement
│   │       ├── formats.rs        # Image decode/encode (image crate + resvg)
│   │       └── types.rs          # Shared types: ProcessResult, Progress, MaskSettings, Error
│   │
│   ├── prunr-models/           # Library: model embedding (isolated for build speed)
│   │   └── src/lib.rs            # include_bytes! + zstd decompress for silueta, u2net, birefnet-lite
│   │                             # Dev feature: load from filesystem
│   │
│   └── prunr-app/              # Binary: single binary for both CLI and GUI
│       └── src/
│           ├── main.rs           # Entry point: no args → GUI, files → CLI
│           ├── cli.rs            # clap args: prunr photo.jpg [options] (indicatif progress)
│           ├── gui/
│           │   ├── mod.rs        # eframe::run_native entry point
│           │   ├── app.rs            # App struct, eframe::App impl, event routing
│           │   ├── worker.rs         # Background inference thread + dual-engine fallback
│           │   ├── state.rs          # State machine (Empty → Loaded → Processing → Done)
│           │   ├── settings.rs       # Settings model (persisted to ~/.config/prunr/)
│           │   ├── zoom_state.rs     # Zoom/pan state (extracted from app.rs)
│           │   ├── status_state.rs   # Progress/status text (extracted from app.rs)
│           │   ├── background_io.rs  # Background channel bundle (file/thumb/decode/save)
│           │   ├── theme.rs          # Design tokens: colors, spacing, fonts, sizes
│           │   ├── views/
│           │   │   ├── canvas.rs     # Image viewer: textures, zoom/pan, checkerboard
│           │   │   ├── sidebar.rs    # Batch queue: thumbnails, drag-reorder, select
│           │   │   ├── toolbar.rs    # Open/Model/Settings/RemoveBG/ProcessAll/Save
│           │   │   ├── statusbar.rs  # Status text + progress bar + progress %
│           │   │   ├── settings.rs   # Settings modal (jobs, mask tuning, backend)
│           │   │   ├── shortcuts.rs  # F1 keyboard shortcuts overlay
│           │   │   └── cli_help.rs   # F2 CLI reference with copy-to-clipboard
│
├── xtask/                        # Developer tooling
│   ├── Cargo.toml
│   └── src/main.rs               # cargo xtask fetch-models (SHA256-verified download)
│
├── models/                       # ONNX model files (.gitignored, fetched via xtask)
│   ├── silueta.onnx              # ~4MB — fast model, default
│   ├── u2net.onnx                # ~170MB — quality model
│   └── birefnet_lite.onnx        # ~214MB — best detail, 1024×1024
│
├── assets/                       # App icon, fonts
├── ARCHITECTURE.md               # This file
└── .planning/                    # GSD planning docs
```

## Crate Dependency Graph

```
prunr-models  (no deps on other workspace crates)
      │
      ▼
prunr-core    (depends on: prunr-models)
      │
      ▼
prunr-app     (single binary: CLI + GUI, depends on: prunr-core)
```

**Single binary architecture:** `prunr` (no args) opens the GUI. `prunr photo.jpg` runs CLI mode. One binary to distribute.

**Why this matters:**
- `prunr-models` compiles independently. Its ~380MB embed only recompiles when model files change, not on every source edit.
- `prunr-core` owns all inference logic. The app binary is a thin presentation layer.
- `prunr-app` contains both CLI and GUI code in one binary — `prunr` (no args) = GUI, `prunr photo.jpg` = CLI.

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
  [UI Thread]  ──receive result──►  Crossfade original → result (0.4s)
```

### Batch Processing (GUI)

```
User opens N images (file dialog or drag-and-drop)
       │
       ▼
  [File Load Thread]  ──std::fs::read per file──►  send (bytes, name) via mpsc
       │
       │  UI drains max 5 files per frame (non-blocking)
       ▼
  [UI Thread]  ──add_to_batch()──►  Vec<BatchItem> (dims via ImageReader header-only)
       │                             Thumbnails created lazily (1 per frame in sidebar)
       │
       │  "Process All" sends batch request via channel
       ▼
  [Worker Thread]  ──rayon::par_iter──►  N images processed in parallel
       │                                   (thread pool sized to avoid
       │                                    oversubscription with ORT)
       │
       │  send BatchItemDone per image via channel
       │  call ctx.request_repaint() on each completion
       ▼
  [UI Thread]  ──cache results on BatchItem──►  result_rgba + result_texture
               ──if viewed item: sync to canvas──►
               (switching images = batch item lookup, no re-inference)
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
| **UI thread** (egui render loop) | Renders frames, handles input, polls channels via `try_recv()` | Blocks on inference, file I/O, image decode, or any operation >16ms |
| **Worker dispatch** (single, long-lived) | Receives `BatchProcess` messages, spawns processing threads | Blocks on inference (delegates immediately) |
| **Processing threads** (per batch, short-lived) | Each spawns a rayon pool for parallel inference, sends `BatchItemDone`/`BatchProgress` per image | Multiple batches can run concurrently |
| **File loader thread** (per open dialog) | Reads image files from disk, sends `(bytes, name)` via mpsc | UI drains max 5 per frame |
| **Thumbnail threads** (per image) | Decodes + resizes source images to 160px thumbnails | Sends `(id, w, h, pixels)` via channel |
| **Decode threads** (per image) | Pre-decodes source `RgbaImage` for instant canvas switching | Sends `(id, RgbaImage)` via channel |
| **Save thread** (per save operation) | PNG encode + `fs::write`, sends completion via `save_done` channel | UI shows "Saving..." toast, receives result |
| **Model pre-warm** (startup, one-shot) | Creates ORT engine to populate CoreML/CUDA disk cache | Dropped after compilation; cache persists for future creates |

### Engine Pooling Strategy

Engines are created **once per batch**, not per-image. `create_engine_pool()` handles sizing:

| Backend | Pool size | intra_threads | Rayon threads |
|---------|-----------|---------------|---------------|
| **CPU** (or GPU not ready) | 1 | num_cpus | 1 |
| **GPU** (CUDA/CoreML/DirectML) | min(jobs, 2) | num_cpus / pool_size | pool_size |

On macOS, the first `OrtEngine::new()` triggers CoreML model compilation (~2-5 min).
The pre-warm thread does this at launch. If the user processes before it finishes,
the worker falls back to `new_cpu_only()` for instant start. Future batches use GPU.

### Thread Oversubscription Prevention

```
ORT intra_op_threads = num_cpus / pool_size
```

If rayon has 4 workers and the machine has 16 cores, each ORT session uses 4 intra-op threads. Total: 4 × 4 = 16 threads = no oversubscription.

## Inference Pipeline Detail

Model-aware preprocessing and postprocessing:

```
1. Input: DynamicImage (any format, any size)
2. Resize to model target (320×320 for Silueta/U2Net, 1024×1024 for BiRefNet-lite)
3. Convert to f32 tensor [1, 3, H, H] (NCHW layout)
   - Silueta/U2Net: divide by max(max_pixel, 1e-6), then ImageNet normalize
   - BiRefNet-lite: divide by 255.0, then ImageNet normalize
     mean = [0.485, 0.456, 0.406]  std = [0.229, 0.224, 0.225]
4. Run ONNX session (GPU or CPU)
5. Take first output tensor [1, 1, H, H]
6. Normalize to [0, 1]:
   - Silueta/U2Net: min-max normalization (val - min) / (max - min)
   - BiRefNet-lite: sigmoid activation 1/(1+exp(-x))
7. Apply mask settings: gamma curve, optional binary threshold
8. Resize mask back to original image dimensions (Lanczos3)
9. Optional: edge shift (morphological erode/dilate)
10. Optional: guided filter edge refinement (uses original image colors)
11. Apply mask as alpha channel to original image → RGBA output
```

## GPU Execution Provider Strategy

```rust
// engine.rs: build_session() — cpu_only flag selects provider set
if cpu_only {
    builder.with_execution_providers([CPUExecutionProvider::default().build()])
} else {
    builder.with_execution_providers([
        #[cfg(not(target_os = "macos"))]
        CUDAExecutionProvider::default()
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .with_cuda_graph(true)   // fixed input shapes → graph replay
            .with_tf32(true)         // TF32 on Ampere+ GPUs
            .build(),
        #[cfg(target_os = "macos")]
        CoreMLExecutionProvider::default()
            .with_model_cache_dir(coreml_cache_dir())
            .build(),
        #[cfg(windows)]
        DirectMLExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ])
}
```

- EPs are feature-gated per platform — only the relevant GPU EP is compiled in
- CUDA EP tuned: `arena_extend_strategy(SameAsRequested)`, `cuda_graph(true)`, `tf32(true)` for 10-30% speedup
- `new_cpu_only()` skips all GPU providers for instant startup
- The active EP is exposed via `Engine::active_provider() -> &str`
- **Dual-engine fallback**: GPU compiles in background; CPU used until ready
- **FP16/INT8 model variants**: `new_with_fallback()` tries optimized model first, falls back to FP32 if ORT rejects it

## Model Embedding

```rust
// prunr-models/src/lib.rs

// Pre-compressed .zst blobs embedded via plain include_bytes!
#[cfg(not(feature = "dev-models"))]
static SILUETA_ZST: &[u8] = include_bytes!("../../../models/silueta.onnx.zst");
#[cfg(not(feature = "dev-models"))]
static U2NET_ZST: &[u8] = include_bytes!("../../../models/u2net.onnx.zst");
#[cfg(not(feature = "dev-models"))]
static BIREFNET_LITE_ZST: &[u8] = include_bytes!("../../../models/birefnet_lite.onnx.zst");

// Runtime decompression via zstd::bulk::decompress
#[cfg(not(feature = "dev-models"))]
pub fn silueta_bytes() -> Vec<u8> {
    zstd::bulk::decompress(SILUETA_ZST, 50 * 1024 * 1024).expect("decompress failed")
}

#[cfg(feature = "dev-models")]
pub fn silueta_bytes() -> Vec<u8> {
    std::fs::read("models/silueta.onnx").expect("model not found")
}
```

- Production: models stored as pre-compressed `.onnx.zst` files, embedded via `include_bytes!`, decompressed at runtime via `zstd::bulk::decompress`
- Development: `--features dev-models` loads from filesystem (no recompilation on model changes)
- Isolated crate: changing source code in core/gui/cli does not trigger model recompilation
- Dependency changed from `include-bytes-zstd` (compile-time macro) to `zstd` (runtime decompression) for simpler builds
- **Optimized variants**: FP16 (GPU) and INT8 (CPU) models generated by `scripts/convert_models.py`, loaded at runtime via `model_fp16_bytes()`/`model_int8_bytes()` with FP32 fallback

## State Machine (GUI)

```
        ┌───────────┐
        │   Empty   │  (app just launched, no image loaded)
        └─────┬─────┘
              │ load image (open dialog, drag-drop, or batch file channel)
              ▼
        ┌───────────┐
        │  Loaded    │  (image displayed, ready to process)
        └─────┬─────┘
              │ user clicks Remove BG / Process All / Ctrl+R
              ▼
        ┌───────────┐
        │ Processing │  (worker thread running, progress shown)
        └─────┬─────┘
              │ inference complete
              ▼
        ┌───────────┐
        │   Done     │  (result displayed, can save/copy/compare/save all)
        └─────┴─────┘
              │ load new image → back to Loaded
              │ Escape during Processing → back to Loaded
              │ Click different batch item → state follows that item's status
```

**Batch state**: In batch mode, switching sidebar items sets `AppState` to match the *viewed item's*
`BatchStatus` (Pending→Loaded, Processing→Processing, Done→Done). The global state reflects
the currently viewed image, not the overall batch progress.

## Key Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ort` | 2.0.0-rc.12 | ONNX Runtime bindings (CUDA, CoreML, DirectML, CPU) |
| `egui` + `eframe` | 0.34.1 | GPU-accelerated immediate-mode GUI |
| `image` | 0.25 | Image decode/encode (PNG, JPEG, WebP, BMP) |
| `ndarray` | 0.17 | Tensor manipulation (ort 2.x compatible) |
| `rayon` | 1.11 | Work-stealing thread pool for batch parallelism |
| `clap` | 4.5 | CLI argument parsing |
| `resvg` | 0.47 | SVG → raster conversion |
| `arboard` | 3.4 | Clipboard (with `wayland-data-control` feature) |
| `indicatif` | 0.17 | CLI progress bars |
| `zstd` | 0.13 | Runtime model decompression (replaced include-bytes-zstd) |
| `rfd` | 0.15 | Native file dialogs (open/save/folder picker) |
| `num_cpus` | 1.x | Detect CPU count for parallel jobs setting |
| `dirs` | 6.x | Cross-platform config directory for persistent settings |
| `egui-notify` | 0.22 | Toast notification system |
| `egui_material_icons` | 0.6 | Material Design icons throughout the UI |
| `serde` + `serde_json` | 1.x | Settings serialization/persistence |

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
| 2026-04-08 | Multi-select open dialog, background file loading thread | Batch UX improvement |
| 2026-04-08 | Sidebar moved to right, lazy thumbnail decode (1/frame) | UI polish |
| 2026-04-08 | Replaced include-bytes-zstd with zstd runtime decompression | Simpler build, no proc-macro dependency |
| 2026-04-08 | Settings modal: .open() close button, stepper for parallel jobs | Settings UI overhaul |
| 2026-04-08 | Save All button + folder picker for batch export | Batch workflow completion |
| 2026-04-08 | Reveal animation for batch items, dimension safety check | Batch animation support |
| 2026-04-08 | Fit-to-window zoom on image load/switch | Large image UX |
| 2026-04-08 | Batch selection: per-item checkboxes, Select All/Clear, Remove Selected | Multi-image workflow |
| 2026-04-08 | Save Selected / Remove BG Selected: actions follow checkbox state | Consistent batch UX |
| 2026-04-08 | Click-drag pan (no Space key needed), sidebar always visible with 1+ images | Photo editor UX |
| 2026-04-08 | Background thumbnail generation via thread+channel (non-blocking UI) | Performance |
| 2026-04-08 | Interactive modal backdrop (click outside to close), fixed Settings button | Modal UX |
| 2026-04-08 | Processing animations: shimmer sweep + pulsing border on thumbnails/canvas | Visual feedback |
| 2026-04-08 | Extracted process_items(), clear_to_empty(), sync_after_batch_change() helpers | Code cleanup |
| 2026-04-08 | ProcessResult carries raw rgba_image — eliminates UI-thread PNG decode (50-200ms) | Performance |
| 2026-04-08 | Arc<Vec<u8>> for source_bytes — eliminates multi-MB clones on click/send | Performance |
| 2026-04-08 | Background thread save (encode+write off UI thread) with save_done channel | Performance |
| 2026-04-08 | Removed all UI-thread image::load_from_memory fallbacks | Performance |
| 2026-04-08 | Cow::Borrowed for clipboard copy, fixed paste encode roundtrip | Performance |
| 2026-04-08 | Added egui_material_icons, egui-notify, egui_animation dependencies | UI polish |
| 2026-04-08 | Material Design icons on all toolbar buttons, toast notifications | UI polish |
| 2026-04-08 | Pre-decoded source images via background thread for instant switching | Performance |
| 2026-04-08 | Plum purple theme from logo (#7B2D8E accent, #5B8C3E leaf green) | Branding |
| 2026-04-08 | Eliminated ProcessImage path — all processing via BatchProcess with item_id | Architecture |
| 2026-04-08 | Worker spawns threads per batch for true parallel concurrent processing | Architecture |
| 2026-04-08 | Per-stage progress via BatchProgress (6 stages reported to UI) | UX |
| 2026-04-08 | Sidebar hover actions: trash icon (delete), save icon (per-item save) | UX |
| 2026-04-08 | Simplified CLI: prunr photo.jpg works without subcommand, short flags | CLI |
| 2026-04-08 | Canvas fade-in (200ms) on image switch, thumbnail fade-in on load | UI polish |
| 2026-04-08 | Non-interactive settings backdrop (fixes frozen settings modal) | Bug fix |
| 2026-04-08 | Fixed concurrent processing: results tracked by item_id not selected index | Bug fix |
| 2026-04-08 | Mask tuning: gamma, threshold, edge shift controls in Settings + CLI | Feature |
| 2026-04-08 | Ctrl+Z undo / Ctrl+Y redo for background removal | Feature |
| 2026-04-08 | Settings persistence via ~/.config/prunr/settings.json | Feature |
| 2026-04-08 | Smart parallel jobs default: 2 for GPU, num_cpus/2 for CPU | Performance |
| 2026-04-08 | Removed CLI subcommand — prunr photo.jpg works directly | CLI simplification |
| 2026-04-08 | F2 CLI reference modal with copy-to-clipboard buttons | UX |
| 2026-04-08 | Cycling tips on empty canvas with fade animation | UX |
| 2026-04-09 | BiRefNet-lite model (1024×1024, ~214 MB) for fine detail | Feature |
| 2026-04-09 | Guided filter edge refinement (pure Rust, O(1) box filter) | Feature |
| 2026-04-09 | Removed reveal animation — replaced with simple crossfade (0.4s) | Simplification |
| 2026-04-09 | Loading spinner on canvas while source image decodes | UX |
| 2026-04-09 | Model dropdown in toolbar with per-model icons | UI |
| 2026-04-09 | Progress percentage in status bar during processing | UX |
| 2026-04-09 | Save dialogs default to source image directory | UX |
| 2026-04-09 | Backend detected at startup via compile-time feature flags | Architecture |
| 2026-04-09 | GitHub Actions release workflow with embedded models | CI/CD |
| 2026-04-12 | God Object refactor: ZoomState, StatusState, BackgroundIO extracted | Architecture |
| 2026-04-12 | logic() decomposed into 5 focused sub-methods (340→8 lines) | Architecture |
| 2026-04-12 | Engine pooling: create_engine_pool() replaces per-image OrtEngine::new() | Performance |
| 2026-04-12 | Dual-engine CPU fallback: instant start while GPU compiles in background | Performance |
| 2026-04-12 | Arc<RgbaImage> throughout — zero-copy image switching/saving | Performance |
| 2026-04-12 | VRAM-safe GPU pool: capped at 2 sessions per batch | Stability |
| 2026-04-12 | Lazy PNG encoding: removed from pipeline, encode on save only | Performance |
| 2026-04-12 | Parallel guided filter: rayon::join for 4-way box_filter, parallel prefix sums | Performance |
| 2026-04-12 | Parallel apply_edge_shift with Rayon (par_chunks_mut rows, threshold gate) | Performance |
| 2026-04-12 | Raw buffer ops: to_nchw precomputed scale/bias, channel-planar writes | Performance |
| 2026-04-12 | Raw buffer postprocess: flat pred_slice, in-place alpha compositing | Performance |
| 2026-04-12 | Fast PNG encoding: Compression::Fast + FilterType::Sub (3-5x faster) | Performance |
| 2026-04-12 | Resize directly from DynamicImage (skip full-res RGB intermediate) | Performance |
| 2026-04-12 | Arc in request_decode (eliminates 5-50MB clone per import) | Performance |
| 2026-04-12 | Drag-and-drop file reads moved to background thread | Performance |
| 2026-04-12 | CUDA EP tuned: arena strategy, cuda_graph, tf32 | Performance |
| 2026-04-12 | FP16/INT8 model variants with fallback to FP32 | Performance |
| 2026-04-12 | egui_extras trimmed from all_loaders to image (~15MB binary reduction) | Build |
| 2026-04-12 | Release profile: fat LTO, panic=abort, codegen-units=1 | Build |
| 2026-04-12 | Fix: sidebar thumb_pending never cleared (100% CPU when idle) | Bug fix |
| 2026-04-12 | Fix: GPU warming status only shown when GPU actually present | Bug fix |
| 2026-04-12 | selected_item() helper replaces direct batch indexing (5 call sites) | Code quality |
| 2026-04-12 | LoadingModelCpuFallback progress stage for CPU fallback visibility | UX |
