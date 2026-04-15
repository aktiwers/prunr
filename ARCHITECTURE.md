# Prunr Architecture

Living document describing how Prunr is built. For user-facing info see [README.md](README.md).

## Design Principles

1. **The UI thread never waits** — inference, file I/O, decoding, PNG encoding, texture prep all run on background threads. All communication is via `mpsc` channels drained non-blockingly each frame.
2. **Single binary** — one `prunr` executable hosts both GUI and CLI. No `prunr-gui` vs `prunr-cli` split. Models are embedded as zstd blobs decompressed at runtime.
3. **Platform parity** — every feature works on Linux x86_64, macOS aarch64, and Windows x86_64. Platform-specific code is isolated behind `#[cfg(...)]`.
4. **Progressive performance** — start fast, improve in place. Use real hardware (GPU/NE) when available, fall back gracefully to CPU, never block startup on compilation.

## Workspace

```
prunr/
├── crates/
│   ├── prunr-models/       # Embedded zstd-compressed ONNX blobs + runtime decompression
│   ├── prunr-core/         # Inference pipeline, image I/O, batch processing
│   └── prunr-app/          # Single binary: GUI (eframe/egui) + CLI (clap)
├── xtask/                  # Developer tooling (cargo xtask fetch-models)
├── packaging/              # AUR PKGBUILD, Homebrew formula template
├── scripts/                # Model conversion (FP16/INT8 variants, DexiNed export)
├── assets/                 # Icon, Info.plist, .desktop file
├── .github/workflows/      # CI + multi-platform release packaging
├── ARCHITECTURE.md         # This file
├── README.md
└── LICENSE                 # Apache-2.0
```

**Dependency direction:** `prunr-models` → `prunr-core` → `prunr-app`. Reverse deps are forbidden; `prunr-models` has no workspace-internal dependencies so its ~380 MB embed blob only recompiles when models change.

## Threading Model

| Thread | Spawned | Lives for | Purpose |
|--------|---------|-----------|---------|
| UI (egui render loop) | main | app lifetime | Render frames, handle input, drain channels via `try_recv()` |
| Worker dispatch | startup | app lifetime | Receives `WorkerMessage::BatchProcess`, spawns processing threads |
| Per-batch processing | per batch | one batch | Builds engine pool, drives rayon scope for parallel inference |
| File loader | per drag-drop / open | until files loaded | `std::fs::read` in a loop, sends `(bytes, name)` via mpsc |
| Image decoder | per imported image | ~20-80ms | `image::load_from_memory` → `Arc<RgbaImage>` for instant canvas switching |
| Thumbnail builder | per imported image | ~5-50ms | Lanczos resize to 160px, sends `(id, w, h, pixels)` |
| Texture prep | per canvas texture | ~5-50ms | Builds `egui::ColorImage` off the UI thread |
| Save writer | per save | until PNG written | Background PNG encode + `fs::write` |
| Model pre-warm | startup | app lifetime | Creates one GPU engine early so CoreML/CUDA compile cache populates |
| Temp file cleanup | app exit | brief | Removes `prunr-drag/*` temp files |

All background threads use `std::thread::spawn` — deliberately **not** `rayon::spawn`. Rayon's global pool is reserved for inference/guided filter parallelism; mixing them caused pool saturation during batch processing (earlier iterations of this code used rayon for everything and the global pool got blocked by decode/thumbnail tasks while `guided_filter::rayon::join` waited for workers).

### Engine pooling

Engines are created **once per batch**, not per-image:

| Backend | Pool size | Intra-op threads |
|---------|-----------|------------------|
| CPU | `jobs` | `num_cpus / pool_size` |
| GPU (CUDA/DirectML/CoreML) | `jobs.min(2)` | `num_cpus / pool_size` |

GPU is capped at 2 engines — more doesn't help because the GPU driver serializes anyway, and each extra session doubles VRAM. The per-batch rayon thread pool matches the engine count so every worker has its own engine (no Mutex contention during inference).

### Pre-warm + GPU fallback

At startup, a background thread calls `OrtEngine::new()` to populate the CoreML/CUDA disk cache. If the user triggers processing before that finishes, the worker falls back to `new_cpu_only()` and reports `ProgressStage::LoadingModelCpuFallback` so the UI can show "GPU warming up — using CPU". After the first successful engine pool creation, the batch thread drops its Arc clone of the pre-warm engine to free its VRAM.

### Repaint throttling

With N rayon workers each calling `request_repaint()` on progress, we'd get N frames per 33ms window. Instead, the progress callback uses a single `Arc<Mutex<Instant>>` across all workers — only one repaint fires per 33ms regardless of how many workers are active.

## Data Flow

### Single-image GUI processing

```
User drops/opens image
  ↓ (UI thread)
add_to_batch(bytes) → BatchItem with Arc<Vec<u8>> source_bytes
  ↓ (spawns decode + thumbnail threads)
Image decoder → Arc<RgbaImage> → item.source_rgba
Thumbnail builder → (id, w, h, pixels) → sidebar texture
Texture prep → ColorImage → ctx.load_texture() → canvas texture
  ↓ (user clicks "Remove BG")
WorkerMessage::BatchProcess { items, model, jobs, ... } → worker
  ↓ (worker spawns per-batch thread)
create_engine_pool(model, jobs, cpu_only)
  ↓ (pool.scope() — rayon parallel)
per item: preprocess → session.run → postprocess → apply_alpha
  ↓ (per image: mpsc send)
WorkerResult::BatchItemDone { item_id, result }
  ↓ (UI thread drains 8/frame)
item.result_rgba = Some(Arc::new(rgba))
  ↓ (spawns texture prep for result)
Texture prep → ctx.load_texture() → result_texture
  ↓
Canvas crossfade (0.4s) original → result
```

### Chain mode

When "chain mode" is on, processing an already-processed image feeds the previous result as input. The worker receives `Option<Arc<RgbaImage>>` as a chain input; if present, it's wrapped in a `DynamicImage` (via `Arc::unwrap_or_clone` — zero-copy when sole owner) and passed to `process_image_from_decoded()` which skips the PNG encode/decode round-trip.

### CLI

```
main.rs
  ↓ Cli::parse()
  ↓ inputs non-empty → cli::run_remove(&cli)
  ↓
single image: pipeline::process_image_with_mask() → PNG encode → fs::write
batch:        batch::batch_process_with_mask() → rayon → PNG encodes in parallel
  ↓
exit code: 0 all ok / 1 all fail / 2 partial
```

## Inference Pipeline

```
1.  load_image_from_bytes()             → DynamicImage
2.  check_large_image()                 → error if > 8000px (configurable via --large-image)
3.  preprocess(img, model)              → Array4<f32> [1, 3, H, H] NCHW
      - Silueta/U2Net: divide by max_pixel, ImageNet normalize
      - BiRefNet-lite: divide by 255, ImageNet normalize
      (single-pass: sequential reads of RGB source, scattered writes to 3 planes)
4.  engine.with_session(|s| s.run(...)) → raw output tensor [1, 1, H, H]
5.  postprocess(raw_output.view(), img, mask, model):
      - Normalize to [0, 1]
          - Silueta/U2Net: min-max via single scan
          - BiRefNet-lite: sigmoid
      - Short-circuit uniform output (everything foreground or everything background)
          → fill mask with constant, skip per-pixel loop
      - Otherwise: single pass with precomputed inv_range (division → multiplication)
      - Apply gamma + optional hard threshold
      - Resize mask to original dimensions (Lanczos3)
      - Optional: apply_edge_shift (morphological erode/dilate, pre-allocated buffers)
      - Optional: guided_filter_alpha (O(1) box filter with f32 integral images)
      - Compose mask as alpha channel
6.  Optional: apply_background_color() → fills transparent pixels with solid color
```

### Postprocess fast paths

- Division by range → precomputed `inv_range` multiplier (10x faster per pixel)
- Uniform-output detection happens before the per-pixel loop (skipped entirely for uniform masks)
- Guided filter uses `f32` prefix sums (was `f64` — halved memory bandwidth with no precision loss at typical image sizes)
- Apply_edge_shift's ring buffers are allocated once and swapped via `std::mem::swap` across iterations

## Edge Detection (DexiNed)

A separate `EdgeEngine` handles line extraction with its own ONNX session. Three line modes:

| Mode | Behaviour |
|------|-----------|
| `Off` | Normal background removal only |
| `LinesOnly` | Skip segmentation, run DexiNed on original image |
| `AfterBgRemoval` | Segmentation first, flatten result onto white (prevents ghost edges from transparent regions), then DexiNed |

Pipeline: resize to 480×640 → BGR float32 with mean subtraction → run DexiNed → take fused output → sigmoid + smoothstep threshold controlled by `line_strength` slider → resize mask to original → compose as alpha (optionally override RGB with solid line color).

## GPU Execution Providers

```rust
if cpu_only {
    // CPUExecutionProvider only
} else {
    // Try in order, fall through to CPU:
    #[cfg(not(target_os = "macos"))]
    CUDAExecutionProvider::default()
        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
        .with_cuda_graph(true)      // fixed-shape models → graph replay
        .with_tf32(true)            // 10-30% speedup on Ampere+
        .build(),
    #[cfg(target_os = "macos")]
    CoreMLExecutionProvider::default()
        .with_model_cache_dir(coreml_cache_dir())  // persistent cache
        .build(),
    #[cfg(windows)]
    DirectMLExecutionProvider::default().build(),
    CPUExecutionProvider::default().build(),
}
```

- EPs are feature-gated per platform; only the relevant GPU EP is compiled in
- The active EP is detected once and cached via `OrtEngine::detect_active_provider()` (a `OnceLock<String>`). Settings UI uses this (not `active_backend`, which reflects what the last batch ran on) to decide whether to show "Force CPU".

### Model variants

`OrtEngine::new_with_fallback()` selects the right model for the active backend:

| Platform | Preferred | Fallback |
|----------|-----------|----------|
| Linux/Windows GPU | FP16 | FP32 |
| Linux/Windows CPU | INT8 | FP32 |
| **macOS (all backends)** | **FP32 always** | n/a |

macOS uses FP32 because CoreML silently converts to FP16 internally on Apple Silicon. Feeding it our FP16 variant stacks two conversions — precision loss can collapse the mask to near-zero, producing a fully transparent output (the "entire image removed" bug).

### Model bytes cache

Decompressed ONNX bytes are cached in `OnceLock<Vec<u8>>` per model. First call to e.g. `silueta_bytes()` takes ~200ms (zstd decompress 50 MB); subsequent calls clone the cached Vec (~1 ms). Previously every engine creation triggered a fresh decompression — visible as a 200 ms stall every time the user changed models or started a batch.

## GUI State Machine

```
      Empty  ──(add_to_batch)──►  Loaded  ──(process)──►  Processing  ──(done)──►  Done
        ▲                            ▲                        │                      │
        │                            │                        │ (cancel/Escape)      │
        │                            └────────────────────────┤                      │
        │                                                     │                      │
        │                                                     ▼                      │
        │                                              Back to Loaded                │
        │                                                                            │
        └────────────────────── (remove all batch items) ────────────────────────────┘
```

In batch mode, switching sidebar items sets `AppState` to match the **viewed item's** `BatchStatus` (Pending→Loaded, Processing→Processing, Done→Done). The global state reflects the currently viewed image, not the overall batch progress.

## Canvas & Texture Lifecycle

- Textures are built on background threads via `spawn_tex_prep()` — the expensive `ColorImage::from_rgba_unmultiplied()` (full RGBA copy) runs off the UI thread; `ctx.load_texture()` itself is cheap (queues a GPU upload)
- Each `BatchItem` tracks `source_tex_pending` / `result_tex_pending` flags independently so source and result texture prep don't block each other
- During a sidebar switch the previous texture is kept visible until the new one is ready — prevents a spinner flash for 25-130 ms while decode + texture prep complete
- The checkerboard behind transparent results is a single 256×256 pre-generated texture (per theme: light and dark), tiled via `~40` `painter.image()` calls — down from ~8100 `rect_filled()` calls on a 1080p canvas in an earlier iteration
- Off-screen sidebar items skip painting entirely (virtualization by viewport intersection)

## Drag-Out (OS-level drag to external apps)

Implemented via the `drag` crate (Windows/macOS only — Linux lacks a GTK window under winit). Files are written to `std::env::temp_dir()/prunr-drag/*.png` and handed to the OS drag session.

Two defences against footguns:

1. **Self-drop rejection** — if a drop event contains a path inside `prunr-drag/`, it's silently discarded. Otherwise dragging a thumbnail back onto the Prunr canvas would re-ingest it as a new image
2. **Stuck-drag recovery** — on Windows, the `drag` crate's completion callback sometimes doesn't fire. When a self-drop is detected, we clear `drag_out_active` and `drag_out_items` so the sidebar dimming resets and a new drag can start

Temp files are removed at app shutdown via a `Drop` impl on `PrunrApp` that spawns a cleanup thread (also cancels the worker via `cancel_flag`).

## Windows-Specific: Console Subsystem

Release builds on Windows use `#[windows_subsystem = "windows"]` — without this, launching the GUI pops an empty `cmd` window. When CLI arguments are detected at startup, we call `AttachConsole(ATTACH_PARENT_PROCESS)` so `prunr.exe photo.jpg` invoked from `cmd`/PowerShell still prints to the user's terminal.

The GUI has a renderer fallback chain: glow (OpenGL) first, then wgpu (DX12/Vulkan). Mac-hosted Windows VMs often ship unreliable OpenGL drivers; wgpu works better in virtualized environments. If both fail, startup errors are written to `prunr-startup-error.log` next to the exe.

## Atomic Memory Ordering

`cancel_flag` and `drag_out_active` use `Release` on store, `Acquire` on load (rather than `Relaxed`). On x86 this is free; on ARM (Apple Silicon, and eventually ARM Windows) the memory barriers are necessary to guarantee cross-thread visibility of the cancel signal.

## Build & Release

### Build profile

```toml
[profile.release]
strip = true
lto = "fat"
opt-level = 3
panic = "abort"
codegen-units = 1
```

LTO takes longer to link but shrinks the binary by ~15% and enables cross-crate inlining across the workspace.

### GitHub Actions release pipeline

Tag push (`v*`) triggers parallel builds on three runners:

| Matrix target | Runner | Artifacts |
|---------------|--------|-----------|
| linux-x86_64 | ubuntu-latest | `.tar.gz`, `.AppImage`, `.deb`, `.rpm` |
| macos-aarch64 | macos-latest | `.dmg`, `.tar.gz` (version patched into Info.plist from git tag) |
| windows-x86_64 | windows-latest | `.zip`, Inno Setup `.exe` |

All 8 artifacts are uploaded to a single GitHub Release. The RPM is built from an inline `.spec` file; the `.deb` is built directly via `dpkg-deb`.

### Homebrew tap

`brew install aktiwers/prunr/prunr` installs from [aktiwers/homebrew-prunr](https://github.com/aktiwers/homebrew-prunr) (a separate repo, since Homebrew requires taps to live in `homebrew-<name>` repos). The formula downloads the macOS tarball from the main repo's GitHub Release. Updates require bumping `version` + `sha256` in that tap repo after each release.

### Version sync

The workspace `Cargo.toml` `version` field is the single source of truth:
- CLI: `clap` reads `CARGO_PKG_VERSION` automatically
- Info.plist: patched from `${GITHUB_REF_NAME#v}` in CI before the macOS build
- Inno Setup + deb + rpm: each reads the tag name via env in their respective package steps
- Homebrew: manually synced in the tap after release (documented in `packaging/homebrew/prunr.rb`)

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ort` | 2.0.0-rc.12 | ONNX Runtime bindings |
| `eframe` / `egui` | 0.34.1 | Immediate-mode GUI + windowing |
| `wgpu` (transitive) | 29 | GPU rendering fallback on Windows VMs |
| `image` | 0.25 | PNG/JPEG/WebP/BMP decode/encode |
| `ndarray` | 0.17 | ORT-compatible tensor manipulation |
| `rayon` | 1.11 | Work-stealing parallelism (scoped pools per batch) |
| `zstd` | 0.13 | Runtime model decompression |
| `clap` | 4.5 | CLI argument parsing |
| `rfd` | 0.15 | Native file dialogs |
| `arboard` | 3.x | Cross-platform clipboard |
| `dirs` | 6.x | Platform config dirs |
| `drag` | 2.1 | OS drag-out (Windows/macOS only) |
| `windows-sys` | 0.59 | AttachConsole on Windows GUI builds |
| `serde` / `serde_json` | 1.x | Settings persistence |
