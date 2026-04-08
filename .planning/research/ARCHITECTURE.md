# Architecture Research

**Domain:** Local AI desktop image processing application (Rust)
**Researched:** 2026-04-06
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
├────────────────────────────┬────────────────────────────────────┤
│  prunr-gui (egui/eframe) │  prunr-cli (clap binary)        │
│  ┌──────────────────────┐  │  ┌──────────────────────────────┐  │
│  │  App struct (state)  │  │  │  Arg parsing → JobRequest    │  │
│  │  update() per frame  │  │  │  Progress → stderr/stdout    │  │
│  │  Texture cache       │  │  │  Batch via rayon             │  │
│  └──────────┬───────────┘  │  └──────────────┬───────────────┘  │
│             │ mpsc channels │                 │ direct call      │
└─────────────┼───────────────┴─────────────────┼─────────────────┘
              │                                  │
┌─────────────┼──────────────────────────────────┼─────────────────┐
│                         Core Library (prunr-core)               │
│  ┌──────────▼──────────────────────────────────▼──────────────┐  │
│  │                    pub API surface                          │  │
│  │  process_image(input, model, options) → Result<RgbaImage>  │  │
│  │  batch_process(inputs, model, options, progress_cb)        │  │
│  └──────────┬───────────────────────────────────┬─────────────┘  │
│             │                                   │                 │
│  ┌──────────▼──────────┐           ┌────────────▼─────────────┐  │
│  │   Image I/O Layer   │           │   Inference Engine       │  │
│  │  image crate        │           │   ort (ONNX Runtime)     │  │
│  │  resvg (SVG)        │           │   Session pool           │  │
│  │  decode/encode PNG  │           │   Execution providers    │  │
│  └──────────┬──────────┘           └────────────┬─────────────┘  │
│             │                                   │                 │
│  ┌──────────▼──────────┐           ┌────────────▼─────────────┐  │
│  │  Pre/Post Processor │           │   Model Registry         │  │
│  │  resize → 320x320   │           │   silueta (embedded)     │  │
│  │  normalize CHW      │           │   u2net (embedded)       │  │
│  │  mask → alpha chan  │           │   include_bytes_zstd     │  │
│  └─────────────────────┘           └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────-┘
```

### Component Responsibilities

| Component | Responsibility | Talks To |
|-----------|----------------|----------|
| `prunr-core` (lib) | Inference pipeline, image I/O, model management | Nothing upstream — pure library |
| `prunr-gui` (bin) | egui/eframe UI, state, texture rendering, user events | `prunr-core` via mpsc worker thread |
| `prunr-cli` (bin) | Clap arg parsing, batch orchestration, stdout/stderr | `prunr-core` directly (blocking calls) |
| `InferenceEngine` (in core) | Session creation, execution provider negotiation, tensor I/O | `ort` crate, `ModelRegistry` |
| `PreProcessor` (in core) | Resize, normalize, channel-first layout, f32 tensor | `image` crate |
| `PostProcessor` (in core) | Sigmoid, threshold, mask resize, alpha channel merge | `image` crate |
| `ModelRegistry` (in core) | Static model bytes (zstd-compressed), lazy session init | `include_bytes_zstd`, `ort` |
| `ImageIO` (in core) | Decode PNG/JPEG/WebP/BMP/SVG, encode PNG with alpha | `image`, `resvg` |

## Recommended Project Structure

```
BgPrunr/                          # Workspace root
├── Cargo.toml                    # [workspace] members, shared deps
├── Cargo.lock
│
├── crates/
│   ├── prunr-core/             # Shared inference library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API exports
│   │       ├── engine.rs         # ort Session management, exec providers
│   │       ├── model.rs          # ModelKind enum, embedded bytes, lazy init
│   │       ├── preprocess.rs     # resize, normalize, NCHW tensor building
│   │       ├── postprocess.rs    # sigmoid, mask threshold, alpha merge
│   │       ├── imageio.rs        # decode all formats, encode RGBA PNG
│   │       ├── batch.rs          # rayon parallel batch with progress callback
│   │       └── error.rs          # unified BgPrunrError type
│   │
│   ├── prunr-gui/              # egui/eframe desktop application
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs           # eframe::run_native entry point
│   │       ├── app.rs            # App struct impl, update() loop
│   │       ├── state.rs          # AppState, InferenceState enum
│   │       ├── worker.rs         # background thread, mpsc channels
│   │       ├── ui/
│   │       │   ├── canvas.rs     # before/after image viewer, zoom/pan
│   │       │   ├── toolbar.rs    # action buttons, keyboard shortcuts
│   │       │   ├── settings.rs   # settings dialog panel
│   │       │   └── progress.rs   # progress bar, spinner overlay
│   │       └── texture.rs        # egui TextureHandle management
│   │
│   └── prunr-cli/              # clap CLI binary
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs           # clap derive parse, dispatch
│           ├── commands/
│           │   ├── process.rs    # single image command
│           │   └── batch.rs      # batch directory command
│           └── progress.rs       # indicatif progress bar
│
└── models/                       # Source model files (not committed if large)
    ├── silueta.onnx              # ~4MB, compressed at build time
    └── u2net.onnx                # ~170MB, compressed at build time
```

### Structure Rationale

- **`crates/` subdirectory:** Keeps workspace members away from root, allows clean separation. `prunr-core` has no binary targets so it cannot accidentally gain a `main.rs`.
- **`engine.rs` separate from `model.rs`:** Session lifecycle (GPU/CPU negotiation, session pool) is independent from which model bytes to load. Swap models without touching engine code.
- **`preprocess.rs` / `postprocess.rs` split:** Preprocessing (image → tensor) and postprocessing (mask → RGBA) are distinct pipeline stages. Easier to unit-test independently, easier to tune normalization constants without touching inference.
- **`worker.rs` in GUI:** Keeps thread and channel boilerplate out of `app.rs`. The App struct only holds channel ends, not OS thread handles.
- **`ui/` subdirectory in GUI:** Each panel is its own module. `canvas.rs` is the most complex; isolating it prevents it from polluting `app.rs`.

## Architectural Patterns

### Pattern 1: Worker Thread with mpsc Channels (GUI)

**What:** The GUI never calls inference directly. It sends a `WorkRequest` message on an mpsc sender and polls a `WorkResult` receiver each frame via `try_recv()` (non-blocking). The worker thread owns the `Session` and calls `prunr-core`.

**When to use:** Any operation that blocks longer than ~16ms. Inference on CPU can take 200ms–2s. Blocking `update()` freezes the window.

**Trade-offs:** Slight complexity (channel plumbing). Gain: UI stays responsive, progress updates arrive naturally, cancellation is expressible as a message.

**Sketch:**
```rust
// In worker.rs
pub enum WorkRequest {
    Process { path: PathBuf, model: ModelKind },
    Cancel,
}

pub enum WorkResult {
    Progress(f32),
    Done(RgbaImage),
    Error(BgPrunrError),
}

pub fn spawn_worker(
    rx: Receiver<WorkRequest>,
    tx: Sender<WorkResult>,
    ctx: egui::Context,   // for ctx.request_repaint()
) -> JoinHandle<()> { ... }

// In app.rs update():
if let Ok(result) = self.result_rx.try_recv() {
    match result {
        WorkResult::Done(img) => self.state = AppState::Done(img),
        ...
    }
}
```

### Pattern 2: Lazy Session Initialization with Execution Provider Fallback

**What:** The `ort` Session is created once on first use (not at startup). Execution providers are tried in priority order: CUDA → CoreML → DirectML → CPU. The first successful provider is used; the rest are silently dropped.

**When to use:** Always — startup time must not be gated on GPU driver availability. If CUDA libs are absent the binary still works.

**Trade-offs:** First inference has initialization cost. Subsequent inferences on same session are fast (session is kept alive). No per-image session creation.

**Sketch:**
```rust
// Execution provider priority list
let providers = vec![
    #[cfg(feature = "cuda")]      CUDAExecutionProvider::default().build(),
    #[cfg(target_os = "macos")]   CoreMLExecutionProvider::default().build(),
    #[cfg(windows)]               DirectMLExecutionProvider::default().build(),
    CPUExecutionProvider::default().build(),
];

let session = Session::builder()?
    .with_execution_providers(providers)?
    .commit_from_memory(&model_bytes)?;
```

### Pattern 3: Compile-Time Model Embedding with zstd Decompression

**What:** Model ONNX bytes are embedded at compile time using `include_bytes_zstd!` macro (compression level 19). At runtime, the macro output is a `Vec<u8>` that is already decompressed by the generated code, ready to pass to `Session::commit_from_memory`.

**When to use:** For self-contained single-binary distribution where model files must not be separate downloads.

**Trade-offs:** Compile time increases (~30s extra for u2net). Binary on disk is smaller than raw bytes (zstd typical ratio ~3:1 for ONNX). Decompression at first use adds ~100ms, which is acceptable.

**Sketch:**
```rust
// In model.rs
static SILUETA_BYTES: &[u8] = include_bytes_zstd::include_bytes_zstd!(
    "../../models/silueta.onnx", 19
);

static U2NET_BYTES: &[u8] = include_bytes_zstd::include_bytes_zstd!(
    "../../models/u2net.onnx", 19
);

pub fn model_bytes(kind: ModelKind) -> &'static [u8] {
    match kind {
        ModelKind::Silueta => SILUETA_BYTES,
        ModelKind::U2Net   => U2NET_BYTES,
    }
}
```

Note: `static` with `include_bytes_zstd` works because the macro emits a `const fn` decompressor evaluated at compile time for smaller files. For u2net (170MB), the macro decompresses lazily at first access via `once_cell::sync::Lazy` pattern.

### Pattern 4: CLI Direct-Call (No Channels)

**What:** The CLI binary calls `prunr-core` functions directly on the main thread using rayon's parallel iterator for batch. Progress is reported via a closure passed to `batch_process`.

**When to use:** CLI does not have a render loop, so blocking is fine. Simpler than channels.

**Trade-offs:** Zero overhead. The same `prunr-core` API works for both GUI (async dispatch) and CLI (blocking).

**Sketch:**
```rust
// In prunr-cli batch.rs
core::batch_process(&inputs, model, options, |done, total| {
    pb.set_position(done as u64);
    pb.set_length(total as u64);
})?;
```

## Data Flow

### Single Image (GUI)

```
User drops file onto window
    ↓
app.rs: send WorkRequest::Process to worker_tx
    ↓
worker.rs: receive request, call prunr_core::process_image()
    ↓
prunr_core::imageio: decode file → DynamicImage
    ↓
prunr_core::preprocess: resize(320×320) → normalize([0.485,0.456,0.406]/[0.229,0.224,0.225]) → NCHW f32 tensor
    ↓
prunr_core::engine: Session::run(tensor) → output tensor [1,1,320,320]
    ↓
prunr_core::postprocess: sigmoid → threshold(0.5) → resize(original dims) → merge as alpha channel
    ↓
worker.rs: send WorkResult::Done(RgbaImage) to result_tx + ctx.request_repaint()
    ↓
app.rs update(): try_recv → upload pixels to egui texture via ctx.load_texture()
    ↓
ui/canvas.rs: render before/after panels using egui::Image widget
```

### Batch (CLI)

```
clap args: Vec<PathBuf> input, output dir, model, parallelism
    ↓
prunr_cli::batch: rayon::par_iter over inputs
    ↓ (N threads in parallel, one per CPU core by default)
prunr_core::process_image() per image (each call is self-contained)
    ↓
progress closure → indicatif progress bar on stderr
    ↓
write RGBA PNG to output dir
```

### Inference Tensor Flow

```
DynamicImage (RGBA u8, original size)
    ↓ convert to RGB, resize bilinear → 320×320
    ↓ per-pixel: (pixel/255 - mean) / std  [per channel]
    ↓ layout: HWC → CHW (channels first)
    ↓ shape: [1, 3, 320, 320] f32 — ONNX input tensor
          ↓
     ONNX Session::run()
          ↓
    output tensor [1, 1, 320, 320] f32
    ↓ sigmoid: 1/(1+e^-x)  (map to 0..1)
    ↓ threshold at 0.5 → binary mask
    ↓ resize mask bilinear → original (W, H)
    ↓ multiply original RGBA with mask as alpha
    ↓
DynamicImage RGBA u8 (original size, transparent background)
```

## Build Order

The Cargo dependency graph dictates compilation order:

```
1. prunr-core  (no workspace deps — builds first)
        ↓ (depends on)
2. prunr-gui   (depends on prunr-core)
   prunr-cli   (depends on prunr-core, builds in parallel with gui)
```

**Phase implications:**
- Phase 1 (core inference) must be fully functional before any GUI or CLI work begins.
- `prunr-core` should expose a stable API boundary early so GUI and CLI can be developed concurrently.
- Integration tests can be written against `prunr-core` alone before any binary targets exist.

## Anti-Patterns

### Anti-Pattern 1: Creating a Session Per Image

**What people do:** Call `Session::builder().commit_from_memory(model_bytes)` inside the per-image processing function.

**Why it's wrong:** Session initialization is expensive (~500ms on CPU, longer with CUDA init). For batch processing of 100 images this is 50 seconds of wasted overhead. Session is not reentrant by default so this also causes unnecessary lock contention.

**Do this instead:** Initialize one `Session` per model kind at startup (or lazily on first use), store it in the engine, and reuse it across all images. ONNX Runtime sessions are safe to call from multiple threads.

### Anti-Pattern 2: Blocking the egui Update Loop

**What people do:** Call `prunr_core::process_image()` directly inside `App::update()`.

**Why it's wrong:** `update()` is called on the GUI thread every frame (60fps target). Any call taking >16ms causes visible stutter. Inference takes 200ms–2s even on fast hardware. The window freezes and appears crashed to the user.

**Do this instead:** Always dispatch to the worker thread via mpsc channel. Poll `result_rx.try_recv()` (non-blocking) in `update()` and call `ctx.request_repaint()` from the worker when a result is ready.

### Anti-Pattern 3: Storing Raw Image Bytes in App State

**What people do:** Store the full `Vec<u8>` of decoded RGBA pixels in the `App` struct and re-upload to egui every frame.

**Why it's wrong:** A 4K image is 33MB of raw RGBA. Re-uploading to GPU each frame uses PCIe bandwidth and causes frame drops. egui warns against calling `load_texture` every frame for static images.

**Do this instead:** Upload once via `ctx.load_texture()` and store only the returned `TextureHandle`. The handle is cheap to clone and keeps the texture alive on the GPU. Invalidate and re-upload only when the image changes.

### Anti-Pattern 4: Feature-Flagging GPU at Runtime

**What people do:** Check for CUDA availability at runtime and conditionally load a different binary or dynamic library.

**Why it's wrong:** Complicates distribution (multiple binaries or dynamic deps), breaks the single-binary guarantee, and requires users to manage GPU driver detection themselves.

**Do this instead:** Compile with all desired execution providers linked via `ort` feature flags. Let ONNX Runtime's provider negotiation handle fallback transparently at runtime — if CUDA is unavailable the session falls through to CPU automatically.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `prunr-core` → `ort` | Direct function call (sync) | Session::run() blocks until inference done |
| `prunr-core` → `image` | Direct function call | DynamicImage conversion, resize, encode |
| `prunr-core` → `resvg` | Direct function call at decode time | SVG only; rasterize to DynamicImage immediately |
| `prunr-gui` → `prunr-core` | mpsc channel (async dispatch) | worker.rs mediates; App holds Sender/Receiver ends |
| `prunr-cli` → `prunr-core` | Direct blocking call + rayon | No channel needed; CLI blocks until batch done |
| `prunr-gui` → egui textures | `ctx.load_texture()` once, `TextureHandle` retained | Never re-upload per frame for static results |

### No External Services

Prunr has zero network integration by design. No external service boundaries exist. The only "external" data is the ONNX model bytes, which are embedded at compile time.

## Scaling Considerations

This is a desktop application, not a server. "Scaling" means handling large images and large batches gracefully.

| Concern | Small inputs | Large inputs / Long batches |
|---------|--------------|-------------------------------|
| Memory | Single image fits in RAM comfortably | 8000px+ images are ~256MB RGBA; warn and offer downscale |
| CPU batch | rayon defaults to logical core count | Expose parallelism setting; let users cap it |
| GPU VRAM | 320×320 inference tensors are tiny (~1MB) | Not a concern — inference resolution is fixed at 320×320 |
| Session memory | One session per model = ~200MB (u2net weights) | Only load the selected model; don't hold both sessions simultaneously |

## Sources

- [ORT Rust crate (pykeio/ort)](https://github.com/pykeio/ort) — ONNX Runtime bindings, execution provider architecture (MEDIUM confidence — GitHub overview only, API guide behind 403)
- [rembg pipeline description](https://github.com/danielgatis/rembg) — normalization constants, tensor format, preprocessing dimensions (MEDIUM confidence — confirmed via community implementations)
- [eframe App trait documentation](https://deepwiki.com/membrane-io/egui/5-eframe-application-framework) — lifecycle methods, state management (HIGH confidence — matches official docs.rs)
- [include-bytes-zstd crate](https://docs.rs/include-bytes-zstd) — compile-time model embedding pattern (HIGH confidence — official docs)
- [Rust mpsc channels + egui pattern](https://docs.rs/egui-async/latest/egui_async/) — non-blocking `try_recv` in update loop (HIGH confidence — Rust stdlib + egui demo code)
- [Cargo Workspaces — The Rust Book](https://doc.rust-lang.org/cargo/reference/workspaces.html) — workspace structure (HIGH confidence — official)
- [Async vs worker threads for CPU work](https://wyeworks.com/blog/2025/02/25/async-rust-when-to-use-it-when-to-avoid-it/) — CPU-bound tasks need real threads not async (HIGH confidence — well-established pattern)

---
*Architecture research for: Local AI desktop background removal (Rust/egui/ort)*
*Researched: 2026-04-06*
