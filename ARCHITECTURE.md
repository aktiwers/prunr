# Prunr Architecture

Living document describing how Prunr is built. For user-facing info see [README.md](README.md).

## Design Principles

1. **The UI thread never waits** — inference, file I/O, decoding, PNG encoding, texture prep all run on background threads or subprocesses. All communication is via `mpsc` channels drained non-blockingly each frame.
2. **Single binary** — one `prunr` executable hosts GUI, CLI, and subprocess worker (`--worker`). Models are embedded as zstd blobs decompressed at runtime.
3. **Platform parity** — every feature works on Linux x86_64, macOS aarch64, and Windows x86_64. Platform-specific code is isolated behind `#[cfg(...)]`.
4. **Progressive performance** — start fast, improve in place. Use real hardware (GPU/NE) when available, fall back gracefully to CPU, never block startup on compilation.
5. **Crash isolation** — AI inference runs in a subprocess. If it OOMs, the main app survives, re-queues work with reduced concurrency, and continues.

## Workspace

```
prunr/
├── crates/
│   ├── prunr-models/       # Embedded zstd-compressed ONNX blobs + runtime decompression
│   ├── prunr-core/         # Inference pipeline, image I/O, batch processing
│   └── prunr-app/          # Single binary: GUI (eframe/egui) + CLI (clap) + subprocess worker
│       └── src/
│           ├── main.rs             # Entry point: --worker / CLI / GUI dispatch
│           ├── cli.rs              # CLI batch processing
│           ├── worker_process.rs   # Subprocess child entry point
│           ├── lib.rs              # Library root (gui + subprocess modules)
│           ├── subprocess/         # IPC protocol, framing, parent-side manager
│           │   ├── protocol.rs     # SubprocessCommand / SubprocessEvent types
│           │   ├── ipc.rs          # Length-prefixed bincode framing
│           │   └── manager.rs      # SubprocessManager (spawn, send, poll, kill)
│           └── gui/                # GUI application
│               ├── app.rs             # PrunrApp coordinator (UI flags + handles to coordinators)
│               ├── item.rs            # Pure data types: BatchItem, HistorySlot, ImageSource, …
│               ├── history_manager.rs # Per-item undo/redo + preset history (no own state)
│               ├── drag_export_state.rs # OS drag-out lifecycle state (4 fields + reset)
│               ├── batch_manager.rs   # batch items, selection, bg_io, memory governance, decode/thumb requests
│               ├── processor.rs       # worker channels, cancel flag, admission, live preview, dispatch state
│               ├── item_settings.rs   # Per-image processing settings
│               ├── live_preview.rs    # In-process Tier 2 dispatcher
│               ├── presets_fs.rs      # On-disk preset store (one JSON per preset)
│               ├── worker.rs          # Bridge thread (subprocess ↔ app)
│               ├── memory.rs          # Admission control, RSS monitoring
│               ├── history_disk.rs    # Cold-tier history on disk
│               ├── settings.rs        # App-wide settings
│               └── views/             # UI components (canvas, sidebar, toolbar,
│                                      #   adjustments_toolbar, chip, lines_popover,
│                                      #   preset_dropdown, settings, …)
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

## Process Architecture

Prunr uses a **two-process model** for batch inference, inspired by Chrome's renderer isolation:

```
Parent process (GUI/CLI)                  Child process (prunr --worker)
┌──────────────────────┐                 ┌──────────────────────┐
│ PrunrApp             │   stdin/stdout  │ worker_process       │
│                      │   (bincode)     │                      │
│ Bridge thread ───────┼────────────────►│ Read commands        │
│   - translate msgs   │                 │ Load ORT engine pool │
│   - retry on crash   │◄───────────────┤│ Process images       │
│   - RSS-based pace   │                 │ Report RSS           │
│                      │   exit code     │                      │
│ Crash → re-queue ◄───┼────────────────┤│ OOM → process dies   │
│   reduce concurrency │                 └──────────────────────┘
│   spawn new child    │
└──────────────────────┘
```

### Why subprocess isolation?

ONNX Runtime allocates unpredictable amounts of memory at runtime (varies by model, input size, graph optimization, arena behaviour). Static memory estimates are always wrong. The subprocess model means:
- **OOM kills only the child** — the parent stays responsive, the desktop never freezes
- **Auto-retry with reduced concurrency** — crash at 4 jobs → retry at 2 → retry at 1
- **Memory can only be fully reclaimed by process exit** — ORT's arena allocator + CUDA/CoreML contexts keep allocations alive for the lifetime of the session, so process exit is the only guaranteed reclaim mechanism

### IPC Protocol

Communication uses **length-prefixed bincode frames** over stdin/stdout:

| Direction | Message | Purpose |
|-----------|---------|---------|
| Parent → Child | `Init` | Load model, create engine pool |
| Parent → Child | `ProcessImage` | Full pipeline: decode + infer + postprocess |
| Parent → Child | `RePostProcess` | Tier 2 mask rerun from cached tensor (batched reruns only — live preview runs in-process) |
| Parent → Child | `Cancel` / `Shutdown` | Graceful stop |
| Child → Parent | `Ready` | Engines loaded |
| Child → Parent | `Progress` | Per-stage progress (Decode, Infer, etc.) |
| Child → Parent | `ImageDone` | Result + optional seg-tensor path and DexiNed-tensor path for Tier 2 caches |
| Child → Parent | `ImageError` | Non-fatal error |
| Child → Parent | `RssUpdate` | Current process RSS (for admission throttling) |

**Image data transfer:** Large payloads (image bytes, result RGBA, raw tensors) go via temp files, not through the pipe. On Linux, temp files are placed in `/dev/shm/prunr-ipc-{pid}/` (RAM-backed tmpfs — zero disk I/O). On Windows/macOS, `std::env::temp_dir()` is used. Cancel path calls `cleanup_ipc_temp()` to prevent tmpfs leaks.

## Threading Model

| Thread / Process | Spawned | Lives for | Purpose |
|------------------|---------|-----------|---------|
| UI (egui render loop) | main | app lifetime | Render frames, handle input, drain channels |
| Bridge thread | startup | app lifetime | Receives `WorkerMessage`, spawns subprocess, translates IPC, handles crash+retry |
| Subprocess (prunr --worker) | per batch | one batch | Loads ORT engines, processes images, reports RSS |
| Subprocess reader thread | per subprocess | subprocess lifetime | Reads child stdout events non-blockingly |
| File loader | per drag-drop / open | until paths sent | Sends `(PathBuf, name)` via mpsc (lazy — no file read) |
| Image decoder | on demand | ~20-80ms | `image::load_from_memory` → `Arc<RgbaImage>` |
| Thumbnail builder | per imported image | ~5-50ms | Lanczos resize to 160px (bg fill happens at render time, not here) |
| Texture prep | per canvas texture | ~5-50ms | Builds `egui::ColorImage` off the UI thread (transparent — bg is render-time) |
| Live-preview rayon job | per tweak dispatch | ~20-500ms | Tier 2 postprocess or edge rerun; shared rayon pool |
| Save writer | per save | until PNG written | Background PNG encode + `fs::write` (with bg composited) |
| Temp file cleanup | app exit + periodic | brief | Removes `prunr-drag/*`, `prunr-history/*`, `prunr-ipc/*` |

### Engine pooling (in subprocess)

| Backend | Pool size | Intra-op threads |
|---------|-----------|------------------|
| CPU | user's `parallel_jobs` setting | `num_cpus / pool_size` |
| GPU (CUDA/DirectML/CoreML) | `min(jobs, 2)` | `num_cpus / pool_size` |

GPU is capped at 2 engines — more doesn't help because the GPU driver serializes anyway. CPU respects the user's setting; the admission controller manages overall memory pressure.

**Postprocess serialization:** A global `POSTPROCESS_LOCK` mutex in the subprocess ensures only one image runs the CPU-intensive Lanczos3 resize at a time. This prevents concurrent resize spikes from causing OOM when multiple engines finish inference simultaneously.

**ORT CPU arena disabled:** `CPUExecutionProvider::with_arena_allocator(false)` reduces baseline memory. The subprocess isolation handles any resulting OOM.

### Retry flow on crash

```
Attempt 1: N engines → crash → re-queue ALL in-flight images
Attempt 2: N/2 engines → success → continue at N/2
         or → crash → re-queue ALL in-flight
Attempt 3: 1 engine → success → continue at 1
         or → crash → mark remaining images as "insufficient memory"
```

The parent shows a toast: "Memory pressure — retrying X images with Y parallel jobs". Crash diagnostics detect the exit signal (SIGKILL = "Process killed by OS (out of memory)", SIGSEGV = "segmentation fault") for user-facing messages.

## Tiered Recipe Pipeline

Re-processing avoids redundant work by classifying each change into a tier:

| Tier | Name | When | Cost |
|------|------|------|------|
| 1 | FullPipeline | Model, line mode, or chain changed | Full inference |
| 2a | MaskRerun | Mask params changed (gamma, threshold, edge_shift, refine_edges) | Postprocess cached seg tensor (~50-200ms) |
| 2b | EdgeRerun | Line params changed (line_strength, solid_line_color) | Re-threshold cached DexiNed tensor (~20-100ms) |
| 3 | CompositeOnly | bg_color changed | Render-time GPU rect — zero CPU |
| — | Skip | Recipe identical | No work |

Each `BatchItem` stores the recipe that produced its current result, snapshotted at dispatch time — so settings changes mid-batch can't corrupt the stored tag.

**Two independent tensor caches.** A `BatchItem` holds a segmentation tensor (for MaskRerun) and a DexiNed tensor (for EdgeRerun) as separate zstd-compressed fields — a model swap invalidates only the seg cache, a line_mode change only the edge cache. Combined budget: 512 MB, oldest-evicted first.

**Two Tier 2 paths.** Batched reruns (e.g. preset applied across many selected items) go through the subprocess. Live-preview reruns run in-process on the rayon pool to avoid IPC overhead — see [Live Preview](#live-preview).

## Per-Image Settings

Each `BatchItem` owns its own processing settings, so tweaking the adjustments toolbar edits one image instead of broadcasting to the whole batch. App-wide config (parallel_jobs, chain_mode, live_preview, etc.) stays separate on `AppSettings`. Per-image settings are `Copy`-sized and forward-compatible: older preset files load cleanly when new fields are added.

## Live Preview

Mask and edge tweaks auto-rerun Tier 2 during slider drag. A tweak is debounced ~150 ms; a new tweak on the same item cancels the in-flight one and dispatches a fresh rerun on the rayon pool.

**Why in-process.** The subprocess path costs ~20-50 ms per rerun in IPC alone; live preview needs 60 fps feel during drag. Batched Tier 2 reruns still go through the subprocess — that overhead amortizes across many images.

**Preview trades quality for speed.** Guided-filter edge refinement is skipped in preview and restored on commit. Cancel drops the result but doesn't interrupt the CPU pipeline mid-run — a dispatch started just before cancel completes and its output is discarded.

## Presets

A preset is a named snapshot of per-image settings, stored as one JSON file per preset in the platform config dir. Human-readable and self-contained: a user sends a `.json` to a friend, the friend drops it in the folder, the preset appears in their dropdown next launch.

A reserved `"Prunr"` entry is the factory default (cannot be overwritten or deleted). Each `BatchItem` carries preset undo/redo stacks (Ctrl+Shift+Z/Y) separate from image-result history, so rolling back an accidental preset swap doesn't touch pixels.

## Memory Management

### Three-tiered history cache

Undo/redo history uses a tiered strategy to bound RAM while preserving Ctrl+Z:

| Tier | Storage | Size vs raw | Access time | When |
|------|---------|-------------|-------------|------|
| Hot | `Arc<RgbaImage>` in RAM | 100% | instant | Currently viewed result |
| Warm | Zstd-compressed `Vec<u8>` in RAM | ~25-30% | ~8ms decompress | Non-visible history entries |
| Cold | Zstd file on disk (`~/.cache/prunr/history/`) | 0% RAM | ~50-100ms | Under memory pressure |

History seeding is lazy: for images that haven't been decoded yet (lazy file loading), the seed is skipped at process time and created on demand during the first undo. This eliminates UI freezes when processing large batches.

History entries compress to Tier 2 (warm) by default. Demotion to Tier 3 (cold) happens automatically when the subprocess reports high RSS via `under_memory_pressure()`.

Each history entry carries its recipe alongside the pixels, so undoing restores the recipe too — a subsequent reprocess can tier-route correctly from the restored state.

### Lazy file loading (ImageSource)

Batch items store file paths, not bytes:

```rust
pub(crate) enum ImageSource {
    Path(PathBuf),           // File-opened images — zero RAM until processed
    Bytes(Arc<Vec<u8>>),     // Clipboard/paste — bytes already in memory
}
```

74 images at idle: ~0 MB instead of ~700 MB. Bytes are read on demand when the admission controller admits an image for processing.

### Result eviction

Non-visible processed results are compressed to Tier 2 (warm) on sidebar navigation. When the user clicks back, the result is decompressed from the compressed history entry (~8ms). Thumbnails (160px, ~100KB each) remain in RAM.

### Admission controller

The `AdmissionController` uses a **sliding window with greedy best-fit** to pace how many images are sent to the subprocess:

1. Estimate per-image cost from dimensions: `W × H × 4 × 2 + file_size`
2. Query available RAM, subtract model overhead
3. Admit largest pending image that fits remaining budget
4. On each `ImageDone`: release budget, admit next
5. Force-admit if nothing is in-flight (prevents deadlock on oversized images)

Additionally, the subprocess reports its own RSS after each image. The parent pauses admission when child RSS exceeds 80% of available RAM, resumes at 70% (hysteresis).

### Model-aware parallel jobs

`Settings::max_jobs()` limits the settings slider based on available RAM and selected model. Switching to a heavier model auto-clamps `parallel_jobs` if needed. At batch start, `safe_max_jobs()` provides a final safety clamp.

### Zero-copy model bytes

Model decompression happens once (via `OnceLock<Vec<u8>>`). All callers receive `&'static [u8]` — no 250 MB clones per engine creation. This eliminated ~2 GB of redundant allocations that previously caused OOM at "Loading model 0%".

## Data Flow

### Batch GUI processing (subprocess path)

```
User opens 74 images
  ↓ (UI thread)
add_to_batch_path(PathBuf) → BatchItem with ImageSource::Path (zero RAM)
  ↓ (sidebar thumbnail generation on background thread)

User clicks "Process All"
  ↓ (UI thread: process_items)
evict_all_tensors() — clear stale tensor caches
For each item: resolve_tier(applied_recipe, current_recipe) → Skip/Tier2/MaskRerun/FullPipeline
dispatch_recipe = snapshot of current settings
Tier 1 items: AdmissionController estimates costs, admits initial window
Tier 2 items: dispatched directly as Tier2WorkItem (no admission needed)
  ↓ (bridge thread)
Spawn `prunr --worker` subprocess (if Tier 1 work exists)
Send Init { model, jobs, ProcessingConfig }
  ↓ (subprocess)
create_engine_pool → ORT sessions loaded
Ready { active_provider }
  ↓ Tier 1: bridge sends ProcessImage
  ↓ Tier 2: bridge sends RePostProcess (tensor from BatchItem.cached_tensor)
  ↓ (subprocess processes one at a time, locked by POSTPROCESS_LOCK)
ImageDone { result_path, tensor_cache_path?, ... }
  ↓ (bridge reads result, reads tensor if present → compresses → WorkerResult::BatchItemDone)
item.result_rgba = Some(Arc::new(rgba))
item.cached_tensor = Some(CompressedTensor)     ← Tier 1 only
item.applied_recipe = dispatch_recipe           ← snapshot, not current settings
  ↓ (background texture prep → canvas crossfade)

If subprocess crashes:
  → Bridge re-queues ALL in-flight items
  → Reduces concurrency (4 → 2 → 1)
  → Spawns new subprocess
  → Continues from where it left off
```

### Chain mode

When "chain mode" is on, processing feeds the previous result as input. The bridge thread writes the chain input RGBA to a temp file (in `/dev/shm` on Linux) and passes the path in `ProcessImage`. The subprocess reads it, wraps in `DynamicImage`, and passes to `process_image_from_decoded()`.

### CLI

```
main.rs
  ↓ Cli::parse()
  ↓ inputs non-empty → cli::run_remove(&cli)
  ↓
single image: pipeline::process_image_with_mask() → PNG encode → fs::write (in-process)
batch (2+ images): spawn `prunr --worker` subprocess → IPC → results → PNG encode → fs::write
  ↓
exit code: 0 all ok / 1 all fail / 2 partial
```

All CLI processing (single and batch) uses subprocess isolation for OOM protection. File paths are passed directly to the subprocess — image bytes are never loaded into the parent process. Only oversized images that need downscaling are loaded temporarily. Auto-retry with concurrency reduction (4→2→1) on OOM.

## Inference Pipeline

```
1.  load_image_from_bytes()             → DynamicImage
2.  check_large_image()                 → error if > 8000px (configurable via --large-image)
3.  preprocess(img, model)              → Array4<f32> [1, 3, H, H] NCHW
      - Silueta/U2Net: divide by max_pixel, ImageNet normalize
      - BiRefNet-lite: divide by 255, ImageNet normalize
4.  engine.with_session(|s| s.run(...)) → raw output tensor [1, 1, H, H]  ← Tier 1 only
5.  postprocess(raw_output.view(), img, mask, model):
      - Allocates RGBA once (shared by guided filter + mask application)
      - Normalize to [0, 1] (min-max for Silueta/U2Net, sigmoid for BiRefNet)
      - Short-circuit uniform output (skip per-pixel loop)
      - Apply gamma + optional hard threshold
      - Resize mask to original dimensions (SIMD Lanczos3 via `fast_image_resize`)
      - Optional: apply_edge_shift (morphological erode/dilate)
      - Optional: guided_filter_alpha (O(1) box filter)
      - Write mask as alpha channel into the shared RGBA buffer
6.  Optional: apply_background_color()

Tier 2 path uses postprocess_from_flat(tensor: &[f32], h, w, original, mask, model)
  → reshapes flat bytes from IPC into ArrayView4 → calls postprocess() directly
  → eliminates inference entirely
```

### Postprocess fast paths

- All Lanczos3 resizes use `fast_image_resize` (SSE4.1, AVX2, NEON) — 10-20x faster than `image` crate
- Division by range → precomputed `inv_range` multiplier
- Uniform-output detection before the per-pixel loop
- Alpha composition row-parallel via `par_chunks_mut` above 256k pixels (memory-bandwidth-bound, so the ceiling is ~1.1-1.2× on 4K regardless of core count)
- Guided filter uses `f32` prefix sums (halved bandwidth vs f64)
- Edge shift's ring buffers allocated once, swapped via `std::mem::swap`
- Single RGBA allocation in `postprocess()` — shared across guided filter and mask application (saves ~48 MB per Tier 2 run on a 4000×3000 image)

#### Phase 10-07 numbers (4000×3000 image, 8-core x86_64, `cargo test --release`)

| Stage                         | Serial (median) | Row-parallel (median) | Speedup |
|-------------------------------|----------------:|----------------------:|--------:|
| `apply_mask_inplace` (alone)  |          5.68ms |                4.83ms |   1.18× |
| `postprocess_from_flat` (E2E) |         94.45ms |               89.80ms |   1.05× |

End-to-end delta is small because the mask-to-alpha loop is ~5-6% of
total postprocess time (the dominant costs are the 48 MB RGBA allocation
and the SIMD Lanczos resize). The `apply_mask_inplace_4k_bench` and
`postprocess_4k_bench` `#[ignore]` tests in `postprocess.rs` reproduce
these numbers.

## Edge Detection (DexiNed)

A separate `EdgeEngine` handles line extraction with its own ONNX session. Three line modes:

| Mode | User-facing label | Behaviour |
|------|-------------------|-----------|
| `Off` | "Off" | Normal background removal only |
| `EdgesOnly` | "Edges only (full image)" | Skip segmentation, run DexiNed on original image |
| `SubjectOutline` | "Outline only (no fill)" | Segmentation first, then DexiNed on the result — edges only within subject, body transparent |

Inference is split from the threshold-and-composite step so the raw DexiNed tensor can be cached and re-thresholded without re-running the model. This powers EdgeRerun Tier 2.

## GPU Execution Providers

```rust
if cpu_only {
    CPUExecutionProvider::default()
        .with_arena_allocator(false)  // lower memory; subprocess handles OOM
        .build()
} else {
    CUDAExecutionProvider::default()
        .with_arena_extend_strategy(SameAsRequested)
        .with_cuda_graph(true)
        .with_tf32(true)
        .build(),
    CoreMLExecutionProvider::default()
        .with_model_cache_dir(coreml_cache_dir())
        .build(),
    DirectMLExecutionProvider::default().build(),
    CPUExecutionProvider::default()
        .with_arena_allocator(false)
        .build(),
}
```

### Model variants

`OrtEngine::new_with_fallback()` tries the optimized variant first, falls back to FP32:

| Platform | Preferred | Fallback |
|----------|-----------|----------|
| Linux/Windows GPU | FP16 | FP32 |
| Linux/Windows CPU | INT8 | FP32 |
| **macOS (all)** | **FP32 always** | n/a |

macOS uses FP32 because CoreML silently converts to FP16 internally — feeding our FP16 stacks two conversions, causing precision loss.

### Model bytes cache

Decompressed ONNX bytes are cached in `OnceLock<Vec<u8>>` per model. Callers receive `&'static [u8]` (zero-copy borrow). Previously every engine creation cloned ~250 MB — now it borrows.

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

## Canvas & Texture Lifecycle

- Textures built on background threads via `spawn_tex_prep()` — `ColorImage::from_rgba_unmultiplied()` runs off the UI thread
- Previous texture stays visible until the new one is ready (no flash on sidebar switch)
- Zoom resets only on explicit user navigation (sidebar click, arrow keys), not on background texture arrivals
- Checkerboard behind transparent results: single 256×256 pre-generated texture, tiled
- Off-screen sidebar items skip painting entirely (viewport virtualization)
- `result_switch_id` is used as the animation seed for the result crossfade

### Render-time bg_color fill

**bg_color is never composited into pixels for display.** The result texture stays transparent where pixels were removed; at draw time the canvas paints a filled rect (or the checkerboard) *behind* the texture, and the GPU alpha-blends the result on top. Changing the bg color costs one rect repaint — no CPU compositing, no texture rebuild, no subprocess dispatch.

The sidebar thumbnail uses the same pattern. Save/export is the exception: PNG has no separate canvas-bg concept, so the bg is composited into pixels on demand at save time. Display and export paths diverge intentionally.

## Drag-Out (OS-level drag to external apps)

Implemented via the `drag` crate (Windows/macOS only). Files written to `temp_dir/prunr-drag/*.png`.

Self-drop rejection prevents re-ingesting thumbnails. Stuck-drag recovery clears state when the drag callback doesn't fire.

## Persistent Config

User data lives in the platform config dir (`dirs::config_dir()`):

| File / Folder | Linux | macOS | Windows |
|---|---|---|---|
| `settings.json` | `~/.config/prunr/` | `~/Library/Application Support/prunr/` | `%APPDATA%\prunr\` |
| `presets/*.json` | `~/.config/prunr/presets/` | `~/Library/Application Support/prunr/presets/` | `%APPDATA%\prunr\presets\` |

## Temp File Lifecycle

| Directory | Purpose | Created by | Cleaned by |
|-----------|---------|------------|------------|
| `{temp_dir}/prunr-drag/` | OS drag-out PNG files | drag_export | Stale at startup (>10 min), all at exit |
| `{cache_dir}/prunr-history/` | Tier 3 history (cold) | history_disk | Stale at startup (>30 min), periodic (10 min), all at exit |
| `/dev/shm/prunr-ipc/` (Linux) or `{temp_dir}/prunr-ipc/` | Subprocess image transfer | manager/worker_process | Per-image after read, all on new subprocess spawn |

## Windows-Specific: Console Subsystem

Release builds use `#[windows_subsystem = "windows"]`. CLI mode calls `AttachConsole(ATTACH_PARENT_PROCESS)`. GUI has a renderer fallback chain: glow (OpenGL) → wgpu (DX12/Vulkan).

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

### GitHub Actions release pipeline

Tag push (`v*`) triggers parallel builds on three runners:

| Matrix target | Runner | Artifacts |
|---------------|--------|-----------|
| linux-x86_64 | ubuntu-latest | `.tar.gz`, `.AppImage`, `.deb`, `.rpm` |
| macos-aarch64 | macos-latest | `.dmg`, `.tar.gz` |
| windows-x86_64 | windows-latest | `.zip`, Inno Setup `.exe` |

### Version sync

Workspace `Cargo.toml` `version` is the single source of truth. CLI reads `CARGO_PKG_VERSION`; platform packages read the git tag.

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ort` | 2.0.0-rc.12 | ONNX Runtime bindings |
| `eframe` / `egui` | 0.34 | Immediate-mode GUI + windowing |
| `image` | 0.25 | PNG/JPEG/WebP/BMP decode/encode |
| `ndarray` | 0.17 | ORT-compatible tensor manipulation |
| `rayon` | 1.11 | Work-stealing parallelism |
| `zstd` | 0.13 | Model decompression + history compression |
| `bincode` | 2 | Subprocess IPC serialization |
| `sysinfo` | 0.37 | Cross-platform available RAM query |
| `memory-stats` | 1 | Process RSS monitoring (subprocess self-reporting) |
| `fast_image_resize` | 6 | SIMD-accelerated Lanczos3 resize (SSE4.1/AVX2/NEON) |
| `clap` | 4.5 | CLI argument parsing |
| `serde` / `serde_json` | 1.x | Settings + IPC serialization |
| `rfd` | 0.15 | Native file dialogs |
| `arboard` | 3.x | Cross-platform clipboard |
| `dirs` | 6.x | Platform config/cache dirs |
| `drag` | 2.1 | OS drag-out (Windows/macOS only) |
