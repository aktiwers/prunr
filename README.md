# BgPrunR

Local AI background removal. One binary, no cloud, no API keys.

BgPrunR removes backgrounds from images using ONNX neural networks running entirely on your machine. It ships as a single binary with embedded models — download, run, done.

## Features

- **GUI and CLI** in one binary — `bgprunr` opens the GUI, `bgprunr photo.jpg` runs headless
- **Two bundled models** — Silueta (~4 MB, fast) and U2Net (~170 MB, higher quality)
- **GPU acceleration** — CUDA (Linux/Windows), CoreML (macOS), DirectML (Windows), with automatic CPU fallback
- **Batch processing** — open multiple images, process in parallel, save all to a folder
- **True parallel processing** — start processing multiple images independently, switch between them while AI works
- **Reveal animation** — dissolve effect when background removal completes (toggle in settings)
- **Drag-and-drop** — drop images onto the window to queue them (X11; Wayland pending winit support)
- **Toast notifications** — animated feedback for save, copy, process complete, errors
- **Material Design icons** — crisp vector icons throughout the UI (via egui_material_icons)
- **Non-blocking architecture** — all image decoding, saving, and thumbnail generation runs on background threads
- **Keyboard-driven** — full shortcut set for power users
- **Cross-platform** — Linux x86_64, macOS x86_64/aarch64, Windows x86_64
- **Formats** — PNG, JPEG, WebP, BMP input; PNG output (transparent background)

## Quick Start

### Prerequisites

- Rust toolchain (1.75+)
- On Linux: GTK3 development libraries for file dialogs

### Build and Run

```bash
# 1. Fetch models (one-time, downloads ~174 MB)
cargo xtask fetch-models

# 2. Run the GUI (dev mode — loads models from filesystem)
cargo run -p bgprunr-app --features dev-models

# 3. Or build a release binary (models embedded in binary)
cargo build --release -p bgprunr-app
./target/release/bgprunr
```

## GUI

Launch with no arguments:

```bash
bgprunr
```

### Toolbar

| Button | Description |
|--------|-------------|
| **Open** | Open one or more images (multi-select supported) |
| **Remove BG** / **Remove BG Selected** | Process current image, or all checked images in parallel |
| **Process All** | Process all queued images in parallel (appears with 2+ images) |
| **Save** / **Save Selected** | Save current result, or all checked results to a folder |
| **Save All** | Save all processed images to a folder (appears with 2+ done) |
| **Remove Selected** | Remove all checked images from the sidebar (appears when any are checked) |
| **Model** | Switch between Silueta (fast) and U2Net (quality) |
| **Settings** | Open settings dialog |

### Sidebar

The sidebar appears on the right when any images are loaded. Each thumbnail shows:

- **Status indicator** (bottom-right) — gray dot (pending), pulsing purple dot (processing), green checkmark (done), red error icon
- **Selection checkbox** (top-left) — check to include in batch operations
- **Delete button** (top-right, on hover) — trash icon to remove individual image
- **Save button** (bottom-left, on hover) — save icon to export individual processed image
- **Processing animation** — purple shimmer sweep + pulsing border while AI is working
- **Loading spinner** — shown while thumbnail is being decoded in background
- **Fade-in** — 200ms fade when thumbnail loads or image switches

At the top: **Select All** checkbox and **Clear** button for quick batch selection.

Click a thumbnail to view it on the canvas with a smooth fade transition. Drag to reorder. Thumbnails update to show the result after processing.

### Settings

Open with the gear button or `Ctrl+,` (`Cmd+,` on macOS):

| Setting | Description |
|---------|-------------|
| **Model** | Silueta (fast, ~4 MB) or U2Net (quality, ~170 MB) |
| **Auto-remove on import** | Automatically process images when added to the queue |
| **Parallel jobs** | Number of images to process simultaneously (1 to CPU count) |
| **Reveal animation** | Play dissolve effect when background removal completes |
| **Inference backend** | Shows active GPU/CPU backend (read-only) |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file(s) |
| `Ctrl+R` | Remove background |
| `Ctrl+S` | Save result |
| `Ctrl+C` | Copy result to clipboard |
| `Escape` | Cancel processing / close dialog |
| `F1` | Show keyboard shortcuts |
| `B` | Toggle before/after comparison |
| `[ / ]` | Previous / next image in batch |
| `Ctrl+0` | Fit image to window |
| `Ctrl+1` | Actual size (1:1 pixels) |
| `Tab` | Show/hide batch queue sidebar |
| `Ctrl+,` | Open settings |
| `Click+drag` | Pan image |
| `Scroll wheel` | Zoom in/out (cursor-centered) |

On macOS, replace `Ctrl` with `Cmd`.

## CLI

Pass image files directly — no subcommand needed:

```bash
bgprunr photo.jpg                    # saves photo_nobg.png alongside input
bgprunr photo.jpg -o result.png      # custom output path
bgprunr *.jpg -o clean/              # batch to folder
bgprunr -m u2net portrait.jpg        # use quality model
bgprunr -j 4 *.jpg -o out/           # 4 parallel jobs
bgprunr remove photo.jpg             # backward compatible subcommand
```

### Full CLI Reference

```
bgprunr [OPTIONS] [INPUTS]...
```

| Option | Description |
|--------|-------------|
| `[INPUTS]...` | Input image file(s) |
| `-o, --output <PATH>` | Output file (single) or directory (batch) |
| `-m, --model <MODEL>` | `silueta` (default, fast) or `u2net` (quality) |
| `-j, --jobs <N>` | Parallel inference jobs (default: 1) |
| `--large-image <MODE>` | `downscale` (default) or `process` (full size) |
| `-f, --force` | Overwrite existing output files |
| `-q, --quiet` | Suppress progress output |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

### Examples

```bash
# Single image with quality model
bgprunr -m u2net portrait.jpg

# Batch with parallel jobs, force overwrite
bgprunr -j 8 -f photos/*.jpg -o clean/

# Large image at full resolution
bgprunr --large-image process poster.png

# Quiet mode for scripting
bgprunr -q photo.jpg -o output.png
```

## Project Structure

```
bgprunr/
├── crates/
│   ├── bgprunr-core/       # Inference pipeline, image I/O, batch processing
│   ├── bgprunr-models/     # Model embedding (zstd-compressed ONNX, ~174 MB)
│   └── bgprunr-app/        # Single binary: GUI (egui) + CLI (clap)
├── xtask/                   # Developer tooling (cargo xtask fetch-models)
├── models/                  # ONNX model files (.gitignored)
├── ARCHITECTURE.md          # Detailed architecture documentation
└── README.md                # This file
```

### Crate Dependencies

```
bgprunr-models  (standalone — no workspace deps)
      |
      v
bgprunr-core    (inference pipeline)
      |
      v
bgprunr-app     (GUI + CLI binary)
```

## Models

| Model | Size | Speed | Quality | Default |
|-------|------|-------|---------|---------|
| **Silueta** | ~4 MB | Fast | Good for clean subjects | Yes |
| **U2Net** | ~170 MB | Slower | Better edge detail | No |

Both models are ONNX format, compatible with the rembg Python library's preprocessing pipeline.

## GPU Acceleration

BgPrunR automatically selects the best available inference backend:

1. **CUDA** (Linux/Windows with NVIDIA GPU)
2. **CoreML** (macOS — Neural Engine on Apple Silicon)
3. **DirectML** (Windows — AMD/Intel GPUs)
4. **CPU** (always available, automatic fallback)

The active backend is shown in Settings. No configuration needed — it just works.

## Development

```bash
# Run with dev-models feature (loads models from disk, faster iteration)
cargo run -p bgprunr-app --features dev-models

# Run tests
cargo test --workspace

# Fetch models (required before first build without dev-models)
cargo xtask fetch-models
```

## License

MIT OR Apache-2.0
