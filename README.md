# BgPrunR

Local AI background removal. One binary, no cloud, no API keys.

BgPrunR removes backgrounds from images using ONNX neural networks running entirely on your machine. It ships as a single binary with embedded models — download, run, done.

## Features

- **GUI and CLI** in one binary — `bgprunr` opens the GUI, `bgprunr remove` runs headless
- **Two bundled models** — Silueta (~4 MB, fast) and U2Net (~170 MB, higher quality)
- **GPU acceleration** — CUDA (Linux/Windows), CoreML (macOS), DirectML (Windows), with automatic CPU fallback
- **Batch processing** — open multiple images, process all in parallel, save all to a folder
- **Reveal animation** — dissolve effect when background removal completes (toggle in settings)
- **Drag-and-drop** — drop images onto the window to queue them (X11; Wayland pending winit support)
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
| **Open Image** | Open one or more images (multi-select supported) |
| **Remove BG** | Remove background from current image; if checkboxes are selected, processes all selected |
| **Process All** | Process all queued images in parallel (appears with 2+ images) |
| **Save** / **Save Selected** | Save current result; if checkboxes are selected, saves all selected to a folder |
| **Save All** | Save all processed images to a folder (appears with 2+ done) |
| **Remove Selected** | Remove all checked images from the sidebar (appears when any are checked) |
| **Copy Image** | Copy result to clipboard |
| **Model** | Switch between Silueta (fast) and U2Net (quality) |
| **Settings** | Open settings dialog |

### Sidebar

The sidebar appears on the right when any images are loaded. Each thumbnail shows:

- **Status badge** (bottom-right) — gray dot (pending), pulsing blue dot (processing), green checkmark (done), red X (error)
- **Selection checkbox** (top-left) — check to include in batch operations (Remove BG Selected, Save Selected, Remove Selected)
- **Remove button** (top-right, on hover) — click X to remove individual image
- **Processing animation** — blue shimmer sweep + pulsing border while AI is working

At the top: **Select All** checkbox and **Clear** button for quick batch selection.

Click a thumbnail to view it on the canvas. Drag to reorder. Thumbnails update to show the result after processing.

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

### Remove background from a single image

```bash
bgprunr remove photo.jpg
# Output: photo_nobg.png (same directory)

bgprunr remove photo.jpg -o result.png
# Output: result.png
```

### Batch processing

```bash
bgprunr remove *.jpg --output-dir ./results/
# Output: results/photo1_nobg.png, results/photo2_nobg.png, ...

bgprunr remove *.jpg --output-dir ./results/ --jobs 4
# Process 4 images in parallel
```

### Full CLI Reference

```
bgprunr remove [OPTIONS] [INPUTS]...
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUTS...` | Input image file(s). Pass multiple paths for batch mode. |

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output file path (single-image mode only) |
| `--output-dir <DIR>` | Output directory for batch mode. Files named `{stem}_nobg.png` |
| `--model <MODEL>` | `silueta` (default, fast) or `u2net` (quality) |
| `--jobs <N>` | Parallel inference jobs for batch mode (default: 1) |
| `--large-image <MODE>` | `downscale` (default, cap at 4096px) or `process` (full size) |
| `--force` | Overwrite existing output files without prompting |
| `--quiet` | Suppress progress output (errors still go to stderr) |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

### Examples

```bash
# High-quality processing with U2Net
bgprunr remove portrait.jpg --model u2net

# Batch with parallel jobs, overwriting existing
bgprunr remove photos/*.jpg --output-dir clean/ --jobs 8 --force

# Process a very large image at full resolution
bgprunr remove poster.png --large-image process

# Quiet mode for scripting
bgprunr remove input.jpg -o output.png --quiet
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
