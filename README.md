<h1><img src="img/logo-nobg.png" height="40" valign="middle">&nbsp;Prunr</h1>

Local AI background removal. One binary, no cloud, no API keys.

Prunr removes backgrounds from images using ONNX neural networks running entirely on your machine. Ships as a single binary with embedded models — download, run, done. GUI and CLI in the same executable.

Website: [prunr.io](https://prunr.io/)

<p align="center">
  <img src="img/demo.gif" alt="Prunr demo" width="720">
</p>

## Download

### Linux (x86_64)

| Format | Link |
|--------|------|
| AppImage (any distro) | [Prunr-x86_64.AppImage](https://github.com/aktiwers/prunr/releases/latest/download/Prunr-x86_64.AppImage) |
| Debian / Ubuntu | [.deb](https://github.com/aktiwers/prunr/releases/latest/download/prunr-linux-x86_64.deb) |
| Fedora / RHEL / openSUSE | [.rpm](https://github.com/aktiwers/prunr/releases/latest/download/prunr-linux-x86_64.rpm) |
| Portable | [tar.gz](https://github.com/aktiwers/prunr/releases/latest/download/prunr-linux-x86_64.tar.gz) |

The .deb / .rpm packages declare their runtime deps. If you use the
AppImage or tar.gz on a minimal install and hit a missing-library error,
install: `libgtk-3-0`, `libxkbcommon0`, `libfontconfig1`. Already present
on Ubuntu 22.04+ / Fedora 40+ / openSUSE Tumbleweed.

### macOS (Apple Silicon)

| Format | Link |
|--------|------|
| Disk image | [Prunr-macos-aarch64.dmg](https://github.com/aktiwers/prunr/releases/latest/download/Prunr-macos-aarch64.dmg) |
| Homebrew | `brew install aktiwers/prunr/prunr` |
| Portable | [tar.gz](https://github.com/aktiwers/prunr/releases/latest/download/prunr-macos-aarch64.tar.gz) |

### Windows (x86_64)

| Format | Link |
|--------|------|
| Installer | [prunr-windows-x86_64-setup.exe](https://github.com/aktiwers/prunr/releases/latest/download/prunr-windows-x86_64-setup.exe) |
| Portable | [zip](https://github.com/aktiwers/prunr/releases/latest/download/prunr-windows-x86_64.zip) |

All releases: [github.com/aktiwers/prunr/releases](https://github.com/aktiwers/prunr/releases)

## Features

- **Three bundled models** — Silueta (fast), U2Net (quality), BiRefNet-lite (best detail at 1024×1024)
- **Line extraction** — edges/outlines via DexiNed, standalone or combined with background removal
- **Object removal (Eraser)** — paint over an unwanted area and let LaMa fill it in
- **Hardware acceleration** — CUDA (NVIDIA), CoreML (Apple), DirectML (Windows), OpenVINO (Intel iGPU/NPU), automatic CPU fallback
- **Batch processing** — parallel inference across multiple images, switch between them while they work
- **Mask tuning** — removal strength, hard cutoff, edge shift, guided filter for fine detail
- **Drag-and-drop** — in and out of the window (drop into Finder/Explorer/Word/PowerPoint)
- **No cloud** — everything runs locally, no telemetry, no API keys

## CLI

The binary runs headless when given image arguments:

```bash
prunr photo.jpg                              # saves photo_nobg.png
prunr *.jpg -o clean/                        # batch to a folder
prunr -m u2net portrait.jpg                  # quality model
prunr -j 4 *.jpg -o out/                     # 4 parallel jobs
prunr --lines logo.png                       # line extraction only
prunr --inpaint photo.jpg --mask m.png       # erase area defined by mask
```

Common flags:

| Flag | Description |
|------|-------------|
| `-o, --output <PATH>` | Output file or directory |
| `-m, --model <MODEL>` | `silueta` (default), `u2net`, `birefnet-lite` |
| `-j, --jobs <N>` | Parallel jobs (default: 1) |
| `-f, --force` | Overwrite existing output |
| `--cpu` | Force CPU inference |
| `--gamma <N>` | Removal strength (default: 1.0) |
| `--threshold <N>` | Hard cutoff (0.0–1.0) |
| `--refine-edges` | Guided filter for fine detail (hair, leaves) |
| `--lines` | Extract lines/edges only |
| `--bg-color <HEX>` | Fill transparent background (e.g. `ffffff`) |
| `--inpaint --mask <PATH>` | Eraser mode — fill a masked region using LaMa |
| `--doctor` | Diagnostic dump for support tickets (hardware, ORT, models, paths) |
| `--clear-ep-cache` | Wipe persistent EP failure cache (after driver/runtime updates) |

`prunr --help` for the full list.

## Models

| Model | Size | Bundled? | Resolution | Best for |
|-------|------|---------|-----------|----------|
| Silueta | ~4 MB | yes | 320×320 | Fast, clean subjects (default) |
| BiRefNet-lite | ~214 MB | yes | 1024×1024 | Fine detail (hair, leaves) |
| DexiNed | ~140 MB | yes | full-res | Line / edge extraction |
| U2Net | ~170 MB | **on-demand** | 320×320 | Better edges than Silueta |
| LaMa (Eraser) | ~199 MB | **on-demand** | 512×512 (tiled) | Object removal — paint over a region, fill it in |
| Big-LaMa (Eraser) | ~199 MB | **on-demand** | 512×512 (tiled) | Sharper fills than LaMa, same architecture trained on more data |
| MI-GAN (Eraser) | ~26 MB | **on-demand** | 512×512 (tiled) | Lightweight GAN — sharper detail, less smooth on flat backgrounds |
| Stable Diffusion 1.5 Inpaint | ~2 GB | **on-demand** | 512×512 (tiled) | Generative inpaint — phone-app-class quality. **GPU strongly recommended.** |

The default install bundles Silueta + BiRefNet-lite + DexiNed for the common cases. Heavier models download on first use from prunr's GitHub releases. Open the model dropdown and pick **More models…** for the Model Store, where you can browse, download, view progress, retry, and delete installed models. SHA256 is verified after every download; partial files clean up automatically on cancel.

## Hardware Acceleration

Prunr picks the best available backend automatically per-model. Backends bundled or installed on-demand:

| Backend | Hardware | Bundled? | Notes |
|---|---|---|---|
| CUDA | NVIDIA GPU | yes (Linux/Windows) | Auto when `nvidia-smi` is present |
| CoreML | Apple Silicon | yes (macOS) | Neural Engine + GPU |
| DirectML | Any GPU on Windows | yes (Windows) | Vendor-agnostic |
| **OpenVINO** | **Intel CPU/iGPU/NPU** | **on-demand** | Settings → Hardware → Install (~80 MB) |
| CPU | Always | yes (everywhere) | Automatic fallback |

**OpenVINO is the big win for Intel users.** On a CPU-only or Intel-iGPU laptop, installing OpenVINO Runtime lets us route inference (most importantly Stable Diffusion) through Intel's accelerated kernels. The first launch on Intel hardware shows a one-shot prompt to install it; you can also accept later from Settings → Hardware. The runtime lives at `~/.local/share/prunr/runtimes/` (Linux) and uses ~150 MB of disk after extraction.

**Per-model EP compatibility** is declared in the model registry (`incompatible_eps` field). Models that don't load on a specific EP — e.g. Silueta's ONNX has graph cycles that OpenVINO rejects — skip that EP entirely instead of paying a failed-load cost. User-specific failures (driver bugs, etc.) are also auto-cached at `~/.local/share/prunr/ep_compat.json` and persist across runs. Wipe the cache with `prunr --clear-ep-cache` after installing new drivers or upgrading OpenVINO.

The active backend is shown in the status bar. For diagnostics, run `prunr --doctor` to dump the full hardware profile, runtime status, model availability, and resolution chain — pasteable directly into bug reports.

## Build from Source

Requires Rust 1.75+ and (on Linux) GTK3 development libraries.

```bash
cargo xtask fetch-models                          # one-time model download, ~174 MB
cargo build --release -p prunr-app                # release binary
cargo run --release -p prunr-app                  # build + run GUI

# Optional — install OpenVINO Runtime (Intel iGPU/NPU acceleration)
cargo xtask install-runtime onnxruntime-openvino 1.24.1

# Diagnostic
target/release/prunr --doctor                     # hardware + runtime status dump
target/release/prunr --clear-ep-cache             # reset EP compatibility cache
```

The app uses `ort` with `load-dynamic` — the ONNX Runtime shared library is resolved at startup from (in order):

1. `ORT_DYLIB_PATH` env var (developer override)
2. User Runtime Store: `~/.local/share/prunr/runtimes/<id>/libonnxruntime.so`
3. Bundled fallback: `<exe>/runtime/libonnxruntime.so`

Cargo-dist installers ship a CPU-only ORT in (3) so the app works out of the box; opt-in EP-specific runtimes (OpenVINO today, more on the roadmap) install via Settings → Hardware or `cargo xtask install-runtime`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for internals.

## License

Apache-2.0 — see [LICENSE](LICENSE).
