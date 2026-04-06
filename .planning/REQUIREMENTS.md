# Requirements: BgPrunR

**Defined:** 2026-04-06
**Core Value:** One-click local background removal that is fast, private, and works offline

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Inference

- [ ] **CORE-01**: User can remove the background from a single image and receive a transparent PNG result
- [ ] **CORE-02**: User can select between silueta (fast, ~4MB) and u2net (quality, ~170MB) models
- [ ] **CORE-03**: Inference automatically uses GPU (CUDA/Metal/DirectML) when available, falls back to CPU
- [ ] **CORE-04**: User sees a progress indicator while inference is running
- [ ] **CORE-05**: Inference pipeline produces pixel-accurate results matching rembg Python output for the same model

### GUI - Image Loading

- [ ] **LOAD-01**: User can drag and drop an image file onto the app window to load it
- [ ] **LOAD-02**: User can open an image via file browser dialog (Ctrl+O)
- [ ] **LOAD-03**: App accepts PNG, JPEG, WebP, BMP, and SVG (rasterized on load) input formats
- [ ] **LOAD-04**: User is prompted to downscale if image exceeds 8000px in either dimension

### GUI - Image Viewer

- [ ] **VIEW-01**: User can zoom in/out with scroll wheel
- [ ] **VIEW-02**: User can pan by holding Space and dragging
- [ ] **VIEW-03**: Transparency is displayed as a checkerboard pattern
- [ ] **VIEW-04**: User can toggle between original and processed image (Before/After with B key)
- [ ] **VIEW-05**: User can fit image to window (Ctrl+0) or view at actual size (Ctrl+1)

### GUI - Reveal Animation

- [ ] **ANIM-01**: When background removal completes, the removed areas dissolve/particle away while the subject stays sharp
- [ ] **ANIM-02**: The animation plays over ~0.5-1s and transitions smoothly to the checkerboard transparency view
- [ ] **ANIM-03**: User can skip the animation by pressing any key or clicking

### GUI - Output

- [ ] **OUT-01**: User can save the processed image as PNG with transparency (Ctrl+S)
- [ ] **OUT-02**: User can copy the processed image to clipboard (Ctrl+C)

### Batch Processing

- [ ] **BATCH-01**: User can drop multiple images at once; they appear in a sidebar queue
- [ ] **BATCH-02**: User can click between images in the sidebar to view each one
- [ ] **BATCH-03**: User can reorder images by dragging items in the sidebar
- [ ] **BATCH-04**: User can process all queued images at once with parallel inference
- [ ] **BATCH-05**: Results are cached — switching between images does not re-process
- [ ] **BATCH-06**: User can enable auto-remove in Settings to process images automatically on import

### CLI

- [ ] **CLI-01**: User can process a single image via `bgprunr input.jpg -o output.png`
- [ ] **CLI-02**: User can batch process via glob/directory: `bgprunr *.jpg --output-dir ./results/`
- [ ] **CLI-03**: User can select model with `--model silueta|u2net`
- [ ] **CLI-04**: User can control parallelism with `--jobs N`
- [ ] **CLI-05**: CLI exits with appropriate exit codes (0 success, 1 error, 2 partial failure in batch)

### Distribution

- [ ] **DIST-01**: Application is distributed as a single self-contained binary per platform
- [x] **DIST-02**: Both ONNX models (silueta + u2net) are embedded in the binary
- [ ] **DIST-03**: Binary runs on Linux x86_64, macOS x86_64 + aarch64, Windows x86_64
- [x] **DIST-04**: No runtime dependencies — user downloads one file and runs it

### Settings & UX

- [ ] **UX-01**: All keyboard shortcuts work as specified (Ctrl/Cmd+O, Ctrl/Cmd+R, Ctrl/Cmd+S, Ctrl/Cmd+C, B, [/], scroll, Space+drag, Ctrl/Cmd+0, Ctrl/Cmd+1, Escape, Ctrl/Cmd+,, ?) — Cmd on macOS, Ctrl on Linux/Windows
- [ ] **UX-02**: User can open settings dialog (Ctrl/Cmd+,) to configure model, auto-remove, parallelism
- [ ] **UX-03**: User can cancel in-progress inference with Escape
- [ ] **UX-04**: User can press ? to see all keyboard shortcuts
- [ ] **UX-05**: User can navigate between images with [ and ] keys

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Output

- **ENH-01**: User can choose output format (PNG, WebP with alpha)
- **ENH-02**: User can adjust mask threshold/sensitivity before saving

### Advanced Models

- **ADV-01**: Support BiRefNet model for higher-quality edges
- **ADV-02**: Support ISNet for anime/illustration content

### UX Polish

- **UXP-01**: Recent files list
- **UXP-02**: Drag-and-drop from web browser
- **UXP-03**: System tray integration for quick access

## Out of Scope

| Feature | Reason |
|---------|--------|
| Manual brush/refinement tool | Scope explosion — requires brush engine, undo/redo stack, mask compositor; full editor territory |
| Background replacement/compositing | Output transparent PNG, use design tools for compositing |
| Video background removal | Image-only for v1; video requires frame extraction pipeline and temporal consistency |
| Real-time camera feed | Desktop file processing only |
| Cloud/server processing | Violates core privacy principle |
| Web UI or embedded web server | Pure native Rust only |
| Subscription/licensing/activation | Free, no DRM |
| Custom model training | Use pre-trained ONNX models only |
| Plugin/extension system | Keep it simple and self-contained |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIST-01 | Phase 1 | Pending |
| DIST-02 | Phase 1 | Complete |
| DIST-03 | Phase 1 | Pending |
| DIST-04 | Phase 1 | Complete |
| CORE-01 | Phase 2 | Pending |
| CORE-02 | Phase 2 | Pending |
| CORE-03 | Phase 2 | Pending |
| CORE-04 | Phase 2 | Pending |
| CORE-05 | Phase 2 | Pending |
| LOAD-03 | Phase 2 (PNG/JPEG/WebP/BMP); Phase 6 (SVG via resvg) | Pending |
| LOAD-04 | Phase 2 | Pending |
| CLI-01 | Phase 3 | Pending |
| CLI-02 | Phase 3 | Pending |
| CLI-03 | Phase 3 | Pending |
| CLI-04 | Phase 3 | Pending |
| CLI-05 | Phase 3 | Pending |
| LOAD-01 | Phase 4 | Pending |
| LOAD-02 | Phase 4 | Pending |
| OUT-01 | Phase 4 | Pending |
| OUT-02 | Phase 4 | Pending |
| UX-01 | Phase 4 | Pending |
| UX-03 | Phase 4 | Pending |
| UX-04 | Phase 4 | Pending |
| VIEW-01 | Phase 5 | Pending |
| VIEW-02 | Phase 5 | Pending |
| VIEW-03 | Phase 5 | Pending |
| VIEW-04 | Phase 5 | Pending |
| VIEW-05 | Phase 5 | Pending |
| ANIM-01 | Phase 5 | Pending |
| ANIM-02 | Phase 5 | Pending |
| ANIM-03 | Phase 5 | Pending |
| BATCH-01 | Phase 5 | Pending |
| BATCH-02 | Phase 5 | Pending |
| BATCH-03 | Phase 5 | Pending |
| BATCH-04 | Phase 5 | Pending |
| BATCH-05 | Phase 5 | Pending |
| BATCH-06 | Phase 5 | Pending |
| UX-02 | Phase 5 | Pending |
| UX-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

**Note on LOAD-03:** PNG, JPEG, WebP, and BMP formats are delivered in Phase 2 (core image I/O via the `image` crate). SVG rasterization via resvg is delivered in Phase 6 alongside clean-VM distribution verification where the added dependency is tested end-to-end.

---
*Requirements defined: 2026-04-06*
*Last updated: 2026-04-06 — traceability populated after roadmap creation*
