# Feature Research

**Domain:** Desktop AI background removal tool (local inference, CLI + GUI)
**Researched:** 2026-04-06
**Confidence:** HIGH (table stakes), MEDIUM (differentiators — informed by competitor analysis + user reviews)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One-click automatic background removal | Core promise of the tool; anything less is not a background remover | MEDIUM | ONNX inference pipeline; must match rembg preprocessing exactly |
| Transparent PNG output | Industry standard output format for cutouts | LOW | `image` crate alpha channel composition |
| Drag-and-drop input | Every desktop image tool supports drag-and-drop; absence feels broken | LOW | egui file drop events |
| PNG, JPEG, WebP input support | These three formats cover >95% of user image files | LOW | `image` crate handles all three natively |
| Progress indication during inference | Inference takes 0.5–5s; silent processing feels like a freeze | LOW | Progress bar or spinner in egui during ONNX call |
| Before/after comparison view | Users must be able to verify the removal worked correctly | MEDIUM | Split-slider or toggle view; standard in Clipping Magic, Photoroom, etc. |
| Zoom and pan on result | Users need to inspect edges at pixel level to evaluate quality | MEDIUM | egui viewport transform; scroll + drag |
| Batch processing (folder input) | A primary use case for photographers and e-commerce workflows | MEDIUM | rayon parallel inference; folder picker or multi-file drop |
| Output saved as file (not just clipboard) | Users need a persistent artifact they can use elsewhere | LOW | Write PNG to disk alongside source or to user-specified directory |
| Copy result to clipboard | Paste directly into design tools (Figma, Slides, etc.) without saving | LOW | Platform clipboard API via arboard crate |
| Model quality selection | Fast vs. quality trade-off is expected in any AI tool with multiple models | LOW | Settings dialog: silueta vs. u2net toggle |
| Cross-platform: Linux, macOS, Windows | Desktop tool must run on all major platforms | HIGH | Cargo cross-compilation; platform GPU backends differ |
| Keyboard shortcuts for major actions | Power users expect keyboard-driven workflows | LOW | egui key bindings: open file, process, save, copy |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Fully local / zero network access | Privacy guarantee competitors cannot match (remove.bg, Photoroom are cloud-only) | LOW | Already architecture decision; no net calls in binary |
| Single-binary distribution | No Python runtime, no pip install, no dependencies — just download and run | HIGH | `include_bytes!` model embedding + zstd decompression at startup |
| GPU acceleration with auto CPU fallback | Significantly faster for users with NVIDIA/Apple Silicon; degrades gracefully without | HIGH | `ort` execution providers: CUDA (Linux/Windows), CoreML (macOS), CPU always |
| CLI with scriptable interface | Developers and power users can integrate into shell pipelines, cron jobs, CI workflows | MEDIUM | `clap`-based CLI sharing the same core inference crate |
| SVG input support | Vector assets (logos, icons) need background removal too; rare in competitors | MEDIUM | `resvg` rasterizes SVG to pixels before inference |
| Embedded dual model (fast + quality) | Users get immediate results with silueta; can upgrade to u2net without any download | HIGH | Bakes ~174MB into binary; download-nothing UX |
| BMP input support | Legacy files from Windows Paint, scanned documents; not universally supported | LOW | `image` crate handles it |
| Large image safety guard (>8000px) | Prevents silent OOM crashes on enormous raws; builds user trust | LOW | Dimension check pre-inference; warn + offer downscale prompt |
| Parallelism control in settings | Advanced users can tune CPU core usage for background vs. foreground workloads | LOW | `rayon` thread pool size exposed in settings dialog |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Manual refinement brush (erase/restore) | Users want to fix AI mistakes without exporting to Photoshop | Scope explosion: requires separate rendering pipeline, undo/redo stack, brush engine, mask compositing — a full editor | Deliver high-quality first-pass; tell users to open result in GIMP/Photoshop for touch-ups in v1. Revisit post-validation. |
| Background replacement (color/image fill) | Obvious next step after removal | Doubles UI complexity; now you need a compositor, layer system, and image picker. Orthogonal to removal quality. | Output transparent PNG; let design tools handle compositing. Anti-feature for v1. |
| Video background removal | Visible feature request in user reviews | Fundamentally different pipeline: frame extraction, temporal consistency, encoding. Multiplies codebase by 3x. | Explicitly out of scope per PROJECT.md. Mention in README as future direction. |
| Real-time camera feed | "Live green-screen" appeal | Requires continuous capture loop, latency budget under 33ms per frame, separate UI paradigm | Out of scope per PROJECT.md. Not a desktop image tool anymore. |
| Cloud sync / account system | Persistent history, cross-device access | Violates core privacy principle; adds auth, backend, infra, GDPR concerns | Local-only. History is the user's file system. |
| Auto-update with remote download | Convenient for users | Requires network permission, signing infra, update server. Contradicts zero-network-access model. | Ship new binaries; let users download manually or via package manager. |
| Plugin/extension system | Power-user extensibility | Massively increases API surface; forces stability guarantees on internals; security considerations | Keep core small and composable via CLI. Scripting via shell is the extension system. |
| Subscription / credit model | Revenue | Creates friction, locked UX, support burden. Primary user pain point with competitors (Snapclear, remove.bg). | One-time purchase or open-source. Users explicitly choose BgPrunR to escape subscriptions. |
| AI background generation (inpainting) | "Fill with AI-generated scene" trend | Requires a generative model (Stable Diffusion class) — gigabytes of weights, totally different problem domain | Out of scope. Recommend Stable Diffusion desktop apps for that use case. |

## Feature Dependencies

```
[ONNX Inference Core]
    └──required by──> [Single-image removal]
    └──required by──> [Batch processing]
    └──required by──> [CLI mode]
    └──required by──> [GPU acceleration]
                          └──required by──> [CUDA execution provider]
                          └──required by──> [CoreML execution provider]
                          └──required by──> [CPU fallback]

[Model selection (silueta vs u2net)]
    └──requires──> [Model embedding in binary]

[Single-image removal]
    └──required by──> [Before/after comparison view]
    └──required by──> [Zoom/pan viewport]
    └──required by──> [Copy to clipboard]
    └──required by──> [Save output file]

[Drag-and-drop input]
    └──enhances──> [Single-image removal]
    └──enhances──> [Batch processing]

[Batch processing]
    └──requires──> [Single-image removal]
    └──requires──> [Progress indication]

[CLI mode]
    └──requires──> [ONNX Inference Core]
    └──conflicts──> [GUI-only features: drag-drop, zoom/pan, before/after view]

[Large image guard (>8000px)]
    └──requires──> [Single-image removal]
    └──enhances──> [Batch processing] (prevents OOM in unattended runs)

[SVG input]
    └──requires──> [resvg rasterization]
    └──enhances──> [Single-image removal]
```

### Dependency Notes

- **Batch processing requires Single-image removal:** Batch is parallelized single-image calls through rayon; the core pipeline must be stable first.
- **Before/after view requires completed removal:** The comparison view has nothing to show until inference produces a result; build inference first, add view as a UI layer on top.
- **GPU acceleration requires ONNX inference core:** The execution provider is a configuration parameter on the `ort` Session; CPU path must be validated before GPU variants.
- **Model embedding requires binary build pipeline:** The `include_bytes!` + zstd approach must be proven in a real build before dual-model selection UI makes sense.
- **CLI mode conflicts with GUI-only features:** CLI shares the inference crate but not the egui rendering crate; maintain clean crate boundary so core is GUI-free.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] ONNX inference pipeline (silueta model, CPU) — the entire product is built on this
- [ ] Single-image background removal from file picker and drag-and-drop
- [ ] Transparent PNG output written to disk
- [ ] Progress indication during inference
- [ ] Before/after toggle or split view
- [ ] Zoom and pan on result
- [ ] Copy result to clipboard
- [ ] CLI mode (`bgrprunr remove <input> <output>`)
- [ ] Keyboard shortcuts for open, process, save, copy
- [ ] Cross-platform builds: Linux x86_64, macOS aarch64, Windows x86_64
- [ ] GPU acceleration with CPU fallback (CUDA + CoreML)
- [ ] u2net model bundled alongside silueta (quality option in settings)
- [ ] Large image warning and downscale prompt
- [ ] Batch processing with folder input and progress bar

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Parallelism tuning in settings — add when power users report CPU saturation
- [ ] BMP input — trivial to add, defer until someone requests it
- [ ] SVG input via resvg — add if users report using vector source files
- [ ] Settings persistence (last-used model, output directory) — add after dog-fooding reveals the pain

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Manual refinement brush — only build if user feedback consistently shows AI quality is insufficient for their use case
- [ ] Background replacement — only if users demonstrate the transparent PNG handoff is too much friction
- [ ] Additional ONNX models (BiRefNet, SAM) — evaluate quality vs. binary size trade-off after shipping

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| ONNX inference (silueta, CPU) | HIGH | HIGH | P1 |
| Single-image removal + save | HIGH | LOW | P1 |
| Drag-and-drop input | HIGH | LOW | P1 |
| Progress indication | HIGH | LOW | P1 |
| Before/after comparison view | HIGH | MEDIUM | P1 |
| Zoom/pan viewport | HIGH | MEDIUM | P1 |
| Copy to clipboard | HIGH | LOW | P1 |
| Keyboard shortcuts | MEDIUM | LOW | P1 |
| CLI mode | HIGH | MEDIUM | P1 |
| Cross-platform builds | HIGH | MEDIUM | P1 |
| GPU acceleration + CPU fallback | HIGH | HIGH | P1 |
| Batch processing | HIGH | MEDIUM | P1 |
| u2net model + model selection | MEDIUM | MEDIUM | P1 |
| Single-binary distribution | HIGH | HIGH | P1 |
| Large image guard | MEDIUM | LOW | P1 |
| SVG input (resvg) | MEDIUM | LOW | P2 |
| Parallelism settings | LOW | LOW | P2 |
| BMP input | LOW | LOW | P2 |
| Settings persistence | MEDIUM | LOW | P2 |
| Manual refinement brush | HIGH | HIGH | P3 |
| Background replacement | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | rembg (Python) | Snapclear (desktop) | remove.bg (cloud) | BgPrunR approach |
|---------|---------------|---------------------|-------------------|-----------------|
| Local inference | Yes | Yes | No (cloud-only) | Yes — core differentiator |
| CLI | Yes | No | API only | Yes — shared core crate |
| GUI | Community forks only | Yes | Web UI | Yes — egui native |
| Drag-and-drop | No | Yes | Web drop | Yes |
| Batch processing | Yes (folder mode) | Yes | Paid tier | Yes — rayon parallel |
| Multiple models | 15+ (auto-download) | Unknown | 1 (proprietary) | 2 bundled (silueta + u2net) |
| Single binary | No (Python deps) | Yes (installer) | N/A | Yes — no installer needed |
| GPU acceleration | CUDA + ROCm | Unknown | Cloud-side | CUDA + CoreML + CPU fallback |
| SVG input | No | No | No | Yes — resvg |
| Zero network access | Partial (model download on first run) | Partial (activation on first launch) | No (cloud required) | Yes — fully air-gapped |
| Open source | Yes (MIT) | No (proprietary) | No | Yes — pure Rust |
| Before/after view | No | Unknown | Yes | Yes |
| Subscription / credits | No | Yes (10 free/month) | Yes (credits) | No — one-time or free |
| Clipboard copy | No | Unknown | No | Yes |

## Sources

- [rembg GitHub (danielgatis/rembg)](https://github.com/danielgatis/rembg) — feature set, model list, CLI flags
- [Snapclear.app](https://www.snapclear.app/) — offline desktop competitor feature set
- [remove.bg desktop app](https://www.remove.bg/a/background-remover-windows-mac-linux) — cloud desktop wrapper, batch features
- [Removedo: AI background remover with manual editing](https://www.removedo.com/blog/ai-powered-background-remover-with-manual-editing-options) — refinement brush patterns
- [Best offline background removers 2026 (MadFable)](https://www.madfable.com/blog/best-background-remover-offline-pc) — user expectations for offline tools
- [Background removal software with batch processing (Removedo)](https://www.removedo.com/blog/background-removal-software-with-batch-processing) — batch UX patterns
- [Clipping Magic](https://clippingmagic.com/) — zoom/pan, before/after, undo/redo patterns
- [Photoroom technology comparison](https://www.photoroom.com/blog/image-background-removal-technology-comparison) — zoom, manual refinement patterns
- [User complaints: remove.bg reviews (SaaSWorthy)](https://www.saasworthy.com/product/remove-bg) — batch processing pain points, subscription complaints
- [rembg vs Background Remover comparison 2026](https://www.backgroundremover.com/blog/rembg-vs-background-remover) — open-source vs. cloud trade-offs

---
*Feature research for: Desktop AI background removal tool (BgPrunR)*
*Researched: 2026-04-06*
