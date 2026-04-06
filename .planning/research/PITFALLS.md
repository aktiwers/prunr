# Pitfalls Research

**Domain:** Rust desktop AI inference app — ONNX Runtime + egui image processing
**Researched:** 2026-04-06
**Confidence:** HIGH (most pitfalls verified via official docs, ort source, egui issue tracker, ONNX Runtime docs)

---

## Critical Pitfalls

### Pitfall 1: Preprocessing Pipeline Mismatch Against rembg

**What goes wrong:**
Output masks look noisy, fail to segment anything meaningful, or return all-white/all-black results. The model runs without errors but produces garbage. This is the single most likely cause of "it runs but doesn't work" during initial integration.

**Why it happens:**
The U2-Net and silueta ONNX models were exported expecting a very specific preprocessing contract: resize to 320×320, convert to float32, scale pixels from [0, 255] to [0.0, 1.0], then normalize per-channel using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), and arrange as a NCHW tensor (batch=1, channels=3, height=320, width=320). Any deviation — wrong channel order, wrong normalization order, wrong axis layout, values not divided by 255 first — produces silently incorrect output. The Rust `image` crate returns RGB pixel data, but it is easy to accidentally pass HWC byte buffers directly without transposing to CHW, or to normalize before dividing by 255.

**How to avoid:**
- Implement preprocessing in a single, tested pure function with no side effects.
- Write a unit test that runs the preprocessing on a known reference image, feeds it through the ONNX session, and compares the output mask pixel-by-pixel against rembg's Python output for that same image. This test catches any future regression.
- Follow this exact order: decode → resize to 320×320 (bilinear) → convert to f32 → divide by 255.0 → subtract mean → divide by std → transpose HWC → CHW → add batch dim → create ONNX tensor.
- Verify the tensor shape is `[1, 3, 320, 320]` (not `[1, 320, 320, 3]`) before the session run call.

**Warning signs:**
- Output mask is a uniform gray, white, or black image regardless of input.
- Output has correct shape but no contrast (all values near 0.5).
- Results vary randomly between runs when input is identical (indicates a buffer reuse/aliasing bug).
- Worked on one test image but fails on images of different aspect ratios (indicates resize mode is wrong or crops instead of squeezing).

**Phase to address:** Core inference engine phase (the first phase that runs any model against real images). The reference-output unit test must pass before any GUI work begins.

---

### Pitfall 2: GPU Execution Provider Silently Falls Back to CPU

**What goes wrong:**
The application reports "GPU enabled" or shows no warning, but all inference runs on CPU at CPU speeds. Users with CUDA or Apple Silicon hardware get no benefit. On benchmarks the app looks slow and the GPU usage monitor stays at 0%.

**Why it happens:**
The `ort` crate registers execution providers in priority order and silently falls back to the next available EP if the requested one cannot be initialized. CUDA EP will fail silently if the CUDA toolkit is not present, the driver is too old, or cuDNN is missing. CoreML EP on macOS does not route to Metal/GPU on all configurations — documented tests on Apple M3 showed CoreML EP may not use Metal acceleration despite being registered. The `ort` download strategy only provides prebuilt binaries with CUDA and TensorRT support for Windows/Linux; CoreML and DirectML require compiling ONNX Runtime from source. If you use `ort`'s default download strategy and try to enable CoreML or DirectML, the build will fail or silently skip the EP.

**How to avoid:**
- After session initialization, log which execution provider is actually active. Do this by querying the session's metadata or by checking for EP-specific errors during `init_from`.
- Show a visible status in the GUI settings dialog: "Inference: CUDA (GPU)" vs "Inference: CPU (no GPU detected)". Never let the user guess.
- For distribution, decide the EP strategy per-platform at build time using Cargo features, and document what each platform binary supports. Do not promise GPU support on macOS unless you have verified CoreML actually routes to the ANE or GPU on that hardware.
- For CUDA on Linux/Windows, use `ort`'s download strategy (provides CUDA prebuilts). For CoreML on macOS, you must compile ONNX Runtime from source with `--use_coreml` and use the `system` strategy.
- Set `SessionBuilder::with_disable_per_session_threads(false)` and verify thread counts are reasonable.

**Warning signs:**
- `cargo build --features cuda` succeeds but benchmark time is identical to the CPU-only build.
- System GPU monitor shows 0% utilization during inference.
- No error or warning printed during session initialization when CUDA is supposedly unavailable.
- Build succeeds on CI (no GPU) but the EP feature flag is still active.

**Phase to address:** Core inference engine phase. Add an EP status check and log output before any GUI work. GPU acceleration should be validated as a separate sub-task, not assumed to be working.

---

### Pitfall 3: Blocking the egui Update Loop with Inference

**What goes wrong:**
The GUI freezes entirely while inference runs. On CPU with u2net (~170MB), a single inference pass can take 3–15 seconds. The entire window becomes unresponsive: no spinner animation, no cancel button, no window drag. On macOS this triggers the spinning beachball. Users force-quit the app thinking it crashed.

**Why it happens:**
egui/eframe runs in a single-threaded immediate-mode loop. Every call inside `App::update()` must return quickly. Calling `session.run()` directly inside `update()` blocks the render thread for the full inference duration. The pattern is obvious in hindsight but easy to miss when first porting a simple inference script.

**How to avoid:**
- Spawn inference on a separate thread (or `tokio::spawn` if using an async runtime) and communicate results back via `std::sync::mpsc` channel or `Arc<Mutex<Option<Result>>>`.
- Store a `InferenceState` enum in your app struct: `Idle`, `Running { progress_rx }`, `Done(MaskData)`, `Failed(String)`.
- In `update()`, poll the channel non-blocking with `try_recv()`, update state, and call `ctx.request_repaint()` from the worker thread when done (egui exposes `Context::clone()` which is `Send`).
- Never call any ONNX Runtime function from the render thread after session creation. Session creation itself can be slow (model loading) and should also be deferred off the main thread.

**Warning signs:**
- The progress spinner freezes mid-animation during inference.
- Window title bar becomes unresponsive ("not responding" on Windows).
- The CPU core for the UI thread is at 100% during inference (instead of the inference being distributed across all cores).
- Adding a `println!` inside `update()` shows it stops printing during inference.

**Phase to address:** GUI integration phase — the threading architecture must be in place before any real inference is wired to the UI.

---

### Pitfall 4: Texture Re-Upload Every Frame for the Result Image

**What goes wrong:**
After inference completes, displaying the result image causes persistent 100% CPU/GPU usage even when nothing is changing. The app drains battery, fans spin up. On large images (4K source, meaning a large output PNG) the frame rate drops below usable levels.

**Why it happens:**
egui's texture system requires you to call `ctx.load_texture()` to register image data and get a `TextureHandle`. If this call is made every frame (e.g., inside `update()` without caching), the texture is re-uploaded to GPU every frame. `load_texture` with the same name will update the texture data, but calling it unconditionally is still a GPU upload every 60fps frame. A common mistake is putting the image decode + texture load inside `update()` without guarding it behind a "has image changed" flag.

**How to avoid:**
- Hold the `TextureHandle` as a field on the app struct. Only call `load_texture` once when new inference results arrive (when state transitions to `Done`). Store the `TextureHandle` and reuse it every frame until the next inference result.
- Use `retain_image` pattern: decode → create `ColorImage` → call `load_texture` once → store handle → use handle for all subsequent frames.
- For very large images, decode at display resolution (downscale before uploading texture) and separately keep the full-res data for export. Do not upload a 8000×6000 RGBA texture to the GPU for display.

**Warning signs:**
- GPU memory usage grows with each inference result instead of staying constant.
- Task manager shows the app consuming GPU resources continuously even when idle.
- `perf` or Activity Monitor shows `egui_glow::painter` functions at >50% CPU when the app is just sitting with a result displayed.

**Phase to address:** GUI integration phase. Establish the texture caching pattern before implementing the before/after comparison view, which makes the problem worse if done naively (two full-res textures per frame).

---

### Pitfall 5: Thread Oversubscription from Rayon + ONNX Runtime Thread Pools

**What goes wrong:**
Batch processing of multiple images runs slower than single-image processing. CPU usage is at 100% but throughput is poor. On systems with 8 cores, batch processing spawns 8 rayon workers × 8 ONNX intra-op threads = 64 threads competing for 8 cores. This causes constant context-switching and cache thrashing that degrades performance below what sequential processing achieves.

**Why it happens:**
ONNX Runtime creates its own intra-op thread pool, defaulting to one thread per physical core. Rayon also creates a thread pool defaulting to the number of logical cores. When you use `rayon::par_iter()` to process multiple images in parallel, each parallel worker creates its own ONNX Runtime session run that internally spawns its own threads. The result is N*M threads where N=rayon workers and M=ORT intra-op threads.

**How to avoid:**
- For batch processing, use one of two approaches, not both simultaneously:
  - **Option A (preferred for this app):** Process images sequentially within ONNX Runtime, but use rayon for preprocessing/postprocessing only (image decoding, mask application). Keep one ONNX session per thread at most.
  - **Option B:** Use a single rayon thread pool for batch dispatch, but configure ONNX Runtime `intra_op_num_threads(1)` per session so each rayon worker is single-threaded internally.
- Set `SessionBuilder::with_intra_threads(n)` where n is `num_cpus::get() / rayon_pool_size`.
- Benchmark with `cargo bench` before deciding the parallelism strategy. The correct number is workload-dependent.

**Warning signs:**
- Batch throughput (images/second) is lower with 4 parallel workers than with 1.
- `htop` shows all CPU cores at 100% but system load average exceeds core count (indicates context switching).
- Inference time per image increases as batch size grows.

**Phase to address:** Batch processing phase. Do not wire rayon + ONNX together without a performance benchmark. Start with sequential, measure, then add parallelism with explicit thread counts.

---

### Pitfall 6: ONNX Runtime DLL Hell on Windows Distribution

**What goes wrong:**
The binary works perfectly in development but crashes on end-user Windows machines with "The program can't start because onnxruntime.dll was not found" or silently uses the wrong version. Windows 11 ships `onnxruntime.dll` v1.10.0 in `System32`; if `ort` requires v1.13+, system DLLs take precedence and the app fails to start.

**Why it happens:**
Windows DLL resolution searches `System32` before the application directory by default. Any system that has ONNX Runtime installed (e.g., via Windows ML) may have a different version. The `ort` crate's `load-dynamic` feature can mitigate this by specifying an explicit path, but only if the path is correctly bundled.

**How to avoid:**
- Enable `ort`'s `copy-dylibs` Cargo feature, which copies the required ONNX Runtime DLLs into the Cargo target folder automatically.
- Package DLLs alongside the executable in the distribution archive. Never rely on the system having a compatible version.
- Use `load-dynamic` with `ORT_DYLIB_PATH` set to a path relative to the executable at launch time if you cannot statically link.
- Prefer static linking where possible (`prefer-compile-strategy` feature) — it embeds ONNX Runtime into the binary and eliminates the DLL problem entirely, at the cost of a larger binary and longer compile time.
- Test distribution on a clean Windows VM with no Rust toolchain or ONNX Runtime installed.

**Warning signs:**
- The binary works on your dev machine but crashes on a colleague's clean machine.
- Error is a DLL load failure, not an ONNX Runtime logic error.
- `Dependency Walker` or `dumpbin /dependents` shows external ONNX Runtime DLL references not satisfied.

**Phase to address:** Distribution/packaging phase. Also requires a CI step that produces and smoke-tests the release artifact on a clean runner.

---

### Pitfall 7: include_bytes! with 170MB u2net Model Causing CI Compile Times

**What goes wrong:**
Every incremental build recompiles the crate containing the `include_bytes!` call even when no Rust source changed, because Cargo tracks the embedded file as a dependency. A 170MB binary blob embedded via `include_bytes!` can add minutes to every CI build cycle. The `rustc` issue tracker documents this as a known performance problem with large binary blobs.

**Why it happens:**
`include_bytes!` copies the entire file into the compilation unit at compile time. The compiler has to process and emit the blob as part of the object file. This is an O(size) operation that does not cache separately from the Rust source. On a cold CI runner this means 170MB of I/O plus LLVM processing time on every build.

**How to avoid:**
- Put the model embedding in a dedicated, isolated crate (e.g., `bgprunr-models`) with no other code. This crate only recompiles when the model files change, not when app logic changes.
- Use `include_flate` or a similar crate that compresses the blob at compile time and decompresses at runtime. This reduces the compilation I/O and the binary size at the cost of ~50ms startup decompression.
- For CI, consider separating a "fast build" (no models embedded, loads from filesystem) from a "release build" (models embedded). Use Cargo features: `default = ["embedded-models"]` and `[cfg(feature = "embedded-models")]` guards around the include calls.
- Cache the compiled `bgprunr-models` artifact in CI using `sccache` or GitHub Actions cache keyed on the model file hash.

**Warning signs:**
- `cargo build --timings` shows the model crate taking >30 seconds even on small source changes.
- CI builds take longer after adding u2net than before.
- Total binary size matches expectations but intermediate `.o` files are unexpectedly large.

**Phase to address:** Project setup / workspace layout phase. Get the crate structure right before embedding any models. Retrofitting is disruptive.

---

### Pitfall 8: Clipboard Image Copy Failing on Linux Wayland

**What goes wrong:**
"Copy to clipboard" works on macOS and Windows, works on Linux X11, but silently does nothing (or copies then immediately loses the data) on Linux Wayland. Users on GNOME/KDE Wayland desktops report the clipboard feature is broken.

**Why it happens:**
X11 and Wayland implement clipboard ownership differently. On Wayland, the application that copied data must remain alive to serve paste requests — the data is not persisted by the compositor. When the `arboard` crate (or egui's built-in clipboard) drops the clipboard owner thread too early, the data vanishes before other apps can paste it. Additionally, `arboard` requires the `wayland-data-control` feature flag to be enabled for Wayland support; without it, clipboard operations silently do nothing on pure Wayland sessions. egui uses `smithay-clipboard` for Wayland by default when a Wayland window handle is present, but this only covers text, not arbitrary image data.

**How to avoid:**
- For image-to-clipboard, use `arboard` directly with the `ImageData` API, and enable the `wayland-data-control` feature flag.
- Keep the `Clipboard` instance alive (store it on the app struct) for the lifetime of the application. Do not create and immediately drop it.
- Test clipboard on both X11 and Wayland explicitly. The simplest test: copy, switch to a text editor, paste, verify.
- Document in the app that clipboard image copy requires a Wayland compositor that supports the `wlr-data-control` or `ext-data-control` protocol (most modern compositors do, but old ones don't).

**Warning signs:**
- Clipboard works in `--wayland` mode test but fails in default Wayland session.
- The "copy" action completes without error but nothing appears in clipboard managers.
- Works after a delay (race condition with clipboard ownership handoff thread).

**Phase to address:** GUI polish phase (clipboard integration). Wayland-specific testing requires explicit CI or manual verification on a Wayland runner.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single `main.rs` with all GUI and inference logic | Faster initial development | Cannot unit-test inference without starting the GUI; tight coupling makes CLI reuse impossible | Never — the PROJECT.md requires a Cargo workspace with shared core from day one |
| Hardcode preprocessing constants inline rather than in a tested function | Less boilerplate | Any edit breaks the pipeline silently; hard to compare against rembg spec | Never |
| Run ONNX session creation on startup synchronously | Simpler code | 2–5 second blocking startup on slow machines (model decompression + session init) | Never for the u2net model; acceptable for silueta (~4MB) during prototyping |
| Embed both models in both CLI and GUI binaries instead of a shared models crate | One Cargo.toml to maintain | 340MB duplication; 2x compile time for model changes | Never |
| Use `Arc<Mutex<Session>>` to share one ONNX session across rayon threads | Saves memory | Sessions hold graph state; contention under batch load serializes all inference; ORT sessions are not designed for concurrent use from multiple threads | Never — create one session per batch worker instead |
| Skip GPU fallback detection; always register CUDA EP | Simpler initialization code | Users with no CUDA get cryptic errors or silent CPU fallback with no feedback | Only acceptable if the binary is explicitly labeled "GPU-required" |
| Load model from filesystem at runtime instead of embedding | Fast iteration during development | Requires distributing model files separately; breaks single-binary constraint | Acceptable as a dev-only feature flag (`--no-embed-models`); must be removed from release |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `ort` execution provider registration | Registering CUDA/CoreML/DirectML EPs using the `download` build strategy — these EPs are not included in prebuilt binaries | Use `download` for CUDA on Linux/Windows only; compile ONNX Runtime from source for CoreML (macOS) and DirectML (Windows with older hardware) |
| `image` crate → ONNX tensor | Passing the raw `ImageBuffer<Rgb<u8>>` bytes directly as f32 without dividing by 255 | Always: cast to f32, divide by 255.0, then normalize. Write a test. |
| `rayon` + ONNX Runtime | Using `par_iter` to call `session.run()` concurrently without limiting ORT's intra-op threads | Set `intra_op_num_threads` = `total_cores / rayon_workers` before batch dispatch |
| egui texture management | Calling `ctx.load_texture` inside `update()` unconditionally | Store `TextureHandle` in app struct; reload only on `InferenceState` transition to `Done` |
| `arboard` clipboard on Linux | Creating a `Clipboard` instance, setting image data, then dropping it immediately | Store `Clipboard` on app struct for the application lifetime; enable `wayland-data-control` feature |
| ONNX Runtime on Windows | Relying on system PATH for `onnxruntime.dll` | Enable `copy-dylibs` feature or prefer static linking via `prefer-compile-strategy` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Decoding full-res image for display texture | Memory spike to 500MB+ when loading a 20MP JPEG; GPU VRAM exhausted | Decode at native size for export; create a display-resolution copy (max 2048px wide) for the texture | Any image > ~4MP |
| Sequential mask post-processing (pixel-by-pixel in Rust without SIMD) | Mask application to a 20MP image takes 2–3 seconds after inference | Use `rayon::par_chunks_mut` over the pixel buffer for the multiply step | Images > 2MP with naive loops |
| Creating a new ONNX `Session` per image in batch mode | Memory usage grows linearly; session creation overhead 200–500ms per image | Create session once, reuse across batch. Sessions are designed for repeated `run()` calls | Batch sizes > 5 images |
| Allocating output tensor buffers inside the inference loop | Steady heap fragmentation in batch mode; GC pressure (allocator thrashing) | Pre-allocate output buffers; reuse across runs using `ort`'s pre-allocated output API if available | Batch sizes > 50 images |
| Logging to stdout/stderr in hot path (per inference call) | Throughput degrades 10–30% in batch mode due to I/O serialization | Use `tracing` with level guards; only emit at `DEBUG` level inside inference; disable in release builds | Batch processing with verbose logging enabled |

---

## Security Mistakes

The local-only, privacy-first design eliminates most web security concerns. Domain-specific risks:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Processing untrusted ONNX model files from user-provided paths | Malformed ONNX models can trigger memory safety bugs in ORT's C++ runtime (known CVEs exist) | Only load models from `include_bytes!` embedded at compile time; do not accept user-supplied model files in v1 |
| Decompressing embedded model data (zstd) without size bounds | Zip-bomb style decompression attacks if model bytes are tampered with post-distribution | Verify decompressed size against known constant before allocation; validate checksum of embedded model at startup |
| Writing temp files for intermediate images to world-readable `/tmp` | Privacy violation — user photos visible to other processes | Write to OS temp dir with restricted permissions; delete immediately after use; prefer in-memory buffers throughout the pipeline |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No progress indication during u2net inference (3–15s on CPU) | Users think the app has frozen; force-quit | Show animated spinner + elapsed time + "Cancel" button from the moment inference is dispatched; update via `request_repaint` calls from worker |
| Silently downscaling images > 8000px without telling the user | User gets a mask for a downscaled version and wonders why fine details are missing | Show a modal: "Image is 12000×8000px. Downscale to 8000px for processing? [Proceed] [Cancel]" — match PROJECT.md requirement |
| Showing raw ONNX error messages to users | "OrtException: [CUDA] cudaMemcpy error: driver shutting down" confuses users | Map ORT exceptions to user-friendly messages: "GPU inference failed, retrying on CPU..." |
| No visual diff between original and result | Users cannot judge quality without toggling between windows | Implement the before/after slider view early — it is the primary quality-assessment tool for this app |
| Exporting as PNG without transparency (RGBA vs RGB confusion) | Output looks correct on white background but compositing fails downstream | Always export as RGBA PNG; verify alpha channel is present and non-trivial before export |

---

## "Looks Done But Isn't" Checklist

- [ ] **Preprocessing:** Inference runs and returns a mask — verify the mask matches rembg's Python output on the same image, not just "has the right shape"
- [ ] **GPU acceleration:** EP is registered and build succeeds — verify via GPU monitor that actual GPU utilization occurs during inference
- [ ] **Single binary:** Binary builds and runs on dev machine — verify on a clean VM/container with no Rust toolchain, no CUDA toolkit, no onnxruntime installed
- [ ] **Cross-platform build:** CI matrix passes — verify the Linux build runs on Wayland (not just X11) and the macOS build runs on arm64 (not just x86_64 via Rosetta)
- [ ] **Batch processing:** Multiple images process successfully — verify throughput improves with parallelism (not degrades) and session memory does not grow per image
- [ ] **Clipboard:** "Copy" action appears to succeed — verify the clipboard data is actually pasteable in another application, including on Linux Wayland
- [ ] **Large image handling:** 8000px images load — verify the downscale warning fires and the output mask corresponds to the full original resolution
- [ ] **Alpha channel export:** PNG is produced — verify the output has transparent pixels (not white background) by opening in an image editor

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Preprocessing mismatch discovered after GUI is built | MEDIUM | The inference core is isolated from GUI by design (Cargo workspace). Fix is contained to `core::preprocess`. Rerun reference test, no GUI changes needed. |
| GPU EP silent fallback discovered late | LOW | Add EP detection logging + UI status indicator. ONNX session already exists; just need to query and surface the active EP name. |
| UI freeze discovered after wiring inference to button | HIGH | Requires architectural change — moving inference off the render thread and introducing a state machine + channel. Harder to retrofit than to design in from the start. |
| Texture re-upload discovered via profiling | LOW | Refactor `update()` to cache `TextureHandle`. One-time change, no architectural impact. |
| Thread oversubscription discovered in batch benchmark | LOW | Add `intra_op_num_threads` configuration to `SessionBuilder`. Benchmark-driven tuning, no architectural change. |
| Windows DLL hell discovered in distribution testing | MEDIUM | Enable `copy-dylibs` or switch to static linking. Requires rebuild of all Windows artifacts. CI pipeline update needed. |
| Model embed compile time discovered when CI is slow | MEDIUM | Extract model crate, update workspace Cargo.toml, add feature flag. Moderate refactor but well-isolated. |
| Wayland clipboard discovered in user reports | LOW | Enable `wayland-data-control` arboard feature, store `Clipboard` instance. Narrow change, limited scope. |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Preprocessing pipeline mismatch | Phase 1 — Core inference engine | Reference test: pixel-accurate comparison against rembg Python output on 3 test images |
| GPU EP silent fallback | Phase 1 — Core inference engine | Log active EP name at session init; GPU monitor shows utilization during bench |
| Blocking the egui update loop | Phase 2 — GUI integration | Spinner animates continuously during inference; window is draggable while model runs |
| Texture re-upload every frame | Phase 2 — GUI integration | CPU/GPU usage stays <5% when inference is idle and result is displayed |
| Thread oversubscription in batch | Phase 3 — Batch processing | Batch throughput benchmark shows improvement, not degradation, with N workers vs 1 |
| Windows DLL hell | Phase 4 — Distribution & packaging | Clean Windows VM smoke test: binary launches without installing any prerequisites |
| include_bytes compile time | Phase 0 — Project scaffolding | CI build time stays below 5 minutes on incremental builds after workspace split |
| Wayland clipboard failure | Phase 2 — GUI integration | Manual test: copy result, paste in GNOME Text Editor on Wayland session |

---

## Sources

- [ort crate GitHub — pykeio/ort](https://github.com/pykeio/ort) — execution provider fallback behavior, linking strategies, load-dynamic feature
- [ort execution providers docs](https://ort.pyke.io/perf/execution-providers) — EP registration order, silent fallback documentation
- [ort linking docs](https://ort.pyke.io/setup/linking) — static vs dynamic vs load-dynamic strategies, copy-dylibs
- [ONNX Runtime threading docs](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) — intra-op/inter-op thread pool behavior, oversubscription risks
- [ONNX Runtime CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) — macOS limitations, build requirements
- [ONNX Runtime fallback discussion — onnx/onnx #6623](https://github.com/onnx/onnx/discussions/6623) — community recognition of silent fallback as a design problem
- [egui texture performance discussions — emilk/egui #5718, #4932, #797](https://github.com/emilk/egui/discussions/5718) — texture re-upload bottleneck, wgpu recommendation
- [egui async/threading patterns — emilk/egui #484](https://github.com/emilk/egui/discussions/484) — channel-based worker thread pattern
- [arboard — 1Password/arboard](https://github.com/1Password/arboard) — Wayland clipboard ownership model, wayland-data-control requirement
- [egui Wayland clipboard fix — emilk/egui #1613](https://github.com/emilk/egui/pull/1613) — smithay-clipboard integration
- [Rust include_bytes compile time issue — rust-lang/rust #65818](https://github.com/rust-lang/rust/issues/65818) — known compile time problem with large blobs
- [Bundling ONNX Runtime in Rust — blog.stark.pub](https://blog.stark.pub/posts/bundling-onnxruntime-rust-nix/) — real-world DLL and linking war stories
- [U2-Net ONNX preprocessing — xuebinqin/U-2-Net #270](https://github.com/xuebinqin/U-2-Net/issues/270) — confirmed preprocessing contract for ONNX inference
- [Windows DLL precedence — ort crate docs](https://crates.io/crates/ort/1.13.1) — System32 DLL version conflict documentation

---
*Pitfalls research for: Rust desktop AI background removal (ort + egui)*
*Researched: 2026-04-06*
