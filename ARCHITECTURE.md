# Prunr Architecture

How Prunr is built. For user-facing info see [README.md](README.md).

## Design Principles

1. **The UI thread never waits.** Inference, I/O, decoding and texture prep run off-thread; results come back over `mpsc` channels drained with `try_recv` each frame.
2. **One executable.** `prunr` is the GUI, the CLI and the worker (`--worker`). Core models are embedded as zstd blobs, the rest download on demand; ONNX Runtime is loaded at run time from `runtime/` beside it or a user-installed GPU runtime.
3. **Platform parity.** Every feature on Linux x86_64, macOS aarch64 and Windows x86_64; platform code behind `#[cfg]`.
4. **Progressive performance.** GPU or Neural Engine when present, CPU otherwise; startup never waits on graph compilation.
5. **Crash isolation where memory is unpredictable.** Batch segmentation and the Stable Diffusion eraser each run in a subprocess, so an OOM kill takes the child, not the app. Bounded models (LaMa, MI-GAN, SAM 2, upscalers) run in-process on background threads.

## Workspace

```
crates/
  prunr-models/            # Model registry + embedded zstd blobs; no workspace deps
  prunr-core/              # Pure pipeline: segmentation, edges, selection, SAM 2,
                           #   inpaint (LaMa, SD), upscale, image I/O, ORT runtime
  prunr-runtime-install/   # PyPI wheel fetch + repackage for GPU runtimes
  prunr-app/src/
    main.rs                # --worker / CLI / GUI dispatch
    cli.rs                 # CLI frontend
    worker_process.rs      # Subprocess child
    subprocess/            # IPC protocol, framing, parent-side manager
    gui/                   # egui app: coordinators, views/, automation/, tests/
xtask/                     # fetch-models, install-runtime, golden diffs
packaging/                 # AUR, Homebrew
assets/                    # Icon, Info.plist, .desktop
```

**Dependency direction:** `prunr-models` → `prunr-core` → `prunr-app`, never back; `prunr-models` depends on nothing in the workspace so its blobs recompile only when models change. `prunr-runtime-install` is a leaf shared by the GUI and `xtask`.

## GUI Coordinators

`PrunrApp` (`gui/app/`, one file per concern: dispatch, results, preview, tools, import, export, textures, adjustments) is a **coordinator, not an owner**: it holds UI visibility flags, view state with no other home, and the domain coordinators below. The coordinators don't know about each other; cross-domain work (a finished image updating the batch, the history and admission) is orchestrated on `PrunrApp`.

| Coordinator        | Owns                                                                                                   |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `BatchManager`     | Batch items, selection, `BackgroundIO` (decode, thumbnail, texture-prep, save channels), memory governance, progress counts |
| `Processor`        | Every dispatch: worker IPC and admission, live preview, cancellation, eraser (in-process and the SD bridge), upscale, colorize, the depth queue, SAM 2.1 sessions, the shared progress slot, and the run queue that takes ticked images one at a time through the single upscale / colorize slot |
| `BrushState` / `MagicBrushState` | Tool on/off and the in-progress stroke; brush settings live once on `settings.brush` for both |
| `DownloadManager`  | On-demand model downloads, one at a time                                                                |
| `DragExportState`  | OS drag-out lifecycle, shared with the drag crate's callback thread                                     |
| `SystemBridge`     | The only importer of `rfd` (file dialogs) and `arboard` (clipboard). One file dialog at a time: a surface asks, `logic()` shows it parented to the window and drains the answer, which carries image ids so it resumes the action that asked; while it is up it counts as a modal |
| `Automation`       | The control socket; `None` unless `PRUNR_CONTROL_PORT` is set                                          |

**Who decides what.** A decision about one image is a method on `BatchItem` (how much of a run it needs, what a landed decode, texture, overlay or result does to it, whether a run's result still fits the state it started from); a loop over the batch is a method on `BatchManager` (classify a batch, prepare its work, mark what waits); admission and the streams to the workers are on `Processor`. `PrunrApp` routes between them and owns nothing a coordinator could. Process targets the ticked images for every model: cut-outs as one admitted batch, upscales, colorize and erasers one after the other.

**Frame loop.** `logic()` drains channels, handles input, then runs `reconcile_selected`, one idempotent step that tops up the selected image's lazy state (decoded source, textures, Magic Brush embedding) and, on a selection change, frees the other items' results. `ui()` only renders. Wayland drops other threads' wake-ups while the window is idle, so the UI thread polls every 50 ms while work is pending.

**Intents.** Every keyboard action is an `Action`, routed with its gates through `PrunrApp::perform`, which the control socket calls too; toolbar buttons call the same `handle_*` methods, so a gate cannot drift between surfaces. `views/shortcuts.rs::SHORTCUTS` is the one binding table; user rebinds (`settings.hotkeys`) layer over it into `Bindings`, installed into the egui context each frame so tooltips and the F1 list show the live keys.

## Secondary surfaces

Every popover, panel and dialog is one of five kinds, chosen by how the user works with it:

| Kind | Mechanism | Used for |
|---|---|---|
| Tool strip | `egui::Area` over the canvas top while a brush is on; the tool's toggle removes it | Knobs touched every stroke (mode, size, hardness or confidence, opacity or expand/blend) |
| Flyout | `chip::flyout_for` / `GroupChip { live: true }`; its chip, Escape or another chip closes it; fades while a slider is held | Knobs that change the image while open: Mask, Sketch, Effect, Background, Refine, the strip's full panels |
| Popover | `chip::popup_for`; a click outside or a pick closes it | Set-and-go choices: Model, Preset, Quality, Scale, Prompt, Advanced, Help |
| Modal | `theme::standard_modal_window`, listed in `PrunrApp::open_modals` | Rare and global: Settings, Model Store, licence, runtime prompt, help references, Save preset |
| Status | Painted on the canvas | Progress capsule and tile map, the Loading Magic Brush pill, the Rendering / Preparing pill once quiet background work (a preview, a thumbnail, a texture) has run for a second with nothing else to show for it, toasts |

The canvas stays live under strips and flyouts; only a popover holds it still, so the click that dismisses one is not also a stroke. Escape does one thing per press: close the open popup, else the top-most modal, else cancel a run, else clear the selection. `gui/tests/surface_tests.rs` pins these contracts.

## Automation

Tests and outside tools drive the app through **the control tree**: the AccessKit tree egui builds from the real widgets (role, name, value, enabled, rectangle). A control's name is its text; icon-only buttons and bare sliders get theirs from `chip::tooltip` / `chip::slider_row` / `chip::named`.

**In tests**, `egui_kittest` runs the real `PrunrApp` without a display; tests find controls by name, click, step frames and assert on state. `gui/tests/tree_tests.rs` renders every surface and fails on any nameless control, so the naming rule enforces itself.

**At run time**, `PRUNR_CONTROL_PORT=<port>` opens a line-JSON socket on localhost. Clicks, drags and keys are injected as egui input, one batch per frame, and the reply waits until those frames have run, so the app reacts exactly as to a user. `intent` goes through `PrunrApp::perform`; `state` is a serde dump; `open` and `screenshot` reuse the app's own paths. The window must be receiving frames: with the output powered off a Wayland compositor sends no frame callbacks and nothing answers.

## Progress

Every long-running pipeline takes a `prunr_core::Progress` handle. It names its steps from one table of user words (`Step`), counts at two levels (outer: crops, passes; inner: tiles, steps, images), lays out the tiles it works on, and checks cancel on the same handle. Listeners implement `ProgressSink`: the subprocess sends each report as one `Report` event, in order with its results; the GUI folds reports into one `DispatchProgress` through a run-scoped sink, so a replaced run's late reports never land; the CLI sets its spinner text. The capsule and tile map draw that snapshot without knowing the pipeline, so a new model needs no widget work. `progress::check_contract` is the shape every finished run must have. Pictures (finished tiles, a step's estimate) travel apart from the counts and are made only while the run's result can still land; each names where it sits in the photo, so the result later reveals over the preview they build.

Time left runs from the moment the count first moves, so model loading and graph compiles never inflate it.

## Process Architecture

ONNX Runtime's memory use is unpredictable, and its arena and GPU contexts hold allocations until the session dies, so process exit is the only guaranteed reclaim. Work that can OOM runs in a `prunr --worker` child, where an OOM kills the child, not the window; small or interactive work stays in-process.

| Work | Runs in | Lifetime |
|------|---------|----------|
| Segmentation + edges (Process, CLI) | seg worker subprocess | pre-warmed at startup; a batch reuses the warm worker when its model, processor and pool size fit and sends it the run's mask and Sketch settings (`Configure`, engines built on first need); the worker stays warm 5 min after a batch, 1 min after a model whose working memory stays loaded (BiRefNet: 5.5 GB); released on model switch, Cancel or a crash, and the newly picked cut-out model loaded once it has stayed picked 1 s |
| SD erases | inpaint-only worker subprocess (no seg engines) | spawned on first SD erase; killed after 5 min idle, on model switch, or on Cancel |
| LaMa / MI-GAN erases, SAM, live preview | in-process, rayon | sessions cached; LaMa released after 5 min idle |
| Upscale, colorize | in-process, one thread per run | engine kept warm until model or hardware change |
| Depth maps | in-process, one queue thread, photos one at a time | engine kept warm until model or hardware change; the ~1 MB map stays on the item until its photo changes |
| Filters only (no model) | background thread per item | — |

**First look.** A slow cut-out (BiRefNet, U2Net) on the image on screen first runs Silueta, on the CPU, and reports its subject through the run's progress, so the canvas outlines the subject in about a second instead of at the end. It runs before the slow model rather than beside it: measured on a 4-core CPU, the outline lands at 1.1 s instead of 2.9 s and the run ends sooner. The app sets which image is on screen and the worker settles it as each run starts (`Configure` carries it; the bridge re-sends it when it changes mid-batch), so background images of a batch never run Silueta. Silueta's session lives in the seg worker beside the slow model's, loaded on first need and let go when first looks go off (Animations off, or the model timed under 2 s per image on its provider this session).

SD has its own process because its multi-GB bundle is the likeliest OOM and a UNet step can't be interrupted: Cancel kills the process so RAM frees at once, and a watchdog kills it if free system RAM drops below 1 GB mid-erase.

### IPC

Length-prefixed bincode frames over the child's stdin/stdout (`subprocess/protocol.rs`):

| Direction | Kinds |
|-----------|-------|
| Parent → child | `Init`; work (`ProcessImage`, `RePostProcess`, `AddEdgeInference`, `Inpaint`); `CancelItem`, `Shutdown` |
| Child → parent | `Ready` / `InitError`; `Report` (any progress update); results / errors per work kind; `RssUpdate`; `Finished` |

Payloads (image bytes, result RGBA, seg and edge tensors for the Tier 2 caches) go through temp files in `prunr-ipc-{parent pid}`, under `/dev/shm` on Linux (unless it is under 1 GiB, as in a container) and the OS temp dir elsewhere; the reader deletes each file (sweep rules in [Temp File Lifecycle](#temp-file-lifecycle)).

Cancel is per item: the bridge drops queued items and sends `CancelItem` for in-flight ones, which the worker checks before and during inference. A cancelled item shows its own state again (its result, or the photo), not an error, so its caches survive.

## Threading Model

| Thread | Purpose |
|--------|---------|
| UI (egui) | Render, input, drain result channels with `try_recv` |
| `prunr-bridge` | Owns the seg worker: spawn, IPC, RSS pacing, crash retry, hang detection (60 s silent = crash; a run repeats its last step every 10 s for up to 10 min, so one long model run on a slow CPU is not taken for a hang) |
| `prunr-inpaint-bridge` | Owns the SD worker; spawns it on a helper thread so Cancel works during Init |
| `subprocess-reader` | One per child; blocking reads of child stdout |
| `prunr-control` | Only with `PRUNR_CONTROL_PORT`: local JSON socket; requests are answered on the UI thread |
| `file-dialog` | One per open file dialog: waits for the user's answer and wakes the frame loop. The dialog itself starts on the UI thread, which macOS needs for its sheet |
| Upscale | Builds the session off the UI thread (a cold OpenVINO compile takes tens of seconds); Cancel terminates the running ORT call |
| Background I/O | Short-lived: file listing, decode, thumbnails, texture prep, history spill, save, downloads. Decode-heavy ones share a slot pool capped at the core count so a big batch doesn't stack per-image peaks |
| rayon global pool | Live preview, LaMa / MI-GAN, SAM |

### Engine pool (seg worker)

| Backend | Engines | ORT intra-op threads |
|---------|---------|----------------------|
| CPU (OpenVINO included: it runs on the host CPU here) | parallel-jobs setting for Silueta and U2Net; 1 for BiRefNet, whose single run fills every core (a second engine only doubled RAM) | `num_cpus / engines` |
| GPU | `min(jobs, 2)` (VRAM) | same |

`batch::engine_pool_size` is the one rule; the worker builds by it and the admission estimate reads it.

The ORT CPU arena is off to lower the baseline; the subprocess absorbs any OOM. Jobs share a 64-unit semaphore weighted by megapixels, so small images run in parallel, a 40 MP image runs nearly alone, and FIFO order keeps small arrivals from starving it.

### Crash retry

On a crash or hang the bridge re-queues unfinished images, halves the job count and spawns a new child (toast: "Memory pressure — retrying…"). In-flight Tier 2 items error instead, since their tensor temp file is gone; the next Process reruns them in full. A crash at one job fails only the image it ran, with the cause from the exit signal (SIGKILL = out of memory, SIGSEGV = segfault), and the batch goes on with a new worker. The CLI runs on the same rules (`subprocess::batch_rules`).

## Tiered Recipe Pipeline

Each `BatchItem` keeps the `ProcessingRecipe` that produced its result (`applied_recipe`). On Process, `prunr_core::resolve_tier(old, new)` picks the cheapest tier that covers the change (the most expensive change wins):

| Tier | When | Work |
|------|------|------|
| `FullPipeline` | Model, segmentation on/off, or a line-mode switch that changes TEED's input | Full inference |
| `AddEdgeInference` | Lines turned on, or `InputTransform` changed, with the seg tensor cached | TEED only |
| `MaskRerun` | Mask knobs (incl. fill style, bg effect, brush correction), the Depth chip's look, or lines turned off | Postprocess the cached seg tensor (the look re-renders the photo from the kept depth map) |
| `EdgeRerun` | Line knobs (strength, colour, thickness, scale, compose mode, style, fade with distance) | Re-threshold the cached TEED tensor |
| `UpscaleRerun` / `UpscaleTier2` | Upscale knobs before / after the model | See [Upscale](#upscale) |
| `ColorizeTier2` | Colorize Strength (a colorizer change is a model change) | Re-render the kept colour plane at the new strength |
| `CompositeOnly` | Background colour or image | Render-time only |
| `Skip` | Nothing output-relevant | None |

The recipe of an in-flight batch lives on `Processor` (`InFlightBatch`), not on the item, so a result landing after a settings edit is stamped with what actually ran.

**Two tensor caches.** The seg tensor and the TEED tensors are separate zstd-compressed fields on the item, so a model swap invalidates only one and a line-mode change only the other; both share a 512 MB budget, evicted in batch order, the selected item exempt.

**Batch vs interactive Tier 2.** Batch reruns go through the subprocess (a mask rerun with lines on takes the AddEdge path so the outline is recomposed; edge reruns run the full pipeline). Interactive reruns of the selected item run in-process; see [Live Preview](#live-preview).

**What each model shows.** A model has a role (cut-out, filters, erase, upscale, colorize), and `knob_catalog::shows(model, Surface)` is the one table of which chips and brush controls appear for it: only those that act on that model's image. Dispatch routing stays on the role and model predicates. The toolbar, the brush strip and panels, the selection overlay and the selection actions all read it; a test renders the toolbar for every model against the table. A control that is shown but idle under another setting (Mask while Sketch is Full) is greyed with its reason instead.

**Two sources of tier truth, one test.** `resolve_tier` compares whole recipes; `gui/knob_catalog.rs` maps each toolbar knob to `(tier, cache impact, dispatch)`, with context-aware specs for knobs whose cost depends on what is cached (Off → Subject outline is a live recompose with the edge tensor warm, an AddEdge run otherwise). A test mutates every knob and asserts the two agree.

## Per-Image Settings

Every `BatchItem` owns its `ItemSettings`, so a toolbar edit changes one image; app-wide behaviour (parallel jobs, brush) lives on `Settings`. `ItemSettings` is `Copy` and `#[serde(default)]`, so older presets load when fields are added.

### Creative compose layer

Five per-item enums sit on top of the AI output, applied in one `postprocess → compose` step in `prunr-core` from cached tensors:

| Enum | Drives | Tier |
|---|---|---|
| `ComposeMode` | How subject α and edge α combine (Subject outline mode) | EdgeRerun |
| `LineStyle` | Edge colouring (solid, gradients, rainbow, noise, DualScale, …) | EdgeRerun |
| `SubjectEffect` | RGB transform of the masked subject (desaturate, duotone, halftone, …) | MaskRerun |
| `BgEffect` | Source-derived backdrop baked into transparent areas | MaskRerun |
| `InputTransform` | Transform of TEED's input | AddEdgeInference |

`DualScale` draws the Fine and Bold TEED scales at once. The Bold tensor is decompressed only while it is on, and never replaces the hot active-scale tensor, so the next edge tweak still hits the cache.

### Selection and brushes

**One plane.** Every selection is a `prunr_core::selection::MaskArtifact`: a signed `i8` grid at source resolution behind an `Arc`, so undo snapshots and cross-thread reads are refcount bumps. Sign is direction (toward subject or background), magnitude is coverage. Region consumers (outline, inpaint region, bounding box) count a cell at half coverage or more as selected; continuous consumers (Delete / Copy / Cut alpha, the segmentation correction, Invert) read the signed value. Paint strokes and Magic Brush results merge onto `BatchItem.selection_mask` through `MaskArtifact::merge_stroke`. On a cut-out a stroke's sign is its direction: zero leaves a cell, the same sign keeps the stronger value, the opposite sign lets the newer stroke win. Elsewhere (No model, the eraser) the plane is an amount: Add grows it, Subtract carves it away, and it is stored negative so it reads as an erase if a background-removal model later uses it.

**One commit path, one rule per model.** Every author calls `PrunrApp::commit_selection_and_dispatch`: `BatchItem::commit_selection_mask` swaps the plane and records the undo step, then `apply_selection_to_active_model` decides what the stroke means:

| Active model | A stroke is | Applied |
|---|---|---|
| Background removal | Restore or Erase on the cut-out | At once, by an in-process mask rerun from the cached seg tensor, but only while a cut-out is on screen (`BatchItem::shows_cutout`); otherwise the selection waits for the next result |
| Eraser | Add to / subtract from the region to fill | On Process, since a run costs seconds to minutes |
| Upscale, Colorize | Not offered: the brushes and selection actions are hidden, since both cover the whole image | — |

**Correction.** `MaskArtifact::apply_to_alpha` runs last, on the full-resolution mask after edge refinement and feather (Erase scales toward 0, Restore lifts toward 1, by magnitude): a stroke is exact intent, and refining it would pull it onto nearby image edges. The plane it gets is the selection at the photo's own size (`BatchItem::correction_plane`), feathered by the selection Feather once the overlay has built that plane off the UI thread; its landing re-cuts once more. The overlay draws the same plane by coverage, so what it shows is what the cut-out gets.

**Actions and overlay.** Delete, Copy, Cut, Invert and Clear share one handler. Fill, outline and feather are baked off-thread into one texture per item keyed by `(plane hash, style)`; a stroke uploads only the changed patch. The outline is in image pixels, so the style is taken at the current zoom: an outline keeps at least 1.5 screen points, widened in odd whole-pixel steps so zooming rebuilds the texture only a few times. With Feather on, the actions use the same guided-filter-feathered plane the overlay shows, so what is cut is what was shown.

**Painters.** `prunr_core::brush` has Circle, Square and Line stamps; an in-progress stroke paints into its own plane on `BrushState` and merges on release. Paint and Magic Brush are mutually exclusive tools sharing `Settings.brush`.

### Undo timeline

Each image has one timeline (`gui/timeline.rs`): the states it has been in and a cursor, the shape Krita, GIMP and Photoshop share. A state is complete: result, recipe, selection, item settings and applied preset. Undo and redo move the cursor and copy that state into the item's working fields; nothing re-runs, because the image is stored. Every other edit cuts the redo tail and appends one step: Process, Erase, Upscale, a stroke or Magic Brush click, Clear and Invert, Delete and Cut, a preset, and a knob change (changes within about a second are one step, so a slider drag is one). 64 steps per image, the oldest dropped first.

**Shared, tiered payloads.** A step points at the stored result and selection of the step before it when it did not change them, so a stroke copies no image. A stored payload stays as-is while anything else holds it (the canvas, a job), is zstd'd off-thread once nothing does, and moves to `<cache dir>/prunr-history/{pid}/` under memory pressure; a disk file goes with its last state.

**Jobs bind to a version.** The timeline's version changes on every step, undo and redo. A run (Process, erase, upscale) notes it at dispatch and lands only if it still matches: as its own step, or, when a knob change or preset asked for it, folded into that step. A live-preview re-cut carries the version the same way and fills in the shown step. A step that keeps the shown pixels (a knob for the next run, a preset, Reset) carries an erase, upscale or colorize under way along, so minutes of work land on top of it; a cut keeps the strict rule, as its knob changes re-cut. Edits that take a step on the image (brushes, Magic Brush, the selection actions) wait for any run on it. A result for a state the image has left is dropped, and undo asks a running job to stop. Switching images releases the working result; the timeline holds it.

**A result that cannot land ends its run.** When a result arrives for a state the image has left (new pixels, an undo, or for a cut any step while the run was under way), the image stops waiting, the user hears why, and under streamed admission its share of the budget is released; when a batch completes, any image the worker never answered is ended the same way. Nothing is ever left "processing" with nothing in flight.

### Magic Brush (SAM 2.1)

SAM 2.1 Hiera Small ships bundled as an encoder + decoder pair (category `Selection`); it authors selections and never enters the background-removal pipeline. `prunr_core::sam` is pure; the two CPU sessions (~180 MB) live on `Processor`, built on the rayon pool at startup and kept resident.

**Encoder.** The selected image is encoded in the background once its source is decoded and it has stayed on screen for 250 ms (paging past images must not queue an encode each; an encode cannot be cancelled), even before the tool is on: refused below ~800 MB free RAM, cached as `BatchItem.magic_brush_embedding` (16 MB). The shown image keeps its embedding, and with the tool on so does the one shown before it, so flipping between two stays instant; a refused speculative encode is not retried. Until the shown image has an embedding the tool reads as Preparing and holds clicks.

**Decoder.** A click, or a stroke decimated to 8 points, is decoded on the pool; the best candidate's 256² logits are upsampled to source size (~8 ms at 4K) and pixels reaching the Confidence probability get the mode's sign (Restore / Erase on a cut-out, Add / Subtract elsewhere). The result merges like a paint stroke, so undo, overlay and re-cut behave the same for both brushes. The last decoder output is kept, so a Confidence drag re-thresholds it in place without a new undo step.

### Eraser (inpaint)

Eraser models are on-demand downloads. The region is `MaskArtifact::region_mask`, grown or shrunk by Expand region; each run starts from the previous result so earlier erases stay, and its output is archived as a `Result` step while the selection stays for Reprocess. A per-item generation counter drops superseded results.

| Family | Runs | Why |
|---|---|---|
| LaMa, Big-LaMa, MI-GAN | In-process on rayon (`prunr_core::inpaint`) | Under 1 GB |
| Stable Diffusion 1.5 / LCM, RORem | Inpaint-only subprocess (`gui/inpaint_bridge.rs`), spawned on first use, dropped after 5 min idle | Multi-GB; an OOM must kill the worker, not the GUI |

**Fill scale.** Every eraser fills the hole at the size its model likes (`FillScale::Auto`: a hole's long side near 400 px, so a large region fills at half or quarter resolution and is resampled back); the footer reports what ran. Full resolution tiles a large hole and shows the seams, so Auto is the default and Full suits small regions only.

**LaMa family.** 512 px tiles with 64 px feathered overlap; empty masks return before the session loads. Tiles run sequentially on one session (ORT already uses every core; do not parallelise). Sessions try GPU EPs through the same compatibility cache as seg models and are released after 5 min idle or on a model switch.

**Stable Diffusion.** CLIP, VAE, the 9-channel UNet and five schedulers (LCM, DDIM, DPM++ 2M Karras, Euler-A, UniPC) are in `inpaint_sd.rs`, checked against Diffusers. The UNet always gets a binary mask, as SD 1.5 was trained on. Free RAM picks one tall crop up to 768 px or 512² tiles (see [SD RAM policy](#sd-ram-policy)); CPU fallback is refused. Inside the worker, sessions are dropped after each stroke unless "Keep the Stable Diffusion eraser loaded" is on (~16 GB resident); the worker itself stays until 5 min idle.

**RORem** is SDXL distilled to four LCM steps for object removal: its prompt is baked into the export, so it takes no prompt or guidance, only Steps (2–8; fewer are a faster rough draft, more change the fill about as much as a new seed does) and the shared seed. Its sessions run without ONNX Runtime's memory-pattern planning, a registry fact (`SdSpec::lean_sessions`) measured on it as the same speed at 15 GB peak instead of 21.7; SD 1.5 keeps ORT's default until measured. The bench's `PRUNR_SD_*` overrides (`SdTuning::with_env_overrides`) are part of the session key.

**Cancel and progress.** The `Progress` handle carries the cancel flag, checked between tiles and SD steps (ORT has no per-op cancel), and reports the masked tiles, crops and denoising steps.

**Seam.** Softness is composite-time only (`inpaint_blend::finalize_inpaint`, both families): colour-match to a ring of source pixels, then an edge-preserving guided filter over a band of Edge blend width around the seam, then optional sharpen. Edges survive where a Gaussian-blur composite would smear them.

## Live Preview

Mask, edge and post-upscale knob changes on the selected item, and brush re-cuts, rerun Tier 2 in-process on the rayon pool; the subprocess's ~20-50 ms of IPC per run would ruin a drag, while batch reruns amortise it.

**Throttle, not cancel.** The first tweak of a drag opens a 150 ms window and the item dispatches when it closes, so a drag yields a frame every ~150 ms; releasing a slider flushes at once. Runs are never cancelled mid-drag (the pipeline can't stop between stages, so cancelling only threw away finished frames); each dispatch carries a generation and stale results are dropped. Previews honour every mask setting, guided-filter refine included. `UpscaleRerun` is refused here: a multi-second job would look like a hang.

**Reuse.** A drag decompresses the seg tensor once and caches the masked base and edge planes keyed on the knobs that built them, so e.g. a Lines thickness drag only re-dilates.

## Presets

A preset is one JSON file in the platform config dir (`prunr/presets/`), readable and shared by sending the file. It holds an entry per model (keyed by name so unknown future models round-trip): item settings, brush settings and, for SD, the prompt and a bundle per scheduler. Saving merges into the file, keeping other models' entries; v1 files migrate on load. Curated built-ins are seeded once (a marker keeps deleted ones deleted). `"Prunr"` is the synthetic factory default and cannot be saved over or deleted; `default_preset` is what new imports inherit. Applying a preset is a `PresetApply` undo step.

## Memory Management

Estimate before loading, release what the user isn't looking at, and let a subprocess die instead of the GUI when the estimate is wrong.

### Where pixels live

| Data | Kept as | Released |
|------|---------|----------|
| Source | A path (`ImageSource::Path`), read only at dispatch; paste keeps bytes | Decoded copy stays once viewed |
| Non-selected result | Released from the item; the timeline's stored copy is zstd'd off the UI thread | Restored from the timeline on reselect |
| Undo timeline | Shared hot `Arc` → warm zstd in RAM → cold file in `<cache dir>/prunr-history/{pid}/` | 64 steps per image; warm → cold under pressure |
| Seg / edge tensors | zstd on the item; decoded only during a drag | See [Tiered Recipe Pipeline](#tiered-recipe-pipeline) |
| Thumbnail (160 px) | RAM, always | With the item |
| Magic Brush embedding (16 MB) | On the item | Kept for the shown image, and with the tool on the previous one |

Timeline states carry their recipe, so undo also restores what the next Process tier-routes from. Memory pressure (available < 20 % of total) demotes stored states to disk and evicts all non-selected tensors.

### Admission and parallel jobs

Budget = 85 % of available RAM − the model's CPU needs × engines (GPU pools cap at 2). Per-image cost (`2 × W × H × 4` + file + history residue) is pessimistic on purpose: undercounting OOMs long batches, overcounting only slows them. Best-fit admits the largest image that fits; with nothing in flight the smallest is forced in, so oversized images can't deadlock. Admission also pauses while the child's RSS exceeds 80 % of available RAM (resume at 70 % of that). `safe_max_jobs` (half of available RAM ÷ the CPU needs) caps the jobs slider, clamps on model switch and at batch start.

### Per-model working set

`ModelDescriptor.needs` holds each model's needs per accelerator (system RAM and card memory, each figure marked measured or estimated with its reason), one field per accelerator and no default, so a model does not build without its numbers and a test holds the count of unmeasured accelerators. An accelerator not measured yet is charged the CPU figure in system RAM. Batch admission, `safe_max_jobs`, the upscale gate and the SD gate read it; a new model is one registry entry.

### SD RAM policy

One gate, `prunr_core::inpaint_sd::check_ram_for`, run by the process that runs the stroke: need = the model's needs (zero if the bundle is already resident) + the user's safety margin (default 2 GB), + 3 GB for a tall crop. A region up to 768 px long runs as one crop instead of 512² tiles when that fits (faster, no seam), as tiles when only they fit, and is refused otherwise. The GUI never decides: a kept bundle lives in the SD subprocess, so only that process can see it is loaded rather than count it twice.

## Data Flow

### Segmentation batch

```
UI      resolve_tier per item → skip | RePostProcess (cached tensor) | TEED-only | full pipeline
        archive results, snapshot dispatch recipe, admit full-pipeline items
bridge  inputs / tensors / chain input → IPC temp files → commands
worker  ImageDone { result, tensor + edge cache paths }
UI      store result + compressed tensors, applied_recipe = snapshot, admit next
```

A crash or 60 s of silence triggers [Crash retry](#crash-retry).

### Starting image

There is no switch for what a run reads; the undo timeline decides. Each step records the engine that made it and the image that run started from (`Origin`). A run of that same engine redoes the step from that image, so a re-cut never stacks on its own output; any other engine builds on the shown result, and the eraser always does, since each erase adds to the last. The upscaler is the one exception: with its settings unchanged since the shown upscale, Process enlarges that upscale again, while a changed Scale or upscaler redoes it (`BatchItem::next_upscale_start`). A hand edit (Delete, Cut) has no origin, so the next run keeps it. `BatchItem::run_start` is the one place the rule lives; Process, the live preview, filter-only runs and upscale all ask it. A segmentation run that starts from an earlier result always runs the full pipeline, since the cached tensor never saw those pixels.

One exception, by design: Process with another cut-out model (or No model) than the one that cut the shown step builds on that cut-out, so a second model only takes more away and undo is how to start over. Its start is the shown step's own colours seen through the shown cut-out's transparency, not the shown pixels, so an effect, lines or faces are applied once; the worker bridge builds it, where the file is decoded anyway, and it becomes the step's start. A cut never shows what its start hides (`keep_hidden`, before brush edits, so Restore still wins). A cached segmentation re-cuts only the step its model cut, never No model's pass, whose start already carries the cut.

### CLI

`main.rs` routes `--worker`, `--doctor`, `bench`, `--open <path>` and plain inputs. `--inpaint` runs LaMa in-process; every other input, single or batch, goes to a `prunr --worker` subprocess by path, so the parent never holds image bytes (only `--large-image downscale` rewrites oversized inputs to a temp file). Crashes halve the jobs and retry. Exit code: 0 all ok, 1 all failed, 2 partial.

`prunr bench` (`prunr_app::bench`) runs each installed model in-process through the app's own entry points, once CPU-only and once through the EP ladder, one session at a time, released before the next model. Every session records a `SessionPick` (provider, weights, each provider that failed on the way), so the report says where a model landed and why. The report is rewritten after every step: one a crash left behind names the step it crashed in.

## Inference Pipeline

```
load_image_from_bytes  → DynamicImage (SVG sniffed, rasterized via resvg)
check_large_image      → error above 8000 px per side (CLI --large-image overrides)
preprocess             → NCHW f32: Silueta/U2Net 320², ÷ max pixel; BiRefNet 1024², ÷ 255; ImageNet norm
infer                  → raw [1,1,S,S]                                ← Tier 1
postprocess            → RGBA                                         ← Tier 2 starts here
  normalize (min-max; BiRefNet: sigmoid, then min-max) → gamma
  → Lanczos3 to source size → edge shift → guided filter → feather → contrast
  → selection correction (full resolution) → alpha into the one RGBA buffer
  → fill style → source backdrop
```

Feather runs after the guided filter: sharpen to colour edges first, then soften. The full pipeline postprocesses the output while it is still borrowed from ORT's `IoBinding` buffer, so the tensor is never copied. Tier 2 (`postprocess_from_flat`) starts from a cached tensor and skips inference entirely.

### Postprocess performance choices

- **One resizer.** Every resize goes through `fast_image_resize` (SIMD, rayon rows) from a borrowed buffer. Its pool nests safely inside the subprocess worker.
- **One RGBA allocation** per `postprocess`, shared by the guided filter and the alpha write.
- **Row-parallel only above 512² pixels.** Alpha writes are memory-bound, so the gain caps at about 1.2× (`apply_mask_inplace` 5.7 → 4.8 ms at 4000×3000, 1.05× end to end). The `#[ignore]` tests `apply_mask_inplace_4k_bench` and `postprocess_4k_bench` reproduce these numbers.
- **Guided filter**: O(1) box filters over f32 integral images, with each plane freed at its last use. Peak is the four parallel first-stage filters (12 planes, about 576 MB at 4K). Pairing them saves two planes but runs about 9 % slower, so that RAM-for-speed trade is left open.
- **Edge shift** (`morphology::shift_mask`): N iterated 3×3 min/max passes equal one separable (2N+1)² window, computed with van Herk / Gil-Werman passes. Cost per pixel is the same at any shift, and RAM stays at two mask planes. Line thickness uses the same kernel.

### Criterion microbenches

`crates/prunr-core/benches/`, run with `cargo bench -p prunr-core --bench <name>`. CI skips them because runner variance causes false alarms. Reference numbers are from an 8-core x86_64 machine at 4K unless noted.

| `--bench`        | Configuration                                         | Median |
|------------------|-------------------------------------------------------|-------:|
| `guided_filter`  | 512² / 2048²                                          | 4.4 / 74 ms |
| `edge_shift`     | 1 / 2.5 / 10 / 50 px                                  | 16 / 35 / 15 / 16 ms |
| `sam_decode`     | 256² logits → selection plane                         | 8 ms |
| `sam_preprocess` | photo → 1024² tensor                                  | 48 ms |
| `edge_compose`   | `edge_plane` (2 px); lines only plain / solid          | 32; 11 / 9 ms |
| `edge_compose`   | Solid / Gradient Y / Radial / Rainbow / Noise; dual   | 19 / 32 / 36 / 36 / 37; 34 ms |


## Upscale

Upscaling runs **in-process**: tiled `OrtEngine` inference on a background thread, outside the subprocess (`crates/prunr-core/src/upscale/`), or part by part for a model that normalizes over its whole input. Its tiler is separate from the LaMa inpaint tiler because the two have different shapes: upscale maps an image to scale × its size, while inpaint maps an image and mask to the same size. They share only the smoothstep seam weight.

Models are registry data: tile size, tile multiple, `needs`, an optional fp16 sibling, and `UpscaleModelKnobs` (native scale, window attention, input name, dtype).

| Model | Native | Tile | Overlap | ORT level |
|-------|-------:|-----:|--------:|-----------|
| Real-ESRGAN x4plus, 4x NMKD Siax-CX, 4x NMKD Superscale | 4× | 512 | 16 | Level3 |
| Real-ESRGAN x2plus | 2× | 512, even | 16 | Level2 |
| 4xNomos8kSCHAT-L (fp16) | 4× | 256, ×16 | 32 | Level2 |
| 4xNomos2 RealPLKSR | 4× | parts, 256 crops | each part's reach | Level3 |

- **Whole-input norms run part by part.** Nomos2 normalizes its features over the whole image (GroupNorm), so tiles toned on their own statistics leave seams no overlap hides. It ships cut at its norms; each part runs crop by crop with a border covering its reach, and each norm takes the whole image's statistics, so the result matches a one-piece run. The features stay whole, about 0.5 KB per input pixel; short of free memory a run holds each block's output in half precision, or runs each block twice instead of holding it. OpenVINO is declared incompatible: it keeps a memory pool per compiled part.
- **Level2 for any model with a tile multiple.** Level3 bakes the first tile's shape into the graph, and the next differently padded tile fails.
- **Overlap follows window attention, not the tile multiple.** Window-attention models need two windows of overlap to hide seams; CNNs need only their receptive field.
- **fp16 siblings** download automatically when a GPU EP is active. Dispatch reads the loaded session's input dtype, not the registry flag.
- **Warm engine.** `Processor` caches one session and builds it on the dispatch thread, so a cold EP compile never freezes the window. Upscale uses the normal EP ladder.
- **Alpha** bypasses the 3-channel model. It is Lanczos3-resampled from the source and recombined at the end.
- **Output scale.** The model runs at its native scale, and `fit_to_scale` Lanczos3-resamples to the requested 2×, 3× or 4× whenever the two differ. 4× two-pass chains x2plus twice and is offered only on Real-ESRGAN x4plus once x2plus is installed.
- **Knobs.** Denoise and Brightness lift run before the model, so changing either is an `UpscaleRerun`: live preview refuses it, and the user commits with Process. Sharpen (default 0.2), AI blend (toward a cached CatmullRom resize), Saturation and Color match run after the model as `UpscaleTier2` and preview live on the cached output.
- **Single flight, fast cancel.** One job runs at a time. Cancel also calls `RunOptions::terminate`, which stops the running tile in about 50 ms. OpenVINO checks only between tiles, so its tiles are capped at 256: on an HD 530 a cancel lands in 11 s instead of 51 s, for 5 % more run time.
- **Admission.** Dispatch is refused when free RAM is below the model's CPU needs, the same figure subprocess admission uses.

## Colorize (DDColor)

Colorize is its own role (`ModelRole::Colorize`), not a filter: the picker shows DDColor, the toolbar shows a Strength chip and nothing that cuts out. `prunr_core::colorize` runs the model at 512² on the photo's lightness and keeps the predicted colour planes (`ColorizeCache` on the item's step, like a kept upscale run); Strength (0–150 %) scales them against the photo's own colour in a live re-render (`ColorizeTier2`), so the model runs once per photo. Colorize shares the single run slot with upscale and goes through the same run queue for ticked images.

## Depth

Depth Anything V2 Small measures a photo once (518 px long side, ~0.5 s) into a `DepthMap` kept on the item; every depth feature reads that map, and a photo with no map asks the depth queue on `Processor` for one (epoch-tagged, so a late map for an earlier source is dropped). Readers:

| Feature | Where | How |
|---|---|---|
| Lens blur, Point / Tilt-shift | Depth chip (No model); Background → Lens blur (cut-outs) | Blur grows with distance from the focus depth (or the band); the cut-out's own subject sets the focus behind a cut-out. `blur::blur_rgba` is a parallel three-box Gaussian (24 MP in 0.5 s); the Blurred photo background and Refine's Sharpen use it too |
| Fog, Aerial perspective | Depth chip | Haze thickening with distance; far parts turn cooler and paler |
| Fade with distance | Sketch | `line_fade_pct` on the edge recipe: lines lighten by depth (an `EdgeRerun`) |
| By depth | Magic Brush | The click's depth ± Range selects a band of the scene, ground included |
| Depth map | Save options, `--depth-map` | 16-bit PNG, white = near |

The depth look re-renders the photo in the live preview (`MaskRerun` for a cut-out, the photo rerun under No model), never the model, so a drag costs a re-render and no inference. While a slider is held the look is made at 2048 px on the long side and resampled back (`DepthLook::apply_quick`, 0.5 s instead of 2.9 s at 24 MP); the render on release is full size.

## Edge Detection (TEED)

`EdgeEngine` owns its own TEED session (58 k parameters). Input is 640×480 BGR with mean subtraction, whatever the photo's aspect, alpha flattened onto white, after the user's input transform (Grayscale, Contrast boost, Posterize).

| `LineMode` | Label | Behaviour |
|------------|-------|-----------|
| `Off` | Off | Background removal only |
| `EdgesOnly` | Full | Skip segmentation; lines of the whole image |
| `SubjectOutline` | Subject | Segment first, then draw lines over the masked subject |

**Two runs, three scales.** `Bold` and `Fused` are TEED's `out_3` and `block_cat` from the 640×480 run; `Fine` is `block_cat` from a second run at 1280×960, so its lines are half as wide. TEED's sigmoid floor is not 0 (≈ 0.44 on `block_cat`), so each plane is stretched min..max to 0..1 as TEED's own inference does, with the range floored so a flat image does not stretch its noise into lines; the cached planes hold these probabilities. A run makes the scales its lines draw (with the one sharing their run) and sends its result; for the image on screen the worker then makes the rest from the photo it kept resized, and the app adds them under a token so a later run's lines never take them. Fine is about three quarters of the detector's time, so a result that does not draw it no longer waits for it. The scales are kept zstd-compressed on the item, so a scale switch costs a decompress (`EdgeRerun`); a scale the item lacks (a background image of a batch) runs the lines again.

**Live-preview plane caches.** A Lines tweak redoes only the stage whose inputs changed:

| Cached on the item | Key |
|--------------------|-----|
| Undilated edge plane (threshold, then Lanczos) | strength, scale |
| Dilated plane (a thickness drag is about 32 ms at 4K) | strength, scale, thickness |
| Masked-subject base (SubjectOutline) | mask recipe, model |

Edge planes are kept per scale: Bold, for the dual-scale style, and the newest other scale. A colour, style or compose change costs only the composition. The composition is row-parallel and branch-free per row, and it blends colour only at edge pixels.

## GPU Execution Providers

### ONNX Runtime loading

`ort` (2.0.0-rc.13) is built with `load-dynamic`: no ONNX Runtime is linked in. `prunr_core::ort_runtime` resolves `libonnxruntime` once per process, first hit wins:

1. `ORT_DYLIB_PATH` (developer override; set but missing is an error, never a fall-through)
2. **Preferred**: the runtime `<data>/prunr/runtimes/preferred` names, a Runtime Store install (`runtimes/<name>-<ver>-<rid>/`) or `built-in`. One process loads one runtime and no wheel carries another's EP, so the choice is a file every process reads (GUI, worker, CLI); installing a runtime makes it the preferred one, and Settings › Hardware picks among what can load
3. **Without a preference**: on Windows the bundled runtime (the DirectML build) before the store's first entry by name; elsewhere the store before the bundled one
4. **Bundled** is `<exe>/runtime/` or `<exe>/`; in a macOS `.app`, `Contents/Frameworks/`

Every session in the workspace starts from `ort_runtime::session_builder()`, which initialises first; `clippy.toml` disallows the raw `Session::builder` and `ort::init_from`. Under `load-dynamic` a builder on an unloaded runtime hangs instead of failing, so the one entry point turns a missing runtime into an error. The app exits at startup if the runtime cannot load; `prunr --doctor` runs before that check.

| Platform | Shipped runtime | GPU route |
|---|---|---|
| Linux | CPU-only ORT 1.30.0 (PyPI wheel) | OpenVINO build via Settings → Hardware (Runtime Store); CUDA build via `cargo xtask install-runtime` or `ORT_DYLIB_PATH` |
| Windows | DirectML ORT 1.24.1 (PyPI `onnxruntime-directml`: `onnxruntime.dll` + `DirectML.dll`, +23.5 MB over the CPU build) | built in; OpenVINO build via Settings → Hardware (Runtime Store), with a choice of which loads; CUDA build via `cargo xtask install-runtime` or `ORT_DYLIB_PATH` |
| macOS (arm64, 14+) | ORT 1.30.0 PyPI wheel, which carries CoreML | built in |

The plain wheel runs ahead of the accelerator builds (1.24.1; their wheels lag, and CUDA's need CUDA 13 from 1.27): constant folding corrupts weights on 1.24 to 1.28, and sessions there run with it off, which costs CoreML most of its models. 1.31 moved CoreML onto device-based provider selection and stopped applying our CoreML options, so the plain wheel stays at 1.30.

Runtime Store entries come from the official PyPI wheels; `prunr-runtime-install` (shared by the GUI installer and xtask) verifies the SHA and repackages the EP's shared libs under the canonical dylib name. A first-launch install prompt (14-day snooze) offers OpenVINO for an Intel iGPU, unless the built-in DirectML already has a graphics card to run on. GPU detection reads `/sys/class/drm` on Linux and DXGI's adapter list on Windows (no device created), so it needs no graphics context.

### EP ladder

Each session walks a ladder of the GPU EPs the loaded runtime actually has compiled in (`is_available`), registering one EP per attempt so the log names the winner and a crashing EP cannot abort the CPU fallback. If every GPU EP fails the engine retries CPU-only. Every EP is built in one place, `EpKind::provider`, set to fail its registration: ort's default logs the failure and commits on the CPU, under the GPU's name. A failed registration is a fact about the machine, so it moves to the next EP without an `ep_compat` record.

| Platform | Ladder |
|---|---|
| Linux | OpenVINO → CUDA → CPU |
| Windows | OpenVINO → CUDA → DirectML → CPU |
| macOS | CoreML → CPU |

CoreML compiles ML Programs (the format fp16 graphs need). A CoreML session swaps ORT's log sink for one that keeps CoreML's own `GetCapability` summary and logs it on commit as `partitions`, `nodes` and `coreml_nodes`: every partition edge is a CPU↔GPU round trip, so these counts are the data for deciding per model whether CoreML pays.

OpenVINO goes first because the common non-NVIDIA desktop is Intel; NVIDIA machines don't have an OpenVINO runtime loaded, so they fall through to CUDA. OpenVINO's thread pool is capped to the rayon budget to avoid oversubscription. `cpu_only` (Settings, CLI) skips the ladder.

### Per-(model, EP) compatibility filter

Two layers skip a doomed EP before paying the failed-load cost (~5 s for OpenVINO + Silueta):

1. **Static**: `ModelDescriptor.incompatible_eps` next to the registry entry, for verified-bad pairs (Silueta + OpenVINO: graph cycles).
2. **Dynamic**: `<data>/prunr/ep_compat.json`, written when an uncached session fails, per copy of the weights so FP16 failing never bars FP32. Keyed on the app version, so an upgrade retries everything; `prunr --clear-ep-cache` wipes it.

### Compiled-model cache

`<data>/prunr/ep_cache/<ep>/<model>-<version>-ort<runtime>-v<format>/` (macOS: under `~/Library/Caches/prunr/`) holds the compiled artefact so the second session build skips graph optimisation: ORT's optimised graph for CPU and CUDA, CoreML's compiled model on macOS. OpenVINO and DirectML are not cached (OpenVINO re-keys its cache every launch and rewrote ~3 GB per build for no speedup; DirectML relies on the OS shader cache). The format version is bumped with every `ort` upgrade because cached graphs are not portable across ORT versions. A commit that fails on a cached graph deletes the cache file instead of marking the EP incompatible. Stale versions are collected at session build; model uninstall and a Settings button clear it.

### Model variants

`OrtEngine` tries an optimised variant from disk before the FP32 bytes: FP16 on a GPU EP, INT8 on CPU. macOS uses FP32: ORT asks CoreML for FP32 compute on the GPU and CPU (the Neural Engine computes in FP16 whatever it is given), and whether FP16 weights would pay on a Mac is not measured yet. The CPU retry after a GPU failure also tries INT8 first.

## GUI State Machine

Nothing about an image's progress is stored. `BatchStatus` is derived from the undo timeline and the item's job: a cut under way is Processing, a result on the shown step (on screen or parked) is Done, a failure with nothing to show is Error, otherwise Pending. `AppState` (Empty / Loaded / Processing / Done) is derived in turn from the selected item's status. Undo, cancel and parking therefore cannot leave a stale state behind.

**One answer to "may this run".** `gui/capability` derives an image's facts (what it shows: the photo, a run's output or a hand edit; whether a run is queued or under way; which caches it holds) and answers each need as Ready, Wait or Never with a reason. The toolbar greys a surface by the same answer (`knob_catalog::surface_gate`) that the live-preview queue checks at its door, so a control the dispatch would refuse is never live, and a render that can never be built leaves the queue instead of waiting for good.

**Process is for big-model output.** The cut-out, Upscale, Colorize and the erasers run on Process. Controls that need that output (Mask, Outline & Shadow, Effect and Background on a cut-out model, Refine, Strength) wait for it, greyed with "Process first."; everything else (Faces, Depth, Magic Brush, brushes, Lines on No model) works on what the canvas shows. Exports write what the canvas shows: a result, or the photo with its restored faces.

## Canvas & Texture Lifecycle

- **Off-thread prep, on-thread upload.** `ColorImage` conversion runs on worker threads behind the shared decode slot pool (bounding peak RAM during a batch-completion burst); the upload happens in the per-frame channel drain, never in a render closure.
- **One reconcile step.** Textures for the selected item (source, result, background image, selection) are requested by `reconcile_selected` (see [GUI Coordinators](#gui-coordinators)), each behind a pending flag.
- The old texture stays bound until its replacement uploads, so result arrivals and preview ticks never blank the canvas. A result switch bumps `result_switch_id`, which seeds the 0.4 s crossfade.
- Zoom resets on navigation (selecting an item, a fresh decode), never on a result arriving.
- Checkerboard: one 256×256 texture, tiled. Off-screen sidebar rows skip painting.

### Backgrounds

One Background chip picks transparent, solid colour, image, or a source-derived effect; the choices are mutually exclusive.

| Kind | Stored on | Display | Export | Tier |
|---|---|---|---|---|
| Colour | `ItemSettings.bg` | rect painted under the result texture | composited at save | CompositeOnly |
| Image | `BatchItem.bg_image` (`Arc` + content hash); hash and fit on `ItemSettings` | textured quad under the result | `apply_background_image` at save | CompositeOnly |
| Effect (Blurred / Inverted / Desaturated source) | `ItemSettings.bg_effect` | baked into the result RGBA | already in the pixels | MaskRerun |

**Colour and image are render-time.** The result texture keeps its transparency; the canvas paints checkerboard, then the backdrop, then the result, and the GPU blends. A change costs a repaint: no compositing, no texture rebuild, no dispatch. Only export bakes it in, because PNG has no separate backdrop. The sidebar thumbnail uses the same layering. Image bytes live on the item (they don't fit the `Copy` settings) with only the hash on settings, the same shape as brush corrections. `BgImageFit` (Cover / Contain / Stretch / Tile / Center) is UV math on one texture uploaded with repeat wrap, so changing the fit never re-uploads. The CLI's `--bg-image` beats `--bg-color` when both are set.

**Effects are baked** because they need source pixels behind the subject, not a flat rect; a per-frame backdrop texture would double render bandwidth and complicate the compositor. They are applied inside postprocess, so live preview of mask knobs shows the current backdrop.

## Drag-Out and Layer Export

Dragging a sidebar thumbnail out hands the OS real PNGs written to `{temp_dir}/prunr-drag/` (`drag` crate, Windows and macOS; on Linux it needs a GTK window winit cannot give, so it is excluded and a toast explains). Drops from that folder are rejected so our own drag is not re-imported; such a drop also clears the drag state, since the completion callback does not always fire on Windows.

A drag is one PNG matching the canvas, background baked in like Save and Copy. With `export_split_layers` it is up to three layers (subject, lines, mask) re-rendered from cached tensors, subject without fill or background effect; missing tensors skip a layer, and none falls back to the composite. Save writes the same layers to a folder, which is the Linux path.

## Where Data Lives

| What | Location |
|------|----------|
| `settings.json`, `presets/*.json` | `{config_dir}/prunr/` |
| On-demand models | `{data_dir}/prunr/models/` |
| Runtime Store, compiled-model cache, EP compatibility | `{data_dir}/prunr/runtimes/`, `ep_cache/`, `ep_compat.json` |
| Compiled-model cache on macOS | `~/Library/Caches/prunr/ep_cache/`: CoreML can fail to open a compiled model under a path with a space |

Directories come from `dirs` (`~/.config` and `~/.local/share` on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows). Settings hold user choices only, hotkey overrides and accepted model licenses included; CPU forcing and the active backend reset every launch. Old files load through serde defaults and a v1 migration.

## Model Registry & Distribution

Every model is one row in `prunr_models::REGISTRY` (source, GPU requirement, known-bad EPs, RAM working set, upscale knobs), so adding a model is a data edit.

| Source | Models |
|--------|--------|
| `Bundled` (zstd `include_bytes!`) | Silueta, BiRefNet-lite, TEED, the YuNet face finder (0.2 MB, so faces are found without a face model) |
| `MultiPartBundled` | SAM 2.1 Hiera Small, so Magic Brush works offline on first click |
| `OnDemand` | U2Net, LaMa, Big-LaMa, MI-GAN, five upscale models (the RRDB ones with an fp16 sibling for GPU EPs), DDColor, Depth Anything V2 Small, the face models (GFPGAN, RestoreFormer++) |
| `MultiPartOnDemand` | SD 1.5 Inpaint, its LCM variant, TAESD, RORem (its SDXL UNet in 2 GB parts), 4xNomos2 (cut at its norms); restrictive licenses need an explicit accept |

`resolve_bytes` / `resolve_part_bytes` are the only byte entry points. Bundled blobs decompress once; on-demand files are read at each load and dropped once the session has copied them, so a model used once doesn't hold its file's size for the rest of the run; SHA-256 is checked at download, not load. A missing model errors with a pointer to the Model Store, the only install surface. In the GUI every surface says the same thing about a model that isn't installed (`views::get_model`: "Get <model> · <size>", from the registry) and opens the Store on that model's section; a chip whose model is missing shows "Get" on its face, and the Model picker lists every model, downloaded or not. `DownloadManager` runs one download at a time (`.partial`, verify, rename; transient errors retry, multi-part bundles keep finished parts across a cancel).

Our exports live on this repo's GitHub releases (`models-v1`, `lcm-inpaint-v1.0.1`, `taesd-v1.0.0`) under versioned names, so a URL an old build knows never changes; a fixed export gets a new name. SD 1.5 Inpaint comes from Hugging Face.

`--features dev-models` reads `models/*.onnx` (or its `.zst`) instead of embedding; `cargo xtask fetch-models` fetches the list in `xtask/src/models.rs`, writes the blobs and mirrors on-demand models into the data dir. xtask is outside `default-members`, or its `dev-models` dependency would unify into a release build and ship no models.

## Temp File Lifecycle

| Directory | Purpose | Cleanup |
|-----------|---------|---------|
| `{temp_dir}/prunr-drag/` | Drag-out PNGs | >10 min at startup; all on exit |
| `{cache_dir}/prunr-history/{pid}/` | Cold undo history | Emptied on exit; dead PIDs' dirs at startup; never while the session runs |
| `/dev/shm/prunr-ipc-{pid}/` (Linux, when it holds 1 GiB), else `{temp_dir}/prunr-ipc-{pid}/` | Subprocess transfer | After read; own dir at startup; dead PIDs' dirs |

The IPC and history dirs are per PID so two instances never sweep each other (`fs_util::sweep_dead_pid_dirs`), and crash recovery deletes only the crashed subprocess's file prefixes, sparing the SD subprocess and the CLI's staged retry inputs.

## Windows Console and Renderer Fallback

Windows release builds use the GUI subsystem so no empty console opens; CLI mode and `--debug` attach to the parent console. On every platform the GUI tries glow (OpenGL), then wgpu, for VMs and old drivers.

## Build & Release

Release profile: fat LTO, one codegen unit, `panic = "abort"`, stripped. Rust is pinned to 1.92.0 in `rust-toolchain.toml` and the workflows so clippy `-D warnings` does not drift.

CI on pushes to master: clippy, then build and test with `dev-models` on Linux, macOS (x86_64, aarch64) and Windows. Only Linux installs ONNX Runtime and sets `PRUNR_REQUIRE_ORT`, so its inference tests fail instead of soft-skipping. macOS arm64 stages the runtime the release ships and runs `cargo xtask probe-load-dynamic --require-ep CoreMLExecutionProvider` on it: it must load under our `ort`, carry CoreML and run a model.

A `v*` tag builds and publishes; `workflow_dispatch` builds without publishing.

| Target | Artifacts |
|--------|-----------|
| linux-x86_64 | `.tar.gz`, `.AppImage`, `.deb`, `.rpm` |
| macos-aarch64 | `.dmg`, `.tar.gz` |
| windows-x86_64 | `.zip`, Inno Setup `.exe` |

Each artifact ships an ONNX Runtime so a clean machine needs no Runtime Store install. Every platform stages Microsoft's wheel with `cargo xtask install-runtime` (the in-app store's code): Linux the CPU one and Windows the DirectML one into `<exe dir>/runtime/`, hence `/usr/bin/runtime/` in the `.deb` / `.rpm`; macOS the arm64 one, which carries CoreML, into `Contents/Frameworks/`. The DirectML build runs on the CPU too: without a DirectX 12 GPU (VM, remote desktop) DirectML fails to register, `available_gpu_eps` drops it after a one-time probe, and every session is a CPU one from the same DLL. The macOS wheel sets the Mac floor: its minimum macOS (14.0) is the app's `LSMinimumSystemVersion` and the binary's deployment target.

The workspace `version` is the app version; packages take theirs from the tag. The ORT version pinned in the workflow YAML is mirrored in a Rust const, and a test fails when they differ.

## Key Dependencies

| Crate | Why |
|-------|-----|
| `ort` =2.0.0-rc.13 | ONNX Runtime with `load-dynamic` and the CUDA / CoreML / DirectML / OpenVINO EPs; exact pin, the RC API moves |
| `eframe` / `egui` 0.34 | GUI |
| `accesskit`; `egui_kittest` (dev) | Widget tree for the control socket and headless GUI tests |
| `image`, `resvg`, `fast_image_resize` | Decode / encode, SVG, SIMD Lanczos3 |
| `ndarray`, `half`, `rand_chacha`, `instant-clip-tokenizer` | Tensors; SD fp16, seeded noise, prompts |
| `zstd`, `bincode` | Model blobs and cold history; IPC wire format |
| `sysinfo`, `memory-stats` | Free RAM for gates; subprocess RSS |
| `reqwest`, `sha2`, `zip` | Downloads, verification, wheel extraction |
| `drag` | OS drag-out, Windows and macOS only |
