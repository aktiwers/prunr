# Changelog

All notable user-facing changes. Releases on GitHub carry the full commit list.

## Unreleased — 0.5.0

### Added

- **Magic Brush.** Click or drag over anything and SAM 2 (Hiera Small, bundled) finds it. On a background-removal result it restores or erases that object at once; on the eraser it adds to or takes away from the region. Shift and Alt override the mode for one click, a Confidence knob filters weak candidates. The encoder runs once per image in the background, so switching images keeps the tool ready.
- **Restore and Erase brushes.** On a background-removal result the Paint Brush restores what the model took or erases what it left, the moment the stroke ends, with Shift and Alt overriding the mode. The knobs touched every stroke sit in a strip over the image while a brush is on; its dots open the full panel.
- **Selections as a first-class object.** Paint Brush and Magic Brush author the same selection. Delete, Copy, Cut, Invert and Clear act on it, with keyboard shortcuts; Feather softens its edge against the photo.
- **Upscale.** Real-ESRGAN x4plus and x2plus, 4x NMKD Siax-CX and Superscale, and 4xNomos8kSCHAT-L, at 2×, 3×, 4× or a 4× two-pass look. Six refinement knobs: Denoise and Brightness lift before the model, Sharpen, AI blend, Saturation and Color match after it, the last four previewed live. fp16 variants download automatically on GPUs. Cancel aborts mid-tile.
- **Eraser with Stable Diffusion.** SD 1.5 Inpaint and its LCM fast variant join LaMa, Big-LaMa and MI-GAN. Prompt and negative prompt, guidance, five schedulers (LCM, DDIM, DPM++ 2M Karras, Euler-A, UniPC), steps, denoising strength, pinned seed, and the TAESD fast decoder. Quality presets (Fast / Balanced / Quality) pick scheduler and steps. SD runs in its own subprocess, so an out-of-memory stop never takes the app down.
- **Presets v2.** Per-model presets with brush settings, merge-save that keeps other models' entries, and automatic migration of v1 files.
- **Unified progress** banner or modal (your choice in Settings) for processing, erasing and upscaling, with step-level progress for SD.
- **Compiled-model cache** on CPU, CUDA and CoreML, so the second session with a model skips graph optimisation; Settings can clear it.
- **Help menu** with keyboard shortcuts, the command-line reference and pipeline diagrams; `prunr --open <path>` starts the app with an image loaded.
- **Keep the Stable Diffusion eraser loaded** (Settings › Behavior): repeated erases skip the session build, at the cost of about 16 GB of RAM while the app is open.
- **Screen-reader names.** Every control, icon buttons and sliders included, carries a readable name.
- **Rebindable hotkeys.** Settings › Hotkeys lists every action with two key slots; click one and press the new keys. A key already in use moves over, and the F1 list and tooltips show your bindings.

### Changed

- **One toolbar row.** Every pipeline stage is a group chip (Mask, Lines, Fill style, Background, Refine, Quality, Prompt, Advanced) with a header and a reset; the tool cluster is Magic, Paint, the selection actions and Compare, then Preset and Reset.
- **Panels that stay out of the way.** Mask, Lines, Fill style, Background and Refine stay open while you work on the image and fade while a slider is held; the chip whose panel is open keeps its accent outline. Picks in Model, Preset, Quality and Scale close their list. Escape closes one thing per press: the open panel, then the top dialog, then a running job, then the selection.
- Knob names follow the words other editors use: Opacity, Expand region, Edge blend, Feather, Edge shift, Denoise, Sharpen. American spelling throughout.
- Paint Brush and Magic Brush share size and shape; while either is on, its knobs sit in a strip over the image, and `[` / `]` change the size.
- Undo is one timeline across strokes, results and presets, with Cmd+Shift+Z as an alternate redo; stroke history is bounded at 32 snapshots per image.
- Chain mode defaults on; switching to an upscale model turns it on so the cut-out feeds the upscaler.
- Eraser models are on-demand downloads; LaMa is no longer embedded in the binary.
- Line color choices (Original, Solid color, styles) live in one list.
- The window can be made narrower: 1020 px minimum, down from 1100.
- The Protect selection lock is gone: on a background-removal result a stroke applies at once, and the eraser's region waits for Apply strokes.

### Performance

- Edge shift runs in constant time per pixel: a 4K mask at the knob's 50 px limit went from 2.5 s to 16 ms, and 1 px from 52 ms to 16 ms.
- Magic Brush decode (logits to a 4K selection) went from 147 ms to 8 ms per click.
- The selection overlay uploads only the changed region per stroke instead of a full-size texture.
- The guided filter frees its planes as soon as they are consumed; BiRefNet postprocess and the guided filter's interior loop gained row-parallel fast paths.
- A per-frame step keeps the selected image's textures, embedding and overlay current, replacing three ad-hoc polls.
- Live preview reuses the chained image across dispatches; SD schedulers reuse their scratch buffers; the upscale engine stays warm between runs.
- SD erases encode a prompt once per session instead of once per tile, and log the time of each stage.
- SD erases run a region up to 768 px long as one crop instead of two blended tiles when enough RAM is free: 21 percent faster on a 328 by 607 region, and no seam.
- The LCM eraser bundle (1.0.1) is re-exported with dynamic batch and size axes: a guided step is one UNet call instead of two, and it takes the single tall crop too. Existing installs download the new bundle, 2.1 GB.
- Cancelling an upscale on OpenVINO takes effect about five times sooner (11 s instead of 51 s on an Intel HD 530), for about 5 percent more upscale time.
- Magic Brush holds its image data for the shown image and the one before it, instead of 16 MB for every image visited, and paging quickly through images no longer prepares each one on the way.
- Stroke undo keeps only the area each stroke changed: a small stroke on a 4K image costs kilobytes instead of 8 MB, so 32 undo steps no longer hold up to a quarter gigabyte per image.
- One ONNX Runtime init and one session builder for the whole app: a missing or broken runtime now fails with a message instead of hanging.

### Fixed

- Magic Brush Confidence does what its hint says: it is the probability a pixel must reach, so higher keeps the sure core and lower grows into the rim, and moving it retunes the last click or stroke live. It used to gate the candidate choice and change nothing.
- Undo restores the cut-out after a Magic Brush or Invert stroke in chain mode; only Paint strokes archived the previous result.
- The selection outline stays visible when a large image is zoomed out to fit.
- Magic Brush no longer gets stuck on Preparing when Paint Brush is turned on while an image is being prepared.
- A fast Paint Brush stroke is continuous instead of a row of dots between pointer samples.
- Undoing the first stroke no longer leaves it drawn on the image until the next stroke.
- The click that closes a brush popover no longer paints a stroke.
- A Magic Brush click beside the image no longer selects whatever sits at the image's edge.
- Undo and redo step every stroke on every model, before or after a result, and a stroke is one step.
- A stroke on an image undone back to the original no longer brings the cut-out back.
- Switching to another image and back no longer costs an undo step.
- Processing again or deleting after undoing back to the original can itself be undone, and so can a Process started before a large image has finished loading.
- Undo steps older than half an hour no longer vanish, and two open Prunr windows no longer clear each other's undo files.
- With the Stable Diffusion eraser kept loaded, later erases keep the faster single crop instead of falling back to tiles.
- Ctrl+click also subtracts with Magic Brush on Linux and Windows, where window managers often take Alt+click.
- Magic Brush is ready on first use: its sessions warm at startup and the selected image is encoded in the background.
- The "Click or stroke to select" text is no longer painted over the image.
- Big-LaMa downloads load again (the release asset was an unpatched export).
- Delete and Cut work before an image is processed, and Delete is undoable.
- Stroke direction, softness and strength survive the trip into the selection and the correction.
- The selection and the Magic Brush embedding reset when an image's source is replaced.
- Upscale at 3× works (it errored), and a native-2× model asked for 4× now resamples up instead of returning a 2× image.
- Upscale: the right model kind reaches the recipe diff, undo works after an upscale, the brush and in-flight jobs are cancelled on a model switch, and the GUI stays responsive while a cancel drains.
- SD: the UNet always receives a binary mask, the mask is average-pooled into latent space, scheduler defaults match the Diffusers reference, and CPU fallback is refused for the SD bundle instead of silently taking minutes.
- Progress counters are scoped to the current dispatch; cancel routing is per kind.
- Buttons react to hover and press; choice rows no longer wrap inside a label; a stray accent line is gone.
- Windows CI no longer trips over `Instant` arithmetic on fresh runners.

## 0.4.8

Last release before this changelog. See the GitHub release for its commit list.
