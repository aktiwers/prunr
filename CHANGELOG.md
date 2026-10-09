# Changelog

All notable user-facing changes. Releases on GitHub carry the full commit list.

## Unreleased — 0.5.0

### Added

- **Magic Brush.** Click or drag over anything and SAM 2 (Hiera Small, bundled) selects it. Shift adds to the selection, Alt subtracts, a Confidence knob filters weak candidates. The encoder runs once per image in the background, so switching images keeps the tool ready.
- **Selections as a first-class object.** Paint Brush and Magic Brush author the same selection. Delete, Copy, Cut, Invert and Clear act on it, with keyboard shortcuts; Feather softens its edge against the photo; an Auto-apply switch decides whether strokes correct the cut-out at once or wait.
- **Upscale.** Real-ESRGAN x4plus and x2plus, 4x NMKD Siax-CX and Superscale, and 4xNomos8kSCHAT-L, at 2×, 3×, 4× or a 4× two-pass look. Six refinement knobs: Denoise and Brightness lift before the model, Sharpen, AI blend, Saturation and Color match after it, the last four previewed live. fp16 variants download automatically on GPUs. Cancel aborts mid-tile.
- **Eraser with Stable Diffusion.** SD 1.5 Inpaint and its LCM fast variant join LaMa, Big-LaMa and MI-GAN. Prompt and negative prompt, guidance, five schedulers (LCM, DDIM, DPM++ 2M Karras, Euler-A, UniPC), steps, denoising strength, pinned seed, and the TAESD fast decoder. Quality presets (Fast / Balanced / Quality) pick scheduler and steps. SD runs in its own subprocess, so an out-of-memory stop never takes the app down.
- **Presets v2.** Per-model presets with brush settings, merge-save that keeps other models' entries, and automatic migration of v1 files.
- **Unified progress** banner or modal (your choice in Settings) for processing, erasing and upscaling, with step-level progress for SD.
- **Compiled-model cache** per hardware backend, so the second session with a model on OpenVINO or DirectML skips graph compilation; Settings can clear it.
- **Help menu** with keyboard shortcuts, the command-line reference and pipeline diagrams; `prunr --open <path>` starts the app with an image loaded.

### Changed

- **One toolbar row.** Every pipeline stage is a group chip (Mask, Lines, Fill style, Background, Refine, Quality, Prompt, Advanced) that opens a popover with a header and a reset; the tool cluster is Magic, Paint, the tool's chip, the selection actions and Compare, then Preset and Reset.
- Knob names follow the words other editors use: Opacity, Expand region, Edge blend, Feather, Edge shift, Denoise, Sharpen. American spelling throughout.
- Both brush popovers share the cursor block (size, hardness, shape, live preview); changing it in either changes both tools.
- Undo is one timeline across strokes, results and presets, with Cmd+Shift+Z as an alternate redo; stroke history is bounded at 32 snapshots per image.
- Chain mode defaults on; switching to an upscale model turns it on so the cut-out feeds the upscaler.
- Eraser models are on-demand downloads; LaMa is no longer embedded in the binary.
- Line color choices (Original, Solid color, styles) live in one list; Protect selection became the Auto-apply strokes switch.

### Performance

- Edge shift runs in constant time per pixel: a 4K mask at the knob's 50 px limit went from 2.5 s to 16 ms, and 1 px from 52 ms to 16 ms.
- Magic Brush decode (logits to a 4K selection) went from 147 ms to 8 ms per click.
- The selection overlay uploads only the changed region per stroke instead of a full-size texture.
- The guided filter frees its planes as soon as they are consumed; BiRefNet postprocess and the guided filter's interior loop gained row-parallel fast paths.
- A per-frame step keeps the selected image's textures, embedding and overlay current, replacing three ad-hoc polls.
- Live preview reuses the chained image across dispatches; SD schedulers reuse their scratch buffers; the upscale engine stays warm between runs.
- One ONNX Runtime init and one session builder for the whole app: a missing or broken runtime now fails with a message instead of hanging.

### Fixed

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
