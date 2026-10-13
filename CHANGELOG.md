# Changelog

All notable user-facing changes. Releases on GitHub carry the full commit list.

## Unreleased — 0.5.0

### Added

- Eraser (RORem) has an Advanced chip: Steps (2–8; fewer are faster but rougher, more than the 4 it was trained for changes little) and Pin seed.
- **Process works on the ticked images for every model.** Upscalers and Colorize take them one at a time (the capsule counts "Image 2 of 5"); an eraser fills every ticked image that has a painted region and says how many had none. Tick boxes are bigger. The footer shows how large an erase filled its region ("Fill 24%"), and a pill at the bottom of the canvas says "Rendering…" or "Preparing…" when work goes on for more than a second with nothing else to show for it.
- `--feather` and `--edge-width` on the command line, so every Mask setting has a flag.
- **Magic Brush.** Click or drag over anything and SAM 2.1 (Hiera Small, bundled) finds it and adds it to the selection; Alt (or Ctrl) takes it away. A Confidence knob filters weak candidates. The encoder runs once per image in the background, so switching images keeps the tool ready.
- **Restore and Erase brushes.** Both brushes paint a selection: a stroke adds to it, Alt (or Ctrl) removes from it. On a background-removal result the switch Restore · Erase · Select says what the selection does, live as you paint: bring back what the model took, remove what it left, or only select. Switching re-applies the area; Restore and Delete in the toolbar keep the change and start a new area. The knobs touched every stroke sit in a strip over the image while a brush is on; its dots open the full panel.
- **Selections as a first-class object.** Paint Brush and Magic Brush author the same selection. Delete, Copy, Cut, Invert and Clear act on it, with keyboard shortcuts; Feather softens its edge against the photo.
- **Upscale.** Real-ESRGAN x4plus, 4x NMKD Siax-CX and Superscale, 4xNomos8kSCHAT-L, and 4xNomos2 RealPLKSR (about a third faster than Real-ESRGAN, with natural skin and hair texture), at 2×, 3× or 4×; a 2× Real-ESRGAN upscale runs Real-ESRGAN's own 2× model when it is installed, about four times faster. Refine: Denoise before the model; Sharpen, AI blend and Color match after it, previewed live. fp16 variants download automatically on GPUs. Cancel aborts mid-tile.
- **Eraser with Stable Diffusion.** SD 1.5 Inpaint joins LaMa, Big-LaMa and MI-GAN. Quality is Fast (the LCM download, a few steps) or Quality (DPM++ 2M Karras on the full model); without the Fast download it runs Quality. A prompt (empty continues the surroundings) and a negative prompt; under Advanced, steps and guidance in each quality's range, Pin seed, which keeps the seed of the fill on screen, and the TAESD fast decoder, on by default for Fast. Switching quality keeps the prompt and seed. SD runs in its own subprocess, so an out-of-memory stop never takes the app down.
- **Eraser (RORem)** in the Model Store: Stable Diffusion XL trained for object removal. On a large object (a person, a car, a bench across a third of the photo) it fills in the background (the road, the water, the snow) where LaMa leaves streaks and SD 1.5 draws something new in the shape of the hole. Four steps; about 75 s per erase on a 2015 4-core CPU after a one-time load, much faster on a GPU. Needs about 18 GB of free RAM on the CPU and a 5.5 GB download. Try another works on it too.
- **Restore faces** (the Faces chip, on background removal, No model and the upscalers): redraws soft, small or compressed faces with GFPGAN or RestoreFormer++, downloaded from the Model Store. Strength updates live without running the model again: up to 50% it adds the redrawn detail over the photo's own shading and skin tone, above it the redrawn shading too. A face larger than the restorer's 512 px takes its detail only as far as that resolution reaches; All faces or the Largest; Include hair widens it to hair and ears. On an upscale the faces are found in the enlarged image and go in after Refine. Saves, copies, drags and layers carry the restored faces.
- **Fill scale** for every eraser (Auto, Full, 1/2, 1/4; `--fill-scale` on the command line). A large object used to fill as flat grey blocks, because the model works in 512 px squares and the middle of a big hole had no photo around it. Auto keeps small areas at full size and fills large ones at a smaller scale: a bench across a 12 MP photo now fills as snow, in 22 s instead of 99 s. Each area is judged on its own, so a small mark next to a large one stays sharp. The Stable Diffusion eraser fills a large area as one crop instead of a patchwork of tiles, and a selection in many pieces (a slatted bench from Magic Brush) is filled as one area instead of hundreds of crops.
- **Presets v2.** Per-model presets with brush settings, merge-save that keeps other models' entries, and automatic migration of v1 files.
- **Animations switch** (Settings › Editing): off, every result appears at once and the working highlights hold still; the progress spinner keeps turning.
- **See the upscale as it happens.** Each finished tile shows its upscaled result in place, sharpened down to fit its square, and the final image reveals over the finished preview. The preview is at most screen-sized, so it costs a few MB whatever the output size, and the run takes as long as before.
- **Progress you can follow.** A capsule at the bottom of the canvas names each step for every model, counts tiles, steps, crops or images, and estimates the time left. A tile map over the image shows which tiles are done and which one is running. Magic Brush says "Loading Magic Brush" beside the cursor while it gets ready.
- **Compiled-model cache** on CPU, CUDA and CoreML, so the second session with a model skips graph optimisation; Settings can clear it.
- **Help menu** with the command-line reference and pipeline diagrams; `prunr --open <path>` starts the app with an image loaded.
- **Keep the Stable Diffusion eraser loaded** (Settings › Performance): repeated erases skip the session build, at the cost of about 16 GB of RAM, until you quit.
- **Screen-reader names.** Every control, icon buttons and sliders included, carries a readable name.
- **Outline & Shadow** for cut-outs: a sticker outline (width, color, round corners that follow soft hair edges) and a Soft or Drop shadow (opacity, blur, distance), under the subject, in the preview, Save and drag-out. `--outline PX[:HEX]` and `--shadow soft|drop` on the command line.
- **Try another** on the Stable Diffusion eraser's strip: fills the same region again from the image it started on, with a new seed. Undo brings the previous fill back.
- **Colorize** (DDColor, in the Model Store): colours a black-and-white photo. Pick "Colorize (DDColor)" and Process; Strength then sets how much of the colour shows, from grey to more saturated than predicted, without running the model again. `--colorize` with `--color-strength` on the command line.
- **Upscale on the command line**: `--upscale realesrgan|nomos|siax|superscale|nomos2` with `--scale`, `--denoise`, `--sharpen`, `--ai-blend`, `--color-match`, and `--restore-faces gfpgan|restoreformer` with `--face-strength`. A batch loads the model once.
- **Depth** (Depth Anything V2 Small, 95 MB in the Model Store, about half a second per photo on the CPU): the app measures how near each part of the photo is once, then every depth effect updates live without running the model again.
  - **Lens blur background** for cut-outs: the photo behind the subject blurs more the farther it is from the subject's depth, like a wide-open lens. `--bg-effect lens[:N]` on the command line.
  - **Depth chip** on No model: Lens blur (Point or Tilt-shift, with Amount and Set focus: click the image where it should be sharp), Fog that thickens with distance, and Aerial perspective, which turns far things cooler and paler like distant hills. Sketch lines draw over the result.
  - **Fade with distance** in Sketch's settings: lines lighten the farther away they are. `--line-fade <0-100>` on the command line.
  - **Select by depth** with Magic Brush: turn on By depth on its strip and a click selects everything at about that depth; Range sets how far it reaches. Shift adds and Alt subtracts as usual.
  - **Depth map export**: Save options › Depth map also writes `<name>-depth.png` (16-bit grey, white = near) beside each saved image, layers included. `--depth-map` on the command line, `--depth-only` for the maps alone.
- `--effect <name>` and `--bg-effect blur[:N]|bw` on the command line, matching the app's Effect list and background effects.
- Sketch's Pre-filter gains **Smooth**, which calms skin and noise before the lines are traced; Contrast boost and Posterize pick from a few steps.
- Wide Sketch lines draw with a round pen: round ends and corners instead of square ones.
- Color splash shows which hues it keeps on a strip under its sliders.
- **Rebindable hotkeys.** Settings › Hotkeys lists every action with two key slots; click one and press the new keys. A key already in use moves over, and tooltips show your bindings. F1 opens this tab, with the mouse gestures below the keys.

### Changed

- **Lines now come from TEED** instead of DexiNed: cleaner lines, faster (a Sketch on a 20 MP photo takes 1.8 s instead of 2.7 s), and the app is about 130 MB smaller. The Balanced scale is folded into Fused; saved settings, presets and `--line-scale balanced` read it as Fused.
- An upscaled result keeps its place on screen: a fitted image stays fitted, a zoomed-in view keeps its spot.
- **A new look.** Panels float as dark glass with round pill corners. Mask, Sketch, Effect, Outline & Shadow, Background and Refine open as a quick strip of their everyday knobs; its dots open the full panel under it. Effects and backgrounds are picture tiles, colors are swatch dots, choices are segmented pills. The model picker groups models by what they do. Settings has a sidebar of pages (Performance, Hardware, Images, Editing, Hotkeys, Credits) and the Model Store is a grid of cards that open in place. The brush panels hold only what their strip lacks, and on a cut-out Erase is drawn in warm red.
- Refine's Brightness lift and Saturation are gone (Brightness lift clipped highlights and banded shadows; no upscaler offers either), and so is the 4× two-pass Scale.
- The Fill style chip is called **Effect**, with None first in its list. The Channel swap effect is gone (saved presets with it load as None); Duotone looks as before, now drawn as a two-color gradient map.
- **Backgrounds are one choice**: Transparent, Solid color, Image, Blurred photo or Black & white photo. Picking one replaces the others. The Inverted background is gone. A solid color is opaque, as it is saved. Image fit is Cover, Contain or Tile: presets with Stretch load as Cover, with Center as Contain.
- The lines feature is called **Sketch** everywhere, with modes Off, Subject and Whole photo (was Full). Picking No model with Sketch on Subject switches to Whole photo, so the lines stay. A greyed-out Subject says why. On the command line `--line-detail` names `--line-strength`, and `--line-width` and `--line-composition` are new.
- The Inverse mask composition is gone: it only ever drew a ring just outside the subject. Presets that used it load as Lines only.
- Picking Subject filled while lines are in the photo's own colors switches them to black, since those lines are invisible on the filled subject.
- Line color styles work with Sketch on Full (gradients, Rainbow, Dual scale), and Dual scale's preview matches Process. The Chromatic and Noise styles are gone; presets that used them draw solid lines. Every style in the list says what it looks like.
- Presets: one Reset everywhere (a panel's Reset returns to your default preset, like the toolbar's); a background-removal preset applies on every background-removal model; applying a preset sets its brush settings as Reset does; deleting one asks for a second click.
- Refine edges reads Edge width (was Refine radius); its precision is a Detail percentage under Advanced instead of a number in scientific notation. The Mask panel and its F3 diagram list the steps in the order they run, Contrast last.
- **Shift edge** (was Edge shift) reads the way other editors do: + grows the cut-out, − shrinks it, over ±20 px. `--edge-shift` on the command line follows, so a positive value now grows the subject.
- **Contrast replaces Hard threshold** in the Mask panel: one slider that sharpens the cut-out's edge at the photo's full size, where 100% is a hard edge. Hard threshold cut at the model's lower resolution, so its edge was never really hard. Saved presets with it on load as Contrast 100%. On the command line `--contrast` replaces `--threshold`.
- Magic Brush shows only what changes its result (mode, Confidence, Remove specks) and draws a small fixed cursor; its size, hardness and shape only ever changed the cursor.
- How a selection looks on the image is one app setting (Settings › Editing: Selection fill and Outline the selection) instead of three sliders in each brush panel; outline width is gone, since the outline already widens as you zoom out.
- Each model shows only the controls that act on its image: no Mask chip without a background-removal model, no selection on an upscaler, and brushes say Add / Subtract wherever they shape a selection rather than a cut-out.
- Dragging a thumbnail out of the window (Windows, macOS) starts at once: its files are written in the background while the pointer rests on it or presses it, instead of when the drag begins.
- **Undo rebuilt on one timeline per image**, the way Krita, GIMP and Photoshop do it. Every step stores its whole state, so undo shows the stored image instantly instead of re-running a model. Knob changes are now undoable (a slider drag is one step), Clear selection is undoable, and an erase that lands removes its brush marks, which undo brings back. A result that finishes after you undo or move on is dropped instead of landing on the wrong state, and undo stops a run in progress.

- **One toolbar row.** Every pipeline stage is a group chip (Mask, Sketch, Effect, Background, Refine, Quality, Prompt, Advanced) with a header and a reset; the tool cluster is Magic, Paint, the selection actions and Compare, then Preset and Reset.
- **Panels that stay out of the way.** Mask, Sketch, Effect, Background and Refine stay open while you work on the image and fade while a slider is held; the chip whose panel is open keeps its accent outline. Picks in Model, Preset, Quality and Scale close their list. Escape closes one thing per press: the open panel, then the top dialog, then a running job, then the selection.
- Knob names follow the words other editors use: Opacity, Expand region, Edge blend, Feather, Edge shift, Denoise, Sharpen. American spelling throughout.
- Paint Brush and Magic Brush share size and shape; while either is on, its knobs sit in a strip over the image, and `[` / `]` change the size.
- Cmd+Shift+Z is an alternate redo; up to 64 undo steps per image.
- The "Use last result as input" switch is gone: each step remembers what it started from, so a new tool builds on the result while changing a setting or running the same tool again redoes the step instead of stacking on it. Undo goes back to the original.
- Eraser models are on-demand downloads; LaMa is no longer embedded in the binary.
- Line color choices (Original, Solid color, styles) live in one list.
- **Save as PNG, JPEG or WebP.** The arrow beside Save picks the format and, for a cut-out, Separate layers (moved from Settings). JPEG flattens transparency onto white; WebP is lossless and keeps it.
- **Processor: Auto or CPU only**, saved, for every model (was "Force CPU this session", which only reached background removal).
- **SD memory guard: Relaxed, Balanced or Strict** replaces the GB safety-margin slider; the memory note in Settings uses the same rule as the erase.
- The Model picker shows in the empty window, and its last row opens the Model Store (which left the Help menu).
- Reset to defaults keeps your presets, licence acceptances, background pictures and model, and the shipped hotkeys apply at once.
- The Live preview switch is gone: changes always preview.
- The window can be made narrower: 900 px minimum, down from 1100; a narrow toolbar shows its chips as icons.
- The Protect selection lock is gone: on a background-removal result a stroke applies at once, and the eraser's region waits for Process.
- The eraser's Feather fades the fill into the photo over a ring just outside the painted region, so a wide Feather never brings back the edge of what was erased; 0 is a hard edge. Sharpen starts at 0 on the Stable Diffusion eraser, which draws its own detail.

### Performance

- Eraser (RORem) peaks at 15 GB of memory instead of 20.5 GB, at the same speed: its sessions run without ONNX Runtime's memory-pattern planning.
- Edge shift runs in constant time per pixel: a 4K mask at the knob's 50 px limit went from 2.5 s to 16 ms, and 1 px from 52 ms to 16 ms.
- Magic Brush decode (logits to a 4K selection) went from 147 ms to 8 ms per click.
- The selection overlay uploads only the changed region per stroke instead of a full-size texture.
- The guided filter frees its planes as soon as they are consumed; BiRefNet postprocess and the guided filter's interior loop gained row-parallel fast paths.
- A per-frame step keeps the selected image's textures, embedding and overlay current, replacing three ad-hoc polls.
- Live preview reuses its starting image across dispatches; SD schedulers reuse their scratch buffers; the upscale engine stays warm between runs.
- SD erases encode a prompt once per session instead of once per tile, and log the time of each stage.
- SD erases run a region up to 768 px long as one crop instead of two blended tiles when enough RAM is free: 21 percent faster on a 328 by 607 region, and no seam.
- The LCM eraser bundle (1.0.1) is re-exported with dynamic batch and size axes: a guided step is one UNet call instead of two, and it takes the single tall crop too. Existing installs download the new bundle, 2.1 GB.
- Cancelling an upscale on OpenVINO takes effect about five times sooner (11 s instead of 51 s on an Intel HD 530), for about 5 percent more upscale time.
- Magic Brush holds its image data for the shown image and the one before it, instead of 16 MB for every image visited, and paging quickly through images no longer prepares each one on the way.
- Stroke undo keeps only the area each stroke changed: a small stroke on a 4K image costs kilobytes instead of 8 MB, so 32 undo steps no longer hold up to a quarter gigabyte per image.
- One ONNX Runtime init and one session builder for the whole app: a missing or broken runtime now fails with a message instead of hanging.

### Fixed

- A photo whose file is gone by the time a batch reaches it fails with "Failed to load" instead of staying Pending forever and holding its share of the batch's memory budget.
- A No-model result no longer blanks the image for a frame while its texture is rebuilt: the old one stays until the new one is uploaded, as every other result does.
- The Stable Diffusion eraser on Fast no longer finishes with nothing changed on a wide or tall area: its fast decoder only takes 512 px squares, so such an area failed and was skipped silently. It now runs at 512 px, and a part that fails stops the erase with a message.
- Erasing a large object no longer leaves a ring of stray coloured pixels along the edge on flat backgrounds (a road at night, snow, a wall). The same fix makes Refine edges exact on large photos.
- Upscale Sharpen starts off and sharpens detail rather than grain: it defaulted on with a one-pixel radius that boosted the model's texture on every upscale. Its radius now follows the enlargement.
- Upscale AI blend and Color match compare with the image that was upscaled, not the original photo: an erased object no longer ghosts back, and noise Denoise removed stays gone. The result is finished in the upscale thread instead of the UI thread.
- Scale and Refine hold still while an upscale runs; moving one used to throw the run away. Refine previews whatever the Live preview setting, and after an undo only re-finishes the shown step's own upscale.
- Process on an upscale model right after opening a large photo runs instead of doing nothing.
- An upscale model loads nothing of Magic Brush, so it no longer competes with the upscale for the CPU.
- Background effects work under Sketch: lines over a blurred or black-and-white photo, on Whole photo and Subject alike. They used to do nothing on Whole photo, break the Subject compositions and make the lines trace the backdrop. Changing one is instant instead of a re-run, and saving does the blur off the UI thread.
- A background image applies to every image in a run, not only the one on screen; it loads off the UI thread, and cancelling the file dialog keeps the background you had.
- Tile backgrounds tile by image pixels, so the screen matches the export at any zoom; a see-through background image keeps its color instead of darkening.
- With an effect on Sketch Subject, the lines are traced from the plain subject, so Pixelate or Halftone no longer turns them into steps after Process.
- The exported Lines layer of a split drag-out matches the canvas, Dual scale included.
- Separate layers keep your brush corrections and start from the image the cut-out ran on; an image whose cut-out data is gone saves as one file, and a toast says so.
- Model Store: Retry works after any failed download, a cancelled one returns to Download, a queued one says so, and disk usage counts the Stable Diffusion files.
- Auto-process on import runs only for background removal and No model; it used to start a wrong run with an eraser or upscaler selected.
- An upscale uses every core whatever Parallel jobs is set to.
- A refused or failed erase no longer leaves the app busy; switching the SD eraser between Fast and Quality with Keep loaded on no longer runs out of memory.
- Cmd+S works whenever the Save button shows; the sidebar's tick, delete and save are real buttons.
- `prunr a.jpg -o b.png` writes `b.png` (it wrote into a folder `b.png/` and failed), and outside a terminal each result and error prints a line instead of going to an invisible spinner.
- With No model, Process never runs a cut-out: an image left on Sketch Subject switches to Whole photo, and a batch takes the filter path whenever the shown image has no lines.
- Smooth softens large photos too (it blurred at full size, which shrank to nothing at the size the line detector reads), and the line detector's dots in the photo's corners are gone.
- Lines no longer mix modes: in Full a Pre-filter change no longer produces a subject outline, and going Full → Off → Subject traces the subject again instead of reusing lines from the whole photo.
- Compare after an upscale shows the original at the result's size instead of a quarter of it, and the B key follows the same rule as the button.
- The model chip's hint describes the selected kind of model; it said LaMa for every eraser and talked about Sketch with No model selected.
- The Mask Feather is continuous: values under half a pixel used to do nothing and the rest moved in whole-pixel steps.
- A Sketch "Full" result is no longer replaced by a cut-out when a Mask setting differs from its default, and a settings change on an eraser or upscale result no longer paints the original photo over it.
- Feather softens what the brushes do to a cut-out, not only Delete / Copy / Cut, and brush edits keep their edge at full resolution instead of following the model's coarser grid. The selection on the image now shows feathered edges and strokes under 50 % opacity.
- The eraser fills exactly what the brush cursor covers; at low Hardness the region used to shrink to half the cursor. Hardness is gone from the eraser, and Edge blend is now its Feather, the one softness control where the fill meets the photo.
- Switching models keeps your brush size, hardness, opacity, mode and selection look; only the eraser's region settings and the Stable Diffusion settings follow the model.
- Subtract and Alt take away from the selection with No model and from the eraser's region; they used to add to it. A fresh Magic Brush click restores, as its hint says.
- Restore and Erase strokes on a cut-out land exactly where you paint them; edge refinement no longer pulls them onto nearby edges.
- Magic Brush Confidence does what its hint says: it is the probability a pixel must reach, so higher keeps the sure core and lower grows into the rim, and moving it retunes the last click or stroke live. It used to gate the candidate choice and change nothing.
- Undo restores the cut-out after a Magic Brush or Invert stroke; only Paint strokes archived the previous result.
- Upscaling an upscaled image again, or dragging an effect or background slider, no longer stacks on its own output (grainy 16x upscales, effects piling up).
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
- Magic Brush no longer leaves a grid of specks where the model is unsure: its output is smoothed and tiny islands and pinholes are removed. A Remove specks switch in its settings panel (on by default) turns this off.
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
