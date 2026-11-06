# Prunr on the command line

The `prunr` program that opens the app also works without a window: give it image files and it processes them and exits. Without any image, it opens the app.

```
prunr [OPTIONS] [INPUTS]...
prunr bench [OPTIONS]
```

`prunr --help` lists every option; `prunr --version` prints the version.

## Examples

```bash
prunr photo.jpg                              # cut out the subject, saves photo.prunr.png
prunr photo.jpg -o result.png                # choose the file
prunr *.jpg -o clean/                        # a batch into a folder
prunr -m u2net portrait.jpg                  # another model
prunr -j 4 *.jpg -o out/                     # 4 images at once
prunr photo.jpg --bg-color ffffff            # white instead of transparent
prunr photo.jpg --bg-effect lens:60          # lens-blur the photo behind the subject
prunr --preset Studio *.jpg -o out/          # a built-in showcase from the app's Presets menu
prunr -m none --preset "My look" photo.jpg   # your preset's whole-photo look, no cut-out
prunr --lines logo.png                       # lines only
prunr --inpaint photo.jpg --mask mask.png    # erase the white part of the mask
prunr --upscale nomos2 --scale 4 photo.jpg   # saves photo_x4.png
prunr --colorize old.jpg                     # saves old_color.png
prunr photo.jpg --depth-map --depth-only     # only photo-depth.png
```

## Where settings come from

Every setting starts from Prunr's own setting for the model in use (the ones the app starts with), or from `--preset`'s when you give one. A flag you give overrides it. `--preset` takes a name from the app's Presets menu (any case), including the built-in showcases, or the path of a preset file exported from the app.

## Where results go

| Mode | Saved as |
|---|---|
| Cut-out, lines, No model | `{stem}.prunr.png` |
| `--inpaint` | `{stem}_erased.png` |
| `--upscale` | `{stem}_x{scale}.png` |
| `--colorize` | `{stem}_color.png` |
| `--depth-map` | `{stem}-depth.png`, beside the result |

Without `-o` each result goes beside its input. With one input, `-o` names the file (a name without an extension is saved as `.png`) or an existing folder. With several inputs, `-o` must be a folder; it is made when missing. Inputs whose results would share a name in one folder each get their own, as in the app: `IMG.jpg` and `IMG.png` save as `IMG.prunr.png` and `IMG-2.prunr.png`. An existing result is skipped unless you give `-f`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Every image was saved |
| 1 | None was saved, or the command could not start (a wrong option, a model not installed) |
| 2 | Some were saved, some failed |

## Models

Silueta, BiRefNet-lite and TEED (lines) come with Prunr. Every other model the command line uses (U2Net, LaMa, the upscalers, DDColor, the face restorers, Depth Anything) is downloaded once in the app's Model Store; the command line says which one is missing. The Magic Brush and the other erasers (Big-LaMa, MI-GAN, Stable Diffusion, RORem) are only in the app.

## Options

### Input and output

| Option | What it does |
|---|---|
| `[INPUTS]...` | The images to process |
| `-o, --output <PATH>` | Where to save: a file for one image, a folder for several |
| `-f, --force` | Overwrite results that are there already |
| `-q, --quiet` | Print errors only |
| `-j, --jobs <N>` | Images processed at once. Default: as the app, half the processor cores, 2 on a graphics card, fewer when memory is short |
| `--large-image <POLICY>` | Images over 8000 px on a side: `process` (default) keeps their size, and a photo too large for the free memory is refused; `downscale` shrinks them to 4096 px first |

### Model

| Option | What it does |
|---|---|
| `-m, --model <MODEL>` | `birefnet-lite` (default, the finest detail), `silueta` (fast), `u2net`, or `none`: no cut-out, the preset's whole-photo look (effect, depth look, faces) or lines over the photo |
| `--preset <NAME\|FILE>` | Start from this preset's settings for the model |
| `--cpu` | Run every model on the processor, even with a graphics card |

### Mask

| Option | What it does |
|---|---|
| `--gamma <N>` | Removal strength: above 1 removes more, below 1 keeps more |
| `--contrast <N>` | Edge contrast, 0 to 1: 0 keeps the soft edge, 1 cuts it hard |
| `--edge-shift <PX>` | Grow (positive) or shrink (negative) the cut-out by up to 20 px |
| `--feather <PX>` | Soften the edge by this many pixels; fractions allowed |
| `--refine-edges[=BOOL]` | Refine the edge for hair, fur and leaves; `--refine-edges=false` turns a preset's off |
| `--edge-width <PX>` | How far from the edge `--refine-edges` looks |
| `--refine-guide <GUIDE>` | What `--refine-edges` follows: `brightness`, or `color` too, which also finds an edge between two colours of the same brightness (slower) |

### Lines

| Option | What it does |
|---|---|
| `--lines` | Draw the photo's lines instead of cutting it out |
| `--lines-after-bg` | Cut out, then draw lines on the result |
| `--line-detail <N>` | How much of the photo becomes lines, 0 to 1: lower draws bold outlines only. `--line-strength` is the same |
| `--line-width <PX>` | Line width, 0 to 20 px |
| `--line-scale <SCALE>` | `fine` (thin, detailed), `bold` (big outlines only) or `fused` (both); `balanced` is read as `fused` |
| `--line-composition <MODE>` | With `--lines-after-bg`: `lines-only`, `subject-filled`, `engraving` or `ghost` |
| `--line-color <HEX>` | Paint every line one colour, e.g. `000000` |
| `--line-fade <PERCENT>` | Lines lighten with distance, 0 to 100 (needs Depth Anything) |

### Subject

| Option | What it does |
|---|---|
| `--effect <NAME>` | Restyle the subject's colours with the effect's default settings: `desaturate`, `invert`, `duotone`, `sepia`, `threshold`, `posterize`, `solarize`, `hue-shift`, `saturate`, `color-splash`, `pixelate`, `cross-process`, `halftone` or `gradient-map` |
| `--outline <PX[:HEX]>` | A sticker outline around the cut-out: `8` (white) or `8:000000` |
| `--shadow <KIND>` | A shadow under the cut-out: `soft` (all around) or `drop` (below) |

### Background

What goes behind the cut-out; without one it stays transparent. When several are given, `--bg-effect` wins over `--bg-image`, which wins over `--bg-color`, as in the app.

| Option | What it does |
|---|---|
| `--bg-color <HEX>` | A colour, e.g. `ffffff` |
| `--bg-image <PATH>` | A picture (PNG, JPEG, WebP or BMP) |
| `--bg-image-fit <MODE>` | How the picture fits: `cover` (fills, may crop), `contain` (fits inside) or `tile` (repeats at its own size) |
| `--bg-effect <EFFECT>` | A copy of the photo: `blur` or `blur:N` (N = 1 to 64 px), `lens` or `lens:N` (N = 0 to 100: blurs more the farther from the subject, needs Depth Anything), or `bw` (black and white) |

### Eraser

| Option | What it does |
|---|---|
| `--inpaint` | Fill the area the mask marks with LaMa. One input image |
| `--mask <PATH>` | The mask, the size of the image: white is filled, black is kept |
| `--fill-scale <SCALE>` | How large the area is filled: `auto` (small areas at full size, large ones smaller), `full`, `half` or `quarter`. A large area fills more convincingly at a smaller scale, a little softer |

### Upscale

| Option | What it does |
|---|---|
| `--upscale <MODEL>` | Enlarge instead of cutting out: `realesrgan` (all-round), `nomos2` (fast, natural skin and hair), `siax` (clean or mildly compressed photos), `superscale` (noisy or compressed photos) or `nomos` (4xNomos8kSCHAT-L: the most detailed, the slowest) |
| `--scale <N>` | 2, 3 or 4 times larger |
| `--denoise <N>` | Denoise before the model, 0 (off) to 1 |
| `--sharpen <N>` | Sharpen after it, -1 (soften) to 1; 0 is off |
| `--ai-blend <N>` | 1 keeps the model's result, 0 the plain enlargement |
| `--color-match[=BOOL]` | Match the colours to the original; `--color-match=false` turns a preset's off |

A batch loads the model once.

### Faces

| Option | What it does |
|---|---|
| `--restore-faces <MODEL>` | Redraw soft, small or compressed faces: `gfpgan` (the crispest) or `restoreformer` (closest to the person's own features). After an upscale, otherwise into the photo before the cut-out, effect and lines. Not with `--inpaint` or `--colorize` |
| `--face-strength <PERCENT>` | 0 to 100: up to 50 adds detail and keeps the photo's shading, above it the redrawn shading too |

### Colorize

| Option | What it does |
|---|---|
| `--colorize` | Colour a black-and-white photo with DDColor |
| `--color-strength <N>` | 0 is grey, 1 the predicted colour, up to 1.5 |

### Depth

| Option | What it does |
|---|---|
| `--depth-map` | Also write each image's depth map, `{stem}-depth.png` (16-bit grey, white is near) |
| `--depth-only` | With `--depth-map`: write only the depth maps |

`--bg-effect lens` and `--line-fade` use the depth too. Each photo is measured once, whatever reads its depth.

### App and diagnostics

| Option | What it does |
|---|---|
| `--open <PATH>` | Open the app with this image loaded |
| `--debug` | Detailed log for a bug report; on Windows the app also writes it to the terminal it started from |
| `--doctor` | Print the computer's hardware, Prunr's runtime, the installed models and Prunr's folders, to paste into a bug report |
| `--clear-ep-cache` | Forget which models failed on which graphics engine, so each is tried there again (after a driver update) |
| `-h, --help` | Every option |
| `-V, --version` | The version |

Upscale, Colorize and Eraser are modes of their own: one at a time, and not with the background flags (`--lines`, `--lines-after-bg`, `--line-color`, `--line-fade`, `--bg-color`, `--bg-image`, `--bg-effect`, `--effect`, `--contrast`).

## `prunr bench`

Checks the graphics card path model by model: each installed model runs on the processor and on the accelerator the app would pick, timed (session build, first run, steady runs), and the two results are compared. When they differ, the report says which one looks right, and both pictures are saved beside it. It writes `prunr-bench.md` and `prunr-bench.json` to send back.

| Option | What it does |
|---|---|
| `--models <NAMES>` | Models to run, comma-separated (e.g. `silueta,migan`); every installed one when left out |
| `--runs <N>` | Timed runs on each side after the first, untimed one, 0 to 100 (default 3) |
| `--image <PATH>` | Run on this picture instead of the built-in one |
| `--heavy` | Also run the multi-GB models: Stable Diffusion 1.5 (both), RORem and 4xNomos8kSCHAT-L. Minutes each on a processor |
| `--out <PATH>` | Where to write the report (default `prunr-bench.md`); the JSON copy goes beside it |
| `--table <LABEL>` | Also print where each model should run on this computer, as the model registry's lines, labelled with where it was measured (e.g. `"RTX 3060, 2026-10-25"`) |
| `--needs` | Measure memory instead of speed: each model's peak memory and graphics memory on the processor and on the accelerator, each in a fresh process |
| `--tile <PIXELS>` | With `--needs`: give an upscaler a square input of this side, 16 to 2048 |
