# Phase 3: CLI Binary - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

CLI mode of the `bgprunr` binary — `bgprunr remove` subcommand for single image and batch processing. Exercises the full bgprunr-core API under real scripting conditions. No GUI work — that's Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Command Structure
- **Subcommand**: `bgprunr remove input.jpg -o output.png` for single image
- **Batch**: `bgprunr remove *.jpg --output-dir ./results/` for batch processing
- **No args**: launches GUI (Phase 4 — already decided in Phase 1)
- **Default model**: silueta (fast). Override with `--model u2net`
- Flags: `--model silueta|u2net`, `--jobs N` (default 1), `--large-image=downscale|process`, `--output-dir DIR`, `--force`, `--quiet`

### Output Naming
- Default naming: `{stem}_nobg.png` (e.g., `photo.jpg` → `photo_nobg.png`)
- Without `--output-dir`: output goes alongside the input file
- With `--output-dir DIR`: output goes into the specified directory
- **--force required to overwrite** existing files — without it, refuse and print error
- Output is always PNG (transparency requires it)

### Progress Display
- **Single image**: Stage spinner with current stage name (⠋ Preprocessing... → ⠋ Inferring... → ⠋ Applying alpha...)
- **Batch**: indicatif MultiProgress — overall progress bar (3/10 images) + per-image spinner showing current stage. When each image completes, print a summary line: `✓ car-1.jpg (1.2s)`
- **--quiet**: Suppresses all progress output. Only errors go to stderr. Good for piping/scripting.

### Exit Codes
- 0: All images processed successfully
- 1: Total failure (no images processed, or fatal error)
- 2: Partial failure in batch (some images succeeded, some failed)

### Claude's Discretion
- Exact clap derive struct layout
- Error message formatting
- Whether to use `clap::Parser` or `clap::Command` builder
- indicatif bar style/template
- How to detect single vs batch mode (file count vs explicit flag)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture
- `ARCHITECTURE.md` — CLI data flow diagram, single-binary dispatch pattern

### Existing Code
- `crates/bgprunr-app/src/main.rs` — Current placeholder (prints version). This becomes the CLI entry point.
- `crates/bgprunr-core/src/lib.rs` — Public API: `process_image()`, `batch_process()`, `OrtEngine`, `ModelKind`, etc.
- `crates/bgprunr-core/src/pipeline.rs` — `process_image()` with callback closure
- `crates/bgprunr-core/src/batch.rs` — `batch_process()` with per-image callback
- `crates/bgprunr-core/src/formats.rs` — `load_image_from_path()`, `check_large_image()`, `downscale_image()`, `encode_rgba_png()`
- `crates/bgprunr-core/src/types.rs` — `ModelKind`, `ProgressStage`, `ProcessResult`, `CoreError`

### Research
- `.planning/research/STACK.md` — clap 4.5, indicatif 0.17

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bgprunr_core::process_image()` — single image with progress callback
- `bgprunr_core::batch_process()` — parallel batch with per-image callback, `--jobs N`
- `bgprunr_core::OrtEngine::new()` — model loading with ModelKind enum
- `bgprunr_core::formats::load_image_from_path()` — image loading from filesystem
- `bgprunr_core::formats::check_large_image()` — returns bool for >8000px check
- `bgprunr_core::formats::downscale_image()` — downscale to max dimension
- `bgprunr_core::formats::encode_rgba_png()` — RGBA to PNG bytes

### Established Patterns
- Callback closures for progress: `|stage: ProgressStage, pct: f32|`
- Per-image batch callback: `|image_idx: usize, stage: ProgressStage, pct: f32|`
- thiserror enums with `#[from]` conversions
- `ModelKind::Silueta` and `ModelKind::U2net` variants

### Integration Points
- `main.rs` dispatches: no args → GUI (Phase 4), `remove` subcommand → CLI processing
- CLI uses the same `bgprunr_core` API that GUI will use in Phase 4
- clap `Parser` derive macro on a `Cli` struct with `Subcommand` enum

</code_context>

<specifics>
## Specific Ideas

- The completed summary line per image in batch (`✓ car-1.jpg (1.2s)`) gives users confidence that work is happening and lets them estimate remaining time
- `--force` for overwrite protection matches standard CLI conventions (cp, mv use similar patterns)
- `_nobg` suffix makes it obvious what happened to the file when browsing a directory

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-cli-binary*
*Context gathered: 2026-04-07*
