use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Prunr — local AI background removal.
///
/// No arguments launches the GUI. Pass image files directly to process them:
///   prunr photo.jpg              # removes background, saves photo_nobg.png
///   prunr *.jpg -o clean/        # batch to folder
///   prunr -m u2net portrait.jpg  # use quality model
#[derive(Parser, Debug)]
#[command(name = "prunr", version, about, long_about = None)]
pub struct Cli {
    /// Input image file(s).
    pub inputs: Vec<PathBuf>,

    /// Output path. File for single image, directory for batch.
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Launch GUI mode with this image pre-loaded (skips the file picker).
    /// Useful for shortcuts ("Open with Prunr…") and for the test harness,
    /// which needs to land an image without driving a modal file dialog.
    #[arg(long, value_name = "PATH")]
    pub open: Option<PathBuf>,

    /// Model: birefnet-lite (default), silueta (fast), or u2net (quality).
    #[arg(short = 'm', long, default_value = "birefnet-lite")]
    pub model: CliModel,

    /// Number of parallel jobs for batch processing.
    #[arg(short = 'j', long, default_value_t = 1)]
    pub jobs: usize,

    /// How to handle images exceeding 8000px in either dimension.
    #[arg(long, default_value = "downscale")]
    pub large_image: LargeImagePolicy,

    /// Overwrite existing output files.
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Suppress progress output (errors still go to stderr).
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Verbose diagnostics: bumps tracing filter to `prunr=debug` and, on
    /// Windows, attaches the parent console so GUI-mode stderr is visible
    /// in the launching terminal. Use to capture pipeline diagnostics
    /// when reporting a bug.
    #[arg(long)]
    pub debug: bool,

    /// Mask gamma (removal strength). >1 = more aggressive, <1 = gentler.
    #[arg(long, default_value_t = 1.0)]
    pub gamma: f32,

    /// Binary threshold (0.0–1.0). Pixels below become fully transparent.
    #[arg(long)]
    pub threshold: Option<f32>,

    /// Edge refinement in pixels. Positive erodes (shrinks), negative dilates (expands).
    #[arg(long, default_value_t = 0.0)]
    pub edge_shift: f32,

    /// Refine mask edges using guided filter for better detail on hair, leaves, etc.
    #[arg(long)]
    pub refine_edges: bool,

    /// Extract lines/edges instead of removing background (uses DexiNed model).
    #[arg(long)]
    pub lines: bool,

    /// Run line extraction after background removal (combine both).
    #[arg(long)]
    pub lines_after_bg: bool,

    /// Line detection sensitivity (0.0–1.0). Lower = bold outlines, higher = fine detail.
    #[arg(long, default_value_t = 0.5)]
    pub line_strength: f32,

    /// Paint all lines a solid color (hex, e.g. "000000" for black, "ff0000" for red).
    #[arg(long)]
    pub line_color: Option<String>,

    /// DexiNed output scale: `fine` (micro-edges), `balanced` (mid-scale),
    /// `bold` (abstract outlines), `fused` (default, combined).
    #[arg(long, default_value_t = prunr_core::EdgeScale::Fused)]
    pub line_scale: prunr_core::EdgeScale,

    /// Fill transparent background with a color (hex, e.g. "ffffff" for white).
    #[arg(long)]
    pub bg_color: Option<String>,

    /// Fill transparent background with an image (PNG/JPEG/WebP/BMP). Wins
    /// over `--bg-color` when both are set (matches GUI mutual exclusion).
    #[arg(long, value_name = "PATH")]
    pub bg_image: Option<std::path::PathBuf>,

    /// How the background image is positioned: cover (default — fills both
    /// dims, may crop), contain (fits inside, may letterbox), stretch
    /// (distort to fill), tile (repeat at native size), center (1:1 native).
    #[arg(long, default_value_t = prunr_core::BgImageFit::Cover)]
    pub bg_image_fit: prunr_core::BgImageFit,

    /// Force CPU inference even when GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// Internal: run as subprocess worker for GUI batch processing.
    #[arg(long, hide = true)]
    pub worker: bool,

    /// Print a diagnostic dump (hardware profile, ORT runtime status,
    /// installed models, paths) and exit. Paste the output into bug
    /// reports when hardware acceleration misbehaves.
    #[arg(long)]
    pub doctor: bool,

    /// Clear the persistent EP × model compatibility cache. Use after
    /// updating drivers / OpenVINO Runtime to re-discover which EPs
    /// can actually run each model on this machine.
    #[arg(long)]
    pub clear_ep_cache: bool,

    /// Chain mode: process the previous result instead of the original.
    /// Only meaningful in multi-pass scripting workflows where the user
    /// runs prunr multiple times on the same file. For a single invocation
    /// this flag has no effect since there is no previous result.
    #[arg(long)]
    pub chain: bool,

    /// Object removal (Eraser) mode. Reads a binary mask from --mask
    /// (white = inpaint, black = keep) and runs LaMa to fill in the
    /// masked region. Output saved as {stem}_erased.png by default.
    #[arg(long)]
    pub inpaint: bool,

    /// Path to the inpaint mask image. Required with --inpaint.
    /// Any image format; pixels >128 in the first channel are treated
    /// as "inpaint here". Mask must match the input image's dimensions.
    #[arg(long)]
    pub mask: Option<PathBuf>,
}

/// Model selection
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliModel {
    /// Silueta (~4MB, fast)
    Silueta,
    /// U2Net (~170MB, higher quality)
    U2net,
    /// BiRefNet-lite (~214MB, best detail at 1024×1024) — default
    BirefnetLite,
}

impl From<CliModel> for prunr_core::ModelKind {
    fn from(m: CliModel) -> Self {
        match m {
            CliModel::Silueta => prunr_core::ModelKind::Silueta,
            CliModel::U2net => prunr_core::ModelKind::U2net,
            CliModel::BirefnetLite => prunr_core::ModelKind::BiRefNetLite,
        }
    }
}

/// How to handle images exceeding LARGE_IMAGE_LIMIT
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargeImagePolicy {
    /// Downscale to DOWNSCALE_TARGET (4096px) before inference — safe default
    Downscale,
    /// Process at original size — may be slow or OOM on limited hardware
    Process,
}

use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use prunr_core::{
    MaskSettings, ModelKind, ProgressStage, CoreError,
    DOWNSCALE_TARGET,
    load_image_from_path, check_large_image, downscale_image, encode_rgba_png,
};

use crate::gui::settings::LineMode;

fn parse_hex_color(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 { return None; }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some([r, g, b])
}

impl Cli {
    fn mask_settings(&self) -> MaskSettings {
        MaskSettings {
            gamma: self.gamma,
            threshold: self.threshold,
            edge_shift: self.edge_shift,
            refine_edges: self.refine_edges,
            ..Default::default()
        }
    }

    fn line_mode(&self) -> LineMode {
        if self.lines_after_bg {
            LineMode::SubjectOutline
        } else if self.lines {
            LineMode::EdgesOnly
        } else {
            LineMode::Off
        }
    }
}

/// Entry point for processing. Returns exit code (0/1/2).
pub fn run_remove(args: &Cli) -> i32 {
    if args.inputs.is_empty() {
        eprintln!("error: no input files specified. Usage: prunr <images...> [options]");
        eprintln!("Run `prunr --help` for more info.");
        return 1;
    }

    // Ensure output directory exists if -o points to a dir (create eagerly, handle error)
    if let Some(ref out) = args.output {
        if args.inputs.len() > 1 {
            if let Err(e) = std::fs::create_dir_all(out) {
                eprintln!("error: cannot create output directory {}: {e}", out.display());
                return 1;
            }
        }
    }

    if args.inpaint {
        return run_inpaint(args);
    }

    // All processing uses subprocess isolation for OOM protection
    run_batch(args)
}

/// Eraser mode: load image + mask, run LaMa, save inpainted result.
/// Single-file only for now — batch eraser via per-image mask paths
/// would need a `--mask-dir` flag (not yet wired).
fn run_inpaint(args: &Cli) -> i32 {
    if args.inputs.len() != 1 {
        eprintln!("error: --inpaint accepts exactly one input image (got {})", args.inputs.len());
        return 1;
    }
    let input = &args.inputs[0];
    let Some(mask_path) = args.mask.as_ref() else {
        eprintln!("error: --inpaint requires --mask <path>");
        return 1;
    };

    let out_path = output_path_with_suffix(input, &args.output, false, "_erased.png");
    if let Err(e) = check_overwrite(&out_path, args.force) {
        eprintln!("error: {e}");
        return 1;
    }

    let img = match image::open(input) {
        Ok(i) => i.into_rgba8(),
        Err(e) => {
            eprintln!("error: failed to read {}: {e}", input.display());
            return 1;
        }
    };
    let mask = match image::open(mask_path) {
        Ok(i) => i.into_luma8(),
        Err(e) => {
            eprintln!("error: failed to read mask {}: {e}", mask_path.display());
            return 1;
        }
    };
    if img.dimensions() != mask.dimensions() {
        eprintln!(
            "error: mask {:?} dimensions don't match image {:?}",
            mask.dimensions(), img.dimensions()
        );
        return 1;
    }

    if !args.quiet {
        eprintln!("Inpainting {} (mask: {})...", input.display(), mask_path.display());
    }
    // CLI defaults to LaMaFp32 — Big-LaMa selection from the CLI is
    // tracked in PLAN 17-10 (would add `--inpaint-backend big-lama`).
    let raw = match prunr_core::inpaint::process_inpaint(&img, &mask, prunr_models::ModelId::LaMaFp32) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: inpaint failed: {e}");
            return 1;
        }
    };
    // Post-process: color-match + seam guided blend. Same defaults as
    // the GUI path so CLI output matches what the user sees on-screen.
    let color_matched = prunr_core::inpaint_blend::color_match_inpainted(
        &raw, &img, &mask,
        prunr_core::inpaint_blend::COLOR_MATCH_RING_PX,
    );
    let result = prunr_core::inpaint_blend::seam_guided_blend(
        &color_matched, &img, &mask,
        prunr_core::inpaint_blend::SEAM_BLEND_RADIUS,
        prunr_core::inpaint_blend::SEAM_BLEND_EPSILON,
        prunr_core::inpaint_blend::SEAM_BLEND_BAND_PX,
    );
    if let Err(e) = result.save(&out_path) {
        eprintln!("error: failed to save {}: {e}", out_path.display());
        return 1;
    }
    if !args.quiet {
        eprintln!("Saved {}", out_path.display());
    }
    0
}

/// Variant of `output_path` that lets callers override the `_nobg.png`
/// suffix. Eraser writes `_erased.png`; the BG-removal path keeps its
/// existing default.
fn output_path_with_suffix(
    input: &std::path::Path,
    output: &Option<PathBuf>,
    is_batch: bool,
    suffix: &str,
) -> std::path::PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let name = format!("{stem}{suffix}");
    match output {
        Some(out) if is_batch || out.is_dir() => out.join(name),
        Some(out) => out.clone(),
        None => input.with_file_name(name),
    }
}

// ── Output path helpers ──────────────────────────────────────────────────────

/// Compute output path for an input image.
/// Batch mode or -o is a directory: write {stem}_nobg.png into that directory.
/// Single mode with -o file.png: use that path directly.
/// No -o: write alongside input as {input_dir}/{stem}_nobg.png.
fn output_path(input: &std::path::Path, output: &Option<PathBuf>, is_batch: bool) -> std::path::PathBuf {
    output_path_with_suffix(input, output, is_batch, "_nobg.png")
}

/// Check if output exists and --force is not set. Returns Err with message if blocked.
fn check_overwrite(out: &std::path::Path, force: bool) -> Result<(), String> {
    if out.exists() && !force {
        Err(format!(
            "output '{}' already exists. Use --force to overwrite.",
            out.display()
        ))
    } else {
        Ok(())
    }
}

// ── Image loading with large-image handling ──────────────────────────────────

/// Load image bytes applying the LargeImagePolicy.
/// Returns PNG-encoded bytes ready for process_image / process_image_unchecked.
fn load_with_policy(
    path: &std::path::Path,
    policy: LargeImagePolicy,
    quiet: bool,
) -> Result<Vec<u8>, CoreError> {
    let img = load_image_from_path(path)?;

    if let Some(_large_err) = check_large_image(&img) {
        match policy {
            LargeImagePolicy::Downscale => {
                if !quiet {
                    eprintln!(
                        "warning: '{}' exceeds 8000px. Downscaling to {}px (use --large-image=process to skip).",
                        path.display(), DOWNSCALE_TARGET
                    );
                }
                let downscaled = downscale_image(img, DOWNSCALE_TARGET);
                // Encode to PNG bytes for process_image
                let rgba = downscaled.into_rgba8();
                return encode_rgba_png(&rgba);
            }
            LargeImagePolicy::Process => {
                if !quiet {
                    eprintln!(
                        "warning: '{}' exceeds 8000px. Processing at full size (--large-image=process).",
                        path.display()
                    );
                }
                // Fall through — process_image_unchecked will skip the guard
            }
        }
    }

    // Encode as PNG bytes for uniform handling
    let rgba = img.into_rgba8();
    encode_rgba_png(&rgba)
}

// ── Spinner label mapping ────────────────────────────────────────────────────

fn stage_label(stage: ProgressStage) -> &'static str {
    match stage {
        ProgressStage::LoadingModel => "Loading model...",
        ProgressStage::LoadingModelCpuFallback => "GPU warming up \u{2014} using CPU...",
        ProgressStage::Decode      => "Decoding...",
        ProgressStage::Resize      => "Resizing...",
        ProgressStage::Normalize   => "Normalizing...",
        ProgressStage::Infer       => "Inferring...",
        ProgressStage::Postprocess => "Postprocessing...",
        ProgressStage::Alpha       => "Applying alpha...",
    }
}

// ── Single-image execution path ──────────────────────────────────────────────

// ── Batch execution path ─────────────────────────────────────────────────────

fn run_batch(args: &Cli) -> i32 {
    // 4 Hz redraw cap (default is 20 Hz). With 1 overall + N spinners
    // every redraw recomputes every bar's template; on a 6-image batch
    // the per-tick fan-out cost the perf trace caught was ~13% of
    // prunr-binary self-time. 4 Hz feels indistinguishable for image
    // processing cadence and is the standard non-game-CLI rate.
    let mp = if !args.quiet {
        Some(MultiProgress::with_draw_target(
            indicatif::ProgressDrawTarget::stderr_with_hz(4),
        ))
    } else {
        None
    };

    // Overall progress bar: "3/10 images"
    let overall = mp.as_ref().map(|m| {
        let pb = m.add(ProgressBar::new(args.inputs.len() as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "{bar:40.cyan/blue} {pos}/{len} images  {elapsed_precise}"
            )
            .expect("static indicatif template compiles"),
        );
        pb
    });

    // Per-image spinners (one per input, added to MultiProgress)
    let spinners: Vec<Option<ProgressBar>> = args.inputs.iter().map(|input| {
        mp.as_ref().map(|m| {
            let pb = m.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg}")
                    .expect("static indicatif template compiles")
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            pb.set_message(format!("{} (waiting...)", input.display()));
            // 250 ms (was 80 ms): the worker emits Progress at fixed
            // pipeline checkpoints, with multi-second silence across
            // `session.run()`. Without a tick the spinner glyph would
            // freeze during the most-uncertain wait. The outer
            // `MultiProgress` 4 Hz cap throttles total redraws, so
            // per-spinner cost is bounded regardless of N.
            pb.enable_steady_tick(std::time::Duration::from_millis(250));
            pb
        })
    }).collect();

    // Validate images lazily — only check dimensions, don't load bytes into RAM.
    // The subprocess reads file bytes on demand when processing each image.
    let start_times: Vec<Instant> = args.inputs.iter().map(|_| Instant::now()).collect();
    let mut valid_indices: Vec<usize> = Vec::new();
    let mut valid_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut load_fail_count = 0usize;

    for (idx, input) in args.inputs.iter().enumerate() {
        match std::fs::File::open(input)
            .ok()
            .and_then(|f| {
                image::ImageReader::new(std::io::BufReader::new(f))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
            })
        {
            Some((w, h)) if args.large_image == LargeImagePolicy::Downscale
                && (w > prunr_core::LARGE_IMAGE_LIMIT || h > prunr_core::LARGE_IMAGE_LIMIT) =>
            {
                // Oversized — downscale to temp file, pass temp path to subprocess
                match load_with_policy(input, args.large_image, args.quiet) {
                    Ok(bytes) => {
                        let temp = prunr_app::subprocess::protocol::ipc_temp_dir()
                            .join(format!("cli_ds_{idx}.img"));
                        if std::fs::write(&temp, &bytes).is_ok() {
                            valid_indices.push(idx);
                            valid_paths.push(temp);
                        } else {
                            load_fail_count += 1;
                        }
                    }
                    Err(e) => {
                        load_fail_count += 1;
                        if let Some(Some(pb)) = spinners.get(idx) {
                            pb.finish_with_message(format!("X {} — {e}", input.display()));
                        }
                    }
                }
            }
            Some(_) => {
                // Normal size — pass original path directly (zero RAM cost)
                valid_indices.push(idx);
                valid_paths.push(input.clone());
            }
            None => {
                load_fail_count += 1;
                if let Some(Some(pb)) = spinners.get(idx) {
                    pb.finish_with_message(format!("X {} — not a valid image", input.display()));
                }
            }
        }
    }

    let model: ModelKind = args.model.into();
    let spinners_arc = std::sync::Arc::new(spinners);
    let inputs_arc = std::sync::Arc::new(args.inputs.clone());
    let quiet = args.quiet;
    let mask = args.mask_settings();

    let line_mode = args.line_mode();
    let solid_line_color = args.line_color.as_deref().and_then(parse_hex_color);
    let bg_color = args.bg_color.as_deref().and_then(parse_hex_color);
    let bg_image = match args.bg_image.as_deref() {
        Some(p) => match prunr_core::load_image_from_path(p) {
            Ok(img) => Some(std::sync::Arc::new(img)),
            Err(e) => {
                eprintln!("error: --bg-image: {e}");
                return 1;
            }
        },
        None => None,
    };
    let bg_image_fit = args.bg_image_fit;

    let edge = prunr_core::EdgeSettings {
        line_strength: args.line_strength,
        solid_line_color,
        edge_thickness: 0,
        edge_scale: args.line_scale,
        compose_mode: prunr_core::ComposeMode::default(),
        line_style: prunr_core::LineStyle::default(),
        input_transform: prunr_core::InputTransform::default(),
    };
    let batch_results = run_batch_subprocess(
        &valid_paths, &valid_indices, model, args.jobs, mask, args.cpu,
        line_mode, edge, bg_color, bg_image.clone(), bg_image_fit,
        &spinners_arc, &inputs_arc, quiet,
    );

    // Compute output paths and write results
    let mut success_count = 0usize;
    let mut fail_count = 0usize;
    fail_count += load_fail_count;

    for (batch_idx, result) in batch_results.iter().enumerate() {
        let original_idx = valid_indices[batch_idx];
        let input = &args.inputs[original_idx];
        let elapsed = start_times[original_idx].elapsed();
        let out_path = output_path(input, &args.output, true);

        let spinner_opt = spinners_arc.get(original_idx).and_then(|o| o.as_ref());

        match result {
            Ok(pr) => {
                // Overwrite check
                if let Err(msg) = check_overwrite(&out_path, args.force) {
                    if let Some(pb) = spinner_opt {
                        pb.finish_with_message(format!("X {} — {msg}", input.display()));
                    } else if !args.quiet {
                        eprintln!("skipped '{}': {msg}", input.display());
                    }
                    fail_count += 1;
                    continue;
                }
                let png_bytes = match prunr_core::encode_rgba_png(&pr.rgba_image) {
                    Ok(b) => b,
                    Err(e) => {
                        fail_count += 1;
                        if let Some(pb) = spinner_opt {
                            pb.finish_with_message(format!("X {} — encode error: {e}", input.display()));
                        }
                        continue;
                    }
                };
                match std::fs::write(&out_path, &png_bytes) {
                    Ok(_) => {
                        success_count += 1;
                        if let Some(pb) = spinner_opt {
                            pb.finish_with_message(format!(
                                "✓ {} ({:.1}s)",
                                input.display(),
                                elapsed.as_secs_f32()
                            ));
                        } else if !args.quiet {
                            println!("✓ {} ({:.1}s)", input.display(), elapsed.as_secs_f32());
                        }
                        if let Some(ref opb) = overall { opb.inc(1); }
                    }
                    Err(e) => {
                        fail_count += 1;
                        if let Some(pb) = spinner_opt {
                            pb.finish_with_message(format!("X {} — write error: {e}", input.display()));
                        } else {
                            eprintln!("error writing '{}': {e}", input.display());
                        }
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                if let Some(pb) = spinner_opt {
                    pb.finish_with_message(format!("X {} — {e}", input.display()));
                } else {
                    eprintln!("error processing '{}': {e}", input.display());
                }
            }
        }
    }

    if let Some(ref opb) = overall { opb.finish_and_clear(); }

    if !args.quiet {
        println!("{} succeeded, {} failed.", success_count, fail_count);
    }

    // Exit codes: 0 = all success, 1 = all failed, 2 = partial
    if fail_count == 0 {
        0
    } else if success_count == 0 {
        1
    } else {
        2
    }
}

/// Run batch processing via subprocess with auto-retry on OOM.
/// Accepts file paths — bytes are read lazily by the subprocess.
fn run_batch_subprocess(
    valid_paths: &[std::path::PathBuf],
    valid_indices: &[usize],
    model: ModelKind,
    initial_jobs: usize,
    mask: MaskSettings,
    force_cpu: bool,
    line_mode: LineMode,
    edge: prunr_core::EdgeSettings,
    bg_color: Option<[u8; 3]>,
    bg_image: Option<std::sync::Arc<image::DynamicImage>>,
    bg_image_fit: prunr_core::BgImageFit,
    spinners: &[Option<ProgressBar>],
    inputs: &[std::path::PathBuf],
    quiet: bool,
) -> Vec<Result<prunr_core::ProcessResult, CoreError>> {
    use prunr_app::subprocess::manager::SubprocessManager;
    use prunr_app::subprocess::protocol::SubprocessEvent;

    let mut results: Vec<Option<Result<prunr_core::ProcessResult, CoreError>>> =
        (0..valid_paths.len()).map(|_| None).collect();
    let mut pending: std::collections::VecDeque<usize> = (0..valid_paths.len()).collect();
    let mut max_jobs = initial_jobs;

    loop {
        if pending.is_empty() { break; }

        // Spawn subprocess — cap engines at pending item count
        let effective_jobs = max_jobs.min(pending.len());
        let (mut sub, _provider) = match SubprocessManager::spawn(
            model, effective_jobs, mask, force_cpu, line_mode, edge,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Can't spawn — fail all remaining
                for &idx in &pending {
                    results[idx] = Some(Err(CoreError::Model(e.clone())));
                }
                break;
            }
        };

        let mut in_flight: Vec<usize> = Vec::new();

        // Send initial burst
        let burst = max_jobs.min(pending.len());
        for _ in 0..burst {
            if let Some(idx) = pending.pop_front() {
                let item_id = valid_indices[idx] as u64;
                if sub.send_image_path(item_id, valid_paths[idx].clone()).is_err() {
                    pending.push_front(idx);
                    break;
                }
                in_flight.push(idx);
            }
        }

        // Event loop
        let mut crashed = false;
        loop {
            if !sub.is_alive() && !in_flight.is_empty() {
                crashed = true;
                break;
            }

            // Block up to 50 ms for the next event. Wakes immediately on
            // arrival — admission follows ImageDone without a sleep-tail
            // delay — and parks the thread instead of fabricating a poll
            // every 50 ms.
            let events = sub.poll_events_blocking(std::time::Duration::from_millis(50));
            if events.is_empty() && in_flight.is_empty() && pending.is_empty() {
                break;
            }

            for event in events {
                match event {
                    SubprocessEvent::Progress { item_id, stage, .. } => {
                        let orig_idx = item_id as usize;
                        if !quiet {
                            if let Some(Some(pb)) = spinners.get(orig_idx) {
                                pb.set_message(format!(
                                    "{} \u{2014} {}",
                                    inputs[orig_idx].display(),
                                    stage_label(stage),
                                ));
                            }
                        }
                    }
                    SubprocessEvent::ImageDone { item_id, result_path, width, height, .. } => {
                        let orig_idx = item_id as usize;
                        let batch_idx = in_flight.iter().position(|&i| valid_indices[i] as u64 == item_id);
                        if let Some(pos) = batch_idx {
                            in_flight.remove(pos);
                        }

                        let result = std::fs::read(&result_path)
                            .ok()
                            .and_then(|data| image::RgbaImage::from_raw(width, height, data))
                            .map(|mut rgba_image| {
                                // Image bg wins over color bg — matches GUI mutual exclusion.
                                if let Some(bg) = bg_image.as_deref() {
                                    prunr_core::apply_background_image(&mut rgba_image, bg, bg_image_fit);
                                } else if let Some(bg) = bg_color {
                                    prunr_core::apply_background_color(&mut rgba_image, bg);
                                }
                                prunr_core::ProcessResult {
                                    rgba_image,
                                    active_provider: String::new(),
                                }
                            })
                            .ok_or_else(|| CoreError::Model("Failed to read subprocess result".into()));
                        let _ = std::fs::remove_file(&result_path);

                        // Find the batch_results index for this item
                        if let Some(ridx) = valid_indices.iter().position(|&vi| vi == orig_idx) {
                            results[ridx] = Some(result);
                        }

                        // Admit next
                        if !sub.should_pause_admission() {
                            if let Some(next) = pending.pop_front() {
                                let next_id = valid_indices[next] as u64;
                                if sub.send_image_path(next_id, valid_paths[next].clone()).is_ok() {
                                    in_flight.push(next);
                                }
                            }
                        }
                    }
                    SubprocessEvent::ImageError { item_id, error } => {
                        let orig_idx = item_id as usize;
                        let batch_idx = in_flight.iter().position(|&i| valid_indices[i] as u64 == item_id);
                        if let Some(pos) = batch_idx {
                            in_flight.remove(pos);
                        }
                        if let Some(ridx) = valid_indices.iter().position(|&vi| vi == orig_idx) {
                            results[ridx] = Some(Err(CoreError::Model(error)));
                        }
                    }
                    SubprocessEvent::Finished => {
                        if in_flight.is_empty() && pending.is_empty() {
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if in_flight.is_empty() && pending.is_empty() { break; }
            // No sleep here — `poll_events_blocking` above already
            // waited up to 50 ms (or returned immediately when an event
            // arrived). Re-sleeping would just stack idle delay.
        }

        if crashed {
            // Re-queue in-flight items
            for idx in in_flight.into_iter().rev() {
                pending.push_front(idx);
            }
            let old_jobs = max_jobs;
            max_jobs = (max_jobs / 2).max(1);
            if old_jobs == 1 {
                // Give up on remaining
                for &idx in &pending {
                    results[idx] = Some(Err(CoreError::Model(
                        "Insufficient memory \u{2014} try a smaller model".into()
                    )));
                }
                break;
            }
            if !quiet {
                eprintln!("Memory pressure \u{2014} retrying with {} parallel jobs", max_jobs);
            }
            sub.kill();
            prunr_app::subprocess::protocol::cleanup_seg_pipeline_temps();
            continue;
        }

        // Normal completion of this subprocess run — matches the GUI's 5s
        // grace budget for model-cache teardown; Drop's 1s covers any slippage.
        let _ = sub.shutdown_with_timeout(std::time::Duration::from_secs(5));
        break;
    }

    // Convert Option<Result> to Result (None should not happen, but handle gracefully)
    results.into_iter()
        .map(|r| r.unwrap_or_else(|| Err(CoreError::Model("Not processed".into()))))
        .collect()
}

/// `prunr --doctor`: dump everything a support ticket would ask for.
/// Runs before `ort::init_from`, so a missing/broken runtime shows up
/// here instead of aborting the whole process.
pub fn run_doctor() {
    let p = prunr_app::hardware::profile();
    let diag = prunr_app::ort_runtime::diagnose();

    println!("Prunr Diagnostic Report");
    println!("{}", "=".repeat(23));
    println!();
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Build:   {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("OS:      {} {}", p.os, p.arch);
    println!();

    section("Hardware");
    println!("CPU vendor:  {}", p.cpu_vendor);
    println!("CPU brand:   {}", p.cpu_brand);
    println!("dGPU:        {}", p.dgpu.map_or("None".to_string(), |g| g.to_string()));
    println!("iGPU:        {}", p.igpu.map_or("None".to_string(), |g| g.to_string()));
    println!("Recommends OpenVINO: {}", p.recommends_openvino());
    println!("Recommends ROCm:     {}", p.recommends_rocm());
    println!();

    section("ONNX Runtime");
    println!("ORT_DYLIB_PATH: {}", diag.env_path.as_ref()
        .map_or("(unset)".to_string(), |p| p.display().to_string()));
    match (&diag.store_root, diag.store_entries.is_empty()) {
        (None, _) => println!("Runtime store:  (not present)"),
        (Some(root), true) => println!("Runtime store:  {} (empty)", root.display()),
        (Some(root), false) => {
            println!("Runtime store:  {}", root.display());
            for (name, has_dylib) in &diag.store_entries {
                let mark = if *has_dylib { "OK" } else { "MISSING" };
                println!("  - {name} [{mark}]");
            }
        }
    }
    if let Some((path, exists)) = &diag.bundled {
        let mark = if *exists { "OK" } else { "(absent)" };
        println!("Bundled:        {} {mark}", path.display());
    }
    match &diag.resolved {
        Some((p, src)) => println!("Active source:  {src} → {}", p.display()),
        None => println!("Active source:  NONE — `prunr` will refuse to start without --doctor"),
    }
    println!();

    section("Models");
    for desc in prunr_models::REGISTRY {
        let avail = if prunr_models::is_available(desc.id) { "installed" } else { "not installed" };
        println!("  {:<32} {:<10} {avail}",
            desc.display_name, desc.source.kind_label());
    }
    println!();

    section("Paths");
    println!("Data dir:     {}", path_or_unknown(prunr_models::data_dir()));
    println!("Models dir:   {}", path_or_unknown(prunr_models::on_demand_dir()));
    println!("Settings:     {}", path_or_unknown(
        dirs::config_dir().map(|d| d.join("prunr").join("settings.json"))));
    println!();

    section("Environment");
    for var in ["RUST_LOG", "ORT_DYLIB_PATH", "PRUNR_DEBUG_LOG"] {
        println!("  {var}: {}",
            std::env::var(var).unwrap_or_else(|_| "(unset)".to_string()));
    }
}

fn section(title: &str) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
}

fn path_or_unknown(p: Option<std::path::PathBuf>) -> String {
    p.map_or("(unresolvable on this platform)".to_string(),
        |p| p.display().to_string())
}
