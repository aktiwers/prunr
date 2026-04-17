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

    /// Model: silueta (fast, default) or u2net (quality).
    #[arg(short = 'm', long, default_value = "silueta")]
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

    /// Fill transparent background with a color (hex, e.g. "ffffff" for white).
    #[arg(long)]
    pub bg_color: Option<String>,

    /// Force CPU inference even when GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// Internal: run as subprocess worker for GUI batch processing.
    #[arg(long, hide = true)]
    pub worker: bool,

    /// Chain mode: process the previous result instead of the original.
    /// Only meaningful in multi-pass scripting workflows where the user
    /// runs prunr multiple times on the same file. For a single invocation
    /// this flag has no effect since there is no previous result.
    #[arg(long)]
    pub chain: bool,
}

/// Model selection
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliModel {
    /// Silueta (~4MB, fast) — default
    Silueta,
    /// U2Net (~170MB, higher quality)
    U2net,
    /// BiRefNet-lite (~214MB, best detail at 1024×1024)
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
        }
    }

    fn line_mode(&self) -> LineMode {
        if self.lines_after_bg {
            LineMode::AfterBgRemoval
        } else if self.lines {
            LineMode::LinesOnly
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

    // All processing uses subprocess isolation for OOM protection
    run_batch(args)
}

// ── Output path helpers ──────────────────────────────────────────────────────

/// Compute output path for an input image.
/// Batch mode or -o is a directory: write {stem}_nobg.png into that directory.
/// Single mode with -o file.png: use that path directly.
/// No -o: write alongside input as {input_dir}/{stem}_nobg.png.
fn output_path(input: &std::path::Path, output: &Option<PathBuf>, is_batch: bool) -> std::path::PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let nobg_name = format!("{stem}_nobg.png");
    match output {
        Some(out) if is_batch || out.is_dir() => out.join(nobg_name),
        Some(out) => out.clone(),
        None => input.with_file_name(nobg_name),
    }
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
    let mp = if !args.quiet { Some(MultiProgress::new()) } else { None };

    // Overall progress bar: "3/10 images"
    let overall = mp.as_ref().map(|m| {
        let pb = m.add(ProgressBar::new(args.inputs.len() as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "{bar:40.cyan/blue} {pos}/{len} images  {elapsed_precise}"
            ).unwrap(),
        );
        pb
    });

    // Per-image spinners (one per input, added to MultiProgress)
    let spinners: Vec<Option<ProgressBar>> = args.inputs.iter().map(|input| {
        mp.as_ref().map(|m| {
            let pb = m.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg}")
                    .unwrap()
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            pb.set_message(format!("{} (waiting...)", input.display()));
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
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

    let batch_results = run_batch_subprocess(
        &valid_paths, &valid_indices, model, args.jobs, mask, args.cpu,
        line_mode, args.line_strength, solid_line_color, bg_color,
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
    line_strength: f32,
    solid_line_color: Option<[u8; 3]>,
    bg_color: Option<[u8; 3]>,
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
            model, effective_jobs, mask, force_cpu, line_mode,
            line_strength, solid_line_color,
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

            let events = sub.poll_events();
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
                                // Apply bg_color on the parent side (non-destructive in subprocess)
                                if let Some(bg) = bg_color {
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
            std::thread::sleep(std::time::Duration::from_millis(50));
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
            prunr_app::subprocess::protocol::cleanup_ipc_temp();
            continue;
        }

        // Normal completion of this subprocess run
        let _ = sub.send_shutdown();
        break;
    }

    // Convert Option<Result> to Result (None should not happen, but handle gracefully)
    results.into_iter()
        .map(|r| r.unwrap_or_else(|| Err(CoreError::Model("Not processed".into()))))
        .collect()
}
