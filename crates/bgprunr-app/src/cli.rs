use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// BgPrunR — local background removal
#[derive(Parser, Debug)]
#[command(name = "bgprunr", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Remove background from one or more images
    Remove(RemoveArgs),
}

#[derive(clap::Args, Debug)]
pub struct RemoveArgs {
    /// Input image file(s). Pass multiple paths for batch mode.
    /// Shell globs are expanded by the shell before bgprunr sees them.
    pub inputs: Vec<PathBuf>,

    /// Output file path. Only valid for single-image mode.
    /// Mutually exclusive with --output-dir.
    #[arg(short = 'o', long, conflicts_with = "output_dir")]
    pub output: Option<PathBuf>,

    /// Output directory for batch mode.
    /// Files are named {stem}_nobg.png inside this directory.
    #[arg(long, conflicts_with = "output")]
    pub output_dir: Option<PathBuf>,

    /// Model to use for inference.
    #[arg(long, default_value = "silueta")]
    pub model: CliModel,

    /// Number of parallel inference jobs (batch mode only).
    /// Default 1 (sequential). Each job creates its own ORT session.
    #[arg(long, default_value_t = 1)]
    pub jobs: usize,

    /// How to handle images exceeding 8000px in either dimension.
    #[arg(long, default_value = "downscale")]
    pub large_image: LargeImagePolicy,

    /// Overwrite existing output files without prompting.
    #[arg(long)]
    pub force: bool,

    /// Suppress all progress output. Errors still go to stderr.
    #[arg(long)]
    pub quiet: bool,
}

/// Model selection
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliModel {
    /// Silueta (~4MB, fast) — default
    Silueta,
    /// U2Net (~170MB, higher quality)
    U2net,
}

impl From<CliModel> for bgprunr_core::ModelKind {
    fn from(m: CliModel) -> Self {
        match m {
            CliModel::Silueta => bgprunr_core::ModelKind::Silueta,
            CliModel::U2net => bgprunr_core::ModelKind::U2net,
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
use bgprunr_core::{
    OrtEngine, ModelKind, ProgressStage, CoreError,
    DOWNSCALE_TARGET,
    process_image, process_image_unchecked,
    batch_process,
    load_image_from_path, check_large_image, downscale_image, encode_rgba_png,
};

/// Entry point for the `remove` subcommand. Returns exit code (0/1/2).
pub fn run_remove(args: RemoveArgs) -> i32 {
    if args.inputs.is_empty() {
        eprintln!("error: no input files specified. Run `bgprunr remove --help`.");
        return 1;
    }

    // Ensure output directory exists if specified
    if let Some(ref dir) = args.output_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: cannot create output directory {}: {e}", dir.display());
            return 1;
        }
    }

    if args.inputs.len() == 1 {
        run_single(&args)
    } else {
        run_batch(&args)
    }
}

// ── Output path helpers ──────────────────────────────────────────────────────

/// Compute output path for a single input using {stem}_nobg.png convention.
/// With -o: use that path directly.
/// With --output-dir DIR: write into DIR/{stem}_nobg.png.
/// Without either: write alongside input as {input_dir}/{stem}_nobg.png.
fn output_path(input: &std::path::Path, args: &RemoveArgs) -> std::path::PathBuf {
    if let Some(ref out) = args.output {
        return out.clone();
    }
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let filename = format!("{stem}_nobg.png");
    if let Some(ref dir) = args.output_dir {
        dir.join(&filename)
    } else {
        input.with_file_name(&filename)
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
        ProgressStage::Decode      => "Decoding...",
        ProgressStage::Resize      => "Resizing...",
        ProgressStage::Normalize   => "Normalizing...",
        ProgressStage::Infer       => "Inferring...",
        ProgressStage::Postprocess => "Postprocessing...",
        ProgressStage::Alpha       => "Applying alpha...",
    }
}

// ── Single-image execution path ──────────────────────────────────────────────

fn run_single(args: &RemoveArgs) -> i32 {
    let input = &args.inputs[0];
    let out_path = output_path(input, args);

    if let Err(msg) = check_overwrite(&out_path, args.force) {
        eprintln!("error: {msg}");
        return 1;
    }

    // Load and apply large-image policy
    let img_bytes = match load_with_policy(input, args.large_image, args.quiet) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error loading '{}': {e}", input.display());
            return 1;
        }
    };

    // Stage spinner (only when not quiet)
    let spinner = if !args.quiet {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        Some(pb)
    } else {
        None
    };

    let model: ModelKind = args.model.into();
    let engine = match OrtEngine::new(model, 1) {
        Ok(e) => e,
        Err(e) => {
            if let Some(pb) = &spinner { pb.finish_and_clear(); }
            eprintln!("error: failed to load model: {e}");
            return 1;
        }
    };

    let spinner_ref = spinner.as_ref();
    let progress = spinner_ref.map(|pb| {
        move |stage: ProgressStage, _pct: f32| {
            pb.set_message(stage_label(stage));
        }
    });

    let result = if args.large_image == LargeImagePolicy::Process {
        process_image_unchecked(&img_bytes, &engine, progress, None)
    } else {
        process_image(&img_bytes, &engine, progress, None)
    };

    if let Some(pb) = &spinner { pb.finish_and_clear(); }

    match result {
        Ok(pr) => {
            if let Err(e) = std::fs::write(&out_path, &pr.rgba_bytes) {
                eprintln!("error writing '{}': {e}", out_path.display());
                return 1;
            }
            if !args.quiet {
                println!("done: {} -> {}", input.display(), out_path.display());
            }
            0
        }
        Err(e) => {
            eprintln!("error processing '{}': {e}", input.display());
            1
        }
    }
}

// ── Batch execution path ─────────────────────────────────────────────────────

fn run_batch(args: &RemoveArgs) -> i32 {
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

    // Load all image bytes upfront (fail-fast per image, not globally)
    let start_times: Vec<Instant> = args.inputs.iter().map(|_| Instant::now()).collect();
    let img_bytes_store: Vec<Result<Vec<u8>, CoreError>> = args.inputs.iter().map(|input| {
        load_with_policy(input, args.large_image, args.quiet)
    }).collect();

    // Update spinner for images that failed to load
    for (idx, res) in img_bytes_store.iter().enumerate() {
        if let Err(e) = res {
            if let Some(Some(pb)) = spinners.get(idx) {
                pb.finish_with_message(format!(
                    "X {} — load error: {e}", args.inputs[idx].display()
                ));
            }
        }
    }

    // Build refs for batch_process (only successfully loaded images)
    // Track which indices actually get processed
    let mut valid_indices: Vec<usize> = Vec::new();
    let mut valid_bytes: Vec<Vec<u8>> = Vec::new();
    for (idx, res) in img_bytes_store.iter().enumerate() {
        if res.is_ok() {
            valid_indices.push(idx);
            valid_bytes.push(res.as_ref().unwrap().clone());
        }
    }
    let valid_refs: Vec<&[u8]> = valid_bytes.iter().map(|b| b.as_slice()).collect();

    let model: ModelKind = args.model.into();

    // Progress callback: update per-image spinner with stage label
    // Use Arc so it's Send + Sync (required by batch_process)
    let spinners_arc = std::sync::Arc::new(spinners);
    let inputs_arc = std::sync::Arc::new(args.inputs.clone());
    let valid_indices_arc = std::sync::Arc::new(valid_indices.clone());
    let quiet = args.quiet;

    let progress_cb = {
        let spinners = spinners_arc.clone();
        let valid_indices = valid_indices_arc.clone();
        let inputs = inputs_arc.clone();
        move |batch_idx: usize, stage: ProgressStage, _pct: f32| {
            let original_idx = valid_indices[batch_idx];
            if !quiet {
                if let Some(Some(pb)) = spinners.get(original_idx) {
                    pb.set_message(format!(
                        "{} — {}",
                        inputs[original_idx].display(),
                        stage_label(stage)
                    ));
                }
            }
        }
    };

    let batch_results = batch_process(
        &valid_refs,
        model,
        args.jobs,
        Some(progress_cb),
    );

    // Compute output paths and write results
    let mut success_count = 0usize;
    let mut fail_count = 0usize;

    // Account for images that failed to load
    let load_fail_count = img_bytes_store.iter().filter(|r| r.is_err()).count();
    fail_count += load_fail_count;

    for (batch_idx, result) in batch_results.iter().enumerate() {
        let original_idx = valid_indices[batch_idx];
        let input = &args.inputs[original_idx];
        let elapsed = start_times[original_idx].elapsed();
        let out_path = output_path(input, args);

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
                match std::fs::write(&out_path, &pr.rgba_bytes) {
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


