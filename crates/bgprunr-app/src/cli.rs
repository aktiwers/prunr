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
