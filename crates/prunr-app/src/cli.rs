use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Prunr — local AI background removal.
///
/// No arguments launches the GUI. Pass image files directly to process them:
///   prunr photo.jpg              # removes background, saves photo.prunr.png
///   prunr *.jpg -o clean/        # batch to folder
///   prunr -m u2net portrait.jpg  # use quality model
#[derive(Parser, Debug)]
#[command(name = "prunr", version, about, long_about = None)]
pub struct Cli {
    /// Input image file(s).
    pub inputs: Vec<PathBuf>,

    /// Output path: a file for one image (a name without an extension is saved as .png), a directory for a batch.
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

    /// Edge contrast (0.0–1.0): 0 keeps the soft edge, 1 cuts it hard.
    #[arg(long, default_value_t = 0.0)]
    pub contrast: f32,

    /// Shift the cut-out's edge in pixels: positive grows the subject,
    /// negative shrinks it.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub edge_shift: f32,

    /// Refine mask edges using guided filter for better detail on hair, leaves, etc.
    #[arg(long)]
    pub refine_edges: bool,

    /// How far from the edge --refine-edges looks, in pixels.
    #[arg(long, default_value_t = 8)]
    pub edge_width: u32,

    /// Soften the cut-out's edge by this many pixels (fractions allowed).
    #[arg(long, default_value_t = 0.0)]
    pub feather: f32,

    /// Extract lines/edges instead of removing background (uses TEED model).
    #[arg(long)]
    pub lines: bool,

    /// Run line extraction after background removal (combine both).
    #[arg(long)]
    pub lines_after_bg: bool,

    /// How much of the detected edges become lines (0.0–1.0): lower draws
    /// bold outlines only, higher adds fine detail.
    #[arg(long = "line-detail", visible_alias = "line-strength", default_value_t = 0.5)]
    pub line_strength: f32,

    /// Line width in pixels (0–20).
    #[arg(long, default_value_t = 0)]
    pub line_width: u32,

    /// How the lines combine with the subject (with --lines-after-bg):
    /// lines-only, subject-filled, engraving or ghost.
    #[arg(long, default_value = "lines-only", value_parser = str::parse::<prunr_core::ComposeMode>)]
    pub line_composition: prunr_core::ComposeMode,

    /// Paint all lines a solid color (hex, e.g. "000000" for black, "ff0000" for red).
    #[arg(long)]
    pub line_color: Option<String>,

    /// Line scale: `fine` (thin, detailed lines), `bold` (big outlines
    /// only), `fused` (default, every scale combined). `balanced` is read
    /// as `fused`.
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
    /// dims, may crop), contain (fits inside, may letterbox), tile (repeat
    /// at native size).
    #[arg(long, default_value_t = prunr_core::BgImageFit::Cover)]
    pub bg_image_fit: prunr_core::BgImageFit,

    /// Restyle the subject's colors, with the effect's default settings:
    /// desaturate, invert, duotone, sepia, threshold, posterize, solarize,
    /// hue-shift, saturate, color-splash, pixelate, cross-process,
    /// halftone or gradient-map.
    #[arg(long, value_name = "NAME", value_parser = str::parse::<prunr_core::SubjectEffect>)]
    pub effect: Option<prunr_core::SubjectEffect>,

    /// Fill transparent background with a copy of the photo: `blur`
    /// (optionally `blur:N`, N = 1-64 px) or `bw` (black and white). Wins
    /// over `--bg-image` and `--bg-color`, as in the app.
    #[arg(long, value_name = "blur[:N]|lens[:N]|bw", value_parser = parse_bg_effect)]
    pub bg_effect: Option<prunr_core::BgEffect>,

    /// Sticker outline around the cut-out: width in px, optionally with a
    /// color, e.g. `8` (white) or `8:000000`.
    #[arg(long, value_name = "PX[:HEX]", value_parser = parse_outline)]
    pub outline: Option<(f32, [u8; 3])>,

    /// Shadow under the cut-out: `soft` (all around) or `drop` (below),
    /// with the app's default opacity, blur and distance.
    #[arg(long, value_name = "soft|drop", value_parser = str::parse::<prunr_core::ShadowKind>)]
    pub shadow: Option<prunr_core::ShadowKind>,

    /// Run every model on the CPU, even when a GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// Internal: run as subprocess worker for GUI batch processing.
    #[arg(long, hide = true)]
    pub worker: bool,

    /// Internal: the version of the app that started this worker.
    #[arg(long, hide = true)]
    pub worker_of: Option<String>,

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

    /// Object removal (Eraser) mode. Reads a binary mask from --mask
    /// (white = inpaint, black = keep) and runs LaMa to fill in the
    /// masked region. Output saved as {stem}_erased.png by default.
    #[arg(long)]
    pub inpaint: bool,

    /// Path to the inpaint mask image. Required with --inpaint.
    /// Any image format; pixels >128 in the first channel are treated
    /// as "inpaint here". Mask must match the input image's dimensions.
    /// `requires` enforces this at parse time so `--mask m.png foo.jpg`
    /// (no `--inpaint`) errors out instead of silently dropping the flag.
    #[arg(long, requires = "inpaint")]
    pub mask: Option<PathBuf>,

    /// How large the eraser fills the masked area: auto (full size for
    /// small areas, large ones shrink to fit one model tile), full, half
    /// or quarter. A large area fills more convincingly at a smaller
    /// scale, a little softer.
    #[arg(long, requires = "inpaint", default_value = "auto")]
    pub fill_scale: prunr_core::inpaint::FillScale,

    /// Upscale mode: enlarge the image with this model instead of
    /// removing its background. Saves {stem}_x{scale}.png.
    #[arg(long, value_name = "MODEL")]
    pub upscale: Option<CliUpscaler>,

    /// How much larger the upscale makes the image.
    #[arg(long, requires = "upscale", default_value_t = 4, value_parser = clap::value_parser!(u8).range(2..=4))]
    pub scale: u8,

    /// Denoise before the upscale model, 0 (off) to 1.
    #[arg(long, requires = "upscale", default_value_t = 0.0)]
    pub denoise: f32,

    /// Sharpen after the upscale, -1 (soften) to 1; 0 is off.
    #[arg(long, requires = "upscale", default_value_t = prunr_core::DEFAULT_SHARPEN, allow_negative_numbers = true)]
    pub sharpen: f32,

    /// AI blend: 1 keeps the model's output, 0 the plain enlargement.
    #[arg(long, requires = "upscale", default_value_t = 1.0)]
    pub ai_blend: f32,

    /// Match the upscale's colours to the original.
    #[arg(long, requires = "upscale")]
    pub color_match: bool,

    /// Restore faces with this model after the upscale (downloaded in the
    /// app's Model Store).
    #[arg(long, requires = "upscale", value_name = "MODEL")]
    pub restore_faces: Option<CliFaceModel>,

    /// Restore faces strength in percent: up to 50 adds detail and keeps
    /// the photo's shading, above it the redrawn shading too.
    #[arg(long, requires = "restore_faces", default_value_t = 50, value_parser = clap::value_parser!(u8).range(0..=100))]
    pub face_strength: u8,

    /// Colorize mode: color a black-and-white photo with DDColor
    /// (downloaded in the app's Model Store). Saves {stem}_color.png.
    #[arg(long)]
    pub colorize: bool,

    /// Color strength: 0 is grey, 1 the predicted color, up to 1.5.
    #[arg(long, requires = "colorize", default_value_t = 1.0)]
    pub color_strength: f32,

    /// Lines fade with distance, 0-100 (needs the depth model, from the
    /// app's Model Store).
    #[arg(long, value_name = "PERCENT", default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=100))]
    pub line_fade: u8,

    /// Also write each input's depth map as {stem}-depth.png (16-bit grey,
    /// white = near) beside the result. Needs the depth model.
    #[arg(long)]
    pub depth_map: bool,

    /// Write only the depth maps, nothing else.
    #[arg(long, requires = "depth_map")]
    pub depth_only: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Time every installed model on the CPU and on the GPU path the app
    /// takes, check the two outputs agree, and write a report to send back.
    Bench(BenchArgs),
}

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Models to run, comma-separated (e.g. silueta,real_esrgan_x2plus,migan);
    /// every installed one when left out.
    #[arg(long, value_delimiter = ',', value_name = "NAMES")]
    pub models: Vec<String>,

    /// Timed runs on each side after the first, untimed one.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(0..=100))]
    pub runs: u32,

    /// Run on this picture instead of the built-in one.
    #[arg(long, value_name = "PATH")]
    pub image: Option<PathBuf>,

    /// Also run the multi-GB models: Stable Diffusion 1.5 (both), RORem
    /// and 4xNomos8kSCHAT-L. Minutes each on a CPU.
    #[arg(long)]
    pub heavy: bool,

    /// Where to write the Markdown report; a JSON copy goes beside it.
    #[arg(long, value_name = "PATH", default_value = "prunr-bench.md")]
    pub out: PathBuf,
}

/// Upscale model selection.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliUpscaler {
    /// Real-ESRGAN x4plus: the all-round photo upscaler.
    Realesrgan,
    /// 4xNomos8kSCHAT-L: the most detailed, the slowest.
    Nomos,
    /// 4x NMKD Siax-CX: clean or mildly compressed photos.
    Siax,
    /// 4x NMKD Superscale: noisy or compressed photos.
    Superscale,
    /// 4xNomos2 RealPLKSR: fast, natural skin and hair texture.
    Nomos2,
}

impl CliUpscaler {
    fn model_id(self) -> prunr_models::ModelId {
        use prunr_models::ModelId;
        match self {
            Self::Realesrgan => ModelId::RealEsrganX4Plus,
            Self::Nomos => ModelId::Nomos8kSchatL,
            Self::Siax => ModelId::FourXNmkdSiaxCx,
            Self::Superscale => ModelId::FourXNmkdSuperscale,
            Self::Nomos2 => ModelId::FourXNomos2RealPlksr,
        }
    }
}

/// Restore faces model selection.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliFaceModel {
    /// GFPGAN v1.4: the crispest.
    Gfpgan,
    /// RestoreFormer++: closest to the person's own features.
    Restoreformer,
}

impl CliFaceModel {
    fn model_id(self) -> prunr_models::ModelId {
        match self {
            Self::Gfpgan => prunr_models::ModelId::FaceGfpgan,
            Self::Restoreformer => prunr_models::ModelId::FaceRestoreFormerPp,
        }
    }
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
    MaskSettings, ModelKind, CoreError,
    DOWNSCALE_TARGET,
    load_image_from_path, downscale_image, encode_rgba_png,
};

use crate::gui::settings::LineMode;
use prunr_core::inpaint_blend::{fill_region, finalize_inpaint, DEFAULT_EDGE_BLEND_PX, DEFAULT_GROW_PX};

/// What goes behind each result, from `--bg-effect`, `--bg-image` and
/// `--bg-color`.
enum CliBackdrop<'a> {
    Transparent,
    Color([u8; 3]),
    Image(std::sync::Arc<image::DynamicImage>, prunr_core::BgImageFit),
    /// An effect, and where a lens blur gets its depth.
    Effect(prunr_core::BgEffect, &'a CliDepth),
}

impl CliBackdrop<'_> {
    /// Put the backdrop under `result`; an effect is made from the photo
    /// at `photo_path`.
    fn apply(&self, result: &mut image::RgbaImage, photo_path: &std::path::Path) {
        match self {
            CliBackdrop::Transparent => {}
            CliBackdrop::Color(c) => prunr_core::apply_background_color(result, *c),
            CliBackdrop::Image(img, fit) => prunr_core::apply_background_image(result, img, *fit),
            CliBackdrop::Effect(effect, depth) => match prunr_core::load_image_from_path(photo_path) {
                Ok(photo) => {
                    let photo = photo.into_rgba8();
                    let depth = effect.needs_depth()
                        .then(|| depth.for_backdrop(photo_path, &photo)
                            .map_err(|err| tracing::warn!(path = %photo_path.display(), %err, "bg effect: depth"))
                            .ok())
                        .flatten();
                    prunr_core::apply_bg_effect(result, &photo, *effect, depth.as_deref());
                }
                Err(err) => tracing::warn!(path = %photo_path.display(), %err, "bg effect: photo unreadable"),
            },
        }
    }
}

/// `blur`, `blur:N` (1-64 px), `lens`, `lens:N` (0-100 %) or `bw`.
fn parse_bg_effect(s: &str) -> Result<prunr_core::BgEffect, String> {
    use prunr_core::BgEffect;
    match s.split_once(':') {
        None if s == "blur" => Ok(BgEffect::BlurredSource { radius: 12 }),
        None if s == "lens" => Ok(BgEffect::LensBlur { amount_pct: 50 }),
        None if s == "bw" => Ok(BgEffect::DesaturatedSource),
        Some(("blur", n)) => match n.parse::<u32>() {
            Ok(radius @ 1..=64) => Ok(BgEffect::BlurredSource { radius }),
            _ => Err(format!("'{n}' is not a blur radius from 1 to 64")),
        },
        Some(("lens", n)) => match n.parse::<u8>() {
            Ok(amount_pct @ 0..=100) => Ok(BgEffect::LensBlur { amount_pct }),
            _ => Err(format!("'{n}' is not a lens blur amount from 0 to 100")),
        },
        _ => Err(format!("'{s}' is not one of blur, blur:N, lens, lens:N, bw")),
    }
}

/// Depth in the CLI: one engine for the run, and each input measured
/// once when a lens backdrop and `--depth-map` both want it.
#[derive(Default)]
struct CliDepth {
    measurer: prunr_core::depth::Measurer,
    /// With `--depth-map`: the maps backdrops measured and their photo's
    /// size (~0.7 MB each), until written.
    kept: Option<std::cell::RefCell<std::collections::HashMap<PathBuf, KeptDepth>>>,
}

type KeptDepth = (std::sync::Arc<prunr_core::depth::DepthMap>, (u32, u32));

impl CliDepth {
    fn new(args: &Cli) -> Self {
        Self { kept: args.depth_map.then(Default::default), ..Self::default() }
    }

    /// The map of `photo`, read from `path`, for its backdrop.
    fn for_backdrop(&self, path: &std::path::Path, photo: &image::RgbaImage) -> Result<std::sync::Arc<prunr_core::depth::DepthMap>, CoreError> {
        let map = std::sync::Arc::new(self.measurer.measure(photo)?);
        if let Some(kept) = &self.kept {
            kept.borrow_mut().insert(path.to_path_buf(), (std::sync::Arc::clone(&map), photo.dimensions()));
        }
        Ok(map)
    }

    /// `input`'s map and its photo's size: the one its backdrop kept, else
    /// measured now.
    fn for_file(&self, input: &std::path::Path) -> Result<KeptDepth, String> {
        if let Some(kept) = self.kept.as_ref().and_then(|kept| kept.borrow_mut().remove(input)) {
            return Ok(kept);
        }
        let photo = prunr_core::load_image_from_path(input).map_err(|e| could_not(input, "read", e))?;
        let map = self.measurer.measure_decoded(&photo).map_err(|e| format!("depth failed: {e}"))?;
        Ok((std::sync::Arc::new(map), (photo.width(), photo.height())))
    }
}

/// Where an input's depth map goes, beside its result.
fn depth_map_path(args: &Cli, input: &std::path::Path) -> PathBuf {
    let mut name = input.file_stem().unwrap_or_default().to_os_string();
    name.push(prunr_core::depth::MAP_FILE_SUFFIX);
    args.result_path(input).with_file_name(name)
}

/// Write each input's depth map (`--depth-map`). Returns how many failed.
fn write_depth_maps(args: &Cli, depth: &CliDepth) -> usize {
    let mut failed = 0;
    for input in &args.inputs {
        let out = depth_map_path(args, input);
        let written = check_overwrite(&out, args.force)
            .and_then(|()| depth.for_file(input))
            .and_then(|(map, (w, h))| prunr_core::depth::encode_depth_png(&map, w, h).map_err(|e| format!("encode failed: {e}")))
            .and_then(|bytes| prunr_app::fs_util::write_bytes_replacing(&out, &bytes).map_err(|e| format!("failed to save {}: {e}", out.display())));
        match written {
            Ok(()) if !args.quiet => eprintln!("Saved {}", out.display()),
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {e}");
                failed += 1;
            }
        }
    }
    failed
}

/// The depth model is needed (a depth map, fading lines, a lens blur)
/// but not installed: the line to print.
fn missing_depth_model(args: &Cli) -> Option<String> {
    let id = prunr_core::depth::MODEL.into();
    (wants_depth(args) && !prunr_models::is_available(id)).then(|| prunr_models::not_installed_error(id))
}

/// Whether the run reads a depth map: a depth-map file, faded lines (only
/// when lines are drawn), or a lens-blur backdrop.
fn wants_depth(args: &Cli) -> bool {
    let draws_lines = args.lines || args.lines_after_bg;
    args.depth_map || (draws_lines && args.line_fade > 0) || args.bg_effect.is_some_and(prunr_core::BgEffect::needs_depth)
}

fn parse_outline(s: &str) -> Result<(f32, [u8; 3]), String> {
    let (width, color) = s.split_once(':').unwrap_or((s, "ffffff"));
    let width: f32 = width.parse().map_err(|_| format!("'{width}' is not an outline width in px"))?;
    let max = prunr_core::Decoration::MAX_OUTLINE_PX;
    if !(0.0..=max).contains(&width) {
        return Err(format!("outline width {width} is outside 0-{max} px"));
    }
    let color = parse_hex_color(color).ok_or_else(|| format!("'{color}' is not a hex color like ffffff"))?;
    Ok((width, color))
}

/// A colour flag's value: `None` when the flag is absent, `Err` (after
/// telling the user, with `example`) when it is not 6-digit hex.
fn flag_color(flag: &str, value: Option<&str>, example: &str) -> Result<Option<[u8; 3]>, ()> {
    let Some(s) = value else { return Ok(None) };
    parse_hex_color(s).map(Some).ok_or_else(|| {
        eprintln!("error: {flag}: '{s}' is not a 6-digit hex color (e.g. {example})");
    })
}

fn parse_hex_color(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 { return None; }
    prunr_core::color::parse_hex_rgba(s).map(|[r, g, b, _]| [r, g, b])
}

impl Cli {
    /// Whether this run writes to the terminal it was started from. A
    /// Windows release build has no console of its own, so it borrows
    /// that one, or its output goes nowhere.
    #[cfg_attr(not(windows), allow(dead_code))] // only Windows borrows a console; the tests check the rule everywhere
    pub fn writes_to_terminal(&self) -> bool {
        self.debug || !self.inputs.is_empty() || self.doctor || self.clear_ep_cache || self.command.is_some()
    }

    fn mask_settings(&self) -> MaskSettings {
        MaskSettings {
            gamma: self.gamma,
            contrast: self.contrast,
            edge_shift: MaskSettings::edge_shift_for_grow(self.edge_shift),
            refine_edges: self.refine_edges,
            guided_radius: self.edge_width.clamp(1, 64),
            feather: self.feather.max(0.0),
            effect: self.effect.unwrap_or_default(),
            decoration: self.decoration(),
            ..Default::default()
        }
    }

    /// Where an input's result is written: `-o` names the file for one
    /// input and the folder for several.
    fn result_path(&self, input: &std::path::Path) -> PathBuf {
        output_path(input, &self.output, self.inputs.len() > 1)
    }

    fn decoration(&self) -> prunr_core::Decoration {
        let mut d = prunr_core::Decoration::default();
        if let Some((width, color)) = self.outline {
            d.outline_width = width;
            d.outline_color = color;
        }
        if let Some(kind) = self.shadow {
            d.shadow = kind;
        }
        d
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

    // `create_dir_all` can return Ok on an existing *file*; `is_dir()` catches it.
    if let Some(ref out) = args.output {
        if args.inputs.len() > 1 {
            if let Err(e) = std::fs::create_dir_all(out) {
                eprintln!("error: cannot create output directory {}: {e}", out.display());
                return 1;
            }
            if !out.is_dir() {
                eprintln!("error: --output must be a directory for batch input (got a file path: {})", out.display());
                return 1;
            }
        }
    }

    // Eraser and upscale modes ignore the background-removal flags; say so
    // up front rather than quietly throw them away.
    let modes = selected_modes(args);
    if modes.len() > 1 {
        eprintln!("error: {} cannot be combined.", modes.join(" and "));
        return 1;
    }
    let mode = modes.first().copied();
    if let Some(mode) = mode {
        let conflicts = background_removal_flags(args);
        if !conflicts.is_empty() {
            eprintln!(
                "error: {mode} cannot be combined with {} \u{2014} those flags belong to background-removal mode.",
                conflicts.join(", "),
            );
            return 1;
        }
    }
    if let Some(err) = missing_depth_model(args) {
        eprintln!("error: {err}");
        return 1;
    }
    let depth = CliDepth::new(args);
    let code = if args.depth_only {
        0
    } else if args.inpaint {
        run_inpaint(args)
    } else if let Some(upscaler) = args.upscale {
        run_upscale(args, upscaler)
    } else if args.colorize {
        run_colorize(args)
    } else {
        // All processing uses subprocess isolation for OOM protection
        run_batch(args, &depth)
    };
    if args.depth_map && write_depth_maps(args, &depth) > 0 {
        return 1;
    }
    code
}

/// One upscale as the app runs it: denoise, the model, Refine, faces.
struct UpscaleJob<'a> {
    engine: &'a prunr_core::OrtEngine,
    model: prunr_models::ModelId,
    scale: prunr_core::OutputScale,
    refine: prunr_core::upscale::Refine,
    denoise: f32,
    /// Loaded once for the batch.
    faces: Option<(prunr_core::face::FaceRestorer, prunr_core::face::FaceBlend)>,
}

impl UpscaleJob<'_> {
    fn run(&mut self, img: image::RgbaImage) -> Result<image::RgbaImage, String> {
        let img = std::sync::Arc::new(img);
        let model_input = if self.denoise > 0.0 {
            std::sync::Arc::new(prunr_core::denoise::apply_denoise(&img, self.denoise.min(1.0)))
        } else {
            img
        };
        let needed = prunr_core::upscale::needed_bytes(self.model, model_input.width(), model_input.height(), self.scale.factor());
        let free = prunr_app::hardware::available_ram_bytes_now();
        if needed > free {
            use crate::gui::views::format_byte_size;
            return Err(format!("too large to upscale: needs about {} of free memory, {} is free", format_byte_size(needed), format_byte_size(free)));
        }
        let progress = prunr_core::Progress::none();
        let raw = prunr_core::upscale::upscale_rgba_with_engine(&model_input, self.engine, self.model, self.scale.factor(), &progress, None)
            .map_err(|e| format!("upscale failed: {e}"))?;
        let (shown, _) = prunr_core::upscale::refine(&std::sync::Arc::new(raw), &model_input, None, &self.refine);
        let mut out = std::sync::Arc::unwrap_or_clone(shown);
        if let Some((restorer, blend)) = self.faces.as_mut() {
            let faces = restorer.find_and_restore(&out, *blend, &progress)
                .map_err(|e| format!("restore faces failed: {e}"))?;
            prunr_core::face::compose_faces_into(&mut out, &faces, *blend);
        }
        Ok(out)
    }
}

/// The mode flags set on the command line; at most one may be.
fn selected_modes(args: &Cli) -> Vec<&'static str> {
    [(args.inpaint, "--inpaint"), (args.upscale.is_some(), "--upscale"), (args.colorize, "--colorize")]
        .into_iter()
        .filter_map(|(set, flag)| set.then_some(flag))
        .collect()
}

/// The background-removal flags set on the command line.
fn background_removal_flags(args: &Cli) -> Vec<&'static str> {
    [
        (args.lines, "--lines"),
        (args.lines_after_bg, "--lines-after-bg"),
        (args.line_color.is_some(), "--line-color"),
        (args.bg_color.is_some(), "--bg-color"),
        (args.bg_image.is_some(), "--bg-image"),
        (args.bg_effect.is_some(), "--bg-effect"),
        (args.effect.is_some(), "--effect"),
        (args.contrast > 0.0, "--contrast"),
        (args.line_fade > 0, "--line-fade"),
    ]
    .into_iter()
    .filter_map(|(set, flag)| set.then_some(flag))
    .collect()
}

/// Upscale mode: enlarge each input with `upscaler`, Refine after it
/// (sharpen, AI blend, colour match) and, when asked, restore faces, as
/// the app does. The model loads once for the whole batch.
fn run_upscale(args: &Cli, upscaler: CliUpscaler) -> i32 {
    use prunr_core::OutputScale;
    let scale = match args.scale {
        2 => OutputScale::X2,
        3 => OutputScale::X3,
        _ => OutputScale::X4,
    };
    let model = prunr_core::upscale::model_for_scale(upscaler.model_id(), scale, prunr_models::is_available(prunr_models::ModelId::RealEsrganX2Plus));
    for needed in std::iter::once(model).chain(args.restore_faces.map(CliFaceModel::model_id)) {
        if !prunr_models::is_available(needed) {
            eprintln!("error: {}", prunr_models::not_installed_error(needed));
            return 1;
        }
    }
    let Some(engine) = load_engine(model, args.cpu, "upscale") else { return 1 };
    let refine = prunr_core::upscale::Refine {
        sharpen: args.sharpen.clamp(-1.0, 1.0),
        sharpen_sigma: scale.sharpen_sigma(),
        ai_blend: args.ai_blend.clamp(0.0, 1.0),
        color_match: args.color_match,
    };
    let blend = prunr_core::face::FaceBlend { strength: f32::from(args.face_strength) / 100.0, ..Default::default() };
    let faces = match args.restore_faces.map(|m| prunr_core::face::FaceRestorer::new(m.model_id())).transpose() {
        Ok(faces) => faces.map(|restorer| (restorer, blend)),
        Err(e) => {
            eprintln!("error: cannot load the face model: {e}");
            return 1;
        }
    };
    let mut job = UpscaleJob { engine: &engine, model, scale, refine, denoise: args.denoise, faces };
    run_each_input(args, &format!("_x{}.png", scale.factor()), "Upscaling", |img| job.run(img))
}

/// Colorize mode: colour each input with DDColor at `--color-strength`.
/// The model loads once for the whole batch.
fn run_colorize(args: &Cli) -> i32 {
    let Some(engine) = load_engine(crate::gui::settings::COLORIZER.into(), args.cpu, "colorize") else { return 1 };
    let strength = args.color_strength.clamp(0.0, prunr_core::colorize::STRENGTH_MAX);
    run_each_input(args, "_color.png", "Colorizing", |img| {
        let plane = prunr_core::colorize::colorize_plane(&engine, &img, &prunr_core::Progress::none())
            .map_err(|e| format!("colorize failed: {e}"))?;
        Ok(prunr_core::colorize::colorized(&img, &plane, strength))
    })
}

/// `model`'s session as the app builds it: every core, the model's graph
/// optimisation level, the CPU alone with `--cpu`. A model not installed
/// or not loading is reported here.
fn load_engine(model: prunr_models::ModelId, cpu: bool, what: &str) -> Option<prunr_core::OrtEngine> {
    if !prunr_models::is_available(model) {
        eprintln!("error: {}", prunr_models::not_installed_error(model));
        return None;
    }
    prunr_core::OrtEngine::for_model(model, prunr_core::batch::ort_intra_threads(1), cpu)
        .map_err(|e| eprintln!("error: cannot load the {what} model: {e}"))
        .ok()
}

/// Read each input, `run` it, and save it beside the input with `suffix`
/// (or into `--output`), reporting each as `doing` it. Returns the exit
/// code: 1 when any input failed.
fn run_each_input(
    args: &Cli,
    suffix: &str,
    doing: &str,
    mut run: impl FnMut(image::RgbaImage) -> Result<image::RgbaImage, String>,
) -> i32 {
    let is_batch = args.inputs.len() > 1;
    let mut failed = 0;
    for input in &args.inputs {
        let out_path = output_path_with_suffix(input, &args.output, is_batch, suffix);
        if let Err(e) = check_overwrite(&out_path, args.force) {
            eprintln!("error: {e}");
            failed += 1;
            continue;
        }
        if !args.quiet {
            eprintln!("{doing} {}...", input.display());
        }
        let result = load_image_from_path(input)
            .map_err(|e| could_not(input, "read", e))
            .and_then(|img| run(img.into_rgba8()))
            .and_then(|out| prunr_app::fs_util::save_image_replacing(&out_path, &out).map_err(|e| could_not(&out_path, "save", e)));
        match result {
            Ok(()) if !args.quiet => eprintln!("Saved {}", out_path.display()),
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {e}");
                failed += 1;
            }
        }
    }
    i32::from(failed > 0)
}

/// Eraser mode: load image + mask, run LaMa, save inpainted result.
/// Single-file only.
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

    let img = match load_image_from_path(input) {
        Ok(i) => i.into_rgba8(),
        Err(e) => {
            eprintln!("error: {}", could_not(input, "read", e));
            return 1;
        }
    };
    let mask = match load_image_from_path(mask_path) {
        Ok(i) => i.into_luma8(),
        Err(e) => {
            eprintln!("error: {}", could_not(mask_path, "read the mask", e));
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
        eprintln!("Erasing {} (mask: {})...", input.display(), mask_path.display());
    }
    let mask = fill_region(&mask, DEFAULT_GROW_PX, DEFAULT_EDGE_BLEND_PX);
    let raw = match prunr_core::inpaint::process_lama(&img, &mask, prunr_models::ModelId::LaMaFp32, args.cpu, args.fill_scale, &prunr_core::Progress::none()) {
        Ok(r) => r.rgba,
        Err(e) => {
            eprintln!("error: {}", could_not(input, "erase", e));
            return 1;
        }
    };
    let result = finalize_inpaint(raw, &img, &mask, DEFAULT_EDGE_BLEND_PX, 0.0, prunr_core::inpaint_blend::EdgeColour::default());
    if let Err(e) = prunr_app::fs_util::save_image_replacing(&out_path, &result) {
        eprintln!("error: failed to save {}: {e}", out_path.display());
        return 1;
    }
    if !args.quiet {
        eprintln!("Saved {}", out_path.display());
    }
    0
}

/// `output_path` with a caller-chosen suffix in place of `.prunr.png`.
fn output_path_with_suffix(
    input: &std::path::Path,
    output: &Option<PathBuf>,
    is_batch: bool,
    suffix: &str,
) -> std::path::PathBuf {
    let mut name = input.file_stem().unwrap_or_default().to_os_string();
    name.push(suffix);
    match output {
        Some(out) if is_batch || out.is_dir() => out.join(name),
        // A name without an extension is a file, saved as PNG: the
        // extension names the format, and every mode writes one.
        Some(out) if out.extension().is_none() => out.with_extension("png"),
        Some(out) => out.clone(),
        None => input.with_file_name(name),
    }
}

/// Compute output path for an input image.
/// Batch mode or -o is a directory: write {stem}.prunr.png into that directory.
/// Single mode with -o file.png: use that path directly.
/// No -o: write alongside input as {input_dir}/{stem}.prunr.png.
fn output_path(input: &std::path::Path, output: &Option<PathBuf>, is_batch: bool) -> std::path::PathBuf {
    output_path_with_suffix(input, output, is_batch, &prunr_core::ExportFormat::Png.result_suffix())
}

/// The CLI's one error shape: "<path>: could not <what>: <reason>".
fn could_not(path: &std::path::Path, what: &str, reason: impl std::fmt::Display) -> String {
    format!("{}: could not {what}: {reason}", path.display())
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

fn downscale_to_png(path: &std::path::Path, quiet: bool) -> Result<Vec<u8>, CoreError> {
    let img = load_image_from_path(path)?;
    if !quiet {
        eprintln!(
            "warning: '{}' exceeds {}px. Downscaling to {}px (use --large-image=process to skip).",
            path.display(), prunr_core::LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET
        );
    }
    encode_rgba_png(&downscale_image(img, DOWNSCALE_TARGET).into_rgba8())
}

fn run_batch(args: &Cli, depth: &CliDepth) -> i32 {
    // Validate and decode --bg-image before setting up progress bars.
    // Placing the decode inside the MultiProgress draw region causes
    // eprintln! on decode failure to get overwritten by the next redraw.
    let bg_image: Option<std::sync::Arc<image::DynamicImage>> = match args.bg_image.as_deref() {
        Some(p) => match prunr_core::load_image_from_path(p) {
            Ok(img) => Some(std::sync::Arc::new(img)),
            Err(e) => {
                eprintln!("error: --bg-image: {e}");
                return 1;
            }
        },
        None => None,
    };

    // 4 Hz redraw cap: at the default 20 Hz, redrawing every bar cost ~13% of
    // self-time on a 6-image batch. Spinners only on a terminal: elsewhere
    // indicatif draws nothing, so each result prints a line instead.
    let mp = if !args.quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        Some(MultiProgress::with_draw_target(
            indicatif::ProgressDrawTarget::stderr_with_hz(4),
        ))
    } else {
        None
    };

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

    let spinners: Vec<Option<ProgressBar>> = args.inputs.iter().map(|input| {
        mp.as_ref().map(|m| {
            let pb = m.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg}")
                    .expect("static indicatif template compiles")
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            pb.set_message(format!("{} (waiting...)", input.display()));
            // Keeps the spinner moving through the multi-second silence of
            // `session.run()`; the 4 Hz cap above bounds total redraws.
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
        match prunr_core::image_dimensions_from_path(input) {
            Some((w, h)) if args.large_image == LargeImagePolicy::Downscale
                && prunr_core::exceeds_large_image_limit(w, h) =>
            {
                match downscale_to_png(input, args.quiet) {
                    Ok(bytes) => {
                        let temp = prunr_app::subprocess::protocol::IpcKind::CliDs
                            .path_for(prunr_app::subprocess::protocol::ipc_temp_dir(), idx as u64);
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
    // "Absent" and "malformed" differ so scripts can detect a typo'd colour.
    let Ok(solid_line_color) = flag_color("--line-color", args.line_color.as_deref(), "ff00aa") else { return 1 };
    let Ok(bg_color) = flag_color("--bg-color", args.bg_color.as_deref(), "ffffff") else { return 1 };
    // The app's rule: an effect wins over an image, an image over a colour.
    let backdrop = match (args.bg_effect, bg_image, bg_color) {
        (Some(effect), _, _) => CliBackdrop::Effect(effect, depth),
        (None, Some(image), _) => CliBackdrop::Image(image, args.bg_image_fit),
        (None, None, Some(color)) => CliBackdrop::Color(color),
        (None, None, None) => CliBackdrop::Transparent,
    };

    let edge = prunr_core::EdgeSettings {
        line_strength: args.line_strength,
        solid_line_color,
        edge_thickness: args.line_width.min(20),
        edge_scale: args.line_scale,
        compose_mode: args.line_composition,
        line_style: prunr_core::LineStyle::default(),
        input_transform: prunr_core::InputTransform::default(),
    };
    let lines = prunr_app::subprocess::manager::WorkerLines {
        mode: line_mode, edge, fade_pct: args.line_fade,
        scales: prunr_app::subprocess::protocol::InferredScales::Drawn,
    };
    let batch_results = run_batch_subprocess(
        &valid_paths, &valid_indices, model, args.jobs, mask, args.cpu,
        lines, &backdrop,
        &spinners_arc, &inputs_arc, quiet,
    );

    let mut success_count = 0usize;
    let mut fail_count = 0usize;
    fail_count += load_fail_count;

    for (batch_idx, result) in batch_results.iter().enumerate() {
        let original_idx = valid_indices[batch_idx];
        let input = &args.inputs[original_idx];
        let elapsed = start_times[original_idx].elapsed();
        let out_path = args.result_path(input);

        let spinner_opt = spinners_arc.get(original_idx).and_then(|o| o.as_ref());

        match result {
            Ok(pr) => {
                if let Err(msg) = check_overwrite(&out_path, args.force) {
                    if let Some(pb) = spinner_opt {
                        pb.finish_with_message(format!("X {} — {msg}", input.display()));
                    } else if !args.quiet {
                        eprintln!("skipped '{}': {msg}", input.display());
                    }
                    fail_count += 1;
                    continue;
                }
                let saved = prunr_app::fs_util::write_replacing(&out_path, |w| {
                    prunr_core::ExportFormat::Png.encode_into(&pr.rgba_image, w).map_err(std::io::Error::other)
                });
                match saved {
                    Ok(_) => {
                        success_count += 1;
                        if let Some(pb) = spinner_opt {
                            pb.finish_with_message(format!(
                                "✓ {} ({:.1}s)",
                                input.display(),
                                elapsed.as_secs_f32()
                            ));
                        } else if !args.quiet {
                            // stdout: per-image success lines and the final summary
                            // are the program's machine-parseable output. Errors and
                            // skips go to stderr. Spinner path uses indicatif (stderr).
                            println!("✓ {} ({:.1}s)", input.display(), elapsed.as_secs_f32());
                        }
                        if let Some(ref opb) = overall { opb.inc(1); }
                    }
                    Err(e) => {
                        fail_count += 1;
                        if let Some(pb) = spinner_opt {
                            pb.finish_with_message(format!("X {} — write error: {e}", input.display()));
                        } else {
                            eprintln!("error: {}", could_not(&out_path, "write", e));
                        }
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                if let Some(pb) = spinner_opt {
                    pb.finish_with_message(format!("X {} — {e}", input.display()));
                } else {
                    eprintln!("error: {}", could_not(input, "process", e));
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
// One call site: the recipe, runtime knobs and UI surface arrive together,
// and a struct would be a builder no other site reuses.
#[allow(clippy::too_many_arguments)]
fn run_batch_subprocess(
    valid_paths: &[std::path::PathBuf],
    valid_indices: &[usize],
    model: ModelKind,
    initial_jobs: usize,
    mask: MaskSettings,
    cpu_only: bool,
    lines: prunr_app::subprocess::manager::WorkerLines,
    backdrop: &CliBackdrop,
    spinners: &[Option<ProgressBar>],
    inputs: &[std::path::PathBuf],
    quiet: bool,
) -> Vec<Result<prunr_core::ProcessResult, CoreError>> {
    use prunr_app::subprocess::batch_rules::{is_stalled, on_crash, room_to_send, OnCrash, HANG_TIMEOUT};
    use prunr_app::subprocess::manager::{Crash, SubprocessManager};
    use prunr_app::subprocess::protocol::SubprocessEvent;

    let mut results: Vec<Option<Result<prunr_core::ProcessResult, CoreError>>> =
        (0..valid_paths.len()).map(|_| None).collect();
    let mut pending: std::collections::VecDeque<usize> = (0..valid_paths.len()).collect();
    let mut max_jobs = initial_jobs;

    // Inverse index: orig_idx (item_id cast to usize) → batch_results slot,
    // so each event finds its slot without scanning valid_indices.
    let orig_to_batch: std::collections::HashMap<usize, usize> = valid_indices
        .iter()
        .enumerate()
        .map(|(batch, &orig)| (orig, batch))
        .collect();

    loop {
        if pending.is_empty() { break; }

        let effective_jobs = max_jobs.min(pending.len());
        let (mut sub, _provider) = match SubprocessManager::spawn(
            model, effective_jobs, mask, cpu_only, lines,
        ) {
            Ok(s) => s,
            Err(e) => {
                for &idx in &pending {
                    results[idx] = Some(Err(CoreError::Model(e.clone())));
                }
                break;
            }
        };

        let mut in_flight: Vec<usize> = Vec::new();
        let mut last_event_at = Instant::now();
        let crash = loop {
            while room_to_send(in_flight.len(), max_jobs, sub.should_pause_admission()) > 0 {
                let Some(idx) = pending.pop_front() else { break };
                match sub.send_image_path(valid_indices[idx] as u64, valid_paths[idx].clone()) {
                    Ok(()) => in_flight.push(idx),
                    // A live worker that cannot take it would refuse it again;
                    // one a dead worker refused is the crash path's to rerun.
                    Err(e) if sub.is_alive() => results[idx] = Some(Err(CoreError::Model(e))),
                    Err(_) => {
                        pending.push_front(idx);
                        break;
                    }
                }
            }

            // Wakes on arrival, so admission follows a finished image without a poll delay.
            let events = sub.poll_events_blocking(std::time::Duration::from_millis(50));
            if !events.is_empty() {
                last_event_at = Instant::now();
            }
            for event in events {
                let (item_id, result) = match event {
                    SubprocessEvent::Report { item_id, update: prunr_core::ProgressUpdate::Step(step) } => {
                        let orig_idx = item_id as usize;
                        if !quiet {
                            if let Some(Some(pb)) = spinners.get(orig_idx) {
                                pb.set_message(format!("{} \u{2014} {}", inputs[orig_idx].display(), step.label()));
                            }
                        }
                        continue;
                    }
                    SubprocessEvent::ImageDone { item_id, result_path, width, height, tensor_cache, edge_cache, .. } => {
                        // Tensors are for a parent that re-runs from them;
                        // the CLI never does, so their files go (in RAM on Linux).
                        for path in tensor_cache.map(|t| t.path).into_iter().chain(edge_cache.map(|e| e.path)) {
                            let _ = std::fs::remove_file(path);
                        }
                        let orig_idx = item_id as usize;
                        let result = prunr_app::subprocess::ipc::read_result_rgba(&result_path, width, height)
                            .map(|mut rgba_image| {
                                backdrop.apply(&mut rgba_image, &inputs[orig_idx]);
                                prunr_core::ProcessResult {
                                    rgba_image,
                                    active_provider: String::new(),
                                }
                            })
                            .ok_or_else(|| CoreError::Model("Failed to read subprocess result".into()));
                        (item_id, result)
                    }
                    SubprocessEvent::ImageError { item_id, error } => (item_id, Err(CoreError::Model(error))),
                    _ => continue,
                };
                if let Some(ridx) = orig_to_batch.get(&(item_id as usize)).copied() {
                    // in_flight holds batch indices, each sent once a pass.
                    in_flight.retain(|&i| i != ridx);
                    results[ridx] = Some(result);
                }
            }

            if in_flight.is_empty() && pending.is_empty() {
                break None;
            }
            if let Some(crash) = sub.died() {
                break Some(crash);
            }
            if is_stalled(last_event_at.elapsed(), in_flight.len(), HANG_TIMEOUT) {
                sub.kill();
                break Some(Crash::Unresponsive);
            }
        };

        let Some(crash) = crash else {
            // Matches the GUI's 5s grace budget for model-cache teardown;
            // Drop's 1s covers any slippage.
            let _ = sub.shutdown_with_timeout(std::time::Duration::from_secs(5));
            break;
        };
        sub.kill();
        prunr_app::subprocess::protocol::cleanup_seg_pipeline_temps();
        match on_crash(max_jobs) {
            OnCrash::FailInFlight => {
                let advice = if crash == Crash::OutOfMemory { " \u{2014} try a smaller model" } else { "" };
                for idx in in_flight {
                    results[idx] = Some(Err(CoreError::Model(format!("worker {crash}{advice}"))));
                }
            }
            OnCrash::RetryWith(jobs) => {
                for idx in in_flight.into_iter().rev() {
                    pending.push_front(idx);
                }
                max_jobs = jobs;
                if !quiet {
                    eprintln!("Worker {crash} \u{2014} retrying with {max_jobs} parallel jobs");
                }
            }
        }
    }

    // Crash recovery leaves `cli_ds_*` temps for retries to re-feed; sweep
    // whatever the worker didn't read-and-delete.
    prunr_app::subprocess::protocol::cleanup_cli_persistent_temps();

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
    let diag = prunr_core::ort_runtime::diagnose();

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
    println!("Recommended runtime: {}", p.recommended_runtime().map_or("none", |rt| rt.display_name()));
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
                let preferred = if diag.preferred.as_ref() == Some(name) { " (preferred)" } else { "" };
                println!("  - {name} [{mark}]{preferred}");
            }
        }
    }
    println!("Preferred:      {}", diag.preferred.as_deref().unwrap_or("(none: first by name)"));
    if let Some((path, exists)) = &diag.bundled {
        let mark = if *exists { "OK" } else { "(absent)" };
        println!("Bundled:        {} {mark}", path.display());
    }
    match &diag.resolved {
        Some((p, src)) => println!("Active source:  {src} → {}", p.display()),
        None => println!("Active source:  NONE — `prunr` will refuse to start without --doctor"),
    }
    for (lib, exists) in &diag.beside_runtime {
        println!("Beside it:      {} {}", lib.display(), if *exists { "OK" } else { "(absent)" });
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
    println!("Settings:     {}", path_or_unknown(prunr_app::gui::settings::Settings::config_path()));
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

/// `prunr bench`: each step on stderr as it starts, the machine and the
/// results table on stdout, and the report rewritten after every step so
/// a crash still leaves one naming the step it crashed in.
pub fn run_bench(args: &BenchArgs) -> i32 {
    use prunr_app::bench::{self, report};
    let models = match bench::select(&args.models, args.heavy, prunr_models::is_available) {
        Ok(models) => models,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let inputs = match &args.image {
        None => bench::Inputs::built_in(),
        Some(path) => match prunr_core::load_image_from_path(path) {
            Ok(img) => bench::Inputs::from_photo(img, path.display().to_string()),
            Err(e) => {
                eprintln!("error: cannot read {}: {e}", path.display());
                return 1;
            }
        },
    };
    let mut report = report::Report { header: report::Header::detect(&inputs.label, args.runs), rows: Vec::new(), running: None };
    print!("{}", report::header_text(&report.header));
    println!();
    let mut files = ReportFiles::new(args.out.clone());
    files.write(&report);
    bench::run(&models, &inputs, args.runs, &mut |event| match event {
        bench::Event::Stage { index, total, model, stage } => {
            eprintln!("[{index}/{total}] {model}: {stage}");
            report.running = Some(format!("{model}: {stage}"));
            files.write(&report);
        }
        bench::Event::Done(row) => {
            eprintln!("      {}: {}", row.model, row.verdict.label());
            report.rows.push(*row);
            report.running = None;
            files.write(&report);
        }
    });
    println!();
    print!("{}", report::terminal_table(&report.rows));
    println!();
    println!("{}.", report::summary(&report.rows));
    println!("Report: {} and {}", files.md.display(), files.json().display());
    0
}

/// The report's Markdown file and its JSON copy. A path that cannot be
/// written falls back once to the temp dir, so the report survives a
/// read-only working directory.
struct ReportFiles {
    md: PathBuf,
    warned: bool,
}

impl ReportFiles {
    fn new(md: PathBuf) -> Self {
        Self { md: std::path::absolute(&md).unwrap_or(md), warned: false }
    }

    fn json(&self) -> PathBuf {
        self.md.with_extension("json")
    }

    fn write(&mut self, report: &prunr_app::bench::report::Report) {
        let save = |files: &Self| -> std::io::Result<()> {
            std::fs::write(&files.md, prunr_app::bench::report::markdown(report))?;
            std::fs::write(files.json(), prunr_app::bench::report::json(report))
        };
        let Err(e) = save(self) else { return };
        if self.warned {
            return;
        }
        self.warned = true;
        let fallback = std::env::temp_dir().join(self.md.file_name().unwrap_or(std::ffi::OsStr::new("prunr-bench.md")));
        eprintln!("warning: cannot write {}: {e}; writing {} instead", self.md.display(), fallback.display());
        self.md = fallback;
        if let Err(e) = save(self) {
            eprintln!("warning: cannot write {} either: {e}", self.md.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn a_colour_flag_is_absent_hex_or_refused() {
        assert_eq!(super::flag_color("--bg-color", None, "ffffff"), Ok(None));
        assert_eq!(super::flag_color("--bg-color", Some("#ff00aa"), "ffffff"), Ok(Some([255, 0, 170])));
        assert_eq!(super::flag_color("--bg-color", Some("ff00aa "), "ffffff"), Err(()));
    }

    /// Every run that prints borrows the terminal on Windows, where a
    /// release build has none of its own; `--doctor` and
    /// `--clear-ep-cache` printed nowhere there. The GUI does not.
    #[test]
    fn every_printing_run_writes_to_the_terminal() {
        let writes = |args: &[&str]| super::Cli::parse_from(["prunr"].iter().chain(args)).writes_to_terminal();
        assert!(writes(&["--doctor"]));
        assert!(writes(&["--clear-ep-cache"]));
        assert!(writes(&["photo.jpg"]));
        assert!(writes(&["--debug"]));
        assert!(!writes(&[]), "the GUI");
        assert!(!writes(&["--open", "photo.jpg"]), "the GUI with an image");
        assert!(writes(&["bench"]));
    }

    /// `prunr bench` takes its own flags; a file named like it, or any
    /// other input, is still an image to process.
    #[test]
    fn bench_is_a_subcommand_beside_the_image_inputs() {
        let cli = super::Cli::parse_from(["prunr", "bench", "--models", "silueta,real_esrgan_x2plus", "--runs", "5", "--heavy", "--out", "r.md"]);
        let Some(super::Command::Bench(args)) = &cli.command else { panic!("no bench: {cli:?}") };
        assert_eq!(args.models, ["silueta", "real_esrgan_x2plus"]);
        assert_eq!((args.runs, args.heavy, args.out.as_path()), (5, true, std::path::Path::new("r.md")));
        assert!(cli.inputs.is_empty());
        let defaults = super::Cli::parse_from(["prunr", "bench"]);
        let Some(super::Command::Bench(args)) = &defaults.command else { panic!("no bench") };
        assert_eq!((args.models.len(), args.runs, args.heavy, args.image.as_ref()), (0, 3, false, None));
        assert_eq!(args.out, std::path::Path::new("prunr-bench.md"));
        let photo = super::Cli::parse_from(["prunr", "bench.jpg", "b.png"]);
        assert!(photo.command.is_none());
        assert_eq!(photo.inputs.len(), 2);
        assert!(super::Cli::try_parse_from(["prunr", "bench", "--runs", "500"]).is_err());
    }

    /// `-o` with one input is the output file; with several it is the
    /// folder each result goes into.
    #[test]
    fn dash_o_is_the_file_for_one_input_and_the_folder_for_several() {
        use std::path::Path;
        let one = super::Cli::parse_from(["prunr", "in/a.jpg", "-o", "out/result.png"]);
        assert_eq!(one.result_path(Path::new("in/a.jpg")), Path::new("out/result.png"));
        let many = super::Cli::parse_from(["prunr", "in/a.jpg", "in/b.jpg", "-o", "out"]);
        assert_eq!(many.result_path(Path::new("in/b.jpg")), Path::new("out/b.prunr.png"));
        let none = super::Cli::parse_from(["prunr", "in/a.jpg"]);
        assert_eq!(none.result_path(Path::new("in/a.jpg")), Path::new("in/a.prunr.png"));
    }

    #[test]
    fn outline_and_shadow_flags_reach_the_mask_settings() {
        let cli = super::Cli::parse_from(["prunr", "a.png", "--outline", "8:000000", "--shadow", "drop"]);
        let d = cli.mask_settings().decoration;
        assert_eq!((d.outline_width, d.outline_color, d.shadow), (8.0, [0, 0, 0], prunr_core::ShadowKind::Drop));
        let white = super::Cli::parse_from(["prunr", "a.png", "--outline", "4"]).mask_settings().decoration;
        assert_eq!(white.outline_color, [255, 255, 255]);
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--outline", "120"]).is_err());
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--shadow", "hard"]).is_err());
    }

    /// An `-o` name without an extension saves a PNG, whatever the mode.
    #[test]
    fn an_output_name_without_an_extension_is_a_png_file() {
        use super::output_path_with_suffix;
        use std::path::PathBuf;
        let input = std::path::Path::new("/photos/a.jpg");
        let out = |o: &str| output_path_with_suffix(input, &Some(PathBuf::from(o)), false, "_x.png");
        assert_eq!(out("/tmp/result"), PathBuf::from("/tmp/result.png"));
        assert_eq!(out("/tmp/result.webp"), PathBuf::from("/tmp/result.webp"));
        assert_eq!(output_path_with_suffix(input, &Some(PathBuf::from("/tmp/dir")), true, "_x.png"), PathBuf::from("/tmp/dir/a_x.png"));
        assert_eq!(output_path_with_suffix(input, &None, false, "_x.png"), PathBuf::from("/photos/a_x.png"));
    }

    /// `--line-fade` reads the depth model only when lines are drawn;
    /// on its own it does nothing and must not refuse the run.
    #[test]
    fn line_fade_needs_depth_only_with_lines() {
        let wants = |args: &[&str]| super::wants_depth(&super::Cli::parse_from(["prunr", "a.png"].iter().chain(args).copied()));
        assert!(!wants(&["--line-fade", "40"]));
        assert!(wants(&["--lines", "--line-fade", "40"]));
        assert!(wants(&["--lines-after-bg", "--line-fade", "40"]));
        assert!(wants(&["--depth-map"]));
        assert!(!wants(&["--lines"]));
    }

    /// The upscale flags parse into a mode; its knobs need it, and faces
    /// need a face model named.
    #[test]
    fn the_upscale_flags_need_upscale_mode() {
        let cli = super::Cli::parse_from(["prunr", "a.png", "--upscale", "nomos", "--scale", "3", "--sharpen", "-0.5", "--restore-faces", "restoreformer", "--face-strength", "70"]);
        assert_eq!(cli.upscale, Some(super::CliUpscaler::Nomos));
        assert_eq!((cli.scale, cli.sharpen, cli.face_strength), (3, -0.5, 70));
        assert_eq!(cli.restore_faces.map(super::CliFaceModel::model_id), Some(prunr_models::ModelId::FaceRestoreFormerPp));
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--scale", "2"]).is_err(), "--scale needs --upscale");
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--upscale", "nomos", "--scale", "5"]).is_err());
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--upscale", "nomos", "--face-strength", "50"]).is_err(), "strength needs a face model");
        let mixed = super::Cli::parse_from(["prunr", "a.png", "--upscale", "siax", "--bg-color", "ff0000"]);
        assert_eq!(super::background_removal_flags(&mixed), ["--bg-color"]);
    }

    /// Colorize is its own mode: Strength needs it, and it takes no other
    /// mode alongside.
    #[test]
    fn the_colorize_flags_need_colorize_mode() {
        let cli = super::Cli::parse_from(["prunr", "a.png", "--colorize", "--color-strength", "0.6"]);
        assert!(cli.colorize);
        assert_eq!(cli.color_strength, 0.6);
        assert_eq!(super::Cli::parse_from(["prunr", "a.png", "--colorize"]).color_strength, 1.0);
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--color-strength", "0.5"]).is_err(), "--color-strength needs --colorize");
        let both = super::Cli::parse_from(["prunr", "a.png", "--colorize", "--upscale", "nomos"]);
        assert_eq!(super::selected_modes(&both), ["--upscale", "--colorize"]);
        let mixed = super::Cli::parse_from(["prunr", "a.png", "--colorize", "--bg-color", "ff0000"]);
        assert_eq!(super::background_removal_flags(&mixed), ["--bg-color"]);
    }

    /// Help › Command-line reference lists every flag `prunr --help`
    /// shows, so a new or renamed flag cannot leave it behind.
    #[test]
    fn the_gui_reference_names_every_visible_flag() {
        use clap::CommandFactory;
        let text = include_str!("gui/views/cli_help.rs");
        let missing: Vec<String> = super::Cli::command()
            .get_arguments()
            .filter(|a| !a.is_hide_set())
            .filter_map(|a| a.get_long())
            .filter(|long| !["help", "version"].contains(long))
            .map(|long| format!("--{long}"))
            .filter(|flag| !text.contains(&format!("\"{flag}")))
            .collect();
        assert!(missing.is_empty(), "missing from the reference: {missing:?}");
    }

    #[test]
    fn bg_effect_reads_blur_with_a_radius_and_bw() {
        use super::parse_bg_effect;
        use prunr_core::BgEffect;
        assert_eq!(parse_bg_effect("blur"), Ok(BgEffect::BlurredSource { radius: 12 }));
        assert_eq!(parse_bg_effect("blur:30"), Ok(BgEffect::BlurredSource { radius: 30 }));
        assert_eq!(parse_bg_effect("bw"), Ok(BgEffect::DesaturatedSource));
        assert_eq!(parse_bg_effect("lens"), Ok(BgEffect::LensBlur { amount_pct: 50 }));
        assert_eq!(parse_bg_effect("lens:80"), Ok(BgEffect::LensBlur { amount_pct: 80 }));
        assert!(parse_bg_effect("lens:120").is_err());
        for bad in ["blur:0", "blur:65", "blur:x", "invert", ""] {
            assert!(parse_bg_effect(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_depth_map_goes_beside_the_result() {
        use clap::Parser;
        let args = super::Cli::parse_from(["prunr", "/p/cat.jpg", "-o", "/out/x.png", "--depth-map"]);
        assert_eq!(super::depth_map_path(&args, std::path::Path::new("/p/cat.jpg")), std::path::PathBuf::from("/out/cat-depth.png"));
        let args = super::Cli::parse_from(["prunr", "/p/cat.jpg", "--depth-map", "--depth-only"]);
        assert_eq!(super::depth_map_path(&args, std::path::Path::new("/p/cat.jpg")), std::path::PathBuf::from("/p/cat-depth.png"));
        assert!(super::Cli::try_parse_from(["prunr", "/p/cat.jpg", "--depth-only"]).is_err(), "--depth-only needs --depth-map");
    }

    /// A stem that is not UTF-8 names its result and depth map as it is:
    /// a mended name would let two inputs write one file.
    #[cfg(unix)]
    #[test]
    fn a_name_that_is_not_utf8_is_kept() {
        use clap::Parser;
        use std::os::unix::ffi::OsStrExt;
        let os = |bytes: &[u8]| std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes));
        let input = os(b"/p/caf\xe9.jpg");
        let args = super::Cli::parse_from([std::ffi::OsStr::new("prunr"), input.as_os_str(), std::ffi::OsStr::new("--depth-map")]);
        assert_eq!(args.result_path(&input), os(b"/p/caf\xe9.prunr.png"));
        assert_eq!(super::depth_map_path(&args, &input), os(b"/p/caf\xe9-depth.png"));
    }

    /// `--depth-map` writes the map a backdrop measured instead of
    /// measuring the input again, and keeps it no longer.
    #[test]
    fn a_depth_map_a_backdrop_measured_is_written_without_measuring_again() {
        use clap::Parser;
        let args = super::Cli::parse_from(["prunr", "/missing/cat.jpg", "--depth-map"]);
        let depth = super::CliDepth::new(&args);
        let map = std::sync::Arc::new(prunr_core::depth::DepthMap::from_raw(1, 2, vec![0.0, 1.0]));
        let input = std::path::Path::new("/missing/cat.jpg");
        depth.kept.as_ref().expect("kept with --depth-map").borrow_mut().insert(input.into(), (std::sync::Arc::clone(&map), (4, 8)));
        let (got, dims) = depth.for_file(input).expect("the kept map");
        assert!(std::sync::Arc::ptr_eq(&got, &map) && dims == (4, 8));
        assert!(depth.for_file(input).is_err(), "taken once; the missing file is read the next time");
        assert!(super::CliDepth::new(&super::Cli::parse_from(["prunr", "/p/cat.jpg"])).kept.is_none(), "nothing kept without --depth-map");
    }

    /// The effect is made from the input photo and fills what the result
    /// leaves transparent; the result's own pixels stay.
    #[test]
    fn a_bg_effect_fills_the_transparent_part_from_the_photo() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("photo.png");
        image::RgbaImage::from_pixel(4, 4, image::Rgba([200, 50, 50, 255])).save(&photo).unwrap();
        let mut result = image::RgbaImage::from_fn(4, 4, |x, _| if x == 0 { image::Rgba([0, 0, 255, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        super::CliBackdrop::Effect(prunr_core::BgEffect::DesaturatedSource, &super::CliDepth::default()).apply(&mut result, &photo);
        assert_eq!(result.get_pixel(0, 0).0, [0, 0, 255, 255]);
        let p = result.get_pixel(3, 3).0;
        assert!(p[3] == 255 && p[0] == p[1] && p[1] == p[2], "grey photo behind: {p:?}");
    }

    /// `--edge-shift` reads like the GUI's Shift edge: + grows the subject,
    /// which the pipeline stores as a negative (dilating) shift.
    #[test]
    fn a_positive_edge_shift_grows_the_subject() {
        let grow = super::Cli::try_parse_from(["prunr", "--edge-shift", "3", "x.png"]).unwrap();
        assert_eq!(grow.mask_settings().edge_shift, -3.0);
        let shrink = super::Cli::try_parse_from(["prunr", "--edge-shift", "-2", "x.png"]).unwrap();
        assert_eq!(shrink.mask_settings().edge_shift, 2.0);
    }

    /// The Sketch flags: --line-detail (and its old name), --line-width
    /// and --line-composition.
    #[test]
    fn sketch_flags_parse() {
        let a = super::Cli::try_parse_from(["prunr", "--line-detail", "0.8", "--line-width", "3", "--line-composition", "ghost", "x.png"]).unwrap();
        assert_eq!((a.line_strength, a.line_width, a.line_composition), (0.8, 3, prunr_core::ComposeMode::Ghost));
        let old = super::Cli::try_parse_from(["prunr", "--line-strength", "0.2", "x.png"]).unwrap();
        assert_eq!(old.line_strength, 0.2);
        assert!(super::Cli::try_parse_from(["prunr", "--line-composition", "inverse", "x.png"]).is_err());
    }

    /// Scripts written for the retired Balanced scale keep working.
    #[test]
    fn line_scale_balanced_reads_as_fused() {
        let a = super::Cli::try_parse_from(["prunr", "--line-scale", "balanced", "x.png"]).unwrap();
        assert_eq!(a.line_scale, prunr_core::EdgeScale::Fused);
        let fine = super::Cli::try_parse_from(["prunr", "--line-scale", "fine", "x.png"]).unwrap();
        assert_eq!(fine.line_scale, prunr_core::EdgeScale::Fine);
    }

    /// Every Mask panel knob has a flag: Feather and Edge width reach the
    /// pipeline, defaulting to the panel's defaults.
    #[test]
    fn feather_and_edge_width_reach_the_mask() {
        let m = super::Cli::try_parse_from(["prunr", "--feather", "1.5", "--edge-width", "12", "x.png"]).unwrap().mask_settings();
        assert_eq!((m.feather, m.guided_radius), (1.5, 12));
        let d = super::Cli::try_parse_from(["prunr", "x.png"]).unwrap().mask_settings();
        let panel = prunr_core::MaskSettings::default();
        assert_eq!((d.feather, d.guided_radius), (panel.feather, panel.guided_radius));
    }
}
