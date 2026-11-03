use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Prunr — local AI background removal.
///
/// No arguments launches the GUI. Pass image files directly to process them:
///   prunr photo.jpg              # removes background, saves photo.prunr.png
///   prunr *.jpg -o clean/        # batch to folder
///   prunr -m u2net portrait.jpg  # use quality model
///   prunr -m none --preset Dreamy photo.jpg  # a whole-photo look, no cut-out
#[derive(Parser, Debug)]
#[command(name = "prunr", version, about, long_about = None)]
pub struct Cli {
    /// Input image file(s).
    pub inputs: Vec<PathBuf>,

    /// Output path: a file for one image (a name without an extension is saved as .png), a directory for a batch.
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Open the app with this image loaded.
    #[arg(long, value_name = "PATH")]
    pub open: Option<PathBuf>,

    /// Model: birefnet-lite (default), silueta (fast), u2net (quality), or
    /// none (no cut-out: a preset's whole-photo look, lines over the photo).
    #[arg(short = 'm', long, default_value = "birefnet-lite")]
    pub model: CliModel,

    /// Start from this preset: a name from the app's Presets menu, or a
    /// preset file. Every setting below that is not given takes the
    /// preset's value for the model; without --preset, Prunr's own.
    #[arg(long, value_name = "NAME|FILE")]
    pub preset: Option<String>,

    /// Images processed at once [default: as the app: half the cores, 2 on
    /// a graphics card, fewer when memory is short].
    #[arg(short = 'j', long, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub jobs: Option<usize>,

    /// Images over 8000 px on either side: keep their size, or shrink
    /// them to 4096 px first.
    #[arg(long, default_value = "process")]
    pub large_image: LargeImagePolicy,

    /// Overwrite existing output files.
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Suppress progress output (errors still go to stderr).
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Detailed log for a bug report; on Windows the app also writes it to
    /// the terminal it started from.
    #[arg(long)]
    pub debug: bool,

    /// Mask gamma (removal strength). >1 = more aggressive, <1 = gentler.
    #[arg(long, value_parser = finite)]
    pub gamma: Option<f32>,

    /// Edge contrast (0.0–1.0): 0 keeps the soft edge, 1 cuts it hard.
    #[arg(long, value_parser = finite)]
    pub contrast: Option<f32>,

    /// Shift the cut-out's edge in pixels, -20 to 20: positive grows the
    /// subject, negative shrinks it.
    #[arg(long, allow_negative_numbers = true, value_parser = edge_shift_px)]
    pub edge_shift: Option<f32>,

    /// Refine mask edges using guided filter for better detail on hair,
    /// leaves, etc. `--refine-edges=false` turns a preset's off.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
    pub refine_edges: Option<bool>,

    /// How far from the edge --refine-edges looks, in pixels.
    #[arg(long)]
    pub edge_width: Option<u32>,

    /// What --refine-edges follows in the photo: brightness, or color too
    /// (also an edge between two colors of the same brightness, slower).
    #[arg(long, value_name = "brightness|color")]
    pub refine_guide: Option<prunr_core::RefineGuide>,

    /// Soften the cut-out's edge by this many pixels (fractions allowed).
    #[arg(long, value_parser = finite)]
    pub feather: Option<f32>,

    /// Extract lines/edges instead of removing background (uses TEED model).
    #[arg(long)]
    pub lines: bool,

    /// Run line extraction after background removal (combine both).
    #[arg(long)]
    pub lines_after_bg: bool,

    /// How much of the detected edges become lines (0.0–1.0): lower draws
    /// bold outlines only, higher adds fine detail.
    #[arg(long = "line-detail", visible_alias = "line-strength", value_parser = finite)]
    pub line_strength: Option<f32>,

    /// Line width in pixels (0–20).
    #[arg(long)]
    pub line_width: Option<u32>,

    /// How the lines combine with the subject (with --lines-after-bg):
    /// lines-only, subject-filled, engraving or ghost.
    #[arg(long, value_parser = str::parse::<prunr_core::ComposeMode>)]
    pub line_composition: Option<prunr_core::ComposeMode>,

    /// Paint all lines a solid color (hex, e.g. "000000" for black, "ff0000" for red).
    #[arg(long)]
    pub line_color: Option<String>,

    /// Line scale: `fine` (thin, detailed lines), `bold` (big outlines
    /// only), `fused` (every scale combined). `balanced` is read as
    /// `fused`.
    #[arg(long)]
    pub line_scale: Option<prunr_core::EdgeScale>,

    /// Fill transparent background with a color (hex, e.g. "ffffff" for white).
    #[arg(long)]
    pub bg_color: Option<String>,

    /// Fill transparent background with an image (PNG/JPEG/WebP/BMP). Wins
    /// over `--bg-color` when both are set, as in the app.
    #[arg(long, value_name = "PATH")]
    pub bg_image: Option<std::path::PathBuf>,

    /// How the background image is positioned: cover (fills both dims, may
    /// crop), contain (fits inside, may letterbox), tile (repeat at native
    /// size).
    #[arg(long)]
    pub bg_image_fit: Option<prunr_core::BgImageFit>,

    /// Restyle the subject's colors, with the effect's default settings:
    /// desaturate, invert, duotone, sepia, threshold, posterize, solarize,
    /// hue-shift, saturate, color-splash, pixelate, cross-process,
    /// halftone or gradient-map.
    #[arg(long, value_name = "NAME", value_parser = str::parse::<prunr_core::SubjectEffect>)]
    pub effect: Option<prunr_core::SubjectEffect>,

    /// Fill transparent background with a copy of the photo: `blur`
    /// (optionally `blur:N`, N = 1-64 px), `lens` (optionally `lens:N`,
    /// N = 0-100: blurs more the farther from the subject; needs the depth
    /// model) or `bw` (black and white). Wins over `--bg-image` and
    /// `--bg-color`, as in the app.
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

    /// Print the computer's hardware, Prunr's AI runtime, the installed
    /// models and Prunr's folders, and exit. Paste it into a bug report
    /// when hardware acceleration misbehaves.
    #[arg(long)]
    pub doctor: bool,

    /// Forget which models failed on which accelerator, so each is tried
    /// there again (after a graphics driver update).
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
    // `requires`: `--mask m.png foo.jpg` without `--inpaint` errors out
    // instead of dropping the flag.
    #[arg(long, requires = "inpaint")]
    pub mask: Option<PathBuf>,

    /// How large the eraser fills the masked area: auto (full size for
    /// small areas, large ones shrink to fit one model tile), full, half
    /// or quarter. A large area fills more convincingly at a smaller
    /// scale, a little softer.
    #[arg(long, requires = "inpaint")]
    pub fill_scale: Option<prunr_core::inpaint::FillScale>,

    /// Upscale mode: enlarge the image with this model instead of
    /// removing its background. Saves {stem}_x{scale}.png.
    #[arg(long, value_name = "MODEL")]
    pub upscale: Option<CliUpscaler>,

    /// How much larger the upscale makes the image.
    #[arg(long, requires = "upscale", value_parser = clap::value_parser!(u8).range(2..=4))]
    pub scale: Option<u8>,

    /// Denoise before the upscale model, 0 (off) to 1.
    #[arg(long, requires = "upscale", value_parser = finite)]
    pub denoise: Option<f32>,

    /// Sharpen after the upscale, -1 (soften) to 1; 0 is off.
    #[arg(long, requires = "upscale", allow_negative_numbers = true, value_parser = finite)]
    pub sharpen: Option<f32>,

    /// AI blend: 1 keeps the model's output, 0 the plain enlargement.
    #[arg(long, requires = "upscale", value_parser = finite)]
    pub ai_blend: Option<f32>,

    /// Match the upscale's colours to the original. `--color-match=false`
    /// turns a preset's off.
    #[arg(long, requires = "upscale", num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
    pub color_match: Option<bool>,

    /// Restore faces with this model (downloaded in the app's Model
    /// Store): after an upscale, else into the photo before the cut-out,
    /// effect and lines, as the app does. The eraser and Colorize do not.
    #[arg(long, value_name = "MODEL", conflicts_with_all = ["inpaint", "colorize"])]
    pub restore_faces: Option<CliFaceModel>,

    /// Restore faces strength in percent: up to 50 adds detail and keeps
    /// the photo's shading, above it the redrawn shading too.
    #[arg(long, requires = "restore_faces", value_parser = clap::value_parser!(u8).range(0..=100))]
    pub face_strength: Option<u8>,

    /// Colorize mode: color a black-and-white photo with DDColor
    /// (downloaded in the app's Model Store). Saves {stem}_color.png.
    #[arg(long)]
    pub colorize: bool,

    /// Color strength: 0 is grey, 1 the predicted color, up to 1.5.
    #[arg(long, requires = "colorize", value_parser = finite)]
    pub color_strength: Option<f32>,

    /// Lines fade with distance, 0-100 (needs the depth model, from the
    /// app's Model Store).
    #[arg(long, value_name = "PERCENT", value_parser = clap::value_parser!(u8).range(0..=100))]
    pub line_fade: Option<u8>,

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

    /// Also print where each model runs on this machine's accelerator, as
    /// the registry's `places` lines, measured where LABEL says (e.g.
    /// "RTX 3060, 2026-10-25"). Places the registry keeps on the CPU are
    /// measured again, except those that fail there.
    #[arg(long, value_name = "LABEL")]
    pub table: Option<String>,

    /// Measure memory instead of speed: each model's peak system RAM and
    /// graphics-card memory on the CPU and on the GPU path, each in a
    /// fresh process, with the registry lines to paste.
    #[arg(long)]
    pub needs: bool,

    /// With `--needs`, give an upscaler a square input of this side (no
    /// larger than the photo) instead of its usual one. Up to the model's
    /// own tile it runs as one tile of that side, so its memory shows how
    /// memory grows with the tile; above it, the model tiles at its own.
    #[arg(long, value_name = "PIXELS", value_parser = clap::value_parser!(u32).range(16..=2048))]
    pub tile: Option<u32>,

    /// One `--needs` measurement, `MODEL:cpu` or `MODEL:gpu`, run in its
    /// own process by `--needs`.
    #[arg(long, hide = true, value_name = "MODEL:SIDE")]
    pub needs_one: Option<String>,

    /// Settings › Hardware › Test this computer: place each installed
    /// model on this machine, its progress on stdout for the app.
    #[arg(long, hide = true)]
    pub here: bool,

    /// After a download onto an accelerator no one measured the model on:
    /// check its result there, run alone.
    #[arg(long, hide = true, value_name = "MODEL")]
    pub check: Option<String>,
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
    fn settings_model(self) -> SettingsModel {
        match self {
            Self::Realesrgan => SettingsModel::RealEsrganUpscale,
            Self::Nomos => SettingsModel::Nomos8kUpscale,
            Self::Siax => SettingsModel::FourXNmkdSiaxCxUpscale,
            Self::Superscale => SettingsModel::FourXNmkdSuperscaleUpscale,
            Self::Nomos2 => SettingsModel::Nomos2Upscale,
        }
    }

    fn model_id(self) -> prunr_models::ModelId {
        // settings_model gives only upscale models, and each names a model.
        self.settings_model().to_model_id().expect("every upscaler names a model")
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
    fn face_model(self) -> crate::gui::item_settings::FaceModel {
        use crate::gui::item_settings::FaceModel;
        match self {
            Self::Gfpgan => FaceModel::Gfpgan,
            Self::Restoreformer => FaceModel::RestoreFormerPp,
        }
    }
}

/// Model selection
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliModel {
    /// Silueta: fast
    Silueta,
    /// U2Net: downloaded in the app's Model Store
    U2net,
    /// BiRefNet-lite: the finest detail (default)
    BirefnetLite,
    /// No model: effects, lines and background over the whole photo
    #[value(name = "none")]
    NoModel,
}

impl CliModel {
    fn settings_model(self) -> SettingsModel {
        match self {
            Self::Silueta => SettingsModel::Silueta,
            Self::U2net => SettingsModel::U2net,
            Self::BirefnetLite => SettingsModel::BiRefNetLite,
            Self::NoModel => SettingsModel::None,
        }
    }

    /// The cut-out model this runs; `None` for No model.
    fn model_kind(self) -> Option<prunr_core::ModelKind> {
        self.settings_model().to_model_kind()
    }

    /// `--model <value>` for the value that runs `kind`, when one does.
    fn flag_for(kind: prunr_core::ModelKind) -> Option<String> {
        let model = Self::value_variants().iter().find(|m| m.model_kind() == Some(kind))?;
        model.to_possible_value().map(|v| format!("--model {}", v.get_name()))
    }
}

/// How to handle images exceeding LARGE_IMAGE_LIMIT
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargeImagePolicy {
    /// Shrink to 4096 px before the model runs
    Downscale,
    /// Keep the original size; a photo too large for the free memory is refused
    Process,
}

use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use prunr_core::{
    MaskSettings, ModelKind, CoreError,
    DOWNSCALE_TARGET,
    load_image_from_path, downscale_image, encode_rgba_png,
};

use crate::gui::brush_state::BrushSettings;
use crate::gui::item_settings::{Backdrop, ItemSettings};
use crate::gui::settings::{LineMode, SettingsModel};
use prunr_core::inpaint_blend::{fill_region, finalize_inpaint};

/// What a run does: `--preset`'s settings for the model in use (Prunr's
/// without one), with every setting flag given laid over them.
struct RunSettings {
    item: ItemSettings,
    brush: BrushSettings,
}

/// `setting` as the flag gives it, when given.
fn lay<T>(setting: &mut T, flag: Option<T>) {
    if let Some(value) = flag {
        *setting = value;
    }
}

/// What goes behind each result, from `--bg-effect`, `--bg-image` and
/// `--bg-color`.
enum CliBackdrop<'a> {
    Transparent,
    Color([u8; 3]),
    Image(std::sync::Arc<image::DynamicImage>, prunr_core::BgImageFit),
    /// An effect, where a lens blur gets its depth, and whether a photo
    /// over the limit is shrunk first, as the run's own copy was.
    Effect(prunr_core::BgEffect, &'a CliDepth, bool),
}

impl CliBackdrop<'_> {
    /// Put the backdrop under `result`; an effect is made from the photo
    /// at `photo_path`, `photo` when it is decoded already.
    fn apply(&self, result: &mut image::RgbaImage, photo_path: &std::path::Path, photo: Option<&image::RgbaImage>) {
        match self {
            CliBackdrop::Transparent => {}
            CliBackdrop::Color(c) => prunr_core::apply_background_color(result, *c),
            CliBackdrop::Image(img, fit) => prunr_core::apply_background_image(result, img, *fit),
            CliBackdrop::Effect(effect, depth, downscale) => {
                let decoded;
                let photo = match photo {
                    Some(photo) => photo,
                    // Read as the run read it: a photo the run shrank is the
                    // one its result was cut from.
                    None => match read_for_run(photo_path, *downscale, true) {
                        Ok(read) => {
                            decoded = read.into_rgba8();
                            &decoded
                        }
                        Err(err) => return tracing::warn!(path = %photo_path.display(), %err, "bg effect: photo unreadable"),
                    },
                };
                let depth = effect.needs_depth()
                    .then(|| depth.for_photo(photo_path, photo, None)
                        .map_err(|err| tracing::warn!(path = %photo_path.display(), %err, "bg effect: depth"))
                        .ok())
                    .flatten();
                prunr_core::apply_bg_effect(result, photo, *effect, depth.as_deref());
            }
        }
    }

    /// An effect is made from the photo.
    fn reads_photo(&self) -> bool {
        matches!(self, CliBackdrop::Effect(..))
    }
}

/// The face restorer a run uses, and how its faces blend in.
type CliFaces = (prunr_core::face::FaceRestorer, prunr_core::face::FaceBlend);

/// The face restorer `s` asks for, loaded; `Err` when its model is not
/// installed or does not load, said already.
fn load_faces(args: &Cli, s: &ItemSettings) -> Result<Option<CliFaces>, ()> {
    let Some(model) = s.faces.wanted() else { return Ok(None) };
    if !prunr_models::is_available(model) {
        eprintln!("error: {}", prunr_models::not_installed_error(model));
        return Err(());
    }
    prunr_core::face::FaceRestorer::new(model, args.cpu)
        .map(|restorer| Some((restorer, s.faces.blend())))
        .map_err(|e| eprintln!("error: cannot load the face model: {e}"))
}

/// The faces in `image`, found and redrawn.
fn restore_faces((restorer, blend): &mut CliFaces, image: &image::RgbaImage) -> Result<Vec<prunr_core::face::RestoredFace>, String> {
    restorer.find_and_restore(image, *blend, &prunr_core::Progress::none()).map_err(|e| format!("restore faces failed: {e}"))
}

/// The app's render of each photo, after the worker or in its place:
/// where faces go into the photo first, and for No model, whose photo is
/// the render. The worker draws a cut-out's run as the app shows it.
struct CliRender<'a> {
    settings: ItemSettings,
    role: crate::gui::knob_catalog::ModelRole,
    faces: Option<CliFaces>,
    depth: &'a CliDepth,
    /// `--large-image downscale`: a photo over the limit is shrunk, as the
    /// worker's copy was, and said so unless `quiet`.
    downscale: bool,
    quiet: bool,
}

impl CliRender<'_> {
    /// Why the worker's answer cannot be rendered: a tensor the run made
    /// did not come back with it.
    fn lacks(&self, seg: Option<&crate::gui::worker::RawSegTensor>, edges: Option<&crate::gui::worker::RawEdgeTensors>) -> Option<&'static str> {
        let cut = self.role == crate::gui::knob_catalog::ModelRole::Cutout && self.settings.line_mode.cuts_out();
        if cut && seg.is_none() {
            Some("the worker's cut-out mask did not come back. Run it again.")
        } else if self.settings.line_mode != LineMode::Off && edges.is_none() {
            Some("the worker's lines did not come back. Run it again.")
        } else {
            None
        }
    }

    /// `path`'s photo as the app renders it, cut and lined from the
    /// worker's tensors when it ran, on `backdrop`.
    fn render(
        &mut self,
        backdrop: &CliBackdrop,
        path: &std::path::Path,
        seg: Option<crate::gui::worker::RawSegTensor>,
        edges: Option<crate::gui::worker::RawEdgeTensors>,
    ) -> Result<image::RgbaImage, String> {
        let photo = read_for_run(path, self.downscale, self.quiet).map_err(|e| e.message())?.into_rgba8();
        let faces = self.faces.as_mut().map(|faces| restore_faces(faces, &photo).map(|found| (std::sync::Arc::new(found), faces.1))).transpose()?;
        let places: Option<Vec<prunr_core::face::FacePlace>> = faces.as_ref().map(|(found, _)| found.iter().map(|face| *face.place()).collect());
        let depth = self.settings.photo_reads_depth(self.role)
            .then(|| self.depth.for_photo(path, &photo, places.as_deref()))
            .transpose()
            .map_err(|e| format!("depth failed: {e}"))?;
        let photo = std::sync::Arc::new(photo);
        // Kept only for an effect backdrop: a photo the render alone holds
        // is not copied for the faces or the effect.
        let kept = backdrop.reads_photo().then(|| std::sync::Arc::clone(&photo));
        let mut image = crate::gui::live_preview::ExportRender { photo, settings: self.settings, role: self.role, seg, edges, faces, depth }.render()?;
        backdrop.apply(&mut image, path, kept.as_deref());
        Ok(image)
    }
}

/// What each result takes before it is saved: the app's render where the
/// worker's result is not what the app shows, then the backdrop.
struct CliFinish<'a> {
    backdrop: CliBackdrop<'a>,
    render: Option<CliRender<'a>>,
}

/// The finish for this run's results; `None` when a background picture or
/// a face model cannot be had, said already.
fn cli_finish<'a>(args: &Cli, depth: &'a CliDepth, s: &ItemSettings) -> Option<CliFinish<'a>> {
    let bg_image = match args.bg_image.as_deref() {
        Some(p) => match prunr_core::load_image_from_path(p) {
            Ok(img) => Some(std::sync::Arc::new(img)),
            Err(e) => {
                eprintln!("error: --bg-image: {e}");
                return None;
            }
        },
        None => None,
    };
    // The app's rule: an effect wins over an image, an image over a colour.
    let backdrop = match (s.backdrop(), bg_image) {
        (Backdrop::Effect(effect), _) => CliBackdrop::Effect(effect, depth, args.large_image == LargeImagePolicy::Downscale),
        (_, Some(image)) => CliBackdrop::Image(image, s.bg_image_fit),
        (Backdrop::Color(color), None) => CliBackdrop::Color(color),
        (Backdrop::Image(_), None) => {
            if !args.quiet {
                eprintln!("note: the preset's background picture is not used on the command line; give it with --bg-image");
            }
            CliBackdrop::Transparent
        }
        (Backdrop::Transparent, None) => CliBackdrop::Transparent,
    };
    let faces = load_faces(args, s).ok()?;
    let role = args.settings_model().role();
    let renders = faces.is_some() || role == crate::gui::knob_catalog::ModelRole::Filters;
    let downscale = args.large_image == LargeImagePolicy::Downscale;
    // A worker read the photo first, and said so, unless the photo is
    // rendered alone.
    let quiet = args.quiet || !args.settings_model().renders_photo_alone(s.line_mode);
    let render = renders.then(|| CliRender { settings: *s, role, faces, depth, downscale, quiet });
    Some(CliFinish { backdrop, render })
}

/// Write `image` to `path` as PNG, replacing a file there.
fn save_png(path: &std::path::Path, image: &image::RgbaImage) -> std::io::Result<()> {
    prunr_app::fs_util::write_replacing(path, |w| prunr_core::ExportFormat::Png.encode_into(image, w).map_err(std::io::Error::other)).map(drop)
}

/// The line a run of images ends with, and its exit code: 0 all saved,
/// 1 none, 2 some.
fn finish_run(quiet: bool, saved: usize, failed: usize) -> i32 {
    if !quiet {
        println!("{saved} succeeded, {failed} failed.");
    }
    match (failed, saved) {
        (0, _) => 0,
        (_, 0) => 1,
        _ => 2,
    }
}

/// No model without lines: no worker runs. Each photo is the app's render
/// of it (faces, the Depth chip's look, the effect) on its backdrop.
fn run_whole_photo(args: &Cli, depth: &CliDepth, s: &ItemSettings) -> i32 {
    let Some(CliFinish { backdrop, render: Some(mut render) }) = cli_finish(args, depth, s) else { return 1 };
    let (mut saved, mut failed) = (0usize, 0usize);
    for (input, out_path) in args.inputs.iter().zip(args.result_paths()) {
        let started = Instant::now();
        if let Err(e) = check_overwrite(&out_path, args.force) {
            failed += 1;
            report_skipped(None, input, &e, args.quiet);
            continue;
        }
        let image = match render.render(&backdrop, input, None, None) {
            Ok(image) => image,
            Err(e) => {
                failed += 1;
                report_failure(None, input, "process", e);
                continue;
            }
        };
        if let Err(e) = save_png(&out_path, &image) {
            failed += 1;
            report_failure(None, &out_path, "write", e);
            continue;
        }
        saved += 1;
        if !args.quiet {
            println!("✓ {} ({:.1}s)", input.display(), started.elapsed().as_secs_f32());
        }
    }
    finish_run(args.quiet, saved, failed)
}

/// `blur`, `blur:N` (1-64 px), `lens`, `lens:N` (0-100 %) or `bw`.
/// A float flag's value: any number but nan and the infinities, which
/// mean nothing as a knob.
fn finite(s: &str) -> Result<f32, String> {
    s.parse::<f32>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("'{s}' is not a finite number"))
}

fn edge_shift_px(s: &str) -> Result<f32, String> {
    let max = prunr_core::MaskSettings::EDGE_SHIFT_MAX_PX;
    finite(s).and_then(|v| if v.abs() <= max { Ok(v) } else { Err(format!("'{s}' is not a shift from -{max} to {max} pixels")) })
}

fn parse_bg_effect(s: &str) -> Result<prunr_core::BgEffect, String> {
    use prunr_core::BgEffect;
    match s.split_once(':') {
        None if s == "blur" => Ok(BgEffect::BlurredSource { radius: BgEffect::BLUR_START_RADIUS }),
        None if s == "lens" => Ok(BgEffect::LensBlur { amount_pct: prunr_core::depth::LENS_BLUR_START_PCT }),
        None if s == "bw" => Ok(BgEffect::DesaturatedSource),
        Some(("blur", n)) => match n.parse::<u32>() {
            Ok(radius) if BgEffect::BLUR_RADII.contains(&radius) => Ok(BgEffect::BlurredSource { radius }),
            _ => Err(format!("'{n}' is not a blur radius from {} to {}", BgEffect::BLUR_RADII.start(), BgEffect::BLUR_RADII.end())),
        },
        Some(("lens", n)) => match n.parse::<u8>() {
            Ok(amount_pct @ 0..=100) => Ok(BgEffect::LensBlur { amount_pct }),
            _ => Err(format!("'{n}' is not a lens blur amount from 0 to 100")),
        },
        _ => Err(format!("'{s}' is not one of blur, blur:N, lens, lens:N, bw")),
    }
}

/// Depth in the CLI: one engine for the run, and each input measured
/// once whatever reads its map (its render, its backdrop, `--depth-map`).
#[derive(Default)]
struct CliDepth {
    measurer: prunr_core::depth::Measurer,
    /// With `--depth-map`: the maps backdrops measured and their photo's
    /// size (~0.7 MB each), until written.
    kept: Option<std::cell::RefCell<std::collections::HashMap<PathBuf, KeptDepth>>>,
    /// The last photo measured, which its render and then its backdrop read.
    last: std::cell::RefCell<Option<(PathBuf, std::sync::Arc<prunr_core::depth::DepthMap>)>>,
}

type KeptDepth = (std::sync::Arc<prunr_core::depth::DepthMap>, (u32, u32));

impl CliDepth {
    fn new(args: &Cli, s: &ItemSettings) -> Self {
        let focused = s.reads_auto_focus(args.settings_model().role());
        let measurer = if focused { prunr_core::depth::Measurer::default() } else { prunr_core::depth::Measurer::lines_only() };
        Self { measurer, kept: args.depth_map.then(Default::default), last: Default::default() }
    }

    /// The map of `photo`, read from `path`, for its render or backdrop;
    /// `faces`, when already found in the photo, are not looked for again.
    fn for_photo(
        &self,
        path: &std::path::Path,
        photo: &image::RgbaImage,
        faces: Option<&[prunr_core::face::FacePlace]>,
    ) -> Result<std::sync::Arc<prunr_core::depth::DepthMap>, CoreError> {
        if let Some(map) = self.last.borrow().as_ref().filter(|(seen, _)| seen == path).map(|(_, map)| std::sync::Arc::clone(map)) {
            return Ok(map);
        }
        let map = std::sync::Arc::new(match faces {
            Some(faces) => self.measurer.measure_with_faces(photo, faces)?,
            None => self.measurer.measure(photo)?,
        });
        *self.last.borrow_mut() = Some((path.to_path_buf(), std::sync::Arc::clone(&map)));
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
        let photo = prunr_core::load_image_from_path(input).map_err(|e| could_not(input, "read", e.message()))?;
        let map = self.measurer.measure_decoded(&photo).map_err(|e| format!("depth failed: {e}"))?;
        Ok((std::sync::Arc::new(map), (photo.width(), photo.height())))
    }
}

/// Where each input's depth map goes, beside its result.
fn depth_map_paths(args: &Cli) -> Vec<PathBuf> {
    args.result_paths().into_iter().zip(args.result_stems()).map(|(result, mut name)| {
        name.push(prunr_core::depth::MAP_FILE_SUFFIX);
        result.with_file_name(name)
    }).collect()
}

/// Write each input's depth map (`--depth-map`). Returns how many failed.
fn write_depth_maps(args: &Cli, depth: &CliDepth) -> usize {
    let mut failed = 0;
    for (input, out) in args.inputs.iter().zip(depth_map_paths(args)) {
        let written = check_overwrite(&out, args.force)
            .and_then(|()| depth.for_file(input))
            .and_then(|(map, (w, h))| prunr_core::depth::encode_depth_png(&map, w, h).map_err(|e| format!("encode failed: {e}")))
            .and_then(|bytes| prunr_app::fs_util::write_bytes_replacing(&out, &bytes).map_err(|e| format!("failed to save {}: {e}", out.display())));
        match written {
            Ok(()) if !args.quiet => eprintln!("Saved {}", out.display()),
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {}", prunr_core::plain::reason(e));
                failed += 1;
            }
        }
    }
    failed
}

/// The depth model is needed (a depth map, fading lines, a lens blur)
/// but not installed: the line to print.
fn missing_depth_model(args: &Cli, s: &ItemSettings) -> Option<String> {
    let id = prunr_core::depth::MODEL.into();
    (wants_depth(args, s) && !prunr_models::is_available(id)).then(|| prunr_models::not_installed_error(id))
}

/// Whether the run reads a depth map: a depth-map file, or what the app
/// reads one for (faded lines, a lens-blur backdrop).
fn wants_depth(args: &Cli, s: &ItemSettings) -> bool {
    args.depth_map || s.wants_depth(args.settings_model().role())
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

/// A colour flag's value: `None` when the flag is absent, `Err` with
/// `example` when it is not 6-digit hex.
fn flag_color(flag: &str, value: Option<&str>, example: &str) -> Result<Option<[u8; 3]>, String> {
    let Some(s) = value else { return Ok(None) };
    parse_hex_color(s).map(Some).ok_or_else(|| format!("{flag}: '{s}' is not a 6-digit hex color (e.g. {example})"))
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

    /// Whether this run opens the window, where a failure to start has
    /// no terminal to be read in: started from Finder, the Start menu or
    /// a launcher.
    pub fn opens_the_window(&self) -> bool {
        !self.worker && !self.doctor && !self.clear_ep_cache && self.command.is_none() && (self.open.is_some() || self.inputs.is_empty())
    }

    /// The app's model this run uses: the eraser, the upscaler, the
    /// colorizer or the background-removal model.
    fn settings_model(&self) -> SettingsModel {
        if self.inpaint {
            SettingsModel::Inpaint
        } else if let Some(upscaler) = self.upscale {
            upscaler.settings_model()
        } else if self.colorize {
            SettingsModel::ColorizeDdcolor
        } else {
            self.model.settings_model()
        }
    }

    /// `--preset`'s settings for the model in use, every flag given laid
    /// over them; why not, when the preset or a colour cannot be read.
    fn run_settings(&self) -> Result<RunSettings, String> {
        use prunr_core::{BgEffect, OutputScale};
        let faces = self.restore_faces.map(CliFaceModel::face_model);
        let (mut s, mut brush) = crate::gui::presets::command_line_settings(self.preset.as_deref(), self.settings_model(), faces)?;
        lay(&mut s.gamma, self.gamma);
        lay(&mut s.contrast, self.contrast);
        lay(&mut s.edge_shift, self.edge_shift.map(MaskSettings::edge_shift_for_grow));
        lay(&mut s.refine_edges, self.refine_edges);
        lay(&mut s.guided_radius, self.edge_width.map(|w| w.clamp(1, 64) as u8));
        lay(&mut s.refine_guide, self.refine_guide);
        lay(&mut s.feather, self.feather.map(|f| f.max(0.0)));
        lay(&mut s.effect, self.effect);
        if let Some((width, color)) = self.outline {
            (s.decoration.outline_width, s.decoration.outline_color) = (width, color);
        }
        lay(&mut s.decoration.shadow, self.shadow);

        let lines = if self.lines_after_bg { Some(LineMode::SubjectOutline) } else { self.lines.then_some(LineMode::EdgesOnly) };
        lay(&mut s.line_mode, lines);
        lay(&mut s.line_strength, self.line_strength);
        lay(&mut s.edge_thickness, self.line_width.map(|w| w.min(20) as u8));
        lay(&mut s.compose_mode, self.line_composition);
        lay(&mut s.edge_scale, self.line_scale);
        lay(&mut s.solid_line_color, flag_color("--line-color", self.line_color.as_deref(), "ff00aa")?.map(Some));
        lay(&mut s.line_fade_pct, self.line_fade);

        // A background given replaces the preset's, whatever its kind.
        if self.bg_effect.is_some() || self.bg_image.is_some() || self.bg_color.is_some() {
            (s.bg, s.bg_effect, s.bg_image_hash) = (None, BgEffect::None, None);
        }
        lay(&mut s.bg_effect, self.bg_effect);
        lay(&mut s.bg, flag_color("--bg-color", self.bg_color.as_deref(), "ffffff")?.map(|[r, g, b]| Some([r, g, b, 255])));
        lay(&mut s.bg_image_fit, self.bg_image_fit);

        lay(&mut s.output_scale, self.scale.map(|n| match n {
            2 => OutputScale::X2,
            3 => OutputScale::X3,
            _ => OutputScale::X4,
        }));
        lay(&mut s.pre_denoise, self.denoise);
        lay(&mut s.sharpen, self.sharpen.map(|v| v.clamp(-1.0, 1.0)));
        lay(&mut s.ai_blend, self.ai_blend.map(|v| v.clamp(0.0, 1.0)));
        lay(&mut s.color_match, self.color_match);
        lay(&mut s.faces.strength_pct, self.face_strength);
        if let Some(strength) = self.color_strength {
            s.set_color_strength(strength.clamp(0.0, prunr_core::colorize::STRENGTH_MAX));
        }
        lay(&mut brush.inpaint_fill_scale, self.fill_scale);
        s.fit_line_mode_to(self.settings_model());
        Ok(RunSettings { item: s, brush })
    }

    /// Where each input's result is written: `-o` names the file for one
    /// input and the folder for several, each named by its `result_stems`.
    fn result_paths(&self) -> Vec<PathBuf> {
        self.output_paths(&prunr_core::ExportFormat::Png.result_suffix())
    }

    /// `result_paths` with `suffix` in place of `.prunr.png`.
    fn output_paths(&self, suffix: &str) -> Vec<PathBuf> {
        let is_batch = self.inputs.len() > 1;
        self.inputs.iter().zip(self.result_stems())
            .map(|(input, stem)| output_path_with_suffix(input, &stem, &self.output, is_batch, suffix))
            .collect()
    }

    /// The stem each input's files are named by: its own, or `stem-2`,
    /// `stem-3`… when an earlier input's go into the same folder by a name
    /// this file system holds as the same.
    fn result_stems(&self) -> Vec<std::ffi::OsString> {
        use crate::gui::drag_export::TakenStems;
        let mut taken: std::collections::HashMap<PathBuf, TakenStems> = Default::default();
        self.inputs.iter().map(|input| {
            let beside = || input.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
            let folder = self.output.as_deref().unwrap_or_else(beside);
            // One folder however it is written: `a`, `./a`, a link to it.
            let folder = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
            taken.entry(folder).or_insert_with(TakenStems::as_this_file_system).take(input.file_stem().unwrap_or_default())
        }).collect()
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
    let run = match args.run_settings() {
        Ok(run) => run,
        Err(err) => {
            eprintln!("error: {err}");
            return 1;
        }
    };
    if let Some(err) = missing_depth_model(args, &run.item) {
        eprintln!("error: {err}");
        return 1;
    }
    let depth = CliDepth::new(args, &run.item);
    let code = if args.depth_only {
        0
    } else if args.inpaint {
        run_inpaint(args, &run.brush)
    } else if let Some(upscaler) = args.upscale {
        run_upscale(args, upscaler, &run.item)
    } else if args.colorize {
        run_colorize(args, &run.item)
    } else if args.settings_model().renders_photo_alone(run.item.line_mode) {
        run_whole_photo(args, &depth, &run.item)
    } else {
        // All processing uses subprocess isolation for OOM protection
        run_batch(args, &depth, &run.item)
    };
    if args.depth_map && write_depth_maps(args, &depth) > 0 {
        return 1;
    }
    code
}

/// One upscale as the app runs it: denoise, the model (on the processor
/// when the card runs out of memory), then `finish_upscale_run`.
struct UpscaleJob<'a> {
    engine: &'a prunr_core::OrtEngine,
    /// Built the first time the card runs out of memory.
    on_cpu: Option<prunr_core::OrtEngine>,
    model: prunr_models::ModelId,
    scale: prunr_core::OutputScale,
    refine: prunr_core::upscale::Refine,
    denoise: f32,
    /// Loaded once for the batch.
    faces: Option<CliFaces>,
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
        let (model, factor, engine, on_cpu) = (self.model, self.scale.factor(), self.engine, &mut self.on_cpu);
        let upscale = |engine: &prunr_core::OrtEngine| prunr_core::upscale::upscale_rgba_with_engine(&model_input, engine, model, factor, &progress, None);
        let raw = prunr_core::engine::retry_on_cpu(model, engine.active_provider(), &progress, || upscale(engine), || {
            let cpu = match on_cpu {
                Some(cpu) => cpu,
                None => on_cpu.insert(prunr_core::OrtEngine::for_model(model, prunr_core::batch::ort_intra_threads(1), true)?),
            };
            upscale(cpu)
        })
        .map_err(|e| format!("upscale failed: {e}"))?;
        finish_upscale_run(raw, &model_input, &self.refine, self.faces.as_mut())
    }
}

/// The model's output finished as the app finishes it: the faces found in
/// `raw` and restored, Refine, then the faces blended in.
fn finish_upscale_run(
    raw: image::RgbaImage,
    model_input: &image::RgbaImage,
    refine: &prunr_core::upscale::Refine,
    faces: Option<&mut CliFaces>,
) -> Result<image::RgbaImage, String> {
    use crate::gui::live_preview::{finish_upscale, FacedKey};
    let raw = std::sync::Arc::new(raw);
    let faces = faces
        .map(|faces| restore_faces(faces, &raw).map(|found| FacedKey { faces: std::sync::Arc::new(found), blend: faces.1 }))
        .transpose()?;
    let (shown, _) = finish_upscale(raw, model_input, None, refine, faces.as_ref());
    Ok(std::sync::Arc::unwrap_or_clone(shown))
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
        (args.contrast.is_some(), "--contrast"),
        (args.line_fade.is_some(), "--line-fade"),
    ]
    .into_iter()
    .filter_map(|(set, flag)| set.then_some(flag))
    .collect()
}

/// Upscale mode: enlarge each input with `upscaler`, Refine after it
/// (sharpen, AI blend, colour match) and, when asked, restore faces, as
/// the app does. The model loads once for the whole batch.
fn run_upscale(args: &Cli, upscaler: CliUpscaler, s: &ItemSettings) -> i32 {
    let scale = s.output_scale;
    let model = prunr_core::upscale::model_for_scale(upscaler.model_id(), scale, prunr_models::is_available(prunr_models::ModelId::RealEsrganX2Plus));
    let Ok(faces) = load_faces(args, s) else { return 1 };
    let Some(engine) = load_engine(model, args.cpu, "upscale") else { return 1 };
    let mut job = UpscaleJob { engine: &engine, on_cpu: None, model, scale, refine: s.upscale_refine(), denoise: s.pre_denoise, faces };
    run_each_input(args, &format!("_x{}.png", scale.factor()), "Upscaling", |img| job.run(img))
}

/// Colorize mode: colour each input with DDColor at `--color-strength`.
/// The model loads once for the whole batch.
fn run_colorize(args: &Cli, s: &ItemSettings) -> i32 {
    let Some(engine) = load_engine(crate::gui::settings::COLORIZER.into(), args.cpu, "colorize") else { return 1 };
    let strength = s.color_strength();
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
    let mut failed = 0;
    for (input, out_path) in args.inputs.iter().zip(args.output_paths(suffix)) {
        if let Err(e) = check_overwrite(&out_path, args.force) {
            eprintln!("error: {e}");
            failed += 1;
            continue;
        }
        if !args.quiet {
            eprintln!("{doing} {}...", input.display());
        }
        let result = load_image_from_path(input)
            .map_err(|e| could_not(input, "read", e.message()))
            .and_then(|img| run(img.into_rgba8()))
            .and_then(|out| prunr_app::fs_util::save_image_replacing(&out_path, &out).map_err(|e| could_not(&out_path, "save", e)));
        match result {
            Ok(()) if !args.quiet => eprintln!("Saved {}", out_path.display()),
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: {}", prunr_core::plain::reason(e));
                failed += 1;
            }
        }
    }
    i32::from(failed > 0)
}

/// Eraser mode: load image + mask, run LaMa, save inpainted result, with
/// the eraser's Expand region, Feather, fill scale, Sharpen and edge
/// colour from `brush`. Single-file only.
fn run_inpaint(args: &Cli, brush: &BrushSettings) -> i32 {
    if args.inputs.len() != 1 {
        eprintln!("error: --inpaint accepts exactly one input image (got {})", args.inputs.len());
        return 1;
    }
    let input = &args.inputs[0];
    let Some(mask_path) = args.mask.as_ref() else {
        eprintln!("error: --inpaint requires --mask <path>");
        return 1;
    };

    let out_path = output_path_with_suffix(input, input.file_stem().unwrap_or_default(), &args.output, false, "_erased.png");
    if let Err(e) = check_overwrite(&out_path, args.force) {
        eprintln!("error: {e}");
        return 1;
    }

    let img = match load_image_from_path(input) {
        Ok(i) => i.into_rgba8(),
        Err(e) => {
            eprintln!("error: {}", could_not(input, "read", e.message()));
            return 1;
        }
    };
    let mask = match load_image_from_path(mask_path) {
        Ok(i) => i.into_luma8(),
        Err(e) => {
            eprintln!("error: {}", could_not(mask_path, "read the mask", e.message()));
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
    let mask = fill_region(&mask, brush.inpaint_grow, brush.inpaint_feather);
    let raw = match prunr_core::inpaint::process_lama(&img, &mask, prunr_models::ModelId::LaMaFp32, args.cpu, brush.inpaint_fill_scale, &prunr_core::Progress::none()) {
        Ok(r) => r.rgba,
        Err(e) => {
            eprintln!("error: {}", could_not(input, "erase", e.message()));
            return 1;
        }
    };
    let band = prunr_core::inpaint_blend::Band::Uniform(brush.inpaint_feather);
    let result = finalize_inpaint(raw, &img, &mask, band, brush.inpaint_sharpen, brush.edge_colour(false));
    if let Err(e) = prunr_app::fs_util::save_image_replacing(&out_path, &result) {
        eprintln!("error: failed to save {}: {e}", out_path.display());
        return 1;
    }
    if !args.quiet {
        eprintln!("Saved {}", out_path.display());
    }
    0
}

/// Where an input's file `{stem}{suffix}` is written. Batch mode or -o is
/// a directory: into that directory. Single mode with -o file.png: that
/// path. No -o: beside the input.
fn output_path_with_suffix(
    input: &std::path::Path,
    stem: &std::ffi::OsStr,
    output: &Option<PathBuf>,
    is_batch: bool,
    suffix: &str,
) -> std::path::PathBuf {
    let mut name = stem.to_os_string();
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

/// The CLI's one error shape: "<path>: could not <what>: <reason>", the
/// reason in plain words.
fn could_not(path: &std::path::Path, what: &str, reason: impl std::fmt::Display) -> String {
    format!("{}: could not {what}: {}", path.display(), prunr_core::plain::reason(reason))
}

/// An image failed: its spinner says why, or stderr when none shows (the
/// output is not a terminal).
fn report_failure(spinner: Option<&ProgressBar>, input: &std::path::Path, what: &str, why: impl std::fmt::Display) {
    match spinner {
        Some(pb) => pb.finish_with_message(format!("X {} — {}", input.display(), prunr_core::plain::reason(why))),
        None => eprintln!("error: {}", could_not(input, what, why)),
    }
}

/// What a batch sends to the worker: each input's index and what the
/// worker reads.
struct Sendable {
    indices: Vec<usize>,
    paths: Vec<ToSend>,
    /// Inputs reported and left out: unreadable, or their output is there
    /// already without `--force`.
    refused: usize,
}

/// A batch input as the worker reads it: the file itself, or a copy shrunk
/// to `DOWNSCALE_TARGET` (`--large-image downscale`), made when its turn
/// comes and sent again after a worker crash.
#[derive(Debug, Clone, PartialEq)]
enum ToSend {
    File(PathBuf),
    Shrunk { from: PathBuf, to: PathBuf },
}

impl ToSend {
    /// The file to send, shrinking it first when that is still to do.
    fn path(&mut self, quiet: bool) -> Result<PathBuf, CoreError> {
        let shrunk = match self {
            ToSend::File(path) => return Ok(path.clone()),
            ToSend::Shrunk { from, to } => {
                std::fs::write(&*to, downscale_to_png(from, quiet)?)?;
                to.clone()
            }
        };
        *self = ToSend::File(shrunk.clone());
        Ok(shrunk)
    }
}

/// The inputs worth a run, checked by their header only: the subprocess
/// reads each file when its turn comes. An output that is there already
/// is refused here, before the model runs on its input.
fn sendable_inputs(args: &Cli, out_paths: &[PathBuf], spinners: &[Option<ProgressBar>]) -> Sendable {
    let mut sendable = Sendable { indices: Vec::new(), paths: Vec::new(), refused: 0 };
    for (idx, (input, out_path)) in args.inputs.iter().zip(out_paths).enumerate() {
        let spinner = spinners.get(idx).and_then(Option::as_ref);
        if let Err(msg) = check_overwrite(out_path, args.force) {
            report_skipped(spinner, input, &msg, args.quiet);
            sendable.refused += 1;
            continue;
        }
        match prunr_core::image_dimensions_from_path(input) {
            Some((w, h)) => {
                let shrink = args.large_image == LargeImagePolicy::Downscale && prunr_core::exceeds_large_image_limit(w, h);
                sendable.indices.push(idx);
                sendable.paths.push(if shrink {
                    let to = prunr_app::subprocess::protocol::IpcKind::CliDs
                        .path_for(prunr_app::subprocess::protocol::ipc_temp_dir(), idx as u64);
                    ToSend::Shrunk { from: input.clone(), to }
                } else {
                    ToSend::File(input.clone())
                });
            }
            None => {
                sendable.refused += 1;
                report_failure(spinner, input, "read", "not a valid image");
            }
        }
    }
    sendable
}

fn report_skipped(spinner: Option<&ProgressBar>, input: &std::path::Path, why: &str, quiet: bool) {
    match spinner {
        Some(pb) => pb.finish_with_message(format!("X {} — {why}", input.display())),
        None if !quiet => eprintln!("skipped '{}': {why}", input.display()),
        None => {}
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

fn downscale_to_png(path: &std::path::Path, quiet: bool) -> Result<Vec<u8>, CoreError> {
    encode_rgba_png(&read_for_run(path, true, quiet)?.into_rgba8())
}

/// `path`'s photo as a run reads it: with `downscale`, one over the limit
/// shrunk to `DOWNSCALE_TARGET`, said unless `quiet`.
fn read_for_run(path: &std::path::Path, downscale: bool, quiet: bool) -> Result<image::DynamicImage, CoreError> {
    let img = load_image_from_path(path)?;
    if !downscale || !prunr_core::exceeds_large_image_limit(img.width(), img.height()) {
        return Ok(img);
    }
    if !quiet {
        eprintln!(
            "note: '{}' is over {} px: shrinking it to {} px (--large-image downscale).",
            path.display(), prunr_core::LARGE_IMAGE_LIMIT, DOWNSCALE_TARGET
        );
    }
    Ok(downscale_image(img, DOWNSCALE_TARGET))
}

fn run_batch(args: &Cli, depth: &CliDepth, s: &ItemSettings) -> i32 {
    // Before the progress bars: an error printed inside their draw region
    // is overwritten by the next redraw.
    let Some(mut finish) = cli_finish(args, depth, s) else { return 1 };

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
            pb
        })
    }).collect();

    let start_times: Vec<Instant> = args.inputs.iter().map(|_| Instant::now()).collect();
    let out_paths = args.result_paths();
    let Sendable { indices: valid_indices, paths: mut to_send, refused: load_fail_count } = sendable_inputs(args, &out_paths, &spinners);

    // Lines alone build no cut-out engine: the model only names the run.
    let model = args.model.settings_model().recipe_kind();
    let spinners_arc = std::sync::Arc::new(spinners);
    let inputs_arc = std::sync::Arc::new(args.inputs.clone());
    let quiet = args.quiet;
    let mask = s.mask_settings();
    // A render fades the lines from its own depth map; the worker's
    // result is not kept, so it measures none.
    let fade_pct = if finish.render.is_some() { 0 } else { s.line_fade_pct };
    let lines = prunr_app::subprocess::manager::WorkerLines {
        mode: s.line_mode, edge: s.edge_settings(), fade_pct,
        scales: prunr_app::subprocess::protocol::InferredScales::Drawn,
    };
    let mut success_count = 0usize;
    let mut fail_count = load_fail_count;

    // Each result is written as it lands: a stopped batch keeps what it
    // finished, and one result is held at a time.
    let mut land = |batch_idx: usize, result: Result<prunr_core::ProcessResult, CoreError>| {
        let original_idx = valid_indices[batch_idx];
        let input = &args.inputs[original_idx];
        let elapsed = start_times[original_idx].elapsed();
        let out_path = &out_paths[original_idx];

        let spinner_opt = spinners_arc.get(original_idx).and_then(|o| o.as_ref());

        match result {
            Ok(pr) => {
                // Another input of this batch may have written it meanwhile.
                if let Err(msg) = check_overwrite(out_path, args.force) {
                    report_skipped(spinner_opt, input, &msg, args.quiet);
                    fail_count += 1;
                    return;
                }
                match save_png(out_path, &pr.rgba_image) {
                    Ok(()) => {
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
                            eprintln!("error: {}", could_not(out_path, "write", e));
                        }
                    }
                }
            }
            Err(e) => {
                fail_count += 1;
                report_failure(spinner_opt, input, "process", e.message());
            }
        }
    };
    run_batch_subprocess(
        &mut to_send, &valid_indices, model, args.jobs, mask, args.cpu,
        lines, &mut finish,
        &spinners_arc, &inputs_arc, quiet, &mut land,
    );

    if let Some(ref opb) = overall { opb.finish_and_clear(); }

    finish_run(args.quiet, success_count, fail_count)
}

/// Run batch processing via subprocess with auto-retry on OOM, each
/// input's result handed to `land` (by its batch index) as it comes.
/// Accepts file paths — bytes are read lazily by the subprocess.
// One call site: the recipe, runtime knobs and UI surface arrive together,
// and a struct would be a builder no other site reuses.
#[allow(clippy::too_many_arguments)]
fn run_batch_subprocess(
    to_send: &mut [ToSend],
    valid_indices: &[usize],
    model: ModelKind,
    jobs: Option<usize>,
    mask: MaskSettings,
    cpu_only: bool,
    lines: prunr_app::subprocess::manager::WorkerLines,
    finish: &mut CliFinish,
    spinners: &[Option<ProgressBar>],
    inputs: &[std::path::PathBuf],
    quiet: bool,
    land: &mut dyn FnMut(usize, Result<prunr_core::ProcessResult, CoreError>),
) {
    use prunr_app::subprocess::batch_rules::{is_stalled, low_free_memory, on_crash, room_to_send, OnCrash, HANG_TIMEOUT};
    use prunr_app::subprocess::manager::{Crash, SubprocessManager};
    use prunr_app::subprocess::protocol::SubprocessEvent;

    let mut landed = vec![false; to_send.len()];
    let mut give = |idx: usize, result| {
        landed[idx] = true;
        land(idx, result);
    };
    let mut pending: std::collections::VecDeque<usize> = (0..to_send.len()).collect();
    // Without -j, the app's default: the processor's until the worker
    // names a graphics card, so this process never probes for one.
    // Lines alone build no cut-out engine: the worker's plan holds the
    // line model's jobs to memory.
    let cuts = lines.mode.cuts_out();
    let default_on = |on_host_cpu| prunr_app::gui::memory::default_jobs(cuts.then_some(model), on_host_cpu, cpu_only);
    let mut max_jobs = jobs.unwrap_or_else(|| default_on(true));

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
        let (mut sub, provider) = match SubprocessManager::spawn(model, effective_jobs, mask, cpu_only, lines) {
            Ok(s) => s,
            Err(e) => {
                for &idx in &pending {
                    give(idx, Err(CoreError::Model(e.clone())));
                }
                break;
            }
        };
        // The worker names the accelerator it found, also under --cpu.
        let on_host_cpu = cpu_only || prunr_core::engine::model_on_host_cpu(model.model_id(), &provider);
        if jobs.is_none() && !on_host_cpu {
            max_jobs = max_jobs.min(default_on(false));
        }
        // Only the worker knows where it runs: this process never probes.
        // On the processor the run is held to the free memory, the
        // worker's own counted; a card short of memory falls back to the
        // processor under the same rule.
        if on_host_cpu {
            let line_fade = lines.fade_pct > 0;
            let engines = prunr_app::gui::memory::CutoutEngines { model, jobs: effective_jobs, cpu_only: true, first_look: false, line_mode: lines.mode, line_fade };
            let held_mb = sub.resident_mb().unwrap_or(0);
            match engines.plan(held_mb).map(|plan| plan.jobs) {
                Ok(fit) if fit < effective_jobs => {
                    sub.kill();
                    max_jobs = fit;
                    continue;
                }
                Ok(_) => {}
                Err(short) => {
                    sub.kill();
                    let (subject, lighter) = if cuts {
                        let lighter = engines.lighter_that_fit(held_mb).into_iter().filter_map(|m| m.to_model_kind()).filter_map(CliModel::flag_for).collect();
                        (CliModel::flag_for(model).unwrap_or_else(|| "the model".into()), lighter)
                    } else {
                        ("the line model".to_string(), Vec::new())
                    };
                    let why = prunr_app::gui::memory::refusal_text(&format!("not enough memory: {subject}"), short, &lighter, |flags| format!("use {flags}"));
                    for &idx in &pending {
                        give(idx, Err(CoreError::Model(why.clone())));
                    }
                    break;
                }
            }
        }

        let mut in_flight: Vec<usize> = Vec::new();
        // Finished by the worker, waiting for their render: it runs once the
        // worker has its next images, so the worker never waits on it.
        let mut to_render: Vec<ToRender> = Vec::new();
        let mut last_event_at = Instant::now();
        let crash = loop {
            while room_to_send(in_flight.len(), max_jobs, sub.should_pause_admission()) > 0 {
                let Some(idx) = pending.pop_front() else { break };
                let path = match to_send[idx].path(quiet) {
                    Ok(path) => path,
                    Err(e) => {
                        give(idx, Err(e));
                        continue;
                    }
                };
                match sub.send_image_path(valid_indices[idx] as u64, path) {
                    Ok(()) => {
                        // Each ticking spinner is a thread of its own: only
                        // the images being worked on tick, through the
                        // silence of a model run (the 4 Hz cap bounds redraws).
                        if let Some(Some(pb)) = spinners.get(valid_indices[idx]) {
                            pb.enable_steady_tick(std::time::Duration::from_millis(250));
                        }
                        in_flight.push(idx);
                    }
                    // A live worker that cannot take it would refuse it again;
                    // one a dead worker refused is the crash path's to rerun.
                    Err(e) if sub.is_alive() => give(idx, Err(CoreError::Model(e))),
                    Err(_) => {
                        pending.push_front(idx);
                        break;
                    }
                }
            }

            render_finished(&mut to_render, finish, inputs, &mut give);

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
                        let input = &inputs[item_id as usize];
                        let result = match finish.render.as_ref() {
                            Some(render) => {
                                // Read now: the files go, and the slot frees.
                                let _ = std::fs::remove_file(&result_path);
                                let seg = crate::gui::worker::read_raw_seg_tensor(tensor_cache.as_ref(), model);
                                let edges = crate::gui::worker::read_raw_edge_tensors(edge_cache.as_ref());
                                match render.lacks(seg.as_ref(), edges.as_ref()) {
                                    Some(why) => Err(CoreError::Model(why.into())),
                                    None => {
                                        if let Some(ridx) = orig_to_batch.get(&(item_id as usize)).copied() {
                                            in_flight.retain(|&i| i != ridx);
                                            to_render.push(ToRender { ridx, input: item_id as usize, seg, edges });
                                        }
                                        continue;
                                    }
                                }
                            }
                            None => {
                                // The tensors are for a render; without one
                                // their files go (in RAM on Linux).
                                for path in tensor_cache.map(|t| t.path).into_iter().chain(edge_cache.map(|e| e.path)) {
                                    let _ = std::fs::remove_file(path);
                                }
                                prunr_app::subprocess::ipc::read_result_rgba(&result_path, width, height)
                                    .ok_or_else(|| CoreError::Model("Failed to read subprocess result".into()))
                                    .map(|mut rgba_image| {
                                        finish.backdrop.apply(&mut rgba_image, input, None);
                                        prunr_core::ProcessResult { rgba_image, active_provider: String::new() }
                                    })
                            }
                        };
                        (item_id, result)
                    }
                    SubprocessEvent::ImageError { item_id, error } => (item_id, Err(CoreError::Model(error))),
                    _ => continue,
                };
                if let Some(ridx) = orig_to_batch.get(&(item_id as usize)).copied() {
                    // in_flight holds batch indices, each sent once a pass.
                    in_flight.retain(|&i| i != ridx);
                    give(ridx, result);
                }
            }

            if in_flight.is_empty() && pending.is_empty() && to_render.is_empty() {
                break None;
            }
            if let Some(crash) = sub.died() {
                break Some(crash);
            }
            if is_stalled(last_event_at.elapsed(), in_flight.len(), HANG_TIMEOUT) {
                sub.kill();
                break Some(Crash::Unresponsive);
            }
            if low_free_memory(in_flight.len(), prunr_app::gui::memory::cutout_watch_free).is_some() {
                break Some(Crash::MemoryLow);
            }
        };

        // What the worker finished before it stopped still renders.
        render_finished(&mut to_render, finish, inputs, &mut give);
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
                let advice = if matches!(crash, Crash::OutOfMemory | Crash::MemoryLow) { " \u{2014} try a smaller model" } else { "" };
                for idx in in_flight {
                    give(idx, Err(CoreError::Model(format!("worker {crash}{advice}"))));
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

    for idx in (0..landed.len()).filter(|&idx| !landed[idx]) {
        land(idx, Err(CoreError::Model("Not processed".into())));
    }
}

/// An image the worker finished, waiting for the app's render of it.
struct ToRender {
    ridx: usize,
    input: usize,
    seg: Option<crate::gui::worker::RawSegTensor>,
    edges: Option<crate::gui::worker::RawEdgeTensors>,
}

/// Render the images waiting in `to_render` and hand each result on.
fn render_finished(
    to_render: &mut Vec<ToRender>,
    finish: &mut CliFinish,
    inputs: &[std::path::PathBuf],
    give: &mut impl FnMut(usize, Result<prunr_core::ProcessResult, CoreError>),
) {
    let Some(render) = finish.render.as_mut() else { return };
    for ToRender { ridx, input, seg, edges } in to_render.drain(..) {
        let result = render.render(&finish.backdrop, &inputs[input], seg, edges)
            .map(|rgba_image| prunr_core::ProcessResult { rgba_image, active_provider: String::new() })
            .map_err(CoreError::Model);
        give(ridx, result);
    }
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
    let device = p.dgpu_device.map_or(String::new(), |id| format!(" (device {id:#06x})"));
    println!("dGPU:        {}{device}", p.dgpu.map_or("None".to_string(), |g| g.to_string()));
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
    if let Some(one) = &args.needs_one {
        return run_needs_one(one, args.image.as_deref(), args.tile);
    }
    if args.here {
        bench::here::run(&args.models, &mut |told| println!("{}", told.line()));
        return 0;
    }
    if let Some(name) = &args.check {
        let Some(id) = bench::model_named(name) else {
            eprintln!("error: no model named {name}");
            return 1;
        };
        bench::here::check(id, &mut |told| println!("{}", told.line()));
        return 0;
    }
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
    if args.needs {
        return run_needs(&models, args);
    }
    if args.table.is_some() {
        prunr_core::engine::remeasure_places();
    }
    let mut report = report::Report { header: report::Header::detect(&inputs.label, args.runs), rows: Vec::new(), running: None, places: None };
    print!("{}", report::header_text(&report.header));
    println!();
    let mut files = ReportFiles::new(args.out.clone());
    files.write(&report);
    let outputs = files.outputs_dir();
    bench::run(&models, &inputs, args.runs, &mut |event| match event {
        bench::Event::Stage { index, total, model, stage, .. } => {
            eprintln!("[{index}/{total}] {model}: {stage}");
            report.running = Some(format!("{model}: {stage}"));
            files.write(&report);
        }
        bench::Event::Done { mut row, pictures, .. } => {
            if let Some(pictures) = pictures {
                match bench::triage::save(&outputs, row.name, &pictures) {
                    Ok(saved) => row.saved = saved,
                    Err(e) => row.add_note(format!("the outputs could not be saved: {e}")),
                }
            }
            eprintln!("      {}: {}", row.model, report::result(&row));
            report.rows.push(*row);
            report.running = None;
            files.write(&report);
        }
    });
    println!();
    print!("{}", report::terminal_table(&report.rows));
    println!();
    println!("{}.", report::summary(&report.rows));
    if let Some(label) = &args.table {
        let column = report.header.gpu_providers.first().map_or("CPU", String::as_str);
        let section = bench::table::section(&report.rows, column, label, prunr_core::engine::column_for);
        print!("{section}");
        report.places = Some(section);
        files.write(&report);
    }
    println!("Report: {} and {}", files.md.display(), files.json().display());
    0
}

/// `prunr bench --needs`: each model measured in a child process, the
/// report written to `--out`.
fn run_needs(models: &[prunr_models::ModelId], args: &BenchArgs) -> i32 {
    use prunr_app::bench::{needs, report};
    let header = report::Header::detect("", 0);
    let machine = format!("{}, {}, {} RAM, {}", header.gpus.join(" + "), header.cpu, header.ram, header.os);
    let points = needs::measure_all(models, args.image.as_deref(), args.tile, &mut |p| {
        let side = p.side_label();
        match &p.error {
            Some(e) => eprintln!("{} ({side}): {e}", p.name),
            None => eprintln!("{} ({side}, {}): RAM {:?} MB, card {:?} MB", p.name, p.provider, p.host_mb, p.card_mb),
        }
    });
    let text = needs::report(&points, &machine);
    print!("{text}");
    match std::fs::write(&args.out, &text) {
        Ok(()) => println!("Report: {}", args.out.display()),
        Err(e) => eprintln!("error: cannot write {}: {e}", args.out.display()),
    }
    0
}

/// One `--needs` point, in this process, printed for the parent.
fn run_needs_one(one: &str, image: Option<&std::path::Path>, tile: Option<u32>) -> i32 {
    let Some((name, side)) = one.split_once(':') else {
        eprintln!("error: --needs-one takes MODEL:cpu or MODEL:gpu");
        return 1;
    };
    let Some(id) = prunr_app::bench::model_named(name) else {
        eprintln!("error: no model named {name}");
        return 1;
    };
    let photo = match image.map(prunr_core::load_image_from_path) {
        Some(Err(e)) => {
            eprintln!("error: cannot read the image: {e}");
            return 1;
        }
        Some(Ok(photo)) => Some(photo),
        None => None,
    };
    prunr_app::bench::needs::measure_here(id, side == "gpu", photo, tile);
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

    /// Beside the report: `prunr-bench.md` saves into `prunr-bench-outputs/`.
    fn outputs_dir(&self) -> PathBuf {
        output_path_with_suffix(&self.md, self.md.file_stem().unwrap_or_default(), &None, false, "-outputs")
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

    use super::SettingsModel;
    use crate::gui::presets::command_line_settings;

    fn cli(args: &[&str]) -> super::Cli {
        super::Cli::parse_from(["prunr", "a.png"].iter().chain(args).copied())
    }

    /// The image settings a run with `args` uses.
    fn settings(args: &[&str]) -> crate::gui::item_settings::ItemSettings {
        cli(args).run_settings().expect("settings").item
    }

    #[test]
    fn every_upscaler_names_a_model() {
        for upscaler in <super::CliUpscaler as clap::ValueEnum>::value_variants() {
            upscaler.model_id();
        }
    }

    /// `-m none` runs No model's settings; Subject lines, with no subject
    /// to outline, draw over the whole photo as in the app; no cut-out
    /// flag names it.
    #[test]
    fn no_model_takes_no_models_settings() {
        assert_eq!(cli(&["-m", "none"]).settings_model(), SettingsModel::None);
        assert_eq!(settings(&["-m", "none", "--lines-after-bg"]).line_mode, prunr_core::LineMode::EdgesOnly);
        assert_eq!(settings(&["--lines-after-bg"]).line_mode, prunr_core::LineMode::SubjectOutline, "a cut-out keeps them");
        let flags: Vec<String> = SettingsModel::ALL.into_iter().filter_map(SettingsModel::to_model_kind).filter_map(super::CliModel::flag_for).collect();
        assert_eq!(flags, ["--model silueta", "--model u2net", "--model birefnet-lite"]);
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--colorize", "--restore-faces", "gfpgan"]).is_err(), "Colorize restores no faces");
        assert_eq!(settings(&["-m", "none", "--restore-faces", "gfpgan"]).faces.wanted(), Some(prunr_models::ModelId::FaceGfpgan));
    }

    /// A No-model run saves the photo as the app renders it, without a
    /// worker: here the effect over the whole photo, from a flag and from
    /// a preset's No model entry.
    #[test]
    fn a_no_model_run_saves_the_apps_render_of_the_photo() {
        let dir = tempfile::tempdir().expect("temp dir");
        let input = dir.path().join("photo.png");
        let photo = image::RgbaImage::from_fn(12, 8, |x, y| image::Rgba([(x * 20) as u8, (y * 30) as u8, 90, 255]));
        photo.save(&input).expect("saved");
        let preset = dir.path().join("look.json");
        std::fs::write(&preset, r#"{ "format_version": 2, "models": { "NoModel": { "item_settings": { "fill_style": "sepia" } } } }"#).expect("written");
        for (effect, flags) in [
            (prunr_core::SubjectEffect::Desaturate, vec!["--effect", "desaturate"]),
            (prunr_core::SubjectEffect::Sepia, vec!["--preset", preset.to_str().expect("utf-8")]),
        ] {
            let out = dir.path().join("out.png");
            let mut args = vec!["prunr", input.to_str().expect("utf-8"), "-m", "none", "-q", "-f", "-o", out.to_str().expect("utf-8")];
            args.extend(flags);
            assert_eq!(super::run_remove(&super::Cli::parse_from(args)), 0, "{effect:?}");
            let mut want = photo.clone();
            prunr_core::apply_subject_effect(&mut want, effect);
            assert_eq!(image::open(&out).expect("the result").to_rgba8(), want, "{effect:?}");
        }
    }

    /// `--large-image downscale` holds on a run the command line renders
    /// itself, as on the worker's: a photo over the limit is shrunk.
    #[test]
    fn a_rendered_run_keeps_the_large_image_policy() {
        let dir = tempfile::tempdir().expect("temp dir");
        let input = dir.path().join("wide.png");
        image::RgbaImage::from_pixel(prunr_core::LARGE_IMAGE_LIMIT + 1, 4, image::Rgba([90, 120, 150, 255])).save(&input).expect("saved");
        let out = dir.path().join("out.png");
        let args = ["prunr", input.to_str().expect("utf-8"), "-m", "none", "-q", "-f", "--large-image", "downscale", "-o", out.to_str().expect("utf-8")];
        assert_eq!(super::run_remove(&super::Cli::parse_from(args)), 0);
        assert_eq!(image::image_dimensions(&out).expect("the result").0, prunr_core::DOWNSCALE_TARGET);
    }

    /// The worker's answer is rendered only with the tensors its run
    /// made: a cut-out without its mask, or lines without theirs, is a
    /// failure, not the photo saved uncut.
    #[test]
    fn a_render_needs_the_tensors_its_run_made() {
        use crate::gui::knob_catalog::ModelRole;
        use prunr_core::LineMode;
        let depth = super::CliDepth::default();
        let render = |role, line_mode| super::CliRender {
            settings: crate::gui::item_settings::ItemSettings { line_mode, ..Default::default() },
            role, faces: None, depth: &depth, downscale: false, quiet: true,
        };
        let seg = crate::gui::worker::RawSegTensor { data: vec![0.0], height: 1, width: 1, model: prunr_core::ModelKind::Silueta };
        let edges = crate::gui::worker::RawEdgeTensors { tensors: Default::default(), height: 1, width: 1, spare: None };
        assert!(render(ModelRole::Cutout, LineMode::Off).lacks(None, None).is_some(), "a cut-out without its mask");
        assert!(render(ModelRole::Cutout, LineMode::Off).lacks(Some(&seg), None).is_none());
        assert!(render(ModelRole::Cutout, LineMode::SubjectOutline).lacks(Some(&seg), None).is_some(), "lines without theirs");
        assert!(render(ModelRole::Filters, LineMode::EdgesOnly).lacks(None, Some(&edges)).is_none(), "No model's lines need no mask");
    }

    /// Without flags or a preset a run takes Prunr's settings for its
    /// model, as the app does: the eraser's Sharpen, for one, is the
    /// app's 0.6, not the 0 the command line had.
    #[test]
    fn without_flags_a_run_takes_prunrs_settings() {
        for (args, model) in [
            (&[][..], SettingsModel::BiRefNetLite),
            (&["-m", "silueta"][..], SettingsModel::Silueta),
            (&["--upscale", "nomos2"][..], SettingsModel::Nomos2Upscale),
            (&["--colorize"][..], SettingsModel::ColorizeDdcolor),
            (&["--inpaint", "--mask", "m.png"][..], SettingsModel::Inpaint),
        ] {
            let run = cli(args).run_settings().unwrap();
            assert_eq!((run.item, run.brush), command_line_settings(None, model, None).unwrap(), "{args:?}");
        }
        let eraser = cli(&["--inpaint", "--mask", "m.png"]).run_settings().unwrap().brush;
        assert_eq!(eraser.inpaint_sharpen, 0.6);
    }

    /// `--preset` is the starting point; a flag given overrides it, and a
    /// background given replaces the preset's, whatever its kind.
    #[test]
    fn a_flag_overrides_the_preset() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("look.json");
        std::fs::write(&file, r#"{ "format_version": 2, "models": { "BiRefNetLite": { "item_settings": {
            "gamma": 1.5, "feather": 2.0, "refine_edges": true, "bg_effect": { "blurred_source": { "radius": 20 } }
        } } } }"#).unwrap();
        let preset = file.to_str().unwrap();
        let s = settings(&["--preset", preset, "--gamma", "0.8"]);
        assert_eq!((s.gamma, s.feather, s.refine_edges), (0.8, 2.0, true));
        assert_eq!(s.bg_effect, prunr_core::BgEffect::BlurredSource { radius: 20 }, "the preset's background");
        assert!(!settings(&["--preset", preset, "--refine-edges=false"]).refine_edges);
        let red = settings(&["--preset", preset, "--bg-color", "ff0000"]);
        assert_eq!(red.backdrop(), crate::gui::item_settings::Backdrop::Color([255, 0, 0]));
        assert_eq!(settings(&["-m", "silueta", "--preset", preset]).gamma, 1.0, "the preset has no entry for Silueta");
    }

    /// An on/off flag followed by an input still takes the input as one:
    /// `--refine-edges photo.jpg` worked before the flags took values.
    #[test]
    fn an_on_off_flag_does_not_swallow_the_input_after_it() {
        let cli = super::Cli::try_parse_from(["prunr", "--refine-edges", "photo.jpg"]).expect("parses");
        assert_eq!((cli.refine_edges, cli.inputs.len()), (Some(true), 1));
        let cli = super::Cli::try_parse_from(["prunr", "--upscale", "nomos", "--color-match", "in.jpg"]).expect("parses");
        assert_eq!((cli.color_match, cli.inputs.len()), (Some(true), 1));
        assert_eq!(super::Cli::parse_from(["prunr", "--refine-edges=false", "a.png"]).refine_edges, Some(false));
    }

    /// `--restore-faces` runs the restorer with its own settings from the
    /// preset, as picking it in the app does.
    #[test]
    fn a_face_restorer_takes_its_own_settings_from_the_preset() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("faces.json");
        std::fs::write(&file, r#"{ "format_version": 2, "models": {
            "FaceGfpgan": { "item_settings": { "faces.strength_pct": 70 } },
            "FaceRestoreFormerPp": { "item_settings": { "faces.strength_pct": 35 } }
        } }"#).unwrap();
        let preset = file.to_str().unwrap();
        let faces = |restorer: &str| settings(&["--upscale", "nomos", "--preset", preset, "--restore-faces", restorer]).faces;
        assert_eq!((faces("restoreformer").on, faces("restoreformer").strength_pct), (true, 35));
        assert_eq!(faces("gfpgan").strength_pct, 70);
        assert_eq!(settings(&["--upscale", "nomos", "--preset", preset, "--restore-faces", "gfpgan", "--face-strength", "20"]).faces.strength_pct, 20, "a flag still wins");
    }

    /// Without faces an upscale saves the model's output after Refine,
    /// pixel for pixel; at Prunr's settings, as it came.
    #[test]
    fn an_upscale_without_faces_is_the_models_output_refined() {
        use std::sync::Arc;
        let raw = image::RgbaImage::from_fn(16, 12, |x, y| image::Rgba([(x * 15) as u8, (y * 20) as u8, ((x * y) % 255) as u8, 255]));
        let input = image::RgbaImage::from_fn(8, 6, |x, y| image::Rgba([(x * 30) as u8, (y * 40) as u8, 90, 255]));
        let knobs = |flags: &[&str]| settings(&[&["--upscale", "realesrgan"][..], flags].concat()).upscale_refine();
        let on = knobs(&["--sharpen", "0.5", "--ai-blend", "0.7", "--color-match"]);
        let (refined, _) = prunr_core::upscale::refine(&Arc::new(raw.clone()), &input, None, &on);
        assert_ne!(*refined, raw, "Refine changes it");
        assert_eq!(super::finish_upscale_run(raw.clone(), &input, &on, None), Ok(Arc::unwrap_or_clone(refined)));
        assert_eq!(super::finish_upscale_run(raw.clone(), &input, &knobs(&[]), None), Ok(raw), "Prunr's settings change nothing");
    }

    /// With Restore faces an upscale's faces are found in the model's
    /// output and blended in after Refine, as the app finishes it: on the
    /// portrait `PRUNR_FACE_PHOTO` names, taken as the model's output.
    #[test]
    fn an_upscale_restores_the_faces_of_the_models_output_as_the_app_does() {
        let Some(path) = std::env::var_os("PRUNR_FACE_PHOTO") else {
            eprintln!("[upscale faces] SKIP: set PRUNR_FACE_PHOTO to a portrait");
            return;
        };
        if let Err(why) = prunr_core::ort_runtime::runtime_for_tests() {
            eprintln!("[upscale faces] SKIP: {why}");
            return;
        }
        let flags = |restorer| ["--upscale", "realesrgan", "--cpu", "--restore-faces", restorer];
        let Some(flags) = ["gfpgan", "restoreformer"].map(flags).into_iter().find(|f| settings(f).faces.wanted().is_some_and(prunr_models::is_available)) else {
            eprintln!("[upscale faces] SKIP: no face model installed");
            return;
        };
        let s = settings(&flags);
        let mut faces = super::load_faces(&cli(&flags), &s).ok().flatten().expect("the face model loads");
        let raw = prunr_core::load_image_from_path(std::path::Path::new(&path)).expect("the portrait loads").to_rgba8();
        let input = image::imageops::resize(&raw, raw.width() / 2, raw.height() / 2, image::imageops::FilterType::Triangle);
        let knobs = s.upscale_refine();

        let found = super::restore_faces(&mut faces, &raw).expect("restored");
        assert!(!found.is_empty(), "a face in the portrait");
        let key = crate::gui::live_preview::FacedKey { faces: std::sync::Arc::new(found), blend: s.faces.blend() };
        let (apps, _) = crate::gui::live_preview::finish_upscale(std::sync::Arc::new(raw.clone()), &input, None, &knobs, Some(&key));
        assert!(super::finish_upscale_run(raw, &input, &knobs, Some(&mut faces)).expect("finished") == *apps, "the app's pixels");
    }

    #[test]
    fn an_unknown_preset_is_refused_with_the_presets_there_are() {
        let err = cli(&["--preset", "No such look 7f3a"]).run_settings().err().unwrap();
        assert!(err.contains("no preset named") && err.contains("Prunr"), "{err}");
        assert!(cli(&["--preset", "prunr"]).run_settings().is_ok(), "Prunr's own, any case");
    }

    #[test]
    fn a_colour_flag_is_absent_hex_or_refused() {
        assert_eq!(super::flag_color("--bg-color", None, "ffffff"), Ok(None));
        assert_eq!(super::flag_color("--bg-color", Some("#ff00aa"), "ffffff"), Ok(Some([255, 0, 170])));
        assert!(super::flag_color("--bg-color", Some("ff00aa "), "ffffff").is_err());
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

    /// A window run tells a failure to start in a dialog; a printing run
    /// prints it and a worker's parent reports it.
    #[test]
    fn only_a_window_run_shows_a_failure_to_start() {
        let opens = |args: &[&str]| super::Cli::parse_from(["prunr"].iter().chain(args)).opens_the_window();
        assert!(opens(&[]));
        assert!(opens(&["--debug"]));
        assert!(opens(&["--open", "photo.jpg"]));
        for printing in [&["photo.jpg"][..], &["--doctor"], &["--clear-ep-cache"], &["bench"], &["--worker"]] {
            assert!(!opens(printing), "{printing:?}");
        }
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

    /// A batch input whose output is there already is refused before it
    /// is sent, not after the model ran on it; `--force` sends it.
    #[test]
    fn an_existing_output_is_refused_before_its_run() {
        let dir = tempfile::tempdir().unwrap();
        let (kept, fresh) = (dir.path().join("kept.png"), dir.path().join("fresh.png"));
        for path in [&kept, &fresh] {
            image::RgbaImage::new(4, 4).save(path).unwrap();
        }
        let out = dir.path().join("out");
        std::fs::create_dir(&out).unwrap();
        let cli = |force: &[&str]| super::Cli::parse_from(["prunr".as_ref(), kept.as_os_str(), fresh.as_os_str(), "-o".as_ref(), out.as_os_str(), "-q".as_ref()].into_iter().chain(force.iter().map(std::ffi::OsStr::new)));
        let sendable_inputs = |args: super::Cli| super::sendable_inputs(&args, &args.result_paths(), &[]);
        std::fs::write(&cli(&[]).result_paths()[0], b"an earlier result").unwrap();
        let sendable = sendable_inputs(cli(&[]));
        assert_eq!((sendable.paths, sendable.indices, sendable.refused), (vec![super::ToSend::File(fresh.clone())], vec![1], 1));
        let forced = sendable_inputs(cli(&["-f"]));
        assert_eq!((forced.paths, forced.refused), (vec![super::ToSend::File(kept.clone()), super::ToSend::File(fresh.clone())], 0));
    }

    /// A photo over the limit is kept at its size unless `--large-image
    /// downscale` asks; then it is shrunk when its turn comes, not before
    /// the first one starts, and a resend after a crash reuses the copy.
    #[test]
    fn a_large_photo_is_shrunk_when_it_is_sent() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.png");
        image::RgbaImage::new(prunr_core::LARGE_IMAGE_LIMIT + 1, 8).save(&big).unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir(&out).unwrap();
        let cli = |policy: &[&str]| super::Cli::parse_from(["prunr".as_ref(), big.as_os_str(), "-o".as_ref(), out.as_os_str(), "-q".as_ref()].into_iter().chain(policy.iter().map(std::ffi::OsStr::new)));
        let sendable_inputs = |args: super::Cli| super::sendable_inputs(&args, &args.result_paths(), &[]);
        assert_eq!(sendable_inputs(cli(&[])).paths, vec![super::ToSend::File(big.clone())], "kept at its size by default");

        let mut sent = sendable_inputs(cli(&["--large-image", "downscale"])).paths;
        let super::ToSend::Shrunk { to, .. } = sent[0].clone() else { panic!("marked to shrink: {sent:?}") };
        assert!(!to.exists(), "nothing is shrunk before its turn");
        let first = sent[0].path(true).unwrap();
        assert_eq!(first, to);
        assert_eq!(prunr_core::image_dimensions_from_path(&first).map(|(w, _)| w), Some(super::DOWNSCALE_TARGET));
        assert_eq!(sent[0], super::ToSend::File(to.clone()), "a resend reuses the copy");
        std::fs::remove_file(&to).unwrap();
    }

    /// No job slots means nothing is ever sent, and the batch waits forever.
    #[test]
    fn zero_jobs_is_refused() {
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "-j", "0"]).is_err());
        assert_eq!(super::Cli::parse_from(["prunr", "a.png", "-j", "3"]).jobs, Some(3));
        assert_eq!(super::Cli::parse_from(["prunr", "a.png", "--cpu"]).jobs, None, "no -j: the app's default");
    }

    /// `-o` with one input is the output file; with several it is the
    /// folder each result goes into.
    #[test]
    fn dash_o_is_the_file_for_one_input_and_the_folder_for_several() {
        use std::path::Path;
        let one = super::Cli::parse_from(["prunr", "in/a.jpg", "-o", "out/result.png"]);
        assert_eq!(one.result_paths(), [Path::new("out/result.png")]);
        let many = super::Cli::parse_from(["prunr", "in/a.jpg", "in/b.jpg", "-o", "out"]);
        assert_eq!(many.result_paths(), [Path::new("out/a.prunr.png"), Path::new("out/b.prunr.png")]);
        let none = super::Cli::parse_from(["prunr", "in/a.jpg"]);
        assert_eq!(none.result_paths(), [Path::new("in/a.prunr.png")]);
    }

    /// A result takes `stem-2` only when an earlier one of the run takes
    /// its name in its folder, ignoring case: not across folders, and its
    /// depth map and every mode's file follow it.
    #[test]
    fn a_result_is_renamed_only_when_its_name_is_taken_in_its_folder() {
        use std::path::{Path, PathBuf};
        let args = |args: &[&str]| super::Cli::parse_from(["prunr"].iter().chain(args).copied());
        let paths = |a: &[&str]| args(a).result_paths();
        // Windows and macOS see `a` and `A` as one name; Linux as two files.
        let second = if cfg!(any(windows, target_os = "macos")) { "out/A-2.prunr.png" } else { "out/A.prunr.png" };
        assert_eq!(paths(&["x/a.jpg", "y/A.png", "y/b.png", "-o", "out"]), ["out/a.prunr.png", second, "out/b.prunr.png"].map(PathBuf::from));
        assert_eq!(paths(&["x/a.jpg", "y/a.png", "-o", "out"]), ["out/a.prunr.png", "out/a-2.prunr.png"].map(PathBuf::from));
        assert_eq!(paths(&["x/a.jpg", "y/a.png"]), ["x/a.prunr.png", "y/a.prunr.png"].map(PathBuf::from), "beside inputs in two folders");
        assert_eq!(paths(&["a.jpg", "./a.png"]), ["a.prunr.png", "./a-2.prunr.png"].map(PathBuf::from), "one folder however it is written");
        let both = args(&["x/a.jpg", "x/a.png", "--depth-map"]);
        assert_eq!(both.output_paths("_x4.png"), ["x/a_x4.png", "x/a-2_x4.png"].map(PathBuf::from));
        assert_eq!(super::depth_map_paths(&both), ["x/a-depth.png", "x/a-2-depth.png"].map(PathBuf::from));

        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("a")).expect("folder");
        let (plain, roundabout) = (dir.path().join("a/IMG.png"), dir.path().join("a/../a/IMG.jpg"));
        let named = super::Cli::parse_from([Path::new("prunr"), plain.as_path(), roundabout.as_path()]).result_paths();
        assert_eq!(named[1].file_name(), Some("IMG-2.prunr.png".as_ref()));
    }

    /// Inputs whose results would share a folder and a name each keep
    /// their own: the later ones as `stem-2`, `stem-3`, as the app names
    /// them. Beside their inputs, same-named photos of two folders keep
    /// their names.
    #[test]
    fn inputs_of_one_name_save_a_result_each() {
        use std::ffi::OsStr;
        let dir = tempfile::tempdir().expect("temp dir");
        let (a, b, out) = (dir.path().join("a"), dir.path().join("b"), dir.path().join("out"));
        for folder in [&a, &b] {
            std::fs::create_dir(folder).expect("folder");
        }
        let inputs = [a.join("IMG.png"), a.join("IMG.bmp"), b.join("IMG.png")];
        for (input, shade) in inputs.iter().zip([40u8, 120, 200]) {
            image::RgbaImage::from_pixel(6, 4, image::Rgba([shade, shade, shade, 255])).save(input).expect("saved");
        }
        let run = |output: &[&OsStr]| {
            let flags = ["-m", "none", "--effect", "desaturate", "-q", "-f"].map(OsStr::new);
            let args = [OsStr::new("prunr")].into_iter().chain(inputs.iter().map(|p| p.as_os_str())).chain(flags).chain(output.iter().copied());
            super::run_remove(&super::Cli::parse_from(args))
        };
        let shade = |path: std::path::PathBuf| image::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).to_rgba8().get_pixel(0, 0).0[0];

        assert_eq!(run(&[OsStr::new("-o"), out.as_os_str()]), 0);
        assert_eq!(["IMG.prunr.png", "IMG-2.prunr.png", "IMG-3.prunr.png"].map(|name| shade(out.join(name))), [40, 120, 200]);

        assert_eq!(run(&[]), 0);
        assert_eq!([a.join("IMG.prunr.png"), a.join("IMG-2.prunr.png"), b.join("IMG.prunr.png")].map(shade), [40, 120, 200]);
    }

    #[test]
    fn outline_and_shadow_flags_reach_the_mask_settings() {
        let mask = |args: &[&str]| settings(args).mask_settings();
        let d = mask(&["--outline", "8:000000", "--shadow", "drop"]).decoration;
        assert_eq!((d.outline_width, d.outline_color, d.shadow), (8.0, [0, 0, 0], prunr_core::ShadowKind::Drop));
        let white = mask(&["--outline", "4"]).decoration;
        assert_eq!(white.outline_color, [255, 255, 255]);
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--outline", "120"]).is_err());
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--shadow", "hard"]).is_err());
        for spelling in ["color", "colour"] {
            assert_eq!(mask(&["--refine-edges", "--refine-guide", spelling]).refine_guide, prunr_core::RefineGuide::Colour);
        }
        assert_eq!(mask(&[]).refine_guide, prunr_core::RefineGuide::Brightness);
    }

    /// An `-o` name without an extension saves a PNG, whatever the mode.
    #[test]
    fn an_output_name_without_an_extension_is_a_png_file() {
        use super::output_path_with_suffix;
        use std::path::PathBuf;
        let input = std::path::Path::new("/photos/a.jpg");
        let stem = std::ffi::OsStr::new("a");
        let out = |o: &str| output_path_with_suffix(input, stem, &Some(PathBuf::from(o)), false, "_x.png");
        assert_eq!(out("/tmp/result"), PathBuf::from("/tmp/result.png"));
        assert_eq!(out("/tmp/result.webp"), PathBuf::from("/tmp/result.webp"));
        assert_eq!(output_path_with_suffix(input, stem, &Some(PathBuf::from("/tmp/dir")), true, "_x.png"), PathBuf::from("/tmp/dir/a_x.png"));
        assert_eq!(output_path_with_suffix(input, stem, &None, false, "_x.png"), PathBuf::from("/photos/a_x.png"));
    }

    /// `--line-fade` reads the depth model only when lines are drawn;
    /// on its own it does nothing and must not refuse the run.
    #[test]
    fn line_fade_needs_depth_only_with_lines() {
        let wants = |args: &[&str]| super::wants_depth(&cli(args), &settings(args));
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
        assert_eq!((cli.scale, cli.sharpen, cli.face_strength), (Some(3), Some(-0.5), Some(70)));
        let s = cli.run_settings().unwrap().item;
        assert_eq!((s.output_scale, s.sharpen), (prunr_core::OutputScale::X3, -0.5));
        assert_eq!((s.faces.wanted(), s.faces.strength_pct), (Some(prunr_models::ModelId::FaceRestoreFormerPp), 70));
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
        assert_eq!(cli.run_settings().unwrap().item.color_strength(), 0.6);
        assert_eq!(settings(&["--colorize"]).color_strength(), 1.0);
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

    /// docs/CLI.md names every option `--help` shows, `bench`'s too, with
    /// the values each takes; it and the app's reference name no option
    /// the command line lacks.
    #[test]
    fn the_cli_reference_names_every_option_and_no_other() {
        use clap::CommandFactory;
        use std::collections::BTreeSet;
        let doc = include_str!("../../../docs/CLI.md");
        let named = |text: &'static str| -> BTreeSet<&'static str> {
            text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                .filter_map(|word| word.strip_prefix("--"))
                .filter(|flag| flag.starts_with(|c: char| c.is_ascii_lowercase()))
                .collect()
        };
        let mut cmd = super::Cli::command();
        cmd.build();
        let args: Vec<clap::Arg> = std::iter::once(&cmd)
            .chain(cmd.get_subcommands())
            .flat_map(|c| c.get_arguments().cloned())
            .collect();
        let longs = |arg: &clap::Arg| -> Vec<String> {
            arg.get_long().into_iter().chain(arg.get_visible_aliases().into_iter().flatten()).map(str::to_owned).collect()
        };
        let known: BTreeSet<String> = args.iter().flat_map(longs).collect();
        let shown = args.iter().filter(|a| !a.is_hide_set());

        let in_doc = named(doc);
        let missing: Vec<String> = shown.clone().flat_map(longs).filter(|long| !in_doc.contains(long.as_str())).collect();
        assert!(missing.is_empty(), "missing from docs/CLI.md: {missing:?}");
        let values_missing: Vec<String> = shown
            .flat_map(|a| a.get_possible_values().into_iter().filter(|v| !v.is_hide_set()).map(|v| v.get_name().to_owned()))
            .filter(|value| !["true", "false"].contains(&value.as_str()) && !doc.contains(&format!("`{value}`")))
            .collect();
        assert!(values_missing.is_empty(), "values missing from docs/CLI.md: {values_missing:?}");
        for (reference, text) in [("docs/CLI.md", doc), ("the app's reference", include_str!("gui/views/cli_help.rs"))] {
            let unknown: Vec<&str> = named(text).into_iter().filter(|flag| !known.contains(*flag)).collect();
            assert!(unknown.is_empty(), "{reference} names options the command line lacks: {unknown:?}");
        }
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
        assert_eq!(super::depth_map_paths(&args), [std::path::PathBuf::from("/out/cat-depth.png")]);
        let args = super::Cli::parse_from(["prunr", "/p/cat.jpg", "--depth-map", "--depth-only"]);
        assert_eq!(super::depth_map_paths(&args), [std::path::PathBuf::from("/p/cat-depth.png")]);
        assert!(super::Cli::try_parse_from(["prunr", "/p/cat.jpg", "--depth-only"]).is_err(), "--depth-only needs --depth-map");
    }

    /// A stem that is not UTF-8 names its result and depth map as it is:
    /// a mended name would let two inputs write one file. Two such stems
    /// are one name only byte for byte.
    #[cfg(unix)]
    #[test]
    fn a_name_that_is_not_utf8_is_kept() {
        use clap::Parser;
        use std::os::unix::ffi::OsStrExt;
        let os = |bytes: &[u8]| std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes));
        let input = os(b"/p/caf\xe9.jpg");
        let args = super::Cli::parse_from([std::ffi::OsStr::new("prunr"), input.as_os_str(), std::ffi::OsStr::new("--depth-map")]);
        assert_eq!(args.result_paths(), [os(b"/p/caf\xe9.prunr.png")]);
        assert_eq!(super::depth_map_paths(&args), [os(b"/p/caf\xe9-depth.png")]);
        let several = super::Cli::parse_from([os(b"prunr"), input.clone(), os(b"/p/caf\xe8.jpg"), os(b"/p/caf\xe9.png")]);
        assert_eq!(several.result_paths(), [os(b"/p/caf\xe9.prunr.png"), os(b"/p/caf\xe8.prunr.png"), os(b"/p/caf\xe9-2.prunr.png")]);
    }

    /// `--depth-map` writes the map a backdrop measured instead of
    /// measuring the input again, and keeps it no longer.
    #[test]
    fn a_depth_map_a_backdrop_measured_is_written_without_measuring_again() {
        use clap::Parser;
        let args = super::Cli::parse_from(["prunr", "/missing/cat.jpg", "--depth-map"]);
        let depth = super::CliDepth::new(&args, &crate::gui::item_settings::ItemSettings::default());
        let map = std::sync::Arc::new(prunr_core::depth::DepthMap::from_raw(1, 2, vec![0.0, 1.0]));
        let input = std::path::Path::new("/missing/cat.jpg");
        depth.kept.as_ref().expect("kept with --depth-map").borrow_mut().insert(input.into(), (std::sync::Arc::clone(&map), (4, 8)));
        let (got, dims) = depth.for_file(input).expect("the kept map");
        assert!(std::sync::Arc::ptr_eq(&got, &map) && dims == (4, 8));
        assert!(depth.for_file(input).is_err(), "taken once; the missing file is read the next time");
        assert!(super::CliDepth::new(&super::Cli::parse_from(["prunr", "/p/cat.jpg"]), &crate::gui::item_settings::ItemSettings::default()).kept.is_none(), "nothing kept without --depth-map");
    }

    /// With --large-image downscale, a worker run's effect background is
    /// made from the photo shrunk as the run's own copy was: the same
    /// background a rendered run makes from the photo it holds.
    #[test]
    fn a_backdrop_is_made_from_the_photo_the_run_shrank() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.png");
        let w = prunr_core::LARGE_IMAGE_LIMIT + 1;
        image::RgbaImage::from_fn(w, 4, |x, _| if x < w / 2 { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 0, 255, 255]) }).save(&path).unwrap();
        let shrunk = super::read_for_run(&path, true, true).unwrap().into_rgba8();
        let depth = super::CliDepth::default();
        let backdrop = super::CliBackdrop::Effect(prunr_core::BgEffect::BlurredSource { radius: 16 }, &depth, true);
        let mut from_file = image::RgbaImage::new(shrunk.width(), shrunk.height());
        let mut from_held = from_file.clone();
        backdrop.apply(&mut from_file, &path, None);
        backdrop.apply(&mut from_held, &path, Some(&shrunk));
        assert!(from_file == from_held, "the worker's background differs from the rendered run's");
    }

    /// The effect is made from the input photo and fills what the result
    /// leaves transparent; the result's own pixels stay.
    #[test]
    fn a_bg_effect_fills_the_transparent_part_from_the_photo() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("photo.png");
        image::RgbaImage::from_pixel(4, 4, image::Rgba([200, 50, 50, 255])).save(&photo).unwrap();
        let mut result = image::RgbaImage::from_fn(4, 4, |x, _| if x == 0 { image::Rgba([0, 0, 255, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        super::CliBackdrop::Effect(prunr_core::BgEffect::DesaturatedSource, &super::CliDepth::default(), false).apply(&mut result, &photo, None);
        assert_eq!(result.get_pixel(0, 0).0, [0, 0, 255, 255]);
        let p = result.get_pixel(3, 3).0;
        assert!(p[3] == 255 && p[0] == p[1] && p[1] == p[2], "grey photo behind: {p:?}");
    }

    #[test]
    fn an_edge_shift_past_the_slider_s_reach_is_refused() {
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--edge-shift", "20"]).is_ok());
        assert!(super::Cli::try_parse_from(["prunr", "a.png", "--edge-shift", "-20.5"]).is_err());
    }

    /// `--edge-shift` reads like the GUI's Shift edge: + grows the subject,
    /// which the pipeline stores as a negative (dilating) shift.
    #[test]
    fn a_positive_edge_shift_grows_the_subject() {
        assert_eq!(settings(&["--edge-shift", "3"]).mask_settings().edge_shift, -3.0);
        assert_eq!(settings(&["--edge-shift", "-2"]).mask_settings().edge_shift, 2.0);
    }

    /// The Sketch flags: --line-detail (and its old name), --line-width
    /// and --line-composition.
    #[test]
    fn sketch_flags_parse() {
        let a = settings(&["--line-detail", "0.8", "--line-width", "3", "--line-composition", "ghost"]);
        assert_eq!((a.line_strength, a.edge_thickness, a.compose_mode), (0.8, 3, prunr_core::ComposeMode::Ghost));
        assert_eq!(settings(&["--line-strength", "0.2"]).line_strength, 0.2);
        assert!(super::Cli::try_parse_from(["prunr", "--line-composition", "inverse", "x.png"]).is_err());
    }

    /// Scripts written for the retired Balanced scale keep working.
    #[test]
    fn line_scale_balanced_reads_as_fused() {
        assert_eq!(settings(&["--line-scale", "balanced"]).edge_scale, prunr_core::EdgeScale::Fused);
        assert_eq!(settings(&["--line-scale", "fine"]).edge_scale, prunr_core::EdgeScale::Fine);
    }

    /// Every Mask panel knob has a flag: Feather and Edge width reach the
    /// pipeline, defaulting to Prunr's for the model.
    #[test]
    fn feather_and_edge_width_reach_the_mask() {
        let m = settings(&["--feather", "1.5", "--edge-width", "12"]).mask_settings();
        assert_eq!((m.feather, m.guided_radius), (1.5, 12));
        let d = settings(&[]).mask_settings();
        let prunr = command_line_settings(None, SettingsModel::BiRefNetLite, None).unwrap().0.mask_settings();
        assert_eq!((d.feather, d.guided_radius), (prunr.feather, prunr.guided_radius));
    }
}
