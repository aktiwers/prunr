// Release builds embed models as zstd blobs; `--features dev-models` reads
// them from disk so model changes do not recompile this crate.
//
// This crate depends on no other workspace crate: prunr-app -> prunr-core ->
// prunr-models, never in reverse.

use std::sync::Arc;

// ── Registry types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelId {
    Silueta,
    U2net,
    BiRefNetLite,
    Teed,
    LaMaFp32,
    BigLaMa,
    Migan,
    /// Stable Diffusion 1.5 Inpainting (FP16). Multi-part bundle:
    /// UNet + VAE encode + VAE decode + text encoder. ~2 GB total.
    SdV15InpaintFp16,
    /// LCM-distilled SD 1.5 Inpainting (FP16). Same UNet architecture as
    /// `SdV15InpaintFp16` but trained to converge in ~4 steps instead of
    /// 20 — ~5× faster on CPU/iGPU at slight quality cost. Selected when
    /// the user picks the LCM scheduler on the SD chip and the bundle
    /// is installed.
    /// Bundle is the output of `scripts/export_lcm_inpaint.py`.
    SdV15LcmInpaintFp16,
    /// TAESD (Tiny AutoEncoder for SD) FP16, its decoder alone (~2.6 MB):
    /// stands in for SD 1.5's VAE decoder, ~3× faster at a slight quality
    /// cost. Used when installed and the Fast decoder switch says so; left
    /// on auto, it follows the LCM scheduler.
    ///
    /// Bundle is the output of `scripts/export_taesd.py`.
    TaesdFp16,
    RealEsrganX4Plus,
    Nomos8kSchatL,
    /// Real-ESRGAN x2plus (RRDB, fp32). OnDemand; runs for a 2x
    /// Real-ESRGAN upscale when installed.
    RealEsrganX2Plus,
    /// 4x NMKD Siax-CX (RRDB-23, fp32). OnDemand. Community-trained
    /// universal upscaler for clean / mildly-compressed photos.
    /// WTFPL — no attribution required.
    FourXNmkdSiaxCx,
    /// 4x NMKD Superscale (RRDB-23, fp32). OnDemand. Community-trained
    /// restoration upscaler — handles noise and JPEG artifacts on real
    /// photos. WTFPL — no attribution required.
    FourXNmkdSuperscale,
    /// 4xNomos2 RealPLKSR with DySample (fp32). MultiPartOnDemand, cut
    /// at its norms. Fast photo upscaler trained on JPEG-degraded photos.
    /// CC-BY-4.0.
    FourXNomos2RealPlksr,
    /// SAM 2.1 Hiera Small. Two-part bundled pair (encoder + decoder
    /// ONNX). Apache 2.0 from Meta. Used by Magic Brush.
    Sam21HieraSmall,
    /// GFPGAN v1.4 face restorer. OnDemand bundle; Apache-2.0. Faces are
    /// found by the bundled `YuNet`.
    FaceGfpgan,
    /// RestoreFormer++ face restorer. Same input and output as
    /// `FaceGfpgan`; Apache-2.0.
    FaceRestoreFormerPp,
    /// YuNet face detector (OpenCV Zoo). Bundled; finds the faces Restore
    /// faces redraws and Auto focus aims at. MIT.
    YuNet,
    /// RORem: SDXL inpainting fine-tuned for object removal, with its
    /// 4-step LCM LoRA fused in and the fixed removal prompt baked into
    /// the UNet (no text encoder). Runs through the SD pipeline.
    RoremFp16,
    /// DDColor paper-tiny colorizer (fp32). OnDemand; predicts the
    /// colour of a black-and-white photo. Apache-2.0.
    DdcolorTiny,
    /// Depth Anything V2 Small (fp32). OnDemand; predicts how near each
    /// part of a photo is. Apache-2.0.
    DepthAnythingV2Small,
}

impl ModelId {
    /// True for inpaint models that route through the dedicated SD
    /// subprocess. LaMa / Big-LaMa / MI-GAN have small footprints and
    /// stay in-process.
    pub fn is_sd_family(&self) -> bool {
        self.sd_spec().is_some()
    }

    /// What the SD pipeline needs to know about an SD-family bundle;
    /// `None` for every other model.
    pub fn sd_spec(&self) -> Option<SdSpec> {
        const SD15: SdSpec = SdSpec { latent: SdLatent::Sd15, cpu_output_correct: false, lean_sessions: false, tall_crop_card_extra_mb: None };
        match self {
            // RTX 3060: the card's peak for the tall 512x768 crop, 10.5 GB
            // (25 steps) and 9.4 GB (LCM, 8 steps at CFG 2).
            ModelId::SdV15InpaintFp16 => Some(SdSpec { tall_crop_card_extra_mb: Some(3_350), ..SD15 }),
            ModelId::SdV15LcmInpaintFp16 => Some(SdSpec { tall_crop_card_extra_mb: Some(2_600), ..SD15 }),
            // Its tall region fits its 1024 px tile: the same 8.6 GB.
            ModelId::RoremFp16 => Some(SdSpec { latent: SdLatent::Sdxl, cpu_output_correct: true, lean_sessions: true, tall_crop_card_extra_mb: Some(0) }),
            _ => None,
        }
    }

    /// Every variant. A test's exhaustive `match` fails the build until a
    /// new variant is added here too.
    pub const ALL: &'static [ModelId] = &[
        ModelId::Silueta,
        ModelId::U2net,
        ModelId::BiRefNetLite,
        ModelId::Teed,
        ModelId::LaMaFp32,
        ModelId::BigLaMa,
        ModelId::Migan,
        ModelId::SdV15InpaintFp16,
        ModelId::SdV15LcmInpaintFp16,
        ModelId::TaesdFp16,
        ModelId::RealEsrganX4Plus,
        ModelId::Nomos8kSchatL,
        ModelId::RealEsrganX2Plus,
        ModelId::FourXNmkdSiaxCx,
        ModelId::FourXNmkdSuperscale,
        ModelId::FourXNomos2RealPlksr,
        ModelId::Sam21HieraSmall,
        ModelId::FaceGfpgan,
        ModelId::FaceRestoreFormerPp,
        ModelId::YuNet,
        ModelId::RoremFp16,
        ModelId::DdcolorTiny,
        ModelId::DepthAnythingV2Small,
    ];

    /// Persistent key (cache directory names, config fields, IPC payloads).
    /// **Do not change existing strings**: they are stamped into user data
    /// on disk, and renaming silently invalidates every cache entry.
    pub fn stable_name(&self) -> &'static str {
        match self {
            ModelId::Silueta => "silueta",
            ModelId::U2net => "u2net",
            ModelId::BiRefNetLite => "birefnet_lite",
            ModelId::Teed => "teed",
            ModelId::LaMaFp32 => "lama_fp32",
            ModelId::BigLaMa => "big_lama",
            ModelId::Migan => "migan",
            ModelId::SdV15InpaintFp16 => "sd_v15_inpaint_fp16",
            ModelId::SdV15LcmInpaintFp16 => "sd_v15_lcm_inpaint_fp16",
            ModelId::TaesdFp16 => "taesd_fp16",
            ModelId::RealEsrganX4Plus => "real_esrgan_x4plus",
            ModelId::Nomos8kSchatL => "nomos8k_schat_l",
            ModelId::RealEsrganX2Plus => "real_esrgan_x2plus",
            ModelId::FourXNmkdSiaxCx => "four_x_nmkd_siax_cx",
            ModelId::FourXNmkdSuperscale => "four_x_nmkd_superscale",
            ModelId::FourXNomos2RealPlksr => "four_x_nomos2_realplksr",
            ModelId::Sam21HieraSmall => "sam2_1_hiera_small",
            ModelId::FaceGfpgan => "face_gfpgan",
            ModelId::FaceRestoreFormerPp => "face_restoreformer_pp",
            ModelId::YuNet => "yunet",
            ModelId::RoremFp16 => "rorem_fp16",
            ModelId::DdcolorTiny => "ddcolor_tiny",
            ModelId::DepthAnythingV2Small => "depth_anything_v2_small",
        }
    }

}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Segmentation,
    EdgeDetection,
    Inpaint,
    Upscale,
    /// Magic Brush — authors a selection mask rather than a final
    /// segmentation. Routes through the SAM dispatch, not the
    /// BG-removal pipeline. Encoder + decoder ship as separate ONNX
    /// files (`MultiPartBundled`).
    Selection,
    /// Face restoration: a detector finds faces, a restorer redraws each
    /// one. Runs after the photo's own model, never in its place.
    Face,
    /// Colorize: predicts colour for a black-and-white photo.
    Colorize,
    /// Depth: how near each part of a photo is, for depth effects.
    Depth,
}

/// Hardware requirement for a model. Drives Model Store + dropdown
/// gating: `Required` greys out entries on CPU-only hardware,
/// `Recommended` shows a warning but allows opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuRequirement {
    /// Runs acceptably on CPU. Default.
    #[default]
    None,
    /// Works on CPU but slow (5-10× GPU). Show "Slow on CPU" badge.
    Recommended,
    /// Not viable on CPU (seconds-to-minutes per call). Disable
    /// download + selection on CPU-only hardware.
    Required,
}

/// Provenance + license metadata shared between OnDemand and
/// MultiPartOnDemand sources. Single source of truth for the strings
/// the Model Store displays and the consent dialog reads.
#[derive(Debug, Clone, Copy)]
pub struct LicenseInfo {
    pub author: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub source_url: &'static str,
}

/// Optional fp16 (or other optimized) companion download for an
/// `OnDemand` entry. When `Some`, the fetcher pulls this alongside the
/// main fp32 file, placing both as siblings in the user data dir; a GPU
/// session then tries the fp16 sibling first.
#[derive(Debug, Clone, Copy)]
pub struct OnDemandVariant {
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_mb: u32,
}

impl OnDemandVariant {
    pub fn size_bytes(self) -> u64 {
        u64::from(self.size_mb) * 1024 * 1024
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ModelSource {
    /// Embedded in the binary.
    Bundled { license: LicenseInfo },
    OnDemand {
        filename: &'static str,
        url: &'static str,
        sha256: &'static str,
        size_mb: u32,
        license: LicenseInfo,
        /// Optional fp16 companion. Downloaded as a sibling alongside
        /// the main file when present. Auto-selected at session-build
        /// time on GPU EPs that support fp16.
        fp16: Option<OnDemandVariant>,
    },
    /// A model composed of multiple ONNX files, downloaded as a unit.
    /// Used by pipelines like Stable Diffusion (UNet + VAE encode +
    /// VAE decode + text encoder). Contrast with `OnDemand` which is
    /// a single file resolved through `resolve_bytes`.
    MultiPartOnDemand {
        /// Subdirectory under `on_demand_dir()` where parts live —
        /// also acts as the install marker for `is_available`.
        subdir: &'static str,
        parts: &'static [ModelPart],
        license: LicenseInfo,
        /// Restrictive licenses (CreativeML Open RAIL-M, NVIDIA SCL, …)
        /// require explicit user acceptance before the download starts.
        license_acceptance_required: bool,
    },
    /// Multi-part model whose ONNX files are zstd-compressed and embedded
    /// in the binary via `include_bytes!`. Used by SAM 2 (encoder +
    /// decoder). Each part unpacks on demand, shared while anyone holds
    /// it (see `shared_bundled`). The license is carried through (unlike
    /// single-file `Bundled`) so the Settings credits row still displays
    /// the upstream author + license.
    MultiPartBundled {
        parts: &'static [BundledPart],
        license: LicenseInfo,
    },
}

impl ModelSource {
    /// Short label for diagnostics + dropdown badges.
    pub fn kind_label(&self) -> &'static str {
        match self {
            ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => "Bundled",
            ModelSource::OnDemand { .. } => "OnDemand",
            ModelSource::MultiPartOnDemand { .. } => "MultiPart",
        }
    }

    /// Embedded in the binary: nothing to download or accept.
    pub fn is_bundled(&self) -> bool {
        matches!(self, ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. })
    }
}

/// One component of a `MultiPartBundled` model. The bytes are resolved
/// by `bundled_multipart_bytes(id, key)` — under release builds via
/// `include_bytes!` + zstd decompress, under `dev-models` via an
/// on-disk read. Adding a new bundled multipart model means
/// adding a row here AND an arm in `bundled_multipart_bytes`.
#[derive(Debug, Clone, Copy)]
pub struct BundledPart {
    /// Logical key for inference dispatch, e.g. `"encoder"` / `"decoder"`.
    pub key: &'static str,
    /// Source ONNX filename. Descriptive only: the byte loaders name
    /// their files themselves (`include_bytes!` takes a literal).
    pub filename: &'static str,
}

/// One component of a `MultiPartOnDemand` bundle. SHA256 verified
/// per-part after download; all parts must succeed before the bundle
/// is considered installed.
#[derive(Debug, Clone, Copy)]
pub struct ModelPart {
    /// Logical key for inference dispatch, e.g. `"unet"` / `"vae_encoder"`.
    pub key: &'static str,
    /// Filename relative to the bundle's subdir.
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
}

/// What one loaded model needs while it runs a typical input (load
/// peaks included), on each accelerator. One field per accelerator and no
/// `Default`: a new accelerator does not build until every model prices
/// it. Size-dependent parts (upscale tiles, Nomos2's ways, SD's tall crop)
/// are added by their own tested formulas on top of these.
///
/// Err high: under-counting crashes the GUI with an OOM; over-counting
/// only throttles admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Needs {
    pub cpu: Cost,
    /// The CPU on macOS, where ORT's Arm build and the footprint macOS
    /// counts (compressed memory in it) differ: BiRefNet-lite peaks at
    /// 12.8 GB there against 7.1 GB on Linux. Unmeasured: `cpu`'s figure.
    pub mac_cpu: Priced,
    pub cuda: Priced,
    pub directml: Priced,
    pub coreml: Priced,
    pub openvino: Priced,
    pub migraphx: Priced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cost {
    /// System RAM, MB.
    pub host_mb: u32,
    /// Memory on the card, MB; 0 on the CPU and on unified memory
    /// (CoreML), where everything is in `host_mb`.
    pub card_mb: u32,
    pub basis: Basis,
}

/// Where a figure comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Measured: the machine, the date and the settings.
    Measured(&'static str),
    /// Not measured: why the figure is what it is.
    Estimate(&'static str),
}

/// A model's needs on one accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priced {
    Same(Cost),
    /// Not measured there yet: the CPU's figure is charged to system RAM,
    /// as every accelerator was before card memory was counted.
    Unmeasured,
}

impl Cost {
    pub const fn measured(host_mb: u32, card_mb: u32, on: &'static str) -> Self {
        Self { host_mb, card_mb, basis: Basis::Measured(on) }
    }

    pub const fn estimate(host_mb: u32, card_mb: u32, why: &'static str) -> Self {
        Self { host_mb, card_mb, basis: Basis::Estimate(why) }
    }
}

impl Needs {
    /// Its needs on the provider ORT names `provider` ("CPU", "CUDA", …);
    /// an accelerator this table does not know is priced as unmeasured.
    pub fn on(&self, provider: &str) -> Priced {
        match provider {
            "CPU" => Priced::Same(self.host_cpu()),
            "CUDA" => self.cuda,
            "DirectML" => self.directml,
            "CoreML" => self.coreml,
            "OpenVINO" => self.openvino,
            "MIGraphX" => self.migraphx,
            _ => Priced::Unmeasured,
        }
    }

    /// What it needs on this machine's CPU: the Mac's figure on macOS
    /// where measured, else `cpu`.
    pub fn host_cpu(&self) -> Cost {
        match self.mac_cpu {
            Priced::Same(cost) if cfg!(target_os = "macos") => cost,
            _ => self.cpu,
        }
    }

    /// Every accelerator's entry with the provider name ORT gives it.
    pub fn accelerators(&self) -> [(&'static str, Priced); 5] {
        [
            ("CUDA", self.cuda),
            ("DirectML", self.directml),
            ("CoreML", self.coreml),
            ("OpenVINO", self.openvino),
            ("MIGraphX", self.migraphx),
        ]
    }
}

/// Where a model runs on each accelerator, from `prunr bench --table` on
/// the hardware each entry names: where its result is the CPU's, then
/// where it is faster, and within 10% of that where it takes less system
/// memory. A model spells out each accelerator it was measured on and
/// fills the rest from `UNMEASURED`; `every_model_says_where_it_runs`
/// counts what is left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Places {
    pub cuda: Place,
    pub directml: Place,
    pub coreml: Place,
    pub migraphx: Place,
    /// OpenVINO on the processor, as it runs unless the Intel graphics
    /// are on.
    pub openvino: Place,
    /// OpenVINO with the Intel graphics on.
    pub openvino_gpu: Place,
}

/// Where a model runs on one accelerator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Place {
    /// On it: its result is the CPU's and it runs faster, or as fast in
    /// less memory. The measurement and where it was taken.
    There(Speed),
    /// On the CPU instead.
    Cpu(Why),
    /// Not measured there: it runs there when its session builds.
    Unmeasured,
}

/// Why a model runs on the CPU rather than an accelerator, with the
/// measurement and where it was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// It fails to build or run there, or is unsafe to try: it hangs or
    /// takes the machine's memory. Never tried again, even by the bench.
    Fails(&'static str),
    /// Its result there is not the CPU's.
    Differs(&'static str),
    /// It runs slower there.
    Slower(&'static str),
    /// Within 10% of the CPU's speed, and the CPU takes less memory.
    NoGain(&'static str),
}

impl Place {
    /// Whether a session skips the accelerator for the CPU. While places
    /// are measured again (`remeasuring`) only a failing one is skipped:
    /// some fail by hanging or by taking the machine's memory.
    pub fn kept_off(self, remeasuring: bool) -> bool {
        match self {
            Place::Cpu(_) if self.fails() => true,
            Place::Cpu(_) => !remeasuring,
            Place::There(_) | Place::Unmeasured => false,
        }
    }

    /// The model does not run on the accelerator.
    pub fn fails(self) -> bool {
        matches!(self, Place::Cpu(Why::Fails(_)))
    }

    /// The model runs on the accelerator, faster than on the CPU.
    pub fn runs_there(self) -> bool {
        matches!(self, Place::There(_))
    }
}

/// A model's steady time on an accelerator against the CPU's, measured
/// side by side: what tells two accelerators that both run it apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speed {
    /// Milliseconds a run takes there.
    pub ms: f32,
    /// Milliseconds a run takes on the CPU, in the same measurement.
    pub cpu_ms: f32,
    /// What the times leave out ("first runs"), or why a place this close
    /// to the CPU's keeps it; empty when nothing.
    pub caveat: &'static str,
    /// Where and when it was measured.
    pub on: &'static str,
}

impl Speed {
    pub const fn on(ms: f32, cpu_ms: f32, on: &'static str) -> Self {
        Self { ms, cpu_ms, caveat: "", on }
    }

    pub const fn but(self, caveat: &'static str) -> Self {
        Self { caveat, ..self }
    }
}

impl std::fmt::Display for Speed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&times_text(self.ms, self.cpu_ms))?;
        if !self.caveat.is_empty() {
            write!(f, ", {}", self.caveat)?;
        }
        write!(f, " ({})", self.on)
    }
}

/// "1.20x: 440 ms against the CPU's 525 ms".
pub fn times_text(ms: f32, cpu_ms: f32) -> String {
    format!("{:.2}x: {} against the CPU's {}", cpu_ms / ms, fmt_ms(f64::from(ms)), fmt_ms(f64::from(cpu_ms)))
}

/// A duration for people: "3.8 ms", "440 ms", "47.11 s", "2m 06s".
pub fn fmt_ms(ms: f64) -> String {
    if ms < 10.0 {
        format!("{ms:.1} ms")
    } else if ms < 1000.0 {
        format!("{ms:.0} ms")
    } else if ms < 60_000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else {
        let secs = (ms / 1000.0).round() as u64;
        format!("{}m {:02}s", secs / 60, secs % 60)
    }
}

impl Why {
    pub fn text(self) -> &'static str {
        match self {
            Self::Fails(text) | Self::Differs(text) | Self::Slower(text) | Self::NoGain(text) => text,
        }
    }
}

impl Places {
    /// Not measured anywhere: a test descriptor's, or a model's before its
    /// first bench (a count in `every_model_says_where_it_runs` holds it).
    pub const UNMEASURED: Self = Self {
        cuda: Place::Unmeasured,
        directml: Place::Unmeasured,
        coreml: Place::Unmeasured,
        migraphx: Place::Unmeasured,
        openvino: Place::Unmeasured,
        openvino_gpu: Place::Unmeasured,
    };

    pub fn get(&self, column: Column) -> Place {
        match column {
            Column::Cuda => self.cuda,
            Column::DirectMl => self.directml,
            Column::CoreMl => self.coreml,
            Column::MiGraphX => self.migraphx,
            Column::OpenVino => self.openvino,
            Column::OpenVinoGpu => self.openvino_gpu,
        }
    }

    pub fn set(&mut self, column: Column, place: Place) {
        *match column {
            Column::Cuda => &mut self.cuda,
            Column::DirectMl => &mut self.directml,
            Column::CoreMl => &mut self.coreml,
            Column::MiGraphX => &mut self.migraphx,
            Column::OpenVino => &mut self.openvino,
            Column::OpenVinoGpu => &mut self.openvino_gpu,
        } = place;
    }
}

/// One accelerator's entry in `Places`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Column {
    Cuda,
    DirectMl,
    CoreMl,
    MiGraphX,
    OpenVino,
    OpenVinoGpu,
}

impl Column {
    pub const ALL: [Self; 6] = [Self::Cuda, Self::DirectMl, Self::CoreMl, Self::MiGraphX, Self::OpenVino, Self::OpenVinoGpu];

    /// Its place in `ALL`.
    pub fn index(self) -> usize {
        match self {
            Self::Cuda => 0,
            Self::DirectMl => 1,
            Self::CoreMl => 2,
            Self::MiGraphX => 3,
            Self::OpenVino => 4,
            Self::OpenVinoGpu => 5,
        }
    }

    /// The provider ORT names.
    pub fn provider(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::DirectMl => "DirectML",
            Self::CoreMl => "CoreML",
            Self::MiGraphX => "MIGraphX",
            Self::OpenVino | Self::OpenVinoGpu => "OpenVINO",
        }
    }

    /// The name the provider list and the bench give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenVinoGpu => "OpenVINO (Intel graphics on)",
            other => other.provider(),
        }
    }

    /// Its field in `Places`, as `prunr bench --table` writes it.
    pub fn field(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::DirectMl => "directml",
            Self::CoreMl => "coreml",
            Self::MiGraphX => "migraphx",
            Self::OpenVino => "openvino",
            Self::OpenVinoGpu => "openvino_gpu",
        }
    }

    /// The provider's own column (ORT's name, any case); OpenVINO's is the
    /// processor's.
    pub fn of(provider: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.provider().eq_ignore_ascii_case(provider))
    }

    /// The column called `name` (`Column::name`).
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }
}

const SILUETA_ON_OPENVINO: Why = Why::Fails(
    "OpenVINO 2025.4.1 rejects its graph as not acyclic; the FP16 copy fails a quantized-cast type check (2026-04-26)",
);

const ROREM_ON_OPENVINO: Why = Why::Fails(
    "OpenVINO widens the 5 GB fp16 UNet to fp32 to compile it: killed for memory with 19 GB free (i7-6700, 32 GB)",
);

const BIREFNET_ON_OPENVINO: Why = Why::Fails(
    "a run fails: with ORT 1.24's constant folding off (it frees weights still in use), OpenVINO's EP matches an \
     output to the wrong name (\"Output names mismatch\"; fixed upstream in microsoft/onnxruntime#29882, after \
     1.24.1). With folding on it runs, 1.1x slower than the CPU (i7-6700 and HD 530, OpenVINO 1.24.1, 2026-10-26)",
);

const MIGAN_ON_OPENVINO: Why = Why::Fails(
    "its session does not build: OpenVINO's CPU plugin has no implementation for one of its nodes (\"No suitable \
     implementations found\") (i7-6700 and HD 530, OpenVINO 1.24.1, 2026-10-25; OpenVINO's plugin 1.8.1 on the \
     processor too, 2026-10-27)",
);

const SAM_ON_OPENVINO: Why = Why::Fails(
    "its encoder does not build: OpenVINO's CPU plugin cannot reshape to a dynamic rank (/image_encoder/trunk/Reshape_2) \
     (i7-6700 and HD 530, OpenVINO 1.24.1, 2026-10-25)",
);

const DDCOLOR_ON_OPENVINO: Why = Why::Fails(
    "a run fails in one of OpenVINO's subgraphs (subgraph 155) (i7-6700 and HD 530, OpenVINO 1.24.1, 2026-10-25)",
);

const NOMOS2_ON_OPENVINO: Why = Why::Fails(
    "compiling its 30 parts takes 73 s and 9.7 GB before a 55 s run, against the CPU's 67 s in 0.95 GB (960×635, \
     i7-6700, 2026-10)",
);

const FILL_ON_INTEL_GRAPHICS: Why = Why::Fails(
    "fills run on OpenVINO's processor: its GPU plugin cannot build LaMa's dynamic shapes, and under AUTO it ran about \
     10% slower (i7-6700, HD 530)",
);

/// The figure the registry carried as one number per model before card
/// memory was counted: measured or estimated per the comment beside it.
const CARRIED: &str = "the registry's earlier single figure; not re-measured since";

#[derive(Debug, Clone, Copy)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub display_name: &'static str,
    pub description: &'static str,
    pub category: ModelCategory,
    pub source: ModelSource,
    pub version: &'static str,
    pub gpu: GpuRequirement,
    /// Where it runs on each accelerator. Read before the failures this
    /// machine recorded, so a first-time user skips a load known to fail.
    pub places: Places,
    /// What the model needs on each accelerator: no default, so a new
    /// model does not build without its numbers.
    pub needs: Needs,
    /// For a segmentation model: whether a batch on the host CPU runs
    /// faster with an engine per job. One run of a heavy model already
    /// fills every core, so a second engine only doubles RAM (BiRefNet-lite:
    /// 4 × 12 MP in 39.7 s with one engine, 42.3 s with two); a light
    /// model's run leaves room to overlap the next image's decode and
    /// encode (Silueta 27 %, U2Net 18 % faster with four, on a 4-core i7).
    /// `false` for every other model.
    pub parallel_jobs_pay_on_cpu: bool,
    /// Window-size multiple constraint for window-attention transformers
    /// (HAT, Swin). Both H and W must be multiples of this at inference
    /// time — the upscale tiler pads bottom-right and trims output.
    /// `None` = no constraint, any tile dimension works.
    pub tile_size_multiple: Option<u32>,
    /// Default tile size for tile-based dispatch (upscale today).
    /// Hardware-aware downsizing may reduce this when RAM is tight.
    /// `None` = caller picks; only required for tile-based models.
    pub recommended_tile: Option<u32>,
    /// The input's sizes the ONNX file leaves free, each by its name there
    /// and the axis it sizes (NCHW). CoreML runs a model compiled for one
    /// size faster and in a fraction of the memory (Real-ESRGAN x4 on a
    /// 256 px tile, MacBook Air: 1.27 s and 0.4 GB against 1.78 s and
    /// 3.3 GB), so an engine there compiles each size it meets. Empty for a
    /// fixed input, or one no engine sizes this way.
    pub free_dims: &'static [(&'static str, usize)],
    /// License mandates in-product credit (CC-BY family): a "CC-BY" badge
    /// in Settings → Model credits, sorted to the top.
    pub attribution_required: bool,
    /// Upscale dispatch knobs; `Some` only on upscale-category entries.
    pub upscale: Option<UpscaleModelKnobs>,
}

/// ONNX export contract for an upscale model.
#[derive(Debug, Clone, Copy)]
pub struct UpscaleModelKnobs {
    /// Name of the input tensor in the ONNX graph. RealESRGAN exports
    /// as `"data"`; Phhofm's HAT-L (Nomos8kSCHAT-L) as `"input"`.
    pub input_name: &'static str,
    /// `true` when the export expects fp16 inputs and emits fp16 outputs.
    pub is_fp16: bool,
    /// Native output scale of the ONNX graph. x2plus is the only
    /// native-2× upscale model.
    pub native_scale: u32,
    /// `true` for window-attention transformers (HAT, Swin), `false`
    /// for CNN-style RRDB.
    ///
    /// Distinct from `tile_size_multiple`: x2plus needs even dimensions
    /// (pixel_unshuffle) but is NOT window-attention; conflating the two
    /// costs it ~7% throughput from an over-wide overlap.
    pub uses_window_attention: bool,
    /// `Some` when the network normalizes over its whole input
    /// (GroupNorm): tiles normalized apart leave seams, so it ships cut at
    /// its norms and runs part by part.
    pub parts: Option<PartLayout>,
}

/// A network run part by part: its feature channels, and how far each
/// part reads around a pixel (input pixels, rounded up), so a crop's
/// middle is exact that far in from a neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartLayout {
    pub features: usize,
    pub head: usize,
    pub block: usize,
    pub tail: usize,
}

/// The keys a model run in parts names its files by: its head, its tail
/// and its norms' weights (JSON); its blocks are the rest, in key order.
pub const PART_HEAD: &str = "head";
pub const PART_TAIL: &str = "tail";
pub const PART_NORMS: &str = "norms";

impl UpscaleModelKnobs {
    /// Tile overlap in pixels: 32 for window-attention transformers
    /// (two window widths hide attention-window seams), 64 for CNN /
    /// RRDB, whose seams at 16 px stood out against one seamless tile.
    pub const fn tile_overlap(&self) -> u32 {
        if self.uses_window_attention { 32 } else { 64 }
    }
}

impl ModelDescriptor {
    pub fn part_layout(&self) -> Option<PartLayout> {
        self.upscale.and_then(|k| k.parts)
    }

    pub fn runs_in_parts(&self) -> bool {
        self.part_layout().is_some()
    }

    /// Upstream author and licence; every model carries one, bundled
    /// ones too, since Settings › Credits is where they are credited.
    pub fn license(&self) -> LicenseInfo {
        match self.source {
            ModelSource::OnDemand { license, .. }
            | ModelSource::MultiPartOnDemand { license, .. }
            | ModelSource::MultiPartBundled { license, .. }
            | ModelSource::Bundled { license } => license,
        }
    }

    /// Download size in MB for OnDemand entries, `None` for Bundled
    /// (the size is already paid in the binary). Multi-part on-demand
    /// bundles return the sum of all parts.
    pub fn size_mb(&self) -> Option<u32> {
        match self.source {
            ModelSource::OnDemand { size_mb, .. } => Some(size_mb),
            ModelSource::MultiPartOnDemand { parts, .. } => {
                let total: u64 = parts.iter().map(|p| p.size_bytes).sum();
                Some((total / (1024 * 1024)) as u32)
            }
            ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => None,
        }
    }

    /// True when this bundle requires explicit license-acceptance click
    /// before the download starts (CreativeML Open RAIL, NVIDIA SCL, …).
    /// Bundled variants are accepted at build time, so they never gate.
    pub fn requires_license_acceptance(&self) -> bool {
        matches!(self.source, ModelSource::MultiPartOnDemand { license_acceptance_required: true, .. })
    }

    /// Human-facing reason this model is gated, or `None` if unrestricted.
    /// Drives the disabled-button tooltip + Model Store badge text.
    pub fn hardware_advisory(&self, provider: &str) -> Option<&'static str> {
        let cpu_only = provider.eq_ignore_ascii_case("CPU");
        match self.gpu {
            GpuRequirement::None => None,
            GpuRequirement::Recommended if cpu_only => Some("Slow on CPU — pick a smaller model unless you have a GPU"),
            GpuRequirement::Recommended => None,
            GpuRequirement::Required if cpu_only => Some("Very slow on CPU — a GPU (CUDA / CoreML / DirectML) is strongly recommended"),
            GpuRequirement::Required => None,
        }
    }
}

/// Single source of truth. Order is the display order in dropdowns.
pub const REGISTRY: &[ModelDescriptor] = &[
    ModelDescriptor {
        id: ModelId::Silueta,
        display_name: "Silueta",
        description: "Small and built in: works without a download. Rougher edges than U2Net.",
        category: ModelCategory::Segmentation,
        source: ModelSource::Bundled {
            license: LicenseInfo {
                author: "Daniel Gatis (rembg), after U\u{b2}-Net by Xuebin Qin",
                license: "MIT",
                license_url: "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
                source_url: "https://github.com/danielgatis/rembg",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(440.0, 525.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(36.42, 618.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::Cpu(Why::NoGain("0.94x: 255 ms against the CPU's 240 ms, within 10%, in 1300 MB of system memory against the CPU's 750 MB (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)")),
            openvino: Place::Cpu(Why::Slower("0.82x: 782 ms against the CPU's 638 ms (i7-6700, OpenVINO plugin 1.8.1 on the processor, 2026-10-27)")),
            openvino_gpu: Place::Cpu(SILUETA_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(600, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(750, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(850, 650, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(350, 400, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_300, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-24")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: true,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::U2net,
        display_name: "U2Net",
        description: "Higher quality on edges and fine subjects.",
        category: ModelCategory::Segmentation,
        source: ModelSource::OnDemand {
            filename: "u2net-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/u2net-1.0.0.onnx",
            sha256: "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
            size_mb: 168,
            license: LicenseInfo {
                author: "Xuebin Qin",
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://github.com/xuebinqin/U-2-Net",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(25.93, 286.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(25.81, 302.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(221.0, 354.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::Cpu(Why::NoGain("1.09x: 369 ms against the CPU's 402 ms, within 10%, in 800 MB of system memory against the CPU's 700 MB (i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25)")),
            openvino_gpu: Place::There(Speed::on(177.0, 402.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(700, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Same(Cost::measured(600, 700, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Unmeasured,
            openvino: Priced::Same(Cost::measured(800, 0, "i7-6700, 2026-10-23")),
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: true,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::BiRefNetLite,
        display_name: "BiRefNet-lite",
        description: "Default model. Best detail on hair and leaves at 1024×1024.",
        category: ModelCategory::Segmentation,
        source: ModelSource::Bundled {
            license: LicenseInfo {
                author: "Peng Zheng (ZhengPeng7)",
                license: "MIT",
                license_url: "https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
                source_url: "https://github.com/ZhengPeng7/BiRefNet",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(397.1, 8820.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::Cpu(Why::Fails("its first run fails in a node DirectML fused: \"not enough memory\" (8007000E) with 11 GB free on the card (RTX 3060, Windows ML 1.30 and DirectML 1.24.1, 2026-10-26)")),
            coreml: Place::Cpu(Why::Fails(
                "its convs without pads do not compile (\"Required param 'pad' is missing\", ORT 1.30); given zero pads it \
                 matched the CPU at half its speed, 17.6 s against 8.2 s, with ScatterND, Equal, Expand, ConstantOfShape, Mod \
                 and Floor left to the CPU (MacBook Air M1, 2026-10-24)",
            )),
            openvino: Place::Cpu(Why::Slower("1.1x slower than the CPU with ORT's constant folding on (i7-6700, OpenVINO 1.24.1, 2026-10-26); OpenVINO's plugin ran it as fast as that build, 8.7 s against 8.4 s (i7-6700, OpenVINO plugin 1.8.1 on the processor, 2026-10-27)")),
            openvino_gpu: Place::Cpu(BIREFNET_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(7_100, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(12_850, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(1_350, 7_700, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Same(Cost::measured(5_150, 0, "i7-6700, 2026-10-23")),
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::Teed,
        display_name: "TEED",
        description: "Draws the lines for Sketch.",
        category: ModelCategory::EdgeDetection,
        source: ModelSource::Bundled {
            license: LicenseInfo {
                author: "Xavier Soria Poma",
                license: "MIT",
                license_url: "https://opensource.org/licenses/MIT",
                source_url: "https://github.com/xavysp/TEED",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(11.79, 127.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(15.76, 153.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::Cpu(Why::Slower("0.59x: 126 ms against the CPU's 74 ms (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)")),
            openvino: Place::There(Speed::on(88.1, 148.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(130.3, 142.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25").but("within 10%; its memory there not measured")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(400, 0, "i5-11600KF, 2026-10-25"),
            mac_cpu: Priced::Same(Cost::measured(250, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-25")),
            cuda: Priced::Same(Cost::measured(650, 400, "RTX 3060 12 GB, i5-11600KF, 2026-10-25")),
            directml: Priced::Same(Cost::measured(250, 300, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_200, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::LaMaFp32,
        display_name: "Eraser (LaMa)",
        description: "Object removal. Smooth fills on simple backgrounds.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "lama_fp32-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/lama_fp32-1.0.0.onnx",
            sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
            size_mb: 199,
            license: LicenseInfo {
                author: "Samsung AI Center / saic-mdal",
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://huggingface.co/Carve/LaMa-ONNX",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(149.8, 2210.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::Cpu(Why::Fails("its first run fails in a MatMul: \"the parameter is incorrect\" (80070057) (RTX 3060, Windows ML 1.30 and DirectML 1.24.1, 2026-10-26)")),
            coreml: Place::Cpu(Why::Fails("CoreML cannot plan it (\"Failed to build the model execution plan\") (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)")),
            openvino: Place::There(Speed::on(2360.0, 2660.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::Cpu(FILL_ON_INTEL_GRAPHICS),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::estimate(700, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::BigLaMa,
        display_name: "Eraser (Big-LaMa)",
        description: "LaMa trained on more photos (Places2): sharper fills on detailed regions.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "big_lama-1.0.1.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/big_lama-1.0.1.onnx",
            sha256: "c9660fc4aea2e62ffbaf3932024c1f4ce29f360e464fc71e1b708f1ad3bfea2e",
            size_mb: 199,
            license: LicenseInfo {
                author: "Samsung AI Center / saic-mdal",
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://huggingface.co/smartywu/big-lama",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(1298.6, 2740.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::Cpu(Why::Fails("its session fails to build: \"the parameter is incorrect\" (80070057) (RTX 3060, Windows ML 1.30 and DirectML 1.24.1, 2026-10-26)")),
            coreml: Place::Cpu(Why::Slower("0.72x: 3.81 s against the CPU's 2.75 s (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)")),
            openvino: Place::Cpu(Why::Fails("its session does not build: \"Conversion is failed for: Transpose-13\", at every optimization level (i7-6700, OpenVINO plugin 1.8.1 on the processor, 2026-10-27)")),
            openvino_gpu: Place::Cpu(FILL_ON_INTEL_GRAPHICS),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::estimate(700, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::Migan,
        display_name: "Eraser (MI-GAN)",
        description: "A light eraser (~25 MB). Sharper than LaMa on detailed regions; less smooth on flat backgrounds.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "migan-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/migan-1.0.0.onnx",
            sha256: "17531b1604e56ff3179a22824c19debf12741dadc551b4500b035bcb216b58ba",
            size_mb: 26,
            license: LicenseInfo {
                author: "Picsart AI Research",
                license: "MIT",
                license_url: "https://opensource.org/license/mit",
                source_url: "https://github.com/Picsart-AI-Research/MI-GAN",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(44.46, 935.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(44.5, 969.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(80.8, 270.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::Cpu(MIGAN_ON_OPENVINO),
            openvino_gpu: Place::Cpu(MIGAN_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(950, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(1_000, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(750, 1_200, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(350, 650, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::SdV15InpaintFp16,
        display_name: "Eraser (Stable Diffusion 1.5)",
        description: "Generative fills with Stable Diffusion. Needs a graphics card. About 2 GB.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "sd15-inpaint-fp16-1.0.0",
            parts: &[
                ModelPart {
                    key: "unet",
                    filename: "unet.onnx",
                    url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                    sha256: "a3726e1944ec2fbc9596096cd001520aea789367f329360ee34eb520c751de16",
                    size_bytes: 1720173976,
                },
                ModelPart {
                    key: "vae_encoder",
                    filename: "vae_encoder.onnx",
                    url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                    sha256: "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea",
                    size_bytes: 68430178,
                },
                ModelPart {
                    key: "vae_decoder",
                    filename: "vae_decoder.onnx",
                    url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                    sha256: "ce0ffca3fcfd0a2729a88d8d90dabf35c66d237404997e0461d52e46ac7a91bc",
                    size_bytes: 99093889,
                },
                ModelPart {
                    key: "text_encoder",
                    filename: "text_encoder.onnx",
                    url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                    sha256: "8af516fb184866f8caed192ca3bf0636ef9d40d4f7799c05600cae431d18d8d1",
                    size_bytes: 246368629,
                },
            ],
            license: LicenseInfo {
                author: "Runway / RanaLLC",
                license: "CreativeML Open RAIL-M",
                license_url: "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
                source_url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16",
            },
            license_acceptance_required: true,
        },
        version: "1.0.0",
        gpu: GpuRequirement::Required,
        places: Places::UNMEASURED,
        // The peak while the bundle loads on OpenVINO's CPU device, its
        // heaviest provider: 13.6-14.1 GB measured for the Fast bundle,
        // which has this one's architecture and size. An estimate of
        // 10 GB let erases start that the memory watchdog then killed
        // halfway through the load.
        needs: Needs {
            cpu: Cost::estimate(14000, 0, "the Fast bundle's 13.6-14.1 GB peak on OpenVINO's CPU device: same architecture and size"),
            mac_cpu: Priced::Unmeasured,
            // Load and 512 px tiles, 25 steps, ORT 1.24.1 with the lean
            // CUDA pool: 4.87 GB of system RAM at the cold load (3.1 GB
            // after), the card's peak 7.1 GB.
            cuda: Priced::Same(Cost::measured(4_900, 7_150, "RTX 3060 12 GB, i7-6700, 2026-10-21")),
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    // Exported with dynamic batch and spatial axes, so guided steps batch
    // into one UNet call and crops other than 512² load.
    ModelDescriptor {
        id: ModelId::SdV15LcmInpaintFp16,
        display_name: "Eraser (Stable Diffusion 1.5, fast)",
        description: "The Stable Diffusion eraser's Fast quality: several times faster than Quality.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "sd15-lcm-inpaint-fp16-1.0.1",
            parts: &[
                ModelPart {
                    key: "unet",
                    filename: "unet.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.1/unet.onnx",
                    sha256: "00708cf68318fb5df5e3dfb112ab6964af0fa739f7e2f6cbba1d7c7df7e2f91c",
                    size_bytes: 1720185037,
                },
                ModelPart {
                    key: "vae_encoder",
                    filename: "vae_encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.1/vae_encoder.onnx",
                    sha256: "33247c04059dff4f7a15ac5fc5525ae46eb13a5f5b121b00b7bcd608a5f169f7",
                    size_bytes: 68427657,
                },
                ModelPart {
                    key: "vae_decoder",
                    filename: "vae_decoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.1/vae_decoder.onnx",
                    sha256: "8cbf506018e52e7339fb3a4ed6bf8ac1d8b9e3108c4e015705e8d1074888129f",
                    size_bytes: 99095895,
                },
                ModelPart {
                    key: "text_encoder",
                    filename: "text_encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.1/text_encoder.onnx",
                    sha256: "5877aeae786c49c139b549f29b61ce7afd51639c25a19d94bf492bd78824f297",
                    size_bytes: 246350850,
                },
            ],
            license: LicenseInfo {
                author: "Latent Consistency Models / Tsinghua",
                license: "CreativeML Open RAIL-M",
                license_url: "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
                source_url: "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
            },
            license_acceptance_required: true,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places::UNMEASURED,
        // Measured peak while the bundle loads on OpenVINO's CPU device:
        // 13.6-14.1 GB.
        needs: Needs {
            cpu: Cost::measured(14000, 0, "i7-6700, OpenVINO CPU device, bundle load peak 13.6-14.1 GB"),
            mac_cpu: Priced::Unmeasured,
            // Load and 512 px tiles, 8 steps at CFG 2 (two UNet passes a
            // step, its heaviest), ORT 1.24.1 with the lean CUDA pool:
            // 4.87 GB of system RAM at the cold load, the card's peak 6.8 GB.
            cuda: Priced::Same(Cost::measured(4_900, 6_800, "RTX 3060 12 GB, i7-6700, 2026-10-21")),
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    // RORem-4S: the UNet's weights are split into files under GitHub's
    // 2 GiB asset limit; the graph names them, so they sit beside it.
    ModelDescriptor {
        id: ModelId::RoremFp16,
        display_name: "Eraser (RORem)",
        description: "Removes large objects cleanly: trained to fill with background, not to draw something new. Stable Diffusion XL, about 5.5 GB.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "rorem-4s-fp16-1.0.0",
            parts: &[
                ModelPart {
                    key: "unet",
                    filename: "unet.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_unet.onnx",
                    sha256: "52680ae80bee80db3ab2eaf85c8dde342b2b8b5f0a8c4d48f86ed676a31bdd12",
                    size_bytes: 3_770_384,
                },
                ModelPart {
                    key: "unet_data_0",
                    filename: "unet_0.data",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_unet_0.data",
                    sha256: "2b10d139a31477c68f401a617bd1d402d4056dcf3e1009f39bad714346cea928",
                    size_bytes: 1_980_548_992,
                },
                ModelPart {
                    key: "unet_data_1",
                    filename: "unet_1.data",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_unet_1.data",
                    sha256: "beb25f4924f71019d6c4496e7ae0f036c4357d733cef97ff2fb2aa4db0739b93",
                    size_bytes: 1_991_050_240,
                },
                ModelPart {
                    key: "unet_data_2",
                    filename: "unet_2.data",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_unet_2.data",
                    sha256: "94fd838901653d7858922f2a5c2242201bad6369a7614cf875ad354559753781",
                    size_bytes: 1_163_650_560,
                },
                ModelPart {
                    key: "vae_encoder",
                    filename: "vae_encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_vae_encoder.onnx",
                    sha256: "7d711a558fbde2186aa900c74ab65649b3fe88dee58bf98ceed4fcbfe59f0cf2",
                    size_bytes: 136_757_971,
                },
                ModelPart {
                    key: "vae_decoder",
                    filename: "vae_decoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/rorem_vae_decoder.onnx",
                    sha256: "4ff36b2a02436e7caa5581d59e2c629a08ae5943398899c8db0c0dc4d1245803",
                    size_bytes: 198_078_716,
                },
            ],
            license: LicenseInfo {
                author: "Li et al. (RORem); Stability AI (SDXL base)",
                license: "Apache 2.0 (RORem) + CreativeML Open RAIL++-M (SDXL)",
                license_url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
                source_url: "https://github.com/leeruibin/RORem",
            },
            license_acceptance_required: true,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            openvino: Place::Cpu(ROREM_ON_OPENVINO),
            openvino_gpu: Place::Cpu(ROREM_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        // Measured on ORT's CPU EP: 15.6 GiB anonymous peak (the fp16
        // weights widened for the CPU kernels, the VAE, activations);
        // the 4.8 GiB of mapped weight files on top are reclaimable.
        needs: Needs {
            cpu: Cost::measured(16000, 0, "i7-6700, ORT CPU EP: 15.6 GiB anonymous peak"),
            mac_cpu: Priced::Unmeasured,
            // Load and erase, 4 steps, lean sessions: 1.45 GB of system
            // RAM (the weights go to the card), the card's peak 8.6 GB.
            cuda: Priced::Same(Cost::measured(1_500, 8_650, "RTX 3060 12 GB, i7-6700, 2026-10-21")),
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::TaesdFp16,
        display_name: "Fast decoder (TAESD)",
        description: "Finishes Stable Diffusion fills about 3\u{00d7} faster, a little softer. Turn on Fast decoder in the eraser's Advanced settings.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "taesd-fp16-1.0.0",
            parts: &[
                ModelPart {
                    key: "decoder",
                    filename: "decoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/taesd-v1.0.0/decoder.onnx",
                    sha256: "4219d97c8bac63a18467ea47768545edcb15f554bfecadc1bed1fb70b8882304",
                    size_bytes: 2632309,
                },
            ],
            license: LicenseInfo {
                author: "Ollin Boer Bohan",
                license: "MIT",
                license_url: "https://github.com/madebyollin/taesd/blob/main/LICENSE",
                source_url: "https://github.com/madebyollin/taesd",
            },
            license_acceptance_required: false,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places::UNMEASURED,
        // Auxiliary VAE — loaded with a SD bundle, never standalone.
        needs: Needs {
            cpu: Cost::estimate(100, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::RealEsrganX4Plus,
        display_name: "Real-ESRGAN x4plus",
        description: "General-purpose 4\u{00d7} upscaler. Strong on photos and illustrations.",
        category: ModelCategory::Upscale,
        source: ModelSource::OnDemand {
            filename: "RealESRGAN_x4plus.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x4plus.onnx",
            sha256: "fb070c21d1e90102859d52328586a8738ebd4b2b076fe611ef42ba4f58076431",
            size_mb: 64,
            license: LicenseInfo {
                author: "Xintao Wang",
                license: "BSD-3-Clause",
                license_url: "https://opensource.org/licenses/BSD-3-Clause",
                source_url: "https://github.com/xinntao/Real-ESRGAN",
            },
            fp16: Some(OnDemandVariant {
                filename: "RealESRGAN_x4plus_fp16.onnx",
                url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x4plus_fp16.onnx",
                sha256: "838e4fad9a14a70e96c1953a08da2a0638c23b1dffc5b740351f0bc525b36250",
                size_mb: 33,
            }),
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(273.0, 8110.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(240.0, 8170.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(1255.8, 7560.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(7130.0, 9320.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(5980.0, 9240.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(2_950, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(4_350, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(1_200, 4_300, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(350, 5_000, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_200, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: Some(512),
        free_dims: &[("h", 2), ("w", 3)],
        attribution_required: false,
        upscale: Some(UpscaleModelKnobs {
            input_name: "data",
            is_fp16: false,
            native_scale: 4,
            uses_window_attention: false,
            parts: None,
        }),
    },
    ModelDescriptor {
        id: ModelId::Nomos8kSchatL,
        display_name: "4xNomos8kSCHAT-L",
        description: "Photo-tuned upscaler (HAT-L). The most detailed and the slowest: about a minute for a small photo without a graphics card.",
        category: ModelCategory::Upscale,
        source: ModelSource::OnDemand {
            filename: "4xNomos8kSCHAT-L.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4xNomos8kSCHAT-L.onnx",
            sha256: "919dff28836ff10fef2d5462e5b82c951211abf24f05196b0a6e2c24f20ed1de",
            size_mb: 155,
            license: LicenseInfo {
                author: "Phhofm / Philip Hofmann",
                license: "CC-BY-4.0",
                license_url: "https://creativecommons.org/licenses/by/4.0/",
                source_url: "https://github.com/Phhofm/models/tree/main/4xNomos8kSCHAT-L",
            },
            // The release is already fp16 (`is_fp16: true`): no sibling.
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::Cpu(Why::Differs(
                "a black picture from this fp16 release, every value 0, where the CPU's lands 3.3/255 off its input shrunk \
                 back (RTX 3060, ORT 1.24.1, 2026-10-24)",
            )),
            coreml: Place::Cpu(Why::Fails(
                "compiled for any size it fails to plan (error -14); for a 256 px tile it splits into 253 parts and its \
                 first run passed 80 s (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)",
            )),
            openvino: Place::There(Speed::on(47110.0, 126000.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25").but("first runs")),
            openvino_gpu: Place::Cpu(Why::Fails("one 256 px tile ran past 40 minutes and was stopped, against the CPU's 2m 06s (HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25)")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::estimate(1200, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: Some(16),
        recommended_tile: Some(256),
        free_dims: &[],
        attribution_required: true,
        upscale: Some(UpscaleModelKnobs {
            input_name: "input",
            is_fp16: true,
            native_scale: 4,
            uses_window_attention: true,
            parts: None,
        }),
    },
    ModelDescriptor {
        id: ModelId::RealEsrganX2Plus,
        display_name: "Real-ESRGAN x2plus",
        description: "Real-ESRGAN's own 2\u{d7} model: a 2\u{d7} Real-ESRGAN upscale runs it when installed, about four times faster than 4\u{d7} and a downscale. Hidden from the model picker.",
        category: ModelCategory::Upscale,
        source: ModelSource::OnDemand {
            filename: "RealESRGAN_x2plus.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x2plus.onnx",
            sha256: "7e0860bb32d903520a244c327b6e3d5e08d680b95c29b3fa8a02cb6ecd230c60",
            size_mb: 64,
            license: LicenseInfo {
                author: "Xintao Wang",
                license: "BSD-3-Clause",
                license_url: "https://opensource.org/licenses/BSD-3-Clause",
                source_url: "https://github.com/xinntao/Real-ESRGAN",
            },
            fp16: Some(OnDemandVariant {
                filename: "RealESRGAN_x2plus_fp16.onnx",
                url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x2plus_fp16.onnx",
                sha256: "700b02d23c0a547441a112ac229424b86039e119c5fefae39ecc74e4ce4b4edc",
                size_mb: 33,
            }),
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(76.14, 1570.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(85.5, 1750.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::Cpu(Why::Fails(
                "its pixel-unshuffle cannot be planned (\"Error in building plan\"), the tile pinned or free (MacBook Air \
                 M1, ORT 1.24 and 1.30)",
            )),
            openvino: Place::There(Speed::on(1727.9, 2350.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(1652.5, 2330.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(1_150, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(1_800, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(1_050, 1_200, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(300, 1_350, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: Some(2),
        recommended_tile: Some(512),
        free_dims: &[],
        attribution_required: false,
        upscale: Some(UpscaleModelKnobs {
            input_name: "data",
            is_fp16: false,
            native_scale: 2,
            uses_window_attention: false,
            parts: None,
        }),
    },
    ModelDescriptor {
        id: ModelId::FourXNmkdSiaxCx,
        display_name: "4x NMKD Siax-CX",
        description: "Community-trained universal upscaler. Sharper than Real-ESRGAN x4plus on clean and mildly compressed photos.",
        category: ModelCategory::Upscale,
        source: ModelSource::OnDemand {
            filename: "4x-NMKD-Siax-CX.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX.onnx",
            sha256: "d7db322397ae1041076a3f9fe9736a7bf9d3364e435eb5b0ece912e050a29f87",
            size_mb: 64,
            license: LicenseInfo {
                author: "Nmkd",
                license: "WTFPL",
                license_url: "https://www.wtfpl.net/about/",
                source_url: "https://openmodeldb.info/models/4x-NMKD-Siax-CX",
            },
            fp16: Some(OnDemandVariant {
                filename: "4x-NMKD-Siax-CX_fp16.onnx",
                url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX_fp16.onnx",
                sha256: "47645594b73a4e5d563babf4674eb756a5853e3cb5620547c02ad6a9e753d818",
                size_mb: 33,
            }),
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(272.2, 8070.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(246.75, 8160.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(1256.8, 7880.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(7180.0, 9160.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(5980.0, 9650.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(3_000, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(4_200, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-25")),
            cuda: Priced::Same(Cost::measured(1_200, 4_300, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(350, 5_000, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_200, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: Some(512),
        free_dims: &[("h", 2), ("w", 3)],
        attribution_required: false,
        upscale: Some(UpscaleModelKnobs {
            input_name: "data",
            is_fp16: false,
            native_scale: 4,
            uses_window_attention: false,
            parts: None,
        }),
    },
    ModelDescriptor {
        id: ModelId::FourXNomos2RealPlksr,
        display_name: "4xNomos2 RealPLKSR",
        description: "Fast photo upscaler trained on compressed photos. Natural skin and hair texture; can invent texture on flat areas like snow.",
        category: ModelCategory::Upscale,
        // Cut from our dynamic-shape export of Phhofm's weights (his ONNX is
        // fixed at 256 × 256): a head, each block up to its norm, a tail
        // and the norms' weights.
        source: ModelSource::MultiPartOnDemand {
            subdir: "4xNomos2_realplksr_dysample-parts-1.0.0",
            parts: &[
                ModelPart { key: PART_HEAD, filename: "nomos2-head.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-head.onnx", sha256: "8af91428f3b0da13ff3f6e2907ccc42522a6d60153b88d884035f2cd25c9b6aa", size_bytes: 9722 },
                ModelPart { key: "block01", filename: "nomos2-block01.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block01.onnx", sha256: "ca135e009bab895383e775a8450fe2faaa381f7d96e08576f15aa3580cc1d719", size_bytes: 1091260 },
                ModelPart { key: "block02", filename: "nomos2-block02.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block02.onnx", sha256: "136a13fd1970785efc2f69f0da652374ad15b74db3332dd8ff6a9af582d2f070", size_bytes: 1091393 },
                ModelPart { key: "block03", filename: "nomos2-block03.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block03.onnx", sha256: "45b0ec3d00fd4bab3c24e60ad232e1cd37ff441d64306598f4bc47d66295ee30", size_bytes: 1091471 },
                ModelPart { key: "block04", filename: "nomos2-block04.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block04.onnx", sha256: "7da56a7f879cc4739dac9fafad8371d6062acbd0044d46e8905133f72c61bd26", size_bytes: 1091479 },
                ModelPart { key: "block05", filename: "nomos2-block05.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block05.onnx", sha256: "5654a15170857bf0bf7de2493a53a379ea5c66909e2f48515221c515ea1d13dd", size_bytes: 1091479 },
                ModelPart { key: "block06", filename: "nomos2-block06.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block06.onnx", sha256: "1f1d4560a79f717f138864dbb145ac3e6087ef609e516fb5aa04d4a7996ac52a", size_bytes: 1091479 },
                ModelPart { key: "block07", filename: "nomos2-block07.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block07.onnx", sha256: "8849c3983181982e610369b082dbad70f59a5166511ee0788e56d7bc5b928391", size_bytes: 1091479 },
                ModelPart { key: "block08", filename: "nomos2-block08.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block08.onnx", sha256: "e25db5fdb325981f27c74368699094fbe5119d404cfb42bb6e4849dbf3dd8642", size_bytes: 1091479 },
                ModelPart { key: "block09", filename: "nomos2-block09.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block09.onnx", sha256: "367a9038f4fcb08450e6f9c588fed736105320701200f605142287873c8e63b0", size_bytes: 1091479 },
                ModelPart { key: "block10", filename: "nomos2-block10.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block10.onnx", sha256: "528c2a139036e18f53eee2486c7afa51379b970f97bfe3705095982f2300112b", size_bytes: 1091618 },
                ModelPart { key: "block11", filename: "nomos2-block11.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block11.onnx", sha256: "62a681b85ff06bc997e76903ccec63eca78e26cdb8fa283621a2407b52bec75f", size_bytes: 1091677 },
                ModelPart { key: "block12", filename: "nomos2-block12.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block12.onnx", sha256: "37efa2382acba7ea7d6bed52174d5605e5ecba4e26968ada721a0e268abf249a", size_bytes: 1091677 },
                ModelPart { key: "block13", filename: "nomos2-block13.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block13.onnx", sha256: "f70c910d5cf484cb906a74ff83f33cef0bb507d89d06f97c3b46d57e774b4a0b", size_bytes: 1091677 },
                ModelPart { key: "block14", filename: "nomos2-block14.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block14.onnx", sha256: "cac696854692d047d1cb4beb03a25ccebdba5a24c80fea5979a4eb4b05565f32", size_bytes: 1091677 },
                ModelPart { key: "block15", filename: "nomos2-block15.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block15.onnx", sha256: "b634dfe0c4d5c157efdfa56a4b6c1e09849d97d2d42d4135917716ad5b6aae36", size_bytes: 1091677 },
                ModelPart { key: "block16", filename: "nomos2-block16.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block16.onnx", sha256: "5ae7ce5487044f76d7c5db2a84bfa6f16cea3a9a55278edde668c230f3657fe7", size_bytes: 1091685 },
                ModelPart { key: "block17", filename: "nomos2-block17.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block17.onnx", sha256: "792f4019f812c7b67e94449f12caea6686a9b413ef50a9d09ab22e785459900c", size_bytes: 1091685 },
                ModelPart { key: "block18", filename: "nomos2-block18.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block18.onnx", sha256: "ff0125a593de52c5cc78556f2b92dfaf744b43d68a693f40d815a2b055a27e68", size_bytes: 1091689 },
                ModelPart { key: "block19", filename: "nomos2-block19.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block19.onnx", sha256: "10ca58f4f763130017f0b17b6a26f5e53fca7630ec84d275d545d9333cd65398", size_bytes: 1091721 },
                ModelPart { key: "block20", filename: "nomos2-block20.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block20.onnx", sha256: "1be962e8dff96968c3416ea3c9e85948fe31cc51187cdffd6b00f872c61c378c", size_bytes: 1091728 },
                ModelPart { key: "block21", filename: "nomos2-block21.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block21.onnx", sha256: "347fc043f1a1734951f8e9377978c0daf0b8dcf5fb581d963b55af3ab06802af", size_bytes: 1091770 },
                ModelPart { key: "block22", filename: "nomos2-block22.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block22.onnx", sha256: "de72dadfc0535342138243442c86d0487d72cfe646b7de6475093c9911b1839a", size_bytes: 1091770 },
                ModelPart { key: "block23", filename: "nomos2-block23.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block23.onnx", sha256: "25e91ffe7553ecfa95fb2804666dc8e0674f7a50ca331322de8856145a9ae9d2", size_bytes: 1091770 },
                ModelPart { key: "block24", filename: "nomos2-block24.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block24.onnx", sha256: "4b968f7c966c8b4c1aa931125964b432b138913005035519a0c3bc77aa68b460", size_bytes: 1091770 },
                ModelPart { key: "block25", filename: "nomos2-block25.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block25.onnx", sha256: "7dc372d0d2c3f62fcd8e6cd72778e80ab05cd9089ba416a3758faae2740d2e38", size_bytes: 1091770 },
                ModelPart { key: "block26", filename: "nomos2-block26.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block26.onnx", sha256: "b50dd916f41b73876e884aeb05d58ae5f674b5bb9bf584a3aed9bc463bef5b19", size_bytes: 1091770 },
                ModelPart { key: "block27", filename: "nomos2-block27.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block27.onnx", sha256: "f39cb2de5874b3f78a840bd7c907897c6ac104b207a87334f6a2035002ae4959", size_bytes: 1091770 },
                ModelPart { key: "block28", filename: "nomos2-block28.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-block28.onnx", sha256: "25d0db817decbffbaff2f5de994917252529b5d11e8591e63dc3c3a37381dcfe", size_bytes: 1091770 },
                ModelPart { key: PART_TAIL, filename: "nomos2-tail.onnx", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-tail.onnx", sha256: "258d899da150dbdcc2d2a5809b933a330e426e72d682630551b176440f4b54b1", size_bytes: 238202 },
                ModelPart { key: PART_NORMS, filename: "nomos2-norms.json", url: "https://github.com/aktiwers/prunr/releases/download/models-v1/nomos2-norms.json", sha256: "415fe6133be0b69c15b7c32115dad65351fdab798d472d9bad89cfea67193ef1", size_bytes: 77124 },
            ],
            license: LicenseInfo {
                author: "Philip Hofmann (Phhofm); cut into parts by Prunr",
                license: "CC-BY-4.0",
                license_url: "https://creativecommons.org/licenses/by/4.0/",
                source_url: "https://github.com/Phhofm/models/releases/tag/4xNomos2_realplksr_dysample",
            },
            license_acceptance_required: false,
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(1100.5, 4050.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::Cpu(Why::Fails("its run fails in a Reshape: \"the parameter is incorrect\" (80070057) (RTX 3060, Windows ML 1.30 and DirectML 1.24.1, 2026-10-26)")),
            coreml: Place::Cpu(Why::Fails(
                "its tail's upsampler is rejected (\"axis 4 is not in valid range\") after 10 s building the other parts \
                 (MacBook Air M1, ORT 1.30)",
            )),
            openvino: Place::Cpu(NOMOS2_ON_OPENVINO),
            openvino_gpu: Place::Cpu(Why::Differs("mean 126.69/255, 99.9% of values more than 2/255 apart (HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25)")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::estimate(750, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Same(Cost::measured(1_550, 7_400, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Unmeasured,
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: true,
        upscale: Some(UpscaleModelKnobs {
            input_name: "input",
            is_fp16: false,
            native_scale: 4,
            uses_window_attention: false,
            // The head's 3 × 3; a block's two 3 × 3, its 17 × 17 and its
            // attention's 3 × 3; the tail's 3 × 3 and DySample's sampling
            // offsets (16 measured to keep the tail exact).
            parts: Some(PartLayout { features: 64, head: 2, block: 12, tail: 16 }),
        }),
    },
    ModelDescriptor {
        id: ModelId::FourXNmkdSuperscale,
        display_name: "4x NMKD Superscale",
        description: "Community-trained restoration upscaler. Handles noise and JPEG artifacts on real photos.",
        category: ModelCategory::Upscale,
        source: ModelSource::OnDemand {
            filename: "4x-NMKD-Superscale.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Superscale.onnx",
            sha256: "6b790a203b341d2db71735040f4f1b26c2bdc5440fdc850e217c2234cd996618",
            size_mb: 64,
            license: LicenseInfo {
                author: "Nmkd",
                license: "WTFPL",
                license_url: "https://www.wtfpl.net/about/",
                source_url: "https://openmodeldb.info/models/4x-NMKD-Superscale",
            },
            fp16: Some(OnDemandVariant {
                filename: "4x-NMKD-Superscale_fp16.onnx",
                url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Superscale_fp16.onnx",
                sha256: "a59481435da813b5b1629b047fc7490a22badd8ed09d54831cd2cf297b0c2bb9",
                size_mb: 33,
            }),
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(272.4, 8030.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(240.5, 8220.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(1389.3, 7530.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(7372.1, 9510.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(6110.0, 9240.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(3_000, 0, "i5-11600KF and i7-6700, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(4_200, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-25")),
            cuda: Priced::Same(Cost::measured(1_200, 4_300, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(350, 5_000, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_200, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: Some(512),
        free_dims: &[("h", 2), ("w", 3)],
        attribution_required: false,
        upscale: Some(UpscaleModelKnobs {
            input_name: "data",
            is_fp16: false,
            native_scale: 4,
            uses_window_attention: false,
            parts: None,
        }),
    },
    ModelDescriptor {
        id: ModelId::DdcolorTiny,
        display_name: "DDColor",
        description: "Colors a black-and-white photo. Strength sets how much of the model's color shows.",
        category: ModelCategory::Colorize,
        source: ModelSource::OnDemand {
            filename: "ddcolor_paper_tiny.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/ddcolor_paper_tiny.onnx",
            sha256: "5b4396ff19dbc9729a39fbebd6d7857dee6ff58a35d8820467efa168ecf7b17b",
            size_mb: 270,
            license: LicenseInfo {
                author: "piddnad (Kang et al., ICCV 2023)",
                license: "Apache-2.0",
                license_url: "https://github.com/piddnad/DDColor/blob/master/LICENSE",
                source_url: "https://github.com/piddnad/DDColor",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::Recommended,
        places: Places {
            cuda: Place::There(Speed::on(75.07, 1120.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(176.0, 1160.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::Cpu(Why::Differs("its colours vary between runs: 4.8%, 14.5% and 9.5% of values more than 2/255 from the CPU's in three runs, against a 10% tolerance (MacBook Air M1, ORT 1.30, 2026-10-23 and 2026-10-25)")),
            openvino: Place::There(Speed::on(901.0, 1130.0, "i7-6700, OpenVINO plugin 1.8.1 on the processor, 2026-10-27")),
            openvino_gpu: Place::Cpu(DDCOLOR_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        // Measured: 1.6 GB peak RSS on the CPU EP, session load included.
        needs: Needs {
            cpu: Cost::measured(1600, 0, "i7-6700, CPU EP, peak RSS with the session load"),
            mac_cpu: Priced::Same(Cost::measured(2_500, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(1_050, 2_850, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(850, 2_400, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::DepthAnythingV2Small,
        display_name: "Depth Anything V2 Small",
        description: "Finds how near each part of a photo is, for lens blur, fog, fading lines and selecting by depth.",
        category: ModelCategory::Depth,
        source: ModelSource::OnDemand {
            filename: "depth_anything_v2_small.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/depth_anything_v2_small.onnx",
            sha256: "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
            size_mb: 95,
            license: LicenseInfo {
                author: "Depth Anything (Yang et al., 2024)",
                license: "Apache-2.0",
                license_url: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/blob/main/LICENSE",
                source_url: "https://github.com/DepthAnything/Depth-Anything-V2",
            },
            fp16: None,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(47.0, 1080.0, "RTX 3060, ORT 1.30, input 518 px on the short side, 2026-10-30")),
            directml: Place::There(Speed::on(65.56, 531.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(231.0, 644.0, "MacBook Air M1 16 GB, ORT 1.30, input 518 px on the short side, 2026-10-30")),
            openvino: Place::There(Speed::on(382.0, 487.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::Cpu(Why::Slower("0.65x: 770 ms against the CPU's 497 ms (HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25)")),
            ..Places::UNMEASURED
        },
        // Measured: 366 MB peak RSS on the CPU EP, session load included
        // (784×518 input, a 3:2 photo).
        needs: Needs {
            cpu: Cost::measured(450, 0, "i5-11600KF, input 518 px on the short side, 2026-10-30"),
            mac_cpu: Priced::Same(Cost::measured(900, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, input 518 px on the short side, 2026-10-30")),
            cuda: Priced::Same(Cost::measured(1_000, 700, "RTX 3060 12 GB, i5-11600KF, input 518 px on the short side, 2026-10-30")),
            directml: Priced::Same(Cost::measured(400, 500, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_400, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, input 518 px on the short side, 2026-10-30")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[("batch_size", 0), ("height", 2), ("width", 3)],
        attribution_required: false,
        upscale: None,
    },
    // The heavy encoder runs once per image, the decoder per click/stroke.
    // Bundled to avoid a first-run download gate.
    ModelDescriptor {
        id: ModelId::Sam21HieraSmall,
        display_name: "Magic Brush (SAM 2.1 Hiera Small)",
        description: "Click or stroke to select objects. Apache 2.0, from Meta. Reads each photo once, then answers every click at once.",
        category: ModelCategory::Selection,
        source: ModelSource::MultiPartBundled {
            parts: &[
                BundledPart {
                    key: "encoder",
                    filename: "sam2.1_hiera_small.encoder.onnx",
                },
                BundledPart {
                    key: "decoder",
                    filename: "sam2.1_hiera_small.decoder.onnx",
                },
            ],
            license: LicenseInfo {
                author: "Meta / facebookresearch",
                license: "Apache 2.0",
                license_url: "https://github.com/facebookresearch/sam2/blob/main/LICENSE",
                source_url: "https://github.com/facebookresearch/sam2",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(379.0, 2510.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(382.52, 1970.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(717.9, 1680.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(1650.0, 1820.0, "the encoder, i7-6700, OpenVINO plugin 1.8.1 on the processor, 2026-10-27")),
            openvino_gpu: Place::Cpu(SAM_ON_OPENVINO),
            ..Places::UNMEASURED
        },
        // One encode and a click, `prunr bench --needs`: 1.12 GB on the
        // CPU; on CUDA 1.0 GB of RAM and the card's peak 2.18 GB during an
        // encode (it hands the pool back after). On a MacBook Air's GPU
        // (ORT 1.30) the encoder takes 0.52 s against 1.99 s on its CPU,
        // with the CPU's masks (20 of 20 clicks, IoU 1.0, two photos), at a
        // 4.9 GB peak against 1.3 GB (2026-10-23).
        needs: Needs {
            cpu: Cost::measured(1_150, 0, "i5-11600KF, 2026-10-23"),
            mac_cpu: Priced::Same(Cost::measured(1_300, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-23")),
            cuda: Priced::Same(Cost::measured(1_000, 2_200, "RTX 3060 12 GB, i5-11600KF, 2026-10-23")),
            directml: Priced::Same(Cost::measured(1_150, 1_300, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(4_900, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-23")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::FaceGfpgan,
        display_name: "Faces (GFPGAN v1.4)",
        description: "Redraws soft, small or compressed faces. The crispest of the two face models. Trained on FFHQ, a non-commercial photo set.",
        category: ModelCategory::Face,
        source: ModelSource::MultiPartOnDemand {
            subdir: "face-gfpgan-1.4-1.0.0",
            parts: &[
                ModelPart {
                    key: "restorer",
                    filename: "gfpgan_v1.4.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/gfpgan_v1.4.onnx",
                    sha256: "0e063e4071211196ec76e9da82f0495e2d4f56e2d9ebcd63c776dbcef60cd087",
                    size_bytes: 340_345_497,
                },
            ],
            license: LicenseInfo {
                author: "Tencent ARC (GFPGAN)",
                license: "Apache 2.0",
                license_url: "https://github.com/TencentARC/GFPGAN/blob/master/LICENSE",
                source_url: "https://github.com/TencentARC/GFPGAN",
            },
            license_acceptance_required: false,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(68.47, 1010.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(69.26, 1140.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(325.0, 1380.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(1280.7, 1460.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::Cpu(Why::Slower("0.43x: 3.33 s against the CPU's 1.43 s (HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25)")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(1_350, 0, "i5-11600KF, 2026-10-25"),
            mac_cpu: Priced::Same(Cost::measured(1_200, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-25")),
            cuda: Priced::Same(Cost::measured(700, 1_700, "RTX 3060 12 GB, i5-11600KF, 2026-10-25")),
            directml: Priced::Same(Cost::measured(700, 1_600, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_350, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    ModelDescriptor {
        id: ModelId::FaceRestoreFormerPp,
        display_name: "Faces (RestoreFormer++)",
        description: "Redraws soft, small or compressed faces. Stays closest to the person's own features. Trained on FFHQ, a non-commercial photo set.",
        category: ModelCategory::Face,
        source: ModelSource::MultiPartOnDemand {
            subdir: "face-restoreformer-pp-1.0.1",
            parts: &[
                ModelPart {
                    key: "restorer",
                    filename: "restoreformer_pp-1.0.1.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/models-v1/restoreformer_pp-1.0.1.onnx",
                    sha256: "ee1fb28557e84b8afdf5a829f20efcadd15ce4dc8bb67fa0564bb85c17a5985a",
                    size_bytes: 295_250_172,
                },
            ],
            license: LicenseInfo {
                author: "Zhouxia Wang et al. (RestoreFormer++)",
                license: "Apache 2.0",
                license_url: "https://github.com/wzhouxiff/RestoreFormerPlusPlus/blob/main/LICENSE",
                source_url: "https://github.com/wzhouxiff/RestoreFormerPlusPlus",
            },
            license_acceptance_required: false,
        },
        version: "1.0.1",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(127.6, 2430.0, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(127.7, 2580.0, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::There(Speed::on(458.0, 2560.0, "MacBook Air M1 16 GB, ORT 1.30, 2026-10-25")),
            openvino: Place::There(Speed::on(2461.5, 2880.0, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25")),
            openvino_gpu: Place::There(Speed::on(976.0, 3080.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25")),
            ..Places::UNMEASURED
        },
        needs: Needs {
            cpu: Cost::measured(1_550, 0, "i5-11600KF, 2026-10-25"),
            mac_cpu: Priced::Same(Cost::measured(1_750, 0, "MacBook Air M1 16 GB, macOS 27, 2026-10-25")),
            cuda: Priced::Same(Cost::measured(800, 1_700, "RTX 3060 12 GB, i5-11600KF, 2026-10-25")),
            directml: Priced::Same(Cost::measured(600, 1_900, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Same(Cost::measured(1_050, 0, "MacBook Air M1 16 GB, macOS 27, ORT 1.30, 2026-10-25")),
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
    // Built in, so faces are found on every install, with or without a
    // face model. Face model folders installed while each carried its
    // own copy keep that file, unread; Remove deletes it with the folder.
    ModelDescriptor {
        id: ModelId::YuNet,
        display_name: "Face finder (YuNet)",
        description: "Finds the faces that Faces redraws and that Auto focus aims at.",
        category: ModelCategory::Face,
        source: ModelSource::Bundled {
            license: LicenseInfo {
                author: "Shiqi Yu et al. (libfacedetection), via OpenCV Zoo",
                license: "MIT",
                license_url: "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE",
                source_url: "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
            },
        },
        version: "2026may",
        gpu: GpuRequirement::None,
        places: Places {
            cuda: Place::There(Speed::on(2.75, 3.8, "RTX 3060, ORT 1.24.1 CUDA 12, 2026-10-25")),
            directml: Place::There(Speed::on(3.77, 5.5, "RTX 3060, Windows ML 1.30, 2026-10-26")),
            coreml: Place::Cpu(Why::Slower("0.87x: 7.2 ms against the CPU's 6.2 ms (MacBook Air M1 16 GB, ORT 1.30, 2026-10-25)")),
            openvino: Place::There(Speed::on(4.95, 5.3, "i7-6700, OpenVINO 1.24.1 on the processor, 2026-10-25").but("within 10%; its memory there not measured")),
            openvino_gpu: Place::There(Speed::on(6.9, 7.0, "HD 530, OpenVINO 1.24.1 with the Intel graphics on, 2026-10-25").but("within 10%; its memory there not measured")),
            ..Places::UNMEASURED
        },
        // A 230 KB model at no more than 1280 px: its input is 20 MB.
        needs: Needs {
            cpu: Cost::estimate(64, 0, CARRIED),
            mac_cpu: Priced::Unmeasured,
            cuda: Priced::Unmeasured,
            directml: Priced::Same(Cost::measured(250, 100, "RTX 3060 12 GB, i5-11600KF, Windows ML 1.30, 2026-10-28")),
            coreml: Priced::Unmeasured,
            openvino: Priced::Unmeasured,
            migraphx: Priced::Unmeasured,
        },
        parallel_jobs_pay_on_cpu: false,
        tile_size_multiple: None,
        recommended_tile: None,
        free_dims: &[],
        attribution_required: false,
        upscale: None,
    },
];

/// The version the models built into the binary ship with: they change
/// only with it.
pub const BUILT_IN_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn descriptor(id: ModelId) -> Option<&'static ModelDescriptor> {
    REGISTRY.iter().find(|d| d.id == id)
}

/// The model's name as the user sees it, "Model" without a registry row.
pub fn display_name(id: ModelId) -> &'static str {
    descriptor(id).map_or("Model", |d| d.display_name)
}

/// Identifies the downloaded files of a model by their checksums, so a
/// fixed or re-exported download (whose version string may not move)
/// is told apart from the one before it. `None` for models built into
/// the binary: they change only with the app version.
pub fn content_tag(id: ModelId) -> Option<u64> {
    match descriptor(id)?.source {
        ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => None,
        ModelSource::OnDemand { sha256, fp16, .. } => {
            Some(tag_of_shas(std::iter::once(sha256).chain(fp16.map(|v| v.sha256))))
        }
        ModelSource::MultiPartOnDemand { parts, .. } => Some(tag_of_shas(parts.iter().map(|p| p.sha256))),
    }
}

/// What names a model's files in what is kept about them (the compiled
/// graph cache, the failure list): a download's checksums (`content_tag`;
/// a re-export may keep its version string), a built-in model's app
/// version, as it ships with the app and may change with any release.
pub fn files_id(id: ModelId) -> String {
    match content_tag(id) {
        Some(tag) => format!("{tag:016x}"),
        None => format!("app{BUILT_IN_VERSION}"),
    }
}

/// FNV-1a over the checksums: stable across builds and platforms, so a
/// tag written to disk by one run matches the next.
fn tag_of_shas<'a>(shas: impl IntoIterator<Item = &'a str>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for sha in shas {
        for b in sha.bytes().chain(std::iter::once(0)) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The registry's places for `model`; unmeasured for a model not in it.
pub fn registry_places(model: ModelId) -> Places {
    descriptor(model).map_or(Places::UNMEASURED, |d| d.places)
}

/// The column a session of `model` on `provider` reads in `places`. On
/// OpenVINO it is the Intel graphics' entry when they are on
/// (`intel_gpu`), the model's kind can use them and that entry keeps it
/// there; else the processor's, so a model the graphics run slower still
/// gets OpenVINO's processor where that beats ORT's CPU. Fills stay on the
/// processor: the GPU plugin cannot build LaMa (dynamic shapes), and under
/// AUTO it ran ~10% slower. `None` for the CPU and providers the table
/// does not know.
pub fn column_in(places: &Places, model: ModelId, provider: &str, intel_gpu: bool, remeasuring: bool) -> Option<Column> {
    let column = Column::of(provider)?;
    let on_graphics = column == Column::OpenVino
        && intel_gpu
        && descriptor(model).is_some_and(|d| d.category != ModelCategory::Inpaint)
        && !places.openvino_gpu.kept_off(remeasuring);
    Some(if on_graphics { Column::OpenVinoGpu } else { column })
}

/// Where `model` runs on `provider` in `places` (`column_in`); unmeasured
/// for a provider the table does not know.
pub fn place_in(places: &Places, model: ModelId, provider: &str, intel_gpu: bool, remeasuring: bool) -> Place {
    column_in(places, model, provider, intel_gpu, remeasuring).map_or(Place::Unmeasured, |column| places.get(column))
}


#[derive(Debug, PartialEq)]
pub enum ResolveError {
    NotInRegistry,
    /// Model is Bundled or single-file OnDemand — has no named parts.
    NotMultiPart,
    UnknownPart(String),
    /// Not yet downloaded, or an I/O error.
    Io(String),
}

/// Resolve bytes for a model. Returns `None` for `OnDemand` entries that
/// haven't been downloaded yet. A session copies the weights it is built
/// from, so nothing keeps them past the last holder: a download is read
/// from disk for each caller, and a bundled model is unpacked once and
/// shared by everyone holding it at the time (see `shared_bundled`).
pub fn resolve_bytes(id: ModelId) -> Option<Arc<Vec<u8>>> {
    resolve_bytes_from(id, on_demand_dir().as_deref())
}

fn resolve_bytes_from(id: ModelId, downloads: Option<&std::path::Path>) -> Option<Arc<Vec<u8>>> {
    let desc = descriptor(id)?;
    match desc.source {
        ModelSource::Bundled { .. } => Some(bundled_bytes(id)),
        ModelSource::OnDemand { filename, .. } => read_on_demand(downloads?, filename).map(Arc::new),
        // Multi-part models load each part via `resolve_part_bytes`.
        ModelSource::MultiPartOnDemand { .. } | ModelSource::MultiPartBundled { .. } => None,
    }
}

/// Resolve bytes for one part of a multi-part model, on-demand (read
/// from disk) or bundled (unpacked). The part key is checked before any
/// disk I/O or decompression.
pub fn resolve_part_bytes(
    id: ModelId,
    part_key: &str,
) -> Result<Arc<Vec<u8>>, ResolveError> {
    let desc = descriptor(id).ok_or(ResolveError::NotInRegistry)?;
    match desc.source {
        ModelSource::MultiPartOnDemand { subdir, parts, .. } => {
            let part = parts.iter().find(|p| p.key == part_key)
                .ok_or_else(|| ResolveError::UnknownPart(part_key.to_string()))?;
            let dir = on_demand_dir()
                .ok_or_else(|| ResolveError::Io("on_demand_dir not resolvable".to_string()))?;
            let path = dir.join(subdir).join(part.filename);
            let bytes = std::fs::read(&path)
                .map_err(|e| ResolveError::Io(format!("read {}: {e}", path.display())))?;
            Ok(Arc::new(bytes))
        }
        ModelSource::MultiPartBundled { parts, .. } => {
            let part = parts.iter().find(|p| p.key == part_key)
                .ok_or_else(|| ResolveError::UnknownPart(part_key.to_string()))?;
            Ok(bundled_multipart_bytes(id, part.key))
        }
        _ => Err(ResolveError::NotMultiPart),
    }
}

/// User-facing error string when an OnDemand model isn't yet installed.
pub fn not_installed_error(id: ModelId) -> String {
    let name = display_name(id);
    format!("{name} is not installed. Open the Model Store from the model dropdown to download it.")
}

pub fn is_available(id: ModelId) -> bool {
    match descriptor(id).map(|d| d.source) {
        Some(ModelSource::Bundled { .. }) | Some(ModelSource::MultiPartBundled { .. }) => true,
        Some(ModelSource::OnDemand { .. }) => download_path(id).is_some_and(|p| p.is_file()),
        Some(ModelSource::MultiPartOnDemand { .. }) => multi_part_paths(id).is_some(),
        None => false,
    }
}

/// Where a single-file download keeps `id`'s own weights, installed or
/// not; `None` for a bundled or multi-part model.
pub fn download_path(id: ModelId) -> Option<std::path::PathBuf> {
    let ModelSource::OnDemand { filename, .. } = descriptor(id)?.source else { return None };
    Some(on_demand_dir()?.join(filename))
}

/// Absolute paths to every part of a multi-part bundle; `None` for
/// non-MultiPart entries or when the bundle isn't fully installed.
pub fn multi_part_paths(id: ModelId) -> Option<Vec<(&'static str, std::path::PathBuf)>> {
    let desc = descriptor(id)?;
    let ModelSource::MultiPartOnDemand { subdir, parts, .. } = desc.source else { return None };
    let root = on_demand_dir()?;
    let bundle_dir = root.join(subdir);
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        let path = bundle_dir.join(p.filename);
        if !path.is_file() {
            return None;
        }
        out.push((p.key, path));
    }
    Some(out)
}

/// The bundle keeps weights in `.data` files beside its graphs (a graph
/// over the 2 GiB protobuf limit). The graph names them relative to its
/// own directory, so it only loads from there.
pub fn keeps_weights_apart(id: ModelId) -> bool {
    descriptor(id).is_some_and(|d| match d.source {
        ModelSource::MultiPartOnDemand { parts, .. } => parts.iter().any(|p| p.filename.ends_with(".data")),
        _ => false,
    })
}

/// See `ModelId::sd_spec`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdSpec {
    pub latent: SdLatent,
    /// The ORT CPU EP's output is right. SD 1.5's FP16 bundles come out
    /// as text-like artifacts on it, so they need a GPU EP; RORem's
    /// matches its PyTorch reference.
    pub cpu_output_correct: bool,
    /// Sessions run without ORT's memory-pattern planning. Measured on
    /// RORem (prunr-export/rorem_mem): 15 GB peak instead of 21.7 at the
    /// same speed. SD 1.5 is unmeasured, so it keeps ORT's default.
    pub lean_sessions: bool,
    /// The card's memory the tall crop adds to the bundle's `needs` on a
    /// card; `None` while the bundle is not measured on one.
    pub tall_crop_card_extra_mb: Option<u32>,
}

/// The latent space an SD-family model denoises in, named by the VAE
/// that encodes and decodes it. RORem is SDXL-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdLatent {
    Sd15,
    Sdxl,
}

impl SdLatent {
    /// Diffusers' `vae.config.scaling_factor`.
    pub fn vae_scale(self) -> f32 {
        match self {
            Self::Sd15 => 0.18215,
            Self::Sdxl => 0.13025,
        }
    }

    /// TAESD can stand in for the VAE: it decodes SD 1.5's latents only.
    pub fn takes_taesd(self) -> bool {
        self == Self::Sd15
    }
}

/// Per-app data root. Linux: `$XDG_DATA_HOME/prunr/`, macOS:
/// `~/Library/Application Support/prunr/`, Windows: `%LOCALAPPDATA%\prunr\`,
/// not the roaming folder: models and runtimes run to gigabytes, which a
/// roaming profile copies from computer to computer.
/// Does NOT create the dir.
pub fn data_dir() -> Option<std::path::PathBuf> {
    dirs::data_local_dir().map(|d| d.join("prunr"))
}

pub fn on_demand_dir() -> Option<std::path::PathBuf> {
    data_dir().map(|d| d.join("models"))
}

/// Files an earlier version downloaded into `on_demand_dir` that nothing
/// reads now. A model that changes its files adds the old names here, so
/// an update leaves nothing behind.
const RETIRED_FILES: &[&str] = &[
    // Nomos2 in one piece, before it shipped in parts.
    "4xNomos2_realplksr_dysample.onnx",
    // TAESD's encoder: the eraser encodes with Stable Diffusion's own VAE.
    "taesd-fp16-1.0.0/encoder.onnx",
];

/// Remove the retired files; the bytes freed. Run off the UI thread.
pub fn remove_retired_files() -> u64 {
    on_demand_dir().map_or(0, |dir| remove_retired_in(&dir, RETIRED_FILES))
}

fn remove_retired_in(dir: &std::path::Path, names: &[&str]) -> u64 {
    names.iter().map(|name| {
        let path = dir.join(name);
        std::fs::metadata(&path).ok()
            .filter(|m| m.is_file() && std::fs::remove_file(&path).is_ok())
            .map_or(0, |m| m.len())
    }).sum()
}

/// Where user-installed ONNX Runtime builds live, one subdir per
/// runtime. Subdir of `data_dir()`.
pub fn runtime_store_dir() -> Option<std::path::PathBuf> {
    data_dir().map(|d| d.join("runtimes"))
}

/// Set in every process before ONNX Runtime loads: from 1.29 on Linux and
/// macOS it starts Microsoft's usage-report uploader as it loads, unless
/// this says no. Its API switch (`with_telemetry(false)`) still leaves the
/// uploader running, a device ID under the user's cache and the machine's
/// OS, processor, memory and network type on their way out.
pub const ORT_TELEMETRY_OFF: (&str, &str) = ("ORT_DISABLE_TELEMETRY", "1");

/// SHA verification happens at download time, not load time: re-hashing
/// 200+ MB on every launch would be wasteful.
fn read_on_demand(dir: &std::path::Path, filename: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(filename)).ok()
}

/// The panic guards against a REGISTRY row that says `Bundled` but has
/// no match arm below.
fn bundled_bytes(id: ModelId) -> Arc<Vec<u8>> {
    match id {
        ModelId::Silueta => silueta_bytes(),
        ModelId::BiRefNetLite => birefnet_lite_bytes(),
        ModelId::Teed => teed_bytes(),
        ModelId::YuNet => yunet_bytes(),
        ModelId::U2net
        | ModelId::LaMaFp32
        | ModelId::BigLaMa
        | ModelId::Migan
        | ModelId::SdV15InpaintFp16
        | ModelId::SdV15LcmInpaintFp16
        | ModelId::TaesdFp16
        | ModelId::RealEsrganX4Plus
        | ModelId::Nomos8kSchatL
        | ModelId::RealEsrganX2Plus
        | ModelId::FourXNmkdSiaxCx
        | ModelId::FourXNmkdSuperscale
        | ModelId::FourXNomos2RealPlksr
        | ModelId::Sam21HieraSmall
        | ModelId::FaceGfpgan
        | ModelId::FaceRestoreFormerPp
        | ModelId::RoremFp16
        | ModelId::DdcolorTiny
        | ModelId::DepthAnythingV2Small => {
            panic!("bundled_bytes called for non-Bundled model {id:?} — REGISTRY/source mismatch");
        }
    }
}

// ── Bundled byte loaders ────────────────────────────────────────────────────

/// A bundled model's unpacked weights while anyone holds them.
type Held = std::sync::Mutex<std::sync::Weak<Vec<u8>>>;

/// The unpacked weights `slot` holds while anyone else does, else
/// `unpack`ed afresh. Sessions built together (an engine pool, the first
/// look beside a run) unpack once; once the last of them is built the
/// weights are freed, where a cache kept them for the process's life
/// (210 MB for BiRefNet Lite in a worker, 150 MB of SAM in the app). The
/// lock is held while unpacking, so callers arriving meanwhile wait for
/// that copy instead of making their own.
fn shared_bundled(slot: &Held, unpack: impl FnOnce() -> Vec<u8>) -> Arc<Vec<u8>> {
    let mut held = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(bytes) = held.upgrade() {
        return bytes;
    }
    let bytes = Arc::new(unpack());
    *held = Arc::downgrade(&bytes);
    bytes
}

#[cfg(any(feature = "dev-models", debug_assertions))]
fn workspace_model_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name)
}

/// On-disk model bytes for `dev-models` builds: the raw `.onnx`, or the
/// `.onnx.zst` sibling decompressed. CI caches only the `.zst` files
/// (half the size), so the fallback is what lets tests run there.
#[cfg(feature = "dev-models")]
fn dev_model_bytes(name: &str) -> Vec<u8> {
    dev_model_bytes_in(&workspace_model_path(name))
}

#[cfg(feature = "dev-models")]
fn dev_model_bytes_in(raw: &std::path::Path) -> Vec<u8> {
    if let Ok(bytes) = std::fs::read(raw) {
        return bytes;
    }
    let compressed = raw.with_extension("onnx.zst");
    if let Ok(file) = std::fs::File::open(&compressed) {
        return zstd::stream::decode_all(std::io::BufReader::new(file))
            .unwrap_or_else(|e| panic!("dev-models: {} is not valid zstd: {e}", compressed.display()));
    }
    panic!(
        "dev-models requires {} (or its .zst sibling) — run `cargo xtask fetch-models` from the workspace root",
        raw.display()
    )
}

/// A bundled model's weights accessor, shared through its own `Held`
/// slot: unpacked from the embedded `.zst` in release builds, read from
/// `models/` under `dev-models`. A macro because the arguments
/// themselves are cfg-split (`include_bytes!` takes a literal).
macro_rules! bundled_model {
    ($name:ident, $file:literal, $cap:expr, $what:literal) => {
        pub fn $name() -> Arc<Vec<u8>> {
            static SLOT: Held = Held::new(std::sync::Weak::new());
            #[cfg(not(feature = "dev-models"))]
            static ZST: &[u8] = include_bytes!(concat!("../../../models/", $file, ".zst"));
            #[cfg(not(feature = "dev-models"))]
            let unpack = || {
                zstd::bulk::decompress(ZST, $cap)
                    .expect(concat!("failed to decompress embedded ", $what))
            };
            #[cfg(feature = "dev-models")]
            let unpack = || dev_model_bytes($file);
            shared_bundled(&SLOT, unpack)
        }
    };
}

bundled_model!(silueta_bytes, "silueta.onnx", 50 * 1024 * 1024, "silueta model");
bundled_model!(birefnet_lite_bytes, "birefnet_lite.onnx", 250 * 1024 * 1024, "birefnet-lite model");
bundled_model!(teed_bytes, "teed.onnx", 1024 * 1024, "TEED model");
bundled_model!(yunet_bytes, "face_detection_yunet_2026may.onnx", 1024 * 1024, "YuNet model");
bundled_model!(sam2_encoder_bytes, "sam2.1_hiera_small.encoder.onnx", 200 * 1024 * 1024, "SAM 2 encoder");
bundled_model!(sam2_decoder_bytes, "sam2.1_hiera_small.decoder.onnx", 32 * 1024 * 1024, "SAM 2 decoder");

/// A new `MultiPartBundled` REGISTRY row needs an arm here AND a
/// per-part accessor above; the panic catches a part with no bytes.
fn bundled_multipart_bytes(id: ModelId, part_key: &'static str) -> Arc<Vec<u8>> {
    match (id, part_key) {
        (ModelId::Sam21HieraSmall, "encoder") => sam2_encoder_bytes(),
        (ModelId::Sam21HieraSmall, "decoder") => sam2_decoder_bytes(),
        _ => panic!(
            "bundled_multipart_bytes called for unknown ({id:?}, {part_key:?}) — \
             REGISTRY declares a part that has no byte accessor"
        ),
    }
}

// ── Optimized model variants (FP16 for GPU, INT8 for CPU) ───────────────────
// `None` when the variant isn't on disk or the model has none; the caller
// falls back to FP32.

pub fn model_fp16_bytes(id: ModelId) -> Option<Vec<u8>> {
    load_variant(id, "fp16")
}

pub fn model_int8_bytes(id: ModelId) -> Option<Vec<u8>> {
    load_variant(id, "int8")
}

/// A half-precision copy a model runs only on a card its own weights do
/// not fit, where the card's memory halves while the result stays the
/// same to the eye: the file the CUDA add-on brings, and what it needs.
#[derive(Debug, Clone, Copy)]
pub struct HalfPrecisionCopy {
    pub file: OnDemandVariant,
    pub need: Cost,
}

/// `id`'s copy for small cards (BiRefNet-lite on an RTX 3060: 3.85 GB
/// against 7.7 GB, 0.2 % of mask pixels more than 8 levels off). `None`:
/// the model runs its own weights, or its copy everywhere.
pub fn half_precision_copy(id: ModelId) -> Option<HalfPrecisionCopy> {
    match id {
        ModelId::BiRefNetLite => Some(HalfPrecisionCopy {
            file: OnDemandVariant {
                filename: "birefnet_lite_fp16.onnx",
                url: "https://github.com/aktiwers/prunr/releases/download/models-v1/birefnet_lite_fp16.onnx",
                sha256: "b32371e6c9c916650df9debbfbb14100b4cf5606b57de6ebb88d63cdaff45d59",
                size_mb: 110,
            },
            need: Cost::measured(1_450, 3_850, "RTX 3060 12 GB, i5-11600KF, 341 ms against fp32's 397, 2026-10-30"),
        }),
        _ => None,
    }
}

pub fn half_precision_copies() -> impl Iterator<Item = (ModelId, HalfPrecisionCopy)> {
    REGISTRY.iter().filter_map(|d| half_precision_copy(d.id).map(|copy| (d.id, copy)))
}

/// Whether the half-precision copy of `id` is on disk, without reading it.
pub fn has_fp16_variant(id: ModelId) -> bool {
    variant_path(id, "fp16").is_some()
}

/// Whether the INT8 copy of `id` is on disk, without reading it.
pub fn has_int8_variant(id: ModelId) -> bool {
    variant_path(id, "int8").is_some()
}

fn load_variant(id: ModelId, suffix: &str) -> Option<Vec<u8>> {
    std::fs::read(variant_path(id, suffix)?).ok()
}

fn variant_path(id: ModelId, suffix: &str) -> Option<std::path::PathBuf> {
    // Seg variants are dropped into the models dir by hand as
    // `{stable_name}_fp16.onnx`; RRDB fp16 companions are downloaded
    // under the release artifact's name, the main file's stem.
    let name = match id {
        ModelId::Silueta | ModelId::U2net | ModelId::BiRefNetLite => id.stable_name(),
        ModelId::RealEsrganX4Plus
        | ModelId::RealEsrganX2Plus
        | ModelId::FourXNmkdSiaxCx
        | ModelId::FourXNmkdSuperscale => on_demand_stem(id)?,
        // Nomos8k's published .onnx is already fp16.
        ModelId::Nomos8kSchatL
        | ModelId::FourXNomos2RealPlksr
        | ModelId::Teed
        | ModelId::LaMaFp32
        | ModelId::BigLaMa
        | ModelId::Migan
        | ModelId::SdV15InpaintFp16
        | ModelId::SdV15LcmInpaintFp16
        | ModelId::TaesdFp16
        | ModelId::Sam21HieraSmall
        | ModelId::FaceGfpgan
        | ModelId::FaceRestoreFormerPp
        | ModelId::YuNet
        | ModelId::RoremFp16
        | ModelId::DdcolorTiny
        | ModelId::DepthAnythingV2Small => return None,
    };
    let filename = format!("{name}_{suffix}.onnx");

    #[cfg(debug_assertions)]
    {
        let dev_path = workspace_model_path(&filename);
        if dev_path.is_file() {
            return Some(dev_path);
        }
    }

    if let Some(dir) = on_demand_dir() {
        let path = dir.join(&filename);
        if path.is_file() {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let path = dir.join("models").join(&filename);
            if path.is_file() {
                return Some(path);
            }
        }
    }

    None
}

fn on_demand_stem(id: ModelId) -> Option<&'static str> {
    let ModelSource::OnDemand { filename, .. } = descriptor(id)?.source else { return None };
    filename.strip_suffix(".onnx")
}

#[cfg(all(test, feature = "dev-models"))]
mod dev_model_bytes_tests {
    use super::dev_model_bytes_in;
    use std::io::Write;

    #[test]
    fn falls_back_to_the_zst_sibling_when_the_raw_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("fake.onnx");
        let payload = b"not really onnx but good enough".to_vec();
        let compressed = zstd::bulk::compress(&payload, 3).unwrap();
        std::fs::File::create(dir.path().join("fake.onnx.zst")).unwrap().write_all(&compressed).unwrap();
        assert_eq!(dev_model_bytes_in(&raw), payload);

        std::fs::write(&raw, b"raw wins").unwrap();
        assert_eq!(dev_model_bytes_in(&raw), b"raw wins");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry's own answer, without what a machine measured.
    fn place(model: ModelId, provider: &str, intel_gpu: bool, remeasuring: bool) -> Place {
        place_in(&registry_places(model), model, provider, intel_gpu, remeasuring)
    }

    fn column_for(model: ModelId, provider: &str, intel_gpu: bool, remeasuring: bool) -> Option<Column> {
        column_in(&registry_places(model), model, provider, intel_gpu, remeasuring)
    }
    use std::collections::HashSet;

    fn ids_in(category: ModelCategory) -> Vec<ModelId> {
        REGISTRY.iter().filter(|d| d.category == category).map(|d| d.id).collect()
    }

    #[test]
    fn every_model_id_has_a_registry_entry() {
        // Add a ModelId variant ⇒ add a REGISTRY row. Guards against a
        // half-finished addition that compiles but has no metadata.
        for &id in ModelId::ALL {
            assert!(descriptor(id).is_some(), "ModelId::{id:?} missing from REGISTRY");
            assert_eq!(display_name(id), descriptor(id).unwrap().display_name);
        }

        // Exhaustiveness fence: adding a variant to `ModelId` without
        // adding an arm here is a compile error. The arm body is empty;
        // its job is to force the developer to look at this test, where
        // the comment above tells them to also extend `ModelId::ALL`.
        // (Rust enums can't be reflected in stable; removal-from-ALL
        // alone is silent — a deliberate code change a reviewer catches.)
        #[allow(non_snake_case)]
        fn _force_developer_to_update_ALL_too(id: ModelId) {
            match id {
                ModelId::Silueta
                | ModelId::U2net
                | ModelId::BiRefNetLite
                | ModelId::Teed
                | ModelId::LaMaFp32
                | ModelId::BigLaMa
                | ModelId::Migan
                | ModelId::SdV15InpaintFp16
                | ModelId::SdV15LcmInpaintFp16
                | ModelId::TaesdFp16
                | ModelId::RealEsrganX4Plus
                | ModelId::Nomos8kSchatL
                | ModelId::RealEsrganX2Plus
                | ModelId::FourXNmkdSiaxCx
                | ModelId::FourXNmkdSuperscale
                | ModelId::FourXNomos2RealPlksr
                | ModelId::Sam21HieraSmall
                | ModelId::FaceGfpgan
                | ModelId::FaceRestoreFormerPp
                | ModelId::YuNet
                | ModelId::RoremFp16
                | ModelId::DdcolorTiny
                | ModelId::DepthAnythingV2Small => {}
            }
        }
    }

    /// Faces are found by the built-in YuNet, so a face model downloads
    /// only its restorer, and an install that also holds the detector it
    /// once downloaded is complete as it is.
    #[test]
    fn faces_are_found_by_the_built_in_finder() {
        assert!(descriptor(ModelId::YuNet).is_some_and(|d| d.source.is_bundled() && d.category == ModelCategory::Face));
        for id in [ModelId::FaceGfpgan, ModelId::FaceRestoreFormerPp] {
            let Some(ModelSource::MultiPartOnDemand { parts, .. }) = descriptor(id).map(|d| d.source) else { panic!("{id:?} is a download") };
            assert_eq!(parts.iter().map(|p| p.key).collect::<Vec<_>>(), ["restorer"], "{id:?}");
        }
    }

    /// Settings › Credits is where the models are credited (THIRD-PARTY.md
    /// points there): every model, bundled ones too, names its authors,
    /// licence and source.
    #[test]
    fn every_model_names_its_authors_license_and_source() {
        for d in REGISTRY {
            let info = d.license();
            assert!(!info.author.is_empty() && !info.license.is_empty() && !info.source_url.is_empty(), "{:?}: {info:?}", d.id);
        }
    }

    /// Only a segmentation model builds an engine pool, so only one can
    /// gain from an engine per job.
    #[test]
    fn only_segmentation_models_pay_for_parallel_jobs() {
        for d in REGISTRY {
            assert!(!d.parallel_jobs_pay_on_cpu || d.category == ModelCategory::Segmentation, "{:?}", d.id);
        }
    }

    #[test]
    fn bundled_models_must_not_require_gpu() {
        // `card_action` short-circuits on Bundled before checking GPU
        // requirement. Pin "Bundled implies !Required" so a future
        // Bundled entry with `Required` doesn't silently render as
        // Built-in while ignoring its own gate.
        for desc in REGISTRY {
            if matches!(desc.source, ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. }) {
                assert!(
                    !matches!(desc.gpu, GpuRequirement::Required),
                    "Bundled descriptor {:?} declares GpuRequirement::Required — \
                     this combination is unreachable from the Model Store gate",
                    desc.id,
                );
            }
        }
    }

    #[test]
    fn registry_ids_are_unique() {
        let mut seen: HashSet<ModelId> = HashSet::new();
        for d in REGISTRY {
            assert!(seen.insert(d.id), "duplicate REGISTRY entry: {:?}", d.id);
        }
    }

    #[test]
    fn segmentation_category_holds_the_seg_models_only() {
        let ids = ids_in(ModelCategory::Segmentation);
        assert!(ids.contains(&ModelId::Silueta));
        assert!(ids.contains(&ModelId::U2net));
        assert!(ids.contains(&ModelId::BiRefNetLite));
        assert!(!ids.contains(&ModelId::Teed));
        assert!(!ids.contains(&ModelId::LaMaFp32));
    }

    #[test]
    fn bundled_descriptors_are_always_available() {
        for desc in REGISTRY {
            if matches!(desc.source, ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. }) {
                assert!(is_available(desc.id), "Bundled {:?} reports unavailable", desc.id);
            }
        }
    }

    #[test]
    fn ondemand_descriptors_have_complete_metadata() {
        for desc in REGISTRY {
            if let ModelSource::OnDemand { filename, url, sha256, size_mb, license, fp16 } = desc.source {
                let is_placeholder = sha256.starts_with("PLACEHOLDER");
                assert!(!filename.is_empty(), "{:?} filename empty", desc.id);
                assert!(url.starts_with("https://"), "{:?} non-HTTPS url: {url}", desc.id);
                if !is_placeholder {
                    assert_eq!(sha256.len(), 64, "{:?} sha256 not 64 hex chars: {sha256}", desc.id);
                    assert!(size_mb > 0, "{:?} size_mb=0", desc.id);
                }
                assert!(!license.license.is_empty(), "{:?} license empty", desc.id);
                assert!(license.license_url.starts_with("https://"), "{:?} bad license_url", desc.id);
                assert!(license.source_url.starts_with("https://"), "{:?} bad source_url", desc.id);
                if let Some(variant) = fp16 {
                    assert_variant_complete(desc.id, variant);
                }
            }
        }
        for (id, copy) in half_precision_copies() {
            assert_variant_complete(id, copy.file);
        }
    }

    fn assert_variant_complete(id: ModelId, variant: OnDemandVariant) {
        assert!(!variant.filename.is_empty(), "{id:?} {} filename empty", variant.filename);
        assert!(variant.url.starts_with("https://"), "{id:?} {} non-HTTPS url", variant.filename);
        assert!(variant.sha256.len() == 64 && variant.sha256.bytes().all(|b| b.is_ascii_hexdigit()), "{id:?} {} sha256 not 64 hex chars", variant.filename);
        assert!(variant.size_mb > 0, "{id:?} {} size_mb=0", variant.filename);
    }

    /// A graph whose weights live in separate `.data` files resolves
    /// them next to the graph: a copy of the graph elsewhere (an
    /// optimized-model cache) cannot load.
    #[test]
    fn only_rorem_keeps_its_weights_apart_from_its_graph() {
        for &id in ModelId::ALL {
            assert_eq!(keeps_weights_apart(id), id == ModelId::RoremFp16, "{id:?}");
        }
    }

    #[test]
    fn is_sd_family_routes_only_sd_models_to_subprocess() {
        // SD-family routes to the dedicated subprocess (5-10 GB bundle
        // — process isolation prevents an OOM from killing the GUI);
        // LaMa, Big-LaMa and MI-GAN stay in-process, as does every
        // non-inpaint model.
        for &id in ModelId::ALL {
            let sd = matches!(id, ModelId::SdV15InpaintFp16 | ModelId::SdV15LcmInpaintFp16 | ModelId::RoremFp16);
            assert_eq!(id.is_sd_family(), sd, "{id:?}");
        }
    }

    #[test]
    fn rorem_denoises_sdxl_latents_and_only_sd15_takes_taesd() {
        let latent = |id: ModelId| id.sd_spec().map(|s| s.latent);
        assert_eq!(latent(ModelId::SdV15InpaintFp16), Some(SdLatent::Sd15));
        assert_eq!(latent(ModelId::SdV15LcmInpaintFp16), Some(SdLatent::Sd15));
        assert_eq!(latent(ModelId::RoremFp16), Some(SdLatent::Sdxl));
        assert_eq!((SdLatent::Sd15.vae_scale(), SdLatent::Sdxl.vae_scale()), (0.18215, 0.13025));
        assert!(SdLatent::Sd15.takes_taesd() && !SdLatent::Sdxl.takes_taesd());
    }

    #[test]
    fn every_descriptor_has_a_calibrated_working_set() {
        // Floor catches accidental zero-init when adding a new model;
        // ceiling catches obvious typos (no real model needs > 64 GB).
        for desc in REGISTRY {
            let costs = std::iter::once(desc.needs.cpu).chain([("Mac CPU", desc.needs.mac_cpu)].into_iter().chain(desc.needs.accelerators()).filter_map(|(_, p)| match p {
                Priced::Same(cost) => Some(cost),
                Priced::Unmeasured => None,
            }));
            for cost in costs {
                assert!(cost.host_mb + cost.card_mb >= 50, "{:?} {cost:?} too small: every real model uses ≥ 50 MB", desc.id);
                assert!(cost.host_mb < 65_536 && cost.card_mb < 65_536, "{:?} {cost:?} suspiciously large (> 64 GB)", desc.id);
                let (Basis::Measured(why) | Basis::Estimate(why)) = cost.basis;
                assert!(!why.trim().is_empty(), "{:?} {cost:?}: a figure says where it comes from", desc.id);
            }
        }
    }

    /// An SD bundle measured on a card says what the tall crop adds there.
    #[test]
    fn an_sd_bundle_on_a_card_prices_its_tall_crop_there() {
        for desc in REGISTRY {
            let Some(spec) = desc.id.sd_spec() else { continue };
            let on_a_card = desc.needs.accelerators().into_iter().any(|(_, p)| matches!(p, Priced::Same(c) if c.card_mb > 0));
            assert_eq!(on_a_card, spec.tall_crop_card_extra_mb.is_some(), "{:?}", desc.id);
        }
    }

    /// A model is priced on every accelerator, and no more accelerators
    /// left unmeasured than counted here. Measuring one lowers its count;
    /// a new model left unmeasured raises it, on purpose, here.
    #[test]
    fn every_model_prices_every_accelerator() {
        const UNMEASURED_AT_MOST: [(&str, usize); 5] =
            [("CUDA", 6), ("DirectML", 9), ("CoreML", 14), ("OpenVINO", 21), ("MIGraphX", 23)];
        let mut unmeasured: Vec<(&str, Vec<ModelId>)> = UNMEASURED_AT_MOST.iter().map(|&(p, _)| (p, Vec::new())).collect();
        for desc in REGISTRY {
            for (provider, priced) in desc.needs.accelerators() {
                if priced == Priced::Unmeasured {
                    // provider names come from the same table
                    unmeasured.iter_mut().find(|(p, _)| *p == provider).unwrap().1.push(desc.id);
                }
            }
        }
        const MAC_CPU_UNMEASURED_AT_MOST: usize = 10;
        let mac_unmeasured: Vec<ModelId> = REGISTRY.iter().filter(|d| d.needs.mac_cpu == Priced::Unmeasured).map(|d| d.id).collect();
        assert_eq!(mac_unmeasured.len(), MAC_CPU_UNMEASURED_AT_MOST, "unmeasured on the Mac's CPU: {mac_unmeasured:?}; set the count");
        for ((provider, at_most), (_, models)) in UNMEASURED_AT_MOST.iter().zip(&unmeasured) {
            assert!(models.len() <= *at_most, "{} models unmeasured on {provider}, more than {at_most}: measure them ({models:?})", models.len());
            assert!(models.len() == *at_most, "only {} unmeasured on {provider} now: lower its count to {}", models.len(), models.len());
        }
    }

    #[test]
    fn multi_part_descriptors_have_at_least_one_part() {
        for desc in REGISTRY {
            if let ModelSource::MultiPartOnDemand { subdir, parts, license, .. } = desc.source {
                assert!(!subdir.is_empty(), "{:?} subdir empty", desc.id);
                assert!(!parts.is_empty(),  "{:?} multi-part has zero parts", desc.id);
                for p in parts {
                    assert!(!p.key.is_empty(),       "{:?} part key empty",       desc.id);
                    assert!(!p.filename.is_empty(),  "{:?} part filename empty",  desc.id);
                    assert!(p.url.starts_with("https://"), "{:?} part non-HTTPS: {}", desc.id, p.url);
                    assert_eq!(p.sha256.len(), 64,   "{:?} part sha256 not 64 hex chars", desc.id);
                    assert!(p.size_bytes > 0,        "{:?} part size_bytes=0",    desc.id);
                }
                assert!(license.license_url.starts_with("https://"), "{:?} bad license_url", desc.id);
                assert!(license.source_url.starts_with("https://"),  "{:?} bad source_url",  desc.id);
            }
        }
    }

    #[test]
    fn resolve_bytes_returns_some_for_bundled() {
        // Silueta is the cheapest decompression to validate the dispatch
        // path; the others would allocate hundreds of MB at test time.
        let bytes = resolve_bytes(ModelId::Silueta).expect("Silueta must resolve");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn load_variant_returns_none_for_models_without_variants() {
        // Models whose `variant_path` match arm explicitly returns None
        // (no variant lookup possible regardless of disk state). Update
        // this set when a new model's arm in `variant_path` changes from
        // a return-None arm to a stem-returning arm.
        //
        // Excludes (these resolve to a filename stem and may legitimately
        // pick up a real `_fp16.onnx` / `_int8.onnx` file from the user's
        // dev path / on_demand_dir / exe-adjacent dir, so asserting None
        // here would be machine-dependent):
        //   Silueta, U2net, BiRefNetLite — seg fp16/int8 dev artifacts
        //   RealEsrganX4Plus, RealEsrganX2Plus, FourXNmkdSiaxCx,
        //   FourXNmkdSuperscale — RRDB upscale fp16 siblings
        let resolves_to_filename_stem = |id| matches!(
            id,
            ModelId::Silueta
                | ModelId::U2net
                | ModelId::BiRefNetLite
                | ModelId::RealEsrganX4Plus
                | ModelId::RealEsrganX2Plus
                | ModelId::FourXNmkdSiaxCx
                | ModelId::FourXNmkdSuperscale,
        );
        for &id in ModelId::ALL {
            if resolves_to_filename_stem(id) {
                continue;
            }
            assert!(model_fp16_bytes(id).is_none(), "{id:?} fp16");
            assert!(model_int8_bytes(id).is_none(), "{id:?} int8");
        }
    }

    #[test]
    fn on_demand_dir_path_ends_with_prunr_models() {
        let p = on_demand_dir().expect("data_dir resolves on supported platforms");
        assert!(p.ends_with(std::path::Path::new("prunr/models")),
            "unexpected path tail: {}", p.display());
    }

    #[test]
    fn a_download_is_read_for_each_load_and_not_kept() {
        let dir = tempfile::tempdir().unwrap();
        let ModelSource::OnDemand { filename, .. } = descriptor(ModelId::U2net).unwrap().source else {
            panic!("U2Net is a download");
        };
        std::fs::write(dir.path().join(filename), b"weights").unwrap();
        let bytes = resolve_bytes_from(ModelId::U2net, Some(dir.path())).expect("on disk");
        assert_eq!(
            Arc::strong_count(&bytes), 1,
            "a session copies the weights it is built from: a kept copy holds the file's size in RAM for nothing",
        );
        assert_eq!(&*bytes, b"weights");
        std::fs::write(dir.path().join(filename), b"redownloaded").unwrap();
        assert_eq!(&*resolve_bytes_from(ModelId::U2net, Some(dir.path())).unwrap(), b"redownloaded");
    }

    #[test]
    fn read_on_demand_returns_none_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_on_demand(dir.path(), "nope.onnx").is_none());
    }

    #[test]
    fn downloaded_models_carry_a_content_tag_and_built_in_ones_none() {
        let mut seen = std::collections::HashSet::new();
        for d in REGISTRY {
            let tag = content_tag(d.id);
            match d.source {
                ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => assert_eq!(tag, None, "{:?}", d.id),
                ModelSource::OnDemand { .. } | ModelSource::MultiPartOnDemand { .. } => {
                    let tag = tag.unwrap_or_else(|| panic!("{:?} has no tag", d.id));
                    assert!(seen.insert(tag), "{:?} shares a tag", d.id);
                }
            }
        }
        assert_ne!(tag_of_shas(["aa", "bb"]), tag_of_shas(["aa", "bc"]), "any part's checksum moves the tag");
    }

    /// A provider is matched in any case, OpenVINO by whether it may use
    /// the Intel graphics; a model runs wherever its place is not the CPU.
    #[test]
    fn a_model_runs_where_its_place_is_not_the_cpu() {
        let off = |model, ep, on_gpu| place(model, ep, on_gpu, false).kept_off(false);
        for ep in ["OpenVINO", "openvino", "OPENVINO"] {
            for on_gpu in [false, true] {
                assert!(off(ModelId::Silueta, ep, on_gpu), "Silueta + {ep}");
            }
        }
        for ep in ["CPU", "CUDA", "DirectML", "MIGraphX"] {
            assert!(!off(ModelId::Silueta, ep, false), "Silueta + {ep}");
        }
        assert!(off(ModelId::Silueta, "CoreML", false), "no gain, more memory: on the CPU");
        assert!(off(ModelId::Nomos8kSchatL, "cuda", false), "a different result keeps it on the CPU");
        assert_eq!(place(ModelId::Silueta, "TensorRT", false, false), Place::Unmeasured, "an unknown provider is unmeasured");
    }

    /// A place on the CPU keeps the model off the accelerator; while places
    /// are measured again only a failing one does, since it may hang or
    /// take the machine's memory.
    #[test]
    fn a_cpu_place_keeps_the_model_off_unless_measured_again() {
        for (place, kept, kept_while_measuring) in [
            (Place::Unmeasured, false, false),
            (Place::There(Speed::on(300.0, 400.0, "test")), false, false),
            (Place::Cpu(Why::Slower("0.8x")), true, false),
            (Place::Cpu(Why::Differs("black")), true, false),
            (Place::Cpu(Why::NoGain("1.05x, heavier")), true, false),
            (Place::Cpu(Why::Fails("hangs")), true, true),
        ] {
            assert_eq!((place.kept_off(false), place.kept_off(true)), (kept, kept_while_measuring), "{place:?}");
            assert_eq!(place.fails(), kept_while_measuring, "{place:?}");
            assert_eq!(place.runs_there(), matches!(place, Place::There(_)), "{place:?}");
        }
    }

    /// A speed reads as the bench wrote it: the ratio and both times, what
    /// they leave out, then where.
    #[test]
    fn a_speed_reads_as_its_times_then_where() {
        assert_eq!(Speed::on(36.42, 618.0, "RTX 3060").to_string(), "16.97x: 36 ms against the CPU's 618 ms (RTX 3060)");
        assert_eq!(
            Speed::on(47110.0, 126000.0, "i7-6700").but("first runs").to_string(),
            "2.67x: 47.11 s against the CPU's 2m 06s, first runs (i7-6700)",
        );
        assert_eq!(fmt_ms(3.84), "3.8 ms");
    }

    /// Each column is found by its name and by its provider, and sits at
    /// its index in `ALL`.
    #[test]
    fn a_column_is_found_by_name_and_by_provider() {
        for (i, column) in Column::ALL.into_iter().enumerate() {
            assert_eq!(Column::named(column.name()), Some(column));
            assert_eq!(column.index(), i);
        }
        assert_eq!(Column::of("openvino"), Some(Column::OpenVino));
        assert_eq!(Column::of("CPU"), None);
    }

    /// With the Intel graphics on, OpenVINO reads a model's graphics entry
    /// only where that entry keeps it there; a model the graphics run
    /// slower reads the processor's entry, and a fill never reads it.
    #[test]
    fn openvino_reads_the_graphics_entry_only_where_it_keeps_the_model_there() {
        let column = |model, on, remeasuring| column_for(model, "OpenVINO", on, remeasuring);
        assert_eq!(column(ModelId::RealEsrganX4Plus, true, false), Some(Column::OpenVinoGpu));
        assert_eq!(column(ModelId::DepthAnythingV2Small, true, false), Some(Column::OpenVino), "0.65x on the graphics");
        let processor = descriptor(ModelId::DepthAnythingV2Small).unwrap().places.openvino;
        assert!(matches!(processor, Place::There(_)), "Depth runs on OpenVINO's processor");
        assert_eq!(place(ModelId::DepthAnythingV2Small, "OpenVINO", true, false), processor);
        assert_eq!(column(ModelId::DepthAnythingV2Small, true, true), Some(Column::OpenVinoGpu), "measured again by the bench");
        assert_eq!(column(ModelId::Nomos8kSchatL, true, true), Some(Column::OpenVino), "a failing entry, even when measuring");
        assert_eq!(column(ModelId::LaMaFp32, true, false), Some(Column::OpenVino), "a fill");
        assert_eq!(column(ModelId::RealEsrganX4Plus, false, false), Some(Column::OpenVino), "the switch is off");
        assert_eq!(column_for(ModelId::RealEsrganX4Plus, "CUDA", true, false), Some(Column::Cuda), "the switch is OpenVINO's only");
    }

    /// A fill never runs on the Intel graphics (OpenVINO keeps fills on
    /// the processor), so its entry there never says it runs there.
    #[test]
    fn a_fill_never_runs_on_the_intel_graphics() {
        for desc in REGISTRY.iter().filter(|d| d.category == ModelCategory::Inpaint) {
            assert!(!matches!(desc.places.openvino_gpu, Place::There(_)), "{:?}", desc.id);
        }
    }

    /// Every place says where it was measured, and no more columns are
    /// left unmeasured than counted here: measuring lowers a count, a new
    /// model left unmeasured raises it, on purpose, here.
    #[test]
    fn every_model_says_where_it_runs() {
        const UNMEASURED: [(&str, usize); 6] = [
            ("CUDA", 4),
            ("DirectML", 5),
            ("CoreML", 4),
            ("MIGraphX", 23),
            ("OpenVINO", 3),
            ("OpenVINO (Intel graphics on)", 3),
        ];
        for (i, &(column, at_most)) in UNMEASURED.iter().enumerate() {
            let mut unmeasured = Vec::new();
            for desc in REGISTRY {
                assert_eq!(Column::ALL[i].name(), column, "the counts follow `Column::ALL`");
                let place = desc.places.get(Column::ALL[i]);
                match place {
                    Place::Unmeasured => unmeasured.push(desc.id),
                    Place::There(speed) => assert!(!speed.on.trim().is_empty(), "{:?} on {column}: say where it was measured", desc.id),
                    Place::Cpu(why) => assert!(!why.text().trim().is_empty(), "{:?} on {column}: say why", desc.id),
                }
            }
            assert_eq!(unmeasured.len(), at_most, "unmeasured on {column}: {unmeasured:?}; set the count");
        }
    }

    #[test]
    fn prunr_models_has_no_workspace_deps() {
        // prunr-models must remain a leaf — no path deps on sibling workspace crates.
        // Adding one would risk creating circular deps through prunr-core → prunr-models.
        let toml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")
        ).expect("Cargo.toml must be readable");
        assert!(
            !toml.contains(r#"path = ".."#),
            "prunr-models must remain a leaf crate with no workspace-internal path deps — see CLAUDE.md Layers table"
        );
    }

    #[test]
    fn upscale_attribution_required_matches_license_class() {
        for desc in REGISTRY {
            let lic = desc.license();
            if lic.license.starts_with("CC-BY") {
                assert!(
                    desc.attribution_required,
                    "{:?} has CC-BY license but attribution_required=false",
                    desc.id
                );
            } else if matches!(lic.license, "BSD-3-Clause" | "MIT" | "Apache-2.0") {
                assert!(
                    !desc.attribution_required,
                    "{:?} has permissive license but attribution_required=true",
                    desc.id
                );
            }
        }
    }

    /// `upscale: Option<UpscaleModelKnobs>` carries the per-model
    /// dispatch knobs the upscale path reads (input-tensor name, fp16
    /// flag). It must be `Some` iff `category == Upscale` — a mismatch
    /// either makes a Segmentation/Inpaint descriptor accidentally
    /// expose upscale knobs, or leaves an Upscale entry without the
    /// data the dispatcher needs. Either direction silently breaks
    /// the data-only "add another upscale model" contract.
    #[test]
    fn upscale_knobs_set_iff_category_is_upscale() {
        for desc in REGISTRY {
            match desc.category {
                ModelCategory::Upscale => assert!(
                    desc.upscale.is_some(),
                    "{:?} is Upscale category but upscale=None — \
                     dispatcher would read default knobs",
                    desc.id
                ),
                _ => assert!(
                    desc.upscale.is_none(),
                    "{:?} is non-Upscale category but upscale=Some — \
                     non-upscale models must not carry upscale knobs",
                    desc.id
                ),
            }
        }
    }

    /// A network that normalizes over its whole input ships cut at its
    /// norms: a head, its blocks, a tail and the norms' weights, which
    /// the part-by-part runner reads by these keys. Only Nomos2 is one.
    #[test]
    fn a_whole_input_norm_upscaler_ships_in_parts() {
        let parted: Vec<ModelId> = REGISTRY.iter().filter(|d| d.runs_in_parts()).map(|d| d.id).collect();
        assert_eq!(parted, [ModelId::FourXNomos2RealPlksr]);
        for id in parted {
            let ModelSource::MultiPartOnDemand { parts, .. } = descriptor(id).unwrap().source else { panic!("{id:?} is not in parts") };
            let keys: Vec<&str> = parts.iter().map(|p| p.key).collect();
            for key in [PART_HEAD, PART_TAIL, PART_NORMS] {
                assert!(keys.contains(&key), "{id:?} lacks {key}");
            }
            let blocks: Vec<&&str> = keys.iter().filter(|k| k.starts_with("block")).collect();
            assert_eq!(blocks.len(), 28, "{id:?}: its 28 blocks");
            assert!(blocks.windows(2).all(|w| w[0] < w[1]), "blocks in order by key");
            assert!(parts.iter().all(|p| p.sha256.len() == 64 && p.size_bytes > 0 && p.url.ends_with(p.filename)));
        }
    }

    /// Each upscaler's dispatch knobs, as its export needs them: the
    /// input tensor's name and type, its native scale, the 32 px overlap
    /// a window-attention transformer needs (64 px for a CNN), and the
    /// tile multiple HAT's windows and RRDB's pixel_unshuffle demand.
    #[test]
    fn every_upscaler_carries_the_knobs_its_export_needs() {
        // (id, input, fp16, native scale, window attention, tile multiple, overlap, tile)
        let table = [
            (ModelId::RealEsrganX4Plus, "data", false, 4, false, None, 64, Some(512)),
            (ModelId::RealEsrganX2Plus, "data", false, 2, false, Some(2), 64, Some(512)),
            (ModelId::Nomos8kSchatL, "input", true, 4, true, Some(16), 32, Some(256)),
            (ModelId::FourXNmkdSiaxCx, "data", false, 4, false, None, 64, Some(512)),
            (ModelId::FourXNmkdSuperscale, "data", false, 4, false, None, 64, Some(512)),
            // Run in parts, never tiled.
            (ModelId::FourXNomos2RealPlksr, "input", false, 4, false, None, 64, None),
        ];
        assert_eq!(table.len(), ids_in(ModelCategory::Upscale).len(), "a row for every upscaler");
        for (id, input, fp16, scale, window, multiple, overlap, tile) in table {
            let desc = descriptor(id).unwrap_or_else(|| panic!("{id:?} in REGISTRY"));
            let knobs = desc.upscale.unwrap_or_else(|| panic!("{id:?} has upscale knobs"));
            assert_eq!(
                (knobs.input_name, knobs.is_fp16, knobs.native_scale, knobs.uses_window_attention, desc.tile_size_multiple, knobs.tile_overlap(), desc.recommended_tile),
                (input, fp16, scale, window, multiple, overlap, tile),
                "{id:?}",
            );
        }
    }

    /// Gigabytes of models and runtimes stay outside the roaming folder a
    /// domain profile copies from computer to computer.
    #[cfg(windows)]
    #[test]
    fn the_data_stays_on_this_computer() {
        assert!(!data_dir().expect("a data dir").starts_with(dirs::data_dir().expect("a roaming data dir")));
    }

    /// A copy for small cards is the file the engine reads as the model's
    /// half-precision weights, and its address names that file.
    #[test]
    fn a_small_card_copy_is_the_file_the_engine_reads() {
        assert!(half_precision_copies().count() > 0);
        for (id, copy) in half_precision_copies() {
            assert_eq!(copy.file.filename, format!("{}_fp16.onnx", id.stable_name()), "{id:?}");
            assert!(copy.file.url.ends_with(copy.file.filename), "{id:?}");
        }
    }

    /// The sweep removes the retired names it finds and nothing else.
    #[test]
    fn the_retired_sweep_removes_only_retired_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("old.onnx"), b"old").expect("write");
        std::fs::write(dir.path().join("empty.onnx"), b"").expect("write");
        std::fs::write(dir.path().join("kept.onnx"), b"kept").expect("write");
        std::fs::create_dir(dir.path().join("old-dir")).expect("mkdir");
        assert_eq!(remove_retired_in(dir.path(), &["old.onnx", "empty.onnx", "missing.onnx", "old-dir"]), 3);
        assert!(!dir.path().join("old.onnx").exists() && !dir.path().join("empty.onnx").exists());
        assert!(dir.path().join("kept.onnx").is_file());
        assert!(dir.path().join("old-dir").is_dir(), "a folder is never swept");
    }

    /// A retired name is never one a model downloads now.
    #[test]
    fn no_retired_file_is_one_a_model_reads() {
        for d in REGISTRY {
            let live: Vec<String> = match d.source {
                ModelSource::OnDemand { filename, fp16, .. } => std::iter::once(filename).chain(fp16.map(|v| v.filename)).map(String::from).collect(),
                ModelSource::MultiPartOnDemand { subdir, parts, .. } => std::iter::once(subdir.to_string()).chain(parts.iter().map(|part| format!("{subdir}/{}", part.filename))).collect(),
                ModelSource::Bundled { .. } | ModelSource::MultiPartBundled { .. } => Vec::new(),
            };
            for name in live {
                assert!(!RETIRED_FILES.contains(&name.as_str()), "{:?} reads {name}", d.id);
            }
        }
    }

    // ── RealEsrganX2Plus tests ────────────────────────────────────────────

    #[test]
    fn x2plus_sha256_is_real_hex() {
        let desc = descriptor(ModelId::RealEsrganX2Plus).expect("x2plus in REGISTRY");
        if let ModelSource::OnDemand { sha256, .. } = desc.source {
            assert_eq!(sha256.len(), 64, "sha256 must be 64 hex chars, got {}", sha256.len());
            assert!(
                sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
                "sha256 must be lowercase hex, got: {sha256}"
            );
            assert!(
                !sha256.starts_with("PLACEHOLDER"),
                "sha256 must not be a placeholder"
            );
        } else {
            panic!("RealEsrganX2Plus must be OnDemand");
        }
    }

    // ── NMKD community RRDB models ────────────────────────────────────────

    #[test]
    fn nmkd_models_have_wtfpl_license() {
        for id in [ModelId::FourXNmkdSiaxCx, ModelId::FourXNmkdSuperscale] {
            let desc = descriptor(id).unwrap();
            let license = match desc.source {
                ModelSource::OnDemand { license, .. } => license,
                _ => panic!("{id:?} must be OnDemand"),
            };
            assert_eq!(license.license, "WTFPL");
            assert_eq!(license.author, "Nmkd");
            assert!(!desc.attribution_required, "WTFPL does not require attribution");
        }
    }

    // ── fp16 variants ─────────────────────────────────────────────────────

    #[test]
    fn rrdb_upscale_models_carry_fp16_companion() {
        // All four RRDB upscale entries must declare an fp16 sibling.
        // Nomos8k's main file IS fp16 (no sibling needed) — that's
        // covered separately.
        for id in [
            ModelId::RealEsrganX4Plus,
            ModelId::RealEsrganX2Plus,
            ModelId::FourXNmkdSiaxCx,
            ModelId::FourXNmkdSuperscale,
        ] {
            let desc = descriptor(id).unwrap();
            let fp16 = match desc.source {
                ModelSource::OnDemand { fp16, .. } => fp16,
                _ => panic!("{id:?} must be OnDemand"),
            };
            let variant = fp16.unwrap_or_else(|| {
                panic!("{id:?} RRDB upscale model must declare an fp16 companion")
            });
            assert!(
                variant.filename.ends_with("_fp16.onnx"),
                "{id:?} fp16 filename must end with _fp16.onnx, got {}",
                variant.filename
            );
            assert!(
                variant.url.contains("_fp16.onnx"),
                "{id:?} fp16 url must reference _fp16.onnx"
            );
            // fp16 file should be roughly half the fp32 size (within
            // 5 MB tolerance for ONNX graph metadata overhead).
            let main_size = match desc.source {
                ModelSource::OnDemand { size_mb, .. } => size_mb,
                _ => unreachable!(),
            };
            let expected_half = main_size / 2;
            let delta = (variant.size_mb as i32 - expected_half as i32).abs();
            assert!(
                delta <= 5,
                "{id:?} fp16 size {} MB not roughly half of fp32 {} MB",
                variant.size_mb, main_size
            );
        }
    }

    /// The fetcher saves the fp16 companion under its registry filename;
    /// `variant_path` must look for that same name.
    #[test]
    fn rrdb_fp16_lookup_name_is_the_downloaded_companion() {
        for id in [
            ModelId::RealEsrganX4Plus,
            ModelId::RealEsrganX2Plus,
            ModelId::FourXNmkdSiaxCx,
            ModelId::FourXNmkdSuperscale,
        ] {
            let Some(ModelSource::OnDemand { fp16: Some(variant), .. }) = descriptor(id).map(|d| d.source) else {
                panic!("{id:?} must be OnDemand with an fp16 companion")
            };
            let stem = on_demand_stem(id).expect("filename ends in .onnx");
            assert_eq!(format!("{stem}_fp16.onnx"), variant.filename, "{id:?}");
        }
    }

    #[test]
    fn depth_anything_is_on_demand_depth() {
        let desc = descriptor(ModelId::DepthAnythingV2Small).expect("DepthAnythingV2Small in REGISTRY");
        assert_eq!(desc.category, ModelCategory::Depth);
        assert!(matches!(desc.source, ModelSource::OnDemand { filename: "depth_anything_v2_small.onnx", .. }));
        assert_eq!(ids_in(ModelCategory::Depth), [ModelId::DepthAnythingV2Small]);
    }

    #[test]
    fn ddcolor_is_on_demand_colorize() {
        let desc = descriptor(ModelId::DdcolorTiny).expect("DdcolorTiny in REGISTRY");
        assert_eq!(desc.category, ModelCategory::Colorize);
        assert!(matches!(desc.source, ModelSource::OnDemand { filename: "ddcolor_paper_tiny.onnx", .. }));
        assert!(desc.upscale.is_none());
        assert_eq!(ids_in(ModelCategory::Colorize), [ModelId::DdcolorTiny]);
    }

    // ── SAM 2.1 Hiera Small boundary tests ─────────────────────────────────

    #[test]
    fn sam2_registry_entry() {
        let desc = descriptor(ModelId::Sam21HieraSmall)
            .expect("Sam21HieraSmall must be registered");
        assert_eq!(desc.category, ModelCategory::Selection);
        assert!(matches!(desc.source, ModelSource::MultiPartBundled { .. }));
        assert_eq!(desc.needs.cpu.host_mb, 1_150);
        assert!(!desc.attribution_required);
        assert!(desc.upscale.is_none());
    }

    #[test]
    fn sam2_license_is_apache_2_no_acceptance_required() {
        let desc = descriptor(ModelId::Sam21HieraSmall).unwrap();
        let ModelSource::MultiPartBundled { license, .. } = desc.source else {
            panic!("expected MultiPartBundled");
        };
        assert_eq!(license.license, "Apache 2.0");
        assert!(!desc.requires_license_acceptance());
    }

    /// Pins the contract that encoder and decoder are SEPARATE ONNX files —
    /// dispatched through SEPARATE OrtEngine instances. If a future change
    /// consolidates them into one part, this test fails.
    #[test]
    fn sam2_encoder_decoder_separate_engines() {
        let desc = descriptor(ModelId::Sam21HieraSmall).unwrap();
        let ModelSource::MultiPartBundled { parts, .. } = desc.source else {
            panic!("expected MultiPartBundled");
        };
        assert_eq!(parts.len(), 2);
        let filenames: std::collections::HashSet<&str> =
            parts.iter().map(|p| p.filename).collect();
        assert_eq!(filenames.len(), parts.len(),
            "every part must have a distinct filename");
        assert!(parts.iter().any(|p| p.key == "encoder" && p.filename.contains("encoder")));
        assert!(parts.iter().any(|p| p.key == "decoder" && p.filename.contains("decoder")));
    }

    /// `resolve_part_bytes` for a bundled multi-part returns real bytes —
    /// pins the contract that Magic Brush dispatch can load SAM 2 without
    /// any prior download step.
    #[test]
    fn sam2_bundled_resolve_returns_bytes() {
        let enc = resolve_part_bytes(ModelId::Sam21HieraSmall, "encoder")
            .expect("encoder bytes must resolve");
        let dec = resolve_part_bytes(ModelId::Sam21HieraSmall, "decoder")
            .expect("decoder bytes must resolve");
        // ONNX files start with the protobuf magic bytes 0x08 0x07 0x12 (or
        // similar varint headers); 4 MiB is a sane lower-bound smoke test —
        // the real encoder is ~150 MiB, the decoder ~20 MiB.
        assert!(enc.len() > 4 * 1024 * 1024,
            "SAM 2 encoder decompressed too small: {} bytes", enc.len());
        assert!(dec.len() > 4 * 1024 * 1024,
            "SAM 2 decoder decompressed too small: {} bytes", dec.len());
    }

    /// Callers at the same time share one unpacked copy; once the last
    /// lets go it is freed, not kept for the process's life.
    #[test]
    fn bundled_bytes_are_shared_while_held_and_then_freed() {
        let a = resolve_part_bytes(ModelId::Sam21HieraSmall, "encoder").unwrap();
        let b = resolve_part_bytes(ModelId::Sam21HieraSmall, "encoder").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "unpacked once for both");
        // Other tests hold SAM's real slot at times; a slot of its own
        // shows the freeing.
        static SLOT: Held = Held::new(std::sync::Weak::new());
        let unpacks = std::sync::atomic::AtomicUsize::new(0);
        let get = || shared_bundled(&SLOT, || { unpacks.fetch_add(1, std::sync::atomic::Ordering::Relaxed); vec![7; 16] });
        let (first, second) = (get(), get());
        assert!(Arc::ptr_eq(&first, &second));
        let held = Arc::downgrade(&first);
        drop((first, second));
        assert!(held.upgrade().is_none(), "freed with its last holder");
        drop(get());
        assert_eq!(unpacks.load(std::sync::atomic::Ordering::Relaxed), 2, "unpacked again once it was freed");
    }

    // ── resolve_part_bytes error-path tests ──────────────────────────────

    #[test]
    fn resolve_part_bytes_non_multipart_returns_not_multipart() {
        // Silueta is Bundled — has no named parts.
        let result = resolve_part_bytes(ModelId::Silueta, "encoder");
        assert_eq!(result, Err(ResolveError::NotMultiPart));
    }

    #[test]
    fn resolve_part_bytes_non_multipart_ondemand_returns_not_multipart() {
        // U2net is single-file OnDemand — not a MultiPart bundle.
        let result = resolve_part_bytes(ModelId::U2net, "encoder");
        assert_eq!(result, Err(ResolveError::NotMultiPart));
    }

    #[test]
    fn resolve_part_bytes_unknown_key_returns_unknown_part() {
        let result = resolve_part_bytes(ModelId::Sam21HieraSmall, "nonexistent");
        assert!(
            matches!(result, Err(ResolveError::UnknownPart(_))),
            "expected UnknownPart, got {result:?}"
        );
    }

}
