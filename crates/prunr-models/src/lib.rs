// Production: models are embedded as pre-compressed zstd blobs and decompressed at runtime.
// Development: --features dev-models loads from the filesystem so model changes
// do not trigger recompilation of the prunr-models crate.
//
// IMPORTANT: This crate has NO dependencies on other workspace crates.
// The dependency arrow is prunr-app -> prunr-core -> prunr-models (never in reverse).
//
// Models are declared in `REGISTRY`; bytes are resolved via `resolve_bytes`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ── Registry types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    Silueta,
    U2net,
    BiRefNetLite,
    DexiNed,
    LaMaFp32,
    BigLaMa,
    Migan,
    /// Stable Diffusion 1.5 Inpainting (FP16). Multi-part bundle:
    /// UNet + VAE encode + VAE decode + text encoder. ~2 GB total.
    /// GPU-required (`GpuRequirement::Required`).
    SdV15InpaintFp16,
    /// LCM-distilled SD 1.5 Inpainting (FP16). Same UNet architecture as
    /// `SdV15InpaintFp16` but trained to converge in ~4 steps instead of
    /// 20 — ~5× faster on CPU/iGPU at slight quality cost. Selected
    /// automatically when `Settings.sd_fast_mode` resolves true.
    /// Bundle is the output of `scripts/export_lcm_inpaint.py`.
    SdV15LcmInpaintFp16,
    /// TAESD (Tiny AutoEncoder for SD) FP16. Two-part bundle: encoder
    /// + decoder, ~5 MB each. Drop-in replacement for SD 1.5's standard
    /// VAE — ~3× faster decode at slight quality cost. Used as the VAE
    /// backend when fast mode is on AND the bundle is installed.
    /// Bundle is the output of `scripts/export_taesd.py`.
    TaesdFp16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Segmentation,
    EdgeDetection,
    Inpaint,
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
    pub license: &'static str,
    pub license_url: &'static str,
    pub source_url: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelSource {
    Bundled,
    OnDemand {
        filename: &'static str,
        url: &'static str,
        sha256: &'static str,
        size_mb: u32,
        license: LicenseInfo,
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
}

impl ModelSource {
    /// Short label for diagnostics + dropdown badges. Adding a variant
    /// fails this match loudly — there's no `_ =>` fall-through.
    pub fn kind_label(&self) -> &'static str {
        match self {
            ModelSource::Bundled => "Bundled",
            ModelSource::OnDemand { .. } => "OnDemand",
            ModelSource::MultiPartOnDemand { .. } => "MultiPart",
        }
    }
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

#[derive(Debug, Clone, Copy)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub display_name: &'static str,
    pub description: &'static str,
    pub category: ModelCategory,
    pub source: ModelSource,
    pub version: &'static str,
    pub gpu: GpuRequirement,
    /// EPs known to fail this specific model export. Checked before
    /// the dynamic compat cache so first-time users skip the failed-
    /// load tax. Verified causes; speculative listings are cargo-cult.
    /// Empty `&[]` means "no static incompatibilities — try every
    /// available EP." See `is_ep_compatible` for the helper.
    ///
    /// Stays `&str`-typed even after `prunr-core::EpKind` exists: this
    /// crate is a leaf with no workspace deps, and pulling `prunr-core`
    /// in just to spell the EP names would be a layering inversion.
    /// Callers convert via `EpKind::Display` at the boundary.
    pub incompatible_eps: &'static [&'static str],
}

impl ModelDescriptor {
    /// Download size in MB for OnDemand entries, `None` for Bundled.
    /// Multi-part bundles return the sum of all parts.
    pub fn size_mb(&self) -> Option<u32> {
        match self.source {
            ModelSource::OnDemand { size_mb, .. } => Some(size_mb),
            ModelSource::MultiPartOnDemand { parts, .. } => {
                let total: u64 = parts.iter().map(|p| p.size_bytes).sum();
                Some((total / (1024 * 1024)) as u32)
            }
            ModelSource::Bundled => None,
        }
    }

    /// True when this bundle requires explicit license-acceptance click
    /// before the download starts (CreativeML Open RAIL, NVIDIA SCL, …).
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
        description: "Fast, clean subjects. The default.",
        category: ModelCategory::Segmentation,
        source: ModelSource::Bundled,
        version: "1.0.0",
        gpu: GpuRequirement::None,
        // OpenVINO 2025.4.1 rejects Silueta's ONNX with
        // "graph is not acyclic" (graph-cycle validation is stricter
        // than ORT's CPU EP). FP16 variant additionally fails on a
        // quantized_cast type mismatch. Both verified 2026-04-26.
        incompatible_eps: &["OpenVINO"],
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
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://github.com/xuebinqin/U-2-Net",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    ModelDescriptor {
        id: ModelId::BiRefNetLite,
        display_name: "BiRefNet-lite",
        description: "Best detail on hair and leaves at 1024×1024.",
        category: ModelCategory::Segmentation,
        source: ModelSource::Bundled,
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    ModelDescriptor {
        id: ModelId::DexiNed,
        display_name: "DexiNed",
        description: "Edge / line extraction. Used by line-mode toolbar.",
        category: ModelCategory::EdgeDetection,
        source: ModelSource::Bundled,
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    ModelDescriptor {
        id: ModelId::LaMaFp32,
        display_name: "Eraser (LaMa)",
        description: "Object removal via inpainting. Smooth fills on simple backgrounds.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "lama_fp32-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/lama_fp32-1.0.0.onnx",
            sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
            size_mb: 199,
            license: LicenseInfo {
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://huggingface.co/Carve/LaMa-ONNX",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    ModelDescriptor {
        id: ModelId::BigLaMa,
        display_name: "Eraser (Big-LaMa)",
        description: "Same architecture as LaMa, trained on more data (Places2). Sharper fills on detailed regions.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "big_lama-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/big_lama-1.0.0.onnx",
            sha256: "523e84eb2ec2df933714cbab6983627a9909f9f23cd848fbbe977356c54bdaa0",
            size_mb: 199,
            license: LicenseInfo {
                license: "Apache-2.0",
                license_url: "https://www.apache.org/licenses/LICENSE-2.0",
                source_url: "https://huggingface.co/smartywu/big-lama",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    ModelDescriptor {
        id: ModelId::Migan,
        display_name: "Eraser (MI-GAN)",
        description: "Lightweight GAN inpainter (~25 MB). Sharper than LaMa on detailed regions; less smooth on flat backgrounds.",
        category: ModelCategory::Inpaint,
        source: ModelSource::OnDemand {
            filename: "migan-1.0.0.onnx",
            url: "https://github.com/aktiwers/prunr/releases/download/models-v1/migan-1.0.0.onnx",
            sha256: "17531b1604e56ff3179a22824c19debf12741dadc551b4500b035bcb216b58ba",
            size_mb: 26,
            license: LicenseInfo {
                license: "MIT",
                license_url: "https://opensource.org/license/mit",
                source_url: "https://github.com/Picsart-AI-Research/MI-GAN",
            },
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    // SD 1.5 Inpainting FP16: GPU-required CreativeML-licensed bundle.
    ModelDescriptor {
        id: ModelId::SdV15InpaintFp16,
        display_name: "Eraser (Stable Diffusion 1.5)",
        description: "Generative inpainting via Stable Diffusion. Phone-app-class quality, GPU strongly preferred. ~2 GB.",
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
                license: "CreativeML Open RAIL-M",
                license_url: "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
                source_url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16",
            },
            license_acceptance_required: true,
        },
        version: "1.0.0",
        gpu: GpuRequirement::Required,
        incompatible_eps: &[],
    },
    // LCM-distilled SD 1.5 Inpaint FP16. ~2 GB total. Same UNet
    // architecture as SdV15InpaintFp16 but trained to converge in ~4
    // steps instead of 20 — ~5× faster on CPU/iGPU at slight quality
    // cost. Selected automatically when Settings.sd_fast_mode resolves
    // true. Released at https://github.com/aktiwers/prunr/releases/tag/lcm-inpaint-v1.0.0.
    ModelDescriptor {
        id: ModelId::SdV15LcmInpaintFp16,
        display_name: "Eraser (SD 1.5 LCM, fast)",
        description: "Latent Consistency Model variant of SD 1.5 inpaint. ~5\u{00d7} faster on CPU/iGPU; lower fidelity. Auto-selected when Fast SD inpaint is on.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "sd15-lcm-inpaint-fp16-1.0.0",
            parts: &[
                ModelPart {
                    key: "unet",
                    filename: "unet.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.0/unet.onnx",
                    sha256: "bd62a44265e8610921d98fa851fb9507cc5ec16eebf359b8e10b0a043f17d4d7",
                    size_bytes: 1723163151,
                },
                ModelPart {
                    key: "vae_encoder",
                    filename: "vae_encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.0/vae_encoder.onnx",
                    sha256: "9a5f1ade2a09a69496e1c48532fc54f11c8d118686115226c0de2698523b7826",
                    size_bytes: 68826952,
                },
                ModelPart {
                    key: "vae_decoder",
                    filename: "vae_decoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.0/vae_decoder.onnx",
                    sha256: "fbe8a7ab071fe0d63484ac9764423505585c4db650cbe7734fb456cf415f7896",
                    size_bytes: 99634701,
                },
                ModelPart {
                    key: "text_encoder",
                    filename: "text_encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/lcm-inpaint-v1.0.0/text_encoder.onnx",
                    sha256: "76c497febe21922a096f368558a18b3549f1fe1b08f7a49267a14d0184c8155f",
                    size_bytes: 246346236,
                },
            ],
            license: LicenseInfo {
                license: "CreativeML Open RAIL-M",
                license_url: "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
                source_url: "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
            },
            license_acceptance_required: true,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
    // TAESD FP16: Tiny distilled VAE for SD 1.5. ~2.4 MB encoder + ~2.5
    // MB decoder. Released at https://github.com/aktiwers/prunr/releases/tag/taesd-v1.0.0.
    // Used as the VAE backend when fast mode + LCM are both active.
    ModelDescriptor {
        id: ModelId::TaesdFp16,
        display_name: "TAESD VAE (fast SD)",
        description: "Tiny distilled VAE for SD 1.5 fast mode. ~3\u{00d7} faster decode at slight quality cost.",
        category: ModelCategory::Inpaint,
        source: ModelSource::MultiPartOnDemand {
            subdir: "taesd-fp16-1.0.0",
            parts: &[
                ModelPart {
                    key: "encoder",
                    filename: "encoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/taesd-v1.0.0/encoder.onnx",
                    sha256: "e60041ed5718ff2dd5d3bff24be82e07c9e26b10850b5837468f7bd352625f98",
                    size_bytes: 2463603,
                },
                ModelPart {
                    key: "decoder",
                    filename: "decoder.onnx",
                    url: "https://github.com/aktiwers/prunr/releases/download/taesd-v1.0.0/decoder.onnx",
                    sha256: "4219d97c8bac63a18467ea47768545edcb15f554bfecadc1bed1fb70b8882304",
                    size_bytes: 2632309,
                },
            ],
            license: LicenseInfo {
                license: "MIT",
                license_url: "https://github.com/madebyollin/taesd/blob/main/LICENSE",
                source_url: "https://github.com/madebyollin/taesd",
            },
            license_acceptance_required: false,
        },
        version: "1.0.0",
        gpu: GpuRequirement::None,
        incompatible_eps: &[],
    },
];

pub fn descriptor(id: ModelId) -> Option<&'static ModelDescriptor> {
    REGISTRY.iter().find(|d| d.id == id)
}

/// True when the static catalog says this (model, EP) pair should NOT
/// be attempted. Cheap — pointer-walk a small `&'static [&str]`.
pub fn is_ep_compatible(model: ModelId, ep: &str) -> bool {
    descriptor(model)
        .map(|d| !d.incompatible_eps.iter().any(|s| s.eq_ignore_ascii_case(ep)))
        .unwrap_or(true)
}

pub fn descriptors_for(category: ModelCategory) -> impl Iterator<Item = &'static ModelDescriptor> {
    REGISTRY.iter().filter(move |d| d.category == category)
}

/// Resolve bytes for a model. Returns `None` for `OnDemand` entries that
/// haven't been downloaded yet. Bundled entries return `Cow::Borrowed`
/// (zero-copy from the embedded zstd cache); OnDemand entries are read
/// once from disk, then cached for the process lifetime so repeated
/// `OrtEngine::new` calls — e.g. user switching between U2Net and a
/// bundled model in a single session — don't re-read 168 MB each time.
pub fn resolve_bytes(id: ModelId) -> Option<Cow<'static, [u8]>> {
    let desc = descriptor(id)?;
    match desc.source {
        ModelSource::Bundled => Some(Cow::Borrowed(bundled_bytes(id))),
        ModelSource::OnDemand { filename, .. } => on_demand_bytes(id, filename).map(Cow::Borrowed),
        // Multi-part bundles aren't a single byte-blob — callers go
        // through `multi_part_paths(id)` and load each part separately.
        ModelSource::MultiPartOnDemand { .. } => None,
    }
}

/// Process-lifetime cache for OnDemand model bytes. Entries are
/// `Box::leak`'d into `&'static [u8]` so repeated lookups can return
/// borrowed slices without re-allocating or holding a lock. Memory
/// cost is bounded by REGISTRY size and matches what the user would
/// pay if the model were bundled.
fn on_demand_cache() -> &'static Mutex<HashMap<ModelId, &'static [u8]>> {
    static CACHE: OnceLock<Mutex<HashMap<ModelId, &'static [u8]>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn on_demand_bytes(id: ModelId, filename: &str) -> Option<&'static [u8]> {
    {
        let cache = on_demand_cache().lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(slice) = cache.get(&id) {
            return Some(slice);
        }
    }
    let bytes = read_on_demand_from_any_path(filename)?;
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let mut cache = on_demand_cache().lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A concurrent caller may have inserted between our check and
    // load — keep the first-inserted slice and drop ours. Cheap;
    // beats holding the lock during the 200 MB read.
    Some(*cache.entry(id).or_insert(leaked))
}

fn read_on_demand_from_any_path(filename: &str) -> Option<Vec<u8>> {
    let dir = on_demand_dir()?;
    read_on_demand(&dir, filename)
}

/// Drop the cached `&'static [u8]` slot for `id`. Call this after the
/// file is removed from disk so a subsequent re-download repopulates
/// from the new bytes instead of returning stale cached content. The
/// previously-leaked slice is intentionally NOT freed — anyone who
/// already received it (e.g. a still-loaded session) keeps using it
/// safely; the OS reclaims on process exit.
pub fn evict_on_demand_cache(id: ModelId) {
    let mut cache = on_demand_cache().lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.remove(&id);
}

/// User-facing error string when an OnDemand model isn't yet installed.
/// Single source of truth for the "go download it" guidance — referenced
/// by both `engine.rs` and `inpaint.rs` so the wording stays consistent.
pub fn not_installed_error(id: ModelId) -> String {
    let name = descriptor(id).map_or("Model", |d| d.display_name);
    format!("{name} is not installed. Open the Model Store from the model dropdown to download it.")
}

pub fn is_available(id: ModelId) -> bool {
    match descriptor(id).map(|d| d.source) {
        Some(ModelSource::Bundled) => true,
        Some(ModelSource::OnDemand { filename, .. }) => {
            on_demand_dir().is_some_and(|d| d.join(filename).is_file())
        }
        Some(ModelSource::MultiPartOnDemand { .. }) => multi_part_paths(id).is_some(),
        None => false,
    }
}

/// Resolve absolute paths to every part of a multi-part bundle. Returns
/// `None` for non-MultiPart entries or when the bundle isn't installed.
/// Inference modules (e.g. `prunr-core::inpaint::sd`) iterate the
/// returned slice to load each part by its `key`.
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

/// Per-app data root. Linux: `$XDG_DATA_HOME/prunr/`, macOS:
/// `~/Library/Application Support/prunr/`, Windows: `%APPDATA%\prunr\`.
/// Returns `None` when the platform's data dir can't be resolved (rare —
/// typically only on stripped-down test envs). Does NOT create the dir.
pub fn data_dir() -> Option<std::path::PathBuf> {
    dirs::data_dir().map(|d| d.join("prunr"))
}

/// Where on-demand models are stored. Subdir of `data_dir()`.
pub fn on_demand_dir() -> Option<std::path::PathBuf> {
    data_dir().map(|d| d.join("models"))
}

/// Pure helper: read `dir/filename` as bytes, or None if missing/unreadable.
/// SHA verification happens at download time, not load time — re-hashing
/// 200+ MB on every app launch would be wasteful.
fn read_on_demand(dir: &std::path::Path, filename: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(filename)).ok()
}

/// OnDemand ids are unreachable here — `resolve_bytes` routes them
/// through the disk read path. The panic guards against a REGISTRY
/// row that says `Bundled` but lists no match arm below.
fn bundled_bytes(id: ModelId) -> &'static [u8] {
    match id {
        ModelId::Silueta => silueta_bytes(),
        ModelId::BiRefNetLite => birefnet_lite_bytes(),
        ModelId::DexiNed => dexined_bytes(),
        ModelId::U2net
        | ModelId::LaMaFp32
        | ModelId::BigLaMa
        | ModelId::Migan
        | ModelId::SdV15InpaintFp16
        | ModelId::SdV15LcmInpaintFp16
        | ModelId::TaesdFp16 => {
            // OnDemand / MultiPart in REGISTRY — resolve_bytes routes
            // via disk (or via `multi_part_paths`), not here.
            panic!("bundled_bytes called for non-Bundled model {id:?} — REGISTRY/source mismatch");
        }
    }
}

// ── Bundled byte loaders ────────────────────────────────────────────────────

#[cfg(not(feature = "dev-models"))]
static SILUETA_ZST: &[u8] = include_bytes!("../../../models/silueta.onnx.zst");
#[cfg(not(feature = "dev-models"))]
static BIREFNET_LITE_ZST: &[u8] = include_bytes!("../../../models/birefnet_lite.onnx.zst");
#[cfg(not(feature = "dev-models"))]
static DEXINED_ZST: &[u8] = include_bytes!("../../../models/dexined.onnx.zst");

#[cfg(not(feature = "dev-models"))]
static SILUETA_CACHE: OnceLock<Vec<u8>> = OnceLock::new();
#[cfg(not(feature = "dev-models"))]
static BIREFNET_LITE_CACHE: OnceLock<Vec<u8>> = OnceLock::new();
#[cfg(not(feature = "dev-models"))]
static DEXINED_CACHE: OnceLock<Vec<u8>> = OnceLock::new();

#[cfg(not(feature = "dev-models"))]
pub fn silueta_bytes() -> &'static [u8] {
    SILUETA_CACHE.get_or_init(|| {
        zstd::bulk::decompress(SILUETA_ZST, 50 * 1024 * 1024)
            .expect("failed to decompress embedded silueta model")
    })
}

#[cfg(not(feature = "dev-models"))]
pub fn birefnet_lite_bytes() -> &'static [u8] {
    BIREFNET_LITE_CACHE.get_or_init(|| {
        zstd::bulk::decompress(BIREFNET_LITE_ZST, 250 * 1024 * 1024)
            .expect("failed to decompress embedded birefnet-lite model")
    })
}

#[cfg(not(feature = "dev-models"))]
pub fn dexined_bytes() -> &'static [u8] {
    DEXINED_CACHE.get_or_init(|| {
        zstd::bulk::decompress(DEXINED_ZST, 150 * 1024 * 1024)
            .expect("failed to decompress embedded dexined model")
    })
}

#[cfg(feature = "dev-models")]
static DEV_SILUETA: OnceLock<Vec<u8>> = OnceLock::new();
#[cfg(feature = "dev-models")]
static DEV_BIREFNET: OnceLock<Vec<u8>> = OnceLock::new();
#[cfg(feature = "dev-models")]
static DEV_DEXINED: OnceLock<Vec<u8>> = OnceLock::new();

#[cfg(feature = "dev-models")]
fn dev_model_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name)
}

#[cfg(feature = "dev-models")]
pub fn silueta_bytes() -> &'static [u8] {
    DEV_SILUETA.get_or_init(|| std::fs::read(dev_model_path("silueta.onnx"))
        .expect("models/silueta.onnx not found — run `cargo xtask fetch-models`"))
}

#[cfg(feature = "dev-models")]
pub fn birefnet_lite_bytes() -> &'static [u8] {
    DEV_BIREFNET.get_or_init(|| std::fs::read(dev_model_path("birefnet_lite.onnx"))
        .expect("models/birefnet_lite.onnx not found — run `cargo xtask fetch-models`"))
}

#[cfg(feature = "dev-models")]
pub fn dexined_bytes() -> &'static [u8] {
    DEV_DEXINED.get_or_init(|| std::fs::read(dev_model_path("dexined.onnx"))
        .expect("models/dexined.onnx not found — run `cargo xtask fetch-models`"))
}

// ── Optimized model variants (FP16 for GPU, INT8 for CPU) ───────────────────
// Generated by: python scripts/convert_models.py
// Returns None when the variant hasn't been generated yet, or when the model
// has no optimized variant (DexiNed, LaMa). Caller falls back to FP32.

pub fn model_fp16_bytes(id: ModelId) -> Option<Vec<u8>> {
    load_variant(id, "fp16")
}

pub fn model_int8_bytes(id: ModelId) -> Option<Vec<u8>> {
    load_variant(id, "int8")
}

fn load_variant(id: ModelId, suffix: &str) -> Option<Vec<u8>> {
    // Only segmentation models ship optimized variants today.
    let name = match id {
        ModelId::Silueta => "silueta",
        ModelId::U2net => "u2net",
        ModelId::BiRefNetLite => "birefnet_lite",
        ModelId::DexiNed
        | ModelId::LaMaFp32
        | ModelId::BigLaMa
        | ModelId::Migan
        | ModelId::SdV15InpaintFp16
        | ModelId::SdV15LcmInpaintFp16
        | ModelId::TaesdFp16 => return None,
    };
    let filename = format!("{name}_{suffix}.onnx");

    #[cfg(debug_assertions)]
    {
        let dev_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&dev_path) {
            return Some(bytes);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Ok(bytes) = std::fs::read(dir.join("models").join(&filename)) {
                return Some(bytes);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn model_api_compiles() {
        let _: fn() -> &'static [u8] = silueta_bytes;
        let _: fn() -> &'static [u8] = birefnet_lite_bytes;
        let _: fn() -> &'static [u8] = dexined_bytes;
    }

    #[test]
    fn every_model_id_has_a_registry_entry() {
        // Add a ModelId variant ⇒ add a REGISTRY row. Guards against a
        // half-finished addition that compiles but has no metadata.
        for id in [
            ModelId::Silueta,
            ModelId::U2net,
            ModelId::BiRefNetLite,
            ModelId::DexiNed,
            ModelId::LaMaFp32,
            ModelId::BigLaMa,
            ModelId::Migan,
            ModelId::SdV15InpaintFp16,
        ] {
            assert!(descriptor(id).is_some(), "ModelId::{id:?} missing from REGISTRY");
        }
    }

    #[test]
    fn bundled_models_must_not_require_gpu() {
        // `card_action` short-circuits on Bundled before checking GPU
        // requirement. Pin "Bundled implies !Required" so a future
        // Bundled entry with `Required` doesn't silently render as
        // Built-in while ignoring its own gate.
        for desc in REGISTRY {
            if matches!(desc.source, ModelSource::Bundled) {
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
    fn descriptors_for_segmentation_returns_seg_models_only() {
        let ids: Vec<ModelId> = descriptors_for(ModelCategory::Segmentation)
            .map(|d| d.id)
            .collect();
        assert!(ids.contains(&ModelId::Silueta));
        assert!(ids.contains(&ModelId::U2net));
        assert!(ids.contains(&ModelId::BiRefNetLite));
        assert!(!ids.contains(&ModelId::DexiNed));
        assert!(!ids.contains(&ModelId::LaMaFp32));
    }

    #[test]
    fn bundled_descriptors_are_always_available() {
        for desc in REGISTRY {
            if matches!(desc.source, ModelSource::Bundled) {
                assert!(is_available(desc.id), "Bundled {:?} reports unavailable", desc.id);
            }
        }
    }

    #[test]
    fn ondemand_descriptors_have_complete_metadata() {
        for desc in REGISTRY {
            if let ModelSource::OnDemand { filename, url, sha256, size_mb, license } = desc.source {
                assert!(!filename.is_empty(),    "{:?} filename empty",    desc.id);
                assert!(url.starts_with("https://"), "{:?} non-HTTPS url: {url}", desc.id);
                assert_eq!(sha256.len(), 64,     "{:?} sha256 not 64 hex chars: {sha256}", desc.id);
                assert!(size_mb > 0,             "{:?} size_mb=0",         desc.id);
                assert!(!license.license.is_empty(), "{:?} license empty", desc.id);
                assert!(license.license_url.starts_with("https://"), "{:?} bad license_url", desc.id);
                assert!(license.source_url.starts_with("https://"),  "{:?} bad source_url",  desc.id);
            }
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
        for id in [
            ModelId::DexiNed,
            ModelId::LaMaFp32,
            ModelId::BigLaMa,
            ModelId::Migan,
            ModelId::SdV15InpaintFp16,
        ] {
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
    fn read_on_demand_returns_bytes_when_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("blob.onnx"), b"hello-bytes").unwrap();
        assert_eq!(
            read_on_demand(dir.path(), "blob.onnx").as_deref(),
            Some(b"hello-bytes".as_slice()),
        );
    }

    #[test]
    fn read_on_demand_returns_none_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_on_demand(dir.path(), "nope.onnx").is_none());
    }
}
