//! Model URL + SHA256 manifest. Lives in its own file so the release
//! workflow can hash *just this file* for the model cache key —
//! formerly the key included all of `xtask/src/main.rs`, so any edit
//! to argument parsing or the install-runtime command invalidated the
//! ~700 MB models cache and forced a full re-fetch on every CI run.
//!
//! Bumping a SHA256 / URL here busts the cache; touching `main.rs`
//! does not.

pub(crate) struct ModelSpec {
    /// Registry id; OnDemand mirroring uses the filename from
    /// `prunr_models::descriptor(id)` so xtask and registry can't drift.
    pub id: prunr_models::ModelId,
    /// Dev-mode unversioned filename in `models/`.
    pub name: &'static str,
    pub url: &'static str,
    /// Empty string = bootstrap mode (skip verification, print hash).
    pub sha256: &'static str,
}

// After first run, replace empty strings with the printed SHA256 values.
pub(crate) const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: prunr_models::ModelId::Silueta,
        name: "silueta.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        sha256: "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
    },
    ModelSpec {
        id: prunr_models::ModelId::U2net,
        name: "u2net.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        sha256: "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    },
    ModelSpec {
        id: prunr_models::ModelId::BiRefNetLite,
        name: "birefnet_lite.onnx",
        url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
        sha256: "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
    },
    // DexiNed is exported from PyTorch weights via scripts/export_dexined.py
    // and hosted on prunr's own releases (separate tag from app versions).
    ModelSpec {
        id: prunr_models::ModelId::DexiNed,
        name: "dexined.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/dexined.onnx",
        sha256: "cba9193b1e3fbcb5bd196001a9aae13bafaa309442f6cb074330c426cc61ec5a",
    },
    // LaMa for the Eraser tool. OnDemand: distributed via Model Store at
    // runtime. Dev-mode uses `models/lama_fp32.onnx`; this xtask also
    // mirrors it to the user data dir so the dev workflow exercises the
    // same code path as production.
    ModelSpec {
        id: prunr_models::ModelId::LaMaFp32,
        name: "lama_fp32.onnx",
        url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
    },
    // MI-GAN + Big-LaMa: xtask fetches the already-published artefacts
    // from prunr's own GitHub release rather than rebuilding from source
    // each time — the export scripts (`scripts/export_migan.py` /
    // `scripts/export_big_lama.py`) need a Python env that the Rust dev
    // workflow doesn't otherwise require. To regenerate from source,
    // run the export script directly and re-upload via gh release.
    ModelSpec {
        id: prunr_models::ModelId::Migan,
        name: "migan.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/migan-1.0.0.onnx",
        sha256: "17531b1604e56ff3179a22824c19debf12741dadc551b4500b035bcb216b58ba",
    },
    ModelSpec {
        id: prunr_models::ModelId::BigLaMa,
        name: "big_lama.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/big_lama-1.0.1.onnx",
        sha256: "c9660fc4aea2e62ffbaf3932024c1f4ce29f360e464fc71e1b708f1ad3bfea2e",
    },
    // RRDB upscale models. xtask fetches only the fp32 main file; the
    // registry's `OnDemand.fp16` companion is downloaded at runtime by
    // DownloadManager, not at build/dev-fetch time. Filenames use the
    // capitalized human-facing form from the GitHub release artefacts
    // (matching `load_variant`'s per-model stems).
    ModelSpec {
        id: prunr_models::ModelId::RealEsrganX4Plus,
        name: "RealESRGAN_x4plus.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x4plus.onnx",
        sha256: "fb070c21d1e90102859d52328586a8738ebd4b2b076fe611ef42ba4f58076431",
    },
    ModelSpec {
        id: prunr_models::ModelId::RealEsrganX2Plus,
        name: "RealESRGAN_x2plus.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/RealESRGAN_x2plus.onnx",
        sha256: "7e0860bb32d903520a244c327b6e3d5e08d680b95c29b3fa8a02cb6ecd230c60",
    },
    ModelSpec {
        id: prunr_models::ModelId::FourXNmkdSiaxCx,
        name: "4x-NMKD-Siax-CX.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Siax-CX.onnx",
        sha256: "d7db322397ae1041076a3f9fe9736a7bf9d3364e435eb5b0ece912e050a29f87",
    },
    ModelSpec {
        id: prunr_models::ModelId::FourXNmkdSuperscale,
        name: "4x-NMKD-Superscale.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4x-NMKD-Superscale.onnx",
        sha256: "6b790a203b341d2db71735040f4f1b26c2bdc5440fdc850e217c2234cd996618",
    },
    // Phhofm's official Nomos8k release IS fp16 (single file, no fp32
    // sibling) — `UpscaleModelKnobs::is_fp16: true` drives runtime
    // dispatch. Filename is `4xNomos8kSCHAT-L.onnx` with no `_fp16`
    // suffix because the upstream artefact doesn't carry one.
    ModelSpec {
        id: prunr_models::ModelId::Nomos8kSchatL,
        name: "4xNomos8kSCHAT-L.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4xNomos8kSCHAT-L.onnx",
        sha256: "919dff28836ff10fef2d5462e5b82c951211abf24f05196b0a6e2c24f20ed1de",
    },
    // SAM 2 Hiera Small — bundled (MultiPartBundled) so each file ships
    // as a zstd-compressed `include_bytes!` blob in the release binary.
    // Two rows share the same ModelId because the model is a single
    // logical entry with two parts; `ondemand_target` returns None for
    // MultiPartBundled, so xtask fetches + compresses but does not mirror
    // to the user data dir. URLs are HuggingFace upstream (vietanhdev's
    // ONNX exports of Meta's SAM 2 Hiera Small).
    ModelSpec {
        id: prunr_models::ModelId::Sam2HieraSmall,
        name: "sam2_hiera_small.encoder.onnx",
        url: "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_small.encoder.onnx",
        sha256: "f6a7c74dee5b2e71cce3f0475b778f0f28fa3e6c3646c79027302123d2197f40",
    },
    ModelSpec {
        id: prunr_models::ModelId::Sam2HieraSmall,
        name: "sam2_hiera_small.decoder.onnx",
        url: "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main/sam2_hiera_small.decoder.onnx",
        sha256: "e07f799d2afe8640ef21f47096ad154d9289bb53041191499ebbea8933ef047b",
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use prunr_models::ModelSource;

    /// xtask `MODELS` and `prunr_models::REGISTRY` both carry SHA256s
    /// for OnDemand artefacts — divergent values silently produce
    /// dev-mode files that the in-app DownloadManager rejects (or vice
    /// versa). The xtask download uses its own `sha256` field; the
    /// in-app `OnDemand` path verifies against the registry. Pin the
    /// two together so a one-sided edit fails the build.
    #[test]
    fn xtask_models_match_registry_ondemand_sha256() {
        for spec in MODELS {
            let Some(desc) = prunr_models::descriptor(spec.id) else { continue };
            let ModelSource::OnDemand { sha256: registry_sha, filename, .. } = desc.source
            else {
                continue;
            };
            assert_eq!(
                registry_sha, spec.sha256,
                "SHA256 drift for {:?} ({}): xtask has {:?}, registry has {:?}. \
                 The same artefact must hash the same in both places.",
                spec.id, filename, spec.sha256, registry_sha,
            );
        }
    }

    /// Every single-file OnDemand registry entry must have a matching
    /// xtask `MODELS` row, otherwise `cargo xtask fetch-models` skips
    /// it silently and the dev workflow can't exercise that model.
    /// `MultiPartOnDemand` bundles (SD15 etc.) are out of scope —
    /// xtask doesn't mirror multi-part bundles by design.
    #[test]
    fn every_ondemand_registry_entry_has_xtask_row() {
        let xtask_ids: std::collections::HashSet<prunr_models::ModelId> =
            MODELS.iter().map(|s| s.id).collect();
        for id in prunr_models::ModelId::ALL {
            let Some(desc) = prunr_models::descriptor(*id) else { continue };
            if matches!(desc.source, ModelSource::OnDemand { .. }) && !xtask_ids.contains(id) {
                panic!(
                    "registry has OnDemand entry for {id:?} but xtask MODELS has no matching row \
                     — `cargo xtask fetch-models` would silently skip it. Add the row in \
                     xtask/src/models.rs or document why this id is dev-mode-only.",
                );
            }
        }
    }
}
