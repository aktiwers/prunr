//! Model URL + SHA256 manifest. Its own file because the release
//! workflow hashes just this file for the ~700 MB models cache key:
//! editing `main.rs` must not force a re-fetch.

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
    // TEED (MIT, Xavier Soria Poma) exported from the BIPED checkpoint and
    // hosted on prunr's own releases (separate tag from app versions).
    ModelSpec {
        id: prunr_models::ModelId::Teed,
        name: "teed.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/teed.onnx",
        sha256: "8aab32bf7b9da0a4f03108c7f5dd9b9799e22601f212249659ea3ab1d59a5a1a",
    },
    // YuNet (MIT, OpenCV Zoo), pinned to the zoo commit it was taken from.
    ModelSpec {
        id: prunr_models::ModelId::YuNet,
        name: "face_detection_yunet_2026may.onnx",
        url: "https://github.com/opencv/opencv_zoo/raw/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
        sha256: "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    },
    // OnDemand: also mirrored to the user data dir so dev runs take the
    // production path.
    ModelSpec {
        id: prunr_models::ModelId::LaMaFp32,
        name: "lama_fp32.onnx",
        url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
    },
    // MI-GAN + Big-LaMa: fetched from prunr's release, since their export
    // scripts need a Python env the Rust workflow doesn't otherwise have.
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
    // RRDB upscale models: only the fp32 main file; the fp16 companion
    // downloads at runtime.
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
    ModelSpec {
        id: prunr_models::ModelId::DdcolorTiny,
        name: "ddcolor_paper_tiny.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/ddcolor_paper_tiny.onnx",
        sha256: "5b4396ff19dbc9729a39fbebd6d7857dee6ff58a35d8820467efa168ecf7b17b",
    },
    ModelSpec {
        id: prunr_models::ModelId::DepthAnythingV2Small,
        name: "depth_anything_v2_small.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/depth_anything_v2_small.onnx",
        sha256: "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
    },
    // Phhofm's release is fp16 already, without an `_fp16` suffix.
    ModelSpec {
        id: prunr_models::ModelId::Nomos8kSchatL,
        name: "4xNomos8kSCHAT-L.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/4xNomos8kSCHAT-L.onnx",
        sha256: "919dff28836ff10fef2d5462e5b82c951211abf24f05196b0a6e2c24f20ed1de",
    },
    // SAM 2.1 Hiera Small, bundled: two rows (encoder, decoder) share one
    // ModelId. Exported from Meta's checkpoint with vietanhdev/samexporter
    // (opset 17).
    ModelSpec {
        id: prunr_models::ModelId::Sam21HieraSmall,
        name: "sam2.1_hiera_small.encoder.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/sam2.1_hiera_small.encoder.onnx",
        sha256: "d15d9f152326e95f788a6122343b1f055cbae4b689dba041a90548782dbdd4e2",
    },
    ModelSpec {
        id: prunr_models::ModelId::Sam21HieraSmall,
        name: "sam2.1_hiera_small.decoder.onnx",
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/sam2.1_hiera_small.decoder.onnx",
        sha256: "f059fb650ec6cd7c512f9fa59bcd76469cb24359969ad3898dfd85a81e13d92e",
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

    /// xtask mirrors no downloaded bundle, so a row for one fetches a file
    /// nothing reads: what a model that moved to parts leaves behind.
    #[test]
    fn no_xtask_row_is_for_a_downloaded_bundle() {
        for spec in MODELS {
            assert!(
                !prunr_models::descriptor(spec.id).is_some_and(|d| matches!(d.source, ModelSource::MultiPartOnDemand { .. })),
                "xtask row {} is for {:?}, a downloaded bundle",
                spec.name, spec.id,
            );
        }
    }
}
