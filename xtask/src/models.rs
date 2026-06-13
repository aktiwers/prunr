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
        url:
            "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
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
        url: "https://github.com/aktiwers/prunr/releases/download/models-v1/big_lama-1.0.0.onnx",
        sha256: "523e84eb2ec2df933714cbab6983627a9909f9f23cd848fbbe977356c54bdaa0",
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
            let Some(desc) = prunr_models::descriptor(spec.id) else {
                continue;
            };
            let ModelSource::OnDemand {
                sha256: registry_sha,
                filename,
                ..
            } = desc.source
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
            let Some(desc) = prunr_models::descriptor(*id) else {
                continue;
            };
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
