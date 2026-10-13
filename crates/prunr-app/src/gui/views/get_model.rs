//! One wording and one route for a model that is not installed, on
//! every surface that offers it: "Get <model> · <size>", the Model
//! Store opened on the model's section, and "Get" on a chip's face.

use std::sync::OnceLock;

use prunr_models::{descriptor, ModelId};

use super::adjustments_toolbar::ModelStoreRequest;
use super::chip;

/// The chip-face word for a feature whose model is missing.
pub(crate) const BADGE: &str = "Get";

/// "Depth Anything V2 Small · 94 MB"; a bundled model has no size.
pub(crate) fn name_and_size(id: ModelId) -> String {
    let Some(d) = descriptor(id) else { return format!("{id:?}") };
    match d.size_mb() {
        Some(mb) => format!("{} \u{00b7} {}", d.display_name, size(mb)),
        None => d.display_name.to_string(),
    }
}

fn size(mb: u32) -> String {
    if mb >= 1000 { format!("{:.1} GB", f64::from(mb) / 1024.0) } else { format!("{mb} MB") }
}

/// "Get Depth Anything V2 Small · 94 MB", built once per model so a
/// strip can show it every frame.
pub(crate) fn label(id: ModelId) -> &'static str {
    &words(id).0
}

/// "Get · 94 MB", for a list row that already names the model.
pub(crate) fn badge_and_size(id: ModelId) -> &'static str {
    &words(id).1
}

fn words(id: ModelId) -> &'static (String, String) {
    static WORDS: OnceLock<Vec<(String, String)>> = OnceLock::new();
    let words = WORDS.get_or_init(|| {
        ModelId::ALL.iter().map(|&id| {
            let size = descriptor(id).and_then(|d| d.size_mb()).map(size);
            let badge = match size {
                Some(size) => format!("{BADGE} \u{00b7} {size}"),
                None => BADGE.to_string(),
            };
            (format!("{BADGE} {}", name_and_size(id)), badge)
        }).collect()
    });
    // every variant is in ALL (a registry test fails the build otherwise)
    &words[ModelId::ALL.iter().position(|&m| m == id).expect("ModelId::ALL is exhaustive")]
}

/// "Depth needs Depth Anything V2 Small · 94 MB."
pub(crate) fn needs(feature: &str, id: ModelId) -> String {
    format!("{feature} needs {}.", name_and_size(id))
}

/// "Faces needs a face model, 295 to 340 MB." for a feature any of
/// `ids` serves.
pub(crate) fn needs_one_of(feature: &str, what: &str, ids: &[ModelId]) -> String {
    let sizes: Vec<u32> = ids.iter().filter_map(|&id| descriptor(id)?.size_mb()).collect();
    match (sizes.iter().min(), sizes.iter().max()) {
        (Some(lo), Some(hi)) if lo != hi => format!("{feature} needs {what}, {} to {}.", size(*lo), size(*hi)),
        (Some(one), _) => format!("{feature} needs {what}, {}.", size(*one)),
        _ => format!("{feature} needs {what}."),
    }
}

/// A button offering the download of `id`; true when clicked.
pub(crate) fn button(ui: &mut egui::Ui, id: ModelId) -> bool {
    chip::button(ui, chip::ButtonKind::Secondary, label(id)).clicked()
}

/// The Model Store opened on the model's section.
pub(crate) fn store_request(id: ModelId) -> ModelStoreRequest {
    ModelStoreRequest { filter: descriptor(id).map(|d| d.category) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every surface says the same thing about a missing model: its
    /// name and download size, "Get" in front of it, and the Store
    /// opens on its section.
    #[test]
    fn one_wording_for_a_missing_model() {
        let depth = ModelId::DepthAnythingV2Small;
        let depth_label = label(depth);
        assert!(depth_label.starts_with("Get Depth Anything V2 Small \u{00b7} ") && depth_label.ends_with(" MB"), "{depth_label}");
        assert_eq!(needs("Depth", depth), format!("Depth needs {}.", name_and_size(depth)));
        assert!(badge_and_size(depth).starts_with("Get \u{00b7} "));
        assert_eq!(label(ModelId::Silueta), "Get Silueta", "a bundled model has no download size");
        assert_eq!(badge_and_size(ModelId::Silueta), "Get");
        assert_eq!(size(2048), "2.0 GB");
        assert_eq!(size(94), "94 MB");
        let faces = needs_one_of("Faces", "a face model", &[ModelId::FaceGfpgan, ModelId::FaceRestoreFormerPp]);
        assert!(faces.starts_with("Faces needs a face model, ") && faces.contains(" to ") && faces.ends_with(" MB."), "{faces}");
        assert_eq!(store_request(depth).filter, Some(prunr_models::ModelCategory::Depth));
    }
}
