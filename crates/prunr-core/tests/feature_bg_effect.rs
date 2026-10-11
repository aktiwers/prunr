//! Every `BgEffect` runs end to end on a real cut-out: the cut-out keeps
//! its transparency, and the effect's picture of the photo fills it when
//! composed behind, as export does.
//!
//! NOT a pixel-bit-exact gate (that's `golden_e2e`). Skip-on-missing-ORT
//! mirrors `golden_e2e`'s pattern.

mod test_common;

use prunr_core::{
    apply_background_image, effect_backdrop, process_image_from_decoded, BgEffect, BgImageFit,
    MaskSettings, ModelKind, OrtEngine,
};
use test_common::{multi_subject_canary, skip_if_no_ort};

#[test]
fn the_cutout_stays_transparent_and_each_effect_fills_it() {
    if skip_if_no_ort("bg_effect") { return; }
    let engine = OrtEngine::new_cpu_only(ModelKind::Silueta, 1)
        .expect("OrtEngine::new_cpu_only(Silueta)");
    let photo = multi_subject_canary();
    let cutout = process_image_from_decoded(&photo, &engine, &MaskSettings::default(), &prunr_core::Progress::none())
        .expect("pipeline should succeed")
        .rgba_image;
    assert_eq!(cutout.dimensions(), (256, 256));
    let (min_a, max_a) = cutout.pixels().fold((255u8, 0u8), |(lo, hi), p| (lo.min(p.0[3]), hi.max(p.0[3])));
    assert!(max_a > 200 && min_a < 50, "subject and transparent background: α {min_a}..{max_a}");

    for effect in BgEffect::ALL.iter().filter(|e| **e != BgEffect::None) {
        let picture = effect_backdrop(&photo.to_rgba8(), *effect).expect("an effect makes a picture");
        let mut composed = cutout.clone();
        apply_background_image(&mut composed, &image::DynamicImage::ImageRgba8(picture), BgImageFit::Cover);
        let min = composed.pixels().map(|p| p.0[3]).min().unwrap_or(0);
        assert_eq!(min, 255, "{effect:?}: the picture fills every transparent pixel");
    }
}
