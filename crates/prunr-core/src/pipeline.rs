use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use image::DynamicImage;
use ort::{inputs, value::Tensor};

use crate::{
    engine::{InferenceEngine, OrtEngine},
    formats::{check_large_image, encode_rgba_png, load_image_from_bytes},
    postprocess::postprocess,
    preprocess::preprocess,
    types::{CoreError, MaskSettings, ProcessResult, ProgressStage},
};

/// Process a single image: remove background and return a transparent PNG.
///
/// # Arguments
/// - `img_bytes`: Raw image bytes (PNG, JPEG, WebP, or BMP)
/// - `engine`: OrtEngine with a loaded session. Create once, reuse across images.
/// - `progress`: Optional progress callback. Called at each pipeline stage.
///   Signature: `|stage: ProgressStage, pct: f32|`
///   Stages (in order): Decode(0.0) → Resize(0.2) → Normalize(0.4) → Infer(0.5)
///   → Postprocess(0.8) → Alpha(0.95)
///
/// # Errors
/// - `CoreError::LargeImage` if the image exceeds 8000px in either dimension
/// - `CoreError::ImageFormat` if the bytes cannot be decoded as a supported format
/// - `CoreError::Inference` if ORT inference fails
pub fn process_image<F>(
    img_bytes: &[u8],
    engine: &OrtEngine,
    progress: Option<F>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<ProcessResult, CoreError>
where
    F: Fn(ProgressStage, f32),
{
    process_image_with_mask(img_bytes, engine, &MaskSettings::default(), progress, cancel)
}

/// Process a single image with custom mask settings.
pub fn process_image_with_mask<F>(
    img_bytes: &[u8],
    engine: &OrtEngine,
    mask: &MaskSettings,
    progress: Option<F>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<ProcessResult, CoreError>
where
    F: Fn(ProgressStage, f32),
{
    let report = |stage: ProgressStage, pct: f32| {
        if let Some(ref cb) = progress {
            cb(stage, pct);
        }
    };

    // Stage 1: Decode
    report(ProgressStage::Decode, 0.0);
    let img = load_image_from_bytes(img_bytes)?;

    // Large image guard — return error before allocating a huge tensor
    if let Some(err) = check_large_image(&img) {
        return Err(err);
    }

    process_image_from_decoded(img, engine, mask, progress, cancel)
}

/// Process a single image without the large-image size guard.
///
/// Use only when the caller has already handled the size check
/// (e.g., `--large-image=process` CLI flag). Skips the 8000px dimension guard
/// and proceeds directly to inference on the original image.
///
/// # Arguments
/// - `img_bytes`: Raw image bytes (PNG, JPEG, WebP, or BMP)
/// - `engine`: OrtEngine with a loaded session. Create once, reuse across images.
/// - `progress`: Optional progress callback. Called at each pipeline stage.
///
/// # Errors
/// - `CoreError::ImageFormat` if the bytes cannot be decoded as a supported format
/// - `CoreError::Inference` if ORT inference fails
pub fn process_image_unchecked<F>(
    img_bytes: &[u8],
    engine: &OrtEngine,
    progress: Option<F>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<ProcessResult, CoreError>
where
    F: Fn(ProgressStage, f32),
{
    let img = load_image_from_bytes(img_bytes)?;
    process_image_from_decoded(img, engine, &MaskSettings::default(), progress, cancel)
}

/// Internal helper: run the full pipeline on an already-decoded image.
///
/// Called by both `process_image` (after the large-image guard) and
/// `process_image_unchecked` (bypassing the guard).
fn process_image_from_decoded<F>(
    img: DynamicImage,
    engine: &OrtEngine,
    mask: &MaskSettings,
    progress: Option<F>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<ProcessResult, CoreError>
where
    F: Fn(ProgressStage, f32),
{
    let is_cancelled = || {
        cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    };

    let report = |stage: ProgressStage, pct: f32| {
        if let Some(ref cb) = progress {
            cb(stage, pct);
        }
    };

    // Stage 2: Resize (happens inside preprocess)
    report(ProgressStage::Resize, 0.2);
    if is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    // Stage 3: Normalize (happens inside preprocess, reported before the call)
    report(ProgressStage::Normalize, 0.4);
    let input_array = preprocess(&img);

    if is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    // Stage 4: Inference
    report(ProgressStage::Infer, 0.5);

    let raw_output = engine.with_session(|session| {
        // Query input name at runtime — do NOT hardcode
        let input_name = session.inputs()[0].name().to_string();

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| CoreError::Inference(format!("Failed to create input tensor: {e}")))?;

        let outputs = session
            .run(inputs![input_name.as_str() => &input_tensor])
            .map_err(|e| CoreError::Inference(format!("ORT inference failed: {e}")))?;

        let raw = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| CoreError::Inference(format!("Failed to extract output tensor: {e}")))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| CoreError::Inference(format!("Output reshape error: {e}")))?
            .to_owned();

        Ok(raw)
    })?;

    if is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    // Stage 5: Postprocess (min-max normalize → grayscale mask → resize to original dims)
    report(ProgressStage::Postprocess, 0.8);
    let rgba_image = postprocess(raw_output.view(), &img, mask);

    // Stage 6: Alpha merge + PNG encode
    report(ProgressStage::Alpha, 0.95);
    let rgba_bytes = encode_rgba_png(&rgba_image)?;

    Ok(ProcessResult {
        rgba_bytes,
        rgba_image,
        active_provider: engine.active_provider().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};
    use std::io::Cursor;

    /// Encode a DynamicImage to PNG bytes for use as test input
    fn encode_as_png(img: DynamicImage) -> Vec<u8> {
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn make_png(width: u32, height: u32) -> Vec<u8> {
        encode_as_png(DynamicImage::ImageRgb8(RgbImage::from_pixel(
            width,
            height,
            Rgb([100, 150, 200]),
        )))
    }

    #[test]
    fn test_process_image_large_returns_err() {
        // 9000x100 exceeds LARGE_IMAGE_LIMIT — check_large_image should catch it
        use crate::formats::check_large_image;
        let big = DynamicImage::ImageRgb8(RgbImage::new(9000, 100));
        let err = check_large_image(&big);
        assert!(err.is_some(), "Expected LargeImage error for 9000px wide image");
        match err.unwrap() {
            CoreError::LargeImage { width, .. } => assert_eq!(width, 9000),
            other => panic!("Expected LargeImage, got {:?}", other),
        }
    }

    #[test]
    fn test_process_image_bad_bytes() {
        // load_image_from_bytes should return ImageFormat error for garbage bytes
        use crate::formats::load_image_from_bytes;
        let result = load_image_from_bytes(b"not_an_image");
        assert!(matches!(result, Err(CoreError::ImageFormat(_))));
    }

    // Full integration tests require dev-models feature and downloaded model files.
    #[cfg(feature = "dev-models")]
    #[test]
    fn test_process_image_produces_rgba_bytes() {
        use crate::{engine::OrtEngine, types::ModelKind};

        let engine = OrtEngine::new(ModelKind::Silueta, 1)
            .expect("Need downloaded models — run `cargo xtask fetch-models`");

        let png_bytes = make_png(64, 64);
        let result = process_image(&png_bytes, &engine, None::<fn(ProgressStage, f32)>, None);
        assert!(result.is_ok(), "process_image failed: {:?}", result.err());
        let pr = result.unwrap();
        assert!(!pr.rgba_bytes.is_empty(), "rgba_bytes must not be empty");
        // PNG magic bytes
        assert_eq!(&pr.rgba_bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[cfg(feature = "dev-models")]
    #[test]
    fn test_process_image_progress_stages_called() {
        use crate::{engine::OrtEngine, types::ModelKind};
        use std::sync::{Arc, Mutex};

        let engine = OrtEngine::new(ModelKind::Silueta, 1)
            .expect("Need downloaded models");

        let stages: Arc<Mutex<Vec<ProgressStage>>> = Arc::new(Mutex::new(Vec::new()));
        let stages_clone = stages.clone();

        let png_bytes = make_png(64, 64);
        let _ = process_image(&png_bytes, &engine, Some(move |stage, _pct| {
            stages_clone.lock().unwrap().push(stage);
        }), None);

        let recorded = stages.lock().unwrap();
        assert!(
            recorded.len() >= 5,
            "Expected at least 5 progress callbacks, got {}",
            recorded.len()
        );
    }
}
