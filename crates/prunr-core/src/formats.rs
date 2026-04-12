use std::io::Cursor;
use std::path::Path;
use image::{DynamicImage, ImageReader, RgbaImage, imageops::FilterType};
use image::codecs::png::{PngEncoder, CompressionType, FilterType as PngFilter};
use image::ImageEncoder;
use crate::types::{CoreError, LARGE_IMAGE_LIMIT};

/// Load an image from a file path. Format detected by file extension.
/// Supports PNG, JPEG, WebP, BMP (via image crate feature flags in Cargo.toml).
pub fn load_image_from_path(path: &Path) -> Result<DynamicImage, CoreError> {
    image::open(path).map_err(CoreError::from)
}

/// Load an image from raw bytes. Format detected by magic bytes (not extension).
/// Use this for drag-and-drop or embedded data where no path is available.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| CoreError::Io(e))?
        .decode()
        .map_err(CoreError::from)
}

/// Check whether the image exceeds the large image limit (8000px in either dimension).
/// Returns Some(CoreError::LargeImage) if exceeded; None otherwise.
/// The caller decides what to do: prompt user, auto-downscale, or abort.
pub fn check_large_image(img: &DynamicImage) -> Option<CoreError> {
    let (w, h) = (img.width(), img.height());
    if w > LARGE_IMAGE_LIMIT || h > LARGE_IMAGE_LIMIT {
        Some(CoreError::LargeImage { width: w, height: h, limit: LARGE_IMAGE_LIMIT })
    } else {
        None
    }
}

/// Downscale an image so its largest dimension does not exceed max_dim.
/// Uses Lanczos3 resampling. Preserves aspect ratio.
/// Default max_dim: DOWNSCALE_TARGET (4096).
pub fn downscale_image(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let largest = w.max(h);
    if largest <= max_dim {
        return img;
    }
    let scale = max_dim as f32 / largest as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    img.resize_exact(nw, nh, FilterType::Lanczos3)
}

/// Encode an RgbaImage as PNG bytes with fast compression.
pub fn encode_rgba_png(img: &RgbaImage) -> Result<Vec<u8>, CoreError> {
    let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
    let encoder = PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, PngFilter::Sub);
    encoder.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
        .map_err(CoreError::from)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DOWNSCALE_TARGET;
    use image::{DynamicImage, RgbaImage, Rgba};

    /// Minimal 1x1 red PNG as raw bytes (generated once, hardcoded for unit test isolation)
    fn minimal_png_bytes() -> Vec<u8> {
        // Create a 1x1 red RGBA image and encode to PNG in-memory
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        let mut buf = Vec::new();
        img.write_to(
            &mut Cursor::new(&mut buf),
            image::ImageFormat::Png,
        ).unwrap();
        buf
    }

    #[test]
    fn test_load_image_from_bytes_png() {
        let bytes = minimal_png_bytes();
        let result = load_image_from_bytes(&bytes);
        assert!(result.is_ok(), "Failed to load PNG bytes: {:?}", result.err());
        let img = result.unwrap();
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    #[test]
    fn test_check_large_image_over_limit() {
        // Create a thin but wide image > 8000px
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(9000, 100));
        let result = check_large_image(&img);
        assert!(result.is_some(), "Expected LargeImage error for 9000px wide image");
        match result.unwrap() {
            CoreError::LargeImage { width, height: _, limit } => {
                assert_eq!(width, 9000);
                assert_eq!(limit, LARGE_IMAGE_LIMIT);
            }
            other => panic!("Expected LargeImage, got {:?}", other),
        }
    }

    #[test]
    fn test_check_large_image_under_limit() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        assert!(check_large_image(&img).is_none());
    }

    #[test]
    fn test_downscale_image_respects_max_dim() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(8000, 4000));
        let scaled = downscale_image(img, DOWNSCALE_TARGET);
        assert!(
            scaled.width().max(scaled.height()) <= DOWNSCALE_TARGET,
            "Downscaled image {}x{} exceeds max_dim {}",
            scaled.width(), scaled.height(), DOWNSCALE_TARGET
        );
    }

    #[test]
    fn test_downscale_image_no_upscale() {
        // Images already within limit should be returned as-is
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        let scaled = downscale_image(img, DOWNSCALE_TARGET);
        assert_eq!(scaled.width(), 800);
        assert_eq!(scaled.height(), 600);
    }

    #[test]
    fn test_encode_rgba_png() {
        let mut img = RgbaImage::new(10, 10);
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 128, 255, 200]);
        }
        let result = encode_rgba_png(&img);
        assert!(result.is_ok(), "encode_rgba_png failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "Encoded PNG is empty");
        // PNG magic bytes: \x89PNG
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
