use std::io::Cursor;
use std::path::Path;
use image::{DynamicImage, GrayImage, ImageReader, RgbaImage};
use image::codecs::png::{PngEncoder, CompressionType, FilterType as PngFilter};
use image::ImageEncoder;
use fast_image_resize::{images::Image, PixelType, Resizer};
use crate::types::{CoreError, LARGE_IMAGE_LIMIT};

/// SIMD-accelerated Lanczos3 resize for single-channel (gray) images.
pub fn resize_gray_lanczos3(src: &GrayImage, dst_width: u32, dst_height: u32) -> GrayImage {
    let src_image = Image::from_vec_u8(
        src.width(), src.height(), src.as_raw().to_vec(), PixelType::U8,
    ).expect("valid gray image buffer");
    let mut dst_image = Image::new(dst_width, dst_height, PixelType::U8);
    Resizer::new().resize(&src_image, &mut dst_image, None).expect("resize failed");
    GrayImage::from_raw(dst_width, dst_height, dst_image.into_vec()).expect("valid dimensions")
}

/// SIMD-accelerated Lanczos3 resize for RGB images.
pub fn resize_rgb_lanczos3(img: &DynamicImage, dst_width: u32, dst_height: u32) -> image::RgbImage {
    let rgb = img.to_rgb8();
    let src = Image::from_vec_u8(rgb.width(), rgb.height(), rgb.into_raw(), PixelType::U8x3)
        .expect("valid RGB buffer");
    let mut dst = Image::new(dst_width, dst_height, PixelType::U8x3);
    Resizer::new().resize(&src, &mut dst, None).expect("resize failed");
    image::RgbImage::from_raw(dst_width, dst_height, dst.into_vec()).expect("valid dimensions")
}

/// Load an image from a file path. Format detected by file extension.
/// Supports PNG, JPEG, WebP, BMP, and SVG (rasterized via resvg).
pub fn load_image_from_path(path: &Path) -> Result<DynamicImage, CoreError> {
    if has_svg_extension(path) {
        let bytes = std::fs::read(path).map_err(CoreError::Io)?;
        return rasterize_svg(&bytes);
    }
    image::open(path).map_err(CoreError::from)
}

/// Load an image from raw bytes. Format detected by content sniff. SVG is
/// detected first (XML prologue / `<svg`); otherwise the `image` crate's
/// magic-byte detection takes over for PNG/JPEG/WebP/BMP.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    if looks_like_svg(bytes) {
        return rasterize_svg(bytes);
    }
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| CoreError::Io(e))?
        .decode()
        .map_err(CoreError::from)
}

fn has_svg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("svg"))
}

/// Sniff the leading bytes for an SVG marker. Allows leading whitespace +
/// optional UTF-8 BOM. Probes only the first 512 bytes. We deliberately
/// match `<?xml` and `<svg` only — `<!--` and `<!DOCTYPE` would false-
/// positive on non-SVG XML or HTML; SVG that starts with a leading
/// comment can still be loaded by extension via `load_image_from_path`.
fn looks_like_svg(bytes: &[u8]) -> bool {
    let probe = &bytes[..bytes.len().min(512)];
    let probe = probe.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(probe);
    let head = std::str::from_utf8(probe).unwrap_or("");
    let trimmed = head.trim_start();
    trimmed.starts_with("<?xml") || trimmed.starts_with("<svg")
}

/// Rasterize an SVG byte buffer to a `DynamicImage::ImageRgba8`. Uses the
/// SVG's intrinsic size, capped at `LARGE_IMAGE_LIMIT` so a pathological
/// `<svg width="100000">` can't blow memory.
fn rasterize_svg(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    use resvg::tiny_skia::{Pixmap, Transform};
    use resvg::usvg::{Options, Tree};

    let opts = Options::default();
    let tree = Tree::from_data(bytes, &opts)
        .map_err(|e| CoreError::ImageFormat(format!("SVG parse error: {e}")))?;
    let size = tree.size();
    let intrinsic_w = size.width().ceil().max(1.0) as u32;
    let intrinsic_h = size.height().ceil().max(1.0) as u32;
    let max_dim = intrinsic_w.max(intrinsic_h);
    let scale = if max_dim > LARGE_IMAGE_LIMIT {
        LARGE_IMAGE_LIMIT as f32 / max_dim as f32
    } else {
        1.0
    };
    let render_w = ((intrinsic_w as f32) * scale).round().max(1.0) as u32;
    let render_h = ((intrinsic_h as f32) * scale).round().max(1.0) as u32;

    let mut pixmap = Pixmap::new(render_w, render_h)
        .ok_or_else(|| CoreError::ImageFormat("SVG render: invalid dimensions".into()))?;
    resvg::render(&tree, Transform::from_scale(scale, scale), &mut pixmap.as_mut());

    // tiny_skia stores premultiplied RGBA; demultiply so the result matches
    // the alpha convention used by the rest of the pipeline (the seg models
    // expect straight RGB, and our compositor unmultiplies anyway).
    let mut rgba = pixmap.take();
    unmultiply_alpha(&mut rgba);
    let img = RgbaImage::from_raw(render_w, render_h, rgba)
        .ok_or_else(|| CoreError::ImageFormat("SVG render: buffer size mismatch".into()))?;
    Ok(DynamicImage::ImageRgba8(img))
}

/// Convert premultiplied RGBA bytes (resvg/tiny_skia output) to straight RGBA.
/// Pixels with alpha=0 stay zeroed (already correct).
fn unmultiply_alpha(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        let a = px[3];
        if a == 0 || a == 255 {
            continue;
        }
        // ((c * 255) + a/2) / a — round-half-up integer divide.
        let inv = |c: u8| -> u8 {
            let n = (c as u32 * 255 + (a as u32 / 2)) / a as u32;
            n.min(255) as u8
        };
        px[0] = inv(px[0]);
        px[1] = inv(px[1]);
        px[2] = inv(px[2]);
    }
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
    let rgba = img.to_rgba8();
    let src = Image::from_vec_u8(rgba.width(), rgba.height(), rgba.into_raw(), PixelType::U8x4)
        .expect("valid RGBA buffer");
    let mut dst = Image::new(nw, nh, PixelType::U8x4);
    Resizer::new().resize(&src, &mut dst, None).expect("resize failed");
    let out = image::RgbaImage::from_raw(nw, nh, dst.into_vec()).expect("valid dimensions");
    DynamicImage::ImageRgba8(out)
}

/// Alpha-blend an RGBA image onto a solid background color, making all pixels fully opaque.
pub fn apply_background_color(img: &mut RgbaImage, bg: [u8; 3]) {
    // Sequential: this runs inside a rayon worker thread (subprocess),
    // so nested rayon parallelism would cause deadlock or thread starvation.
    let raw = img.as_mut();
    for pixel in raw.chunks_mut(4) {
        let a = pixel[3] as f32 / 255.0;
        if a < 1.0 {
            let inv = 1.0 - a;
            pixel[0] = (pixel[0] as f32 * a + bg[0] as f32 * inv) as u8;
            pixel[1] = (pixel[1] as f32 * a + bg[1] as f32 * inv) as u8;
            pixel[2] = (pixel[2] as f32 * a + bg[2] as f32 * inv) as u8;
            pixel[3] = 255;
        }
    }
}

/// Encode an RgbaImage as PNG bytes with fast compression.
pub fn encode_rgba_png(img: &RgbaImage) -> Result<Vec<u8>, CoreError> {
    let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
    let encoder = PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, PngFilter::Sub);
    encoder.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
        .map_err(CoreError::from)?;
    Ok(buf)
}

/// Encode a GrayImage (L8) as PNG bytes with fast compression.
pub fn encode_gray_png(img: &GrayImage) -> Result<Vec<u8>, CoreError> {
    let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
    let encoder = PngEncoder::new_with_quality(&mut buf, CompressionType::Fast, PngFilter::Sub);
    encoder.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
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

    /// Minimal valid SVG: a 32x16 viewBox with one solid red rect.
    const SAMPLE_SVG: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="16" viewBox="0 0 32 16">
  <rect width="32" height="16" fill="rgb(255,0,0)"/>
</svg>"#;

    #[test]
    fn looks_like_svg_detects_xml_prologue() {
        assert!(looks_like_svg(SAMPLE_SVG));
        assert!(looks_like_svg(b"<svg xmlns='...'><circle/></svg>"));
        assert!(looks_like_svg(b"\xEF\xBB\xBF<svg/>"), "BOM-prefixed SVG");
        assert!(looks_like_svg(b"  \n\t<svg/>"), "leading whitespace");
    }

    #[test]
    fn looks_like_svg_rejects_raster_formats() {
        let png = minimal_png_bytes();
        assert!(!looks_like_svg(&png));
        assert!(!looks_like_svg(b"\xff\xd8\xff\xe0JFIF"), "JPEG header");
        assert!(!looks_like_svg(b"GIF89a"), "GIF header");
        assert!(!looks_like_svg(b""), "empty buffer");
    }

    #[test]
    fn has_svg_extension_case_insensitive() {
        use std::path::Path;
        assert!(has_svg_extension(Path::new("foo.svg")));
        assert!(has_svg_extension(Path::new("foo.SVG")));
        assert!(has_svg_extension(Path::new("foo.SvG")));
        assert!(!has_svg_extension(Path::new("foo.png")));
        assert!(!has_svg_extension(Path::new("foo")));
    }

    #[test]
    fn load_image_from_bytes_rasterizes_svg() {
        let img = load_image_from_bytes(SAMPLE_SVG)
            .expect("SVG bytes should rasterize");
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 16);
        // The center pixel should be roughly pure red after demultiplying.
        let rgba = img.to_rgba8();
        let px = rgba.get_pixel(16, 8);
        assert!(px[0] > 240, "red channel should be ~255, got {}", px[0]);
        assert!(px[1] < 16, "green channel should be ~0, got {}", px[1]);
        assert!(px[2] < 16, "blue channel should be ~0, got {}", px[2]);
        assert_eq!(px[3], 255, "fully opaque rect");
    }

    #[test]
    fn rasterize_svg_caps_oversize_input() {
        // 20000x20000 would normally exceed LARGE_IMAGE_LIMIT (8000); rasterize
        // must scale the render down so we don't allocate a 1.6 GB pixmap.
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="20000" height="20000"><rect width="20000" height="20000" fill="black"/></svg>"#,
        );
        let img = load_image_from_bytes(svg.as_bytes())
            .expect("oversize SVG should still rasterize after capping");
        assert!(img.width() <= LARGE_IMAGE_LIMIT);
        assert!(img.height() <= LARGE_IMAGE_LIMIT);
    }

    #[test]
    fn unmultiply_alpha_recovers_straight_rgba() {
        // Premultiplied (128, 0, 0, 128) → straight (~255, 0, 0, 128).
        let mut buf = vec![128, 0, 0, 128];
        unmultiply_alpha(&mut buf);
        assert!(buf[0] >= 254, "unmultiplied red should be ~255, got {}", buf[0]);
        assert_eq!(buf[3], 128, "alpha must not change");
        // Fully transparent stays untouched.
        let mut zero = vec![0, 0, 0, 0];
        unmultiply_alpha(&mut zero);
        assert_eq!(zero, vec![0, 0, 0, 0]);
        // Fully opaque stays untouched.
        let mut opaque = vec![100, 200, 50, 255];
        unmultiply_alpha(&mut opaque);
        assert_eq!(opaque, vec![100, 200, 50, 255]);
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

    #[test]
    fn test_encode_gray_png_round_trip() {
        // Gradient 0..255 across an 8x1 strip — round-tripping through PNG
        // must preserve every value.
        let mut img = GrayImage::new(8, 1);
        for x in 0..8 {
            img.put_pixel(x, 0, image::Luma([(x * 32) as u8]));
        }
        let bytes = encode_gray_png(&img).expect("gray encode");
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);
        let decoded = image::load_from_memory(&bytes).expect("decode back").to_luma8();
        assert_eq!(decoded.dimensions(), (8, 1));
        for x in 0..8 {
            assert_eq!(decoded.get_pixel(x, 0).0[0], (x * 32) as u8);
        }
    }
}
