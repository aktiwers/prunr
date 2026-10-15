use std::io::Cursor;
use std::path::Path;
use image::{DynamicImage, GrayImage, ImageReader, RgbaImage};
use image::codecs::png::{PngEncoder, CompressionType, FilterType as PngFilter};
use image::ImageEncoder;
use image::buffer::ConvertBuffer;
use fast_image_resize::{images::{Image, ImageRef}, PixelType, ResizeAlg, ResizeOptions, Resizer};
pub use fast_image_resize::FilterType as ResizeFilter;
use crate::types::{CoreError, LARGE_IMAGE_LIMIT};

/// SIMD convolution resize of a borrowed interleaved buffer; the
/// channels are resampled independently (straight alpha, like the rest
/// of the pipeline).
fn resize_raw(src: &[u8], w: u32, h: u32, pixel: PixelType, dst_w: u32, dst_h: u32, filter: ResizeFilter) -> Vec<u8> {
    let src = ImageRef::new(w, h, src, pixel).expect("buffer matches its dimensions");
    let mut dst = Image::new(dst_w, dst_h, pixel);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter));
    Resizer::new().resize(&src, &mut dst, Some(&options)).expect("resize failed");
    dst.into_vec()
}

/// SIMD resize of a borrowed RGB image with the given filter.
fn resize_rgb(src: &image::RgbImage, dst_width: u32, dst_height: u32, filter: ResizeFilter) -> image::RgbImage {
    let out = resize_raw(src.as_raw(), src.width(), src.height(), PixelType::U8x3, dst_width, dst_height, filter);
    image::RgbImage::from_raw(dst_width, dst_height, out).expect("valid dimensions")
}

/// SIMD Lanczos3 resize of an RGB image, borrowed.
pub fn resize_rgb_image_lanczos3(src: &image::RgbImage, dst_width: u32, dst_height: u32) -> image::RgbImage {
    resize_rgb(src, dst_width, dst_height, ResizeFilter::Lanczos3)
}

/// SIMD-accelerated Lanczos3 resize for single-channel (gray) images.
pub fn resize_gray_lanczos3(src: &GrayImage, dst_width: u32, dst_height: u32) -> GrayImage {
    let out = resize_raw(src.as_raw(), src.width(), src.height(), PixelType::U8, dst_width, dst_height, ResizeFilter::Lanczos3);
    GrayImage::from_raw(dst_width, dst_height, out).expect("valid dimensions")
}

/// SIMD-accelerated resize of an RGBA image with the given filter. The
/// source is borrowed, so a 4K photo costs one pass, no copy.
pub fn resize_rgba(src: &RgbaImage, dst_width: u32, dst_height: u32, filter: ResizeFilter) -> RgbaImage {
    let out = resize_raw(src.as_raw(), src.width(), src.height(), PixelType::U8x4, dst_width, dst_height, filter);
    RgbaImage::from_raw(dst_width, dst_height, out).expect("valid dimensions")
}

/// Resize a decoded image to RGBA in its own layout: RGB8 and RGBA8
/// sources are resampled as they are and only the small result is
/// expanded, so a 12 MP JPEG never materialises as a 48 MB RGBA buffer.
pub fn resize_to_rgba(img: &DynamicImage, dst_width: u32, dst_height: u32, filter: ResizeFilter) -> RgbaImage {
    match img {
        DynamicImage::ImageRgba8(rgba) => resize_rgba(rgba, dst_width, dst_height, filter),
        DynamicImage::ImageRgb8(rgb) => resize_rgb(rgb, dst_width, dst_height, filter).convert(),
        other => resize_rgba(&other.to_rgba8(), dst_width, dst_height, filter),
    }
}

/// `part` (x, y, w, h) of `img` as a progress preview's `width × height`
/// straight RGBA: its own pixels at its own size (an owned image asked for
/// whole is handed over without a copy), else Lanczos3-resampled whole and
/// cut, so a part is the same pixels as that place of the whole preview.
pub fn preview_rgba(img: std::borrow::Cow<'_, RgbaImage>, (width, height): (u32, u32), part: (u32, u32, u32, u32)) -> Vec<u8> {
    let whole = part == (0, 0, width, height);
    let cut = |full: &RgbaImage| image::imageops::crop_imm(full, part.0, part.1, part.2, part.3).to_image().into_raw();
    match (img.dimensions() == (width, height), whole) {
        (true, true) => img.into_owned().into_raw(),
        (true, false) => cut(&img),
        (false, true) => resize_rgba(&img, width, height, ResizeFilter::Lanczos3).into_raw(),
        (false, false) => cut(&resize_rgba(&img, width, height, ResizeFilter::Lanczos3)),
    }
}

/// How far from a sample Lanczos3 weighs source pixels, in source pixels
/// when enlarging and in output pixels when shrinking.
const LANCZOS3_SUPPORT: f64 = 3.0;

/// Along one side of `span` source pixels previewed at `out` pixels by
/// `preview_rgba`: the preview pixels a change to `[at, at + len)` can
/// reach, as (start, length). The change itself when nothing is
/// resampled; otherwise every pixel whose Lanczos3 window touches it, a
/// pixel to spare on each side. Never empty.
pub fn preview_reach(at: u32, len: u32, span: u32, out: u32) -> (u32, u32) {
    let last = out.saturating_sub(1);
    if span == out {
        let start = at.min(last);
        return (start, len.clamp(1, out - start));
    }
    let scale = f64::from(span) / f64::from(out.max(1));
    let radius = LANCZOS3_SUPPORT * scale.max(1.0);
    let start = ((f64::from(at) - radius) / scale - 0.5).floor().max(0.0) as u32;
    let end = ((f64::from(at + len) + radius) / scale - 0.5).ceil().min(f64::from(out)) as u32;
    let start = start.min(last);
    (start, end.max(start + 1) - start)
}

/// SIMD-accelerated Lanczos3 resize for RGB images. The `to_rgb8` call
/// is unavoidable when `img` is a `DynamicImage` of a different layout
/// (e.g. RGBA), but the resulting `RgbImage` is then borrowed into the
/// resizer rather than copied again.
pub fn resize_rgb_lanczos3(img: &DynamicImage, dst_width: u32, dst_height: u32) -> image::RgbImage {
    resize_rgb_image_lanczos3(&img.to_rgb8(), dst_width, dst_height)
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
        .map_err(CoreError::Io)?
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

/// `(w, h)` scaled so the long side is `max_side`, the aspect kept and
/// neither side zero.
pub fn fit_long_side(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let k = f64::from(max_side) / f64::from(w.max(h).max(1));
    let side = |v: u32| ((f64::from(v) * k).round() as u32).max(1);
    (side(w), side(h))
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
    let (nw, nh) = fit_long_side(w, h, max_dim);
    DynamicImage::ImageRgba8(resize_to_rgba(&img, nw, nh, ResizeFilter::Lanczos3))
}

/// Alpha-blend an RGBA image onto a solid background color, making all pixels fully opaque.
pub fn apply_background_color(img: &mut RgbaImage, bg: [u8; 3]) {
    // Sequential: this runs inside a rayon worker thread (subprocess),
    // so nested rayon parallelism would cause deadlock or thread starvation.
    let raw = img.as_mut();
    for pixel in raw.chunks_mut(4) {
        if pixel[3] < 255 {
            let rgb = blend_over(pixel, bg);
            pixel[..3].copy_from_slice(&rgb);
            pixel[3] = 255;
        }
    }
}

/// One straight-alpha RGBA pixel composited over `bg`.
pub(crate) fn blend_over(px: &[u8], bg: [u8; 3]) -> [u8; 3] {
    let a = px[3] as f32 / 255.0;
    let inv = 1.0 - a;
    std::array::from_fn(|c| (px[c] as f32 * a + bg[c] as f32 * inv) as u8)
}

/// Alpha-blend an RGBA image onto a background image, making opaque any
/// pixel where the bg layout reaches. `fit` controls how `bg` is positioned
/// inside the foreground frame:
///
/// - `Cover`: scaled to fill, may crop (every alpha-zero pixel covered)
/// - `Contain`: bg may not cover the whole frame; uncovered transparent
///   pixels stay transparent
/// - `Tile`: repeated at native size, fully covers
///
/// Sequential for the same reason as `apply_background_color` (subprocess
/// rayon worker — nested parallelism deadlocks).
pub fn apply_background_image(
    img: &mut RgbaImage,
    bg: &DynamicImage,
    fit: crate::types::BgImageFit,
) {
    let (fw, fh) = (img.width(), img.height());
    let bg_full = build_bg_layer(bg, fw, fh, fit);
    let raw_bg = bg_full.as_raw();
    let raw = img.as_mut();
    for (i, pixel) in raw.chunks_mut(4).enumerate() {
        let a = pixel[3] as f32 / 255.0;
        if a >= 1.0 { continue; }
        let bi = i * 4;
        let bg_a = raw_bg[bi + 3] as f32 / 255.0;
        if bg_a <= 0.0 { continue; }
        let inv = (1.0 - a) * bg_a;
        // Straight alpha: the blended colour is divided by the new alpha.
        let out_a = a + inv;
        for c in 0..3 {
            pixel[c] = ((pixel[c] as f32 * a + raw_bg[bi + c] as f32 * inv) / out_a).round().clamp(0.0, 255.0) as u8;
        }
        pixel[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// Build a `dst_w × dst_h` RGBA buffer with `bg` positioned per `fit`.
/// Pixels outside the bg region have alpha=0 (preserved transparency).
fn build_bg_layer(
    bg: &DynamicImage,
    dst_w: u32,
    dst_h: u32,
    fit: crate::types::BgImageFit,
) -> RgbaImage {
    use crate::types::BgImageFit;
    let mut out = RgbaImage::new(dst_w, dst_h);
    let (sw, sh) = (bg.width(), bg.height());
    if sw == 0 || sh == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }
    match fit {
        BgImageFit::Cover => {
            let scale = (dst_w as f32 / sw as f32).max(dst_h as f32 / sh as f32);
            let scaled_w = ((sw as f32 * scale).ceil() as u32).max(dst_w);
            let scaled_h = ((sh as f32 * scale).ceil() as u32).max(dst_h);
            let scaled = resize_rgb_lanczos3(bg, scaled_w, scaled_h);
            let (ox, oy) = ((scaled_w - dst_w) / 2, (scaled_h - dst_h) / 2);
            let cropped = image::imageops::crop_imm(&scaled, ox, oy, dst_w, dst_h).to_image();
            image::imageops::replace(&mut out, &cropped.convert(), 0, 0);
        }
        BgImageFit::Contain => {
            let scale = (dst_w as f32 / sw as f32).min(dst_h as f32 / sh as f32);
            let inner_w = ((sw as f32 * scale).round() as u32).max(1).min(dst_w);
            let inner_h = ((sh as f32 * scale).round() as u32).max(1).min(dst_h);
            let scaled = resize_rgb_lanczos3(bg, inner_w, inner_h);
            let ox = ((dst_w - inner_w) / 2) as i64;
            let oy = ((dst_h - inner_h) / 2) as i64;
            image::imageops::replace(&mut out, &scaled.convert(), ox, oy);
        }
        BgImageFit::Tile => {
            image::imageops::tile(&mut out, &bg.to_rgba8());
        }
    }
    out
}

/// Stream-encode an RgbaImage into a `Write` sink (typically
/// `BufWriter<File>`). Skips the intermediate `Vec<u8>` that
/// `encode_rgba_png` allocates — relevant when the caller would
/// immediately `fs::write` the bytes anyway. Saves 5–30 MB transient
/// per encode at 4 K. Sink is flushed by the caller.
pub fn encode_rgba_png_into<W: std::io::Write>(img: &RgbaImage, w: W) -> Result<(), CoreError> {
    write_fast_png(w, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
}

fn write_fast_png<W: std::io::Write>(w: W, bytes: &[u8], width: u32, height: u32, ty: image::ExtendedColorType) -> Result<(), CoreError> {
    PngEncoder::new_with_quality(w, CompressionType::Fast, PngFilter::Sub)
        .write_image(bytes, width, height, ty)
        .map_err(CoreError::from)
}

/// Encode an RgbaImage as PNG bytes with fast compression. Convenience
/// wrapper over `encode_rgba_png_into` for callers that need owned bytes
/// (e.g. shipping over a channel). Streaming callers writing to a file
/// should use `encode_rgba_png_into` directly.
pub fn encode_rgba_png(img: &RgbaImage) -> Result<Vec<u8>, CoreError> {
    let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
    encode_rgba_png_into(img, &mut buf)?;
    Ok(buf)
}

/// A saved image's file format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    /// Lossless, keeps transparency.
    #[default]
    Png,
    /// Lossy at quality 92; transparency is flattened onto white.
    Jpeg,
    /// Lossless WebP, keeps transparency.
    Webp,
}

impl ExportFormat {
    pub const ALL: [Self; 3] = [Self::Png, Self::Jpeg, Self::Webp];

    /// The file extensions of this format, the one written first.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Png => &["png"],
            Self::Jpeg => &["jpg", "jpeg"],
            Self::Webp => &["webp"],
        }
    }

    pub fn extension(self) -> &'static str {
        self.extensions()[0]
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp => "WebP",
        }
    }

    /// The format a file name asks for by its extension.
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Self::ALL.into_iter().find(|f| f.extensions().contains(&ext.as_str()))
    }

    pub fn encode(self, img: &RgbaImage) -> Result<Vec<u8>, CoreError> {
        match self {
            Self::Png => encode_rgba_png(img),
            Self::Jpeg => {
                const JPEG_QUALITY: u8 = 92;
                let flat = flatten_on_white(img);
                let mut buf = Vec::with_capacity(img.as_raw().len() / 8);
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY)
                    .encode(flat.as_raw(), flat.width(), flat.height(), image::ExtendedColorType::Rgb8)
                    .map_err(CoreError::from)?;
                Ok(buf)
            }
            Self::Webp => {
                let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
                image::codecs::webp::WebPEncoder::new_lossless(&mut buf)
                    .encode(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
                    .map_err(CoreError::from)?;
                Ok(buf)
            }
        }
    }
}

/// Straight-alpha RGBA composited over white, as RGB (JPEG has no alpha).
fn flatten_on_white(img: &RgbaImage) -> image::RgbImage {
    let rgb: Vec<u8> = img.as_raw().chunks_exact(4).flat_map(|px| blend_over(px, [255; 3])).collect();
    // invariant: 3 bytes for each of width × height pixels.
    image::RgbImage::from_raw(img.width(), img.height(), rgb).expect("one RGB triple per RGBA pixel")
}

/// Encode a GrayImage (L8) as PNG bytes with fast compression.
pub fn encode_gray_png(img: &GrayImage) -> Result<Vec<u8>, CoreError> {
    let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
    encode_gray_png_into(img, &mut buf)?;
    Ok(buf)
}

/// `encode_gray_png` streamed into `w`.
pub fn encode_gray_png_into<W: std::io::Write>(img: &GrayImage, w: W) -> Result<(), CoreError> {
    write_fast_png(w, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
}

/// Encode a 16-bit grey image as PNG bytes with fast compression.
pub fn encode_gray16_png(img: &image::ImageBuffer<image::Luma<u16>, Vec<u16>>) -> Result<Vec<u8>, CoreError> {
    use image::EncodableLayout;
    let mut buf = Vec::with_capacity(img.as_raw().len());
    write_fast_png(&mut buf, img.as_raw().as_bytes(), img.width(), img.height(), image::ExtendedColorType::L16)?;
    Ok(buf)
}

/// `img` as RGBA, borrowed when it already is and converted otherwise.
pub fn rgba_view(img: &DynamicImage) -> std::borrow::Cow<'_, RgbaImage> {
    img.as_rgba8().map_or_else(|| std::borrow::Cow::Owned(img.to_rgba8()), std::borrow::Cow::Borrowed)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_16_bit_grey_png_keeps_every_value() {
        let img = image::ImageBuffer::from_fn(7, 3, |x, y| image::Luma([(x * 9000 + y * 300) as u16]));
        let back = image::load_from_memory(&super::encode_gray16_png(&img).expect("encode")).expect("decode");
        assert_eq!(back.as_luma16(), Some(&img));
    }

    #[test]
    fn an_opaque_rgb_photo_shrinks_to_the_pixels_of_its_rgba_copy() {
        let rgb = image::RgbImage::from_fn(97, 61, |x, y| image::Rgb([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x ^ y) % 256) as u8]));
        let rgba = image::DynamicImage::ImageRgb8(rgb.clone()).to_rgba8();
        let filter = super::ResizeFilter::Lanczos3;
        assert_eq!(super::resize_to_rgba(&image::DynamicImage::ImageRgb8(rgb), 23, 14, filter), super::resize_rgba(&rgba, 23, 14, filter));
    }

    /// `downscale_image` resampling each source in its own layout gives the
    /// same bytes as converting it to RGBA first: noise photos of every
    /// pixel type it meets, at several sizes and ratios.
    #[test]
    fn downscale_in_the_source_layout_matches_the_rgba_copy() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut noise = |n: usize| -> Vec<u8> { (0..n).map(|_| rng.random()).collect() };
        for (w, h, max) in [(1000, 700, 300), (517, 911, 256), (64, 48, 63), (4100, 30, 4096)] {
            let rgb = image::RgbImage::from_raw(w, h, noise((w * h * 3) as usize)).unwrap();
            let rgba = image::RgbaImage::from_raw(w, h, noise((w * h * 4) as usize)).unwrap();
            let luma = image::GrayImage::from_raw(w, h, noise((w * h) as usize)).unwrap();
            for img in [image::DynamicImage::ImageRgb8(rgb), image::DynamicImage::ImageRgba8(rgba), image::DynamicImage::ImageLuma8(luma)] {
                let (nw, nh) = super::fit_long_side(w, h, max);
                let old = super::resize_rgba(&img.to_rgba8(), nw, nh, super::ResizeFilter::Lanczos3);
                let new = super::resize_to_rgba(&img, nw, nh, super::ResizeFilter::Lanczos3);
                assert!(old == new, "{:?} {w}x{h} -> {nw}x{nh}", img.color());
            }
        }
    }

    #[test]
    fn an_rgb_resize_keeps_a_flat_colour() {
        let src = image::RgbImage::from_pixel(64, 32, image::Rgb([20, 140, 230]));
        let out = super::resize_rgb_image_lanczos3(&src, 8, 4);
        assert_eq!(out.dimensions(), (8, 4));
        assert!(out.pixels().all(|p| p.0 == [20, 140, 230]));
    }


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
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20000" height="20000"><rect width="20000" height="20000" fill="black"/></svg>"#.to_string();
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
    fn apply_background_image_blends_under_transparent_pixels() {
        // 2×1 RGBA: opaque red, fully transparent (so bg should show through).
        let mut fg = RgbaImage::new(2, 1);
        fg.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        fg.put_pixel(1, 0, Rgba([0, 0, 0, 0]));
        // Solid green 4×2 bg — gets resize-cover'd to 2×1.
        let mut bg = image::RgbImage::new(4, 2);
        for px in bg.pixels_mut() {
            *px = image::Rgb([0, 255, 0]);
        }
        let bg_dyn = DynamicImage::ImageRgb8(bg);
        apply_background_image(&mut fg, &bg_dyn, crate::types::BgImageFit::Cover);
        assert_eq!(fg.get_pixel(0, 0), &Rgba([255, 0, 0, 255]),
            "opaque pixel must stay unchanged");
        assert_eq!(fg.get_pixel(1, 0), &Rgba([0, 255, 0, 255]),
            "transparent pixel must reveal bg, alpha forced to 255");
    }

    #[test]
    fn apply_background_image_preserves_partial_alpha_blend() {
        let mut fg = RgbaImage::new(1, 1);
        fg.put_pixel(0, 0, Rgba([200, 0, 0, 128])); // half-transparent red
        let bg_dyn = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            1, 1, image::Rgb([0, 0, 200])));
        apply_background_image(&mut fg, &bg_dyn, crate::types::BgImageFit::Cover);
        let p = fg.get_pixel(0, 0);
        // a = 128/255 ≈ 0.502; blended R ≈ 200*0.502 = 100, B ≈ 200*0.498 = 99
        assert!((90..=110).contains(&p[0]), "R was {}", p[0]);
        assert!((90..=110).contains(&p[2]), "B was {}", p[2]);
        assert_eq!(p[3], 255);
    }

    #[test]
    fn apply_background_image_handles_zero_size_bg_gracefully() {
        let mut fg = RgbaImage::from_pixel(2, 2, Rgba([100, 100, 100, 0]));
        // 0×0 bg — every fit produces an all-transparent layer; the
        // foreground keeps its alpha=0 pixels untouched (no division by zero).
        let bg_dyn = DynamicImage::ImageRgb8(image::RgbImage::new(0, 0));
        apply_background_image(&mut fg, &bg_dyn, crate::types::BgImageFit::Cover);
        for p in fg.pixels() {
            assert_eq!(p, &Rgba([100, 100, 100, 0]));
        }
    }

    #[test]
    fn apply_background_image_contain_leaves_letterbox_transparent() {
        // Wide foreground (4×1) with a square 1×1 green bg. Contain centres
        // the bg into a 1×1 block — the surrounding alpha-0 pixels in the
        // foreground stay alpha-0 (transparent letterbox).
        let mut fg = RgbaImage::from_pixel(4, 1, Rgba([0, 0, 0, 0]));
        let bg = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([0, 255, 0])));
        apply_background_image(&mut fg, &bg, crate::types::BgImageFit::Contain);
        // Centre column (or close to it) gets the bg green; edges stay transparent.
        let opaque_count = fg.pixels().filter(|p| p[3] > 0).count();
        let transparent_count = fg.pixels().filter(|p| p[3] == 0).count();
        assert!(opaque_count >= 1, "Contain must paint at least one pixel of bg");
        assert!(transparent_count >= 2, "Contain must letterbox the rest as transparent");
    }

    #[test]
    fn apply_background_image_tile_covers_every_pixel() {
        let mut fg = RgbaImage::from_pixel(4, 2, Rgba([0, 0, 0, 0]));
        // 2×1 red+blue striped bg → tiled, every fg pixel gets red or blue.
        let mut bg = image::RgbImage::new(2, 1);
        bg.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        bg.put_pixel(1, 0, image::Rgb([0, 0, 255]));
        let bg_dyn = DynamicImage::ImageRgb8(bg);
        apply_background_image(&mut fg, &bg_dyn, crate::types::BgImageFit::Tile);
        for p in fg.pixels() {
            assert_eq!(p[3], 255, "Tile must cover every pixel");
            // Each pixel is either red or blue (not black).
            assert!(p[0] == 255 || p[2] == 255, "Tile sample must come from bg");
        }
    }

    /// A preview at the image's own size is its pixels; any other size is
    /// the Lanczos3 resample, borrowed or owned alike; a part is that
    /// place of the whole.
    #[test]
    fn a_preview_is_the_image_at_its_size_or_its_lanczos3_resample() {
        use std::borrow::Cow;
        let img = RgbaImage::from_fn(40, 30, |x, y| image::Rgba([(x * 6) as u8, (y * 8) as u8, ((x + y) * 3) as u8, 255 - x as u8]));
        let cut = |raw: Vec<u8>, w: u32, h: u32, part: (u32, u32, u32, u32)| {
            image::imageops::crop_imm(&RgbaImage::from_raw(w, h, raw).unwrap(), part.0, part.1, part.2, part.3).to_image().into_raw()
        };
        assert_eq!(preview_rgba(Cow::Borrowed(&img), (40, 30), (0, 0, 40, 30)), img.as_raw().clone());
        assert_eq!(preview_rgba(Cow::Owned(img.clone()), (40, 30), (0, 0, 40, 30)), img.as_raw().clone());
        assert_eq!(preview_rgba(Cow::Borrowed(&img), (40, 30), (3, 4, 10, 7)), cut(img.as_raw().clone(), 40, 30, (3, 4, 10, 7)));
        let resampled = resize_rgba(&img, 17, 13, ResizeFilter::Lanczos3).into_raw();
        assert_eq!(preview_rgba(Cow::Borrowed(&img), (17, 13), (0, 0, 17, 13)), resampled);
        assert_eq!(preview_rgba(Cow::Owned(img.clone()), (17, 13), (0, 0, 17, 13)), resampled);
        assert_eq!(preview_rgba(Cow::Owned(img), (17, 13), (2, 1, 9, 5)), cut(resampled, 17, 13, (2, 1, 9, 5)));
    }

    /// Changing a block of an image changes its preview only within the
    /// reach: every preview pixel that differs lies inside it, shrunk or
    /// enlarged, and an unscaled preview's reach is the block itself.
    #[test]
    fn a_change_reaches_only_the_preview_pixels_its_reach_names() {
        use std::borrow::Cow;
        let base = RgbaImage::from_fn(120, 90, |x, y| image::Rgba([(x * 2) as u8, (y * 3) as u8, ((x ^ y) * 5) as u8, 255]));
        let (bx, by, bw, bh) = (37, 21, 19, 11);
        let mut changed = base.clone();
        for y in by..by + bh {
            for x in bx..bx + bw {
                changed.put_pixel(x, y, image::Rgba([255, 0, 255, 255]));
            }
        }
        for (w, h) in [(120, 90), (47, 35), (13, 10), (200, 150)] {
            let (px, pw) = preview_reach(bx, bw, 120, w);
            let (py, ph) = preview_reach(by, bh, 90, h);
            let before = preview_rgba(Cow::Borrowed(&base), (w, h), (0, 0, w, h));
            let after = preview_rgba(Cow::Borrowed(&changed), (w, h), (0, 0, w, h));
            for (i, (a, b)) in before.chunks(4).zip(after.chunks(4)).enumerate() {
                let (x, y) = (i as u32 % w, i as u32 / w);
                if a != b {
                    assert!((px..px + pw).contains(&x) && (py..py + ph).contains(&y), "{w}×{h}: ({x}, {y}) changed outside the reach");
                }
            }
            if (w, h) == (120, 90) {
                assert_eq!((px, py, pw, ph), (bx, by, bw, bh), "unscaled, the reach is the change");
            }
        }
        assert_eq!(preview_reach(0, 1, 100, 1), (0, 1), "never empty, never past the side");
    }

    #[test]
    fn fit_long_side_keeps_the_aspect_and_never_reaches_zero() {
        assert_eq!(fit_long_side(8000, 6000, 4096), (4096, 3072));
        assert_eq!(fit_long_side(6000, 8000, 2048), (1536, 2048));
        assert_eq!(fit_long_side(10_000, 3, 100), (100, 1), "a sliver keeps a pixel");
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
        let mut streamed = Vec::new();
        encode_gray_png_into(&img, &mut streamed).expect("gray encode into");
        assert_eq!(streamed, bytes, "streamed and buffered PNGs are the same bytes");
        let decoded = image::load_from_memory(&bytes).expect("decode back").to_luma8();
        assert_eq!(decoded.dimensions(), (8, 1));
        for x in 0..8 {
            assert_eq!(decoded.get_pixel(x, 0).0[0], (x * 32) as u8);
        }
    }

    #[test]
    fn rgba_view_borrows_rgba_and_converts_the_rest() {
        let rgba = DynamicImage::ImageRgba8(RgbaImage::new(2, 2));
        assert!(matches!(rgba_view(&rgba), std::borrow::Cow::Borrowed(_)));
        let rgb = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let view = rgba_view(&rgb);
        assert!(matches!(view, std::borrow::Cow::Owned(_)));
        assert_eq!(view.dimensions(), (2, 2));
    }

    /// Images hold straight alpha: where a see-through picture shows
    /// through, its colour stays its colour at its own transparency.
    #[test]
    fn a_see_through_background_keeps_its_colour() {
        let mut img = RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 0]));
        let bg = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, image::Rgba([200, 100, 50, 128])));
        apply_background_image(&mut img, &bg, crate::types::BgImageFit::Tile);
        let p = img.get_pixel(1, 1).0;
        assert_eq!(p[3], 128);
        assert!((p[0] as i32 - 200).abs() <= 1 && (p[1] as i32 - 100).abs() <= 1, "not darkened: {p:?}");
    }

    /// A colour fills what is transparent, blends under what is partly
    /// covered, and leaves what is opaque alone; the result is opaque.
    #[test]
    fn a_background_colour_fills_under_the_result() {
        let mut img = RgbaImage::from_vec(3, 1, vec![10, 20, 30, 255, 200, 200, 200, 128, 0, 0, 0, 0]).unwrap();
        apply_background_color(&mut img, [0, 0, 100]);
        assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 255]);
        let half = img.get_pixel(1, 0).0;
        assert!(half[3] == 255 && (99..=101).contains(&half[0]) && (148..=150).contains(&half[2]), "{half:?}");
        assert_eq!(img.get_pixel(2, 0).0, [0, 0, 100, 255]);
    }

    /// Each format round-trips through the decoder: PNG and WebP keep
    /// the pixels and alpha, JPEG flattens a transparent pixel to white.
    #[test]
    fn export_formats_round_trip_and_jpeg_flattens_alpha() {
        let mut img = RgbaImage::from_pixel(32, 32, image::Rgba([200, 40, 10, 255]));
        // A whole 16 px block, so JPEG's chroma subsampling cannot bleed into it.
        for y in 0..16 { for x in 0..16 { img.put_pixel(x, y, image::Rgba([0, 0, 0, 0])); } }
        for f in ExportFormat::ALL {
            let bytes = f.encode(&img).unwrap();
            let back = image::load_from_memory(&bytes).unwrap().to_rgba8();
            assert_eq!(back.dimensions(), (32, 32), "{f:?}");
            match f {
                ExportFormat::Jpeg => {
                    let p = back.get_pixel(4, 4).0;
                    assert!(p[0] > 240 && p[1] > 240 && p[2] > 240 && p[3] == 255, "transparent became white: {p:?}");
                }
                _ => assert_eq!(back, img, "{f:?} is lossless with alpha"),
            }
            let name = format!("a.{}", f.extension());
            assert_eq!(ExportFormat::from_path(std::path::Path::new(&name)), Some(f));
        }
        assert_eq!(ExportFormat::from_path(std::path::Path::new("x.JPEG")), Some(ExportFormat::Jpeg));
        assert_eq!(ExportFormat::from_path(std::path::Path::new("x.tif")), None);
    }
}
