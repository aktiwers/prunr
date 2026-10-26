use std::io::Cursor;
use std::path::Path;
use image::{DynamicImage, GrayImage, ImageReader, RgbaImage};
use image::codecs::png::{PngEncoder, CompressionType, FilterType as PngFilter};
use image::{ImageDecoder, ImageEncoder};
use image::buffer::ConvertBuffer;
use fast_image_resize::{images::{Image, ImageRef}, PixelType, ResizeAlg, ResizeOptions, Resizer};
pub use fast_image_resize::FilterType as ResizeFilter;
use crate::types::{CoreError, LARGE_IMAGE_LIMIT};

/// SIMD convolution resize of a borrowed interleaved buffer. RGBA is
/// weighted by its alpha inside the resize (in and out straight alpha),
/// so a transparent pixel's colour does not bleed into its neighbours;
/// a fully opaque one skips that, which changes nothing but the time.
fn resize_raw(src: &[u8], w: u32, h: u32, pixel: PixelType, dst_w: u32, dst_h: u32, filter: ResizeFilter) -> Vec<u8> {
    let opaque = pixel == PixelType::U8x4 && is_opaque(src);
    resize_raw_with(src, w, h, pixel, (dst_w, dst_h), filter, !opaque)
}

/// Every pixel of interleaved RGBA `rgba` fully opaque.
pub(crate) fn is_opaque(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|p| p[3] == u8::MAX)
}

fn resize_raw_with(src: &[u8], w: u32, h: u32, pixel: PixelType, (dst_w, dst_h): (u32, u32), filter: ResizeFilter, weigh_alpha: bool) -> Vec<u8> {
    let src = ImageRef::new(w, h, src, pixel).expect("buffer matches its dimensions");
    let mut dst = Image::new(dst_w, dst_h, pixel);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter)).use_alpha(weigh_alpha);
    // The resizer fails only on mismatched pixel types or a bad crop:
    // both images share `pixel` and no crop is asked.
    Resizer::new().resize(&src, &mut dst, Some(&options)).expect("same pixel type, no crop");
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

/// `image` grown to `w × h` by mirroring it across its right and bottom
/// edges (borrowed when it is that size already): a model fed a tile
/// sees photo past the edge, not a black band it bleeds into the result.
pub fn pad_mirrored<P>(image: &image::ImageBuffer<P, Vec<P::Subpixel>>, w: u32, h: u32) -> std::borrow::Cow<'_, image::ImageBuffer<P, Vec<P::Subpixel>>>
where
    P: image::Pixel + 'static,
{
    let (iw, ih) = image.dimensions();
    if (iw, ih) == (w, h) {
        return std::borrow::Cow::Borrowed(image);
    }
    if iw == 0 || ih == 0 {
        return std::borrow::Cow::Owned(image::ImageBuffer::new(w, h));
    }
    std::borrow::Cow::Owned(image::ImageBuffer::from_fn(w, h, |x, y| *image.get_pixel(mirror(x, iw), mirror(y, ih))))
}

/// `i` folded back into `0..len` by reflecting at both ends, the edge
/// pixel not repeated.
fn mirror(i: u32, len: u32) -> u32 {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let m = i % period;
    if m < len { m } else { period - m }
}

/// SIMD-accelerated Lanczos3 resize for RGB images. An RGB8 image is
/// read in place; any other layout is converted once first.
pub fn resize_rgb_lanczos3(img: &DynamicImage, dst_width: u32, dst_height: u32) -> image::RgbImage {
    match img {
        DynamicImage::ImageRgb8(rgb) => resize_rgb_image_lanczos3(rgb, dst_width, dst_height),
        other => resize_rgb_image_lanczos3(&other.to_rgb8(), dst_width, dst_height),
    }
}

/// Format by file extension: PNG, JPEG, WebP, BMP, or SVG (rasterized
/// via resvg).
pub fn load_image_from_path(path: &Path) -> Result<DynamicImage, CoreError> {
    if has_svg_extension(path) {
        let bytes = std::fs::read(path).map_err(CoreError::Io)?;
        return rasterize_svg(&bytes);
    }
    decode_upright(ImageReader::open(path).map_err(CoreError::Io)?)
}

/// Format by content sniff: SVG first, then the `image` crate's magic
/// bytes for PNG/JPEG/WebP/BMP.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    if looks_like_svg(bytes) {
        return rasterize_svg(bytes);
    }
    decode_upright(ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(CoreError::Io)?)
}

/// Decode a raster turned the way its EXIF says, as a viewer shows it:
/// phones store a portrait sideways with a note to turn it. Under the
/// same allocation limit `ImageReader::decode` keeps.
///
/// Peak: twice the decoded image while a quarter turn copies it.
fn decode_upright<R: std::io::BufRead + std::io::Seek>(reader: ImageReader<R>) -> Result<DynamicImage, CoreError> {
    let (reader, late) = late_png_orientation(reader);
    let mut decoder = reader.into_decoder()?;
    let orientation = orientation_of(&mut decoder, late);
    let mut limits = image::Limits::default();
    limits.reserve(decoder.total_bytes())?;
    decoder.set_limits(limits)?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// A raster's size as `decode_upright` turns it, from its header.
fn upright_dimensions<R: std::io::BufRead + std::io::Seek>(reader: ImageReader<R>) -> Option<(u32, u32)> {
    use image::metadata::Orientation;
    let (reader, late) = late_png_orientation(reader);
    let mut decoder = reader.into_decoder().ok()?;
    let (w, h) = decoder.dimensions();
    let quarter_turn = matches!(
        orientation_of(&mut decoder, late),
        Orientation::Rotate90 | Orientation::Rotate270 | Orientation::Rotate90FlipH | Orientation::Rotate270FlipH
    );
    Some(if quarter_turn { (h, w) } else { (w, h) })
}

/// The turn the header names, else one read after the pixels (`late`).
fn orientation_of(decoder: &mut impl image::ImageDecoder, late: Option<image::metadata::Orientation>) -> image::metadata::Orientation {
    use image::metadata::Orientation;
    match decoder.orientation() {
        Ok(Orientation::NoTransforms) | Err(_) => late.unwrap_or(Orientation::NoTransforms),
        Ok(turn) => turn,
    }
}

/// A PNG's EXIF turn read wherever its eXIf chunk sits: the PNG rules let
/// it follow the image data (ExifTool before 12.29 wrote it there), and a
/// decoder asked before the pixels has not read it yet. The chunk headers
/// are walked, their data skipped, and the reader put back where it was.
fn late_png_orientation<R: std::io::BufRead + std::io::Seek>(reader: ImageReader<R>) -> (ImageReader<R>, Option<image::metadata::Orientation>) {
    if reader.format() != Some(image::ImageFormat::Png) {
        return (reader, None);
    }
    let mut inner = reader.into_inner();
    let found = inner.stream_position().ok().and_then(|start| {
        let found = png_exif_orientation(&mut inner);
        inner.seek(std::io::SeekFrom::Start(start)).ok().and(found)
    });
    (ImageReader::with_format(inner, image::ImageFormat::Png), found)
}

fn png_exif_orientation<R: std::io::Read + std::io::Seek>(r: &mut R) -> Option<image::metadata::Orientation> {
    // Far past any real EXIF block: a damaged length allocates no more.
    const MAX_EXIF: u32 = 1 << 20;
    let mut signature = [0u8; 8];
    r.read_exact(&mut signature).ok()?;
    if signature != *b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    loop {
        let mut head = [0u8; 8];
        r.read_exact(&mut head).ok()?;
        let len = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        match &head[4..] {
            b"eXIf" if len <= MAX_EXIF => {
                let mut exif = vec![0u8; len as usize];
                r.read_exact(&mut exif).ok()?;
                return image::metadata::Orientation::from_exif_chunk(&exif);
            }
            b"IEND" => return None,
            _ => {
                r.seek(std::io::SeekFrom::Current(i64::from(len) + 4)).ok()?;
            }
        }
    }
}

fn has_svg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("svg"))
}

/// `<?xml` or `<svg` within the first 512 bytes, after whitespace and a
/// BOM. Not `<!--` or `<!DOCTYPE`: those match non-SVG XML and HTML.
fn looks_like_svg(bytes: &[u8]) -> bool {
    let probe = &bytes[..bytes.len().min(512)];
    let probe = probe.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(probe);
    let head = probe.trim_ascii_start();
    head.starts_with(b"<?xml") || head.starts_with(b"<svg")
}

/// Rasterize an SVG byte buffer to a `DynamicImage::ImageRgba8`. Uses the
/// SVG's intrinsic size, capped at `LARGE_IMAGE_LIMIT` so a pathological
/// `<svg width="100000">` can't blow memory.
fn rasterize_svg(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    use resvg::tiny_skia::{Pixmap, Transform};
    use resvg::usvg::{Options, Tree};

    let tree = Tree::from_data(bytes, &Options::default())
        .map_err(|e| CoreError::ImageFormat(format!("SVG parse error: {e}")))?;
    let (render_w, render_h, scale) = svg_render_size(&tree);

    let mut pixmap = Pixmap::new(render_w, render_h)
        .ok_or_else(|| CoreError::ImageFormat("SVG render: invalid dimensions".into()))?;
    resvg::render(&tree, Transform::from_scale(scale, scale), &mut pixmap.as_mut());

    // tiny_skia stores premultiplied RGBA; the pipeline is straight alpha.
    let mut rgba = pixmap.take();
    unmultiply_alpha(&mut rgba);
    let img = RgbaImage::from_raw(render_w, render_h, rgba)
        .ok_or_else(|| CoreError::ImageFormat("SVG render: buffer size mismatch".into()))?;
    Ok(DynamicImage::ImageRgba8(img))
}

/// The size an SVG renders at, and its scale: its intrinsic size, capped
/// at `LARGE_IMAGE_LIMIT` so `<svg width="100000">` cannot blow memory.
fn svg_render_size(tree: &resvg::usvg::Tree) -> (u32, u32, f32) {
    let size = tree.size();
    let intrinsic_w = size.width().ceil().max(1.0) as u32;
    let intrinsic_h = size.height().ceil().max(1.0) as u32;
    let max_dim = intrinsic_w.max(intrinsic_h);
    let scale = if max_dim > LARGE_IMAGE_LIMIT { LARGE_IMAGE_LIMIT as f32 / max_dim as f32 } else { 1.0 };
    let render_w = ((intrinsic_w as f32) * scale).round().max(1.0) as u32;
    let render_h = ((intrinsic_h as f32) * scale).round().max(1.0) as u32;
    (render_w, render_h, scale)
}

/// The size `bytes` decode to, read without decoding a raster's pixels;
/// an SVG is parsed, not rendered. `None` when it is no image.
pub fn image_dimensions_from_bytes(bytes: &[u8]) -> Option<(u32, u32)> {
    if looks_like_svg(bytes) {
        let tree = resvg::usvg::Tree::from_data(bytes, &resvg::usvg::Options::default()).ok()?;
        let (w, h, _) = svg_render_size(&tree);
        return Some((w, h));
    }
    upright_dimensions(ImageReader::new(Cursor::new(bytes)).with_guessed_format().ok()?)
}

/// The size the file at `path` decodes to: a raster's header only, so the
/// file is never loaded; an SVG (by extension, as `load_image_from_path`)
/// is read and parsed.
pub fn image_dimensions_from_path(path: &Path) -> Option<(u32, u32)> {
    if has_svg_extension(path) {
        return std::fs::read(path).ok().and_then(|bytes| image_dimensions_from_bytes(&bytes));
    }
    let file = std::fs::File::open(path).ok()?;
    upright_dimensions(ImageReader::new(std::io::BufReader::new(file)).with_guessed_format().ok()?)
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

/// Either side past `LARGE_IMAGE_LIMIT`, for callers that only have the
/// dimensions (e.g. read from a file header).
pub fn exceeds_large_image_limit(w: u32, h: u32) -> bool {
    w > LARGE_IMAGE_LIMIT || h > LARGE_IMAGE_LIMIT
}

/// `Some(CoreError::LargeImage)` when either side exceeds `LARGE_IMAGE_LIMIT`.
pub fn check_large_image(img: &DynamicImage) -> Option<CoreError> {
    let (w, h) = (img.width(), img.height());
    if exceeds_large_image_limit(w, h) {
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

/// Lanczos3 downscale so the long side is at most `max_dim`, aspect kept.
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
    std::array::from_fn(|c| crate::math::round_u8(px[c] as f32 * a + bg[c] as f32 * inv))
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
/// Sequential: it runs inside a subprocess rayon worker, where nested
/// parallelism deadlocks.
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
) -> std::borrow::Cow<'_, RgbaImage> {
    use crate::types::BgImageFit;
    // At the frame's size Cover and Contain place the picture as it is (a
    // background effect always is): no resample, crop or copy. It goes
    // opaque, as the resampled path's RGB does.
    if fit != BgImageFit::Tile && (bg.width(), bg.height()) == (dst_w, dst_h) {
        if let Some(rgba) = bg.as_rgba8().filter(|rgba| is_opaque(rgba)) {
            return std::borrow::Cow::Borrowed(rgba);
        }
        let mut rgba = bg.to_rgba8();
        rgba.pixels_mut().for_each(|p| p[3] = u8::MAX);
        return std::borrow::Cow::Owned(rgba);
    }
    let mut out = RgbaImage::new(dst_w, dst_h);
    let (sw, sh) = (bg.width(), bg.height());
    if sw == 0 || sh == 0 || dst_w == 0 || dst_h == 0 {
        return std::borrow::Cow::Owned(out);
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
    std::borrow::Cow::Owned(out)
}

/// Fast-compression PNG streamed into `w`, without `encode_rgba_png`'s
/// intermediate buffer (5–30 MB at 4K). The caller flushes `w`.
pub fn encode_rgba_png_into<W: std::io::Write>(img: &RgbaImage, w: W) -> Result<(), CoreError> {
    write_fast_png(w, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
}

fn write_fast_png<W: std::io::Write>(w: W, bytes: &[u8], width: u32, height: u32, ty: image::ExtendedColorType) -> Result<(), CoreError> {
    PngEncoder::new_with_quality(w, CompressionType::Fast, PngFilter::Sub)
        .write_image(bytes, width, height, ty)
        .map_err(CoreError::from)
}

fn write_kept_png<W: std::io::Write>(w: W, bytes: &[u8], width: u32, height: u32, ty: image::ExtendedColorType) -> Result<(), CoreError> {
    PngEncoder::new_with_quality(w, CompressionType::Level(4), PngFilter::Adaptive)
        .write_image(bytes, width, height, ty)
        .map_err(CoreError::from)
}

/// `encode_rgba_png_into` as owned bytes.
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

    /// What follows the source stem in a result's file name: `.prunr.<ext>`.
    pub fn result_suffix(self) -> String {
        format!(".prunr.{}", self.extension())
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Webp => "WebP",
        }
    }

    /// The largest side, in pixels, the format stores: WebP's header has
    /// 14 bits for it, JPEG's 16.
    pub const fn max_side(self) -> u32 {
        match self {
            Self::Png => i32::MAX as u32,
            Self::Jpeg => 65_535,
            Self::Webp => 16_384,
        }
    }

    pub const fn fits(self, width: u32, height: u32) -> bool {
        width <= self.max_side() && height <= self.max_side()
    }

    /// Why a `width × height` image cannot be saved in this format, if it
    /// cannot.
    pub fn refusal(self, width: u32, height: u32) -> Option<String> {
        (!self.fits(width, height)).then(|| {
            format!("{} stores images up to {} px a side; this one is {width}×{height}. Save it as PNG.", self.label(), self.max_side())
        })
    }

    /// The format a file name asks for by its extension.
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Self::ALL.into_iter().find(|f| f.extensions().contains(&ext.as_str()))
    }

    /// `img` encoded for a file the user keeps, streamed into `w`. A PNG
    /// is zlib level 4 with adaptive filters: on a 20 MP cut-out fast is
    /// 0.2 s / 27.9 MB, level 4 1.6 s / 13.4 MB, default 4.9 s / 12.1 MB.
    /// Temp, drag-out and clipboard PNGs stay fast (`encode_rgba_png`).
    pub fn encode_into<W: std::io::Write>(self, img: &RgbaImage, w: W) -> Result<(), CoreError> {
        let (width, height) = img.dimensions();
        if let Some(why) = self.refusal(width, height) {
            return Err(CoreError::ImageFormat(why));
        }
        match self {
            Self::Png => return write_kept_png(w, img.as_raw(), width, height, image::ExtendedColorType::Rgba8),
            Self::Jpeg => {
                const JPEG_QUALITY: u8 = 92;
                let flat = flatten_on_white(img);
                image::codecs::jpeg::JpegEncoder::new_with_quality(w, JPEG_QUALITY)
                    .encode(flat.as_raw(), width, height, image::ExtendedColorType::Rgb8)
            }
            Self::Webp => image::codecs::webp::WebPEncoder::new_lossless(w)
                .encode(img.as_raw(), width, height, image::ExtendedColorType::Rgba8),
        }
        .map_err(CoreError::from)
    }

    /// `encode_into` as bytes, for a caller that holds them before writing.
    pub fn encode(self, img: &RgbaImage) -> Result<Vec<u8>, CoreError> {
        let mut buf = Vec::with_capacity(img.as_raw().len() / 2);
        self.encode_into(img, &mut buf)?;
        Ok(buf)
    }
}

/// Straight-alpha RGBA composited over white, as RGB (JPEG has no alpha).
fn flatten_on_white(img: &RgbaImage) -> image::RgbImage {
    let rgb: Vec<u8> = img.as_raw().chunks_exact(4).flat_map(|px| blend_over(px, [255; 3])).collect();
    // invariant: 3 bytes for each of width × height pixels.
    image::RgbImage::from_raw(img.width(), img.height(), rgb).expect("one RGB triple per RGBA pixel")
}

/// Encode a GrayImage (L8) as PNG with fast compression, streamed into `w`.
pub fn encode_gray_png_into<W: std::io::Write>(img: &GrayImage, w: W) -> Result<(), CoreError> {
    write_fast_png(w, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
}

/// A GrayImage (L8) PNG for a file the user keeps, at the compression
/// `ExportFormat::Png` uses, streamed into `w`.
pub fn encode_gray_png_kept_into<W: std::io::Write>(img: &GrayImage, w: W) -> Result<(), CoreError> {
    write_kept_png(w, img.as_raw(), img.width(), img.height(), image::ExtendedColorType::L8)
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

/// A photo the pipeline reads as straight RGBA: a decoded image in any
/// layout (converted when it is not RGBA), or RGBA already (borrowed).
pub trait AsRgba {
    fn as_rgba(&self) -> std::borrow::Cow<'_, RgbaImage>;
}

impl AsRgba for DynamicImage {
    fn as_rgba(&self) -> std::borrow::Cow<'_, RgbaImage> {
        rgba_view(self)
    }
}

impl AsRgba for RgbaImage {
    fn as_rgba(&self) -> std::borrow::Cow<'_, RgbaImage> {
        std::borrow::Cow::Borrowed(self)
    }
}

#[cfg(test)]
mod tests {

    /// An EXIF block that says "turn right" (orientation 6): what a phone
    /// writes for a portrait it stores sideways.
    fn turn_right_exif() -> Vec<u8> {
        let mut tiff = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x0112u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&6u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff
    }

    /// A photo stored sideways with an EXIF turn loads upright, at the
    /// size it shows, from bytes and from a file, PNG and JPEG alike.
    #[test]
    fn a_photo_loads_the_way_its_exif_turns_it() {
        use image::{GenericImageView, ImageEncoder, Rgb, RgbImage};
        let mut stored = RgbImage::new(16, 8);
        stored.put_pixel(0, 0, Rgb([255, 0, 0]));
        let mut png = Vec::new();
        let mut encoder = image::codecs::png::PngEncoder::new(&mut png);
        encoder.set_exif_metadata(turn_right_exif()).unwrap();
        encoder.write_image(stored.as_raw(), 16, 8, image::ExtendedColorType::Rgb8).unwrap();
        let mut jpeg = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut jpeg);
        encoder.set_exif_metadata(turn_right_exif()).unwrap();
        encoder.write_image(stored.as_raw(), 16, 8, image::ExtendedColorType::Rgb8).unwrap();

        let upright = load_image_from_bytes(&png).unwrap().to_rgb8();
        assert_eq!(upright.dimensions(), (8, 16));
        assert_eq!(*upright.get_pixel(7, 0), Rgb([255, 0, 0]), "the stored top-left turns to the top-right");
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in [("turned.png", &png), ("turned.jpg", &jpeg)] {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(load_image_from_bytes(bytes).unwrap().dimensions(), (8, 16), "{name} from bytes");
            assert_eq!(load_image_from_path(&path).unwrap().dimensions(), (8, 16), "{name} from a file");
            assert_eq!(image_dimensions_from_bytes(bytes), Some((8, 16)), "{name}'s size from bytes");
            assert_eq!(image_dimensions_from_path(&path), Some((8, 16)), "{name}'s size from a file");
        }
    }

    /// A PNG may put its EXIF after the image data (the PNG rules allow
    /// it; ExifTool before 12.29 wrote it there), where a decoder asked
    /// before the pixels has not read it yet: it still loads upright.
    #[test]
    fn a_png_with_its_exif_after_the_pixels_loads_upright() {
        use image::{GenericImageView, ImageEncoder, Rgb, RgbImage};
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = !0u32;
            for &b in bytes {
                crc ^= b as u32;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
                }
            }
            !crc
        }
        let mut stored = RgbImage::new(16, 8);
        stored.put_pixel(0, 0, Rgb([255, 0, 0]));
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png).write_image(stored.as_raw(), 16, 8, image::ExtendedColorType::Rgb8).unwrap();
        let exif = turn_right_exif();
        let mut chunk = (exif.len() as u32).to_be_bytes().to_vec();
        let typed: Vec<u8> = b"eXIf".iter().chain(&exif).copied().collect();
        chunk.extend_from_slice(&typed);
        chunk.extend_from_slice(&crc32(&typed).to_be_bytes());
        let iend = png.len() - 12;
        png.splice(iend..iend, chunk);

        let upright = load_image_from_bytes(&png).unwrap().to_rgb8();
        assert_eq!(upright.dimensions(), (8, 16));
        assert_eq!(*upright.get_pixel(7, 0), Rgb([255, 0, 0]));
        assert_eq!(image_dimensions_from_bytes(&png), Some((8, 16)), "its size from the header");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("late-exif.png");
        std::fs::write(&path, &png).unwrap();
        assert_eq!(load_image_from_path(&path).unwrap().dimensions(), (8, 16), "from a file");
        assert_eq!(image_dimensions_from_path(&path), Some((8, 16)));
    }

    /// A background already the frame's size is used as it is: the same
    /// layer the resample path made (a same-size Lanczos pass is a copy,
    /// and Cover / Contain go opaque), borrowed when it is opaque already.
    #[test]
    fn a_background_at_the_frames_size_is_used_as_it_is() {
        let resampled = |bg: &DynamicImage| -> RgbaImage { resize_rgb_lanczos3(bg, 97, 61).convert() };
        let opaque = DynamicImage::ImageRgba8(RgbaImage::from_fn(97, 61, |x, y| image::Rgba([(x * 37 % 256) as u8, (y * 91 % 256) as u8, ((x * y) % 256) as u8, 255])));
        let seen = DynamicImage::ImageRgba8(RgbaImage::from_fn(97, 61, |x, y| image::Rgba([(x * 5) as u8, y as u8, 9, (x * 2) as u8])));
        for fit in [crate::types::BgImageFit::Cover, crate::types::BgImageFit::Contain] {
            let layer = build_bg_layer(&opaque, 97, 61, fit);
            assert!(matches!(layer, std::borrow::Cow::Borrowed(_)) && *layer == resampled(&opaque));
            assert!(*build_bg_layer(&seen, 97, 61, fit) == resampled(&seen), "its alpha goes opaque as before");
        }
    }

    /// A PNG the user saves reads back as the same pixels, smaller than
    /// the fast one.
    #[test]
    fn a_saved_png_is_lossless_and_smaller_than_the_fast_one() {
        let img = RgbaImage::from_fn(300, 200, |x, y| image::Rgba([(x / 3) as u8, (y / 2) as u8, 120, if x < 150 { 255 } else { 0 }]));
        let saved = ExportFormat::Png.encode(&img).unwrap();
        assert_eq!(image::load_from_memory(&saved).unwrap().into_rgba8(), img);
        assert!(saved.len() < encode_rgba_png(&img).unwrap().len());
    }

    /// An opaque image resizes without the alpha weighting to the very
    /// pixels it gives with it.
    #[test]
    fn an_opaque_resize_skips_the_alpha_weighting_with_the_same_result() {
        let rgba = RgbaImage::from_fn(97, 61, |x, y| image::Rgba([(x * 3) as u8, (y * 4) as u8, (x ^ y) as u8, 255]));
        for (dst, filter) in [((40, 30), ResizeFilter::Lanczos3), ((200, 150), ResizeFilter::Bilinear)] {
            let weighed = resize_raw_with(rgba.as_raw(), 97, 61, PixelType::U8x4, dst, filter, true);
            assert_eq!(resize_rgba(&rgba, dst.0, dst.1, filter).as_raw(), &weighed);
        }
    }

    /// Padding reflects the image at its right and bottom edges, folding
    /// back again when the pad is longer than the image.
    #[test]
    fn padding_mirrors_the_image_past_its_edges() {
        let img = GrayImage::from_fn(3, 2, |x, y| image::Luma([(y * 3 + x) as u8]));
        let padded = pad_mirrored(&img, 7, 4);
        let rows: Vec<Vec<u8>> = padded.rows().map(|r| r.map(|p| p[0]).collect()).collect();
        assert_eq!(rows, vec![vec![0, 1, 2, 1, 0, 1, 2], vec![3, 4, 5, 4, 3, 4, 5], vec![0, 1, 2, 1, 0, 1, 2], vec![3, 4, 5, 4, 3, 4, 5]]);
        assert!(matches!(pad_mirrored(&img, 3, 2), std::borrow::Cow::Borrowed(_)));
        assert_eq!(pad_mirrored(&GrayImage::from_pixel(1, 1, image::Luma([9])), 2, 2).as_raw(), &vec![9; 4]);
    }

    /// An RGB8 image resizes in place to the same pixels an RGBA copy of
    /// it gives.
    #[test]
    fn rgb_resize_reads_rgb8_in_place_with_the_same_result() {
        let rgb = image::RgbImage::from_fn(97, 61, |x, y| image::Rgb([(x * 3) as u8, (y * 4) as u8, (x ^ y) as u8]));
        let rgba = DynamicImage::ImageRgb8(rgb.clone()).to_rgba8();
        let direct = resize_rgb_lanczos3(&DynamicImage::ImageRgb8(rgb), 40, 30);
        assert!(direct == resize_rgb_lanczos3(&DynamicImage::ImageRgba8(rgba), 40, 30));
    }

    /// Every size in, every size out, an empty one included: resizing
    /// never panics. A zero side gave the resizer an empty image.
    #[test]
    fn a_resize_to_or_from_an_empty_size_does_not_panic() {
        for (src, dst) in [((4, 4), (0, 0)), ((4, 4), (0, 3)), ((4, 4), (3, 0)), ((0, 0), (3, 3)), ((0, 4), (0, 0)), ((4, 4), (1, 1))] {
            let rgba = RgbaImage::new(src.0, src.1);
            let out = resize_rgba(&rgba, dst.0, dst.1, ResizeFilter::Lanczos3);
            assert_eq!(out.dimensions(), dst, "{src:?} -> {dst:?}");
            let gray = resize_gray_lanczos3(&GrayImage::new(src.0, src.1), dst.0, dst.1);
            assert_eq!(gray.dimensions(), dst);
            let rgb = resize_rgb_image_lanczos3(&image::RgbImage::new(src.0, src.1), dst.0, dst.1);
            assert_eq!(rgb.dimensions(), dst);
        }
    }

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

    /// An SVG's size is read as it would render, matching the decode; a
    /// raster's from its header; anything else is no image.
    #[test]
    fn image_dimensions_read_svgs_and_rasters_alike() {
        assert_eq!(image_dimensions_from_bytes(SAMPLE_SVG), Some((32, 16)));
        let decoded = load_image_from_bytes(SAMPLE_SVG).unwrap();
        assert_eq!(image_dimensions_from_bytes(SAMPLE_SVG), Some((decoded.width(), decoded.height())));
        let huge = br#"<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="50000"></svg>"#;
        assert_eq!(image_dimensions_from_bytes(huge), Some((LARGE_IMAGE_LIMIT, LARGE_IMAGE_LIMIT / 2)));
        let png = encode_rgba_png(&RgbaImage::new(5, 3)).unwrap();
        assert_eq!(image_dimensions_from_bytes(&png), Some((5, 3)));
        assert_eq!(image_dimensions_from_bytes(b"not an image"), None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logo.svg");
        std::fs::write(&path, SAMPLE_SVG).unwrap();
        assert_eq!(image_dimensions_from_path(&path), Some((32, 16)));
    }

    /// WebP stores at most 16384 px a side: a larger image is refused
    /// before encoding, saying what fits, not with the encoder's error.
    #[test]
    fn a_format_refuses_an_image_past_its_largest_side_with_the_reason() {
        assert!(ExportFormat::Webp.encode(&RgbaImage::new(16384, 1)).is_ok());
        let refused = ExportFormat::Webp.encode(&RgbaImage::new(16385, 1)).unwrap_err().to_string();
        assert!(refused.contains("16384") && refused.contains("PNG"), "{refused}");
        assert!(ExportFormat::Png.fits(16385, 16385) && !ExportFormat::Webp.fits(1, 16385));
        assert!(ExportFormat::Jpeg.fits(65_535, 1) && !ExportFormat::Jpeg.fits(65_536, 1));
    }

    #[test]
    fn looks_like_svg_detects_xml_prologue_and_not_rasters() {
        assert!(looks_like_svg(SAMPLE_SVG));
        assert!(looks_like_svg(b"<svg xmlns='...'><circle/></svg>"));
        assert!(looks_like_svg(b"\xEF\xBB\xBF<svg/>"), "BOM-prefixed SVG");
        assert!(looks_like_svg(b"  \n\t<svg/>"), "leading whitespace");
        assert!(!looks_like_svg(&minimal_png_bytes()));
        assert!(!looks_like_svg(b"\xff\xd8\xff\xe0JFIF"), "JPEG header");
        assert!(!looks_like_svg(b"GIF89a"), "GIF header");
        assert!(!looks_like_svg(b""), "empty buffer");
    }

    /// The first 512 bytes are sniffed: a character cut at that edge, or
    /// a Latin-1 comment, is no reason to refuse the file.
    #[test]
    fn an_svg_is_known_by_its_first_bytes_whatever_follows() {
        let mut cut = b"<?xml version=\"1.0\"?>\n<!-- ".to_vec();
        cut.resize(511, b'x');
        cut.extend_from_slice("é -->\n<svg/>".as_bytes());
        assert!(looks_like_svg(&cut), "a character cut at byte 512");
        assert!(looks_like_svg(b"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><!-- \xa9 2024 --><svg/>"), "Latin-1");
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
        // A thin but wide image past 8000 px.
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(9000, 100));
        match check_large_image(&img) {
            Some(CoreError::LargeImage { width, height: _, limit }) => {
                assert_eq!(width, 9000);
                assert_eq!(limit, LARGE_IMAGE_LIMIT);
            }
            other => panic!("Expected LargeImage, got {:?}", other),
        }
        assert!(check_large_image(&DynamicImage::ImageRgb8(image::RgbImage::new(800, 600))).is_none());
    }

    #[test]
    fn exceeds_large_image_limit_is_strictly_past_either_side() {
        let l = LARGE_IMAGE_LIMIT;
        assert!(!exceeds_large_image_limit(l, l));
        assert!(exceeds_large_image_limit(l + 1, 1));
        assert!(exceeds_large_image_limit(1, l + 1));
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
    fn test_downscale_image_no_upscale() {
        // Images already within limit should be returned as-is
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        let scaled = downscale_image(img, DOWNSCALE_TARGET);
        assert_eq!(scaled.width(), 800);
        assert_eq!(scaled.height(), 600);
    }

    #[test]
    fn test_encode_gray_png_round_trip() {
        // Gradient 0..255 across an 8x1 strip — round-tripping through PNG
        // must preserve every value.
        let mut img = GrayImage::new(8, 1);
        for x in 0..8 {
            img.put_pixel(x, 0, image::Luma([(x * 32) as u8]));
        }
        let mut bytes = Vec::new();
        encode_gray_png_into(&img, &mut bytes).expect("gray encode");
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);
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

    #[test]
    fn a_result_suffix_tags_prunr_before_the_extension() {
        assert_eq!(ExportFormat::Png.result_suffix(), ".prunr.png");
        assert_eq!(ExportFormat::Jpeg.result_suffix(), ".prunr.jpg");
        assert_eq!(ExportFormat::Webp.result_suffix(), ".prunr.webp");
    }
}
