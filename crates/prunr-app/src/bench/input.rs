//! The fixed input every model runs on: a built-in synthetic photo, or
//! the user's `--image`, cut to the size each kind of model sees.

use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use prunr_core::formats;

/// Side of the built-in photo.
pub const PHOTO_SIDE: u32 = 512;
/// Longest side the background-removal, colorize, depth and eraser runs
/// get; a larger `--image` is shrunk to it.
pub const MAX_SIDE: u32 = 1024;
/// Side of the upscalers' input: one tile, so a run times the model and
/// not the tiling around it.
pub const UPSCALE_SIDE: u32 = 256;

/// What the models run on.
pub struct Inputs {
    /// For background removal, colorize, depth and the erasers.
    pub photo: RgbaImage,
    /// The erasers' region: a square in the middle of `photo`.
    pub mask: GrayImage,
    /// For the upscalers: the middle of `photo`.
    pub tile: RgbaImage,
    /// Says in the report what the input was.
    pub label: String,
}

impl Inputs {
    pub fn built_in() -> Self {
        Self::from_photo(DynamicImage::ImageRgba8(synthetic_photo()), format!("built-in {PHOTO_SIDE}×{PHOTO_SIDE} picture"))
    }

    /// `photo` shrunk to `MAX_SIDE` (resampled in its own layout, so a
    /// large JPEG never expands to RGBA at full size), with `label` naming
    /// where it came from.
    pub fn from_photo(photo: DynamicImage, label: String) -> Self {
        let (w, h) = (photo.width(), photo.height());
        let photo = if w.max(h) > MAX_SIDE {
            let (w, h) = formats::fit_long_side(w, h, MAX_SIDE);
            formats::resize_to_rgba(&photo, w, h, formats::ResizeFilter::Lanczos3)
        } else {
            photo.into_rgba8()
        };
        let (w, h) = photo.dimensions();
        let tile = center_crop(&photo, UPSCALE_SIDE);
        let label = format!("{label}, run at {w}×{h}; upscalers get its middle {}×{}", tile.width(), tile.height());
        Self { mask: center_square_mask(w, h), tile, photo, label }
    }
}

/// A sky-to-ground gradient with fine texture, and a shaded figure (body
/// and head) in the middle: a mask has an edge to find, an upscaler
/// detail to draw, an eraser a surround to continue. The same pixels on
/// every machine, so reports compare.
pub fn synthetic_photo() -> RgbaImage {
    let side = PHOTO_SIDE as f32;
    RgbaImage::from_fn(PHOTO_SIDE, PHOTO_SIDE, |x, y| {
        let (u, v) = (x as f32 / side, y as f32 / side);
        let texture = 10.0 * (x as f32 * 0.35).sin() * (y as f32 * 0.27).sin();
        let sky = [110.0, 160.0, 225.0];
        let ground = [70.0, 125.0, 60.0];
        let mut rgb: [f32; 3] = std::array::from_fn(|c| sky[c] + (ground[c] - sky[c]) * v + texture);
        let body = ((u - 0.5) / 0.18).powi(2) + ((v - 0.68) / 0.26).powi(2);
        let head = ((u - 0.5) / 0.11).powi(2) + ((v - 0.33) / 0.11).powi(2);
        if body < 1.0 {
            let shade = 1.0 - 0.35 * body;
            let stripe = if (y / 8) % 2 == 0 { 18.0 } else { 0.0 };
            rgb = [(200.0 + stripe) * shade, (90.0 + stripe) * shade, 60.0 * shade];
        } else if head < 1.0 {
            let shade = 1.0 - 0.3 * head;
            rgb = [230.0 * shade, 190.0 * shade, 160.0 * shade];
        }
        let [r, g, b] = rgb.map(prunr_core::math::round_u8);
        Rgba([r, g, b, 255])
    })
}

/// A square a quarter of the shorter side across, in the middle: 255
/// inside, 0 outside.
pub fn center_square_mask(w: u32, h: u32) -> GrayImage {
    let side = (w.min(h) / 4).max(1);
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    GrayImage::from_fn(w, h, |x, y| {
        let inside = (x0..x0 + side).contains(&x) && (y0..y0 + side).contains(&y);
        Luma([if inside { 255 } else { 0 }])
    })
}

/// The middle `side` × `side` of `img`, or as much of it as there is.
pub fn center_crop(img: &RgbaImage, side: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let (cw, ch) = (side.min(w), side.min(h));
    image::imageops::crop_imm(img, (w - cw) / 2, (h - ch) / 2, cw, ch).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in picture has a subject in the middle and background in
    /// the corners, and is the same every time.
    #[test]
    fn the_built_in_photo_has_a_figure_on_a_background() {
        let photo = synthetic_photo();
        assert_eq!(photo.dimensions(), (PHOTO_SIDE, PHOTO_SIDE));
        assert_eq!(photo, synthetic_photo());
        let body = photo.get_pixel(PHOTO_SIDE / 2, PHOTO_SIDE * 68 / 100).0;
        let corner = photo.get_pixel(4, 4).0;
        assert!(body[0] > body[2] + 60, "warm figure: {body:?}");
        assert!(corner[2] > corner[0] + 60, "blue sky: {corner:?}");
        assert!(photo.pixels().all(|p| p.0[3] == 255));
    }

    #[test]
    fn the_eraser_mask_is_a_centred_square_a_quarter_side_across() {
        let mask = center_square_mask(512, 400);
        let inside = mask.pixels().filter(|p| p.0[0] == 255).count();
        assert_eq!(inside, 100 * 100);
        assert_eq!(mask.get_pixel(256, 200).0, [255]);
        assert_eq!(mask.get_pixel(206, 150).0, [255]);
        assert_eq!(mask.get_pixel(205, 150).0, [0]);
        assert_eq!(mask.get_pixel(0, 0).0, [0]);
    }

    /// A large `--image` is shrunk to `MAX_SIDE` with its aspect kept;
    /// the upscalers get its middle, at most one tile.
    #[test]
    fn a_large_image_is_shrunk_and_the_upscalers_get_its_middle() {
        let big = RgbaImage::from_pixel(3000, 1500, Rgba([1, 2, 3, 255]));
        let inputs = Inputs::from_photo(DynamicImage::ImageRgba8(big), "photo.jpg".into());
        assert_eq!(inputs.photo.dimensions(), (MAX_SIDE, 512));
        assert_eq!(inputs.mask.dimensions(), inputs.photo.dimensions());
        assert_eq!(inputs.tile.dimensions(), (UPSCALE_SIDE, UPSCALE_SIDE));
        assert!(inputs.label.starts_with("photo.jpg, run at 1024×512"), "{}", inputs.label);
        let small = Inputs::from_photo(DynamicImage::ImageRgb8(image::RgbImage::new(100, 60)), "small.png".into());
        assert_eq!((small.photo.dimensions(), small.tile.dimensions()), ((100, 60), (100, 60)));
    }
}
