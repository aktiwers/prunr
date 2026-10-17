//! A full-size image on the GPU, as tiles. egui uploads a texture whole
//! in the frame that creates it (a 12 MP image is 48 MB, an x4 upscale
//! 768 MB) and a GPU caps a texture's side (8192 px on many), so a large
//! image is cut into tiles on the thread that builds it and uploaded a
//! few tiles per frame; it shows once every tile is up.

use std::collections::VecDeque;
use std::sync::Arc;

use egui::{Color32, Rect, TextureHandle, TextureId, TextureOptions};

/// The longest side of a tile's texture, overlap included: within every
/// GPU's texture limit (and egui's default), and 16 MB, a few ms to upload.
pub(crate) const TILE_SIDE: usize = 2048;

/// The part of the image a tile covers: its texture less a pixel of each
/// neighbour on either side.
const TILE_STEP: usize = TILE_SIDE - 2;

/// Bytes of tiles uploaded per frame; a frame always takes at least one.
const UPLOAD_BUDGET: usize = 2 * TILE_SIDE * TILE_SIDE * 4;

/// An image on the GPU: its tiles, each with the part of the image it
/// covers in 0..1 coordinates.
#[derive(Clone)]
pub(crate) struct ImageTexture {
    size: [usize; 2],
    tiles: Arc<[Tile<TextureHandle>]>,
}

/// One tile: the part of the image it covers (0..1), and the part of its
/// texture that shows it. A tile holds a pixel of each neighbour around
/// its part, so smooth sampling at its edge blends across the seam as it
/// would inside one texture; only its own part is drawn.
#[derive(Clone)]
pub(crate) struct Tile<T> {
    pub(crate) part: Rect,
    pub(crate) texture_uv: Rect,
    pub(crate) texture: T,
}

impl ImageTexture {
    /// A texture uploaded whole: a small image.
    pub(crate) fn single(handle: TextureHandle) -> Self {
        let unit = crate::gui::views::effects::UNIT_UV;
        Self { size: handle.size(), tiles: Arc::from([Tile { part: unit, texture_uv: unit, texture: handle }]) }
    }

    pub(crate) fn size_vec2(&self) -> egui::Vec2 {
        egui::vec2(self.size[0] as f32, self.size[1] as f32)
    }

    pub(crate) fn byte_size(&self) -> usize {
        self.tiles.iter().map(|t| t.texture.byte_size()).sum()
    }

    /// The first tile's id: which upload this is, for tests.
    #[cfg(test)]
    pub(crate) fn id(&self) -> TextureId {
        self.tiles[0].texture.id()
    }


    pub(crate) fn tiles(&self) -> impl Iterator<Item = Tile<TextureId>> + '_ {
        self.tiles.iter().map(|t| Tile { part: t.part, texture_uv: t.texture_uv, texture: t.texture.id() })
    }

    /// The image in `rect`, tinted.
    pub(crate) fn paint(&self, painter: &egui::Painter, rect: Rect, tint: Color32) {
        for tile in self.tiles() {
            painter.image(tile.texture, part_of(rect, tile.part), tile.texture_uv, tint);
        }
    }
}

/// The part `part` (0..1) of `rect`.
pub(crate) fn part_of(rect: Rect, part: Rect) -> Rect {
    Rect::from_min_max(rect.lerp_inside(part.min.to_vec2()), rect.lerp_inside(part.max.to_vec2()))
}

impl From<TextureHandle> for ImageTexture {
    fn from(handle: TextureHandle) -> Self {
        Self::single(handle)
    }
}

/// An image cut into tiles ready to upload, made off the UI thread.
pub(crate) struct PreparedImage {
    size: [usize; 2],
    tiles: Vec<Tile<egui::ColorImage>>,
}

impl PreparedImage {
    pub(crate) fn from_rgba(rgba: &image::RgbaImage) -> Self {
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        let raw = rgba.as_raw();
        let mut tiles = Vec::new();
        // An image that fits one texture is one tile, without overlap.
        let step = |len: usize| if len <= TILE_SIDE { TILE_SIDE } else { TILE_STEP };
        let (sx, sy) = (step(w), step(h));
        for y0 in (0..h.max(1)).step_by(sy) {
            for x0 in (0..w.max(1)).step_by(sx) {
                let (tw, th) = (sx.min(w - x0), sy.min(h - y0));
                // The texture reaches a pixel into each neighbour.
                let (gx0, gy0) = (x0.saturating_sub(1), y0.saturating_sub(1));
                let (gx1, gy1) = ((x0 + tw + 1).min(w), (y0 + th + 1).min(h));
                let (gw, gh) = (gx1 - gx0, gy1 - gy0);
                let texture = if (gw, gh) == (w, h) {
                    egui::ColorImage::from_rgba_unmultiplied([w, h], raw)
                } else {
                    let mut pixels = Vec::with_capacity(gw * gh * 4);
                    for row in gy0..gy1 {
                        let start = (row * w + gx0) * 4;
                        pixels.extend_from_slice(&raw[start..start + gw * 4]);
                    }
                    egui::ColorImage::from_rgba_unmultiplied([gw, gh], &pixels)
                };
                let part = Rect::from_min_max(
                    egui::pos2(x0 as f32 / w as f32, y0 as f32 / h as f32),
                    egui::pos2((x0 + tw) as f32 / w as f32, (y0 + th) as f32 / h as f32),
                );
                let texture_uv = Rect::from_min_max(
                    egui::pos2((x0 - gx0) as f32 / gw as f32, (y0 - gy0) as f32 / gh as f32),
                    egui::pos2((x0 + tw - gx0) as f32 / gw as f32, (y0 + th - gy0) as f32 / gh as f32),
                );
                tiles.push(Tile { part, texture_uv, texture });
            }
        }
        Self { size: [w, h], tiles }
    }

}

/// Prepared images waiting for their tiles to upload, oldest first, each
/// with what its landing needs (`T`).
pub(crate) struct Uploads<T> {
    queue: VecDeque<Upload<T>>,
    budget: usize,
}

struct Upload<T> {
    name: String,
    left: std::vec::IntoIter<Tile<egui::ColorImage>>,
    size: [usize; 2],
    done: Vec<Tile<TextureHandle>>,
    landing: T,
}

impl<T> Default for Uploads<T> {
    fn default() -> Self {
        Self { queue: VecDeque::new(), budget: UPLOAD_BUDGET }
    }
}

impl<T> Uploads<T> {
    pub(crate) fn push(&mut self, name: String, image: PreparedImage, landing: T) {
        let size = image.size;
        self.queue.push_back(Upload { name, left: image.tiles.into_iter(), size, done: Vec::new(), landing });
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Upload this frame's tiles; the images whose last tile went up come
    /// back, in the order they were pushed.
    pub(crate) fn pump(&mut self, ctx: &egui::Context, options: TextureOptions) -> Vec<(ImageTexture, T)> {
        let mut spent = 0;
        let mut finished = Vec::new();
        while let Some(upload) = self.queue.front_mut() {
            while spent == 0 || spent < self.budget {
                let Some(tile) = upload.left.next() else { break };
                spent += tile.texture.pixels.len() * 4;
                let name = format!("{}#{}", upload.name, upload.done.len());
                let texture = ctx.load_texture(name, tile.texture, options);
                upload.done.push(Tile { part: tile.part, texture_uv: tile.texture_uv, texture });
            }
            if upload.left.len() > 0 {
                break;
            }
            // just checked: the front upload has no tiles left
            let Some(upload) = self.queue.pop_front() else { break };
            finished.push((ImageTexture { size: upload.size, tiles: upload.done.into() }, upload.landing));
            if spent >= self.budget {
                break;
            }
        }
        if !self.queue.is_empty() {
            ctx.request_repaint();
        }
        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A large image is cut into tiles that cover it exactly, each within
    /// the tile side; a small one is one tile.
    #[test]
    fn a_large_image_is_cut_into_tiles_that_cover_it() {
        let rgba = image::RgbaImage::from_fn((TILE_STEP + 5) as u32, 3, |x, y| image::Rgba([x as u8, y as u8, 7, 255]));
        let prepared = PreparedImage::from_rgba(&rgba);
        assert_eq!(prepared.tiles.len(), 2);
        let (first, second) = (&prepared.tiles[0], &prepared.tiles[1]);
        assert_eq!((first.texture.size, second.texture.size), ([TILE_STEP + 1, 3], [6, 3]), "each reaches a pixel into its neighbour");
        assert_eq!(first.part.max.x, second.part.min.x, "the parts meet");
        assert_eq!((first.part.min, second.part.max), (egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), "and cover the image");
        assert_eq!(second.texture.pixels[0], egui::Color32::from_rgba_unmultiplied((TILE_STEP - 1) as u8, 0, 7, 255), "the second's texture starts a pixel inside the first");
        assert_eq!(second.texture_uv.min.x, 1.0 / 6.0, "and draws from its own first pixel");
        assert_eq!(first.texture_uv, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(TILE_STEP as f32 / (TILE_STEP + 1) as f32, 1.0)));
        assert!(prepared.tiles.iter().all(|t| t.texture.size[0] <= TILE_SIDE), "no texture past the limit");
        assert_eq!(PreparedImage::from_rgba(&image::RgbaImage::new(4, 4)).tiles.len(), 1);
    }

    /// A frame uploads within its budget (at least one tile); an image
    /// lands once its last tile is up, in the order pushed.
    #[test]
    fn tiles_upload_a_few_per_frame_and_land_whole() {
        let ctx = egui::Context::default();
        let big = PreparedImage::from_rgba(&image::RgbaImage::new((3 * TILE_STEP) as u32, 1));
        let small = PreparedImage::from_rgba(&image::RgbaImage::new(2, 2));
        let mut uploads = Uploads { budget: 2 * TILE_SIDE * 4, ..Uploads::default() };
        uploads.push("big".into(), big, 1);
        uploads.push("small".into(), small, 2);
        let mut landed = Vec::new();
        let mut frames = 0;
        while !uploads.is_empty() {
            frames += 1;
            landed.extend(uploads.pump(&ctx, TextureOptions::default()).into_iter().map(|(t, n)| (n, t.tiles.len(), t.size)));
            assert!(frames < 10);
        }
        assert!(frames > 1, "not all in one frame");
        assert_eq!(landed, vec![(1, 3, [3 * TILE_STEP, 1]), (2, 1, [2, 2])]);
    }
}
