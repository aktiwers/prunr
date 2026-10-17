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
    /// One texture that wraps around: a repeating region is one quad.
    repeats: bool,
}

/// One tile: the part of the image it covers (0..1), and the part of its
/// texture that shows it. A tile holds a pixel of each neighbour around
/// its part, so smooth sampling at its edge blends across the seam as it
/// would inside one texture; only its own part is drawn.
#[derive(Clone)]
pub(crate) struct Tile<T> {
    pub(crate) part: Rect,
    pub(crate) texture_uv: Rect,
    /// Where the texture's first pixel is in the image.
    pub(crate) origin: [usize; 2],
    pub(crate) texture: T,
}

impl ImageTexture {
    /// A texture uploaded whole: a small image.
    pub(crate) fn single(handle: TextureHandle) -> Self {
        let unit = crate::gui::views::effects::UNIT_UV;
        Self { size: handle.size(), tiles: Arc::from([Tile { part: unit, texture_uv: unit, origin: [0, 0], texture: handle }]), repeats: false }
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
        self.tiles.iter().map(|t| Tile { part: t.part, texture_uv: t.texture_uv, origin: t.origin, texture: t.texture.id() })
    }

    /// The image in `rect`, tinted.
    pub(crate) fn paint(&self, painter: &egui::Painter, rect: Rect, tint: Color32) {
        self.paint_region(painter, rect, crate::gui::views::effects::UNIT_UV, tint);
    }

    /// `region` of the image (0..1; past 1 it repeats the image) in
    /// `rect`, tinted.
    pub(crate) fn paint_region(&self, painter: &egui::Painter, rect: Rect, region: Rect, tint: Color32) {
        for (screen, texture, uv) in self.region_pieces(rect, region) {
            painter.image(texture, screen, uv, tint);
        }
    }

    /// What `paint_region` draws: each piece's screen rect, texture and
    /// texture coordinates.
    fn region_pieces(&self, rect: Rect, region: Rect) -> Vec<(Rect, TextureId, Rect)> {
        let mut pieces = Vec::new();
        if region.width() <= 0.0 || region.height() <= 0.0 {
            return pieces;
        }
        if self.repeats {
            pieces.push((rect, self.tiles[0].texture.id(), region));
            return pieces;
        }
        let to_screen = |p: egui::Pos2| rect.lerp_inside(egui::vec2((p.x - region.min.x) / region.width(), (p.y - region.min.y) / region.height()));
        // Each repeat of the image the region reaches, then each tile in it.
        for cy in region.min.y.floor() as i32..region.max.y.ceil() as i32 {
            for cx in region.min.x.floor() as i32..region.max.x.ceil() as i32 {
                let offset = egui::vec2(cx as f32, cy as f32);
                let cell = region.intersect(Rect::from_min_size(offset.to_pos2(), egui::vec2(1.0, 1.0)));
                for tile in self.tiles() {
                    let shown = cell.intersect(tile.part.translate(offset));
                    if shown.width() <= 0.0 || shown.height() <= 0.0 {
                        continue;
                    }
                    let in_tile = |p: egui::Pos2| {
                        let local = p - offset;
                        tile.texture_uv.lerp_inside(egui::vec2(
                            (local.x - tile.part.min.x) / tile.part.width(),
                            (local.y - tile.part.min.y) / tile.part.height(),
                        ))
                    };
                    let screen = Rect::from_min_max(to_screen(shown.min), to_screen(shown.max));
                    pieces.push((screen, tile.texture, Rect::from_min_max(in_tile(shown.min), in_tile(shown.max))));
                }
            }
        }
        pieces
    }

    /// Write `patch`, whose first pixel is at `pos` in the image, into
    /// the tiles it overlaps.
    pub(crate) fn set_partial(&self, pos: [usize; 2], patch: &egui::ColorImage, options: TextureOptions) {
        let [pw, ph] = patch.size;
        for tile in self.tiles.iter() {
            let [tw, th] = tile.texture.size();
            let x0 = pos[0].max(tile.origin[0]);
            let y0 = pos[1].max(tile.origin[1]);
            let x1 = (pos[0] + pw).min(tile.origin[0] + tw);
            let y1 = (pos[1] + ph).min(tile.origin[1] + th);
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            let (w, h) = (x1 - x0, y1 - y0);
            let mut pixels = Vec::with_capacity(w * h);
            for row in y0..y1 {
                let start = (row - pos[1]) * pw + (x0 - pos[0]);
                pixels.extend_from_slice(&patch.pixels[start..start + w]);
            }
            let local = [x0 - tile.origin[0], y0 - tile.origin[1]];
            // A handle's clone writes the same texture.
            tile.texture.clone().set_partial(local, egui::ColorImage::new([w, h], pixels), options);
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
        Self::cut(w, h, |[x0, y0], [gw, gh]| {
            if (gw, gh) == (w, h) {
                return egui::ColorImage::from_rgba_unmultiplied([w, h], raw);
            }
            let mut pixels = Vec::with_capacity(gw * gh * 4);
            for row in y0..y0 + gh {
                let start = (row * w + x0) * 4;
                pixels.extend_from_slice(&raw[start..start + gw * 4]);
            }
            egui::ColorImage::from_rgba_unmultiplied([gw, gh], &pixels)
        })
    }

    /// The tiles of an image already in egui's pixels.
    pub(crate) fn from_color_image(image: egui::ColorImage) -> Self {
        let [w, h] = image.size;
        if w <= TILE_SIDE && h <= TILE_SIDE {
            let unit = crate::gui::views::effects::UNIT_UV;
            return Self { size: [w, h], tiles: vec![Tile { part: unit, texture_uv: unit, origin: [0, 0], texture: image }] };
        }
        Self::cut(w, h, |[x0, y0], [gw, gh]| {
            let mut pixels = Vec::with_capacity(gw * gh);
            for row in y0..y0 + gh {
                let start = row * w + x0;
                pixels.extend_from_slice(&image.pixels[start..start + gw]);
            }
            egui::ColorImage::new([gw, gh], pixels)
        })
    }

    /// Cut a `w`×`h` image into tiles; `texture(origin, size)` makes the
    /// pixels of the texture starting at `origin`.
    fn cut(w: usize, h: usize, texture: impl Fn([usize; 2], [usize; 2]) -> egui::ColorImage) -> Self {
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
                let texture = texture([gx0, gy0], [gw, gh]);
                let part = Rect::from_min_max(
                    egui::pos2(x0 as f32 / w as f32, y0 as f32 / h as f32),
                    egui::pos2((x0 + tw) as f32 / w as f32, (y0 + th) as f32 / h as f32),
                );
                let texture_uv = Rect::from_min_max(
                    egui::pos2((x0 - gx0) as f32 / gw as f32, (y0 - gy0) as f32 / gh as f32),
                    egui::pos2((x0 + tw - gx0) as f32 / gw as f32, (y0 + th - gy0) as f32 / gh as f32),
                );
                tiles.push(Tile { part, texture_uv, origin: [gx0, gy0], texture });
            }
        }
        Self { size: [w, h], tiles }
    }

}

/// One frame's uploads: the images whose last tile went up, and those
/// dropped as no longer wanted.
pub(crate) struct Pumped<T> {
    pub(crate) landed: Vec<(ImageTexture, T)>,
    pub(crate) dropped: Vec<T>,
}

/// Prepared images waiting for their tiles to upload, oldest first, each
/// with what its landing needs (`T`).
pub(crate) struct Uploads<T> {
    queue: VecDeque<Upload<T>>,
    budget: usize,
}

struct Upload<T> {
    name: String,
    options: TextureOptions,
    /// One texture uploaded to wrap around.
    repeats: bool,
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
        self.push_with(name, image, landing, TextureOptions::LINEAR);
    }

    pub(crate) fn push_with(&mut self, name: String, image: PreparedImage, landing: T, options: TextureOptions) {
        let size = image.size;
        let repeats = options.wrap_mode == egui::TextureWrapMode::Repeat && image.tiles.len() == 1;
        self.queue.push_back(Upload { name, options, repeats, left: image.tiles.into_iter(), size, done: Vec::new(), landing });
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Upload this frame's tiles; the images whose last tile went up come
    /// back, in the order they were pushed. Those no longer `wanted` (the
    /// image off screen, a newer one asked for) are dropped first, tiles
    /// and all, and come back in `dropped` so their requests can end.
    pub(crate) fn pump(&mut self, ctx: &egui::Context, wanted: impl Fn(&T) -> bool) -> Pumped<T> {
        let mut dropped = Vec::new();
        for upload in std::mem::take(&mut self.queue) {
            if wanted(&upload.landing) {
                self.queue.push_back(upload);
            } else {
                dropped.push(upload.landing);
            }
        }
        let mut spent = 0;
        let mut finished = Vec::new();
        while let Some(upload) = self.queue.front_mut() {
            while spent == 0 || spent < self.budget {
                let Some(tile) = upload.left.next() else { break };
                spent += tile.texture.pixels.len() * 4;
                let name = format!("{}#{}", upload.name, upload.done.len());
                let texture = ctx.load_texture(name, tile.texture, upload.options);
                upload.done.push(Tile { part: tile.part, texture_uv: tile.texture_uv, origin: tile.origin, texture });
            }
            if upload.left.len() > 0 {
                break;
            }
            // just checked: the front upload has no tiles left
            let Some(upload) = self.queue.pop_front() else { break };
            finished.push((ImageTexture { size: upload.size, tiles: upload.done.into(), repeats: upload.repeats }, upload.landing));
            if spent >= self.budget {
                break;
            }
        }
        if !self.queue.is_empty() {
            ctx.request_repaint();
        }
        Pumped { landed: finished, dropped }
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

    /// A patch is written into each tile it overlaps, at the tile's own
    /// coordinates, overlap pixels included.
    #[test]
    fn a_patch_lands_in_every_tile_it_overlaps() {
        let ctx = egui::Context::default();
        let w = TILE_STEP + 4;
        let mut uploads = Uploads::default();
        uploads.push("sel".into(), PreparedImage::from_color_image(egui::ColorImage::new([w, 2], vec![Color32::BLACK; w * 2])), ());
        let (texture, ()) = uploads.pump(&ctx, |_| true).landed.pop().expect("uploaded");
        let _ = ctx.tex_manager().write().take_delta();
        texture.set_partial([TILE_STEP - 2, 0], &egui::ColorImage::new([4, 1], vec![Color32::WHITE; 4]), TextureOptions::LINEAR);
        let set: Vec<_> = ctx.tex_manager().write().take_delta().set.into_iter().map(|(_, d)| (d.pos, d.image.size())).collect();
        assert_eq!(set, vec![(Some([TILE_STEP - 2, 0]), [3, 1]), (Some([0, 0]), [3, 1])], "the first tile to its overlap pixel, the second from its own");
    }

    /// A region past the image repeats it; a region inside it draws only
    /// the tiles it reaches, each at its share of the screen.
    #[test]
    fn a_region_repeats_past_the_image_and_crops_inside_it() {
        let ctx = egui::Context::default();
        let one = ImageTexture::single(ctx.load_texture("one", egui::ColorImage::new([4, 4], vec![Color32::RED; 16]), TextureOptions::LINEAR));
        let screen = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 50.0));
        let repeated = one.region_pieces(screen, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(2.0, 1.0)));
        assert_eq!(repeated.iter().map(|p| p.0).collect::<Vec<_>>(), vec![
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(50.0, 50.0)),
            Rect::from_min_max(egui::pos2(50.0, 0.0), egui::pos2(100.0, 50.0)),
        ], "two copies side by side");
        let cropped = one.region_pieces(screen, Rect::from_min_max(egui::pos2(0.25, 0.0), egui::pos2(0.75, 1.0)));
        assert_eq!(cropped.len(), 1);
        assert_eq!(cropped[0].2, Rect::from_min_max(egui::pos2(0.25, 0.0), egui::pos2(0.75, 1.0)), "the middle half of the texture");
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
            landed.extend(uploads.pump(&ctx, |_| true).landed.into_iter().map(|(t, n)| (n, t.tiles.len(), t.size)));
            assert!(frames < 10);
        }
        assert!(frames > 1, "not all in one frame");
        assert_eq!(landed, vec![(1, 3, [3 * TILE_STEP, 1]), (2, 1, [2, 2])]);
    }

    /// A small image uploaded to wrap draws a repeating region as one
    /// quad; a tiled one cannot wrap and draws a quad per copy and tile.
    #[test]
    fn a_wrapping_texture_repeats_in_one_quad() {
        let ctx = egui::Context::default();
        let wrap = TextureOptions { wrap_mode: egui::TextureWrapMode::Repeat, ..TextureOptions::LINEAR };
        let region = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(3.0, 2.0));
        let screen = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 200.0));
        let mut pieces = Vec::new();
        for (side, options) in [(4, wrap), (4, TextureOptions::LINEAR), (TILE_STEP + 10, wrap)] {
            let mut uploads = Uploads { budget: usize::MAX, ..Uploads::default() };
            uploads.push_with("bg".into(), PreparedImage::from_rgba(&image::RgbaImage::new(side as u32, 1)), (), options);
            let (texture, ()) = uploads.pump(&ctx, |_| true).landed.pop().expect("uploaded");
            pieces.push(texture.region_pieces(screen, region).len());
        }
        assert_eq!(pieces, vec![1, 6, 12]);
    }

    /// An upload no longer wanted is dropped part-way, behind the front
    /// or at it, and comes back as dropped.
    #[test]
    fn an_unwanted_upload_is_dropped_before_its_next_tile() {
        let ctx = egui::Context::default();
        let mut uploads = Uploads { budget: 1, ..Uploads::default() };
        uploads.push("old".into(), PreparedImage::from_rgba(&image::RgbaImage::new((3 * TILE_STEP) as u32, 1)), 1);
        uploads.push("other".into(), PreparedImage::from_rgba(&image::RgbaImage::new(2, 2)), 2);
        let first = uploads.pump(&ctx, |_| true);
        assert!(first.landed.is_empty() && first.dropped.is_empty(), "one tile of the old one");
        let next = uploads.pump(&ctx, |&n| n != 1);
        let landed: Vec<u32> = next.landed.into_iter().map(|(_, n)| n).collect();
        assert_eq!((landed, next.dropped), (vec![2], vec![1]), "the half-uploaded one is gone");
        assert!(uploads.is_empty());
    }
}
