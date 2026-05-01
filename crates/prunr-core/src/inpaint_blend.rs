//! Seam blending — post-process that hides the inpaint↔source boundary.
//!
//! Two complementary stages, both pure functions over RGBA + mask:
//! [`color_match_inpainted`] shifts the inpaint region's mean RGB to
//! match the source ring just outside the mask, and [`seam_guided_blend`]
//! runs a per-channel guided filter (source as guide) and composites it
//! into a band along the seam so the inpaint "tucks under" the
//! surrounding image rather than sitting on top of it as a colour patch.
//!
//! ## RAM budget — peak working set scales with the **bbox** of the mask,
//! not the image. Within `seam_guided_blend`, channels run sequentially
//! and the four `mean_*` buffers are reused across channels via
//! [`crate::guided_filter::box_filter_into`]. Concurrent live `Vec<f32>`
//! at peak: `guide`, `input`, `gi_a`, `gg_b`, `dist`, `mean_g`, `mean_i`,
//! `mean_gi`, `mean_gg` = **9 × n** bytes ÷ 4 = **9n × 4 B**.
//!
//! | bbox      | peak f32 working set |
//! |-----------|----------------------|
//! |  500²     | ~9 MB                |
//! | 1000²     | ~36 MB               |
//! | 4096×3072 | ~450 MB              |
//!
//! Plus the cloned `RgbaImage` output (4 × img_w × img_h u8 — full image,
//! not bbox-scaled) and the byte-sized `crop_mask`.
//!
//! All optimisations are quality-preserving: bbox crop emits bit-exact
//! pixels for unchanged regions, sequential channels run identical math
//! to a parallel-channel variant, and buffer reuse only changes layout.

use image::{GrayImage, RgbaImage};
use rayon::prelude::*;

use crate::guided_filter::{box_filter, box_filter_into};

/// Default ring width for [`color_match_inpainted`] (pixels). Tuned for
/// 1024-class inpaint output; on much-larger sources, the visible band
/// stays the same absolute width — a feature, not a bug, since the
/// underlying inpaint patch size doesn't grow with image resolution.
pub const COLOR_MATCH_RING_PX: u32 = 12;

/// Default radius for [`seam_guided_blend`]. Six pixels picks up enough
/// surrounding texture to drive the locally-affine fit without over-
/// smoothing across a real edge.
pub const SEAM_BLEND_RADIUS: u32 = 6;

/// Default regularisation for [`seam_guided_blend`]. 1e-3 is a good
/// trade-off between edge sharpness (lower) and seam smoothness (higher).
pub const SEAM_BLEND_EPSILON: f32 = 1e-3;

/// Default band width for [`seam_guided_blend`]. Pixels deeper than this
/// stay as the raw inpainted output; pixels at the edge fade toward the
/// guided-filter result.
pub const SEAM_BLEND_BAND_PX: f32 = 8.0;

/// Bounding box of a region inside an image (top-left + size, all in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bbox {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl Bbox {
    fn area(&self) -> usize {
        (self.w as usize) * (self.h as usize)
    }
}

/// Color-correct the inpainted region by matching its mean RGB to the
/// source pixels just outside the mask boundary.
///
/// Returns the input unchanged when the mask is empty, dimensions
/// disagree, or fewer than 16 valid sample pixels exist on either side
/// of the boundary (insufficient sample → noisy correction; common at
/// image edges where the source ring is clipped).
///
/// **Working set scales with the mask bbox, not the image.** A 200×200
/// stroke on an 8 MP source allocates ~400 KB of f32 scratch instead of
/// ~64 MB. Bit-exact vs. the full-image variant — pixels outside the
/// bbox have mask = 0 by construction (the bbox encloses every mask >=
/// 128 pixel), and the offset-apply pass only writes where mask >= 128.
pub fn color_match_inpainted(
    inpainted: &RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    ring_px: u32,
) -> RgbaImage {
    if ring_px == 0
        || inpainted.dimensions() != source.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        return inpainted.clone();
    }
    let (img_w, img_h) = inpainted.dimensions();
    let Some(bbox) = mask_bbox_expanded(mask, ring_px, img_w, img_h) else {
        return inpainted.clone();
    };
    let bwu = bbox.w as usize;
    let bhu = bbox.h as usize;
    let bxu = bbox.x as usize;
    let byu = bbox.y as usize;
    let img_wu = img_w as usize;

    // Boundary detection via box-filtered binary mask, evaluated on the
    // cropped mask only — pixels with mean strictly between 0 and 1 are
    // within `ring_px` of the seam.
    let msk_raw = mask.as_raw();
    let mut mask_bin = vec![0.0f32; bbox.area()];
    mask_bin
        .par_chunks_mut(bwu)
        .enumerate()
        .for_each(|(j, row)| {
            let img_y = byu + j;
            let row_off = img_y * img_wu + bxu;
            for (i, slot) in row.iter_mut().enumerate() {
                *slot = if msk_raw[row_off + i] >= 128 { 1.0 } else { 0.0 };
            }
        });
    let mean = box_filter(&mask_bin, bbox.w, bbox.h, ring_px);

    // Sum source RGB on the outer ring and inpaint RGB on the inner ring.
    // Walk bbox-local indices; map to image-global byte offset for sampling.
    let src_raw = source.as_raw();
    let inp_raw = inpainted.as_raw();
    let mut src_sum = [0u64; 3];
    let mut src_count = 0u64;
    let mut inp_sum = [0u64; 3];
    let mut inp_count = 0u64;
    for j in 0..bhu {
        let img_y = byu + j;
        let mean_row_off = j * bwu;
        let img_row_off = img_y * img_wu;
        for i in 0..bwu {
            let m = mean[mean_row_off + i];
            if !(m > 0.001 && m < 0.999) {
                continue;
            }
            let img_idx = img_row_off + bxu + i;
            let p = img_idx * 4;
            if msk_raw[img_idx] >= 128 {
                inp_count += 1;
                inp_sum[0] += inp_raw[p] as u64;
                inp_sum[1] += inp_raw[p + 1] as u64;
                inp_sum[2] += inp_raw[p + 2] as u64;
            } else {
                src_count += 1;
                src_sum[0] += src_raw[p] as u64;
                src_sum[1] += src_raw[p + 1] as u64;
                src_sum[2] += src_raw[p + 2] as u64;
            }
        }
    }
    if src_count < 16 || inp_count < 16 {
        return inpainted.clone();
    }
    let offset: [f32; 3] = std::array::from_fn(|c| {
        src_sum[c] as f32 / src_count as f32 - inp_sum[c] as f32 / inp_count as f32
    });

    // Apply offset only on bbox rows; mask >= 128 pixels live entirely
    // inside the bbox by construction.
    let mut out = inpainted.clone();
    let out_raw = out.as_mut();
    out_raw[byu * img_wu * 4..(byu + bhu) * img_wu * 4]
        .par_chunks_mut(img_wu * 4)
        .enumerate()
        .for_each(|(j, row)| {
            let img_y = byu + j;
            let img_row_off = img_y * img_wu;
            for i in 0..bwu {
                let img_x = bxu + i;
                if msk_raw[img_row_off + img_x] < 128 {
                    continue;
                }
                let p = img_x * 4;
                row[p] = (row[p] as f32 + offset[0]).clamp(0.0, 255.0) as u8;
                row[p + 1] = (row[p + 1] as f32 + offset[1]).clamp(0.0, 255.0) as u8;
                row[p + 2] = (row[p + 2] as f32 + offset[2]).clamp(0.0, 255.0) as u8;
            }
        });
    out
}

/// Refine the inpaint↔source seam using a 3-channel guided filter (one
/// pass per channel, source as guide). Inside the mask boundary band the
/// output blends from the guided-filter result (at the edge) to the raw
/// inpaint (deep interior). Outside the mask, output is left unchanged
/// (the caller's compose step has already painted source there).
///
/// Returns the input unchanged when the mask is empty or dimensions
/// disagree. Internally crops to the mask bounding box plus a margin
/// of `radius + ceil(band_px) + 2` so a small brush stroke pays for a
/// small bbox, not the whole image.
pub fn seam_guided_blend(
    inpainted: &RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
    band_px: f32,
) -> RgbaImage {
    if band_px <= 0.0
        || radius == 0
        || inpainted.dimensions() != source.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        return inpainted.clone();
    }
    let (img_w, img_h) = inpainted.dimensions();
    let margin = radius + band_px.ceil() as u32 + 2;
    let Some(bbox) = mask_bbox_expanded(mask, margin, img_w, img_h) else {
        return inpainted.clone();
    };

    let n = bbox.area();
    let mut out = inpainted.clone();

    // Distance-from-edge inside the mask, evaluated on the cropped mask
    // so chamfer cost scales with bbox not image. (Reused across channels
    // — distance is independent of colour.)
    let crop_mask =
        image::imageops::crop_imm(mask, bbox.x, bbox.y, bbox.w, bbox.h).to_image();
    let dist = crate::inpaint::chamfer_distance_inside(&crop_mask);
    let crop_mask_raw = crop_mask.as_raw();

    // Allocate working buffers ONCE; reuse across channels.
    // - guide / input: bbox source / inpaint, one channel at a time
    // - gi_a / gg_b: re-purposed (guide×input → coeff `a`, guide² → coeff `b`)
    // - mean_*: box-filter outputs, written via `box_filter_into` so the
    //   four allocations happen once total, not 4×3 = 12 times.
    let mut guide = vec![0.0f32; n];
    let mut input = vec![0.0f32; n];
    let mut gi_a = vec![0.0f32; n];
    let mut gg_b = vec![0.0f32; n];
    let mut mean_g = vec![0.0f32; n];
    let mut mean_i = vec![0.0f32; n];
    let mut mean_gi = vec![0.0f32; n];
    let mut mean_gg = vec![0.0f32; n];
    let mut mean_a = vec![0.0f32; n];
    let mut mean_b = vec![0.0f32; n];

    let inv_band = 1.0 / band_px;

    for c in 0..3 {
        load_channel(source, bbox, c, &mut guide);
        load_channel(inpainted, bbox, c, &mut input);

        // gi = guide * input, gg = guide * guide
        gi_a.par_iter_mut()
            .zip(gg_b.par_iter_mut())
            .zip(guide.par_iter().zip(input.par_iter()))
            .for_each(|((gi, gg), (&g, &i))| {
                *gi = g * i;
                *gg = g * g;
            });

        // Box filters 1-4. Two inner pairs joined for CPU overlap. Each
        // writes into a pre-allocated buffer, no per-call allocation.
        // Buffers are aliased only as exclusive `&mut` slices passed into
        // independent rayon::join branches → no overlap, sound.
        let (mg, mi, mgi, mgg) =
            (&mut mean_g, &mut mean_i, &mut mean_gi, &mut mean_gg);
        rayon::join(
            || {
                rayon::join(
                    || box_filter_into(&guide, bbox.w, bbox.h, radius, mg),
                    || box_filter_into(&input, bbox.w, bbox.h, radius, mi),
                )
            },
            || {
                rayon::join(
                    || box_filter_into(&gi_a, bbox.w, bbox.h, radius, mgi),
                    || box_filter_into(&gg_b, bbox.w, bbox.h, radius, mgg),
                )
            },
        );

        // a = cov / (var + eps), b = mean_i - a * mean_g.
        // Reuse gi_a/gg_b in-place. `var.max(0.0)` guards against
        // sub-epsilon negative variance from f32 rounding on flat regions
        // (would otherwise feed `/(var+eps)` a value below `eps` and
        // produce noise on completely uniform input).
        gi_a.par_iter_mut()
            .zip(gg_b.par_iter_mut())
            .zip(
                mean_g
                    .par_iter()
                    .zip(mean_i.par_iter())
                    .zip(mean_gi.par_iter().zip(mean_gg.par_iter())),
            )
            .for_each(|((a, b), ((&mg_v, &mi_v), (&mgi_v, &mgg_v)))| {
                let var = (mgg_v - mg_v * mg_v).max(0.0);
                let cov = mgi_v - mg_v * mi_v;
                *a = cov / (var + epsilon);
                *b = mi_v - *a * mg_v;
            });

        rayon::join(
            || box_filter_into(&gi_a, bbox.w, bbox.h, radius, &mut mean_a),
            || box_filter_into(&gg_b, bbox.w, bbox.h, radius, &mut mean_b),
        );

        // Composite back into `out`: q = mean_a * source_pixel + mean_b,
        // blended toward raw inpaint deeper in the mask.
        composite_channel(
            &mut out, source, bbox, c, crop_mask_raw,
            &dist, &mean_a, &mean_b, inv_band,
        );
    }

    out
}

/// Find the bounding box of mask>=128 pixels, expanded by `margin` and
/// clamped to image bounds. Parallel row-reduction — each row computes
/// its own min/max in parallel, then merged. Returns `None` for an
/// all-zero mask.
fn mask_bbox_expanded(
    mask: &GrayImage,
    margin: u32,
    img_w: u32,
    img_h: u32,
) -> Option<Bbox> {
    let raw = mask.as_raw();
    let w = mask.width() as usize;
    let h = mask.height() as usize;

    /// Per-row scan: returns (min_x, max_x, has_any) for one row.
    fn row_extent(row: &[u8]) -> (usize, usize, bool) {
        let mut min_x = usize::MAX;
        let mut max_x = 0usize;
        let mut any = false;
        for (x, &v) in row.iter().enumerate() {
            if v >= 128 {
                if !any { min_x = x; }
                max_x = x;
                any = true;
            }
        }
        (min_x, max_x, any)
    }

    let merged = (0..h)
        .into_par_iter()
        .map(|y| {
            let row = &raw[y * w..(y + 1) * w];
            let (rmin_x, rmax_x, any) = row_extent(row);
            if any {
                Some((rmin_x, y, rmax_x, y))
            } else {
                None
            }
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (None, x) | (x, None) => x,
                (Some((axn, ayn, axx, ayx)), Some((bxn, byn, bxx, byx))) => {
                    Some((axn.min(bxn), ayn.min(byn), axx.max(bxx), ayx.max(byx)))
                }
            },
        )?;
    let (min_x, min_y, max_x, max_y) = merged;

    let m = margin as usize;
    let bx = min_x.saturating_sub(m) as u32;
    let by = min_y.saturating_sub(m) as u32;
    let bx_end = (max_x + m + 1).min(img_w as usize) as u32;
    let by_end = (max_y + m + 1).min(img_h as usize) as u32;
    Some(Bbox { x: bx, y: by, w: bx_end - bx, h: by_end - by })
}

fn load_channel(img: &RgbaImage, bbox: Bbox, channel: usize, out: &mut [f32]) {
    let src = img.as_raw();
    let sw = img.width() as usize;
    let bwu = bbox.w as usize;
    let bxu = bbox.x as usize;
    let byu = bbox.y as usize;
    out.par_chunks_mut(bwu).enumerate().for_each(|(j, row)| {
        let src_base = ((byu + j) * sw + bxu) * 4;
        for x in 0..bwu {
            row[x] = src[src_base + x * 4 + channel] as f32;
        }
    });
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut u8);
// SAFETY: callers must partition writes so no two parallel iterations
// touch the same byte; see `composite_channel`'s row-disjoint partition.
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

// Per-channel inner kernel — args are the math context the row-parallel
// loop closes over. Splitting into a struct adds a borrow-lifetime story
// for what is fundamentally a leaf math helper.
#[allow(clippy::too_many_arguments)]
fn composite_channel(
    out: &mut RgbaImage,
    source: &RgbaImage,
    bbox: Bbox,
    channel: usize,
    mask_raw: &[u8],
    dist: &[f32],
    mean_a: &[f32],
    mean_b: &[f32],
    inv_band: f32,
) {
    let src = source.as_raw();
    let sw = source.width() as usize;
    let dst_raw = out.as_mut();
    let bwu = bbox.w as usize;
    let bxu = bbox.x as usize;
    let byu = bbox.y as usize;
    // SAFETY: each parallel iteration `j` writes at most to bytes
    // (byu+j)*sw*4 + ... + channel — i.e., row-disjoint by `j` AND
    // channel-disjoint within a row (we never write the alpha byte or
    // a different channel). `seam_guided_blend` calls this sequentially
    // for c=0,1,2, so different channels also can't race across calls.
    let dp = SendMutPtr(dst_raw.as_mut_ptr());
    (0..bbox.h as usize).into_par_iter().for_each(|j| {
        let row_in_full = (byu + j) * sw + bxu;
        for x in 0..bwu {
            let i = j * bwu + x;
            if mask_raw[i] < 128 {
                continue; // outside mask: leave caller's compose alone
            }
            // Guided-filter prediction at this pixel.
            let g = src[(row_in_full + x) * 4 + channel] as f32;
            let q = mean_a[i] * g + mean_b[i];
            // Blend weight: 0 at the seam (use guided), 1 deep in mask
            // (use raw inpaint already in `out`).
            let w = (dist[i] * inv_band).clamp(0.0, 1.0);
            let pix_off = ((row_in_full + x) * 4 + channel) as isize;
            unsafe {
                let p = dp.0.offset(pix_off);
                let cur = *p as f32;
                let blended = q + w * (cur - q);
                *p = blended.clamp(0.0, 255.0) as u8;
            }
        }
        // Force whole-struct capture of `dp` (Rust 2021 disjoint capture
        // would otherwise grab `dp.0: *mut u8` alone, which isn't Sync;
        // the unsafe Send/Sync impls live on `SendMutPtr`, not on the
        // raw pointer field). Same pattern as `guided_filter::box_filter`.
        let _ = &dp;
    });
}

/// Single source of truth for the inpaint post-process pipeline:
/// color match → seam-guided blend → optional sharpen. Used by the
/// LaMa rayon path and the subprocess worker arm so constants and
/// order can't drift across call sites. `feather_px == 0.0` falls back
/// to `SEAM_BLEND_BAND_PX`; `sharpen <= 0.0` skips the sharpen pass.
pub fn finalize_inpaint(
    raw: &RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    feather_px: f32,
    sharpen: f32,
) -> RgbaImage {
    let color_matched = color_match_inpainted(raw, source, mask, COLOR_MATCH_RING_PX);
    let band_px = if feather_px > 0.0 { feather_px } else { SEAM_BLEND_BAND_PX };
    let mut out = seam_guided_blend(
        &color_matched, source, mask,
        SEAM_BLEND_RADIUS, SEAM_BLEND_EPSILON, band_px,
    );
    if sharpen > 0.0 {
        out = crate::inpaint::sharpen_inpainted(&out, mask, sharpen);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(rgba))
    }
    fn empty_mask(w: u32, h: u32) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([0]))
    }

    #[test]
    fn color_match_empty_mask_is_noop() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let inp = solid(64, 64, [50, 50, 50, 255]);
        let mask = empty_mask(64, 64);
        let out = color_match_inpainted(&inp, &src, &mask, 8);
        assert_eq!(out, inp);
    }

    #[test]
    fn color_match_dim_mismatch_is_noop() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let inp = solid(64, 64, [50, 50, 50, 255]);
        let mask = empty_mask(32, 32);
        let out = color_match_inpainted(&inp, &src, &mask, 8);
        assert_eq!(out, inp);
    }

    #[test]
    fn color_match_shifts_inpaint_toward_source_mean() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let mut inp = solid(64, 64, [200, 200, 200, 255]);
        let mut mask = empty_mask(64, 64);
        for y in 16..48 {
            for x in 16..48 {
                inp.put_pixel(x, y, Rgba([100, 100, 100, 255]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = color_match_inpainted(&inp, &src, &mask, 8);
        let p = out.get_pixel(32, 32);
        assert!(
            p[0] >= 180 && p[0] <= 220,
            "expected ~200 after color match, got {}",
            p[0],
        );
        let outside = out.get_pixel(0, 0);
        assert_eq!(outside[0], 200);
    }

    /// All-mask image: every pixel is inside the mask, so the source
    /// ring is empty (`src_count == 0`). Function must fall through
    /// gracefully and return the input unchanged.
    #[test]
    fn color_match_all_mask_returns_input() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let inp = solid(64, 64, [50, 50, 50, 255]);
        let mask = GrayImage::from_pixel(64, 64, Luma([255]));
        let out = color_match_inpainted(&inp, &src, &mask, 8);
        assert_eq!(out, inp);
    }

    /// Bbox-crop contract: pixels outside the mask's expanded bbox must
    /// stay bit-exact identical to the inpaint input. A 32×32 stroke at
    /// (200, 200) on a 512×512 image leaves the four corners untouched.
    #[test]
    fn color_match_outside_bbox_is_bit_exact() {
        let src = solid(512, 512, [200, 150, 100, 255]);
        let mut inp = solid(512, 512, [60, 90, 120, 255]);
        let mut mask = empty_mask(512, 512);
        for y in 200..232 {
            for x in 200..232 {
                inp.put_pixel(x, y, Rgba([60, 90, 120, 255]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = color_match_inpainted(&inp, &src, &mask, 8);
        // Every pixel >24 px from the stroke edge sits outside the bbox
        // (ring_px = 8 + 16 px slack to clear the box-filter footprint).
        for &(x, y) in &[(0, 0), (511, 0), (0, 511), (511, 511), (100, 100), (400, 400)] {
            assert_eq!(
                out.get_pixel(x, y),
                inp.get_pixel(x, y),
                "outside-bbox pixel ({x}, {y}) should be bit-exact"
            );
        }
    }

    #[test]
    fn seam_blend_empty_mask_is_noop() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let inp = solid(64, 64, [50, 50, 50, 255]);
        let mask = empty_mask(64, 64);
        let out = seam_guided_blend(&inp, &src, &mask, 4, 1e-3, 8.0);
        assert_eq!(out, inp);
    }

    #[test]
    fn seam_blend_smooths_step_at_boundary() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let mut inp = solid(64, 64, [200, 200, 200, 255]);
        let mut mask = empty_mask(64, 64);
        for y in 16..48 {
            for x in 16..48 {
                inp.put_pixel(x, y, Rgba([50, 50, 50, 255]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = seam_guided_blend(&inp, &src, &mask, 4, 1e-3, 8.0);
        let edge = out.get_pixel(16, 32);
        assert!(
            edge[0] > 60,
            "boundary pixel expected to lift toward source; got {}",
            edge[0],
        );
        let deep = out.get_pixel(32, 32);
        assert!(
            deep[0] < 100,
            "deep-interior pixel should keep raw inpaint; got {}",
            deep[0],
        );
        let outside = out.get_pixel(0, 0);
        assert_eq!(outside[0], 200);
    }

    /// Mask touching the image boundary has fewer source samples on
    /// the clipped side. The function must still produce sensible
    /// output (no panic, no negative-variance NaN, mask interior
    /// pulled toward source where source samples exist).
    #[test]
    fn seam_blend_mask_touching_image_boundary() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let mut inp = solid(64, 64, [200, 200, 200, 255]);
        let mut mask = empty_mask(64, 64);
        // Mask runs flush against the top-left corner.
        for y in 0..32 {
            for x in 0..32 {
                inp.put_pixel(x, y, Rgba([50, 50, 50, 255]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = seam_guided_blend(&inp, &src, &mask, 4, 1e-3, 8.0);
        // Pixel near the inside seam (right/bottom of mask) should be
        // lifted toward source. Pixels at the actual image corner have
        // less context so we don't assert on (0,0).
        let near_seam = out.get_pixel(31, 31);
        assert!(
            near_seam[0] > 60,
            "seam-band pixel should lift toward source; got {}",
            near_seam[0],
        );
        // Outside the mask must remain pristine.
        let outside = out.get_pixel(40, 40);
        assert_eq!(outside[0], 200);
    }

    #[test]
    fn bbox_finds_tight_box_with_margin() {
        let mut mask = empty_mask(64, 64);
        for y in 20..30 {
            for x in 25..35 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let bbox = mask_bbox_expanded(&mask, 4, 64, 64);
        assert_eq!(bbox, Some(Bbox { x: 21, y: 16, w: 18, h: 18 }));
    }

    #[test]
    fn bbox_clamps_to_image_bounds() {
        let mut mask = empty_mask(32, 32);
        mask.put_pixel(0, 0, Luma([255]));
        let bbox = mask_bbox_expanded(&mask, 8, 32, 32);
        assert_eq!(bbox, Some(Bbox { x: 0, y: 0, w: 9, h: 9 }));
    }

    #[test]
    fn bbox_empty_mask_returns_none() {
        let mask = empty_mask(32, 32);
        assert!(mask_bbox_expanded(&mask, 4, 32, 32).is_none());
    }

    /// Full-image mask: bbox spans the whole image plus margin (clamped).
    #[test]
    fn bbox_full_image_mask() {
        let mask = GrayImage::from_pixel(32, 32, Luma([255]));
        let bbox = mask_bbox_expanded(&mask, 4, 32, 32);
        assert_eq!(bbox, Some(Bbox { x: 0, y: 0, w: 32, h: 32 }));
    }
}
