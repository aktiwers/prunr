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
//! and the six `mean_*` buffers are reused across channels via
//! [`crate::guided_filter::box_filter_into`]. Concurrent live `Vec<f32>`
//! of n = bbox pixels at peak: `guide`, `input`, `gi_a`, `gg_b`, `dist`,
//! `mean_g`, `mean_i`, `mean_gi`, `mean_gg`, `mean_a`, `mean_b` =
//! **11 × n × 4 B**.
//!
//! | bbox      | peak f32 working set |
//! |-----------|----------------------|
//! |  500²     | ~11 MB               |
//! | 1000²     | ~44 MB               |
//! | 4096×3072 | ~550 MB              |
//!
//! Plus the cloned `RgbaImage` output (4 × img_w × img_h u8 — full image,
//! not bbox-scaled) and the byte-sized `crop_mask`.

use image::{GrayImage, RgbaImage};
use rayon::prelude::*;

use crate::guided_filter::{box_filter, box_filter_into};
use crate::inpaint::{mask_bbox, Bbox};
use crate::math::SendMutPtr;

/// Default ring width for [`color_match_inpainted`] (pixels). Fixed in
/// absolute pixels on purpose: the inpaint patch size doesn't grow with
/// image resolution.
pub const COLOR_MATCH_RING_PX: u32 = 12;

/// Default radius for [`seam_guided_blend`]: enough texture for the
/// locally-affine fit without smoothing across a real edge.
pub const SEAM_BLEND_RADIUS: u32 = 6;

/// Default regularisation for [`seam_guided_blend`]: lower is sharper
/// edges, higher is a smoother seam.
pub const SEAM_BLEND_EPSILON: f32 = 1e-3;

/// The eraser's default Expand region and Edge blend, in pixels.
pub const DEFAULT_GROW_PX: f32 = 2.0;
pub const DEFAULT_EDGE_BLEND_PX: f32 = 4.0;

/// The region a fill replaces: what was painted, grown by `grow_px`, then
/// by the `edge_blend_px` ring the fill fades into the photo over. The
/// ring lies outside the grown region, so the blend never reaches the
/// object being erased.
pub fn fill_region(painted: &GrayImage, grow_px: f32, edge_blend_px: f32) -> GrayImage {
    crate::inpaint::grow_mask(painted, (grow_px + edge_blend_px.max(0.0)).round() as i32)
}

/// What an erase of `painted` will touch, for the overlay to show before
/// it runs: the painted region grown by `grow_px` at full strength, and
/// the `edge_blend_px` band beyond it at half, where the fill is blended
/// into the photo. The same `fill_region` the erase uses draws both.
///
/// Peak working set: two grey planes of the mask's size (1 B/px each)
/// and the returned cells (1 B/px).
pub fn fill_preview(painted: &crate::selection::MaskArtifact, grow_px: f32, edge_blend_px: f32) -> crate::selection::MaskArtifact {
    use crate::selection::{MaskArtifact, FULL};
    let (w, h) = (painted.width, painted.height);
    let region = painted.region_mask(w, h);
    let core = fill_region(&region, grow_px, 0.0);
    let reach = fill_region(&region, grow_px, edge_blend_px);
    let cells = core.as_raw().iter().zip(reach.as_raw())
        .map(|(&c, &r)| if c >= 128 { FULL } else if r >= 128 { FULL / 2 } else { 0 })
        .collect();
    MaskArtifact::from_cells(w, h, cells)
}

/// Color-correct the inpainted region by matching its mean RGB to the
/// source pixels just outside the mask boundary.
///
/// Returns the input unchanged when the mask is empty, dimensions
/// disagree, or fewer than 16 valid sample pixels exist on either side
/// of the boundary (insufficient sample → noisy correction; common at
/// image edges where the source ring is clipped).
///
/// Working set scales with the mask bbox, not the image: a 200×200
/// stroke on an 8 MP source allocates ~400 KB of f32 scratch, not ~64 MB.
pub fn color_match_inpainted(
    inpainted: &mut RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    ring_px: u32,
) {
    if ring_px == 0
        || inpainted.dimensions() != source.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        return;
    }
    let Some(bbox) = mask_bbox(mask, 128, ring_px) else {
        return;
    };
    let bwu = bbox.w as usize;
    let bhu = bbox.h as usize;
    let bxu = bbox.x as usize;
    let byu = bbox.y as usize;
    let img_wu = inpainted.width() as usize;

    // Pixels whose box-filtered mask mean is strictly between 0 and 1 lie
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
        return;
    }
    let offset: [f32; 3] = std::array::from_fn(|c| {
        src_sum[c] as f32 / src_count as f32 - inp_sum[c] as f32 / inp_count as f32
    });

    // Every mask >= 128 pixel lies inside the bbox by construction.
    let out_raw = inpainted.as_mut();
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
                row[p] = crate::math::round_u8(row[p] as f32 + offset[0]);
                row[p + 1] = crate::math::round_u8(row[p + 1] as f32 + offset[1]);
                row[p + 2] = crate::math::round_u8(row[p + 2] as f32 + offset[2]);
            }
        });
}

/// Color-correct the fill point by point along its edge: at each fill
/// pixel within `ring_px` of the edge, the photo's mean just outside less
/// the fill's mean just inside (one `ring_px` box), spread over the whole
/// region by push-pull interpolation and faded out over `reach_px` from
/// the edge. A region bordering different surroundings (dark hair on one
/// side, pale sky on the other) then meets each, which one mean shift for
/// the region cannot; the fade keeps a difference that is content, not
/// cast (an edge across a face), from spreading through the fill.
///
/// Leaves `inpainted` as it is when the mask is empty, dimensions
/// disagree, or no fill pixel has photo within `ring_px`.
///
/// Peak working set, n = mask bbox plus `max(ring_px, reach_px)` (f32):
/// while the offsets are measured, the inside mask, its box share, the
/// edge weight, the fade, the offset and two reused scratch planes (7 n);
/// while one channel spreads, the inside mask, weight, fade and offset
/// plus push-pull's own levels and output (about 4 n) = **about 8 × n ×
/// 4 B**; channels run in turn.
///
/// | bbox      | peak f32 working set |
/// |-----------|----------------------|
/// |  500²     | ~8 MB                |
/// | 1000²     | ~32 MB               |
/// | 2000²     | ~130 MB              |
pub fn match_colour_along_edge(
    inpainted: &mut RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    ring_px: u32,
    reach_px: u32,
) {
    if ring_px == 0
        || inpainted.dimensions() != source.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        return;
    }
    let Some(bbox) = mask_bbox(mask, 128, ring_px.max(reach_px) + 1) else {
        return;
    };
    let (bw, n) = (bbox.w as usize, bbox.area());
    let (bx, by) = (bbox.x as usize, bbox.y as usize);
    let img_w = inpainted.width() as usize;
    let msk = mask.as_raw();
    let mut inside = vec![0.0f32; n];
    inside.par_chunks_mut(bw).enumerate().for_each(|(j, row)| {
        let at = (by + j) * img_w + bx;
        for (slot, &m) in row.iter_mut().zip(&msk[at..at + bw]) {
            *slot = if m >= 128 { 1.0 } else { 0.0 };
        }
    });
    let in_share = box_filter(&inside, bbox.w, bbox.h, ring_px);
    // An edge pixel has photo and fill each fill a fair part of its box,
    // so both means rest on enough samples.
    let weight: Vec<f32> = inside.iter().zip(&in_share)
        .map(|(&i, s)| if i > 0.5 && (0.2..=0.8).contains(s) { 1.0 } else { 0.0 })
        .collect();
    if !weight.iter().any(|&w| w > 0.0) {
        return;
    }
    // 1 at the edge, where the fill box is half inside, to 0 once the
    // `reach_px` box lies wholly in the fill.
    let fade: Vec<f32> = box_filter(&inside, bbox.w, bbox.h, reach_px.max(1)).iter()
        .map(|&share| ((1.0 - share) * 2.0).clamp(0.0, 1.0))
        .collect();
    let (mut plane, mut sum) = (vec![0.0f32; n], vec![0.0f32; n]); // reused per channel
    let mut offset = vec![0.0f32; n];
    for c in 0..3 {
        // The photo's mean just outside, less the fill's just inside.
        load_channel(source, bbox, c, &mut plane);
        plane.iter_mut().zip(&inside).for_each(|(v, &i)| *v *= 1.0 - i);
        box_filter_into(&plane, bbox.w, bbox.h, ring_px, &mut sum);
        for (i, o) in offset.iter_mut().enumerate() {
            *o = if weight[i] > 0.0 { sum[i] / (1.0 - in_share[i]) } else { 0.0 };
        }
        load_channel(inpainted, bbox, c, &mut plane);
        plane.iter_mut().zip(&inside).for_each(|(v, &i)| *v *= i);
        box_filter_into(&plane, bbox.w, bbox.h, ring_px, &mut sum);
        for (i, o) in offset.iter_mut().enumerate() {
            if weight[i] > 0.0 {
                *o -= sum[i] / in_share[i];
            }
        }
        let field = push_pull(&offset, &weight, bw, bbox.h as usize);
        let raw = inpainted.as_mut();
        raw[by * img_w * 4..(by + bbox.h as usize) * img_w * 4]
            .par_chunks_mut(img_w * 4)
            .enumerate()
            .for_each(|(j, row)| {
                for x in 0..bw {
                    let i = j * bw + x;
                    if inside[i] > 0.5 {
                        let p = (bx + x) * 4 + c;
                        row[p] = crate::math::round_u8(row[p] as f32 + field[i] * fade[i]);
                    }
                }
            });
    }
}

/// Fill the gaps of `value` (known where `weight` > 0) smoothly from the
/// known samples: halve with weights summed until one cell is left, then
/// come back up, keeping each known value and taking the coarser level's
/// (bilinear) everywhere else.
fn push_pull(value: &[f32], weight: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Each level: (w, h, weighted sum, weight).
    let mut levels = vec![(w, h, value.iter().zip(weight).map(|(v, k)| v * k).collect::<Vec<f32>>(), weight.to_vec())];
    while let Some((lw, lh, sum, wt)) = levels.last().filter(|(lw, lh, ..)| *lw > 1 || *lh > 1) {
        let (cw, ch) = (lw.div_ceil(2), lh.div_ceil(2));
        let (mut csum, mut cwt) = (vec![0.0f32; cw * ch], vec![0.0f32; cw * ch]);
        for y in 0..*lh {
            for x in 0..*lw {
                let (i, ci) = (y * lw + x, (y / 2) * cw + x / 2);
                csum[ci] += sum[i];
                cwt[ci] += wt[i];
            }
        }
        levels.push((cw, ch, csum, cwt));
    }
    let (_, _, sum, wt) = levels.pop().expect("level 0 is always there");
    let mut coarse: Vec<f32> = sum.iter().zip(&wt).map(|(s, k)| if *k > 0.0 { s / k } else { 0.0 }).collect();
    let mut coarse_dims = (1usize, 1usize);
    while let Some((lw, lh, sum, wt)) = levels.pop() {
        let (cw, ch) = coarse_dims;
        let sample = |fx: f32, fy: f32| {
            let (x0, y0) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
            let (x0, y0) = (x0.min(cw - 1), y0.min(ch - 1));
            let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(ch - 1));
            let (tx, ty) = ((fx - x0 as f32).clamp(0.0, 1.0), (fy - y0 as f32).clamp(0.0, 1.0));
            let top = coarse[y0 * cw + x0] * (1.0 - tx) + coarse[y0 * cw + x1] * tx;
            let bottom = coarse[y1 * cw + x0] * (1.0 - tx) + coarse[y1 * cw + x1] * tx;
            top * (1.0 - ty) + bottom * ty
        };
        let fine: Vec<f32> = (0..lw * lh).map(|i| {
            let (x, y) = (i % lw, i / lw);
            let below = sample((x as f32 - 0.5) / 2.0, (y as f32 - 0.5) / 2.0);
            let known = wt[i].min(1.0);
            if wt[i] > 0.0 { known * (sum[i] / wt[i]) + (1.0 - known) * below } else { below }
        }).collect();
        coarse = fine;
        coarse_dims = (lw, lh);
    }
    coarse
}

/// Refine the inpaint↔source seam using a 3-channel guided filter (one
/// pass per channel, source as guide). Inside the mask boundary band the
/// output blends from the guided-filter result (at the edge) to the raw
/// inpaint (deep interior). Outside the mask, output is left unchanged
/// (the caller's compose step has already painted source there).
///
/// Leaves `inpainted` as it is when the mask is empty or dimensions
/// disagree. Works on the mask bbox plus a `radius + ceil(band_px) + 2`
/// margin.
pub fn seam_guided_blend(
    inpainted: &mut RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    radius: u32,
    epsilon: f32,
    band_px: f32,
) {
    if band_px <= 0.0
        || radius == 0
        || inpainted.dimensions() != source.dimensions()
        || source.dimensions() != mask.dimensions()
    {
        return;
    }
    let margin = radius + band_px.ceil() as u32 + 2;
    let Some(bbox) = mask_bbox(mask, 128, margin) else {
        return;
    };

    let n = bbox.area();

    // Distance is independent of colour, so one pass serves all channels.
    let crop_mask =
        image::imageops::crop_imm(mask, bbox.x, bbox.y, bbox.w, bbox.h).to_image();
    let dist = crate::inpaint::chamfer_distance_inside(&crop_mask);
    let crop_mask_raw = crop_mask.as_raw();

    // Allocated once and reused across channels; gi_a / gg_b hold
    // guide×input and guide² first, then are reused as coefficients a / b.
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

        gi_a.par_iter_mut()
            .zip(gg_b.par_iter_mut())
            .zip(guide.par_iter().zip(input.par_iter()))
            .for_each(|((gi, gg), (&g, &i))| {
                *gi = g * i;
                *gg = g * g;
            });

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

        // `var.max(0.0)`: f32 rounding on flat regions can go sub-zero,
        // which would amplify noise through `/(var+eps)`.
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

        // Writing channel `c` in place is safe: no later channel reads it.
        composite_channel(
            inpainted, source, bbox, c, crop_mask_raw,
            &dist, &mean_a, &mean_b, inv_band,
        );
    }
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

// Leaf math kernel: the args are the context the row-parallel loop
// closes over, and it has a single caller.
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
                *p = crate::math::round_u8(blended);
            }
        }
        // Capture `dp` whole: its raw-pointer field alone is not Sync.
        let _ = &dp;
    });
}

/// How a fill's colours are brought to the photo around it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EdgeColour {
    /// Point by point along the edge (`match_colour_along_edge`).
    #[default]
    AlongEdge,
    /// One shift for the whole region (`color_match_inpainted`). Where the
    /// region borders different surroundings it misses each, and the fill
    /// shows as a lighter or darker patch.
    WholeArea,
    /// The fill's own colours.
    Off,
}

/// How far in from the edge `EdgeColour::AlongEdge` corrects for a
/// Feather of `feather_px`: half of it, as measured best on SD 1.5, LaMa,
/// Big-LaMa and MI-GAN fills (seam step 41-64 % below one shift).
pub fn edge_colour_reach(feather_px: f32) -> u32 {
    (feather_px / 2.0).round().max(4.0) as u32
}

/// The one post-process pipeline for every fill — color match →
/// seam-guided blend → optional sharpen — so order and constants can't
/// drift between callers. `mask` is the [`fill_region`];
/// `feather_px <= 0.0` skips the blend, `sharpen <= 0.0` the sharpen.
pub fn finalize_inpaint(
    mut raw: RgbaImage,
    source: &RgbaImage,
    mask: &GrayImage,
    feather_px: f32,
    sharpen: f32,
    colour: EdgeColour,
) -> RgbaImage {
    match colour {
        EdgeColour::AlongEdge => match_colour_along_edge(&mut raw, source, mask, COLOR_MATCH_RING_PX, edge_colour_reach(feather_px)),
        EdgeColour::WholeArea => color_match_inpainted(&mut raw, source, mask, COLOR_MATCH_RING_PX),
        EdgeColour::Off => {}
    }
    seam_guided_blend(&mut raw, source, mask, SEAM_BLEND_RADIUS, SEAM_BLEND_EPSILON, feather_px);
    crate::inpaint::sharpen_inpainted(&mut raw, mask, sharpen);
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `img` with `f` applied to a copy.
    fn applied(img: &RgbaImage, f: impl FnOnce(&mut RgbaImage)) -> RgbaImage {
        let mut out = img.clone();
        f(&mut out);
        out
    }
    use image::{Luma, Rgba};

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(rgba))
    }

    /// The preview shows the grown region at full strength and the
    /// blend band beyond it at half, exactly as far as the erase reaches.
    #[test]
    fn fill_preview_shows_the_grown_region_and_the_blend_band() {
        use crate::selection::{MaskArtifact, FULL};
        let mut cells = vec![0i8; 11 * 11];
        cells[5 * 11 + 5] = FULL;
        let painted = MaskArtifact::from_cells(11, 11, cells);
        let shown = fill_preview(&painted, 1.0, 1.0);
        let at = |x: usize, y: usize| shown.cells()[y * 11 + x];
        assert_eq!(at(5, 5), FULL);
        assert_eq!(at(4, 5), FULL, "grown by one");
        assert_eq!(at(3, 5), FULL / 2, "the blend band");
        assert_eq!(at(2, 5), 0, "beyond the erase's reach");
        let reach = fill_region(&painted.region_mask(11, 11), 1.0, 1.0);
        let touched: Vec<bool> = reach.as_raw().iter().map(|&v| v >= 128).collect();
        let previewed: Vec<bool> = shown.cells().iter().map(|&v| v != 0).collect();
        assert_eq!(touched, previewed, "the preview covers what the erase touches, no more");
    }
    fn empty_mask(w: u32, h: u32) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([0]))
    }

    /// A fill bordered by two different surroundings meets each along
    /// its edge; one mean shift for the region leaves both sides off.
    #[test]
    fn colour_matched_along_the_edge_meets_each_side() {
        let (w, h) = (80u32, 40u32);
        let src = RgbaImage::from_fn(w, h, |x, _| if x < w / 2 { Rgba([200, 40, 40, 255]) } else { Rgba([40, 40, 200, 255]) });
        let mask = GrayImage::from_fn(w, h, |x, y| Luma([if (10..70).contains(&x) && (10..30).contains(&y) { 255 } else { 0 }]));
        let fill = RgbaImage::from_fn(w, h, |x, y| if mask.get_pixel(x, y).0[0] >= 128 { Rgba([120, 120, 120, 255]) } else { *src.get_pixel(x, y) });
        let along = applied(&fill, |f| match_colour_along_edge(f, &src, &mask, COLOR_MATCH_RING_PX, 6));
        let whole = applied(&fill, |f| color_match_inpainted(f, &src, &mask, COLOR_MATCH_RING_PX));
        let off = |img: &RgbaImage, x, want: [u8; 4]| {
            let p = img.get_pixel(x, 11).0;
            (0..3).map(|c| (p[c] as i32 - want[c] as i32).abs()).sum::<i32>()
        };
        let (red, blue) = ([200, 40, 40, 255], [40, 40, 200, 255]);
        assert!(off(&along, 12, red) < 30 && off(&along, 67, blue) < 30, "each side meets its own surroundings");
        assert!(off(&whole, 12, red) > 100, "one shift for the region leaves the red side off");
        assert_eq!(along.get_pixel(5, 5), src.get_pixel(5, 5), "the photo outside is untouched");
    }

    /// Push-pull keeps every known sample and fills between them smoothly.
    #[test]
    fn push_pull_keeps_known_values_and_fills_between() {
        let (w, h) = (9usize, 1usize);
        let mut value = vec![0.0; w];
        let mut weight = vec![0.0; w];
        (value[0], weight[0]) = (0.0, 1.0);
        (value[8], weight[8]) = (8.0, 1.0);
        let out = push_pull(&value, &weight, w, h);
        assert!((out[0] - 0.0).abs() < 1e-4 && (out[8] - 8.0).abs() < 1e-4, "{out:?}");
        assert!(out.windows(2).all(|p| p[1] >= p[0] - 1e-4), "monotone between: {out:?}");
    }

    /// With nothing to match or blend against (no mask, a mask of another
    /// size, or a mask over all of it so no photo rings it), the fill
    /// stays as it is.
    #[test]
    fn colour_match_and_seam_blend_without_a_ring_leave_the_fill() {
        let src = solid(64, 64, [200, 200, 200, 255]);
        let inp = solid(64, 64, [50, 50, 50, 255]);
        for mask in [empty_mask(64, 64), empty_mask(32, 32), GrayImage::from_pixel(64, 64, Luma([255]))] {
            assert_eq!(applied(&inp, |o| color_match_inpainted(o, &src, &mask, 8)), inp, "{:?}", mask.dimensions());
        }
        let out = applied(&inp, |o| seam_guided_blend(o, &src, &empty_mask(64, 64), 4, 1e-3, 8.0));
        assert_eq!(out, inp, "seam blend, no mask");
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
        let out = applied(&inp, |o| color_match_inpainted(o, &src, &mask, 8));
        let p = out.get_pixel(32, 32);
        assert!(
            p[0] >= 180 && p[0] <= 220,
            "expected ~200 after color match, got {}",
            p[0],
        );
        let outside = out.get_pixel(0, 0);
        assert_eq!(outside[0], 200);
    }

    /// Bbox-crop contract: pixels outside the mask's expanded bbox must
    /// stay bit-exact identical to the inpaint input.
    #[test]
    fn color_match_outside_bbox_is_bit_exact() {
        let src = solid(512, 512, [200, 150, 100, 255]);
        let inp = solid(512, 512, [60, 90, 120, 255]);
        let mut mask = empty_mask(512, 512);
        for y in 200..232 {
            for x in 200..232 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let ring_px = 8;
        let bbox = mask_bbox(&mask, 128, ring_px).unwrap();
        let out = applied(&inp, |o| color_match_inpainted(o, &src, &mask, ring_px));
        for y in 0..512 {
            for x in 0..512 {
                let inside = x >= bbox.x && x < bbox.x + bbox.w
                    && y >= bbox.y && y < bbox.y + bbox.h;
                if !inside {
                    assert_eq!(
                        out.get_pixel(x, y),
                        inp.get_pixel(x, y),
                        "outside-bbox pixel ({x}, {y}) drifted",
                    );
                }
            }
        }
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
        let out = applied(&inp, |o| seam_guided_blend(o, &src, &mask, 4, 1e-3, 8.0));
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

    /// A nearly flat photo (a night road, snow: a few levels of noise)
    /// under an erase the size of a person: the seam must stay between the
    /// photo and the fill, not blow up to pure 0 / 255 colours along the
    /// edge.
    #[test]
    fn seam_blend_stays_in_range_on_a_large_nearly_flat_erase() {
        let n = 1400u32;
        let src = RgbaImage::from_fn(n, n, |x, y| {
            let v = ((x * 7 + y * 13) % 5) as u8;
            Rgba([190 + v, 160 + v, 90 + v, 255])
        });
        let inp = RgbaImage::from_fn(n, n, |x, y| {
            let v = ((x * 3 + y * 11) % 5) as u8;
            Rgba([185 + v, 155 + v, 85 + v, 255])
        });
        let mask = GrayImage::from_fn(n, n, |x, y| {
            let (dx, dy) = (x as f32 - 700.0, y as f32 - 700.0);
            Luma([if dx * dx + dy * dy < 600.0 * 600.0 { 255 } else { 0 }])
        });
        let out = applied(&inp, |o| seam_guided_blend(o, &src, &mask, SEAM_BLEND_RADIUS, SEAM_BLEND_EPSILON, DEFAULT_EDGE_BLEND_PX));
        for (x, y, px) in out.enumerate_pixels() {
            for (c, lo, hi) in [(0, 180, 200), (1, 150, 170), (2, 80, 100)] {
                assert!((lo..=hi).contains(&px[c]), "({x}, {y}) channel {c} = {}", px[c]);
            }
        }
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
        let out = applied(&inp, |o| seam_guided_blend(o, &src, &mask, 4, 1e-3, 8.0));
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

    /// Edge blend 0 is no blend at all: the fill is only colour-matched.
    #[test]
    fn edge_blend_zero_leaves_the_fill_unblended() {
        let src = solid(48, 48, [90, 120, 150, 255]);
        let mut raw = solid(48, 48, [90, 120, 150, 255]);
        let mut mask = empty_mask(48, 48);
        for y in 16..32 { for x in 16..32 {
            mask.put_pixel(x, y, Luma([255]));
            raw.put_pixel(x, y, Rgba([((x * 7) % 255) as u8, 40, 200, 255]));
        } }
        let mut matched = raw.clone();
        color_match_inpainted(&mut matched, &src, &mask, COLOR_MATCH_RING_PX);
        assert_eq!(finalize_inpaint(raw, &src, &mask, 0.0, 0.0, EdgeColour::WholeArea), matched);
    }

    /// The fill region is the painted one grown by Expand and then by the
    /// Edge blend ring.
    #[test]
    fn the_fill_region_grows_by_expand_plus_edge_blend() {
        let mut painted = empty_mask(40, 40);
        painted.put_pixel(20, 20, Luma([255]));
        let region = fill_region(&painted, 2.0, 3.0);
        assert_eq!(region.get_pixel(25, 20)[0], 255, "5 px out is filled");
        assert_eq!(region.get_pixel(26, 20)[0], 0, "6 px out is not");
        assert_eq!(fill_region(&painted, 0.0, 0.0), painted);
    }

    /// The blend fades the fill into the photo over the Edge blend ring,
    /// which lies outside what was painted: the erased object never
    /// shows through, however wide the blend.
    #[test]
    fn a_wide_edge_blend_never_brings_back_the_erased_object() {
        let mut src = solid(64, 64, [100, 100, 100, 255]);
        let mut painted = empty_mask(64, 64);
        for y in 24..40 { for x in 24..40 {
            src.put_pixel(x, y, Rgba([250, 250, 250, 255]));
            painted.put_pixel(x, y, Luma([255]));
        } }
        let raw = solid(64, 64, [100, 100, 100, 255]);
        let region = fill_region(&painted, 0.0, 12.0);
        for colour in [EdgeColour::AlongEdge, EdgeColour::WholeArea, EdgeColour::Off] {
            let out = finalize_inpaint(raw.clone(), &src, &region, 12.0, 0.0, colour);
            let brightest = (24..40).flat_map(|y| (24..40).map(move |x| (x, y)))
                .map(|(x, y)| out.get_pixel(x, y)[0])
                .max().unwrap();
            assert!(brightest <= 110, "{colour:?}: object rim ghosted back: {brightest}");
        }
    }
}
