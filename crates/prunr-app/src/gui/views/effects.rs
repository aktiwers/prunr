//! Light effects over the image for work under way and work finished: a
//! slow breath, a sheen sweeping across, and the reveal of a new result.
//! Each is a mesh with per-vertex colour, so a soft gradient costs one
//! draw call and no texture.

use std::f32::consts::TAU;

use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{Color32, Painter, Pos2, Rect, TextureId};

use crate::gui::theme;

/// Seconds a new result takes to reveal.
pub(crate) const REVEAL_SECS: f64 = 0.6;

/// Whether keeping an image of `held_bytes` on screen through a reveal
/// fits in `available_bytes` of free memory. A reveal keeps the old image
/// alongside the new one; when that would crowd memory the new image
/// simply replaces the old.
pub(crate) fn reveal_fits(held_bytes: usize, available_bytes: u64) -> bool {
    const SHARE_OF_FREE: u64 = 4;
    (held_bytes as u64).saturating_mul(SHARE_OF_FREE) <= available_bytes
}

/// Seconds the subject outline takes to fade in.
pub(crate) const SUBJECT_FADE_SECS: f32 = 0.35;

/// Half-widths, in subject-hint pixels, of the outline's line and its glow.
const OUTLINE_LINE_PX: f32 = 1.5;
const OUTLINE_GLOW_PX: f32 = 6.0;

/// Seconds a found face's outline takes to come in.
pub(crate) const FACE_FADE_SECS: f32 = 0.25;
/// Seconds its arrival lasts: the pop, the bounce and the glow settling.
pub(crate) const FACE_ARRIVAL_SECS: f32 = 2.0;
/// The glow a found face keeps once it has arrived, and its brightest.
const FACE_GLOW_REST: f32 = 0.3;
const FACE_GLOW_PEAK: f32 = 1.8;
/// Width of a glow ring's line, and how far its glow reaches either side,
/// in screen points.
const RING_LINE_PX: f32 = 1.5;
const RING_GLOW_PX: f32 = 7.0;

/// A found face's outline `since` seconds after the faces were found, as
/// (size, glow): it pops in at 80 %, bounces past full size once and
/// settles, while its glow swells bright and eases back to the rest level.
pub(crate) fn face_arrival(since: f32) -> (f32, f32) {
    if !(0.0..FACE_ARRIVAL_SECS).contains(&since) {
        return (1.0, FACE_GLOW_REST);
    }
    // A damped spring from 80 %: past 100 % once, back, settled.
    let size = 1.0 - 0.2 * (-since / 0.18).exp() * (std::f32::consts::TAU * since / 0.45).cos();
    // t·e^(1−t): 0 at the start, 1 at a quarter second, gone by the end.
    let t = since / 0.25;
    let swell = t * (1.0 - t).exp();
    (size, FACE_GLOW_REST + (FACE_GLOW_PEAK - FACE_GLOW_REST) * swell)
}

/// A closed outline through `points` (the first not repeated) in the
/// accent purple, as the subject outline is drawn: a crisp line and a
/// glow either side that falls off with the square of the distance.
/// `opacity` fades the whole; `glow` (0-1) is how strongly, and how far,
/// the halo shows.
pub(crate) fn paint_glow_ring(painter: &Painter, points: &[Pos2], opacity: f32, glow: f32) {
    // Across the ring, outer edge to outer edge: (offset in reaches,
    // strength). The middle pair samples (1 - d)² so the falloff is smooth.
    const PROFILE: [(f32, f32); 5] = [(-1.0, 0.0), (-0.4, 0.36), (0.0, 1.0), (0.4, 0.36), (1.0, 0.0)];
    let n = points.len();
    if n < 3 || opacity <= 0.0 {
        return;
    }
    let reach = RING_GLOW_PX * (0.7 + 0.6 * glow);
    let halo = (0.5 * glow).min(1.0) * opacity;
    let mut mesh = Mesh::default();
    for i in 0..n {
        let along = (points[(i + 1) % n] - points[(i + n - 1) % n]).normalized();
        let out = egui::vec2(-along.y, along.x) * reach;
        for (d, k) in PROFILE {
            mesh.colored_vertex(points[i] + out * d, theme::ACCENT_BRIGHT.gamma_multiply(halo * k));
        }
    }
    let (n, across) = (n as u32, PROFILE.len() as u32);
    for i in 0..n {
        let (a, b) = (across * i, across * ((i + 1) % n));
        for band in 0..across - 1 {
            mesh.add_triangle(a + band, a + band + 1, b + band + 1);
            mesh.add_triangle(a + band, b + band, b + band + 1);
        }
    }
    painter.add(mesh);
    painter.add(egui::Shape::closed_line(points.to_vec(), egui::Stroke::new(RING_LINE_PX, theme::ACCENT_BRIGHT.gamma_multiply(0.95 * opacity))));
}

/// The whole texture, for `Painter::image`.
pub(crate) const UNIT_UV: Rect = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

const GLOW: Color32 = theme::GLOW;

/// Cells per side of an effect mesh; plenty for a gradient this soft.
pub(crate) const GRID: usize = 24;

/// 0 → 1 → 0 over one slow breath, for anything that shows work under way.
pub(crate) fn breath(t: f32) -> f32 {
    const PERIOD_SECS: f32 = 4.5;
    0.5 - 0.5 * (t * TAU / PERIOD_SECS).cos()
}

/// A soft band of light crossing the area corner to corner every few
/// seconds, at `(u, v)` in 0..1.
fn sheen(t: f32, u: f32, v: f32) -> f32 {
    const PERIOD_SECS: f32 = 2.8;
    const HALF_WIDTH: f32 = 0.18;
    let front = (t / PERIOD_SECS).fract() * 1.8 - 0.4;
    let d = ((u + v) * 0.5 - front).abs() / HALF_WIDTH;
    let x = (1.0 - d).max(0.0);
    prunr_core::math::smoothstep(x)
}

/// How far a reveal has come at `p` (0..1) for a point whose turn comes at
/// `order` (0 first, 1 last): the new image arrives along a soft edge.
/// Returns the new image's opacity and the glow riding the edge.
fn reveal_at(p: f32, order: f32) -> (f32, f32) {
    const FEATHER: f32 = 0.35;
    let eased = prunr_core::math::smoothstep(p);
    let front = eased * (1.0 + FEATHER);
    let opacity = ((front - order) / FEATHER).clamp(0.0, 1.0);
    let glow = (1.0 - (2.0 * opacity - 1.0).abs()) * (1.0 - eased);
    (opacity, glow)
}

/// A `GRID × GRID` mesh over `rect` with `texture` mapped across it and
/// each vertex coloured by `color(u, v, vertex index)`.
pub(crate) fn paint_mesh(painter: &Painter, rect: Rect, texture: TextureId, color: impl Fn(f32, f32, usize) -> Color32) {
    paint_mesh_placed(painter, rect, texture, |p| p, color);
}

/// `rect` coloured by `color(u, v)` with its corners rounded to `radius`:
/// the grid's corner vertices are pulled onto the arcs.
pub(crate) fn paint_rounded_mesh(painter: &Painter, rect: Rect, radius: f32, color: impl Fn(f32, f32) -> Color32) {
    paint_mesh_placed(painter, rect, TextureId::default(), |p| round_into(rect, radius, p), |u, v, _| color(u, v));
}

/// `p` moved onto the rounded rect's edge when it lies outside a corner arc.
fn round_into(rect: Rect, radius: f32, p: Pos2) -> Pos2 {
    let center = Pos2::new(
        p.x.clamp(rect.left() + radius, rect.right() - radius),
        p.y.clamp(rect.top() + radius, rect.bottom() - radius),
    );
    let d = p - center;
    let len = d.length();
    if len > radius { center + d * (radius / len) } else { p }
}

fn paint_mesh_placed(
    painter: &Painter,
    rect: Rect,
    texture: TextureId,
    place: impl Fn(Pos2) -> Pos2,
    color: impl Fn(f32, f32, usize) -> Color32,
) {
    let mut mesh = Mesh::with_texture(texture);
    mesh.reserve_vertices((GRID + 1) * (GRID + 1));
    mesh.reserve_triangles(2 * GRID * GRID);
    let n = GRID as u32;
    for j in 0..=GRID {
        for i in 0..=GRID {
            let (u, v) = (i as f32 / GRID as f32, j as f32 / GRID as f32);
            let uv = if texture == TextureId::default() { WHITE_UV } else { Pos2::new(u, v) };
            let index = mesh.vertices.len();
            mesh.vertices.push(Vertex { pos: place(rect.lerp_inside(egui::vec2(u, v))), uv, color: color(u, v, index) });
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            mesh.add_triangle(a, a + 1, a + n + 1);
            mesh.add_triangle(a + 1, a + n + 2, a + n + 1);
        }
    }
    painter.add(mesh);
}

/// The content of `rect` breathing in soft light with a sheen crossing it:
/// what a region looks like while a model works on it.
pub(crate) fn paint_working(painter: &Painter, rect: Rect, t: f32) {
    let s = breath(t);
    paint_mesh(painter, rect, TextureId::default(), |u, v, _| {
        GLOW.gamma_multiply(0.06 + 0.16 * s + 0.22 * sheen(t, u, v))
    });
}

/// The same light shaped by `mask` (a selection overlay texture): the
/// brushed region breathes and the sheen stays inside it.
pub(crate) fn paint_working_masked(painter: &Painter, rect: Rect, mask: TextureId, t: f32) {
    let s = breath(t);
    paint_mesh(painter, rect, mask, |u, v, _| Color32::WHITE.gamma_multiply(0.35 + 0.35 * s + 0.6 * sheen(t, u, v)));
}

/// `to` arriving where `from` was, at `p` (0..1), along a soft edge with a
/// purple glow; `from` leaves along the same edge, so where the new
/// image is transparent the old one is already gone when the edge passes.
/// `order` (one value per mesh vertex, see `subject_glow`) sets where it
/// starts; without one it runs corner to corner.
pub(crate) fn paint_reveal(painter: &Painter, rect: Rect, from: TextureId, to: TextureId, p: f32, order: Option<&[f32]>) {
    let at = |u: f32, v: f32, i: usize| reveal_at(p, order.and_then(|o| o.get(i).copied()).unwrap_or((u + v) * 0.5));
    paint_mesh(painter, rect, from, |u, v, i| Color32::WHITE.gamma_multiply(1.0 - at(u, v, i).0));
    paint_mesh(painter, rect, to, |u, v, i| Color32::WHITE.gamma_multiply(at(u, v, i).0));
    paint_mesh(painter, rect, TextureId::default(), |u, v, i| theme::ACCENT_BRIGHT.gamma_multiply(0.25 * at(u, v, i).1));
}

/// A glowing outline around the subject in `hint`, and the order a reveal
/// over it follows: per mesh vertex, 0 on the subject rising to 1 at the
/// farthest point. `None` when the hint holds no subject.
pub(crate) fn subject_glow(hint: &prunr_core::SubjectHint) -> Option<(egui::ColorImage, Vec<f32>)> {
    let (w, h) = (hint.width as usize, hint.height as usize);
    let inside: Vec<bool> = hint.coverage.iter().map(|&c| c >= 128).collect();
    if inside.len() != w * h || !inside.iter().any(|&i| i) {
        return None;
    }
    let to_subject = prunr_core::math::distance_to_nearest(&inside, w, h);
    let to_background = prunr_core::math::distance_to_nearest(&inside.iter().map(|&i| !i).collect::<Vec<_>>(), w, h);
    let pixels = (0..w * h)
        .map(|i| {
            let d = if inside[i] { to_background[i] } else { to_subject[i] };
            let line = (1.0 - d / OUTLINE_LINE_PX).max(0.0);
            let glow = 0.5 * (1.0 - d / OUTLINE_GLOW_PX).max(0.0).powi(2);
            theme::ACCENT_BRIGHT.gamma_multiply(line.max(glow))
        })
        .collect();
    let far = to_subject.iter().copied().fold(1.0, f32::max);
    let order = (0..=GRID)
        .flat_map(|j| (0..=GRID).map(move |i| (i, j)))
        .map(|(i, j)| {
            let x = (i * (w - 1)) / GRID;
            let y = (j * (h - 1)) / GRID;
            to_subject[y * w + x] / far
        })
        .collect();
    Some((egui::ColorImage { size: [w, h], pixels, source_size: egui::vec2(w as f32, h as f32) }, order))
}

/// The steady frame of the tile a model is working on.
pub(crate) fn paint_working_frame(painter: &Painter, rect: Rect) {
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.5, theme::ACCENT_BRIGHT), egui::StrokeKind::Inside);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A found face pops in small, bounces past full size once and
    /// settles; its glow swells past the rest level and eases back.
    #[test]
    fn a_found_face_pops_in_bounces_and_rests() {
        let (start, glow0) = face_arrival(0.0);
        assert!((start - 0.8).abs() < 1e-4 && (glow0 - FACE_GLOW_REST).abs() < 1e-4);
        let (over, peak) = face_arrival(0.25);
        assert!(over > 1.02, "past full size: {over}");
        assert!((peak - FACE_GLOW_PEAK).abs() < 1e-4, "brightest a quarter second in");
        let (late, glow) = face_arrival(FACE_ARRIVAL_SECS - 0.01);
        assert!((late - 1.0).abs() < 1e-3 && (glow - FACE_GLOW_REST).abs() < 0.02, "settled");
        assert_eq!(face_arrival(f32::MAX), (1.0, FACE_GLOW_REST), "animations off: at rest from the start");
    }

    #[test]
    fn a_rounded_mesh_pulls_only_its_corners_in() {
        let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(60.0, 40.0));
        let corner = round_into(rect, 10.0, Pos2::ZERO);
        assert!(((corner - Pos2::new(10.0, 10.0)).length() - 10.0).abs() < 1e-4, "the corner lands on its arc");
        for p in [Pos2::new(30.0, 0.0), Pos2::new(0.0, 20.0), Pos2::new(30.0, 20.0), Pos2::new(60.0, 40.0 - 10.0)] {
            assert_eq!(round_into(rect, 10.0, p), p, "{p:?} stays put");
        }
    }

    #[test]
    fn a_breath_rises_and_settles_between_zero_and_one() {
        assert!(breath(0.0).abs() < 1e-6);
        assert!((breath(2.25) - 1.0).abs() < 1e-4);
        assert!((0..100).all(|i| (0.0..=1.0).contains(&breath(i as f32 * 0.37))));
    }

    #[test]
    fn the_sheen_is_a_band_that_crosses_the_area() {
        let peak = (0..280).map(|i| sheen(i as f32 * 0.01, 0.5, 0.5)).fold(0.0, f32::max);
        assert!(peak > 0.99, "the band passes the middle");
        assert_eq!(sheen(0.0, 1.0, 1.0), 0.0, "and lights only where it is");
    }

    #[test]
    fn a_reveal_starts_hidden_ends_whole_and_never_steps_back() {
        assert_eq!(reveal_at(0.0, 0.0).0, 0.0);
        assert_eq!(reveal_at(1.0, 1.0), (1.0, 0.0), "fully shown, no glow left");
        let mut last = 0.0;
        for i in 0..=20 {
            let (o, _) = reveal_at(i as f32 / 20.0, 0.5);
            assert!(o >= last);
            last = o;
        }
        assert!(reveal_at(0.5, 0.0).0 > reveal_at(0.5, 1.0).0, "what comes first, arrives first");
    }

    #[test]
    fn a_reveal_is_skipped_when_the_old_image_would_crowd_free_memory() {
        let sixteen_k = 16_384 * 16_384 * 4;
        assert!(reveal_fits(sixteen_k, 8 << 30), "plenty free");
        assert!(!reveal_fits(sixteen_k, 2 << 30), "the old image would take half of what is free");
        assert!(reveal_fits(1024 * 1024 * 4, 64 << 20), "a small image always fits");
    }

    /// The outline sits on the subject's edge, and a reveal over it starts
    /// on the subject and spreads outward.
    #[test]
    fn the_subject_glow_rings_the_subject_and_orders_the_reveal_outward() {
        let (w, h) = (32usize, 32usize);
        let coverage = (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            if (12..20).contains(&x) && (12..20).contains(&y) { 255 } else { 0 }
        }).collect();
        let hint = prunr_core::SubjectHint { width: w as u32, height: h as u32, coverage };
        let (image, order) = subject_glow(&hint).expect("there is a subject");
        let alpha = |x: usize, y: usize| image.pixels[y * w + x].a();
        assert!(alpha(12, 16) > alpha(16, 16), "brightest on the edge, not in the middle");
        assert!(alpha(12, 16) > alpha(2, 2), "nor far outside");
        let centre = order[(GRID / 2) * (GRID + 1) + GRID / 2];
        assert_eq!(centre, 0.0, "the subject reveals first");
        assert!((order[0] - 1.0).abs() < 0.2, "a far corner reveals last");
        assert!(subject_glow(&prunr_core::SubjectHint { width: 2, height: 2, coverage: vec![0; 4] }).is_none());
    }
}
