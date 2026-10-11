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

/// The whole texture, for `Painter::image`.
pub(crate) const UNIT_UV: Rect = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

const GLOW: Color32 = theme::GLOW;

/// Cells per side of an effect mesh; plenty for a gradient this soft.
const GRID: usize = 24;

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
    x * x * (3.0 - 2.0 * x)
}

/// How far a reveal has come at `p` (0..1) for the point `(u, v)`: the new
/// image arrives corner to corner along a soft edge. Returns the new
/// image's opacity and the glow riding the edge.
fn reveal_at(p: f32, u: f32, v: f32) -> (f32, f32) {
    const FEATHER: f32 = 0.35;
    let eased = p * p * (3.0 - 2.0 * p);
    let front = eased * (1.0 + FEATHER);
    let opacity = ((front - (u + v) * 0.5) / FEATHER).clamp(0.0, 1.0);
    let glow = (1.0 - (2.0 * opacity - 1.0).abs()) * (1.0 - eased);
    (opacity, glow)
}

/// A `GRID × GRID` mesh over `rect` with `texture` mapped across it and
/// each vertex coloured by `color(u, v)`.
fn paint_mesh(painter: &Painter, rect: Rect, texture: TextureId, color: impl Fn(f32, f32) -> Color32) {
    let mut mesh = Mesh::with_texture(texture);
    mesh.reserve_vertices((GRID + 1) * (GRID + 1));
    mesh.reserve_triangles(2 * GRID * GRID);
    let n = GRID as u32;
    for j in 0..=GRID {
        for i in 0..=GRID {
            let (u, v) = (i as f32 / GRID as f32, j as f32 / GRID as f32);
            let uv = if texture == TextureId::default() { WHITE_UV } else { Pos2::new(u, v) };
            mesh.vertices.push(Vertex { pos: rect.lerp_inside(egui::vec2(u, v)), uv, color: color(u, v) });
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
    paint_mesh(painter, rect, TextureId::default(), |u, v| {
        GLOW.gamma_multiply(0.06 + 0.16 * s + 0.22 * sheen(t, u, v))
    });
}

/// The same light shaped by `mask` (a selection overlay texture): the
/// brushed region breathes and the sheen stays inside it.
pub(crate) fn paint_working_masked(painter: &Painter, rect: Rect, mask: TextureId, t: f32) {
    let s = breath(t);
    paint_mesh(painter, rect, mask, |u, v| Color32::WHITE.gamma_multiply(0.35 + 0.35 * s + 0.6 * sheen(t, u, v)));
}

/// `to` arriving where `from` was, at `p` (0..1), along a soft edge with a
/// lavender glow; `from` leaves along the same edge, so where the new
/// image is transparent the old one is already gone when the edge passes.
pub(crate) fn paint_reveal(painter: &Painter, rect: Rect, from: TextureId, to: TextureId, p: f32) {
    paint_mesh(painter, rect, from, |u, v| Color32::WHITE.gamma_multiply(1.0 - reveal_at(p, u, v).0));
    paint_mesh(painter, rect, to, |u, v| Color32::WHITE.gamma_multiply(reveal_at(p, u, v).0));
    paint_mesh(painter, rect, TextureId::default(), |u, v| GLOW.gamma_multiply(0.45 * reveal_at(p, u, v).1));
}

/// The steady frame of the tile a model is working on.
pub(crate) fn paint_working_frame(painter: &Painter, rect: Rect) {
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.5, theme::ACCENT_BRIGHT), egui::StrokeKind::Inside);
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(reveal_at(0.0, 0.0, 0.0).0, 0.0);
        assert_eq!(reveal_at(1.0, 1.0, 1.0), (1.0, 0.0), "fully shown, no glow left");
        let mut last = 0.0;
        for i in 0..=20 {
            let (o, _) = reveal_at(i as f32 / 20.0, 0.6, 0.4);
            assert!(o >= last);
            last = o;
        }
        assert!(reveal_at(0.5, 0.0, 0.0).0 > reveal_at(0.5, 1.0, 1.0).0, "the top-left corner arrives first");
    }
}
