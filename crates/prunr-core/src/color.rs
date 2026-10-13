//! sRGB ↔ CIE Lab (D65) for pixel math that works on lightness and
//! colour apart: Reinhard colour match and colorization.

// ── sRGB ↔ linear helpers ──────────────────────────────────────────────────

/// `srgb_to_linear` of each 8-bit value, so per-pixel conversions skip
/// the `powf`. Built from the same expression, so results are identical.
fn srgb_u8_to_linear(v: u8) -> f32 {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))[v as usize]
}

/// sRGB encoded → linear-light (IEC 61966-2-1).
#[inline(always)]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-light → sRGB encoded (IEC 61966-2-1 inverse).
#[inline(always)]
fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ── CIE XYZ ↔ Lab (D65) ──────────────────────────────────────────────────

/// Linear sRGB → CIE XYZ D65.
/// Bradford-adapted D65 matrix from IEC 61966-2-1.
#[inline(always)]
fn linear_rgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;
    (x, y, z)
}

/// CIE XYZ D65 → linear sRGB.
#[inline(always)]
fn xyz_to_linear_rgb(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r =  3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266 * x + 1.876_010_8 * y + 0.041_556_0 * z;
    let b =  0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    (r, g, b)
}

const D65_X: f32 = 0.950_489;
const D65_Y: f32 = 1.000_000;
const D65_Z: f32 = 1.088_84;

#[inline(always)]
fn lab_f(t: f32) -> f32 {
    // CIE standard cube-root nonlinearity
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA;           // ≈ 0.008856
    const INV3DELTA2: f32 = 1.0 / (3.0 * DELTA * DELTA); // ≈ 7.787
    if t > DELTA3 {
        t.cbrt()
    } else {
        INV3DELTA2 * t + (4.0 / 29.0)
    }
}

#[inline(always)]
fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// sRGB u8 pixel → CIE Lab (L in [0,100], a/b unbounded typical ±128).
#[inline(always)]
pub(crate) fn srgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (x, y, z) = linear_rgb_to_xyz(srgb_u8_to_linear(r), srgb_u8_to_linear(g), srgb_u8_to_linear(b));
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_lab = 200.0 * (fy - fz);
    (l, a, b_lab)
}

/// The `L` of `srgb_to_lab`, without the colour channels' work.
#[inline(always)]
pub(crate) fn srgb_lightness(r: u8, g: u8, b: u8) -> f32 {
    let (_, y, _) = linear_rgb_to_xyz(srgb_u8_to_linear(r), srgb_u8_to_linear(g), srgb_u8_to_linear(b));
    116.0 * lab_f(y / D65_Y) - 16.0
}

/// CIE Lab → sRGB u8, clamped.
#[inline(always)]
pub(crate) fn lab_to_srgb(l: f32, a: f32, b_lab: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_lab / 200.0;
    let x = lab_f_inv(fx) * D65_X;
    let y = lab_f_inv(fy) * D65_Y;
    let z = lab_f_inv(fz) * D65_Z;
    let (lr, lg, lb) = xyz_to_linear_rgb(x, y, z);
    let r = (linear_to_srgb(lr) * 255.0).clamp(0.0, 255.0) as u8;
    let g = (linear_to_srgb(lg) * 255.0).clamp(0.0, 255.0) as u8;
    let b = (linear_to_srgb(lb) * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lightness_is_the_l_of_the_full_conversion() {
        for (r, g, b) in [(0, 0, 0), (255, 255, 255), (200, 40, 40), (12, 130, 250), (90, 160, 210)] {
            assert_eq!(srgb_lightness(r, g, b), srgb_to_lab(r, g, b).0, "{r},{g},{b}");
        }
    }
}
