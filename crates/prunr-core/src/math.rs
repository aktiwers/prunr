//! Tiny shared math primitives.

/// Cubic Hermite smoothstep on `t ∈ [0, 1]`. Caller clamps the input.
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}
