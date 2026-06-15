## 2025-05-14 - [Apply Edge Shift Optimization]
**Learning:** Naive 3x3 morphological filters with per-pixel clamping and branching are extremely slow on large images. Hoisting branches and using an interior fast-path with unrolled kernels provides over 10x speedup. Fixed-point integer math for blending also avoids unnecessary f32 conversions.
**Action:** Always prefer unrolled interior fast-paths for spatial filters (min, max, box) to eliminate boundary checks in the hot path.
