## 2025-05-15 - Sigmoid Monotonicity in Stats Pass
**Learning:** For neural network models using sigmoid activation (like BiRefNetLite), calculating min/max stats on raw logits before applying sigmoid to the extrema saves significant computation (~1M `exp()` calls per 1024² tensor). This works because sigmoid is a monotonic function.
**Action:** Always check if an expensive activation function is monotonic before including it in a reduction loop (stats/normalization pass).

## 2025-05-15 - Parallel Global Extrema with Rayon
**Learning:** Finding global min/max in parallel is most efficient using Rayon's `par_iter().fold(...).reduce(...)` pattern. This avoids synchronization overhead while saturating all available CPU cores.
**Action:** Use the fold-then-reduce pattern for all global reductions on large image/tensor buffers.
