## 2025-05-15 - Sigmoid Monotonicity for Extremes
**Learning:** For models using sigmoid activation (like BiRefNet), computing min/max on the raw logits and then applying sigmoid only to the extrema is mathematically equivalent to computing sigmoid on every element and then finding min/max, because sigmoid is a monotonic function. This saves $O(N)$ expensive `exp()` calls in the min-max pass.
**Action:** Always check for monotonic transformations when computing ranges or extrema on large tensors.
