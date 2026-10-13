# Reference notes

The residual lives at the original latent resolution. Resize it only to decide each scale's indices; lift selected codes back to full resolution before subtracting. A shared codebook does not mean every scale sees the same feature map.

The reference computes squared distances by broadcasting. Algebraic norms/dot products are also valid within tolerance; ties must still select the first code. Channel projection happens after code selection. Its effect is part of both the next residual and cumulative reconstruction.

Tests include a hand-computed two-scale example, a scalar nearest-neighbor oracle with nonuniform resolutions and nonidentity channel maps, independently constructed decoder expectations, frozen-output semantics and per-example independence. With a dense scalar codebook and identity last projection, the last-scale error is bounded by half a quantization interval when the residual remains in range.

The toy's fixed codebook happens to improve the selected latent. Monotonic error reduction is not a general invariant of learned projected residual quantization. Autograd of a trainable VQ-VAE is a separate module; this exercise implements deterministic frozen token generation.
