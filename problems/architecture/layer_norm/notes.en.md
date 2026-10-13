# Reference notes

Keep the reduced feature dimension so the mean and variance broadcast back to every feature. Compute population variance from centered values; the default sample-variance correction is wrong here and becomes undefined for one feature.

Apply affine scale and bias after normalization. Epsilon must be added to the variance, not to the standard deviation. For very small variance this difference changes both outputs and gradients substantially.

`post_norm_residual` reuses `layer_norm(x + sublayer_output, ...)`. Normalizing `x` before the sum is a different computation. The exercise disables dropout, which the Transformer paper otherwise places on sublayer outputs before residual addition. It does not implement modern pre-LN or RMSNorm variants.

The toy intentionally varies each token's offset and scale. It checks against scalar arithmetic and visualizes why per-token statistics remain independent of other tokens, with no claim about training stability or paper benchmark reproduction.
