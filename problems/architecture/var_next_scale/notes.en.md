# Reference notes

Keep the cumulative feature map at the **largest** resolution. Add one lifted residual map, area-resize the sum to the next resolution, and only then append that input block. Replacing the cumulative sum with the latest map drops coarse context. Using the next target map in that addition leaks the answer.

A block-causal mask includes the full diagonal block, not only its lower triangle. Its diagonal blocks have sizes `n_k² × n_k²`; resolutions are not token counts. The boolean convention here is `True = visible`, so pass its inverse to `masked_fill(..., -inf)`.

Tests use an independent scalar lookup and channel-sum oracle, exact visibility matrices, target perturbations and separate-example checks. T3 composes your two functions with actual softmax attention; the output effect of each target perturbation must be strictly directed toward finer scales.

Scope: frozen tokenizer target preparation, shared Euclidean codebook, supplied channel projections. The full paper's learned spatial projection, encoder/decoder and Transformer training are intentionally not approximated by performance claims.
