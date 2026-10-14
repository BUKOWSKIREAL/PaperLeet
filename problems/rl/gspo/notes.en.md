# Implementation notes — GSPO Sequence Ratios and Clipping

`logp`, `old_logp`, and binary `mask` have shape `[B,L]`; `advantages` has shape `[B]`. First sum the valid log-ratios and divide by valid length, then exponentiate. Clip that single ratio per response, multiply by the detached advantage, take the smaller surrogate, and negate the batch mean.

The old policy and advantages are frozen. A response with no valid tokens has ratio 1 and contributes zero to the batch mean. Padding must be neutralized **before** exponentiation, including extreme padded scores. Duplicating tokens within a response leaves the loss unchanged. This module implements GSPO, not GSPO-token. The default `.2` clip is an inspectable teaching setting; it is not a claim about Qwen's production hyperparameter.

Try token ratios `[2, 0.5]`: their geometric ratio is 1, so both valid log-probabilities receive the same gradient. A positive-advantage response with ratio 1.5 clips; a negative-advantage response with ratio 0.5 also clips. The toy saves the submitted implementation's ratios and gradient matrix.

## Deliberately wrong variants

- Average token ratios instead of the geometric sequence ratio.
- Omit response-length normalization.
- Backpropagate into the behavior policy.
- Maximize instead of minimize clipped surrogate.

Each mutant differs from the reference in a small, intentional change. T2 includes independent formula or gradient expectations, so equality with the reference is not the only acceptance evidence. T3 produces artifacts from the supplied implementation.

Source: [2507.18071v1](https://arxiv.org/html/2507.18071v1#S4.SS1); [Author explanation (no author training-code release verified)](https://qwenlm.github.io/blog/gspo/). Toy provenance: Analytic three-response diagnostic written for LeetPaper; not Qwen training results.
