# Implementation notes — LoRA Low-rank Updates and Frozen-base Forward

Implement `lora_delta(A,B,alpha)` and `lora_linear(x,W,A,B,alpha,bias)`. Shapes are `A[r,in]`, `B[out,r]`, `W[out,in]`, and `x[...,in]`. The delta is `(alpha/r) * B @ A`. The forward operation is the frozen base linear map plus this low-rank update. Base weight and optional bias must not receive gradients, even when their inputs have `requires_grad=True`; gradients to `x`, `A`, and `B` must remain intact.

The training scaffold initializes A randomly and B to zero. Initially, the adapter changes no outputs: B has a nonzero learning signal while A's first gradient is zero. Freezing the entire base computation with `no_grad()` would incorrectly remove part of the gradient to x. Merge by adding the returned delta to W; the exercise compares merged and unmerged outputs. There is no dropout in this module.

The toy freezes a 6-by-8 map and fits a known rank-two change using 28 adapter parameters. The platform supplies input data, optimizer and 300 CPU training steps. Passing means recovering this controlled target, not reproducing a full language-model benchmark.

## Deliberately wrong variants

- Forget division by rank.
- Accidentally train the base weight.
- Detach the low-rank update.
- Remove the base path gradient to input.

Each mutant differs from the reference in a small, intentional change. T2 includes independent formula or gradient expectations, so equality with the reference is not the only acceptance evidence. T3 produces artifacts from the supplied implementation.

Source: [2106.09685v2](https://arxiv.org/abs/2106.09685v2); [Official Microsoft LoRA implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py). Toy provenance: PaperLeet rank-two synthetic regression, not a language-model benchmark.
