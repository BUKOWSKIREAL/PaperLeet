# Reference notes

The invariant is `l = sum(exp(scores_seen-m))` and `u = sum(exp(scores_seen-m)*values)`. Both depend on the same running maximum. A new larger maximum changes their common scale, not the final ratio.

Using only the new block's maximum can remain algebraically correct on mild inputs while overflowing on a large descending score transition. The extreme-score test explicitly covers this. Detaching the numerator can preserve forward results but corrupt earlier value gradients; the independent gradient test detects it.

The provided traversal is a readable CPU implementation of the recurrence. It is not a fused FlashAttention kernel, and ordinary autograd is not the paper's recomputed backward algorithm.
