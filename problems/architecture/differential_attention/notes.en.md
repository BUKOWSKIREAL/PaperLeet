# Reference notes

Subtracting the logits before one softmax is not equivalent to subtracting two normalized maps. The scale uses the width of each split Q/K branch, not V's doubled width. Both branches obey the same causal dependency structure.

For probabilities `p` and upstream row gradient `u`, the softmax VJP is `p * (u - sum(p*u))`. The tests compute probabilities with scalar loops and then apply this derivative explicitly to each branch. The second branch contributes `−lambda` times its VJP; lambda itself receives `−sum(A2 * dL/dA)`.

The offset experiment follows the row-sum identity. Each attention map maps a constant value offset `c` to `c`, so their difference maps it to `(1−lambda)c`. This statement holds before the head normalization intentionally omitted from this module. A full Differential Transformer includes that normalization and the output projection.
