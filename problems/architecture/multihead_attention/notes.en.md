# Reference notes

The packed projection matrices store each head's features in consecutive columns. For a head width `d`, head `i` reads `w[:, i*d:(i+1)*d]`. The independent oracle constructs those projections one head at a time, avoiding the packed reshape used by the reference.

For cross-attention, derive query length from `query` and key length from `key`. Attention probabilities normalize across keys, but the scale uses a single head's width. The provided kernel already implements this convention.

After attention, `[B,H,Q,d]` must become `[B,Q,H,d]` before flattening to `[B,Q,D]`. Reshape is safer than view for transposed, noncontiguous tensors. The final `@ w_o` mixes head outputs; concatenation alone is incomplete.

The original paper formula has no projection biases. This exercise follows it and omits dropout to make results deterministic. All projection matrices and q/k/v remain differentiable. All-masked rows are explicitly zero, a practical extension beyond the paper's displayed formula.

The toy's head preferences follow analytic softmax probabilities proportional to `[exp(5/sqrt(2)),1,1]` and its permutations. It demonstrates independent feature subspaces, not learned specialization or the original translation benchmark.
