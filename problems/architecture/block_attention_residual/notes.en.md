# Reference notes

Depth is axis 0. Each token computes its own distribution over the embedding, completed blocks and current partial block. Softmax across time instead would make different tokens compete and violate the operator.

RMSNorm has a learned multiplicative weight but no mean subtraction. Its denominator uses the mean of squared features plus eps, inside the square root. Normalize keys; multiplying normalized keys by attention weights loses the original value magnitudes.

A zero query produces uniform depth weights and the arithmetic mean of sources. A single source is returned unchanged, regardless of query. Permuting sources only permutes attention weights, leaving the output unchanged. These are useful independent checks beyond a reference comparison.

The optional partial source must be omitted at a block boundary. A supplied zero source is different: it participates in normalization. The caller supplies completed sums, so the function does not infer how many sublayers each block contains.

The teaching experiment sweeps a fixed query through three hand-chosen vectors. Its plotted error is measured from the submitted output; it is not a training-loss claim or a reproduction of K3 scaling gains.
