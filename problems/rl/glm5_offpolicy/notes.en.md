# Off-policy correction notes

This is masking, not PPO clipping or clamping. A rejected token contributes zero for either advantage sign. Keep the original token count in the denominator: normalizing only survivors changes the objective.

Compute detached ratios and compare with 1-low and 1+high in the tensor dtype, using strict inequalities. Use torch.where to select zero for rejected weights, including infinity from detached exponentiation; multiplication by a zero mask would produce NaN. Exclude padded log-probabilities before the final product as well: zero times NaN is still NaN.

The score-function gradient with respect to each current log-probability is exactly -mask*weight*advantage/N. Numeric finite differences of this stop-gradient surrogate do not test the intended gradient, because they change the frozen weights between evaluations. Use this analytic gradient oracle instead. The paper/source boundary difference is documented in the statement.
