# Reference notes

The paper's [Eq. (4) and Algorithm 2](https://arxiv.org/html/2603.19312v2) define a final-state latent cost and an elite Gaussian refit. We isolate a diagonal distribution and pin the population/MLE variance convention because generic variance functions often default to a K-1 denominator. The test includes K=1 and an explicit two-elite calculation.

A candidate may pass far from the goal during its rollout and still have low terminal cost. Summing all intermediate errors changes the objective; the prefix-invariance check distinguishes that mistake. Likewise, returning standard deviation produces a plausible positive tensor but changes exploration because the sampler already applies a square root.

Stable tie ordering makes deterministic fixtures reproducible without changing CEM's ranking objective. Affine action transforms verify that mean and variance transform respectively as `c*mean+b` and `c^2*variance`. Monotone cost transforms must preserve the selected set.

The toy provides sampling, action bounds and frozen dynamics. It plots the cost of the refitted mean separately from the best sampled candidate; these are not generally the same plan. There is no claim that CEM is globally optimal or that the synthetic rollout measures the paper's robot-control performance.
