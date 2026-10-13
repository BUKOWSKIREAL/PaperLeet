# Reference notes

Raw second moments carry both within-component variance and between-component mean variation. Equivalently, the updated covariance is

$$
\Sigma=\beta\Sigma_{old}+(1-\beta)\Sigma_b+
\beta(1-\beta)(\mu_{old}-\mu_b)(\mu_{old}-\mu_b)^\top.
$$

The final outer product is lost when covariances are averaged directly. The independent T1 oracle constructs sample weights for an explicit old/current population and sums centered outer products, rather than copying the implementation.

For `mean.sum()+covariance.trace()`, the gradient at current row `x_i` is `(1-beta)/B * (1 + 2*(x_i-mean))`. For `||mean||^2+trace(covariance)`, it simplifies to `2*(1-beta)/B*x_i`, since the objective equals the trace of the raw second moment. The latter verifies the toy's current-batch graph.

The source boundary is [official queue.py at 5c03b811](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/frechet_distance/queue.py), especially `_build_feats_stats_ema`, plus Algorithm 1 and Eqs. (3)-(5). Official implementation computes statistics in float64; this CPU exercise preserves the caller's float32/float64 dtype. Its functional return includes the raw second moment so the supplied loop can detach and persist it explicitly. There is no feature queue, in-place module state, distributed gather or image backbone in this exercise.

The official queue path replaces the oldest features and divides centered squared sums by `N-1`. EMA instead mixes normalized raw moments and reconstructs population covariance without a Bessel factor. Warm-started history is supplied; bias correction from zero-initialized EMA is neither required nor introduced.

The toy's full-population warm start is deliberately controlled. It demonstrates why information from previous batches changes estimates and gradients. It does not test generator convergence, choose the paper's production decay, or reproduce its ImageNet ablation. Four mutants target centered second moments, swapped EMA decay, gradients into history, and centering the mixed second moment around the current batch mean.
