# Reference notes

For two-dimensional SPD matrices, an independent identity is

$$
\operatorname{tr}\sqrt{\Sigma_r\Sigma_g}=
\sqrt{\operatorname{tr}(\Sigma_r\Sigma_g)+2\sqrt{\det\Sigma_r\det\Sigma_g}}.
$$

The test oracle differentiates this scalar formula, avoiding the submitted eigenvalue implementation. Noncommuting covariances distinguish the symmetric sandwich from directly applying a Hermitian solver to `Sigma_r @ Sigma_g`. Coordinate-transform and symmetry checks further expose lost off-diagonal terms. Repeated positive eigenvalues are intentional: the sum of square-root eigenvalues remains differentiable on SPD matrices, even when an implementation based on differentiating eigenvectors would become unstable.

Reference statistics are constants. Their square root uses `eigh` after detaching, so no gradient through those eigenvectors is needed. Generated statistics enter only the sandwich and `eigvalsh`. For PSD matrices containing zero eigenvalues, the forward formula still applies, but square-root derivatives may be undefined or nonfinite. Both covariances must be SPD for this module's gradient contract. We do not add epsilon to covariance eigenvalues or clamp the final FD to manufacture a different differentiable objective.

Stopped normalization has gradient `w/(distance+eps)`. Differentiating the full numeric ratio would instead produce `w*eps/(distance+eps)^2`. A naive finite-difference check of changing numerator and denominator would therefore test the wrong function. Tests use an analytic gradient oracle; any finite-difference implementation must freeze the denominator at its base value.

Source implementation: [losses.py at 5c03b811](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/frechet_distance/losses.py), `precompute_sigma_ref_sqrt`, `_compute_trace_term` and `compute_frechet_distance_loss`; normalization is in [main_fd.py](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/main_fd.py). The teaching interface preserves float32/float64 rather than forcing the final scalar to float32. It computes the small frozen reference root per call; production code precomputes it. Distributed collection, nonfinite-failure fallbacks, pretrained feature models and sampling are excluded.

The teaching experiment represents a small affine Gaussian by its exact moments using symmetric quadrature points, with supplied detached EMA history. Its independent evaluator uses the 2D identity above; the optimization path calls the submitted functions. It reports both raw and normalized objectives because near-unit normalization values do not directly reveal distance reduction. Its settings and finite generator are teaching choices, not the paper's visual-generation benchmark.

Mutants remove generated correlations, use a nonsymmetric input to `eigvalsh`, differentiate the normalization denominator, or allow reference-statistic gradients.
