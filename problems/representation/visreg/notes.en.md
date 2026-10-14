# Reference notes

Algorithm 1 makes three useful conventions explicit: `std(unbiased=False)`, the **squared** scale deviation, and means over feature/projection entries. Using VICReg's variance hinge or a covariance penalty changes this module's objective. The Gaussian grid excludes 0 and 1, avoiding infinite inverse-CDF values.

The shape forward uses a data-dependent standard deviation, but its derivative treats that denominator as constant. A valid finite-difference gradient check therefore freezes the denominator at the base point. Recomputing it after every perturbation differentiates a different function. Directions are supplied constants, and normalizing their columns makes their positive magnitude irrelevant. Sorting removes sample order; sorting feature columns instead destroys the one-dimensional distribution comparison.

For invariance, write `r_v = z_v - g`. Every view receives its direct residual gradient. A global view also receives the anchor contribution `-2 sum_v(r_v)/(G*V*N*D)`. Detaching the anchor leaves the scalar loss unchanged and changes these gradients.

The paper's Eq. (2) includes epsilon, while its displayed Algorithm 1 omits it. This package explicitly uses Eq. (2)'s denominator and Algorithm 1's mean reductions. It does not claim that a tied sort has a unique gradient, or that exact collapse must yield a particular optimization trajectory.

Source check: [paper v1, Algorithm 1](https://arxiv.org/html/2606.02572v1#alg1) is the implementation authority used here. The [author repository](https://github.com/HaiyuWu/visreg/blob/7e0a0bb3a4ee302d65678dcc9b85cb6ef718d356/README.md) contained only a redirect when checked on 2026-10-13, and its destination was unavailable. We therefore make no claim of validating a full released training codebase. The synthetic toy is independently constructed, inspired by the paper's scale/shape analysis.
