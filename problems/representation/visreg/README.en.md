# VISReg Scale and Distribution Shape

Implement the regularizer and view-invariance objective in [VISReg](https://arxiv.org/abs/2606.02572), Section 3.1. A batch can have healthy variance but an unsuitable distribution shape. VISReg gives these two properties separate objectives.

## Implement

`visreg_terms(z, directions, eps=1e-4)` receives features `[N,D]` and fixed, nonzero projection columns `[D,K]`. Return three scalar tensors **in this order: scale, shape, center**.

Let the per-feature population mean and standard deviation be

$$\mu_j=\frac1N\sum_n z_{nj},\qquad \sigma_j=\sqrt{\frac1N\sum_n(z_{nj}-\mu_j)^2}.$$

Compute

$$L_{scale}=\operatorname{mean}_j(\sigma_j-1)^2,\quad
\widetilde z=\frac{z-\mu}{\operatorname{sg}(\sigma)+\epsilon},\quad
L_{center}=\operatorname{mean}_j\mu_j^2.$$

Normalize each supplied direction column to unit length, project `z_tilde`, and sort **across the N samples independently for each direction**. For ranks `i=1,...,N`, use `q_i=Phi_inverse(i/(N+1))`, where `Phi` is the standard Gaussian CDF:

$$L_{shape}=\frac1{NK}\sum_{i,k}\left(\operatorname{sort}(\widetilde z W)_{ik}-q_i\right)^2.$$

Direction tensors are fixed and must be detached. Feature gradients flow through centering, projections and sorting; the standard deviation is detached **only in the shape denominator**. The scale term retains its standard-deviation gradient. Use population standard deviation, squared scale error and mean reductions as in Algorithm 1. Use the `+eps` from Eq. (2), which also defines finite forward values for fully collapsed inputs.

`view_invariance(views, num_global=2)` receives `[V,N,D]`. The first `G=num_global` views are global. Form their per-example mean `g=views[:G].mean(0)` and return `mean((views-g)^2)` over **all views, samples and features**. Gradients flow through `g`; there is no frozen teacher in this function.

For example, views `[[[0,2]], [[2,0]], [[5,-1]]]` and `G=2` have anchor `[1,1]` and invariance loss `4`.

## Contract

Finite CPU float32/float64 inputs, matching dtype/device; preserve inputs and output dtype. `N>=2`, `D,K>=1`, each direction nonzero, `eps>0` finite, and integer `1<=G<=V`. Noncontiguous feature tensors are supported. Validation is supplied. At tied projections or exactly zero variance, tests require defined forward values; they do not require a unique mathematical derivative.

This exercise isolates the two loss components. The toy combines them as `0.5 * invariance + 0.5 * mean_over_views(scale + shape + center)`. Encoders, image augmentation, projection sampling and distributed training are supplied or outside the scope.

## Toy experiment

**Teaching adaptation.** Three aligned 24-by-2 feature tables start with small spread and an offset. Your two functions drive 80 gradient updates with fixed projection directions. Charts show the submitted objective and shape term alongside independently measured population spread and squared mean. This is a small feature optimization experiment, not a reproduction of the paper's pretraining results. The preview uses the reference; T3 runs your submission.
