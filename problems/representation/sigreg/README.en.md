# LeJEPA Sliced Gaussian Regularization

Implement SIGReg from [LeJEPA, Sections 4.2.3 and 5.1](https://arxiv.org/html/2511.08544v1). Project embeddings onto unit directions, compare each empirical characteristic function with the standard normal target, then average the statistics across slices. This checks distribution shape beyond its first two moments.

For projections $P=XA\in\mathbb R^{N\times M}$ and frequencies $t_k\ge0$,

$$
\hat c_{mk}=\frac1N\sum_n\cos(t_kP_{nm}),\quad
\hat s_{mk}=\frac1N\sum_n\sin(t_kP_{nm}),\quad
\phi_k=e^{-t_k^2/2},
$$

$$
E_{mk}=\left[(\hat c_{mk}-\phi_k)^2+\hat s_{mk}^2\right]\phi_k,\quad
T_m=2N\sum_{k=0}^{K-2}\frac{t_{k+1}-t_k}{2}(E_{m,k+1}+E_{mk}),\quad
L=\operatorname{mean}_m T_m.
$$

## Implement

| Function | Output |
|---|---|
| `epps_pulley(projected,t)` | One statistic per slice, shape `(M,)` |
| `sigreg_loss(features,directions,t)` | Scalar mean across slices |

`features` has shape `(N,D)`, `directions` has shape `(D,M)` with unit-length columns, and `t` is a strictly increasing vector of at least two nodes starting at zero. `N,D,M>=1`. All inputs share float32/float64 dtype and device. Supplied validators cover these conditions; preserve inputs and dtype/device.

Use the actual grid spacings, including nonuniform grids. The usual supplied grid has 17 nodes on `[0,3]`. Doubling the positive-half integral accounts for its negative half. The Gaussian factor is used both as the target characteristic function and as the integration window. Include the sine term, multiply by `N`, then average over slices.

Directions and quadrature nodes are fixed scaffolding: detach them internally. Keep gradients through features and projected samples. Do not center, standardize, whiten or detach features; the target is specifically `N(0,I)`. This exercise supplies the directions instead of sampling them, making the harness deterministic.

All-zero features have a positive statistic and **zero gradient**. Nearby nonconstant features can have a direction of descent. Neither a finite set of slices nor finite quadrature nodes guarantees identification of every possible distribution.

## Toy: beyond matching moments

**Teaching adaptation.** Optimize 64 distorted 2D embeddings using 16 fixed slices. Compare Gaussian quantiles and balanced `{-1,+1}` samples after giving both exactly zero mean and unit population variance: their characteristic functions still differ. Plots show the submitted loss, independent feature diagnostics and one projected characteristic function. No encoder or full LeJEPA multiview objective is trained.

T1 compares against scalar complex arithmetic on the full symmetric grid. T2 checks gradients, fixed geometry, stationary collapse, joint rotations, sample-count scaling and target mean/scale. T3 executes your functions in the toy and checks the final loss against an independent full-grid oracle. Preview charts use the reference solution.
