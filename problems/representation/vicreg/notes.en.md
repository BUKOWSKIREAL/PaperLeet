# Reference notes

Center along the batch axis. The invariance term uses the original paired values, so independently translating one view still incurs an alignment cost. Centering both views before the MSE would change that objective.

The variance hinge only asks that standard deviation reach a floor. It does not force variance to exactly one. In contrast, penalizing every covariance entry would push diagonal variance back toward zero and conflict with that floor. Two duplicated feature columns can each satisfy the floor while their off-diagonal covariance remains large.

For an active variance hinge, the derivative for one view is

$$
\frac{\partial v}{\partial X_{ij}}=-\frac{X_{ij}-\bar X_j}{2D(B-1)\sqrt{C_{X,jj}+\epsilon}}.
$$

This directly explains why exactly constant features have zero gradient despite positive variance loss. T2 checks that stationary point and a nearby nonconstant perturbation; the toy starts from the latter kind of geometry.

The paper anchor is [v3 Section 4.1, Eqs. (1)-(6)](https://arxiv.org/html/2105.04906v3). Eq. (5) sums feature coordinates inside the squared vector norm, whereas official MSE averages those coordinates. The implementation convention comes from [official `main_vicreg.py` at 4e12602f](https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py), `VICReg.forward`. This module replaces the distributed global batch with a single supplied batch. Unlike the displayed paper sum of per-view variance terms, official `std_loss` averages them; official `cov_loss` sums them. Matching the coefficients requires retaining that distinction.

The scalar-loop oracle computes centered products without the reference implementation. Gradient checks cover both views, and the toy independently measures feature geometry. Four mutants target population instead of sample statistics, an incorrect variance-view sum, inclusion of diagonal covariance, and an accidentally detached second view. The toy is a controlled embedding optimization, not a replication of the paper's ImageNet experiments or a claim of universal collapse avoidance.
