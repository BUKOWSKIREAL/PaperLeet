# VICReg Variance, Invariance and Covariance

Implement the feature-space objective from [VICReg](https://arxiv.org/html/2105.04906v3). Two augmented views should agree, each feature should retain sufficient spread across examples, and different features should avoid redundant covariance.

Given paired features $X,Y\in\mathbb R^{B\times D}$, let $X_c=X-\operatorname{mean}_B(X)$ and $C_X=X_c^TX_c/(B-1)$, with the same definitions for $Y$.

$$
s=\operatorname{mean}((X-Y)^2),\qquad
v=\frac{1}{2D}\sum_j\left[\max(0,1-\sqrt{C_{X,jj}+\epsilon})+\max(0,1-\sqrt{C_{Y,jj}+\epsilon})\right],
$$

$$
c=\frac1D\sum_{i\ne j}(C_{X,ij}^2+C_{Y,ij}^2),\qquad
L=\lambda s+\mu v+\nu c.
$$

## Implement

| Function | Output |
|---|---|
| `vicreg_terms(x,y,eps=1e-4)` | Three scalar tensors `(invariance, variance, covariance)` |
| `vicreg_loss(x,y,sim_coeff=25.,std_coeff=25.,cov_coeff=1.,eps=1e-4)` | Weighted scalar loss |

Both inputs have matching float32/float64 dtype, device and `(B,D)` shape, with `B>=2` and `D>=1`. Supplied validators check the contract. Preserve inputs and dtype/device. Both views receive gradients. Coefficients are finite nonnegative Python numbers; epsilon is finite and positive.

Use **sample** variance and covariance with denominator `B-1`. Invariance averages all `B*D` entries. Variance averages the two view penalties, while covariance **sums** them. Divide the off-diagonal squared sum by `D`, not by the number of pairs. Both `(i,j)` and `(j,i)` count. When `D=1`, covariance loss is zero.

These normalizations follow the pinned official code. Its MSE averages over features as well as samples, whereas the paper's Eq. (5) writes a sample mean of squared vector norms. The paper's written combined objective sums its per-view variance terms; the code averages them. This exercise uses the code's factor of one half and its default weights consistently.

Identical all-zero embeddings yield `(0,0.99,0)` at default epsilon. They still have zero gradient: the positive penalty is not a guarantee that optimization can escape exact collapse. A small **nonconstant** feature cloud is a different case.

## Toy: spread and redundancy

**Teaching adaptation.** Optimize two sets of 16 paired 2D embeddings from small, correlated, nonconstant values. The scaffold calls your functions and plots loss, independently measured standard deviation and off-diagonal covariance. Compare the three final terms with exact collapse. No image encoder is trained.

T0 checks the interface. T1 uses a scalar-loop oracle and a one-feature case. T2 checks both-view gradients, the exact-collapse stationary point, translations, permutations and duplicated-feature redundancy. T3 optimizes the supplied geometry using your submitted loss; preview plots use the reference solution.
