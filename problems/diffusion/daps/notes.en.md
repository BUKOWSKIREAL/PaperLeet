# Reference notes

The Gaussian approximation is to `p(x0 | xt)`, not directly to the global unconditional data prior. In a real DAPS outer loop, the ODE recomputes `x0_hat` and the schedule selects `r` at each noise level. This module accepts those quantities explicitly. Specializing the paper's differentiable observation map to a matrix makes the likelihood score independently testable.

The [author sampler](https://github.com/zhangbingliang2019/DAPS/blob/main/cores/mcmc.py) constructs its Gaussian prior contribution as `(x0hat - xt)/sigma^2`, then adds `(xt-x)/sigma^2`; these combine to `(x0hat-x)/sigma^2`. Its Langevin update uses `sqrt(2*lr)`. The exercise exposes the paper's general `r` explicitly, so the code's Gaussian default is the special case `r=sigma`.

For the teaching conditional, precision is `H = I/r^2 + A.T@A/beta^2`, mean is `H^-1(x0_hat/r^2 + A.T@y/beta^2)`, and true covariance is `H^-1`. A stable unadjusted Langevin step has stationary covariance `(H - eta*H@H/2)^-1`. The toy checks both, avoiding an incorrect claim of exact posterior sampling at finite step size.

Appendix G describes a two-component Gaussian mixture and a nonlinear observation toy. The displayed observation equation with `y=0` does not clearly support the stated likelihood modes. The public repository tree inspected on 2026-10-13 did not expose a runnable synthetic/toy script that resolves this convention. Therefore this package uses a clearly labeled new linear-Gaussian exercise rather than silently altering the paper's toy. Its score, Langevin scaling and re-noising kernel still follow Eq. (7) and Algorithm 1.

Fresh noise is an explicit argument to keep the harness deterministic. The caller is responsible for independent draws; the composed test and experiment use distinct MCMC and annealing samples. No claim is made about real-image restoration, global posterior mixing or pretrained-model quality.
