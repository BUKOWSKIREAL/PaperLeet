# Flow Matching: Review after passing

- Interpolation, target sign, and integration direction must agree. `test_point_zero_loss` checks training; `test_gaussian_transport` and `test_learns_eight_modes` check generation, rather than relying only on decreasing loss.
- The point-distribution oracle is singular at t=1. Euler evaluates left endpoints, so its largest evaluation time is (n-1)/n. `test_no_singular_endpoint` records the actual call times.
- Conditional targets have irreducible variance, so CFM loss generally cannot reach zero. Only the special point distribution in `test_point_zero_loss` has zero optimal error; the real training curves retain a nonzero final loss.
- The loss averages over every batch and coordinate element. Summing over coordinates before averaging over the batch differs by the number of dimensions. `test_reference` makes this contract explicit.
- A TimeMLP trained for 2,000 steps is not an exact oracle. The measured outlier fraction beyond radius .5 is .123; the specification's factor of 1.5 gives a threshold of .1845. MMD is calibrated by the same factor. Separately, all eight modes must be covered and each mode's share must be within .05 of .125. Environment details and measurements are in `problem.yaml`. These thresholds apply to a short training budget and do not imply the exact-oracle quality used in problem **8**.

The original exercise checked the time conventions against [Lipman Equations 22–23](https://arxiv.org/html/2210.02747v2), [Rectified Flow](https://arxiv.org/abs/2209.03003), and [SD3 Equation 13](https://arxiv.org/html/2403.03206v1) on 2026-10-13.
