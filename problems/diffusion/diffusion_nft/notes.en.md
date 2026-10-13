# Reference notes

Write `d=current-old` and `e=old-target`. The per-example objective expands to `||e||^2 + beta^2||d||^2 + 2*beta*(2*r-1)<e,d>`. Its current-policy gradient is `2*beta*(beta*d+(2*r-1)*e)` before batch averaging. This independent expansion supplies the gradient and optimum oracles.

At reward one-half, the cross term vanishes: optimization pulls toward the old policy. At reward zero, minimizing the negative branch pushes the current policy away from the target relative to the old policy. This behavior is lost if negative examples are discarded or if a signed MSE replaces the implicit policy construction.

The equation uses a feature-summed Euclidean norm. Production training may absorb feature normalization and time weighting into optimization choices; the harness fixes the exact reduction explicitly. The scope is the paper's basic Eq. (5), without practical adaptive weighting, EMA scheduling or reward normalization. Those are separate choices and are not silently introduced here.

The original paper allows arbitrary black-box samplers to collect clean data. The toy supplies a frozen minibatch instead and independently optimizes each velocity output. Its measured optimum demonstrates the local objective, not a theorem about a finite-capacity denoiser or global reward improvement. Data-to-noise time runs from zero to one, matching the paper's rectified-flow convention.

The old rollout policy, clean target and reward probability are fixed context. Tests intentionally set their `requires_grad` flags to catch accidental gradient leakage. Four mutations target reciprocal beta, the negative branch sign, omission of negative samples, and training the old policy.
