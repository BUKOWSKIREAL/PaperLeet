# PPO clipped objective: Review after passing

- **Keeping only the clipped branch.** `clip(r)*A` also flattens unfavorable directions. `test_core_four_quadrants_and_frozen_targets` checks all four gradient cases directly; `test_four_quadrant_gradient` checks integration with the supplied adapter.
- **Using max instead of min.** PPO takes a pessimistic lower bound on reward. Keep negative advantages signed; taking their absolute value changes the optimization direction.
- **Using the log-probability difference as a ratio.** The ratio is `exp(new_logp - old_logp)`. Identical policies give a ratio of one and recover the ordinary policy gradient.
- **Sending gradients into the old policy or advantages.** Both are fixed targets for this update. Detach them inside the core function; tests call it directly to inspect gradient connectivity.
- **Treating clipping as a strict KL constraint.** Clipping does not bound KL for arbitrary learning rates or optimizers. The supplied bandit experiment uses a fixed training configuration, and its threshold applies to that experiment.

Normalization, value clipping, entropy, and diagnostics are supplied in `ppo_loss`. Their integration tests check that your core function works in training. Studying these implementation conventions is an extension after completing this module.
