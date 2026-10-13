# DDIM: Review after passing

- Eta=1 recovers the posterior variance for adjacent times. The direction term must also subtract sigma². `test_eta_one_posterior` and `test_reference` catch errors in these factors.
- A timestep subsequence usually does not end at index 0, and index 0 itself still represents one noising step. Take an additional jump to alpha_bar=1; see `test_point_final_jump`.
- This exercise uses trailing spacing T//n. The official diffusers trailing sequence uses T/n followed by rounding. They agree for T=1000, n=50 but may differ when division is inexact. `test_trailing_times` explicitly covers T=17, n=5. The original exercise checked the [official source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) on 2026-10-13.
- Determinism does not imply exact finite-step invertibility. Epsilon depends on the current state, and first-order inversion is a local approximation. At the clean starting point, it first uses index zero. `test_inversion_converges` checks that error falls when increasing from 10 to 100 steps.
- `test_step_convergence` fixes the initial noise so sampling randomness cannot be mistaken for discretization error. The fifty-step eta=1 comparison is display-only and does not affect the eta=0 quality score.
