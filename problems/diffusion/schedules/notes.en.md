# Noise schedules: Review after passing

- The first cumulative alpha is not 1; the first *previous-time* cumulative alpha is 1. `test_reference` and `test_linear_endpoints` catch index shifts.
- Beta is a variance. The multiplier on noise must be a standard deviation. Missing square roots shrink the variance at intermediate times; see `test_variance_preserving`.
- Without clipping, the final cosine beta becomes 1 and can cause division by zero in later inversions. `test_cosine_clip` fixes the upper bound at 0.999.
- Each sample has its own time. Index the schedule with t before expanding spatial dimensions. `test_per_sample_time` uses unsorted, nonconsecutive times so incorrect broadcasting cannot pass by coincidence.

The original exercise checked [Improved DDPM Equation 16](https://arxiv.org/abs/2102.09672) and the [official `betas_for_alpha_bar` implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py) on 2026-10-13. The official linear schedule scales endpoints by 1000/T; this exercise keeps them fixed.
