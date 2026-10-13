# DDPM sampling: Review after passing

- The two posterior-mean coefficients look similar but are not interchangeable. `test_precision_product` derives mean and variance independently by adding Gaussian precisions, without relying on the reference implementation.
- The first posterior variance is exactly zero. improved-diffusion replaces it with the second entry before computing log variance to avoid log(0). This exercise uses variance directly, so no replacement is needed. The original exercise checked the [official implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py) on 2026-10-13.
- The final step must omit noise for beta variance too. Testing only posterior variance can miss this mistake; `test_last_step_no_noise` checks both options.
- Copying image x0 clipping into a point cloud of radius 2 damages the distribution. `test_gmm_sampling` checks all eight modes, rather than only the overall center.
- Sampling must visit times in descending order. `test_sampler_time_order` identifies the exact incorrect position; T3 checks the resulting distribution. Its MMD threshold is 1.5 times the measured reference value, each mode's target share is 12.5%±3%, and fewer than 5% of samples may lie more than .5 from a mode center.
