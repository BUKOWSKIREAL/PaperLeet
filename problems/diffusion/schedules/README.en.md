# Closed-Form Forward Noising

## 1. What the paper solves

DDPM gradually transforms complex data into Gaussian noise, then learns to reverse the process. A closed-form forward distribution lets training select any time directly, without simulating every preceding step.

This module selects the closed-form marginal in Equation (4). Implement only `q_sample`; linear schedules, cumulative coefficients, and a cosine comparison schedule are supplied. The cosine schedule comes from Improved DDPM and is included as an optional comparison.

## 2. Core equations

The one-step noise variance is $\beta_t$, the retained signal fraction is $\alpha_t=1-\beta_t$, and the cumulative signal is $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Independent noise from successive steps combines into a single variance:

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

Here $x_0$ and $\epsilon$ have shape `[B,...]`, times t have shape `[B]`, and schedule arrays have shape `[T]`. When data and noise are independent standard Gaussians, the two variance contributions sum to one. The point cloud can therefore keep the same overall scale even as its structure disappears. See the closed-form marginal in [DDPM Equation 4](https://arxiv.org/abs/2006.11239).

## 3. From equations to code

Implement only `q_sample(x0, t, noise, alpha_bars)`:

```text
Extract each sample’s cumulative coefficient using its own t
Move coefficients to x0’s dtype/device and reshape to [B,1,...,1]
Combine x0 and noise using the signal and noise standard deviations
```

Return the same shape and dtype as `x0`. Each sample may use a different time; the first row’s time must not control the batch. Noise is supplied by the caller, so this function draws no random numbers.

`linear_betas`, `cosine_betas`, and `make_schedule` are fully implemented. Read them for context; no changes to these helpers are required.

## 4. Conventions

| Item | This exercise | Common implementations |
| --- | --- | --- |
| Indexing | Array index i=0 corresponds to paper time t=1 | Papers typically start at 1 |
| Schedule precision | float64 | Extracted coefficients are often converted to float32 during training |
| Linear beta | Fixed endpoints 1e-4 and 0.02 | improved-diffusion scales endpoints by the number of steps |
| Cosine beta | Maximum 0.999 | improved-diffusion also limits the maximum beta |

## 5. Interactive visualizations

<viz id="schedule_curves" />

Adjust the number of steps and offset s to compare cumulative signal and log-SNR for both schedules. A nonzero terminal signal means that a standard Gaussian is only an approximate starting boundary for sampling.

<viz id="forward_cloud" />

Move through time to compare how quickly structure disappears from the same eight-Gaussian data and noise samples. Reusing samples prevents random variation from being mistaken for a schedule effect.

## 6. Run your implementation

Open module **6** in the browser, complete `q_sample`, and run the tests. Your function is tested together with the supplied schedule helpers.

Tests check per-sample times and shapes, and use 100,000 samples to verify variance preservation. This statistical check catches valid shapes with incorrect meaning, such as missing square roots.

## 7. Further exploration

The cosine schedule in [Improved DDPM Equation 16](https://arxiv.org/abs/2102.09672) is an optional comparison, supplied in `cosine_betas`:

$$f(\tau)=\cos^2\left(\frac{\tau+s}{1+s}\frac\pi2\right),\qquad\beta_i=\min\left(1-\frac{f((i+1)/T)}{f(i/T)},0.999\right).$$

The log signal-to-noise ratio $\log\bar\alpha-\log(1-\bar\alpha)$ helps compare schedules. Continue with “Posterior and Reverse Denoising Step” under the same DDPM paper.
