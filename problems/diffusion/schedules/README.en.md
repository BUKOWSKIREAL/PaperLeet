# Noise Schedules and Forward Diffusion

## 1. What the paper solves

DDPM gradually transforms complex data into Gaussian noise, then learns to reverse the process. A closed-form forward distribution lets training select any time directly, without simulating every preceding step.

## 2. Core equations

The one-step noise variance is $\beta_t$, the retained signal fraction is $\alpha_t=1-\beta_t$, and the cumulative signal is $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Independent noise from successive steps combines into a single variance:

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

Here $x_0$ and $\epsilon$ have shape `[B,...]`, times t have shape `[B]`, and schedule arrays have shape `[T]`. When data and noise are independent standard Gaussians, the two variance contributions sum to one. The point cloud can therefore keep the same overall scale even as its structure disappears. See the closed-form marginal in [DDPM Equation 4](https://arxiv.org/abs/2006.11239).

A linear beta schedule is simple, but may erase most of the signal early. [Improved DDPM Equation 16](https://arxiv.org/abs/2102.09672) defines the cumulative signal through a cosine curve:

$$f(\tau)=\cos^2\left(\frac{\tau+s}{1+s}\frac\pi2\right),\qquad\beta_i=\min\left(1-\frac{f((i+1)/T)}{f(i/T)},0.999\right).$$

The log signal-to-noise ratio, $\log\bar\alpha-\log(1-\bar\alpha)$, makes changes near the nearly pure-noise tail easier to inspect.

## 3. From equations to code

```text
Generate betas [T] in float64
alphas = 1 - betas; cumulative product over time → alpha_bars
Prepend 1 and drop the final element → alpha_bars_prev
Index with each sample's t and reshape to [B,1,...,1]
Move extracted coefficients to x0's dtype/device
Compute the forward sample
```

Implement `linear_betas`, `cosine_betas`, `make_schedule`, and `q_sample`. `make_schedule` also returns square-root coefficients and log_snr; the starter specifies field names. Make sure one sample's time index does not accidentally control the entire batch.

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

Open problem **6**, implement the four functions in the browser editor, and click **Run tests**. Test details and plots appear in the results panel.

Tests check exact endpoints and use 100,000 samples to verify variance preservation. This statistical check catches implementations with valid shapes but incorrect meaning, such as missing square roots.

## 7. Further exploration

Continue with parameterization conversion, then implement DDPM and DDIM. For terminal SNR and timestep selection, see [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891).
