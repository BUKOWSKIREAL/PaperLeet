# DDIM Deterministic Few-Step Sampling

## 1. What the paper solves

DDPM's training objective specifies noisy marginal distributions at each time without uniquely determining sampling trajectories. DDIM uses this freedom to reuse an existing noise predictor and generate samples in fewer denoising steps.

## 2. Core equations

Let the current cumulative signal be $a=\bar\alpha_t$ and the previous sampling time's cumulative signal be $b=\bar\alpha_{\mathrm{prev}}$, with $b\ge a$. Reconstruct the clean prediction, then combine a deterministic direction and random noise:

$$\hat x_0=\frac{x_t-\sqrt{1-a}\hat\epsilon}{\sqrt a},\qquad\sigma=\eta\sqrt{\frac{1-b}{1-a}}\sqrt{1-\frac ab}.$$

$$x_{\mathrm{prev}}=\sqrt b\hat x_0+\sqrt{1-b-\sigma^2}\hat\epsilon+\sigma z.$$

Sample tensors have shape `[B,D]`; a and b may be scalars or broadcastable to the sample shape. At eta=0, initial noise uniquely determines the trajectory. At eta=1 with adjacent timesteps, the stochastic variance equals the DDPM posterior variance. Subtract the stochastic variance from the direction term to avoid adding too much total noise. See [DDIM Equations 12 and 16](https://arxiv.org/abs/2010.02502) for the update and eta parameterization.

Set b=1 on the final update to reach the true clean endpoint. A point distribution provides a strong check: with perfect x0 predictions, both one step and fifty steps should arrive exactly at the point.

## 3. From equations to code

```text
timesteps[k] = T-1-k*(T//num_steps), for k = 0, ..., n-1
For each descending time: predict epsilon and call ddim_step
When no next time exists, set alpha_bar_prev=1
Invert in ascending order: evaluate epsilon at the current state
Then move to the noisier time
```

Implement `ddim_step`, `ddim_timesteps`, `ddim_sample`, and `ddim_invert`. Sampling can return a trajectory of shape `[n+1,B,D]` including both endpoints. Inversion uses a first-order approximation: the clean endpoint has no trained timestep index, so the first call uses index zero; later calls use the time corresponding to the current state. At finite step counts, this is not an exact algebraic inverse of a nonlinear sampling map.

## 4. Conventions

| Item | This exercise | diffusers DDIMScheduler |
| --- | --- | --- |
| Trailing step spacing | Integer spacing T//n | Uses real-valued T/n before rounding |
| When T is divisible by n | The two sequences agree | The two sequences agree |
| Other options | Fixed trailing spacing only | Also supports leading and linspace |
| Clean boundary | Final b=1 | Controlled by set_alpha_to_one |

The original exercise checked this comparison against the [official scheduler source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) on 2026-10-13. Follow this exercise's definition when T is not divisible by n.

## 5. Interactive visualizations

<viz id="eta_steps_grid" />

Choose an eta and step-count combination. Play trajectories from the same initial noise and compare final MMD to inspect the relationship between randomness and computation.

<viz id="inversion_demo" />

Choose a data point and step count to inspect inverted noise, reconstructed positions, and round-trip error. A finer time grid usually brings the reconstruction closer to the original sample.

## 6. Run your implementation

Open problem **9**, implement the four functions in the browser editor, and click **Run tests**. Inspect numeric checks and trajectory diagnostics in the results panel.

T2 checks determinism, the point-distribution endpoint, posterior variance, inversion, and discretization convergence. T3 uses the same eight-Gaussian oracle to check fifty-step generation quality and records eta=1 as a comparison.

## 7. Further exploration

Problem **10** moves from discrete denoising updates to a continuous velocity field. For few-step error and time boundaries, see [Progressive Distillation](https://arxiv.org/abs/2202.00512).
