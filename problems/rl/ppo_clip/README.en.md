# PPO Clipped Surrogate

## 1. What the paper solves

Training for multiple epochs on the same trajectories can move the policy far from the distribution that collected them. PPO uses a pessimistic clipped surrogate to support simple first-order updates. See [Equations 7 and 9](https://arxiv.org/pdf/1707.06347).

## 2. Core equations and task

Implement `ppo_loss`. All inputs have shape `[B]`; return six scalars: `loss, pg_loss, v_loss, entropy, approx_kl, clipfrac`. Current log-probabilities and values retain gradients. The old policy, old values, returns, and advantages are fixed targets.

$$
r=e^{\log\pi_\theta-\log\pi_{old}},\qquad
L^{PG}=-\mathbb E[\min(r\hat A,\operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)].
$$

For a positive advantage, an excessively large probability ratio stops earning additional reward for increasing the probability. For a negative advantage, an excessively small ratio stops earning additional reward for decreasing it. Moving in the unfavorable direction still incurs a penalty. Clipping the ratio and then multiplying by the advantage alone does not implement this objective.

$$
V_c=V_{old}+\operatorname{clip}(V-V_{old},-c,c),\quad
L^V=\tfrac12\mathbb E[\max((V-R)^2,(V_c-R)^2)].
$$

Minimize the total loss $L=L^{PG}+c_vL^V-c_e\mathbb E[H]$. Defaults are $\epsilon=c=0.2,c_v=0.5,c_e=0$. `vf_clip=None` disables value clipping; `entropy=None` makes the entropy term zero.

## 3. From equations to code

```text
Inputs: six [B] vectors, plus optional entropy [B]
Detach old_logp, old_values, and training targets
Optionally normalize advantages: [B]
Compute per-sample ratios, clipped surrogates, and both value errors
Average over the batch → scalar loss
Compute approx_kl and clipfrac without gradients
```

Use `approx_kl = mean(r-1-log(r))` and `clipfrac = mean(abs(r-1)>epsilon)`. The first expression is nonnegative per sample. The second measures how often ratios fall outside the interval; it does not measure how often the actual objective's gradient is clipped.

## 4. Conventions

| Item | This exercise | CleanRL `ppo.py` |
|---|---|---|
| Advantage standard deviation | Unbiased sample standard deviation, plus $10^{-8}$ | Same |
| Normalization with one sample | Standard deviation is treated as zero; normalized advantage is zero | Additional edge-case rule for this exercise |
| Value clipping | Optional; default is 0.2 | Enabled by default, sharing the clipping coefficient |
| Gradients through old quantities | Detach inside the function | Usually prevented by rollout storage |

The original exercise checked [CleanRL's implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) and [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) on 2026-10-13. PPO clipping does not impose a strict KL bound. This exercise's KL threshold applies to a small experiment with a fixed learning rate and batch; the training loop does not add KL-based early stopping.

The advantage's sign determines whether an action's probability should rise or fall. Establish that direction first, then inspect clipping boundaries. Sketch the four sign combinations and calculate a single sample's gradient before comparing it with the expected result.

## 5. Interactive visualizations

[Open the clipped objective](#viz-clip_objective): adjust $\epsilon$, compare positive and negative advantages, and move the probability ratio to inspect gradient directions and flat regions.

[Open the trust-region experiment](#viz-trust_region): compare per-epoch KL for the reference, a ratio-clipping-only implementation, and an incorrect `max` implementation.

## 6. Run your implementation

Open problem **2**, implement `ppo_loss` in the browser editor, and click **Run tests**. Inspect individual failures and the training plots in the results panel.

The four-quadrant gradient test isolates policy clipping. The value-loss lower-bound test isolates value clipping. A five-armed bandit experiment checks that repeated updates increase the optimal arm's probability above 0.9.

## 7. Further exploration

[DPO](../dpo/README.md) derives a policy objective directly from preferences. You can also study TRPO to compare constrained optimization with PPO's surrogate objective.
