# Direct Preference Optimization

## 1. What the paper solves

DPO rewrites KL-regularized reward maximization as a classification objective on preference pairs, eliminating the need to fit a separate reward model during training. [Section 4](https://arxiv.org/html/2305.18290v3#S4) derives the equivalence from the optimal policy.

## 2. Core equations and task

First implement `sequence_logprob`. Inputs are logits `[B,T,V]`, token IDs `[B,T]`, and a response mask `[B,T]`; output is `[B]`. Logits at position $t$ predict the token at $t+1$. Sum log-probabilities only over response tokens. The first label has no preceding prediction; the last logits have no corresponding next label.

Then implement `dpo_loss`. The four sequence log-probabilities have shape `[B]`. Return a scalar loss and two detached reward vectors.

$$
h=(\log\pi_c-\log\pi^{ref}_c)-(\log\pi_r-\log\pi^{ref}_r),\qquad
L=-\mathbb E[(1-\varepsilon)\log\sigma(\beta h)+\varepsilon\log\sigma(-\beta h)].
$$

Here $c$ and $r$ mean chosen and rejected. Defaults are $\beta=0.1$ and label smoothing $\varepsilon=0$. The implicit reward is $\beta(\log\pi-\log\pi^{ref})$. It measures improvement relative to the reference policy, so the chosen response's raw log-probability alone is not a predicted preference score.

## 3. From equations to code

```text
logits[:, :-1]: [B,T-1,V] → log_softmax
Gather label scores using input_ids[:, 1:]: [B,T-1]
Multiply by loss_mask[:, 1:] and sum over time → [B]
Detach the reference; compute chosen/rejected rewards and their margin
Combine positive and reversed labels with stable log-sigmoid → scalar loss
```

Keep the computation in log space. When $\beta h=-10000$, computing sigmoid first underflows to zero, and taking its logarithm produces infinity. A stable function evaluates the same objective directly, without clipping the margin or erasing gradients for hard examples.

## 4. Conventions

| Item | This exercise | Related methods or libraries |
|---|---|---|
| Token aggregation | Sum of response log-probabilities | SimPO uses length normalization, giving a different objective |
| Mask | Aligned with shifted labels | Excludes both prompt and padding tokens |
| Label smoothing | Convex combination of forward and reversed sigmoid losses | Conservative DPO as described in TRL v0.23.1 |
| Reference and returned rewards | No reference gradients; detached returned rewards | Rewards are diagnostic outputs, not another backward path |

[TRL v0.23.1 documentation](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) connects this smoothing to conservative DPO. It differs from the label-noise correction used by the `robust` loss. T3 uses six candidates, a fixed nonuniform reference, and 20,000 Bradley–Terry preference pairs. It compares the trained policy with $\pi^*\propto\pi_{ref}e^{r^*/\beta}$ and checks that implicit rewards exactly recover the true ranking.

If the reference model already favors one response, the policy must improve its relative probability by more to create a positive preference margin. This explains both the reference distribution's role in the closed-form optimum and the experiment's use of a nonuniform reference.

## 5. Interactive visualizations

[Open the label and mask grid](#viz-shift_mask_grid): inspect which next token each position predicts and which contributions the prompt mask excludes.

[Open the margin curves](#viz-loss_vs_margin): adjust $\beta$ and $\varepsilon$ to inspect the loss and gradient weights. Reward scatter plots show how training recovers the true reward ranking.

## 6. Run your implementation

Open problem **3**, write both functions in the browser editor, and click **Run tests**. Read failure details and experiment outputs in the results panel.

The double-loop test checks shifting. Extreme-margin tests check numerical stability. The behavioral test trains a small categorical policy.

## 7. Further exploration

Continue with [GRPO](../grpo/README.md) to compare offline preference learning with online group sampling. You can also investigate which optimality assumptions change under length normalization or noisy preferences.
