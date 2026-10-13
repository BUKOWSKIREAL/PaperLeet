# DPO preference objective: Review after passing

- **Omitting the reference policy.** The preference margin measures improvements relative to the reference. When policy and reference match, the margin is zero and the loss is `log(2)`.
- **Omitting beta or scaling only the final loss.** Beta scales implicit rewards and the margin inside the sigmoid. `test_beta_margin_scaling` checks this relationship.
- **Underflow in log-sigmoid.** Applying sigmoid first to a very negative margin can produce zero and an infinite loss. `test_extreme_margin_stability` checks finite losses and exact gradients.
- **Summing instead of averaging the batch.** Each preference pair contributes one sample loss; take the mean. Batch size should not change the gradient scale for the same data distribution.
- **Retaining graphs on diagnostic rewards.** Detach returned rewards and keep the reference fixed. `test_gradient_direction_and_rewards` checks values, directions, and graph connectivity.

The supplied `sequence_logprob` uses a label-side mask and sums response-token log-probabilities. Shifting tests check this helper's integration; you do not need to reimplement it. The default `label_smoothing=0` gives original DPO. Nonzero smoothing retains the conservative DPO convention as an extension; see [TRL v0.23.1 documentation](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer).
