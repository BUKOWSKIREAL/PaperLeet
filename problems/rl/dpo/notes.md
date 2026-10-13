# DPO 偏好目标：通过后复盘

- **遗漏参考策略**：偏好 margin 比较策略相对 reference 的提升。策略与 reference 相同时，margin 应为零，loss 应为 `log(2)`。
- **遗漏 beta 或只缩放 loss**：beta 应进入隐式奖励与 sigmoid 内的 margin。`test_beta_margin_scaling` 检查这个关系。
- **log(sigmoid) 下溢**：负的极端 margin 先 sigmoid 为零，导致无穷 loss。`test_extreme_margin_stability` 检查有限 loss 和精确梯度。
- **把 batch 求和当作平均**：每对偏好贡献一个样本损失，最后求均值。batch 大小不应改变同一数据分布的梯度尺度。
- **诊断奖励残留计算图**：返回奖励需要 detach，reference 同样固定。`test_gradient_direction_and_rewards` 检查方向、值与计算图。

已提供的 `sequence_logprob` 使用标签侧 mask，并对回答 token 的 log-prob 求和。相关移位测试检查辅助代码的集成，无需重新实现这部分。默认 `label_smoothing=0` 对应原始 DPO；非零值沿用保守 DPO 的正反标签混合约定，可作为拓展阅读 [TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer)。
