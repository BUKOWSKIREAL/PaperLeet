# PPO：通过后复盘

- **把 clip 当成双向梯度截断**：只用 `clip(r)*A` 会把不利方向也截平；`test_four_quadrant_gradient` 用四个符号组合给出精确梯度。
- **把 policy 的 min 与 value 的 max 混用**：policy 使用悲观收益下界，value 使用悲观误差上界；`test_value_loss_lower_bound` 选择当前预测恰好等于 target、旧预测相距很远的例子，使差异无法隐藏。
- **旧模型收到梯度**：旧 log-prob 或 value 是快照，即使输入带 `requires_grad` 也必须 detach。`test_old_values_are_constants` 检查两种快照。
- **std 默认值和单样本边界**：本题与 [CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) 一样使用样本标准差；B=1 额外约定标准差为零。核验日期 2026-10-13。`test_advantage_affine_invariance` 与参考对照检查归一化。实现细节稳定文章：[ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)。
- **把 PPO 当作严格 KL 约束**：截断不保证任意学习率、优化器下 KL 有界。`test_bandit_learning_and_kl` 固定完整训练配置，在没有 KL 早停的条件下标定门槛；该数字不应直接移用到大模型训练。
