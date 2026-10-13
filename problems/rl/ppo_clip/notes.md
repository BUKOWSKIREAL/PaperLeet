# PPO 截断目标：通过后复盘

- **只计算截断分支**：`clip(r)*A` 会把不利方向也截平。`test_core_four_quadrants_and_frozen_targets` 直接检查核心函数的四种梯度；`test_four_quadrant_gradient` 检查接入训练辅助后的结果。
- **用 max 代替 min**：PPO 取收益的悲观下界。负 advantage 也要保留符号，不能先取绝对值再截断。
- **把 log-prob 差当作概率比**：概率比是 `exp(new_logp - old_logp)`。同一策略的 ratio 应为一，策略梯度应退化为普通 policy gradient。
- **旧策略或 advantage 收到梯度**：二者都是这个优化步骤的固定目标。核心函数应自行 detach；测试会直接调用它检查计算图。
- **把 clipping 当作严格 KL 约束**：截断不能保证任意学习率或优化器下 KL 有界。现成 bandit 实验固定了训练配置，门槛只适用于这个实验。

`ppo_loss` 中的归一化、value clipping、熵和诊断量均已提供。它们的集成测试用于确认核心函数能接入训练；进一步研究这些实现约定可以作为完成本模块后的拓展。
