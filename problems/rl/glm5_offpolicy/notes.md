# 离策略纠偏提示

这里做的是屏蔽，不是 PPO clipping 或直接 clamp。无论优势正负，越界 token 的贡献都为零；分母仍统计原来的有效 token 数，仅对幸存 token 重新归一化会改变目标。

计算冻结的比率，按张量 dtype 与 1-low、1+high 严格比较。对被拒绝的权重使用 torch.where 选择零，包括冻结的指数运算产生的无穷；直接与零 mask 相乘会得到 NaN。最终乘积也要预先排除 padding：零乘 NaN 仍是 NaN。

每个当前 log-prob 的策略梯度应恰好是 -mask*weight*advantage/N。不要用跨次改变冻结权重的数值差分验证这个 stop-gradient surrogate，而应使用上述解析梯度。论文和官方 helper 在端点上的差异已在题面明确。
