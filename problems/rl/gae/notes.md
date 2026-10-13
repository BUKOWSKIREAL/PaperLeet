# GAE：通过后复盘

- **跨 episode 泄漏**：上一局末步受到下一局奖励影响，通常是递推只乘 gamma、lambda，没有乘 `~done`。`test_episode_isolation` 会只改变边界后的奖励，直接定位泄漏。
- **时间截断丢 bootstrap**：把 `terminated | truncated` 同时用于 TD 残差，会低估时间限制处的回报。`test_termination_and_truncation_bootstrap` 用两种结束方式的并排样例区分它们。现代 API 的区分不来自 GAE 原文。
- **反向依赖写成正向循环**：GAE 依赖未来残差；形状正确不代表方向正确。`test_lambda_one_telescopes` 用独立的奖励展开式检查整个时间轴。
- **returns 混淆为 MC**：当 lambda 小于一时，value target 仍是 advantage 加当前 value；`test_returns_contract` 专门检查此约定。
- **误解无偏性**：有价值误差时，lambda 一的 advantage 本身仍可有状态基线偏移，但策略梯度不受动作无关基线影响。`test_bias_variance_tradeoff` 比较首个动作的 score-function 梯度与 DP 精确值，而非仅比较 advantage 均值。原文位置已核对：[式 16–18](https://arxiv.org/html/1506.02438v6#S3)。
