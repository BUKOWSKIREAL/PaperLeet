# GRPO：通过后复盘

- **std 跨组计算**：另一道题的奖励尺度改变会影响当前题。`test_group_advantage_invariants` 只改变第二组，要求第一组保持不变。[TRL v0.23.1 源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py) 使用 `std(dim=1)`，对应 PyTorch 默认 correction=1；核验于 2026-10-13。eps 加在标准差外，正缩放严格不变只在 eps=0 时成立。
- **padding 污染或出现 NaN**：无效 token 的极端概率差可能先溢出，再乘零。`test_padding_isolation` 不只验证普通数值，还验证无效位置相差两千时的隔离及零梯度。
- **k1 偶尔为负**：`log pi - log pi_ref` 的单样本值可以小于零；本题指定 k3，`test_k3_nonnegative` 能区分。k3 的 KL 解释还依赖采样分布，离策略 batch 上不能宣称任何情况下都无偏。补充：[Schulman 2020 原始博客](http://joschu.net/blog/kl-approx.html)（本次检索该站返回错误，公式另由 TRL v0.23.1 与论文核对）。
- **旧策略漏 detach**：loss 数值完全一样，梯度却会流进不该训练的快照。`test_snapshots_no_gradient` 独立于数值对照测试此契约。
- **统一 token 平均**：`m5_token_mean` 表达的是另一种合法聚合思路，并不意味着 DAPO 有错；它违反本题的每回答等权约定。`test_response_duplication_invariance` 只复制一条回答的 token，区分两种目标；不同长度 token 权重将在 R5 中再次出现。
