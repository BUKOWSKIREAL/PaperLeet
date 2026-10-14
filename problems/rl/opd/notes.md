# 实现说明 — 学生前缀上的 On-policy Distillation

实现 `sample_student` 与 `distillation_loss`。采样表为 `[V+1,V]`；第 `V` 行是 BOS，第 `k` 行是 token `k` 之后的条件分布。返回 `[B,L]` 整数 `tokens` 和 `prefixes`。首个前缀为 BOS，其后必须是上一时刻学生采到的 token。使用传入 generator，温度 1 的分类采样，不穿过离散采样反向传播。此 toy 定长、无 EOS。

损失输入为 `[B,L,V]` logits：**学生与教师都在同一批学生前缀上计算**，再提供 `[B,L]` 二元 mask。`reverse` 表示 KL(学生 || 教师)，`forward` 表示 KL(教师 || 学生)。对完整词表求和，仅平均有效 token，冻结教师，使用 log-softmax。全 padding 返回可微零；padding logits 可能非有限，须先清零。有效位置 logits 有限。

本题固定 GKD 的 lambda=1，不是 sampled-token OPD、策略梯度估计量或教师生成数据。提供的训练循环让三 token 模型学习四个条件分布。训练在新学生前缀上使用 reverse KL，评估对所有状态计算平均 forward KL。

## 错误变体

- reverse KL 错用教师分布加权，交换了 KL 方向。
- 错误地更新教师。
- 每一步都从 BOS 采样，没有使用学生真实前缀。
- 归一化分母包含 padding。

每个 mutant 都是相对于参考实现的小幅故意错误。T2 包含独立公式或梯度期望，接受结果不只依赖参考值相等。T3 使用传入的学员实现产生结果。

依据：[2306.13649v3](https://arxiv.org/html/2306.13649v3#S3)；[TRL 实现（独立于论文作者；不是本题参考答案来源）](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py)。实验来源：PaperLeet 定长分类 Markov 模型，不是原作 T5 或混合高斯实验。
