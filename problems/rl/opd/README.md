# 学生前缀上的 On-policy Distillation

从学生分布采样，在相同前缀上蒸馏教师全词表反馈。

论文：[On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649v3)。定位：Section 3, Algorithm 1: GKD and student data fraction lambda；Section 3, Eqs. (2–4): token divergences on student-generated prefixes；Appendix A.7: forward/reverse KL interpretation (separate conceptual example)。

## 精选核心

选取 GKD 的 lambda=1 与明确方向的 KL；提供三 token Markov 语言模型代替大型语言模型。

$$
L=\mathbb E_{y\sim p_S}[\frac1{|y|}\sum_t D_{KL}(p_S(\cdot|y_{<t})\Vert p_T(\cdot|y_{<t}))]
$$

- `sample_student(transition_logits: Tensor, batch_size: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor]`
- `distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, direction: str='reverse') -> Tensor`

## 接口与实验

实现 `sample_student` 与 `distillation_loss`。采样表为 `[V+1,V]`；第 `V` 行是 BOS，第 `k` 行是 token `k` 之后的条件分布。返回 `[B,L]` 整数 `tokens` 和 `prefixes`。首个前缀为 BOS，其后必须是上一时刻学生采到的 token。使用传入 generator，温度 1 的分类采样，不穿过离散采样反向传播。此 toy 定长、无 EOS。

损失输入为 `[B,L,V]` logits：**学生与教师都在同一批学生前缀上计算**，再提供 `[B,L]` 二元 mask。`reverse` 表示 KL(学生 || 教师)，`forward` 表示 KL(教师 || 学生)。对完整词表求和，仅平均有效 token，冻结教师，使用 log-softmax。全 padding 返回可微零；padding logits 可能非有限，须先清零。有效位置 logits 有限。

本题固定 GKD 的 lambda=1，不是 sampled-token OPD、策略梯度估计量或教师生成数据。提供的训练循环让三 token 模型学习四个条件分布。训练在新学生前缀上使用 reverse KL，评估对所有状态计算平均 forward KL。

## Toy 来源

**LeetPaper 教学改编。** LeetPaper 定长分类 Markov 模型，不是原作 T5 或混合高斯实验。 判题 harness 调用本次提交的函数，保存由其产生的指标与曲线。Lab 静态预览来自参考实现，Run/Submit 的产物来自你的代码。`experiment.py` 与 `tests/` 已提供实验和判题脚手架。

## 来源与验证

[TRL 实现（独立于论文作者；不是本题参考答案来源）](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py)。题目实现与合成数据由 LeetPaper 编写；论文链接固定公式依据的版本。

T0 验证接口，T1 对照参考值，T2 检查数学性质与梯度，T3 用你的实现运行 toy。在在线编辑器补全 starter 后 Run 或 Submit，无需部署完整大模型训练系统。
