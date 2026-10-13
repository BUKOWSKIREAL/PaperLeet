# Dr. GRPO · Remove Two Biases

## 1. 这篇论文解决什么

Dr. GRPO 分析组相对优化中的两处归一化：按回答长度平均会改变逐 token 权重，按组内标准差缩放会改变不同问题的权重。[原文第 3.1–3.2 节](https://arxiv.org/html/2503.20783v2#S3.SS2) 给出对应修正。

## 2. 核心公式与题面

本题以前一题为基础，实现 `drgrpo_advantages` 与 `drgrpo_loss`。奖励 `[B*G]` 仍按组相邻排列，三个 log-prob 和 mask 均为 `[N,L]`，advantage 为 `[N]`，$N=BG$。组大小至少为二并整除奖励数。

$$
\hat A_i=r_i-\bar r_g,\qquad
L=-\frac{1}{N L_{max}}\sum_{i,t}m_{i,t}
\left[\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-c,1+c)\hat A_i)-\beta k3_{i,t}\right].
$$

$L_{max}$ 是调用者传入的固定生成长度上限，不是这一批回答的最长长度。$\rho$、$k3$、截断范围与 R4 完全一致。默认 $c=0.2,\beta=0.04$，返回 loss 与有效 token 平均的 `kl_mean, clipfrac, ratio_mean` 三个 detach 诊断量。

长度为五和二十、advantage 相同的负奖励回答，在 GRPO 中逐 token 惩罚之比为四；更长的错误回答得到较轻的单 token 惩罚。改用固定分母后，两个回答的单 token 梯度完全相同。去掉 std 后，奖励乘以常数，advantage 也随之成比例变化，奖励尺度重新成为有意义的训练参数。

## 3. 从公式到代码

```text
rewards [B*G] → [B,G] → 减组均值 → [B*G]
调用已提供的 token_objective → objective/ratio/k3: [N,L]
objective 乘 mask → 全部 token 求和 → 除以 N*max_len
诊断量仍用有效 token 数归一
```

starter 已导入 R4 参考解的 `token_objective(logp, old_logp, ref_logp, advantages, clip_eps, beta)`；它返回 `(objective, ratio, kl)`，其中 objective 已包含减去 $\beta k3$。复用它可以集中注意两处变体差异。聚合前仍需要正确隔离 padding，不能把全零 advantage 的其他回答偷偷引入目标回答的分母。

## 4. 本题约定

| 项目 | 本题 | 原文与常见实现 |
|---|---|---|
| advantage | 只减组均值 | Dr. GRPO 同样移除 std |
| 固定常数 | 外部 `max_len` | 原文建议生成预算，其他固定常数改变梯度尺度 |
| KL 系数 | 保留 R4 的 0.04 | 原文第 2 节与附录 G 采用零 |
| 经验长度趋势 | 展示实验，不对单次趋势判分 | 需多种子验证后才能作为稳定行为门槛 |

这里保留 KL 是为了在教学中隔离两处修改，不应把本题默认值写成原文的训练配置。梯度性质精确可验，而训练得到的平均长度还会受到 EOS、学习率、奖励函数和采样噪声影响，两者需要分别阅读。

## 5. 交互可视化

[打开 token 权重对比](#viz-token_weight_heatmap)：调节两条回答长度，观察 GRPO 与 Dr. GRPO 热图的单 token 权重。

[打开难度偏置](#viz-difficulty_bias)：移动组内正确率，比较标准化与仅中心化的奖励尺度；[长度实验](#viz-t3_length_curves) 展示错误回答长度曲线，并明确标为展示用。

## 6. 怎么运行

```bash
paperleet init rl/dr_grpo
paperleet test rl/dr_grpo
```

编辑 `workspace/rl/dr_grpo.py`。长度梯度测试给出 R4 的对照值；固定上限测试通过改变无关回答的长度，排除对 batch 组成的意外依赖。

## 7. 延伸

回到 [GRPO](../grpo/README.md) 比较 DAPO 的有效 token 平均。进一步实验可以固定采样预算，分别研究归一化对梯度尺度、奖励增长和错误回答长度的影响。
