# DistiLLM-2 双源非对称蒸馏

实现 [DistiLLM-2 v2 §2.2 与 Eq. (2)](https://arxiv.org/html/2503.07067v2) 的损失。教师回答（`_t`）与学生回答（`_s`）是两套不同上下文；每套都提供教师和学生 logits。

## 函数

记教师分布 `p=softmax(teacher_logits)`、学生分布 `q=softmax(student_logits)`：

$$\mathrm{SKL}_\alpha(p,q)=\sum_v p_v\log\frac{p_v}{\alpha p_v+(1-\alpha)q_v},$$
$$\mathrm{SRKL}_\alpha(p,q)=\sum_v q_v\log\frac{q_v}{(1-\alpha)p_v+\alpha q_v}.$$

1. `skew_kl(teacher_logits, student_logits, alpha)`：输入同形 `[...,V]`，返回每个位置的 `[...]`。
2. `skew_reverse_kl(...)`：接口相同，按上式交换散度方向及混合比例。
3. `distillm2_loss(teacher_t, student_t, mask_t, teacher_s, student_s, mask_s, alpha_t=.1, alpha_s=.1, beta=.5)`：返回标量：

$$L=\frac{(1-\beta)\sum_{b,t}m^t_{bt}\,\mathrm{SKL}_{\alpha_t}(p^t_{bt},q^t_{bt})+\beta\sum_{b,t}m^s_{bt}\,\mathrm{SRKL}_{\alpha_s}(p^s_{bt},q^s_{bt})}{2B}.$$

每套 logits 为 `[B,T,V]`，两套可有不同的 `T/V`，非空批次 `B` 必须相同。布尔 mask `[B,T]` 的 `True` 表示有效回答 token。先求 token 与词表之和，再除以 `2B`，**不要按有效 token 数平均**。空回答和全 mask 回答贡献零。三个系数均为 `[0,1]` 内有限数；词表非空，logits 有限，每对 dtype/device 相同。保留 float32/float64、非连续输入和输入值；两套学生 logits 需要梯度，教师 logits 冻结。稳定的 log-space 辅助函数及校验已提供，无需加载模型或生成回答。

## Toy 实验

用提交的 loss 优化两套三词词表的合成条件分布，T3 绘制 60 次更新的损失与两个散度。这是双源目标的**教学改编**，不是论文 Figure 1 的长尾数据或语言模型基准。固定系数，不运行 curriculum；对已给上下文使用全条件词表 KL，并明确保留 Eq. (2) 的 `1/(2B)`。范围说明见题解。
