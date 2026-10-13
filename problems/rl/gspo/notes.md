# 实现说明 — GSPO 序列比率与截断

`logp`、`old_logp`、二元 `mask` 形状为 `[B,L]`，`advantages` 为 `[B]`。先对有效 log-ratio 求和、除以有效长度，再取指数。每个回答只截断这个序列比率，与冻结 advantage 相乘取较小项，最后对 batch 求负均值。

旧策略与 advantage 不参与梯度。全 padding 回答比率为 1、损失贡献为 0，仍计入 batch 大小。必须在指数运算前屏蔽 padding。重复一个回答内部的 token 不改变结果。本题实现 GSPO，不是 GSPO-token。默认 `.2` 截断是便于观察的教学设置，不代表 Qwen 生产训练参数。

实验使用 token 比率 `[2, 0.5]`，它们的几何比率为 1，两个有效 log-probability 的梯度相同。正 advantage 的比率 1.5 和负 advantage 的比率 0.5 均被截断。产物保存学员实现计算的比率和梯度矩阵。

## 错误变体

- 对 token 比率做算术平均，误把几何序列比率改成算术均值。
- 遗漏回答长度归一化。
- 错误地向采样旧策略传播梯度。
- 截断代理目标错误取最大值。

每个 mutant 都是相对于参考实现的小幅故意错误。T2 包含独立公式或梯度期望，接受结果不只依赖参考值相等。T3 使用传入的学员实现产生结果。

依据：[2507.18071v1](https://arxiv.org/html/2507.18071v1#S4.SS1)；[作者说明（尚未验证作者训练代码发布）](https://qwenlm.github.io/blog/gspo/)。实验来源：PaperLeet 自编的三回答解析实验，不是 Qwen 训练结果。
