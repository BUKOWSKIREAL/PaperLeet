# 实现说明 — LoRA 低秩更新与冻结基座前向

实现 `lora_delta(A,B,alpha)` 与 `lora_linear(x,W,A,B,alpha,bias)`。形状为 `A[r,in]`、`B[out,r]`、`W[out,in]`、`x[...,in]`。增量为 `(alpha/r) * B @ A`。前向等于冻结基座线性映射加低秩更新。即使输入基座权重与 bias 的 `requires_grad=True`，也不能向它们传播梯度；`x`、`A`、`B` 的梯度必须保留。

训练脚手架随机初始化 A、零初始化 B。因此初始输出不变，B 有学习信号而 A 的首次梯度为零。对整个基座计算使用 `no_grad()` 会错误删除部分 x 梯度。将 delta 加到 W 即可合并，测试比较合并前后结果。本模块不包含 dropout。

Toy 冻结一个 6×8 映射，用 28 个 adapter 参数拟合已知 rank-2 变化。平台提供输入、优化器和 300 步 CPU 训练。通过表示恢复这个可控目标，不是复现完整语言模型指标。

## 错误变体

- 遗漏除以 rank 的缩放。
- 错误地训练基座权重。
- 误 detach 低秩更新，使 adapter 无法学习。
- 错误移除基座路径对输入的梯度。

每个 mutant 都是相对于参考实现的小幅故意错误。T2 包含独立公式或梯度期望，接受结果不只依赖参考值相等。T3 使用传入的学员实现产生结果。

依据：[2106.09685v2](https://arxiv.org/abs/2106.09685v2)；[微软 LoRA 官方实现](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)。实验来源：PaperLeet rank-2 合成回归，不是语言模型基准。
