# VAR 多尺度残差量化

实现 [VAR](https://arxiv.org/html/2404.02905v2) 使用的冻结潜空间 tokenizer：每个尺度量化前面尺度尚未解释的特征。算法 1–2 给出编码与重建过程，[官方代码](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) 确定了 area 与 bicubic 插值约定。

## 需要实现

`multiscale_quantize(features, codebook, scales, projections) -> (token_maps, reconstructions)`

初始化残差 `e=features` 和最大分辨率重建 `f_hat=0`。依次处理由小到大的尺度 `n_k`：

1. 调用已提供的 `resize_residual(e,n_k)`，逐空间位置在共享码本中寻找平方欧氏距离最小的向量。距离完全相同时选择最小索引。
2. 调用 `lift_codes(indices,codebook,projections[k],H)`，得到最大分辨率贡献 `h_k`。
3. 更新 `e=e−h_k`、`f_hat=f_hat+h_k`，保存整数 token 图和 `f_hat` 的独立快照。

返回 `K` 个 int64 图 `[B,n_k,n_k]`，以及 `K` 个累计浮点重建 `[B,C,H,H]`。最后一个尺度处理剩余的全分辨率残差，不能在每个尺度都重新量化原始特征。

`decode_multiscale(token_maps, codebook, projections) -> Tensor`

将每个 token 图放大、投影到最大分辨率后相加，返回 `[B,C,H,H]`。同一组 token 与投影的解码结果，必须等于编码时最后一个累计重建。

## 已提供 helper 与约定

`resize_residual` 使用 area 插值；`lift_codes` 完成码本查表、`align_corners=False` 的 bicubic 放大、通道矩阵乘法。二者已提供。本教学题用给定 `[K,C,C]` 矩阵替代原系统已学习的空间 `Phi` 层；码本选择不使用余弦归一化。

`features` 是有限、非空、正方形 `[B,C,H,H]`；共享码本为 `[V,C]`，`V,C>0`。浮点输入使用 CPU float32/float64，dtype/device 一致，允许非连续张量。尺度是从 `1` 到 `H` 严格递增的正整数，允许 `K=1,H=1`。解码 token 图遵守同样尺度约定，使用有效范围内的 int64 索引。不得修改输入，各尺度快照不得共享同一可变存储。

这里 tokenizer 已冻结，模板保留 `no_grad`。无需实现直通梯度、码本更新、commitment loss、编码器或图像解码器。

## 小实验

教学改编使用一张 8×8 潜特征图，包含粗粒度区域与小幅交替细节，配以九个标量码和恒等投影。你的编码、解码函数生成重建误差曲线与一行潜特征对照，检查细节恢复及编码/解码一致性。实验不假定任意码本、投影的每一个中间尺度都一定降低误差，也不声称复现 ImageNet 训练。
