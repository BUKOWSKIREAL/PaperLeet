# 实现说明 — QLoRA NF4 分块与可训练 Adapter

实现分块量化、反量化与冻结基座 adapter 前向。**已提供 `NF4` 元组**是固定的非对称 16 值码本：索引 7 为精确零，正负非零值不镜像。按行优先展开权重，每块除以自身绝对值最大值，再选最近码本值；距离相同时选较小索引。全零块 scale=0，索引全为 7；尾部不足整块也单独保存 scale。

返回原权重形状的 uint8 **未打包索引**和一维浮点 scale。反量化把每个 scale 重复到相应块。量化不建立梯度；反量化冻结 scale。前向输入 A[r,in]、B[out,r]，按 alpha/r 缩放，保留输入与 adapter 梯度，冻结可选 bias。默认 block_size=64，测试用小块便于检查。码本常量已在 starter 中提供，无需背诵。

Toy 在冻结量化映射上训练 adapter。这里每个索引占一字节，不是半字节；未包含 nibble 打包、双重量化、分页优化器与 CUDA 内核，所以指标不能证明完整论文显存效果。较大系统见 [QLoRA 官方训练仓库](https://github.com/artidoro/qlora)。

## 错误变体

- 用均匀码本替代 NF4。
- 每块误用全张量的 absmax。
- 错误地训练冻结量化 scale。
- 遗漏 adapter 的 rank 归一化。

每个 mutant 都是相对于参考实现的小幅故意错误。T2 包含独立公式或梯度期望，接受结果不只依赖参考值相等。T3 使用传入的学员实现产生结果。

依据：[2305.14314v1](https://arxiv.org/html/2305.14314v1#S3)；[官方量化库，固定 0.43.3 的 NF4 码本](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/functional.py)。实验来源：PaperLeet 未打包 NF4 索引合成回归；不衡量完整 QLoRA 显存节省。
