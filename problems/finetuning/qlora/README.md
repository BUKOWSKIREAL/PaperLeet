# QLoRA NF4 分块与可训练 Adapter

实现分块 NF4 量化，在冻结量化基座上训练 adapter。

论文：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314v1)。定位：Section 3: NormalFloat4 and frozen quantized base with LoRA；Appendix E: asymmetric NF4 with exact zero；bitsandbytes 0.43.3 functional.py get_4bit_type: pinned 16-level NF4 codebook。

## 精选核心

三个 CPU 函数；索引用未打包 uint8 便于观察，不包含 4-bit 打包存储、双重量化和分页优化器。

$$
W\approx s_b C_{q(W/s_b)},\qquad y=x\mathrm{dequant}(W)^\top+\frac\alpha r xA^\top B^\top
$$

- `nf4_quantize(weight: Tensor, block_size: int=64) -> tuple[Tensor, Tensor]`
- `nf4_dequantize(codes: Tensor, scales: Tensor, block_size: int=64) -> Tensor`
- `qlora_linear(x: Tensor, codes: Tensor, scales: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, block_size: int=64, bias: Tensor | None=None) -> Tensor`

## 接口与实验

实现分块量化、反量化与冻结基座 adapter 前向。**已提供 `NF4` 元组**是固定的非对称 16 值码本：索引 7 为精确零，正负非零值不镜像。按行优先展开权重，每块除以自身绝对值最大值，再选最近码本值；距离相同时选较小索引。全零块 scale=0，索引全为 7；尾部不足整块也单独保存 scale。

返回原权重形状的 uint8 **未打包索引**和一维浮点 scale。反量化把每个 scale 重复到相应块。量化不建立梯度；反量化冻结 scale。前向输入 A[r,in]、B[out,r]，按 alpha/r 缩放，保留输入与 adapter 梯度，冻结可选 bias。默认 block_size=64，测试用小块便于检查。码本常量已在 starter 中提供，无需背诵。

Toy 在冻结量化映射上训练 adapter。这里每个索引占一字节，不是半字节；未包含 nibble 打包、双重量化、分页优化器与 CUDA 内核，所以指标不能证明完整论文显存效果。较大系统见 [QLoRA 官方训练仓库](https://github.com/artidoro/qlora)。

## Toy 来源

**LeetPaper 教学改编。** LeetPaper 未打包 NF4 索引合成回归；不衡量完整 QLoRA 显存节省。 判题 harness 调用本次提交的函数，保存由其产生的指标与曲线。Lab 静态预览来自参考实现，Run/Submit 的产物来自你的代码。`experiment.py` 与 `tests/` 已提供实验和判题脚手架。

## 来源与验证

[官方量化库，固定 0.43.3 的 NF4 码本](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/functional.py)。题目实现与合成数据由 LeetPaper 编写；论文链接固定公式依据的版本。

T0 验证接口，T1 对照参考值，T2 检查数学性质与梯度，T3 用你的实现运行 toy。在在线编辑器补全 starter 后 Run 或 Submit，无需部署完整大模型训练系统。
