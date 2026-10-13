# 题解

按本题布局，激活通道除以 lambda，权重**行**乘以 lambda。截断 SVD 可取 `left=U[:,:r]*singular[:r]`、`right=Vh[:r]`，等价分解也可；先计算残差，再量化。Eq. (5) 的高精度支路接收 xs，低比特支路接收 Q(xs)。两支路都用量化激活，即使满秩也无法恢复精确线性输出。

## 来源与数值范围

[v1 §4.2](https://arxiv.org/html/2411.05007v1) 将平滑权重写作 `W diag(lambda)`，但其 W 定义为 `[M,N]`。该布局维度一致的式子是 `diag(lambda) W`，本题采用后者，并用非方阵测试避免转置歧义。输出遵守 Eq. (5)，测试比较因子乘积，不限定奇异向量符号。

[作者仓库](https://github.com/mit-han-lab/deepcompressor/)是校准与推理系统。本题独立实现给定平滑系数、一次 SVD，并将 Eq. (1) 固定为张量级对称整数 fake quantization；不模拟生产校准配置、低秩迭代细化、group/FP4 变体、存储打包或 Nunchaku 融合内核。CPU float32/float64 代表高精度支路，不能据此声称硬件加速或存储压缩。

受控离群值 toy 计算真实输出误差及残差奇异谱，仅展示选定矩阵；不保证任意输入增大 rank 后量化误差都单调下降。独立 oracle 使用已知旋转奇异向量与 Python 标量 rounding，覆盖 rank=0/满秩和零张量。四个 mutant 对应反向平滑、高精度激活被量化、完整权重被重复量化、遗漏奇异值。
