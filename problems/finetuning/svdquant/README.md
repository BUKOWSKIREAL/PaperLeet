# SVDQuant 平滑、低秩与残差量化

实现 [SVDQuant v1 §4.2 Eq. (5)](https://arxiv.org/html/2411.05007v1) 的数值核心：把激活离群值迁移到权重，保留高精度主导低秩分量，对残差分支做量化。

## 函数

采用 `x[B,M]`、`weight[M,N]`、正且有限的 `scale[M]`。未量化输出是 `x@weight`，与 PyTorch Linear 的转置权重布局不同。

1. `smooth_inputs(x,weight,scale)` 返回 `(xs,ws)`：`xs=x/scale`、`ws=scale[:,None]*weight`，应保持 `xs@ws=x@weight`。
2. `split_weight(weight,rank)` 返回 `(left[M,r],right[r,N],residual[M,N])`，`left@right` 为截断 SVD 的最佳 rank-r 近似，`residual=weight-left@right`。允许奇异向量符号及等价因子分解不同，覆盖 `rank=0` 和 `rank=min(M,N)`。
3. `svdquant_linear(x,weight,scale,rank,bits=4)`：先平滑，再分解**平滑后的权重**，计算：

$$Y=(\hat X L_1)L_2+Q(\hat X)Q(R).$$

低秩分支必须接收**未量化的平滑激活**；低比特分支量化的是残差 R，不能量化完整平滑权重。已提供 `fake_quantize`：每个张量共享 `s=max(abs(z))/(2^(bits-1)-1)`，`Q(z)=s*clamp(round(z/s),-qmax,qmax)`，中点取偶数，零张量输出零，`2<=bits<=8`。

starter 已提供校验和量化。输入为有限 CPU float32/float64，同 dtype/device，维度非空。保留 dtype/device 与输入值，兼容非连续输入。本题为推理数值接口，不考查 SVD/rounding 梯度。

## Toy 实验

T3 构造含一个 30 倍通道的合成激活，通过提交函数计算输出误差及奇异值曲线，比较直接 W4A4、仅平滑与 rank 0–6 双分支结果。这是 Figures 4–5 的**教学改编**：平滑系数已给定，只做一次 SVD；高精度分支使用输入浮点 dtype，不模拟硬件 FP16。不含模型权重、校准搜索、迭代细化、FP4 编码、整数内核或 Nunchaku 延迟复现。核心同时包括平滑、低秩、残差量化，不将单独 SVD 当作完整系统。
