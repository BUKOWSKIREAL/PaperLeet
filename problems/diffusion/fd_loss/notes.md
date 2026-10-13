# 参考说明

二维正定矩阵有一个独立恒等式：

$$
\operatorname{tr}\sqrt{\Sigma_r\Sigma_g}=
\sqrt{\operatorname{tr}(\Sigma_r\Sigma_g)+2\sqrt{\det\Sigma_r\det\Sigma_g}}.
$$

测试对这个标量公式求导，不依赖提交中的特征值算法。不可交换的协方差可以区分正确对称夹积与直接对 `Sigma_r @ Sigma_g` 使用 Hermitian 求解器的错误。坐标变换和对称性测试也会暴露丢失非对角项的问题。测试刻意包含重复正特征值：平方根特征值之和在正定矩阵上仍可微，但对特征向量反向传播的实现可能不稳定。

参考统计是常量。对其 detach 后使用 `eigh` 构造平方根，不需要通过这些特征向量求梯度。生成统计只进入夹积和 `eigvalsh`。半正定矩阵含零特征值时，前向公式仍成立，但平方根导数可能不存在或不有限；因此本题的梯度约定要求两个协方差都正定。不通过给特征值加 epsilon 或裁剪最终 FD 换成另一个目标。

停止分母梯度后的导数是 `w/(distance+eps)`。若对整个数值比值求导，得到的是 `w*eps/(distance+eps)^2`。因此同时改变分子分母的朴素有限差分会验证错误的函数。测试使用解析梯度；如果自行做有限差分，需要把分母固定在原始值。

源实现为[提交 5c03b811 的 losses.py](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/frechet_distance/losses.py) 中的 `precompute_sigma_ref_sqrt`、`_compute_trace_term`、`compute_frechet_distance_loss`；归一化位于 [main_fd.py](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/main_fd.py)。教学接口保留 float32/float64，不把最终标量强制转换为 float32。每次调用重新计算小型冻结参考平方根；生产代码会预计算它。多设备收集、非有限值回退、预训练特征网络和采样不在本题范围内。

教学实验使用对称求积点表达小型仿射高斯的精确矩，脚手架提供已停止梯度的 EMA 历史。独立评估器使用上面的二维公式，优化过程调用提交函数。实验同时记录原始和归一化目标，因为接近一的归一化值不能直接反映距离缩减。这些参数与有限生成器是教学选择，不是论文视觉生成基准。

错误变体分别忽略生成相关性、给 `eigvalsh` 非对称输入、对归一化分母求梯度，以及泄漏真实参考统计梯度。
