# 参考说明

原始二阶矩同时包含组内方差和组间均值变化。更新后的协方差也可以写成

$$
\Sigma=\beta\Sigma_{old}+(1-\beta)\Sigma_b+
\beta(1-\beta)(\mu_{old}-\mu_b)(\mu_{old}-\mu_b)^\top.
$$

直接平均协方差会丢失最后的外积项。T1 的独立答案给旧样本和当前样本分别分配权重，然后对显式总体的中心化外积求和，不照抄被测实现。

对 `mean.sum()+covariance.trace()`，当前第 i 行特征的梯度是 `(1-beta)/B * (1 + 2*(x_i-mean))`。对 `||mean||^2+trace(covariance)`，它简化为 `2*(1-beta)/B*x_i`，因为该目标等于原始二阶矩的迹。Toy 用后者检查当前微批次的计算图。

实现依据是[官方 queue.py，提交 5c03b811](https://github.com/Jiawei-Yang/FD-Loss/blob/5c03b8112fec8b9432631e4ce053c0d918cc24bc/frechet_distance/queue.py) 中的 `_build_feats_stats_ema`，以及论文 Algorithm 1、Eq.(3)-(5)。官方实现用 float64 计算统计量；本 CPU 题保留调用方的 float32/float64 类型。函数额外返回原始二阶矩，方便脚手架显式 detach 并保存。题目不包含特征队列、原地模块状态、多设备收集或图像骨干网络。

官方队列分支覆盖最旧特征，中心化平方和除以 `N-1`；EMA 分支混合已经归一化的原始矩，重建总体协方差时没有 Bessel 因子。历史已经暖启动，不需要从零初始化 EMA 的偏差修正。

Toy 的已知总体暖启动是受控设计，用于展示历史信息如何影响统计量与梯度。它不验证生成器收敛、不选择论文生产配置的 decay，也不复现 ImageNet 消融。四个错误变体分别覆盖中心化二阶矩、调换 EMA 权重、历史梯度泄漏和用当前批次均值中心化混合二阶矩。
