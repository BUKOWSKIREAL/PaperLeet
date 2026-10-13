# 参考实现说明

这里的高斯近似对应 `p(x0 | xt)`，不是直接近似全局无条件数据先验。完整 DAPS 外层循环会在每个噪声层重新运行 ODE 得到 `x0_hat`，并由调度设置 `r`。本题显式接收这些量，将论文的可微观测函数收窄为矩阵，以便独立验证似然 score。

[作者 sampler](https://github.com/zhangbingliang2019/DAPS/blob/main/cores/mcmc.py) 的 Gaussian 分支先得到 `(x0hat-xt)/sigma^2`，再加 `(xt-x)/sigma^2`，相加即 `(x0hat-x)/sigma^2`；Langevin 更新使用 `sqrt(2*lr)`。本题显式保留论文中的一般参数 `r`，作者默认 Gaussian 实现是 `r=sigma` 的特例。

教学条件分布的精度矩阵为 `H = I/r^2 + A.T@A/beta^2`，均值为 `H^-1(x0_hat/r^2 + A.T@y/beta^2)`，真实协方差为 `H^-1`。稳定的未校正 Langevin 更新，其稳态协方差为 `(H-eta*H@H/2)^-1`。toy 同时记录两者，避免把有限步长采样误称为精确后验采样。

Appendix G 使用双分量高斯混合和非线性测量。文中展示的测量公式配合 `y=0`，不能直接解释同段描述的似然模态。2026-10-13 检查的公开仓库文件树中未找到能澄清约定的可运行 synthetic/toy 脚本。因此本题明确使用新设计的线性高斯教学实验，不擅自修改原文 toy；score、Langevin 系数和重加噪内核仍对齐式 (7) 与 Algorithm 1。

显式噪声参数使 harness 可重复，独立抽样由调用者负责；组合测试和实验使用不同的 MCMC 噪声与下一层噪声。本实验不评价真实图像恢复、全局混合或预训练模型质量。
