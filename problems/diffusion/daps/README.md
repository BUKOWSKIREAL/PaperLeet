# DAPS 解耦后验采样

实现 [DAPS 式 (7) 与 Algorithm 1](https://arxiv.org/abs/2407.01521) 的局部后验采样核心。在当前噪声层，由已提供的 probability-flow ODE 输出 `x0_hat` 构建干净样本的高斯近似，在干净空间中结合观测做 MCMC，再独立加噪进入下一层。

线性观测 `y = A x + measurement_noise` 下：

$$
g(x)=\frac{\hat x_0-x}{r^2}+\frac{A^T(y-Ax)}{\beta^2},\quad
x' = x+\eta g(x)+\sqrt{2\eta}\epsilon,\quad
x_{\mathrm{next}}=x_0+\sigma_{\mathrm{next}}\epsilon'.
$$

`r`、`beta` 表示标准差；两份独立标准高斯噪声由调用者传入。下一噪声层以 MCMC 的干净样本为中心，标准差直接使用 `sigma_next`。

## 实现接口

| 函数 | 输入 | 输出 |
|---|---|---|
| `posterior_score(x, x0_hat, operator, measurement, r, beta)` | `x,x0_hat: (B,D)`，`operator: (M,D)`，`measurement: (B,M)` | `(B,D)` 条件 score |
| `langevin_step(x, score, eta, noise)` | 同形状张量与非负步长 | 一次 Langevin 更新 |
| `renoise(x0, sigma_next, noise)` | 同形状张量与非负噪声标准差 | 下一层噪声样本 |

批次中每个样本为行向量，预测观测为 `x @ operator.T`。score 是逐样本导数，不除以 batch 大小或坐标数量。MCMC 内层循环始终固定 `x0_hat`。保留 dtype/device，不修改输入；验证代码已提供，测试使用 CPU float32/float64。样本之间不进行隐式广播。

ODE、观测矩阵、调度、随机数和循环都是脚手架，学习者实现上述三个数学内核。无需下载模型或训练扩散网络。

## Toy：可解析的局部条件分布

**来源分类：teaching adaptation。** 二维 ODE 估计位于原点，高斯条件近似使用 `r=1`；观测为 `x[0]+0.5*x[1]=1.5`，测量标准差 `0.5`。4,096 条链向量化执行 240 步 Langevin，再独立以 `sigma_next=0.3` 加噪。

harness 解析计算条件均值与协方差，也计算有限步长 Langevin 的稳态协方差，显式区分离散步长偏差。图表展示观测一致性、条件均值收敛和协方差。这只验证一次 DAPS 局部转移，不代表完整逆问题求解，也不声称复现论文 Appendix G 的非线性 toy。

T0 检查接口；T1 对独立标量能量作数值求导并验证给定噪声的更新；T2 检查高斯驻点、批次独立性、观测零空间、噪声矩与组合转移；T3 调用你的函数生成实验图表。预览使用参考实现。
