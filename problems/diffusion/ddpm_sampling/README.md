# DDPM 后验与祖先采样

## 1. 论文解决什么问题

DDPM 将生成过程拆成许多小的去噪步骤。训练时可解析的高斯后验提供学习目标，采样时网络预测的噪声决定每一步往哪里走。

## 2. 核心公式

已知干净数据后，$q(x_{t-1}\mid x_t,x_0)$ 是高斯。它的均值由当前样本与干净数据的加权和组成：

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

网络看不到真实 x0，因此先由噪声预测恢复它：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad x_{t-1}=\tilde\mu_t(x_t,\hat x_0)+\sigma_tz.$$

$x_t,\hat\epsilon,z$ 都是 `[B,...]`，t 是 `[B]`，每一行可以选择不同时间。方差是 `[B,1,...]`，由广播扩展到样本。后验及噪声参数化见 [DDPM 式 6、7、11 与算法 2](https://arxiv.org/abs/2006.11239)。

本题将网络换成八高斯混合分布的精确 score，得到 $\epsilon^*=-\sqrt{1-\bar\alpha_t}\nabla\log p_t$。因此无需训练，也能检验采样结果是否恢复原分布。输出仍有有限步离散化和有限样本误差，不能期待经验 MMD 恰好为零。

## 3. 从公式到代码

```text
q_posterior: 按 [B] 时间提取系数，组合 x0 和 xt
p_sample_step: 从 eps 恢复 x0，计算均值，选择方差
把 t=0 的噪声乘零
ddpm_sample: 初始化标准高斯 [B,D]
按 T-1,T-2,...,0 逐步调用 eps_model，再更新样本
```

实现三个公开函数，`return_traj=True` 返回 `(samples, trajectory)`；轨迹形状为 `[T+1,B,...]`，包含初始噪声。采样随机数必须来自调用者提供的 generator。

## 4. 本题约定

| 项目 | 本题 | improved-diffusion |
| --- | --- | --- |
| x0 范围 | 二维数据不裁剪 | 图像常裁剪到 [-1,1] |
| 方差选项 | posterior 或 beta | 还支持网络学习方差 |
| 最后一步 | 两种选项均不加噪 | 同样屏蔽最终噪声 |
| 首个 log 方差 | 不计算 log 方差 | 用下一项方差填充后取 log |

## 5. 交互可视化

<viz id="reverse_particles" />

播放反向轨迹，切换 posterior 与 beta 方差，观察粒子从噪声聚集到八个模式；灰色中心表示真实数据位置。

<viz id="variance_compare" />

拖动时间比较 beta 与后验方差，最后一步两者差异最大，但都必须停止注入噪声。导入自己的报告后，可叠加采样点、真实样本以及 MMD 阈值。

## 6. 怎么运行

```bash
paperleet init diffusion/ddpm_sampling
paperleet test diffusion/ddpm_sampling
```

T2 以精度相加独立推导后验。T3 用 1000 步、2000 个样本检查两种方差，MMD 阈值来自固定种子参考值乘 1.5，并检查模式占比和离群点。

## 7. 延伸

下一题 DDIM 保留同一个噪声预测器，却能沿时间子序列跳步。想研究学习方差，可继续阅读 [Improved DDPM](https://arxiv.org/abs/2102.09672)。
