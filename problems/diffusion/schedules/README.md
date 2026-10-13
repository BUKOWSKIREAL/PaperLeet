# 噪声调度与前向加噪

## 1. 论文解决什么问题

DDPM 将复杂数据逐步变成高斯噪声，再学习反向恢复数据。前向过程的闭式表达式让训练可以随机挑选任意时间，而不必真正执行之前的每一步。

## 2. 核心公式

单步噪声方差为 $\beta_t$，保留信号的比例为 $\alpha_t=1-\beta_t$，累计信号为 $\bar\alpha_t=\prod_{s=1}^t\alpha_s$。由于每一步噪声独立，累计方差可以合并：

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

这里 $x_0,\epsilon$ 是 `[B,...]`，时间 t 为 `[B]`，调度数组为 `[T]`。当数据和噪声均为独立单位高斯，两项方差相加恰好为一。因此数据结构消失时，点云整体尺度仍可能不变。闭式边缘分布见 [DDPM 式 4](https://arxiv.org/abs/2006.11239)。

线性 beta 调度便于实现，但大部分信号可能提前消失。[Improved DDPM 式 16](https://arxiv.org/abs/2102.09672) 通过 cosine 曲线规定累计信号：

$$f(\tau)=\cos^2\left(\frac{\tau+s}{1+s}\frac\pi2\right),\qquad\beta_i=\min\left(1-\frac{f((i+1)/T)}{f(i/T)},0.999\right).$$

对数信噪比是 $\log\bar\alpha-\log(1-\bar\alpha)$，更容易看出接近纯噪声的尾部变化。

## 3. 从公式到代码

```text
生成 betas [T]，使用 float64
alphas = 1 - betas；沿时间维累乘得到 alpha_bars
在累计量前补 1 并去掉末项，得到 alpha_bars_prev
按每行 t 索引，重塑为 [B,1,...,1]
把选出的系数转到 x0 的 dtype/device，再计算前向样本
```

实现 `linear_betas`、`cosine_betas`、`make_schedule` 和 `q_sample`。make_schedule 还需返回平方根系数与 log_snr，字段名见 starter。不要让一个样本的时间索引意外控制整个 batch。

## 4. 本题约定

| 项目 | 本题 | 常见实现 |
| --- | --- | --- |
| 下标 | i=0 对应论文 t=1 | 论文通常从 1 开始 |
| 调度精度 | float64 | 训练时常将提取的系数转为 float32 |
| 线性 beta | 固定首尾 1e-4、0.02 | improved-diffusion 会按步数缩放端点 |
| cosine beta | 上限 0.999 | improved-diffusion 同样限制最大 beta |

## 5. 交互可视化

<viz id="schedule_curves" />

拖动步数与偏移 s，对比两种调度的累计信号与 log-SNR。末端数值不为零意味着初始标准高斯只是近似边界。

<viz id="forward_cloud" />

拖动时间，在同一组八高斯数据及噪声上对比结构消失速度，避免把随机样本差异误认为调度效果。

## 6. 怎么运行

```bash
paperleet init diffusion/schedules
paperleet test diffusion/schedules
```

测试既检查精确端点，也用十万个样本验证方差保持；统计规律用于发现遗漏平方根这种形状正确但含义错误的实现。

## 7. 延伸

完成后进入参数化互转，再实现 DDPM 和 DDIM。关于终端 SNR 与时间子序列的影响，可阅读 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)。
