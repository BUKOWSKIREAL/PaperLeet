# x₀ / ε / v 参数化互转

## 1. 论文解决什么问题

扩散网络可以预测干净数据、加入的噪声，也可以预测速度。Salimans 与 Ho 在少步蒸馏中研究这些参数化：同一个训练样本，换一种预测目标会改变端点稳定性与误差权重。

## 2. 核心公式

设 $x_0,\epsilon,x_t\in\mathbb R^{B\times D}$，$\alpha=\sqrt{\bar\alpha_t}$、$\sigma=\sqrt{1-\bar\alpha_t}$ 可广播到输入形状，且 $\alpha^2+\sigma^2=1$。本题的 alpha 是标准差系数，和 D1 中单步 alpha 的含义不同。

$$x_t=\alpha x_0+\sigma\epsilon,\qquad v=\alpha\epsilon-\sigma x_0.$$

$$x_0=\alpha x_t-\sigma v,\qquad\epsilon=\sigma x_t+\alpha v.$$

令 $\alpha=\cos\phi,\sigma=\sin\phi$，就能直接求导得到 $v=dx_t/d\phi$。逆变换是二维正交旋转，因而不需要除法。相反，由噪声恢复数据要除以 alpha，由数据恢复噪声要除以 sigma，在各自接近零的端点会放大预测误差。对同一个错误预测，三个表示的误差满足

$$\|\hat v-v\|^2=\frac{\|\hat x_0-x_0\|^2}{\sigma^2}=\frac{\|\hat\epsilon-\epsilon\|^2}{\alpha^2}.$$

这解释了预测目标为何也隐含损失权重。以上旋转关系见 [Progressive Distillation 附录 D，式 26、30](https://arxiv.org/html/2202.00512v2#A4)。

## 3. 从公式到代码

实现 starter 中六个函数。所有函数都保持输入 shape 与 dtype，不沿 batch 聚合。

```text
输入 pred、x_t: [B,...]，alpha、sigma: 可广播的系数
根据 src 恢复 x0 或 epsilon
根据 dst 返回目标表示
src == dst 时返回原预测
涉及除法时，将分母下限设为 1e-8
```

`convert` 的来源与目标均在 `x0 / eps / v` 中选择；未知字符串抛出 `ValueError`。同参数化转换不应产生额外数值误差。尤其要直接实现 v 到 x0 与 epsilon 的转换，保持它们在纯信号、纯噪声两端的稳定性。

## 4. 本题约定

| 项目 | 本题 | 常见调度器接口 |
| --- | --- | --- |
| 输入系数 | 已开方的 alpha、sigma | 常保存 alpha_cumprod，再在内部开方 |
| 输出 | 直接返回 Tensor | diffusers 调度器通常返回 step 输出对象 |
| 端点除法 | clamp 到 1e-8 | 由具体调度与稳定性策略决定 |
| 数据维度 | 任意可广播形状 | 图像通常使用 B×C×H×W |

## 5. 交互可视化

<viz id="param_circle" />

拖动旋转角 phi，观察位置向量与切向量：切向量的符号由旋转方向决定。这里画的是系数平面；高维独立噪声与数据近似正交时，也可以直观理解为样本平面。

<viz id="error_amplification" />

拖动时间，查看 $1/\alpha$ 和 $1/\sigma$ 的放大系数，理解某个参数化在端点更难训练的原因。

## 6. 怎么运行

```bash
paperleet init diffusion/param_conversion
paperleet test diffusion/param_conversion
```

T2 会验证六种往返、有限差分、端点和误差权重恒等式；报告中的误差位置能帮助分辨广播错误与公式错误。

## 7. 延伸

下一题实现 DDPM 后验，把 epsilon 预测真正接入采样。D5 会出现另一种速度：它对线性时间求导，与本题对旋转角求导的 v 不应混用。原始噪声参数化见 [Ho 等，2020](https://arxiv.org/abs/2006.11239)。
