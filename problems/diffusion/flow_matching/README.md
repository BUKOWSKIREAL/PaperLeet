# Flow Matching 速度场

## 1. 论文解决什么问题

Flow Matching 学习一个随时间变化的速度场，用常微分方程把噪声搬运到数据。条件训练目标允许直接回归可计算的速度，不必在每次训练时模拟整条生成轨迹。

## 2. 核心公式

本题固定 x0 是噪声，x1 是数据。二者均为 `[B,D]`，时间 t 为 `[B]`，每个样本拥有自己的插值比例：

$$x_t=(1-t)x_0+tx_1,\qquad u=x_1-x_0,\qquad\mathcal L=\operatorname{mean}_{b,d}\left(v_\theta(x_t,t)-u\right)^2.$$

采样从标准高斯开始，沿时间递增积分：

$$x_{k+1}=x_k+\frac1n v_\theta(x_k,k/n),\quad k=0,\ldots,n-1.$$

条件速度 u 是一对端点之间的位移，但网络只看见当前位置和时间。同一位置可能由不同端点对产生，因此它学到的是条件速度的平均值。即使网络达到最优，训练损失一般仍不为零；不可约方差不等于实现错误。梯度等价性及线性条件路径见 [Flow Matching 式 5、9、22、23](https://arxiv.org/html/2210.02747v2)。

单点数据是特殊情况：$v^*(x,t)=(\mu-x)/(1-t)$，代入插值即可得到恒定目标速度，因此 oracle loss 为零，Euler 一步也能到达数据点。一般混合分布的边缘轨迹会弯曲，多步求解才有意义。

## 3. 从公式到代码

```text
训练：独立采 x0、x1 和 t [B]
把 t reshape 到 [B,1]，构造插值 [B,D]
回归 x1-x0，元素平方误差取平均得到标量
采样：从 x0 开始，k=0..n-1 正向循环
在左端点 k/n 调用模型，并按步长 1/n 前进
```

实现时间采样、插值、目标、损失和 Euler 五个函数。`sample_t` 支持均匀与 logit-normal，必须输出开区间内的值。模型接口是 `model(x,t)`；轨迹开关返回 `(samples, trajectory)`，后者包含初始噪声和最终样本。

## 4. 本题约定

| 项目 | 本题 / Lipman | Rectified Flow | SD3 |
| --- | --- | --- | --- |
| 时间零端点 | 噪声 | 源分布，生成时为噪声 | 数据 |
| 时间一端点 | 数据 | 目标分布，生成时为数据 | 噪声 |
| 生成方向 | 0 → 1 | 0 → 1 | 1 → 0 |
| 时间采样 | 默认均匀 | 基础目标均匀 | 研究了 logit-normal 等权重 |

SD3 的反向约定见 [式 13](https://arxiv.org/html/2403.03206v1)。移植代码时要同时核对插值、目标符号和积分方向。

## 5. 交互可视化

<viz id="flow_particles" />

播放粒子流动并拖动时间，查看当前位置的速度场箭头。切换 Euler 步数，比较直线 oracle 与学习到的弯曲轨迹。

<viz id="convention_bug" />

切换反向采样错误，观察同一模型如何生成错误结果；同时查看 loss 曲线，理解 loss 下降不足以证明采样正确。

## 6. 怎么运行

```bash
paperleet init diffusion/flow_matching
paperleet test diffusion/flow_matching
```

T3 在 CPU 双线程上真实训练 TimeMLP：Adam、学习率 1e-3、batch 256、2000 步。生成 2000 个点，检查 MMD、八模式覆盖与离群率；阈值和实测耗时保存在 problem.yaml。

## 7. 延伸

阅读 [Rectified Flow](https://arxiv.org/abs/2209.03003) 了解如何通过重新耦合使轨迹更直。这里使用独立噪声与数据配对，不包含 reflow 或全局最优传输配对。
