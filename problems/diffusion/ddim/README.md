# DDIM 确定性少步采样

## 1. 论文解决什么问题

DDPM 的训练目标只要求各时间的边缘加噪分布，并没有把采样轨迹唯一固定。DDIM 利用这一自由度复用已有噪声预测器，以更少的去噪步骤生成样本。

## 2. 核心公式

设当前累计信号为 $a=\bar\alpha_t$，前一个采样时间的累计信号为 $b=\bar\alpha_{\mathrm{prev}}$，因此 $b\ge a$。先恢复干净预测，再组合确定性方向与随机噪声：

$$\hat x_0=\frac{x_t-\sqrt{1-a}\hat\epsilon}{\sqrt a},\qquad\sigma=\eta\sqrt{\frac{1-b}{1-a}}\sqrt{1-\frac ab}.$$

$$x_{\mathrm{prev}}=\sqrt b\hat x_0+\sqrt{1-b-\sigma^2}\hat\epsilon+\sigma z.$$

所有样本张量为 `[B,D]`；a、b 可以是标量，也可广播到样本形状。eta 等于零时，每一条轨迹由初始噪声唯一确定；eta 等于一且相邻时间步时，随机方差与 DDPM 后验相同。这里必须从方向项中扣掉随机项的方差，否则总噪声会过量。更新和 eta 参数化分别见 [DDIM 式 12、16](https://arxiv.org/abs/2010.02502)。

最后一次更新设置 b=1，把样本送到真正的干净端点。单点分布是很强的检查：完美 x0 预测下，无论采样一步还是五十步，都应精确到达该点。

## 3. 从公式到代码

```text
timesteps[k] = T-1-k*(T//num_steps)，k 从 0 到 n-1
对每个降序时间：预测 epsilon，调用 ddim_step
下一时间不存在时，令 alpha_bar_prev=1
反演：按升序遍历，在当前位置求 epsilon，再推到更噪的时间
```

实现 `ddim_step`、`ddim_timesteps`、`ddim_sample`、`ddim_invert`。采样可返回包含起点与终点的 `[n+1,B,D]` 轨迹。反演采用一阶近似：干净端点没有训练索引，首次使用索引零的 epsilon；之后总在当前状态对应的时间预测。它不会在有限步数下严格求得非线性映射的代数逆。

## 4. 本题约定

| 项目 | 本题 | diffusers DDIMScheduler |
| --- | --- | --- |
| trailing 步距 | T 整除 n | 实数 T/n 后取整 |
| T 能整除 n 时 | 两者相同 | 两者相同 |
| 其他选项 | 仅固定 trailing | 另有 leading、linspace |
| 干净边界 | 最后 b=1 | set_alpha_to_one 控制 |

表格按 [官方调度器源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) 于 2026-10-13 核对。非整除情况优先遵循本题定义。

## 5. 交互可视化

<viz id="eta_steps_grid" />

点击 eta 与步数组合，播放同一初始噪声的轨迹并比较最终 MMD，观察随机性与计算预算的关系。

<viz id="inversion_demo" />

选择数据点与步数，查看反演噪声、重建位置及往返误差；网格更密时通常更接近原样本。

## 6. 怎么运行

```bash
paperleet init diffusion/ddim
paperleet test diffusion/ddim
```

T2 检查确定性、单点端点、后验方差、反演与离散化收敛。T3 用相同八高斯 oracle 检查五十步生成质量，并记录 eta=1 的对照数值。

## 7. 延伸

D5 将离散去噪更新推广到连续速度场。关于少步误差与时间边界，也可阅读 [Progressive Distillation](https://arxiv.org/abs/2202.00512)。
