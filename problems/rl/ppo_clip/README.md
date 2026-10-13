# PPO 截断策略目标

## 1. 这篇论文解决什么

同一批轨迹训练多轮会让策略远离采样时的分布。PPO 用概率比的悲观截断目标实现简单的一阶更新；核心见[论文式 7 与式 9](https://arxiv.org/pdf/1707.06347)。

本模块目标：实现悲观截断目标，验证正负 advantage 下的四种梯度情形。 只实现 clipped_surrogate；价值损失、熵、归一化与训练循环已提供。 完成后记为这篇论文的一个核心模块。

## 2. 核心公式与题面

只需实现 `clipped_surrogate(new_logp, old_logp, advantages, clip_eps=0.2)`。三个输入都是 `[B]`，返回用于最小化的策略损失标量。`advantages` 已经准备好，函数内部冻结它与 `old_logp`，保留 `new_logp` 的梯度。

$$
r=e^{\log\pi_\theta-\log\pi_{old}},\qquad
L^{PG}=-\mathbb E[\min(r\hat A,\operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)].
$$

正 advantage 遇到过大的概率比时停止奖励进一步提高概率；负 advantage 遇到过小的概率比时停止奖励继续降低概率。反向移动仍会受到惩罚，所以需要保留未截断分支与截断分支中的较小值。

starter 已完整提供 `ppo_loss`。它调用你的核心函数，负责 advantage 归一化、value clipping、熵和诊断量，最后接入现成的 bandit 实验。保留这段辅助代码即可。

## 3. 从公式到代码

```text
三个 [B] 向量 → 冻结 old_logp 和 advantages
log-prob 差取 exp → ratio [B]
分别计算 ratio * advantage 与 clipped_ratio * advantage
逐元素取较小值 → 对 batch 求负均值 → 标量
```

核心检查是 ratio 高于/低于截断范围、advantage 正/负时的四种梯度。提供的 wrapper 只使用核心函数返回的策略损失；实现这个函数即可完成整套测试。

## 4. 已提供的训练辅助与约定

| 项目 | 已提供行为 |
|---|---|
| advantage 归一化 | 样本标准差加 $10^{-8}$；单样本标准差视为零 |
| value loss | 可选 value clipping，两个平方误差取最大值后乘 $1/2$ |
| entropy | 可选熵奖励；缺省为零 |
| diagnostics | detach 的 `approx_kl = mean(r-1-log(r))` 与 `clipfrac` |

这些辅助项沿用现有判题约定，测试保留它们的集成检查。你需要填写的只有策略截断目标。PPO clipping 不构成严格的 KL 上界；bandit 中的 KL 门槛只针对固定学习率和固定 batch 的小实验。

## 5. 交互可视化

[打开截断目标](#viz-clip_objective)：调节 $\epsilon$，分别看正负 advantage 的曲线和零梯度区，移动概率比观察梯度方向。

[打开信赖域实验](#viz-trust_region)：对照参考解、仅截断版本和错误 `max` 版本的逐 epoch KL，查看目标形式如何改变更新幅度。

## 6. 怎么运行

在浏览器中打开本论文的 **PPO 截断策略目标** 模块，填写 `clipped_surrogate`，保留已提供的 `ppo_loss`，然后运行测试。

```bash
paperleet init rl/ppo_clip
paperleet test rl/ppo_clip
```

本地编辑 `workspace/rl/ppo_clip.py`。先查看核心函数与四象限梯度测试，再查看现成 bandit 实验。

## 7. 延伸

下一题 [DPO](../dpo/README.md) 从偏好直接得到策略目标。也可继续阅读 TRPO，比较真实约束优化与 PPO 代理目标的关系。
