# PPO Clipped Surrogate

## 1. 这篇论文解决什么

同一批轨迹训练多轮会让策略远离采样时的分布。PPO 用概率比的悲观截断目标实现简单的一阶更新；核心见[论文式 7 与式 9](https://arxiv.org/pdf/1707.06347)。

## 2. 核心公式与题面

实现 `ppo_loss`，所有输入是 `[B]`，返回 `loss, pg_loss, v_loss, entropy, approx_kl, clipfrac` 六个标量。当前 log-prob 和价值保留梯度，旧策略、旧价值、回报和 advantage 是固定目标。

$$
r=e^{\log\pi_\theta-\log\pi_{old}},\qquad
L^{PG}=-\mathbb E[\min(r\hat A,\operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)].
$$

正 advantage 遇到过大的概率比时停止奖励进一步提高概率；负 advantage 遇到过小的概率比时停止奖励继续降低概率。反向移动仍会受到惩罚，所以不能单独把概率比截断后乘 advantage。

$$
V_c=V_{old}+\operatorname{clip}(V-V_{old},-c,c),\quad
L^V=\tfrac12\mathbb E[\max((V-R)^2,(V_c-R)^2)].
$$

总损失 $L=L^{PG}+c_vL^V-c_e\mathbb E[H]$，按最小化实现。默认 $\epsilon=c=0.2,c_v=0.5,c_e=0$。`vf_clip=None` 关闭价值截断，`entropy=None` 把熵项视为零。

## 3. 从公式到代码

```text
输入: 六个 [B] 向量，可选 entropy [B]
冻结 old_logp、old_values 与训练目标
可选归一化 advantage: [B]
逐样本构造 ratio、截断 surrogate、两个 value error
各项按 batch 平均 → 标量 loss
在 no_grad 下计算 approx_kl 与 clipfrac
```

`approx_kl = mean(r-1-log(r))`，`clipfrac = mean(abs(r-1)>epsilon)`。前者逐样本非负；后者只是超出区间的比例，并不等价于梯度真正被截掉的比例。

## 4. 本题约定

| 项目 | 本题 | CleanRL `ppo.py` |
|---|---|---|
| advantage 标准差 | 无偏样本标准差，外加 $10^{-8}$ | 相同 |
| 单样本归一化 | 标准差视为零，归一后为零 | 本题补充边界规则 |
| value clipping | 可关闭，默认 0.2 | 默认开启、共用 clip coefficient |
| 旧量梯度 | 函数内部 detach | 通常由 rollout 存储保证 |

对照来源为 [CleanRL 源码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) 与 [37 个实现细节](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)，核对日期为 2026-10-13。PPO 截断并不构成严格的 KL 上界；本题 KL 门槛只针对固定学习率、固定 batch 的小实验，训练循环没有偷偷加入 KL 早停。

训练中 advantage 的正负号决定应该提高还是降低该动作概率；先把这个方向判断清楚，再检查截断边界。建议在纸上画出四个象限，手算一个样本的梯度，与报告中的期望值逐项对照。

## 5. 交互可视化

[打开截断目标](#viz-clip_objective)：调节 $\epsilon$，分别看正负 advantage 的曲线和零梯度区，移动概率比观察梯度方向。

[打开信赖域实验](#viz-trust_region)：对照参考解、仅截断版本和错误 `max` 版本的逐 epoch KL，查看目标形式如何改变更新幅度。

## 6. 怎么运行

```bash
paperleet init rl/ppo_clip
paperleet test rl/ppo_clip
```

编辑 `workspace/rl/ppo_clip.py`。四象限梯度测试定位 policy clipping；价值下界测试定位 value clipping；五臂 bandit 实验验证多轮训练后最优臂概率超过 0.9。

## 7. 延伸

下一题 [DPO](../dpo/README.md) 从偏好直接得到策略目标。也可继续阅读 TRPO，比较真实约束优化与 PPO 代理目标的关系。
