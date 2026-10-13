# PKPO 奖励变换与无偏策略梯度

依据 [PKPO v5 Eq. (12)、Eq. (33)](https://arxiv.org/html/2505.15201v5)，实现优化 k 次采样中最佳结果的奖励变换。奖励可为任意有限实数；二值奖励下，期望最大值就是 pass@k。

## 函数

输入浮点张量 `[...,N]`，最后一维是一组独立策略采样，`1<=k<=N`。

1. `maxk_estimate(rewards,k)` 返回 `[...]`，计算所有 k 元子集的最大值平均。奖励升序排列为 `g[0:N]` 后：

$$\widehat M_k=\sum_{j=0}^{N-1}\frac{\binom{j}{k-1}}{\binom Nk}g_j.$$

2. `pkpo_weights(rewards,k)` 返回**已 detach、恢复原采样顺序**的系数：

$$a_i=\frac{1}{\binom Nk}\sum_{I:\ |I|=k,\ i\in I}\left[\max_{j\in I}g_j-\max_{j\in I\setminus\{i\}}g_j\right].$$

`k=1` 明确约定空集合 baseline 为零，即 `a_i=g_i/N`。`k>=2` 可排序后用前缀和，避免枚举子集：

$$a_{(i)}=\sum_{j<i}\frac{\binom{j}{k-2}}{\binom Nk}(g_{(i)}-g_{(j)}).$$

无效组合数为零。奖励相同的差值自然为零，不应给并列最大值随意分配 advantage。最后将系数恢复原始顺序。

3. `pkpo_loss(log_probs,rewards,k)` 返回 `-mean_groups(sum_i(log_probs_i*a_i))`。`log_probs` 与奖励同形，是已采样动作的标量 log probability。系数已包含样本归一化，**不要再对 N 取 mean**。梯度仅经过 log probability。

可复用已提供的参数校验、组合数辅助函数。输入为有限 CPU float32/float64，前置 group 维非空。保留 dtype/device、输入值和非连续布局兼容性；最大值估计允许奖励梯度，策略系数必须 detach。覆盖负奖励、并列、`N=1`、`k=N`、`N=256`，避免整数溢出。

例：`rewards=[.2,.9,.4], k=3` 得到 `[0,.5,0]`。这些是策略梯度系数，不是概率，也不是 GRPO 的组内标准化 advantage。

## 论文原生 toy

复现 §5.1 的设置：`X~Normal(theta,std=.1)`，区间 `[0,1]` 内奖励为 `X²`，否则为零。T3 用你的函数计算 `k=1,4,8` 的期望最大值曲线，以及提交 loss 的实际策略梯度。采用固定 CPU 小预算（2048 组、N=16、15 个均值），独立重算论文 toy，不声称复用论文随机样本或复现 LLM 训练。
