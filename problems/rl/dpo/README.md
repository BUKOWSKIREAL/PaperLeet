# Direct Preference Optimization

## 1. 这篇论文解决什么

DPO 把 KL 正则化奖励最大化改写为偏好对上的分类目标，训练时不需要单独拟合奖励模型。[论文第 4 节](https://arxiv.org/html/2305.18290v3#S4) 从最优策略推导这个等价关系。

## 2. 核心公式与题面

先实现 `sequence_logprob`：输入 logits `[B,T,V]`、token id 和回答 mask `[B,T]`，输出 `[B]`。位置 $t$ 的 logits 预测位置 $t+1$ 的 token；只求回答部分 log-prob 的总和。第一项标签没有前驱预测，最后一项 logits 没有对应标签。

再实现 `dpo_loss`。四个序列 log-prob 都是 `[B]`，返回 loss 标量，以及两个已经 detach 的奖励向量。

$$
h=(\log\pi_c-\log\pi^{ref}_c)-(\log\pi_r-\log\pi^{ref}_r),\qquad
L=-\mathbb E[(1-\varepsilon)\log\sigma(\beta h)+\varepsilon\log\sigma(-\beta h)].
$$

$c,r$ 分别代表 chosen 和 rejected，$\beta$ 默认为 0.1，标签平滑 $\varepsilon$ 默认为零。隐式奖励为 $\beta(\log\pi-\log\pi^{ref})$。这是相对参考策略的奖励，所以不能拿 chosen 的原始 log-prob 当作预测偏好分数。

## 3. 从公式到代码

```text
logits[:, :-1]: [B,T-1,V] → log_softmax
用 input_ids[:, 1:] 取出标签分数: [B,T-1]
乘 loss_mask[:, 1:]，沿时间求和 → [B]
冻结 reference；计算 chosen / rejected reward 和 margin
用稳定的 log-sigmoid 合并正反标签 → 标量 loss
```

整个计算应留在 log 空间。对于 $\beta h=-10000$，先求 sigmoid 会下溢为零，再取 log 会产生无穷；稳定函数直接计算同一个数学目标，不需要人为截断 margin，也不会抹掉困难样本的梯度。

## 4. 本题约定

| 项目 | 本题 | 相关方法或库 |
|---|---|---|
| token 聚合 | 回答 log-prob 总和 | SimPO 使用长度归一化，目标不同 |
| mask | 与右移后的标签对齐 | prompt 与 padding 均排除 |
| label smoothing | 正反 sigmoid loss 的凸组合 | TRL v0.23.1 的保守 DPO 说明 |
| reference 与奖励 | reference 无梯度，返回奖励 detach | 奖励用于诊断而非再次反传 |

[TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) 将这一平滑联系到保守 DPO；它与 `robust` loss 的标签噪声校正不是同一公式。T3 使用六个候选、固定非均匀 reference 和两万对 Bradley–Terry 偏好，比较训练策略与 $\pi^*\propto\pi_{ref}e^{r^*/\beta}$，同时检查隐式奖励排序完全一致。

当参考模型更偏爱某个回答时，策略必须相对参考提高得更多，才能形成正的偏好边际。这个相对量解释了为什么固定参考分布会改变闭式最优策略，也解释了实验中选择非均匀参考分布的原因。

## 5. 交互可视化

[打开标签与 mask 网格](#viz-shift_mask_grid)：检查每个位置预测的下一个 token，查看被 prompt mask 排除的贡献。

[打开 margin 曲线](#viz-loss_vs_margin)：调节 $\beta,\varepsilon$，观察 loss 与梯度权重；训练奖励散点展示隐式奖励逐渐恢复真实奖励的排序。

## 6. 怎么运行

```bash
paperleet init rl/dpo
paperleet test rl/dpo
```

编辑 `workspace/rl/dpo.py`。双循环测试帮助核对移位；极端 margin 测试帮助核对数值稳定性；行为测试在本地训练一个小 categorical 策略。

## 7. 延伸

继续 [GRPO](../grpo/README.md)，比较离线偏好训练与在线组采样。也可以研究长度归一化或带噪偏好改变了哪些最优性假设。
