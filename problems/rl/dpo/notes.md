# DPO：通过后复盘

- **标签或 mask 少移一格**：位置 t 的 logits 预测 t+1 的标签；prompt 边界必须用标签侧 mask。`test_sequence_double_loop` 与 `test_shift_and_mask_isolation` 分别检查定义和无关位置隔离。
- **求平均替代求和**：会把序列概率变成长短回答不同的目标；本题实现标准 DPO 的序列 log-prob。`test_sequence_double_loop` 用不等有效长度抓住这一变化。其他方法采用长度归一化不意味着它们在自身约定中有错。
- **log(sigmoid) 下溢**：负的极端 margin 先 sigmoid 为零，导致无穷 loss；`test_extreme_margin_stability` 检查有限 loss 和精确梯度。
- **诊断奖励残留计算图**：返回奖励会被日志保存；本题要求 detach，reference 同样固定。`test_gradient_direction_and_rewards` 检查方向、值与计算图。
- **保守 DPO 与 robust DPO 混淆**：本题使用正反标签损失的凸组合。[TRL v0.23.1 文档](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) 对 cDPO 作了说明；robust loss 另有除以标签噪声校正系数的形式，不能只凭参数名互换。版本核验于 2026-10-13，平滑语义由 `test_reference` 检查。
