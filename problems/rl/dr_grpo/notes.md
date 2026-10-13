# Dr. GRPO：通过后复盘

- **依旧除以回答长度**：短回答中每个 token 仍获得更大的梯度。`test_length_bias_removed` 精确比较长度五与二十的负 advantage 回答，并报告 R4 的梯度比 5/20 作为对照。
- **把固定上限换成 batch 最长回答**：目标回答未变，仅加入或延长别的回答就改变梯度。`test_fixed_generation_limit` 用零 advantage、零 KL 的无关回答排除这种隐式依赖。
- **中心化后又除 std**：奖励翻倍而梯度权重不变，说明保留了 R4 的难度缩放。`test_advantage_scales_with_reward` 检查本题要求的线性尺度关系。
- **复制题目默认 KL 到原文描述**：[原文第 2 节与附录 G](https://arxiv.org/html/2503.20783v2) 明确设 beta=0；第 3.2 节与代码示例使用固定生成预算归一。本题故意保留 R4 的 beta=0.04，只隔离两处归一化改动。`test_reference` 检查本题数值约定。
- **把单种子长度曲线升级为定律**：`test_length_effect_display_only` 只产出真实训练曲线，不根据方向判分。EOS 采样、奖励与学习率会影响结果；若要新增行为门槛，应先做多种子实验。本题目前只把可精确推导的梯度性质用于质量门。
