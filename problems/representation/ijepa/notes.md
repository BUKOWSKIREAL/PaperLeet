# 参考讲解

目标编码器看到完整图像，因此目标准备从完整 token 特征图开始。上下文编码器的掩码和预测器的位置查询发生在本题之前。给定特征后完成这个模块，不等于完成表示学习预训练。

作者的 `forward_target` 在无梯度环境下运行，沿最后的特征维做归一化，收集目标块，再调用 `repeat_interleave_batch`。该辅助函数的名字容易误导：它对每个目标块重复的是**整个 batch**，不是逐张图重复。目标掩码列表顺序和掩码内部索引顺序都要保留。

官方 `loss_fn` 调用默认 beta 为 1、mean 归约的 `F.smooth_l1_loss`。论文中的 L2 描述与发布代码的数值约定需要区分。每个预测值的导数为 `clamp(prediction-target,-1,1)/numel`。目标准备与损失函数分别执行 stop-gradient，测试可检测误训练外部目标的情况。

目标编码器 EMA 中，旧目标参数的权重是 `m`。online 参数固定时，T 次更新后满足 `target_T=m**T*target_0+(1-m**T)*online`。toy 用独立轨迹检查这个恒等式；真实 I-JEPA 训练中 online 参数会随优化器变化。

来源：[论文](https://arxiv.org/abs/2301.08243)、[src/train.py](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py)、[目标收集](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/utils.py)、[batch 重复](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/tensors.py)。函数式 EMA 输出和确定性掩码输入是本题明确采用的教学改编。
