# 复盘：Drifting：成对漂移场与冻结目标

采用 exp(-欧氏距离/temperature)，屏蔽生成样本自身配对，以 sqrt(行质量×列质量) 归一化，再用正负组的交叉质量加权中心差。这是官方 toy 变体。整个 g+V 目标必须冻结，漏掉 detach 可能保持相同 loss 数值却让梯度归零。一步指生成器推理，不指只训练一步。

改编作者 Swiss-roll/checkerboard notebook：从 noise32→3×256、batch2048、2000步缩为 noise8→2×48、生成96/正样本128、260步、temperature=.2，适合CPU判题。不含预训练特征、CFG或图像基准；论文 Figure 3 的双峰实验是另一个实验。

错误变体覆盖：row only kernel, attraction only, unfrozen target, self pairs unmasked。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。

数值改编：用 logsumexp 计算双向归一化核，替代 notebook 的直接 exp 与乘积截断。这样在原始指数下溢时仍保持数学归一化权重；有意移除了原代码绝对 1e-12 的乘积下限。空样本集及非正/非有限温度会报错。
