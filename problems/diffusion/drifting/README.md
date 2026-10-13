# 18. Drifting：成对漂移场与冻结目标

实现距离核漂移场及 stop-gradient 目标，训练一次前向即可采样的生成器。

## 接口

- `drift_field(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`
- `drifting_loss(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`

采用 exp(-欧氏距离/temperature)，屏蔽生成样本自身配对，以 sqrt(行质量×列质量) 归一化，再用正负组的交叉质量加权中心差。这是官方 toy 变体。整个 g+V 目标必须冻结，漏掉 detach 可能保持相同 loss 数值却让梯度归零。一步指生成器推理，不指只训练一步。

## 可运行实验

**来源：Official toy adaptation。** 改编作者 Swiss-roll/checkerboard notebook：从 noise32→3×256、batch2048、2000步缩为 noise8→2×48、生成96/正样本128、260步、temperature=.2，适合CPU判题。不含预训练特征、CFG或图像基准；论文 Figure 3 的双峰实验是另一个实验。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2602.04770
- https://github.com/lambertae/lambertae.github.io/blob/a7fa4baae52abbab2aaeb57136d557caee79ff39/projects/drifting/notebooks/drifting_model_demo.ipynb
- https://github.com/lambertae/drifting/tree/8861066430db11d2fb0858d042c7845b99df49f1

论文定位：arXiv v2 §3 drifting field and fixed-point loss；§5.1 Figure 3 bimodal toy；Official notebook compute_drift and drifting_loss。

数值改编：用 logsumexp 计算双向归一化核，替代 notebook 的直接 exp 与乘积截断。这样在原始指数下溢时仍保持数学归一化权重；有意移除了原代码绝对 1e-12 的乘积下限。空样本集及非正/非有限温度会报错。
