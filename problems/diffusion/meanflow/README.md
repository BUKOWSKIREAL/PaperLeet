# 16. MeanFlow：平均速度与 JVP 目标

实现平均速度的全导数目标、冻结目标损失，以及一次前向采样。

## 接口

- `meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]`
- `meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor`
- `one_step(model, noise: Tensor) -> Tensor`

接口使用论文顺序 `(z,r,t)`；JVP 方向 `(v,0,1)` 固定 r。t 增大表示噪声增多。目标为 `stopgrad(v-(t-r)du/dt)`。本题用逐元素平均 MSE，不包含官方图像实现的自适应权重。r=t 时目标退化为条件 Flow Matching，并不要求未训练网络立即等于 v。

## 可运行实验

**来源：LeetPaper teaching adaptation。** 使用解析单点数据族与学员 JVP 目标学习中心；这是机制演示，不声称是原论文低维实验。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2505.13447
- https://github.com/Gsunshine/py-meanflow/blob/1f6d72d94247c8fdeb89489acca1a8007a6baf6c/meanflow/models/meanflow.py

论文定位：arXiv v1 §3 MeanFlow identity and stop-gradient training objective；Official forward_with_loss and sample。
