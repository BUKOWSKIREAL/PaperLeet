# 19. DMD：分数差分布蒸馏

实现分布匹配生成器梯度、配对回归与独立的 fake-score 更新。

## 接口

- `score_difference(real_score: Tensor, fake_score: Tensor, weight: Tensor) -> Tensor`
- `dmd_loss(generated: Tensor, gradient: Tensor, paired_target: Tensor, regression_weight: float = .1) -> Tensor`
- `fake_score_loss(model, generated: Tensor, noise: Tensor, sigma: Tensor) -> Tensor`

采用 VE 加噪 y=x+sigma*noise。输入是 ∇y log q_sigma 与 ∇y log p_sigma，不是 epsilon 预测。正权重由调用者提供。匹配替代损失的梯度为 weight*(s_fake-s_real)/numel，分数目标冻结。DMD 保留配对回归；高斯 toy 用 MSE 替代图像回归度量。训练 fake-score 的 DSM 不能更新生成器。链接 DMD2 仓库仅用于共同梯度机制，不将本题混为 DMD2。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 平移二维高斯生成器和可训练高斯 fake score 展示分数符号、梯度隔离与配对回归。这不是论文 Figure 3 双峰原代码，而是用解析分数隔离相同分布匹配机制。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2311.18828
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

论文定位：arXiv v1 §3 distribution matching gradient and regression regularization；Figure 3 distribution matching illustration。

第二个 T3 与参考预览增加 **Figure 3 启发的双峰改编实验**：解析双高斯学生的两个中心都从右侧附近出发，比较仅真实分数、分数差匹配、匹配加配对 MSE 回归。这里教师和 fake score 都使用解析混合分数以隔离生成器梯度；第一个高斯 T3 单独验证 fake score 训练。CPU 运行600步，噪声 sigma 从1.5降至0.6，目标中心为 (-1.8,0)/(1.8,0)，分量标准差 .25，报告每个模式占比和覆盖数。这些配置与曲线是 PaperLeet 重实现，不是原 Figure 3 数据或源码。
