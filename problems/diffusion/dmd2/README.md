# 20. DMD2：双时间尺度与对抗更新

在分布匹配基础上实现 critic/生成器更新比例和非饱和 GAN 项。

## 接口

- `generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool`
- `adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]`
- `generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor`

本题 critic/fake diffusion 每更新5次，生成器更新1次；计数表示已完成的 critic 更新，第0次不触发。判别器与生成器在 fake logits 上符号相反。调用方在 critic 训练时 detach 生成数据，在生成器训练时冻结 critic 参数但保留输入梯度。与 DMD 不同，生成器不含配对教师回归。本题聚焦稳定训练和真实数据贡献，不覆盖多步 backward simulation。

## 可运行实验

**来源：LeetPaper teaching adaptation。** 非配对平移高斯 toy 将 fake-score 和判别器训练5次再更新学生1次，展示调度、GAN梯度方向与移除配对回归。用高斯分数参数替代预训练扩散特征。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2405.14867
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

论文定位：arXiv v1 §3 two-time-scale updates and GAN loss；Official EDM compute_generator_clean_cls_loss and compute_guidance_clean_cls_loss。
