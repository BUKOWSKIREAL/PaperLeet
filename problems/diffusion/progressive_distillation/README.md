# 26. Progressive Distillation：教师两步压成学生一步

求出使学生一步 DDIM 精确匹配教师两步的干净数据目标，并训练学生。

## 接口

- `ddim_step(noisy: Tensor, clean_prediction: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distilled_target(noisy: Tensor, teacher_endpoint: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distillation_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor`
- `teacher_pair(teacher, noisy: Tensor, t: Tensor, midpoint: Tensor, end: Tensor) -> Tensor` (provided)

alpha/sigma 是幅度系数，alpha²+sigma²=1，不是累计方差 alpha_bar。要求 sigma_t>0 且结束时间严格更小，保证分母非零。教师做两次去噪，在中间状态重新预测。解出的学生目标必须冻结。每样本权重由调用者给出，toy 使用1；参数化旋转仍在第7题。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 固定仿射去噪教师在余弦调度下走两步 DDIM，小型线性学生学习解出的目标，一次前向匹配终点。演示精确蒸馏代数，不代表图像质量或多阶段反复减半。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2202.00512
- https://github.com/google-research/google-research/tree/0a1b0fe7298a58bca6dcb47112379d829bc0c0a5/diffusion_distillation

论文定位：arXiv v1 §3 Progressive distillation；Algorithm 2 and Appendix B distillation target。
