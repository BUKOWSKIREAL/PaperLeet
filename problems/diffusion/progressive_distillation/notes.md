# 复盘：Progressive Distillation：教师两步压成学生一步

alpha/sigma 是幅度系数，alpha²+sigma²=1，不是累计方差 alpha_bar。要求 sigma_t>0 且结束时间严格更小，保证分母非零。教师做两次去噪，在中间状态重新预测。解出的学生目标必须冻结。每样本权重由调用者给出，toy 使用1；参数化旋转仍在第7题。

固定仿射去噪教师在余弦调度下走两步 DDIM，小型线性学生学习解出的目标，一次前向匹配终点。演示精确蒸馏代数，不代表图像质量或多阶段反复减半。

错误变体覆盖：omits ratio term, wrong ddim sign, target has gradient, drops loss weight。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。
