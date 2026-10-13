# 复盘：DMD2：双时间尺度与对抗更新

本题 critic/fake diffusion 每更新5次，生成器更新1次；计数表示已完成的 critic 更新，第0次不触发。判别器与生成器在 fake logits 上符号相反。调用方在 critic 训练时 detach 生成数据，在生成器训练时冻结 critic 参数但保留输入梯度。与 DMD 不同，生成器不含配对教师回归。本题聚焦稳定训练和真实数据贡献，不覆盖多步 backward simulation。

非配对平移高斯 toy 将 fake-score 和判别器训练5次再更新学生1次，展示调度、GAN梯度方向与移除配对回归。用高斯分数参数替代预训练扩散特征。

错误变体覆盖：equal update frequency, gan wrong sign, no gan generator, target not frozen。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。
