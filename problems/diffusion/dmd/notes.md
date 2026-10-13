# 复盘：DMD：分数差分布蒸馏

采用 VE 加噪 y=x+sigma*noise。输入是 ∇y log q_sigma 与 ∇y log p_sigma，不是 epsilon 预测。正权重由调用者提供。匹配替代损失的梯度为 weight*(s_fake-s_real)/numel，分数目标冻结。DMD 保留配对回归；高斯 toy 用 MSE 替代图像回归度量。训练 fake-score 的 DSM 不能更新生成器。链接 DMD2 仓库仅用于共同梯度机制，不将本题混为 DMD2。

平移二维高斯生成器和可训练高斯 fake score 展示分数符号、梯度隔离与配对回归。这不是论文 Figure 3 双峰原代码，而是用解析分数隔离相同分布匹配机制。

错误变体覆盖：score sign, matching target not frozen, fake updates generator, drops regression。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。

第二个 T3 与参考预览增加 **Figure 3 启发的双峰改编实验**：解析双高斯学生的两个中心都从右侧附近出发，比较仅真实分数、分数差匹配、匹配加配对 MSE 回归。这里教师和 fake score 都使用解析混合分数以隔离生成器梯度；第一个高斯 T3 单独验证 fake score 训练。CPU 运行600步，噪声 sigma 从1.5降至0.6，目标中心为 (-1.8,0)/(1.8,0)，分量标准差 .25，报告每个模式占比和覆盖数。这些配置与曲线是 PaperLeet 重实现，不是原 Figure 3 数据或源码。
