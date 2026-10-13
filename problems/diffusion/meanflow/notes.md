# 复盘：MeanFlow：平均速度与 JVP 目标

接口使用论文顺序 `(z,r,t)`；JVP 方向 `(v,0,1)` 固定 r。t 增大表示噪声增多。目标为 `stopgrad(v-(t-r)du/dt)`。本题用逐元素平均 MSE，不包含官方图像实现的自适应权重。r=t 时目标退化为条件 Flow Matching，并不要求未训练网络立即等于 v。

使用解析单点数据族与学员 JVP 目标学习中心；这是机制演示，不声称是原论文低维实验。

错误变体覆盖：wrong jvp time, target has gradient, reverse sampling。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。
