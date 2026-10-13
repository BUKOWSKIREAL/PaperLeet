# 复盘：JiT：高维空间中的干净数据预测

远离截断端点时 v=x-epsilon；x 预测通过 (x_hat-z)/(1-t) 转换，epsilon 预测通过 (z-epsilon_hat)/t 转换。所有网络都使用速度损失。分母截断遵循官方 x 预测代码，参数化的精确等价性只在截断区域之外成立。t 为 [B]，应广播到其余维度。

根据 Figure 2 固定正交 2D→D 投影思想重实现，数据选用合成螺旋。默认 D=16、宽32、深3、160步；原文维度为2/8/16/512、宽256、深5。使用 toy.py 的 --impl 与 --full --steps 5000 可运行原维度/网络扫描；步数和螺旋数据为教学选择。短程结果不保证原论文排序，未找到原 toy 源码。

错误变体覆盖：wrong time direction, wrong x denominator, clean space loss。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。

实验 depth 计入输出层的全部 Linear 层，因此 --full 总计五个 Linear 层，之间使用 ReLU。
