# 17. JiT：高维空间中的干净数据预测

实现干净数据预测与速度空间监督，在高维嵌入数据上比较 x/epsilon/v 输出。

## 接口

- `make_noisy(clean: Tensor, noise: Tensor, t: Tensor) -> Tensor`
- `prediction_to_velocity(prediction: Tensor, z: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`
- `velocity_loss(prediction: Tensor, clean: Tensor, noise: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`

远离截断端点时 v=x-epsilon；x 预测通过 (x_hat-z)/(1-t) 转换，epsilon 预测通过 (z-epsilon_hat)/t 转换。所有网络都使用速度损失。分母截断遵循官方 x 预测代码，参数化的精确等价性只在截断区域之外成立。t 为 [B]，应广播到其余维度。

## 可运行实验

**来源：Paper experiment reimplementation, reduced。** 根据 Figure 2 固定正交 2D→D 投影思想重实现，数据选用合成螺旋。默认 D=16、宽32、深3、160步；原文维度为2/8/16/512、宽256、深5。使用 toy.py 的 --impl 与 --full --steps 5000 可运行原维度/网络扫描；步数和螺旋数据为教学选择。短程结果不保证原论文排序，未找到原 toy 源码。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2511.13720
- https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py

论文定位：arXiv v1 §3.2 prediction and loss spaces；§3.3 Figure 2 high-dimensional manifold toy；Official Denoiser.forward。

实验 depth 计入输出层的全部 Linear 层，因此 --full 总计五个 Linear 层，之间使用 ReLU。
