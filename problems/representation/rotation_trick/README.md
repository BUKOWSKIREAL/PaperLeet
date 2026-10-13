# Rotation Trick 向量量化梯度

[Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/html/2410.06424v1)（ICLR 2025 Oral）改变向量量化的反向传播方式。输入已经给出选中的码本向量 `q`。实现 `rotation_quantize(e, q, eps=1e-8)`，使前向等于 `q`，同时让编码器接收经过旋转和缩放的梯度。

令：

$$u=e/\lVert e\rVert,\quad v=q/\lVert q\rVert,\quad r=(u+v)/\lVert u+v\rVert,\quad s=\lVert q\rVert/\lVert e\rVert.$$

原文 §4.2 给出高效计算：

$$R=I-2rr^T+2vu^T,\qquad \widetilde q=\operatorname{sg}(sR)e.$$

不必构造 `D×D` 矩阵。**全部方向向量和范数比都要冻结**。若上游列梯度为 `g`，编码器梯度应为 `s Rᵀ g`；这条替代梯度路径不能更新 `q`。完整 VQ-VAE 中用于学习码本的独立损失不在本题内。

## 接口约定

两个输入形状相同且非空，为 `[...,D]`、`D≥2`，有限的 CPU float32/float64，dtype/device 一致，可为非连续张量。保留形状、dtype、device 和输入值。脚手架校验有限正数 `eps`、两个范数都大于 `eps`，以及 `||u+v|| > eps`。

零向量与完全反向的向量无法定义这套构造，抛出 `ValueError`，不需另造退化梯度。正向共线有效：`q=3e` 时，前向为 `q`，编码器梯度为 `3g`。

这里验证的是**替代 Jacobian**。对最近码本选择做有限差分是在测另一个函数，因此测试采用独立解析旋转矩阵来验证梯度。

## Toy 实验

**基于原文的改编。** Figure 2–3 使用二次函数 `f(x,y)=x²+y²` 和 16 个码本向量来说明梯度行为。我们独立设置 4×4 码本，在圆周上采样编码器向量，比较你的 autograd 梯度与 STE，展示梯度角度和模长。保留原函数及码本数，网格、采样和折线图为教学改编，不声称复现 VQ-VAE 训练。

预览使用参考实现，T3 调用你的提交，核验前向值、角度保持与范数缩放。[作者代码](https://github.com/cfifty/rotation_trick)。
