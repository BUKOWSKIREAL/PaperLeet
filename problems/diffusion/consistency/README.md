# 21. Consistency Models：边界与教师目标

实现保持边界的 consistency 函数、冻结相邻时间目标的损失与 EMA 更新。

## 接口

- `consistency_function(model, noisy: Tensor, sigma: Tensor, sigma_min: float = .002, sigma_data: float = .5) -> Tensor`
- `consistency_loss(prediction: Tensor, target_prediction: Tensor) -> Tensor`
- `update_ema(online, target, decay: float = .95) -> None`
- `teacher_euler(noisy: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor` (provided)

在 sigma_min 处函数必须严格等于输入，与网络输出无关。模型接收 c_in*x 与原始 sigma；本题提供原始 sigma 条件而非官方嵌入。skip/output 系数遵循官方边界参数化。较低噪声的 teacher/EMA 分支冻结。已提供 Euler helper 使用递减正噪声尺度。本题是带教师 consistency distillation，不是另外的无教师 consistency training。

## 可运行实验

**来源：PaperLeet teaching adaptation。** 单点数据教师使概率流 Euler 步精确。小型 MLP 蒸馏验证边界和目标梯度路径，无需预训练模型。一半训练对使用正的相邻低噪声 sigma/2，EMA 网络真实参与目标；另一半连接严格 sigma_min 边界作为锚点。

T3 会真实执行你提交的实现并记录点云、曲线、指标；随机种子为 42。`toy.py` 提供实验脚手架，无需下载模型和数据。本练习只覆盖选定机制，不代表完整图像基准复现。

## 来源

- https://arxiv.org/abs/2303.01469
- https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py

论文定位：arXiv v1 §3 consistency function boundary parameterization；§4 consistency distillation and EMA target；Official get_scalings_for_boundary_condition。
