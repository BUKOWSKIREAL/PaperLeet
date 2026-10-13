# 复盘：Consistency Models：边界与教师目标

在 sigma_min 处函数必须严格等于输入，与网络输出无关。模型接收 c_in*x 与原始 sigma；本题提供原始 sigma 条件而非官方嵌入。skip/output 系数遵循官方边界参数化。较低噪声的 teacher/EMA 分支冻结。已提供 Euler helper 使用递减正噪声尺度。本题是带教师 consistency distillation，不是另外的无教师 consistency training。

单点数据教师使概率流 Euler 步精确。小型 MLP 蒸馏验证边界和目标梯度路径，无需预训练模型。一半训练对使用正的相邻低噪声 sigma/2，EMA 网络真实参与目标；另一半连接严格 sigma_min 边界作为锚点。

错误变体覆盖：no boundary shift, unfrozen target, ema reversed, skip input scaling。T1 检查公式值，T2 检查不变量与梯度路径，T3 在 toy 中调用学员实现。
