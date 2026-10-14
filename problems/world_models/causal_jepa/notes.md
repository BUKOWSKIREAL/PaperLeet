# 参考讲解

arXiv 首版标题为 *Object-Level Latent Interventions*，当前版本使用 *Object-Level Latent Masking*。本题依据 [arXiv 2602.11389v2](https://arxiv.org/html/2602.11389v2) 与官方 [输入构造代码](https://github.com/galilai-group/cjepa/blob/412337d0210bf98cee2ca90c3586ab2ea7ca519e/src/cjepa_predictor.py)。`prepare_input` 先构造遮蔽 token、时间与投影锚点之和，再恢复首帧及未遮蔽历史。本接口将投影的共享偏置并入共享遮蔽 token。

官方 [CLEVRER slot 训练损失](https://github.com/galilai-group/cjepa/blob/412337d0210bf98cee2ca90c3586ab2ea7ca519e/src/train/train_causalwm_from_clevrer_slot.py) 将分别平均的选中对象历史和未来 MSE 相加，目标均停止梯度。该循环包含选中对象的时间零；本题则遵循论文公式 (5) 的遮蔽位置指示器，排除可见身份锚点，接口已明确这一差异。若将所有遮蔽 token 统一平均，会按各自数量重新加权；测试故意使两组数量不同。历史选择为空时必须返回零，不能对空张量取平均而得到 NaN。

身份锚点使模型在没有对象序号编码时仍能区分对象。同步置换 slot 与掩码后，构造输入应同样置换。修改选中对象后续历史不得影响输入，修改锚点则可以影响身份查询。逐元素投影真值还会检查权重矩阵的方向。

遮蔽改变预测器能看到什么，不会物理干预系统，也不证明因果可识别性。小实验用已知耦合轨迹显现上下文缺失的影响，不冒充论文视觉或因果推理结果。错误变体分别遮蔽锚点、泄漏隐藏历史、训练冻结目标和监督可见历史。
