# 通过后的复盘

- v 符号反转时，shape 完全正常，但运动方向相反。`test_velocity_tangent` 用有限差分确定正确符号。
- alpha 是累计信号的平方根；多平方一次会在高噪声时放大误差。`test_six_roundtrips` 和 `test_reference` 定位这类错误。
- alpha=0 时噪声无法唯一决定 x0；clamp 只保证有限值，不创造缺失的信息。v 的逆旋转仍精确，见 `test_endpoint_stability`。
- 不同参数化损失隐含不同时间权重。`test_error_weight_identity` 将同一错误预测统一到三种表示，避免比较无关的预测误差。

已核对 [Salimans & Ho 附录 D，式 26、30](https://arxiv.org/html/2202.00512v2#A4)，访问日期 2026-10-13。
