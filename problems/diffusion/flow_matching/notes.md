# 通过后的复盘

- 插值、目标符号、积分方向必须一致。`test_point_zero_loss` 检查训练，`test_gaussian_transport` 与 `test_learns_eight_modes` 检查生成，避免只看 loss 下降。
- 单点 oracle 在 t=1 有奇点；Euler 用左端点求值，最大时间是 (n-1)/n。`test_no_singular_endpoint` 直接记录调用时间。
- 条件目标有不可约方差，CFM loss 通常不能降到零。`test_point_zero_loss` 的特殊单点数据才具有零最优误差；真实训练曲线保留非零终值。
- 本题 loss 是所有 batch 和坐标元素取平均；与对坐标求和再对 batch 平均相差维度因子，`test_reference` 明确这一约定。
- 2000 步 TimeMLP 并非完美 oracle：半径 .5 的离群率实测 .123，阈值按规格规则取 ×1.5=.1845，MMD 也按 ×1.5 标定，并单独要求八模式均覆盖、每模式占比距 .125 不超过 .05。具体环境和数值在 problem.yaml；该阈值针对短训练预算，并不等同 D3 的精确 oracle 质量。

时间方向已核对 [Lipman 式 22–23](https://arxiv.org/html/2210.02747v2)、[Rectified Flow](https://arxiv.org/abs/2209.03003)、[SD3 式 13](https://arxiv.org/html/2403.03206v1)，访问日期 2026-10-13。
