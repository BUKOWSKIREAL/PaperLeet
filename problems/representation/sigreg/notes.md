# 参考讲解

经验特征函数的实部和虚部分别为 cos 与 sin 在 batch 上的均值。应先取均值再平方，比较的是分布；若先计算逐样本平方再平均，就变成另一个目标。

对实数样本，负频率特征函数是正频率特征函数的共轭，因此它与实数高斯目标的距离平方是偶函数。可对 `[0,t_max]` 用梯形积分再乘二；均匀网格的端点权重仍为内部节点的一半，非均匀网格则使用相邻节点间隔。零频率处，两个特征函数都等于一。

来源为 [LeJEPA v1](https://arxiv.org/html/2511.08544v1) 第 4.2.3、5.1 节，以及固定版本 c293d291 的官方 [EppsPulley](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py)、[slicing.py](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py) 和 [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)。论文示例用完整网格，优化代码用正半轴网格和双倍积分权重。本题保持 float32/float64，直接接收单位方向与网格，不包含分布式归约和每步重新抽样方向。

SIGReg 的目标是固定的标准高斯，不能先估计均值和尺度再标准化。均衡二点分布已匹配高斯的一阶、二阶矩，但特征函数仍不同。反过来，有限的方向和频率也可能遗漏差异，因此 toy 展示的是一个具体区别，不是有限检验对所有分布的识别保证。

乘以 `N` 是统计量定义的一部分。复制每个样本不会改变经验特征函数，却会令统计量翻倍；改变 batch 大小不是这里的不变性。四个错误版本分别移除虚部、遗漏 `N`、遗漏正半轴倍增，以及错误地先标准化投影。

在精确零塌缩处，sin 为零而 cos 的导数为零，所以损失虽正，所有特征梯度仍为零。T2 验证这一事实及其附近不恒定扰动的下降方向；toy 则从非恒定的变形特征开始。独立 oracle 使用完整对称网格与复数运算，不依赖用户实现。
