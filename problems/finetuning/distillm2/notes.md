# 题解

用 `log_softmax` 与 log-space 混合，构造两种散度前先 detach 教师。SRKL 中学生既出现在概率权重，也出现在混合分布分母，两条梯度都要保留。alpha 为零时恢复正向/反向 KL；alpha 为一时两者均为零。

教师回答用 SKL，学生回答用 SRKL。mask 掉无效 token，分别求和，再加 beta 权重并除以 `2B`。改成 token mean 会改变不同长度回答的权重。测试用独立标量概率与中心差分，未调用参考解。

## 论文与实现边界

固定 [arXiv v2](https://arxiv.org/html/2503.07067v2)，保留 Eq. (2) 的二分之一因子。论文 Eq. (1) 沿实际回答 token 书写；本题明确对每个已给上下文计算完整条件词表散度，不声称任意上下文采样下它等于完整自回归序列 KL。alpha/beta 固定，回答已提供，不包含论文的 curriculum 和数据筛选，也不声称与[作者训练仓库](https://github.com/jongwooko/distillm-2)逐位一致。

T3 的合成分布实际调用两个分支进行梯度更新，不复制 Figure 1，也不验证论文的模型质量提升。四个 mutant 分别覆盖双源同方向散度、SRKL 混合方向错误、遗漏 Eq. (2) 缩放、学生梯度被截断。
