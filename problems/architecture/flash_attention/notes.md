# 参考说明

不变量为 `l = sum(exp(scores_seen-m))` 与 `u = sum(exp(scores_seen-m)*values)`。两者依赖相同的累计最大值。遇到更大的最大值时，要改变其共同尺度，最终比值不受影响。

只使用当前块最大值，在普通输入上可能仍然代数正确，但 score 大幅下降时会溢出。极值测试专门覆盖这种情况。detach 累计 numerator 可能不改变前向结果，却破坏早期 value 梯度，独立梯度测试能检测它。

提供的循环是便于阅读的 CPU 递推，不是融合 FlashAttention kernel，普通自动求导也不是论文的重计算反向算法。
