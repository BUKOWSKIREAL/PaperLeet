# 37. OPSD 逐词表项散度截断

实现 forward/reverse KL 的逐词表项上限截断。

## 论文与范围

[Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/html/2601.18734v3) · Section 3.2: Per-Token Pointwise Divergence Clipping · [Code](https://github.com/siyan-zhao/OPSD)

实现有符号散度项、逐项截断和回答级平均；阈值校验与实验循环已提供。

## 接口

```python
clipped_distillation_loss(student_logits, teacher_logits, mask, tau, direction='forward')
```

输入两个同形有限 logits `[B,T,V]`、布尔 mask `[B,T]`、有限正标量 tau 以及 direction=`forward`/`reverse`。B,V 为正，T 可以为零；CPU float32/float64。输出标量 loss，保持 dtype/device，只允许学生梯度。每条回答在有效 token 上平均，再对 B 条回答平均；空回答贡献 0。不要修改输入。

## 定义

令 p=softmax(student)，q=stopgrad(softmax(teacher))。forward 项为 `q*(log q-log p)`，reverse 项为 `p*(log p-log q)`。对**每个词表项**取 `min(term,tau)` 后沿词表求和。负项保留；截断后的目标可以为负，不能 clamp_min(0)，也不能先求 KL 再截断。

## 例子

p=[0.03,0.17,0.8]，q=[0.8,0.15,0.05]，tau=0.01 时 forward clipped loss 约为 -0.14741；这个负值是约定允许的。

## 小实验

三分类分布上的 tau 扫描，对比逐项截断与整体截断。属于按公式设计的教学实验，不是论文 Figure 4 的 Qwen 训练复现。

T0 检查接口；T1 使用独立数值例子；T2 检查梯度与不变量；T3 在小实验中实际调用提交实现。
