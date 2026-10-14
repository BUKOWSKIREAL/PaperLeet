# LeetPaper 独立盲测记录：D2 / R4

## 时间与方法

- 首个记录时间：2026-10-13 16:49:31 UTC，紧接四个指定文件的一次完整读取之后；读取命令在工具侧耗时不足 1 秒。没有声称计入该时间之前的调度等待。
- 两题首版实现完成：2026-10-13 16:50:19 UTC。
- 两题首次测试结果均已取得：2026-10-13 16:50:36 UTC。
- 记录整理结束时间：2026-10-13 16:50:55 UTC。
- 从首个记录时间至两题测试结果：65 秒；至记录整理结束：84 秒。两題在一次编辑中实现并并行运行测试，没有独立测量每题用时。
- 执行者是一个独立 Codex 子代理。这里是代理盲测，不是人类学习者耗时评估。

## 信息边界

实现前只读取以下仓库文件：

1. `problems/diffusion/param_conversion/README.md`
2. `problems/diffusion/param_conversion/starter.py`
3. `problems/rl/grpo/README.md`
4. `problems/rl/grpo/starter.py`

另使用了 PyTorch 和基础代数的既有知识，以及题目授权的 CLI 输出。未打开题面链接，没有浏览网页、操作 UI、阅读 solution、测试源码、mutants、规格或其他题解，也没有读取生成的 JSON 报告正文。题面写明的测试覆盖信息本身可见。没有修改仓库题包；独立实现和本记录位于 `/tmp/paperleet-blind-trial`。

## 首次结果与最终结果

| 题目 | 首次实现 | 首次 CLI 结果 | 最终结果 | 修复次数 |
| --- | --- | --- | --- | --- |
| D2 `diffusion/param_conversion` | `d2.py` | PASS：T0 1/1、T1 1/1、T2 5/5、T3 无 | 同首次 | 0 |
| R4 `rl/grpo` | `r4.py` | PASS：T0 1/1、T1 1/1、T2 6/6、T3 1/1 | 同首次 | 0 |

没有先提交失败版本，也没有测试失败后修复。两次 CLI 均退出码 0。通过现有测试不等于已穷尽证明正确性；本次没有额外验证混合 dtype、float16 或题面范围之外的输入。

执行命令：

```sh
.venv/bin/paperleet test diffusion/param_conversion --impl /tmp/paperleet-blind-trial/d2.py --report /tmp/paperleet-blind-trial/d2-report.json
.venv/bin/paperleet test rl/grpo --impl /tmp/paperleet-blind-trial/r4.py --report /tmp/paperleet-blind-trial/r4-report.json
```

## 文档卡点与实现依据

- 没有导致暂停、猜测关键公式或需要外部查阅的文档卡点。
- D2 已给出正反旋转、除数 clamp 下限、三种字符串、未知字符串 ValueError、同参数化直接返回以及 v 转换端点稳定性要求。实现由这些公式直接得到；v 到 x0/eps 使用无除法式，同源同目标直接返回 pred。系数转换为所用数据张量的 dtype/device 是实现选择，没有由测试失败反推约定。
- R4 明确给出相邻分组、样本标准差 correction=1、std+eps、按回答平均、空回答贡献零、old/ref 切断梯度、指数之前隔离 padding、三个诊断量的分母和 detach，以及 clipfrac 的双尾定义。这些信息足够一次实现。
- R4 非法 group_size 抛 ValueError 是依据“G >= 2 且整除奖励数”的自然实现选择；题面没有明确指定异常类型，此处没有形成实际阻塞。
- 本次没有从失败报告中获取额外实现信息，因为首次运行均通过。测试源码与参考解保持未读。
