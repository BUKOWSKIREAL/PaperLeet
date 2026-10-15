# PaperLeet 离线版

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="320">
</picture>

[English](README.md) | **简体中文**

**把研究论文写成能运行的代码，在自己的 CPU 上练习。**

PaperLeet 是一个从论文到代码的练习题库：阅读方法，实现核心函数，再用测试检验理解。这个开源离线版把论文题库、Python 编辑器、数学性质测试和小型教学实验放到你的电脑上。

**49 篇论文 · 63 道练习 · 491 项分层检查 · 257 个刻意设计的错误实现 · MIT**

内容涵盖注意力与现代 LLM 基础模块、强化学习、扩散与 Flow、后训练、表征学习和世界模型。每道题都包含中英文讲解、起始代码、参考解、测试及常见坑点。Python/PyTorch 在本机执行，无需 GPU、账号或云端 API。

[本地启动](#本地启动) · [第一道练习](#完成第一道练习) · [学习路径](#选择学习路径) · [完整题库](#完整论文与练习目录) · [测试如何帮助理解](#通过四层测试理解方法) · [参与贡献](#参与贡献)

![本地论文题库，包含精选合集与论文搜索](assets/readme/library.png)

## 可以在这里学到什么

| 练习中的内容 | 帮助你理解什么 |
| --- | --- |
| 实现论文精选的核心函数 | 把公式对应到 tensor shape、mask、归一化与梯度流；辅助代码已经提供。 |
| 检查独立数学性质并对照参考解 | 找出那些输出看似合理、却违背方法数学约定的实现。 |
| 分析刻意设计的错误实现 | 看清哪个测试会发现符号、梯度截断、维度或边界处理上的错误。 |
| 使用交互实验台与小型 CPU 实验 | 把公式与可观察的行为联系起来，再比较参考结果和自己运行产生的结果。 |
| 按论文完成必做模块 | 一次掌握一个机制，并在本地界面查看学习进度。 |

练习适合已经接触 Python 和张量运算的学习者。了解基本的 PyTorch 广播与自动微分会更容易上手。每题聚焦论文选定的算法，提供辅助设施，并用适合本机运行的小规模实验验证行为。

## 本地启动

需要 **Python 3.10+**、**Node.js 20.19+**、Git 和 `make`。以下命令适用于 macOS/Linux 终端；Windows 用户可在 WSL 中使用这一启动方式。首次安装依赖需要联网；安装完成后，正常做题所用的编辑器资源与 Python 测试环境都在本机。

```bash
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
python3 -m venv .venv
source .venv/bin/activate

# Linux 可选：在安装项目依赖前，先安装仅 CPU 的 PyTorch。
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e .
npm --prefix web ci
make dev
```

打开 **http://127.0.0.1:5173**。`make dev` 会导出当前题库，并同时启动网页与本机判题服务。连接成功后，做题编辑器底部、**运行全部测试**按钮旁会显示 **本机 CPU 已连接**。

下次使用时，在仓库根目录再次运行 `make dev`。**Ctrl-C 会同时停止两个服务。** 网页监听 `127.0.0.1:5173`，判题服务监听 `127.0.0.1:8765`。判题队列一次执行一个任务，测试 fixture 将 PyTorch 限制为两条 CPU 线程。

只使用终端做题时，安装 Python 部分即可，跳过 npm 和 `make dev`，使用下方[命令行示例](#用自己的编辑器或终端做题)。

## 完成第一道练习

可以从 [**61 · RMSNorm**](problems/architecture/rms_norm/README.md)、[**22 · 缩放点积注意力**](problems/architecture/attention/README.md)或 [**6 · 前向加噪**](problems/diffusion/schedules/README.md)开始。

1. 从论文题库或学习合集进入一篇论文，选择其中一个模块。
2. 阅读目标、张量约定和**必做函数**。起始代码已实现辅助函数。
3. 补全实现，点击**运行全部测试**，使用本机 CPU 执行原生 Python/PyTorch。
4. 查看失败测试的依据、期望值与实际值，进入**实验**观察对应行为，修改后再次运行。
5. 通过全部必需检查，模块会标记为完成并解锁**坑点**。需要对照其他实现时，可以展开**参考解**。

![本地练习工作区，包含题面、Python 编辑器与测试反馈](assets/readme/workspace.png)

截图使用英文界面。首次访问时，应用跟随浏览器／系统的首选语言：中文语言环境使用中文，其余使用英文。语言按钮可以切换界面、题面和讲解的中英文，手动选择会保留。也可以切换浅色／深色主题、调整代码字号，拖动分隔线改变题面、编辑器和结果区域的大小。

完成一篇论文，意味着通过它选定的全部必做模块。论文编号与练习编号相互独立：当前包含论文 **1–49** 和练习 **1–63**。只通过部分测试不会标记整题完成；后续运行失败，也不会抹去当前浏览器保存的已通过记录。

## 选择学习路径

题库提供六个精选合集。下列顺序是建议的起点；遇到不熟悉的概念时，可以先学习题面标注的前置内容。合集允许重叠，同一篇论文可以出现在多条路径中。

| 合集 | 建议顺序 | 起点 |
| --- | --- | --- |
| **LLM 基础** | Attention → Transformer 组件 → RMSNorm → RoPE → GQA → FlashAttention → LoRA | [缩放点积注意力](problems/architecture/attention/README.md) |
| **强化学习** | GAE → PPO → GRPO → Dr. GRPO → GSPO / DAPO → 基于反馈的目标 | [GAE 递推](problems/rl/gae/README.md) |
| **扩散与 Flow** | 前向加噪 → 参数化转换 → DDPM / DDIM 采样 → Flow Matching → 蒸馏 | [前向加噪](problems/diffusion/schedules/README.md) |
| **后训练** | LoRA → QLoRA → DPO → On-policy Distillation → OPSD / SDPO / DistiLLM-2 | [低秩适配](problems/finetuning/lora/README.md) |
| **JEPA / 世界模型** | VICReg → I-JEPA → SIGReg / VISReg → 潜空间预测 → 规划与 Causal-JEPA | [VICReg 损失](problems/representation/vicreg/README.md) |
| **视觉生成** | CLIP → Rotation Trick → VAR → REPA → Drifting / FD-Loss | [CLIP 对比损失](problems/representation/clip/README.md) |

使用合集页、搜索、领域与难度筛选来缩小范围。**随机练习**会在当前合集和筛选条件内选择尚未尝试的练习；**继续练习**帮助你返回未完成的内容。

## 覆盖领域

下表按题库的主要领域统计；上方的学习合集则允许交叉收录。

| 领域 | 论文 | 练习 | 代表方法 |
| --- | ---: | ---: | --- |
| 强化学习 | 8 | 9 | GAE、PPO、GRPO、DAPO、GSPO、PKPO |
| 后训练与微调 | 7 | 8 | LoRA、QLoRA、DPO、OPSD、SDPO、DistiLLM-2 |
| 生成模型 | 15 | 19 | DDPM、DDIM、Flow Matching、Consistency、DMD、MeanFlow、FD-Loss |
| 架构与推理 | 8 | 15 | Transformer、FlashAttention、Kimi K3 模块、RMSNorm、GQA、RoPE |
| 优化算法 | 1 | 1 | AdamW |
| 表征学习 | 8 | 8 | CLIP、REPA、VICReg、SIGReg、VISReg、I-JEPA |
| 世界模型 | 2 | 3 | LeWorldModel、Causal-JEPA |
| **合计** | **49** | **63** | |

## 完整论文与练习目录

点击方法名称可打开该论文的第一道练习；本地论文页会展示全部模块及精选范围。[papers.yaml](papers.yaml) 是课程编排的事实来源。

<details>
<summary><strong>展开全部 49 篇论文及其实现范围</strong></summary>

| 论文编号 | 论文 / 方法 | 实现范围 | 练习编号 |
| --- | --- | --- | --- |
| 1 | [GAE](problems/rl/gae/README.md) | 实现 GAE 的反向递推和 episode 边界处理。 | 1 |
| 2 | [PPO](problems/rl/ppo_clip/README.md) | 实现 clipped surrogate 及其梯度行为；价值损失、归一化和统计代码已提供。 | 2 |
| 3 | [DPO](problems/rl/dpo/README.md) | 实现偏好损失与隐式奖励；token 对齐和序列 log-prob 计算已提供。 | 3 |
| 4 | [DeepSeekMath · GRPO](problems/rl/grpo/README.md) | 选取论文的 GRPO 模块：组内相对优势、截断目标和回答级聚合。 | 4 |
| 5 | [Dr. GRPO](problems/rl/dr_grpo/README.md) | 实现去除组内标准差缩放与固定生成长度归一化这两处修正。 | 5 |
| 6 | [DDPM](problems/diffusion/schedules/README.md) | 实现闭式前向加噪，以及高斯后验和 epsilon 参数化的反向一步；调度器与采样循环已提供。 | 6, 8 |
| 7 | [Progressive Distillation](problems/diffusion/param_conversion/README.md) | 先实现 v 参数化转换，再推导两步教师到一步学生的蒸馏目标并验证更新。 | 7, 26 |
| 8 | [DDIM](problems/diffusion/ddim/README.md) | 实现 DDIM 单步更新和干净端点；时间步选择、循环与反演探索代码已提供。 | 9 |
| 9 | [Flow Matching](problems/diffusion/flow_matching/README.md) | 实现条件路径、目标速度和 CFM 回归损失；时间采样、Euler 积分和训练辅助代码已提供。 | 10 |
| 10 | [GSPO](problems/rl/gspo/README.md) | 实现序列概率比与 GSPO 损失，验证掩码、截断区域和梯度。 | 11 |
| 11 | [GKD · On-policy Distillation](problems/rl/opd/README.md) | 实现学生 rollout 和全词表散度；在小型自回归模型上验证。OPD 在此指 on-policy distillation。 | 12 |
| 12 | [DAPO](problems/rl/dapo/README.md) | 实现组筛选与 token 级目标；采样系统和分布式训练不在本模块范围。 | 13 |
| 13 | [LoRA](problems/finetuning/lora/README.md) | 实现低秩增量和线性前向，验证冻结边界及小型回归学习。 | 14 |
| 14 | [QLoRA](problems/finetuning/qlora/README.md) | 实现分块 NF4 量化、反量化与适配器前向；不模拟双重量化或分页优化器。 | 15 |
| 15 | [MeanFlow](problems/diffusion/meanflow/README.md) | 实现平均流的 JVP 目标和梯度边界，并运行小型生成实验。 | 16 |
| 16 | [JiT](problems/diffusion/jit/README.md) | 实现 x-prediction 到速度的转换与去噪目标，使用论文启发的低维流形嵌入实验。 | 17 |
| 17 | [Drifting Models](problems/diffusion/drifting/README.md) | 实现漂移场与停止梯度损失，在小型二维分布上验证生成器行为。 | 18 |
| 18 | [DMD](problems/diffusion/dmd/README.md) | 实现 score 差目标与配对回归项，用解析双峰分布检查模式覆盖。 | 19 |
| 19 | [DMD2](problems/diffusion/dmd2/README.md) | 实现双时间尺度更新与辅助目标，在小型实验中验证参数更新边界。 | 20 |
| 20 | [Consistency Models](problems/diffusion/consistency/README.md) | 实现边界参数化和相邻时刻的一致性蒸馏损失。 | 21 |
| 21 | [Transformer](problems/architecture/attention/README.md) | 分五题实现缩放点积注意力、多头注意力、LayerNorm 与 Add & Norm、正弦位置编码、逐位置 FFN。 | 22, 27–30 |
| 22 | [AdamW](problems/optimization/adamw/README.md) | 实现矩估计、偏差修正与解耦权重衰减的一步更新。 | 23 |
| 23 | [FlashAttention](problems/architecture/flash_attention/README.md) | 实现分块递推与重缩放。这是 CPU 算法练习，不评价 GPU kernel 性能。 | 24 |
| 24 | [CLIP](problems/representation/clip/README.md) | 实现特征归一化、温度缩放和双向对比损失，使用小型配对检索实验。 | 25 |
| 25 | [GLM-5](problems/optimization/muon_split/README.md) | 分别实现 Muon Split 逐头正交化、异步 RL 的双边重要性采样修正。 | 31–32 |
| 26 | [Kimi K3](problems/architecture/kimi_kda/README.md) | 分别实现有界 KDA 递推、块级注意力残差、SiTU-GLU，并重绘论文的门函数和激活曲线。 | 33–35 |
| 27 | [OPSD](problems/finetuning/opsd_alignment/README.md) | 双上下文因果对齐与逐词表项散度截断。 | 36, 37 |
| 28 | [SDPO](problems/rl/sdpo_feedback/README.md) | 同题反馈路由、reverse KL 与 top-k 尾部概率聚合。 | 38, 39 |
| 29 | [REPA](problems/representation/repa/README.md) | 逐 patch 表征对齐损失与梯度边界。 | 40 |
| 30 | [Rotation Trick](problems/representation/rotation_trick/README.md) | 保持量化前向值的旋转梯度变换。 | 41 |
| 31 | [Differential Transformer](problems/architecture/differential_attention/README.md) | 已投影张量上的差分注意力与 lambda 参数化。 | 42 |
| 32 | [Block Diffusion](problems/diffusion/block_diffusion/README.md) | 拼接干净和含噪 token 的单次前向注意力掩码。 | 43 |
| 33 | [DistiLLM-2](problems/finetuning/distillm2/README.md) | 教师轨迹的 skew KL 与学生轨迹的 skew reverse KL。 | 44 |
| 34 | [PKPO](problems/rl/pkpo/README.md) | 连续奖励变换、policy loss 与论文的一维 Gaussian toy。 | 45 |
| 35 | [SVDQuant](problems/finetuning/svdquant/README.md) | 平滑、SVD 低秩分解与 4-bit 残差前向。 | 46 |
| 36 | [MoGe](problems/representation/moge_alignment/README.md) | 截断加权 L1 的鲁棒最优估计。 | 47 |
| 37 | [DAPS](problems/diffusion/daps/README.md) | 线性 Gaussian 后验 score、Langevin 与独立重新加噪。 | 48 |
| 38 | [DiffusionNFT](problems/diffusion/diffusion_nft/README.md) | 正负 beta 隐式速度、双分支目标与微型训练。 | 49 |
| 39 | [VAR](problems/architecture/var_next_scale/README.md) | 下一尺度条件输入与注意力掩码；多尺度残差量化与重建。 | 50–51 |
| 40 | [FD-Loss](problems/diffusion/fd_population/README.md) | 总体矩的 EMA、可微 Fréchet 距离和停止梯度的损失归一化。 | 52–53 |
| 41 | [VICReg](problems/representation/vicreg/README.md) | 方差、视图一致性与非对角协方差损失。 | 54 |
| 42 | [LeJEPA · SIGReg](problems/representation/sigreg/README.md) | 特征函数高斯性统计与随机切片聚合。 | 55 |
| 43 | [VISReg](problems/representation/visreg/README.md) | 尺度、切片形状、中心与多视图一致性。 | 56 |
| 44 | [I-JEPA](problems/representation/ijepa/README.md) | 教师目标块、预测损失与 EMA 更新。 | 57 |
| 45 | [LeWorldModel](problems/world_models/lewm_prediction/README.md) | 联合预测目标与潜空间 rollout；终点代价与 CEM 精英重拟合。 | 58–59 |
| 46 | [Causal-JEPA](problems/world_models/causal_jepa/README.md) | 保留身份锚点的对象遮挡及历史、未来预测。 | 60 |
| 47 | [RMSNorm](problems/architecture/rms_norm/README.md) | RMSNorm、前 k 特征统计、尺度关系与跨特征梯度。 | 61 |
| 48 | [GQA](problems/architecture/gqa/README.md) | KV 投影组内均值转换、连续头共享、共享梯度与缓存容量。 | 62 |
| 49 | [RoFormer / RoPE](problems/architecture/rope/README.md) | 相邻特征对旋转、相对位置内积、解析梯度与缓存位置偏移。 | 63 |

</details>

当前题库围绕精选机制与教学规模实验展开。例如，FlashAttention 练习分块累加与重缩放，不测量 GPU kernel 的速度；QLoRA 练习分块 NF4 量化与适配器前向。每道题都会说明自己的范围、假设和论文出处。

## 通过四层测试理解方法

输出与参考值一致，只是实现正确的一部分。测试还会检查接口约定、数学性质、梯度，以及小型实验中的真实行为。

| 层级 | 要回答的问题 | 常见检查 |
| --- | --- | --- |
| **T0 · 接口契约** | 函数是否遵守输入输出约定？ | 形状、数据类型、必需输出、mask 与边界输入。 |
| **T1 · 参考对齐** | 是否符合题目声明的实现约定？ | 在明确的约定与容差下，对照参考实现的数值结果。 |
| **T2 · 数学性质** | 方法背后的数学关系是否成立？ | 不变性、恒等式、归一化、极限情形与梯度流。 |
| **T3 · 真实行为** | 方法在实验中是否表现符合预期？ | 带实际测量结果的小型训练、采样、检索或规划任务。 |

63 道练习共声明 **491 项检查**。各题按需要设置测试层：例如前向加噪和参数化转换两题没有 T3 要求。通过全部必需检查才算完成；只运行部分测试适合调试，不能代替整题验收。

公共数值断言可以显示不匹配坐标、张量形状、期望与实际值、误差及容差。每个测试的描述会说明检查依据。题包 `mutants/` 下的 **257 个刻意设计的错误实现**，则把测试希望发现的错误具体呈现出来。

例如，[RMSNorm 的错误变体](problems/architecture/rms_norm/mutants)包括错误地对输入中心化、将 epsilon 放在平方根外；[RoPE 的错误变体](problems/architecture/rope/mutants)包括反向旋转和忽略位置。做完题后可以阅读这些变体，也可以自己运行变异测试：

```bash
paperleet mutate architecture/rms_norm
```

质量检查要求参考解通过、错误实现触发真实断言失败。导入失败或语法错误不会算作发现数学实现问题。测试与参考实现全部在仓库中，可以直接阅读、复核和补充。

## 使用交互实验

**实验**页结合交互公式图、随题提供的参考数据，以及自己运行测试产生的结果。可以观察实现变化如何影响曲线、分布、指标或梯度。

仓库已附带参考可视化数据，开始学习前无需重跑全部实验。自己的 T3 结果来自实际运行的实现，界面会将其与参考示意区分开。小型实验的教学改编与来源写在各题包中。

修改实验后，可以重新生成这一题的数据并导出题库：

```bash
paperleet vizdata architecture/rms_norm
paperleet export
```

`make data` 会重新生成全部参考可视化，耗时更长。生成使用固定种子和两条 PyTorch CPU 线程，每题数据文件限制为 500 KB。`make build` 会导出题库、构建网页，并检查离线构建结果。

## 用自己的编辑器或终端做题

网页编辑器与 CLI 使用同一套原生 Python 测试。使用 `paperleet` 命令时，保持虚拟环境处于激活状态。

```bash
# 浏览题目，并创建自己的实现文件。
paperleet list
paperleet init architecture/rms_norm

# 用习惯的编辑器修改 workspace/architecture/rms_norm.py。
paperleet test architecture/rms_norm

# 调试时只运行指定层，或指定报告位置。
paperleet test architecture/rms_norm --layers T0,T2 --report reports/rmsnorm-debug.json

# 显式测试另一个实现，也可以用参考解做对照。
paperleet test architecture/rms_norm --impl problems/architecture/rms_norm/solution.py
```

CLI 同时接受数字练习编号和稳定的目录 ID。`paperleet init` 默认保留已有工作区文件；只有明确使用 `--force` 才会用起始代码替换它。早期的 `leetpaper` 命令仍作为兼容别名保留。

在 IDE 中修改文件后，可在网页点击**载入本机文件**，读取对应的 `workspace/` 文件。若它与编辑器草稿不同，界面会先确认再替换。下载按钮可以把当前编辑器代码保存为 Python 文件。

## 代码与进度保存在哪里

| 位置 | 内容 |
| --- | --- |
| 浏览器存储 | 编辑器草稿、最近反馈、已通过进度，以及语言、主题和布局偏好。 |
| `workspace/<category>/<module>.py` | 自己的本机实现，由 `init` 创建，或在网页运行时保存。 |
| `reports/` | 本机测试结果、参考报告及质量检查报告；普通模块报告对应最近一次运行。 |
| `.paperleet/` | 本机任务状态与源码快照，用于保留实际运行的那份代码。 |

`workspace/`、`reports/` 和 `.paperleet/` 都被 Git 忽略，更新项目时请保留。浏览器记录属于当前浏览器和访问地址：切换 `localhost` / `127.0.0.1`、改变端口或换浏览器，都会使用独立的浏览器存储。清理网站数据会移除浏览器记录，但不会删除仓库中的本机文件。

输入代码时，草稿会自动保存到浏览器。测试使用点击运行时捕获的代码快照，完成后不会替换更新的编辑器内容。如果任务等待期间 IDE 又修改了工作区文件，本机服务会保留较新的文件，并单独保留此次运行的源码。浏览器保存失败时会有提示，可以用**下载实现**保留副本。

本机判题以当前用户权限执行 Python，只绑定回环地址，用于运行自己的练习代码。应用不会把代码或结果上传到 PaperLeet 在线服务。主动打开论文链接、安装依赖和获取题库更新时需要联网。

## 持续更新题库

维护题库跟随 [PaperLeet 公共主干](https://github.com/PaperLeet/paperleet.github.io)。[同步 workflow](.github/workflows/sync-upstream.yml)每天检查，也可以在 **Actions** 中手动运行：

**固定上游版本 → 提取白名单变化 → 离线检查 → 同一个更新 PR → 维护者审核合并。**

同步范围包含题目包、论文信息、判题公共代码、合集成员及单题可视化数据；离线界面和本机服务适配层由本仓库维护。[锁定文件](sync/upstream.lock.json)记录导入的 commit 和各文件哈希。管理文件有本地修改时，同步会停止，不会静默覆盖；上游只有在线界面变更时，不会产生更新 PR。

题库更新合并后，先停止本机应用，再在仓库根目录执行：

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e .
npm --prefix web ci
make dev
```

被 Git 忽略的解答与报告目录会保留。需要长期修改题目文件时，请放到自己的独立分支中。[同步说明](docs/upstream-sync.md)介绍了固定版本、冲突处理、手动运行和仓库权限。

## 目录与质量检查

```text
papers.yaml             论文贡献、必做模块与模块顺序
problems/               双语题目、起始代码、参考解、测试、错误变体和实验
paperleet/              CLI、本机 CPU 判题、本地服务与公共数学工具
web/                    React + TypeScript 练习界面和实验数据
scripts/                本地启动与上游同步
sync/                   同步白名单和固定文件清单
tests/                  框架、本机服务与同步回归检查
docs/                   课程来源记录与同步说明
workspace/              自己的实现（Git 忽略）
reports/                自己的报告（Git 忽略）
.paperleet/             本机判题数据（Git 忽略）
```

```bash
make check  # 导出题库，运行框架和前端检查。
make build  # 导出、构建网页，并验证离线构建。
make ci     # 额外检查全部参考解、错误实现与 CPU 预算；运行一轮。

# 可选：对完整参考解和错误实现重复检查三轮。
paperleet ci --repeats 3 --budget 1200
```

`make ci` 使用一轮检查和 600 秒题库预算；显式运行三轮的命令会进一步比较重复结果是否一致。完整题库检查的工作量明显大于运行一道练习。结果保存在 `reports/` 下，包括 `reports/ci.json`。[GitHub Actions 质量检查](.github/workflows/ci.yml)安装 CPU-only 依赖，并上传前端构建产物。

Git 历史保留了以前的合并应用；当前目录是可以独立运行的离线版。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 做题编辑器底部显示**本机服务未连接** | 使用 `make dev` 启动，并保持终端开启。只打开构建页面不会启动 Python。 |
| 5173 或 8765 端口已被占用 | 对此前的 PaperLeet 终端按 Ctrl-C 停止，再重新启动。 |
| 找不到 `paperleet` 命令 | 激活 `.venv`，在仓库根目录执行 `python -m pip install -e .`。 |
| 看不到新练习 | 重新运行 `make dev` 导出题库，并确认最近一次 `git pull` 已成功。 |
| CLI 修改没有出现在网页编辑器中 | 点击**载入本机文件**；浏览器草稿与磁盘文件是两份独立内容。 |
| 浏览器进度看起来为空 | 检查是否使用相同浏览器、主机名和端口；本机源码和报告文件独立保留。 |

## 参与贡献

欢迎补充更清晰的推导、反例、失败提示、数学性质测试、教学实验、翻译，以及本地编辑与判题体验的改进。

- **共享课程内容：** 新论文、题面、参考解和测试改进，请提交到 [PaperLeet/paperleet.github.io](https://github.com/PaperLeet/paperleet.github.io)，让审核后的内容进入两个版本。
- **离线应用：** 安装、界面、本地存储和原生判题的改进，提交到本仓库。
- **问题反馈：** 请附上练习 ID、最小复现、Python/PyTorch 版本，以及期望和实际行为。

详细说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。请保留来源署名、稳定模块 ID、双语讲解，以及论文主张和测试之间的联系，让后来的学习者既能理解正确实现，也能理解附近容易出现的错误。

如果 PaperLeet 帮你更深入地读懂了一篇论文，欢迎给仓库点一个 **Star**，也可以分享给读书小组。一条清楚的问题反馈、一处讲解修正或一个新测试，同样能帮助题库持续完善。

## 许可证

PaperLeet 离线版采用 [MIT 许可证](LICENSE)。请保留题包已有的来源署名和第三方声明；原论文与引用资源继续注明各自作者。
