# PaperLeet

[English](README.md) | **简体中文**

读论文，写实现，用数学性质和真实行为验证理解。

当前试点包含 **24 篇论文、30 个核心模块**。先选择论文，了解核心贡献和本次练习范围，再进入模块的 LeetCode 风格分栏工作区。辅助代码直接提供，练习聚焦论文的关键机制。“论文完成”表示选定核心模块全部通过。论文组织保存在 `papers.yaml`，题面、实现、测试和可视化数据来自 `problems/`。

## 本地启动

需要 Python 3.10+、Node.js 20.19+。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paperleet export
npm --prefix web ci
make dev
```

打开终端显示的地址（默认 **http://127.0.0.1:5173**）。以后在根目录运行 **`make dev`** 即可同时启动网页与判题服务；按 Ctrl-C 会关闭两个进程。

## 部署邀请制内测

仓库提供小规模在线内测的部署包，用户可以在自己的电脑上打开网页，完成 **30 个精选模块**。默认机构模式需要邀请和机构邮箱验证。首批测试可以使用 `PAPERLEET_AUTH_MODE=test`：管理员为每位测试者创建登录账号和访问码，无需邮箱注册，也无需配置 Supabase 或 SMTP，具体见 [内测账号指南](docs/test-accounts.zh-CN.md)。线上判题在限制资源的独立沙箱中运行；草稿、判题结果与提交历史按账号私密保存。浏览器草稿缓存按用户隔离；其他设备更新了同一草稿时，需要明确选择保留哪个版本。

按照 [部署指南](docs/hosting.zh-CN.md) 配置服务器、沙箱、持久化存储和所选账号模式；只有机构模式需要邮件服务与邀请名单。仓库提供部署文件，不会自动发布到公网。本地 `make dev` 工作流保持不变，仍使用你自己的机器执行代码。

## 开始第一题

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击“运行全部测试”，等待判题结果自动显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。

完整测试全部通过后，完成状态自动更新并解锁坑点。只跑部分测试不会解锁“全部通过”状态。本地模式的草稿和学习进度保存在当前浏览器中；托管模式中，用户登录后，草稿与判题结果会同步到账号。右上角可切换中文 / English，界面、题面和讲解随之切换。论文入口使用数字编号 1–24；模块编号 1–30 和内部目录名保持不变，兼容已有草稿和 CLI 命令。后续运行失败不会抹掉此前已通过的记录。旧草稿会保留；如需新版辅助代码，可先备份代码，再使用编辑器的重置功能。

网页通过判题服务执行代码并自动显示结果，无需手动处理报告文件。本地服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；代码保存到 `workspace/`，由本机 Python/PyTorch 执行。这一启动方式面向个人本地使用。多人在线判题请使用独立的托管配置，不要直接公开开发服务。

## 社区与用户账号

在顶部 **社区 / Community** 浏览讨论、发帖与回复、点赞或转发，并关注其他研究者。个人公开主页展示帖子、回复、点赞、转发，以及关注与粉丝列表。机构模式的写入操作需要先验证 `institutions.yaml` 中已批准机构的邮箱；线上还需要管理员邀请。测试模式使用管理员分配的账号和访问码登录，个人主页标注“内测账号”，不宣称已通过机构邮箱认证。

昵称和生成的头像始终公开。已验证机构、国家／地区分别提供展示开关，默认均隐藏；内测账号没有机构字段，完整邮箱始终私有。国家由用户选填，不查询 IP。托管模式中的练习数据私密同步到账号，本地模式仍保存在浏览器。公开学习统计尚未开放，进度展示开关暂不发布这些记录。

机构邮箱登录使用 Supabase Auth，需配置服务端环境；内测账号单独创建，登录不经过邮件服务。复制 `.env.example` 为 `.env` 并填写后，必须在启动服务的终端显式导出环境变量。OTP 邮件模板、SMTP、机构审核与隐私规则见 [社区配置指南（英文）](docs/community.md)。在线内测请使用 [部署指南](docs/hosting.zh-CN.md) 中的独立判题沙箱和邀请控制。

## Submit paper

**在线内测模式** 下，已登录用户可以提交论文提案并查看自己的提交，管理员可以查看全部提案。生成、验证和发布由维护者通过离线流程完成。线上判题范围固定为部署镜像中的 30 个精选模块。

**本地维护模式** 下，点击顶部 **Submit paper / 提交论文**，填写论文链接、可选的官方代码链接和希望练习的内容。模型读取来源，生成中英文题面、starter、参考实现、分层测试和错误变体；通过隔离验证后，进入人工评审。只有人工批准的当前版本才会分配数字模块编号并进入论文题库。导出时会根据 arXiv 标识或来源 URL，将同篇论文的模块合并到一个入口。目前生成器每次提交生成一个候选模块。生成时会寻找作者 toy 和论文图示，明确标注教学改编，并让 T3 通过有大小限制的 `artifact` fixture 输出实际测量的曲线和指标，供人工审核。

生成需要在启动服务的终端配置 `OPENAI_API_KEY` 和 `PAPERLEET_MODEL`。验证需要启动 Docker，并在仓库根目录构建镜像：

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

未配置模型或 Docker 时仍可保存提交，网页会说明生成或验证尚缺的条件。候选材料与评审记录保存到 `.paperleet/submissions/`。本地评审由工作区所有者完成；托管模式不会向普通用户开放生成、验证或发布操作。完整流程、隔离边界与人工评审标准见 [harness 设计](docs/paper-submission-harness.md)。

## 维护者：CLI 与质量门

维护题目、调试判题或检查参考解时，可以使用 CLI：

```bash
paperleet list
paperleet init diffusion/param_conversion
# 编辑 workspace/diffusion/param_conversion.py
paperleet test diffusion/param_conversion
paperleet test diffusion/param_conversion --layers T0,T2 --report reports/my-result.json
paperleet test rl/gae --impl problems/rl/gae/solution.py
paperleet mutate diffusion/param_conversion
paperleet ci
```

`init` 默认不覆盖已有代码；确实要重置时使用 `--force`。

`paperleet ci` 检查参考解全部通过、每个 mutant 至少被一个真实失败抓到、每个 mutant 差异不超过 10 行、重复三次一致、T3 CPU 预算与总运行时间。扩展后的 30 模块目录总预算为 300 秒（可通过 `--budget` 设置），单个 T3 的时限仍由各题包声明。结果写入 `reports/ci.json`，每题完整参考报告也保存在 `reports/`。导入/语法错误不会被当作有效 mutant 检出。

## 论文与核心模块

| 论文编号 | 论文 | 核心范围 | 模块编号 |
| --- | --- | --- | --- |
| 1 | GAE | 实现 GAE 的反向递推和 episode 边界处理。 | 1 |
| 2 | PPO | 实现 clipped surrogate 及其梯度行为；价值损失、归一化和统计代码已提供。 | 2 |
| 3 | DPO | 实现偏好损失与隐式奖励；token 对齐和序列 log-prob 计算已提供。 | 3 |
| 4 | DeepSeekMath · GRPO | 选取论文的 GRPO 模块：组内相对优势、截断目标和回答级聚合。 | 4 |
| 5 | Dr. GRPO | 实现去除组内标准差缩放与固定生成长度归一化这两处修正。 | 5 |
| 6 | DDPM | 实现闭式前向加噪，以及高斯后验和 epsilon 参数化的反向一步；调度器与采样循环已提供。 | 6, 8 |
| 7 | Progressive Distillation | 先实现 v 参数化转换，再推导两步教师到一步学生的蒸馏目标并验证更新。 | 7, 26 |
| 8 | DDIM | 实现 DDIM 单步更新和干净端点；时间步选择、循环与反演探索代码已提供。 | 9 |
| 9 | Flow Matching | 实现条件路径、目标速度和 CFM 回归损失；时间采样、Euler 积分和训练辅助代码已提供。 | 10 |
| 10 | GSPO | 实现序列概率比与 GSPO 损失，验证掩码、截断区域和梯度。 | 11 |
| 11 | GKD · On-policy Distillation | 实现学生 rollout 和全词表散度；在小型自回归模型上验证。OPD 在此指 on-policy distillation。 | 12 |
| 12 | DAPO | 实现组筛选与 token 级目标；采样系统和分布式训练不在本模块范围。 | 13 |
| 13 | LoRA | 实现低秩增量和线性前向，验证冻结边界及小型回归学习。 | 14 |
| 14 | QLoRA | 实现分块 NF4 量化、反量化与适配器前向；不模拟双重量化或分页优化器。 | 15 |
| 15 | MeanFlow | 实现平均流的 JVP 目标和梯度边界，并运行小型生成实验。 | 16 |
| 16 | JiT | 实现 x-prediction 到速度的转换与去噪目标，使用论文启发的低维流形嵌入实验。 | 17 |
| 17 | Drifting Models | 实现漂移场与停止梯度损失，在小型二维分布上验证生成器行为。 | 18 |
| 18 | DMD | 实现 score 差目标与配对回归项，用解析双峰分布检查模式覆盖。 | 19 |
| 19 | DMD2 | 实现双时间尺度更新与辅助目标，在小型实验中验证参数更新边界。 | 20 |
| 20 | Consistency Models | 实现边界参数化和相邻时刻的一致性蒸馏损失。 | 21 |
| 21 | Transformer | 分五题实现缩放点积注意力、多头注意力、LayerNorm 与 Add & Norm、正弦位置编码、逐位置 FFN。 | 22, 27–30 |
| 22 | AdamW | 实现矩估计、偏差修正与解耦权重衰减的一步更新。 | 23 |
| 23 | FlashAttention | 实现分块递推与重缩放。这是 CPU 算法练习，不评价 GPU kernel 性能。 | 24 |
| 24 | CLIP | 实现特征归一化、温度缩放和双向对比损失，使用小型配对检索实验。 | 25 |

首页按六个领域分类。新增模块包含 CPU toy 实验、出处与改编说明，以及 T0–T3 验证；实验页区分参考曲线和用户代码生成的 T3 结果。练习覆盖精选核心算法，不要求完整模型训练或 GPU kernel 性能复现。

每个模块通过 `module.required_functions` 和 `module.provided_functions` 声明需要实现及已经提供的函数，starter 仅将必做函数留空。判题仍会使用辅助代码调用用户实现。每个模块均包含中英文题面和坑点、参考解、分层测试与错误变体。模块 6、7 无需 T3；模块 5 的长度趋势只作展示，不据此判分。

## 可视化与静态构建

仓库提供预生成数据，可直接启动。修改题目或实验后重新导出：

```bash
paperleet vizdata all
paperleet export
cd web
npm run check
npm run build
```

`vizdata` 使用参考实现、固定种子和两条 PyTorch CPU 线程。每题独立数据文件限制为 500 KB。前端能直接计算的概念图随滑块实时更新。`npm run check` 验证全部维护模块的图表渲染、轨迹解码、报告图、完成状态，以及私有工作区隔离、草稿冲突和通过记录保留。

`web/dist/` 可作为阅读论文和查看可视化的静态预览。账号工作区、代码编辑、同步和判题需要配置 `/api` 服务。交互式内测请使用 [托管部署包](docs/hosting.zh-CN.md)；仓库不会自动部署公开服务。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
institutions.yaml  经审核的机构邮箱域名
problems/          独立核心模块包（强化学习、微调、生成模型、架构、优化、表示学习）
web/               Vite + React + TypeScript
tests/             判题框架的集成测试
workspace/         个人代码，忽略于 Git
reports/           判题和 CI 报告，忽略于 Git
.paperleet/        本地社区数据库与论文提交材料，忽略于 Git
```

出题流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，详细验收依据见 [10 题试点规格](PaperLeet_第一步_10题试点规格.md)。GitHub Actions 运行框架测试、三次质量门、数据生成和前端构建，上传报告与静态站构建产物。

本次实现的实测结果与验证范围见 [试点验收记录](docs/validation/pilot.md)。

24 篇论文扩展版的数值验证、toy 来源、网页检查及 harness 验证见[验证记录](docs/validation/core-curriculum.md)。
