# PaperLeet

[English](README.md) | **简体中文**

读论文，写实现，用数学性质和真实行为验证理解。

当前试点包含 **9 篇论文、10 个核心模块**。先选择论文，了解核心贡献和本次练习范围，再进入模块的 LeetCode 风格分栏工作区。辅助代码直接提供，练习聚焦论文的关键机制。“论文完成”表示选定核心模块全部通过。论文组织保存在 `papers.yaml`，题面、实现、测试和可视化数据来自 `problems/`。

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

仓库提供小规模在线内测的部署包，用户可以在自己的电脑上打开网页，完成 **10 个精选模块**。默认机构模式需要邀请和机构邮箱验证。首批测试可以使用 `PAPERLEET_AUTH_MODE=test`：管理员为每位测试者创建登录账号和访问码，无需邮箱注册，也无需配置 Supabase 或 SMTP，具体见 [内测账号指南](docs/test-accounts.zh-CN.md)。线上判题在限制资源的独立沙箱中运行；草稿、判题结果与提交历史按账号私密保存。浏览器草稿缓存按用户隔离；其他设备更新了同一草稿时，需要明确选择保留哪个版本。

按照 [部署指南](docs/hosting.zh-CN.md) 配置服务器、沙箱、持久化存储和所选账号模式；只有机构模式需要邮件服务与邀请名单。仓库提供部署文件，不会自动发布到公网。本地 `make dev` 工作流保持不变，仍使用你自己的机器执行代码。

## 开始第一题

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击“运行全部测试”，等待判题结果自动显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。

完整测试全部通过后，完成状态自动更新并解锁坑点。只跑部分测试不会解锁“全部通过”状态。本地模式的草稿和学习进度保存在当前浏览器中；托管模式中，用户登录后，草稿与判题结果会同步到账号。右上角可切换中文 / English，界面、题面和讲解随之切换。论文入口使用数字编号 1–9；模块编号 1–10 和内部目录名保持不变，兼容已有草稿和 CLI 命令。后续运行失败不会抹掉此前已通过的记录。旧草稿会保留；如需新版辅助代码，可先备份代码，再使用编辑器的重置功能。

网页通过判题服务执行代码并自动显示结果，无需手动处理报告文件。本地服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；代码保存到 `workspace/`，由本机 Python/PyTorch 执行。这一启动方式面向个人本地使用。多人在线判题请使用独立的托管配置，不要直接公开开发服务。

## 社区与用户账号

在顶部 **社区 / Community** 浏览讨论、发帖与回复、点赞或转发，并关注其他研究者。个人公开主页展示帖子、回复、点赞、转发，以及关注与粉丝列表。机构模式的写入操作需要先验证 `institutions.yaml` 中已批准机构的邮箱；线上还需要管理员邀请。测试模式使用管理员分配的账号和访问码登录，个人主页标注“内测账号”，不宣称已通过机构邮箱认证。

昵称和生成的头像始终公开。已验证机构、国家／地区分别提供展示开关，默认均隐藏；内测账号没有机构字段，完整邮箱始终私有。国家由用户选填，不查询 IP。托管模式中的练习数据私密同步到账号，本地模式仍保存在浏览器。公开学习统计尚未开放，进度展示开关暂不发布这些记录。

机构邮箱登录使用 Supabase Auth，需配置服务端环境；内测账号单独创建，登录不经过邮件服务。复制 `.env.example` 为 `.env` 并填写后，必须在启动服务的终端显式导出环境变量。OTP 邮件模板、SMTP、机构审核与隐私规则见 [社区配置指南（英文）](docs/community.md)。在线内测请使用 [部署指南](docs/hosting.zh-CN.md) 中的独立判题沙箱和邀请控制。

## Submit paper

**在线内测模式** 下，已登录用户可以提交论文提案并查看自己的提交，管理员可以查看全部提案。生成、验证和发布由维护者通过离线流程完成。线上判题范围固定为部署镜像中的 10 个精选模块。

**本地维护模式** 下，点击顶部 **Submit paper / 提交论文**，填写论文链接、可选的官方代码链接和希望练习的内容。模型读取来源，生成中英文题面、starter、参考实现、分层测试和错误变体；通过隔离验证后，进入人工评审。只有人工批准的当前版本才会分配数字模块编号并进入论文题库。导出时会根据 arXiv 标识或来源 URL，将同篇论文的模块合并到一个入口。目前生成器每次提交生成一个候选模块。

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

`paperleet ci` 检查参考解全部通过、每个 mutant 至少被一个真实失败抓到、每个 mutant 差异不超过 10 行、重复三次一致、T3 CPU 预算与总运行时间。结果写入 `reports/ci.json`，每题完整参考报告也保存在 `reports/`。导入/语法错误不会被当作有效 mutant 检出。

## 论文与核心模块

| 论文编号 | 论文 | 选定核心模块 | 模块编号 |
| --- | --- | --- | --- |
| 1 | GAE | 优势反向递推与 episode 边界 | 1 |
| 2 | PPO | 截断策略目标 | 2 |
| 3 | DPO | 偏好损失与隐式奖励 | 3 |
| 4 | DeepSeekMath | GRPO 组内相对优势与策略目标 | 4 |
| 5 | Dr. GRPO | 优势缩放与固定长度归一化 | 5 |
| 6 | DDPM | 前向加噪；后验与反向一步 | 6、8 |
| 7 | Progressive Distillation | v 参数化与转换 | 7 |
| 8 | DDIM | 单步采样更新 | 9 |
| 9 | Flow Matching | 条件路径、目标速度与回归损失 | 10 |

每篇论文单独划定练习范围。Progressive Distillation 当前选取其中的 v 参数化支撑技术，尚未包含蒸馏训练。DDPM 配套的 cosine 调度来自 Improved DDPM，作为相关工作辅助代码提供。完整网络组装和大规模训练不属于本次必做范围。

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

`vizdata` 使用参考实现、固定种子和两条 PyTorch CPU 线程。每题独立数据文件限制为 500 KB。前端能直接计算的概念图随滑块实时更新。`npm run check` 验证 10 题图表渲染、轨迹解码、报告图、完成状态，以及私有工作区隔离、草稿冲突和通过记录保留。

`web/dist/` 可作为阅读论文和查看可视化的静态预览。账号工作区、代码编辑、同步和判题需要配置 `/api` 服务。交互式内测请使用 [托管部署包](docs/hosting.zh-CN.md)；仓库不会自动部署公开服务。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
institutions.yaml  经审核的机构邮箱域名
problems/          独立核心模块包（rl / diffusion）
web/               Vite + React + TypeScript
tests/             判题框架的集成测试
workspace/         个人代码，忽略于 Git
reports/           判题和 CI 报告，忽略于 Git
.paperleet/        本地社区数据库与论文提交材料，忽略于 Git
```

出题流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，详细验收依据见 [10 题试点规格](PaperLeet_第一步_10题试点规格.md)。GitHub Actions 运行框架测试、三次质量门、数据生成和前端构建，上传报告与静态站构建产物。

本次实现的实测结果与验证范围见 [试点验收记录](docs/validation/pilot.md)。
