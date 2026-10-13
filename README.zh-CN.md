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

## 开始第一题

1. 在题库中选择论文，例如 **6 · DDPM**，再打开“前向加噪”模块。
2. 阅读模块目标，在右侧编辑器中补全标注为需要实现的函数；调度器等辅助函数已经提供。
3. 点击“运行全部测试”，等待判题结果自动显示。
4. 根据各层测试的失败原因、期望值与实际值修改代码，再次运行。

完整测试全部通过后，完成状态自动更新并解锁坑点。只跑部分测试不会解锁“全部通过”状态。草稿和学习进度保存在当前浏览器中。右上角可切换中文 / English，界面、题面和讲解随之切换。论文入口使用数字编号 1–9；模块编号 1–10 和内部目录名保持不变，兼容已有草稿和 CLI 命令。后续运行失败不会抹掉此前已通过的记录。旧草稿会保留；如需新版辅助代码，可先备份代码，再使用编辑器的重置功能。

网页通过判题服务执行代码并自动显示结果，无需手动处理报告文件。本地服务监听 `127.0.0.1:8765`，Vite 代理 `/api` 请求；代码保存到 `workspace/`，由本机 Python/PyTorch 执行。目前的执行方式面向个人本地使用，尚未提供隔离的多用户代码沙箱。

## Submit paper

点击顶部 **Submit paper / 提交论文**，填写论文链接、可选的官方代码链接和希望练习的内容。模型读取来源，生成中英文题面、starter、参考实现、分层测试和错误变体；通过隔离验证后，进入人工评审。只有人工批准的当前版本才会分配数字模块编号并进入论文题库。导出时会根据 arXiv 标识或来源 URL，将同篇论文的模块合并到一个入口。目前生成器每次提交生成一个候选模块。

生成需要在启动服务的终端配置 `OPENAI_API_KEY` 和 `PAPERLEET_MODEL`。验证需要启动 Docker，并在仓库根目录构建镜像：

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

未配置模型或 Docker 时仍可保存提交，网页会说明生成或验证尚缺的条件。候选材料与评审记录保存到 `.paperleet/submissions/`。当前评审者是本地工作区所有者；多人权限与线上部署另行接入。完整流程、隔离边界与人工评审标准见 [harness 设计](docs/paper-submission-harness.md)。

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

`vizdata` 使用参考实现、固定种子和两条 PyTorch CPU 线程。每题独立数据文件限制为 500 KB。前端能直接计算的概念图随滑块实时更新。`npm run check` 验证 10 题图表渲染、轨迹解码、报告图和完成状态。

`web/dist/` 可部署到 GitHub Pages 等静态托管，但静态页面本身不能执行 Python。线上运行代码还需要部署判题服务并配置 `/api` 请求转发；仅部署静态文件时可以阅读题目、编辑和下载代码。仓库未配置自动发布，也未部署云端判题服务。

## 目录与贡献

```text
paperleet/         CLI、pytest runner、本地服务、公共数学工具
papers.yaml        论文、核心贡献、练习范围与模块顺序
problems/          独立核心模块包（rl / diffusion）
web/               Vite + React + TypeScript
tests/             判题框架的集成测试
workspace/         个人代码，忽略于 Git
reports/           判题和 CI 报告，忽略于 Git
```

出题流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，详细验收依据见 [10 题试点规格](PaperLeet_第一步_10题试点规格.md)。GitHub Actions 运行框架测试、三次质量门、数据生成和前端构建，上传报告与静态站构建产物。

本次实现的实测结果与验证范围见 [试点验收记录](docs/validation/pilot.md)。
