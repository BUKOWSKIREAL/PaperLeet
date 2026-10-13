# 第一步：RL + Diffusion 10 题试点规格

> 2026-10-13：本文件保留最初 10 模块试点的验收规格。当前目录扩展为 24 篇论文、30 个核心模块，按首页六类组织；最新编号、实现范围和 toy 出处见 [README](README.zh-CN.md#论文与核心模块)、`papers.yaml` 及各模块题面。
> 当前实现范围（2026-10-13 更新）：题库以论文为一级单位，9 篇论文下组织 10 个精选核心模块。每篇说明核心贡献与练习范围，调度、循环和统计等辅助代码直接提供；完成选定核心模块即可完成本篇练习。当前模块划分以 [中英文 README](README.zh-CN.md#论文与核心模块)、`papers.yaml` 与各模块 `problem.yaml` 为准。下文保留最初提案及数学/测试依据，其中旧路径展示、题号和要求实现全部辅助函数的描述已被本次调整取代。

Oct 13, 2026 · @Hongyang

第一步用 10 道题（RL 5 道 + Diffusion 5 道）跑通整条链路：paper 讲解 → 参考解 → 四层测试 → mutant 质量门 → 可视化。本页是给 code agent 的实现规格，按"里程碑与验收"一节的顺序实现；公式与约定以本页为准，原文核对项列在最后。

## 范围与题单

10 道题覆盖两部分。一是面试高频的 RL 四件套（GAE、PPO、DPO、GRPO）加一道变体题；二是 diffusion 的主干链路（调度 → 参数化 → DDPM → DDIM → Flow Matching）。D3、D4 用解析 oracle 代替网络，T3 不需要训练；只有 D5 需要 CPU 上 ≤ 10 s 的小训练。

| ID | 题目 | Paper（arXiv） | 题型 | 难度 | T3 行为测试 | 核心可视化 |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | GAE | Schulman 2016（1506.02438） | 实现 | M | 随机游走 MDP 上，偏差随 λ 增大而降、方差随 λ 增大而升 | γ/λ 滑块下的逐步 δ 与 A；episode 边界 |
| R2 | PPO clipped surrogate | Schulman 2017（1707.06347） | 实现 | M | bandit 上同一 batch 多 epoch 更新，KL 受控且策略收敛 | L^CLIP(r) 曲线 + ε 滑块，零梯度区阴影 |
| R3 | DPO | Rafailov 2023（2305.18290） | 实现 | E | 5 选项 toy 偏好数据上，策略收敛到闭式最优解 | 标签移位与 mask 网格；loss 随 margin 变化 |
| R4 | GRPO | Shao 2024（2402.03300） | 实现 | M | toy 自回归策略上奖励上升 | 组内 advantage 柱状图；逐 token 梯度权重热图 |
| R5 | Dr. GRPO | Liu 2025（2503.20783） | 变体（前置 R4） | M | 长度偏置效应，仅展示、不判分 | GRPO 与 Dr. GRPO 的逐 token 权重对比 |
| D1 | 噪声调度与前向加噪 | Ho 2020（2006.11239）；Nichol 2021（2102.09672） | 实现 | E | 无，T2 已覆盖 | ᾱ 与 log-SNR 曲线；2D 点云随 t 加噪 |
| D2 | x₀ / ε / v 互转 | Ho 2020；Salimans & Ho 2022（2202.00512） | 实现 | E | 无，T2 已覆盖 | (x₀, ε) 平面上的四分之一圆与 v 切向量 |
| D3 | DDPM 后验与祖先采样 | Ho 2020 | 实现 | M | 8-高斯 GMM + 解析 ε：MMD 与模式覆盖 | 反向扩散粒子动画 |
| D4 | DDIM | Song 2021（2010.02502） | 实现 | M | 同 D3，50 步 η=0；反演往返误差 | η × 步数网格的轨迹对比 |
| D5 | Flow Matching | Lipman 2023（2210.02747）；Liu 2023（2209.03003） | 实现 | M | ≤ 10 s 训练小 MLP：MMD 与模式覆盖 | 粒子流动 + 向量场；Euler 步数对比 |

依赖规则：每题的参考解自包含。后续题的签名直接接收调度张量，例如 D3 接收 alpha\_bars，不依赖做题者自己的 D1 实现。变体题 R5 的 starter 直接 import R4 参考解，只改动差异部分。

## 仓库结构与判题

判题完全本地运行：做题者在 workspace 里写实现，`paperleet test` 用 pytest 跑四层测试并输出 report.json，网页再把报告画出来。mutant 门禁用的是同一套测试，只是把被测文件换成 mutant。

```text
paperleet/
├── pyproject.toml            # python>=3.10；torch(CPU)、numpy、pytest、pytest-timeout、pyyaml
├── paperleet/
│   ├── cli.py                # paperleet 命令入口
│   ├── judge/
│   │   ├── loader.py         # 按文件路径加载被测实现
│   │   ├── runner.py         # 跑 pytest，收集结果与 artifacts，写 report.json
│   │   └── mutation.py       # 对参考解与每个 mutant 跑全套测试，输出 mutant × test 矩阵
│   └── common/               # 公共工具，见"公共工具"一节
├── problems/
│   ├── rl/{gae,ppo_clip,dpo,grpo,dr_grpo}/
│   └── diffusion/{schedules,param_conversion,ddpm_sampling,ddim,flow_matching}/
├── workspace/                # 做题者的实现（gitignore）
├── web/                      # Vite + React + TS 静态站
└── .github/workflows/ci.yml  # 质量门
```

### CLI

| 命令 | 作用 |
| --- | --- |
| `paperleet list` | 列出所有题、难度与本地通过状态 |
| `paperleet init rl/gae` | 把 starter.py 复制到 workspace/rl/gae.py |
| `paperleet test rl/gae [--impl FILE] [--layers T0,T2] [--report out.json]` | 默认测 workspace 下的实现；`--impl` 可指定任意文件 |
| `paperleet mutate rl/gae` | 对参考解和每个 mutant 跑测试并打印矩阵；有 mutant 存活即非零退出 |
| `paperleet ci` | 全部题：参考解全过、mutant 全被抓、重复 3 次结果一致、T3 计时达标 |
| `paperleet vizdata rl/gae` | 用参考解生成网页数据 web/public/data/rl-gae.json |

### 测试加载规则

- conftest 注册 `--impl` 选项；`impl` fixture 用 importlib 按路径加载模块，`ref` fixture 加载 solution.py。测试只能通过这两个 fixture 访问函数，禁止直接 import solution，否则 mutant 门禁失效。
- 层级用 pytest marker：`t0` `t1` `t2` `t3`。每个测试 docstring 的第一行写"为什么这个性质应该成立"，runner 把它写进报告。
- 失败信息必须定位到错误：给出索引、期望值与实际值，不能只有 `assert False`。
- 确定性：autouse fixture 固定 torch / numpy 种子，`torch.set_num_threads(2)`，开启 `torch.use_deterministic_algorithms(True)`。
- 时限：T0–T2 每个测试 5 s；T3 每个测试预算 10 s，硬超时 15 s。
- `artifact` fixture：T3 和部分 T2 测试用它把数组存进报告，比如采样点、训练曲线、期望值与实际值；点云最多下采样到 2000 个。

### report.json

```json
{
  "problem": "rl/gae",
  "impl": "workspace/rl/gae.py",
  "created_at": "2026-10-13T12:00:00Z",
  "summary": {"T0": [3, 3], "T1": [1, 1], "T2": [5, 6], "T3": [1, 1]},
  "tests": [
    {
      "id": "test_lambda_zero_is_td",
      "layer": "T2",
      "passed": false,
      "why": "λ=0 时 GAE 应退化为单步 TD 误差",
      "message": "max abs diff 0.31 at (t=4, env=0)",
      "duration_s": 0.02,
      "artifacts": {"expected": [0.1, 0.3], "got": [0.1, 0.61]}
    }
  ],
  "mutation": null
}
```

`paperleet mutate` 写同样的结构，`mutation` 字段为 `{mutant_id: [被哪些测试抓到]}`。

## 题目包规范

每道题是一个自包含目录。网页和 CLI 都只读这个目录，新增题目不需要改任何框架代码。

```text
problems/rl/gae/
├── problem.yaml       # 元数据与约定
├── README.md          # 题面 + paper 讲解（网页渲染）
├── starter.py         # 签名 + docstring + raise NotImplementedError
├── solution.py        # 参考解
├── notes.md           # 坑点：paper 与常见实现的差异（通过后解锁）
├── mutants/
│   ├── m1_no_done_mask.py   # 完整文件，与 solution.py 只差一个 bug
│   └── ...
├── tests/test_gae.py
└── viz/gen.py         # 用参考解生成可视化数据
```

### problem.yaml

```yaml
id: rl/gae
title: Generalized Advantage Estimation
paper:
  title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
  authors: Schulman, Moritz, Levine, Jordan, Abbeel
  venue: ICLR 2016
  arxiv: "1506.02438"
  anchors: ["GAE 定义式", "λ=0 与 λ=1 两个特例"]   # 讲解引用的原文位置，实现时填上公式编号
type: implement          # implement | variant | compare | debug
parent: null             # 变体题填前置题 id，如 rl/grpo
difficulty: M
functions:
  - name: compute_gae
    signature: "compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95) -> tuple[Tensor, Tensor]"
conventions:
  layout: "[T, N]，时间维在前"
  truncation: "bootstrap 用 next_values，但 advantage 不跨 episode 传播"
tests: {T0: 3, T1: 1, T2: 6, T3: 1}
t3_budget_s: 10
mutants:
  - id: m1_no_done_mask
    bug: "递推时没有乘 (1 - done)，advantage 跨 episode 泄漏"
viz: [gae_timeline, bias_variance_vs_lambda]
estimated_minutes: 25
```

### README 讲解模板

讲解写给刚进组的研究生，600–1000 中文字加公式，按以下顺序组织：

1. 一句话：这篇 paper 解决什么问题，不超过 2 句。
2. 核心公式：LaTeX，逐个说明符号的含义与 shape。
3. 从公式到代码：伪代码，标注每个张量的 shape 和循环方向。
4. 本题约定：表格对比本题的选择与 CleanRL、TRL、diffusers 等常见实现的选择。
5. 交互可视化：嵌入 1–2 个 viz，每个配一句"拖动什么、看什么"。
6. 怎么运行：init 与 test 命令。
7. 延伸：下一题和后续 paper。

notes.md 在测试全部通过后解锁，写 3–5 条坑点，每条包括现象、原因和抓它的测试名。

### starter 与 mutant 要求

- starter.py：完整签名、类型标注、docstring 写清 shape 与约定，函数体只有 `raise NotImplementedError`，不留暗示答案的注释。
- mutant：完整文件拷贝，第一行注释写明 bug。CI 用 diff 检查每个 mutant 与 solution.py 的差异 ≤ 10 行，保证一个 mutant 只对应一个 bug。

## 公共工具（paperleet/common）

公共工具的目标是让 T3 不依赖训练大模型：diffusion 题用高斯混合的解析解当"完美网络"，RL 题用可以精确求解的 toy 环境。

| 模块 | 内容 | 被哪些题用 |
| --- | --- | --- |
| `toydata.py` | `eight_gaussians(n, radius=2.0, std=0.15)`；`GaussianMixture(means, std, weights)`，含 `sample`、`log_prob`、`score`，以及 `diffused(alpha_bar)` 返回加噪后的混合分布 | D1、D3、D4、D5 |
| `oracles.py` | `gmm_eps_oracle`、`gaussian_velocity_oracle`、`point_velocity_oracle`，公式见下 | D3、D4、D5 |
| `metrics.py` | `mmd_rbf(x, y)`：无偏 MMD²，多带宽 (0.05, 0.1, 0.2, 0.5, 1.0) 求和；`mode_coverage(samples, centers, radius)`：每个模式的占比与离群占比 | D3、D4、D5 |
| `models.py` | `TimeMLP(dim=2, hidden=128, depth=3, t_dim=32)`，正弦时间嵌入 + SiLU，固定种子初始化 | D5 |
| `rl_envs.py` | `RandomWalkMDP(n_states=7)`，可用动态规划求精确 Q、V；`Bandit(k=5)`；`TablePolicyLM(max_len=8, vocab=5)`：逐位置独立 logits 的 toy 自回归策略，带 EOS | R1、R2、R4、R5 |
| `asserts.py` | `assert_close_located(got, exp, rtol, atol)`：失败时报告最大误差的位置与两边的值 | 全部 |

高斯混合加噪后仍是高斯混合。若 x₀ 服从 Σ π\_k N(μ\_k, s²I)，则 x\_t 服从：

```latex
p_t(x) = \sum_k \pi_k \, \mathcal{N}\!\left(x;\ \sqrt{\bar\alpha_t}\,\mu_k,\ (\bar\alpha_t s^2 + 1 - \bar\alpha_t) I\right)
```

完美 ε 预测器等于负的噪声标准差乘以精确 score：

```latex
\epsilon^*(x_t, t) = -\sqrt{1-\bar\alpha_t}\ \nabla_x \log p_t(x_t)
```

Flow Matching 采用 x\_t = (1−t)x₀ + t·x₁ 的约定，x₀ 为噪声，x₁ 为数据。数据为单点 μ 时，最优速度场为 (μ − x)/(1 − t)。数据为 N(μ, s²I) 时，最优速度场是线性的：

```latex
v^*(x, t) = \mu + \frac{t s^2 - (1-t)}{(1-t)^2 + t^2 s^2}\,(x - t\mu)
```

阈值标定流程：T3 的阈值等于参考解实测值 × 1.5，并写入 problem.yaml。标定时还要确认，所有影响采样的 mutant 都会超过阈值；若有 mutant 存活，先加强测试，而不是放宽阈值。

# RL 五题

## R1 · GAE（实现，M）

Paper：Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. arXiv:1506.02438。

**讲解要点**

- 问题：policy gradient 需要 advantage。MC 回报无偏但方差大；单步 TD 方差小，但会被 V 的误差带偏。GAE 用 λ 在两者之间连续插值。
- 关键结论：GAE 是 k 步 advantage 估计的指数加权平均，等价于 TD 误差的折扣和，所以反向递推一遍就能算完。
- 两个特例：λ=0 时退化为 δ\_t；λ=1 时等于 G\_t − V(s\_t)。
- 论文没写的（放进 notes.md）：terminated 与 truncated 的区分来自 Gymnasium API，论文年代没有这个区分，但它决定了截断步是否 bootstrap。

**签名与约定**

```python
def compute_gae(
    rewards: Tensor,      # [T, N]
    values: Tensor,       # [T, N]  V(s_t)
    next_values: Tensor,  # [T, N]  V(s_{t+1})；truncated 步用重置前的真实末状态
    terminated: Tensor,   # [T, N] bool：s_{t+1} 是终止状态，不 bootstrap
    truncated: Tensor,    # [T, N] bool：时间截断，要 bootstrap，但 advantage 不跨 episode 传播
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:  # (advantages [T, N], returns = advantages + values)
```

```latex
\delta_t = r_t + \gamma (1-\mathrm{term}_t)\, V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \delta_t + \gamma\lambda (1-\mathrm{done}_t)\, \hat A_{t+1}, \qquad \hat A_T = 0
```

其中 done\_t = term\_t ∨ trunc\_t。

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | 输出的 shape、dtype 与输入一致；T=1、N=1 可运行；无 NaN | 基本契约 |
| T1 | 随机输入下与参考解一致，rtol 1e-5 | 约定已完全钉死 |
| T2 | λ=0 时 Â = δ | 定义的特例 |
| T2 | λ=1 且无 done 时，Â\_t = Σ γ^k r\_{t+k} + γ^{T−t} V\_T − V\_t | 带 bootstrap 的 MC 特例 |
| T2 | 在 t 处 done 后扰动之后的奖励，t 及之前的 Â 不变 | advantage 不跨 episode |
| T2 | terminated 步修改 next\_values 不影响结果；truncated 步修改则会影响 | 两种结束方式的语义 |
| T2 | 结果对 rewards 线性；returns == advantages + values | 递推是线性的；返回值约定 |
| T3 | 在 RandomWalkMDP 上用 DP 求精确 Q、V，给 V 加固定扰动当作 V̂。在同一批 4000 条轨迹上，比较 GAE 与精确 advantage 构成的策略梯度估计：λ 取 0、0.5、0.9、1 时，偏差下降、方差上升 | 论文的核心论点：λ 控制偏差-方差折中 |

T3 若中间点的单调性对种子敏感，只断言两端：bias(0) > bias(1)，var(1) > var(0)，且 bias(1) 接近 0。

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_no\_done\_mask | 递推没有乘 (1 − done)，跨 episode 泄漏 | episode 隔离 |
| m2\_trunc\_as\_term | 把 truncated 当成 terminated，不 bootstrap | terminated / truncated 语义 |
| m3\_forward\_loop | 正向循环，用 Â\_{t−1} 递推 | T1、λ=1 特例 |
| m4\_returns\_mc | returns 用折扣奖励和，而不是 Â + V | returns 约定 |

**可视化**

- gae\_timeline：一条 T=20 的轨迹。上方柱状图画 r\_t，折线画 V(s\_t)；下方画 δ\_t 与 Â\_t。γ、λ 用滑块调节，由前端 TS 重算。episode 边界用两种标记区分 terminated 与 truncated；可叠加 λ=0 与 λ=1 两条参考线。
- bias\_variance\_vs\_lambda：T3 实验中偏差与方差随 λ 变化的两条曲线，λ 取 0 到 1 共 11 个点，由 viz/gen.py 预计算。

## R2 · PPO clipped surrogate（实现，M）

Paper：Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* 2017. arXiv:1707.06347。补充材料：Huang et al. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022。

**讲解要点**

- 问题：同一批数据做多轮更新时，vanilla PG 会把策略推得太远。TRPO 用 KL 约束解决，但需要二阶优化；PPO 用截断概率比做一阶近似。
- 关键：min(r·A, clip(r)·A) 是悲观下界，只截断"对目标有利"的方向，不利方向保留梯度。这是本题最容易写错的地方。
- 论文正文之外、但影响结果的实现细节：minibatch 内 advantage 归一化、value clipping、用 approx KL 监控和早停、clipfrac。
- 约定：loss 按最小化写，带负号。

**签名与约定**

```python
def ppo_loss(
    new_logp: Tensor,       # [B] 当前策略下动作的 log-prob（有梯度）
    old_logp: Tensor,       # [B] 采样时策略的 log-prob（常数）
    advantages: Tensor,     # [B]
    new_values: Tensor,     # [B]
    old_values: Tensor,     # [B]
    returns: Tensor,        # [B]
    entropy: Tensor | None = None,  # [B]
    clip_eps: float = 0.2,
    vf_clip: float | None = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    norm_adv: bool = True,   # minibatch 内 (A - mean) / (std + 1e-8)，std 用无偏估计
) -> dict[str, Tensor]:      # loss, pg_loss, v_loss, entropy, approx_kl, clipfrac
```

```latex
r = e^{\log\pi_\theta - \log\pi_{\text{old}}}, \qquad
L^{\text{pg}} = -\,\mathbb{E}\left[\min\left(r\hat A,\ \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\,\hat A\right)\right]
```

```latex
L^{V} = \tfrac12\,\mathbb{E}\left[\max\left((V-R)^2,\ (V_{\text{old}} + \mathrm{clip}(V - V_{\text{old}}, -c, c) - R)^2\right)\right]
```

approx\_kl = mean((r − 1) − log r)，clipfrac = mean(|r − 1| > ε)，两者都在 no\_grad 下计算。总 loss = L^pg + vf\_coef·L^V − ent\_coef·mean(entropy)。

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | 返回全部键，均为标量；entropy=None 可运行 | 基本契约 |
| T1 | 与参考解一致 | 约定已钉死 |
| T2 | new = old 时，∂L^pg/∂new\_logp = −Â/B | ratio=1 时退化为 vanilla PG |
| T2 | 关闭 norm\_adv，在四个象限检查梯度：A>0 且 r>1+ε 为 0；A>0 且 r<1−ε 非 0；A<0 且 r<1−ε 为 0；A<0 且 r>1+ε 非 0 | 只截断有利方向 |
| T2 | old\_logp、old\_values 不产生梯度 | 它们是采样时的常数 |
| T2 | approx\_kl ≥ 0，r=1 时为 0；clipfrac 与定义一致 | k3 估计器非负 |
| T2 | value loss 永不小于不截断的版本；\|V − V\_old\| ≤ c 时两者相等 | 取 max 的悲观形式 |
| T2 | norm\_adv=True 时，对 advantages 做正系数仿射变换，pg\_loss 不变 | 归一化的作用 |
| T3 | Bandit(k=5)：每轮采 256 个动作，做 10 个 epoch 的 minibatch 更新。30 轮内最优臂概率 > 0.9；在单个 batch 上更新 20 个 epoch，KL(π\_old ‖ π\_new) 低于阈值 | 截断带来的信赖域效果 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_clip\_only | 只用 −clip(r)·Â，没有和未截断项取 min | 四象限梯度 |
| m2\_max\_not\_min | min 写成 max（乐观界） | 四象限梯度、T3 KL |
| m3\_old\_logp\_grad | old\_logp 没有 detach | 常数无梯度 |
| m4\_ratio\_no\_exp | r 直接用 log-prob 之差，没有取 exp | T1、ratio=1 退化 |
| m5\_vclip\_min | value clipping 用 min | value loss 下界 |

**可视化**

- clip\_objective：A>0 与 A<0 两个面板，横轴 r ∈ \[0, 2\]。实线画 min(r·A, clip(r)·A)，虚线画 r·A；ε 滑块；零梯度区间加阴影；可拖动 r 上的点，显示该点的梯度箭头。
- trust\_region：T3 实验中 KL 随 epoch 变化的曲线，对比参考解与 m1、m2，直观说明截断为什么必要。数据预计算。

## R3 · DPO（实现，E）

Paper：Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. arXiv:2305.18290。

**讲解要点**

- 问题：RLHF 先训奖励模型、再做 PPO，流程长且不稳定。
- 关键推导：KL 约束下的最优策略有闭式解 π\* ∝ π\_ref·exp(r/β)。反解得到 r = β·log(π/π\_ref) + β·log Z(x)，而 Z(x) 在 Bradley–Terry 的差值里抵消，于是偏好学习变成一个二分类 loss。
- 前置：序列 log-prob 是回答部分每个 token 的 log-prob 之和。注意是求和不是平均，平均是 SimPO 的选择。
- 论文没写的（放进 notes.md）：标签要右移一位；prompt 部分要 mask；用 logsigmoid 保证数值稳定；label smoothing 对应保守 DPO（cDPO）。

**签名与约定**

```python
def sequence_logprob(
    logits: Tensor,      # [B, T, V]，位置 t 的 logits 预测 token t+1
    input_ids: Tensor,   # [B, T]
    loss_mask: Tensor,   # [B, T]，1 表示该位置的 token 属于回答
) -> Tensor:             # [B]：log_softmax(logits[:, :-1]) 取 input_ids[:, 1:]，乘 loss_mask[:, 1:] 后求和

def dpo_loss(
    pi_chosen: Tensor, pi_rejected: Tensor,     # [B] 策略的序列 log-prob
    ref_chosen: Tensor, ref_rejected: Tensor,   # [B] 参考模型的序列 log-prob（常数）
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:  # loss 标量，chosen_reward [B]，rejected_reward [B]（后两者 detach）
```

```latex
h = (\log\pi_c - \log\pi^{\text{ref}}_c) - (\log\pi_r - \log\pi^{\text{ref}}_r), \qquad
L = -(1-\varepsilon)\log\sigma(\beta h) - \varepsilon\log\sigma(-\beta h)
```

隐式奖励为 β·(log π − log π\_ref)，detach 后返回。

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | 输出 shape 正确；B=1 可运行 | 基本契约 |
| T1 | 两个函数都与参考解一致 | 约定已钉死 |
| T2 | sequence\_logprob 与逐 (b, t) 双重循环结果一致 | 定义 |
| T2 | 扰动最后一个位置的 logits，结果不变；扰动 loss\_mask\[:, t+1\]=0 处的 logits\[:, t\]，结果不变 | 右移与 mask 的语义 |
| T2 | policy = ref 时，loss = log 2（ε=0） | h=0 |
| T2 | β 乘 k、h 除以 k，loss 不变 | loss 只依赖 βh |
| T2 | βh = ±1e4 时 loss 与梯度有限 | 数值稳定 |
| T2 | ∂L/∂π\_c < 0，∂L/∂π\_r > 0；返回的 reward 不带梯度且等于 β(π − ref) | 优化方向；返回值约定 |
| T3 | toy 偏好：一个 prompt、6 个候选回答，真实奖励 r\* 固定，π\_ref 为固定的非均匀分布。按 Bradley–Terry 采 2 万对偏好，用 Adam 训练一个 categorical 策略。结束时隐式奖励与 r\* 的 Spearman 相关为 1，且 KL(π\_θ ‖ π\*) 低于阈值，其中 π\* ∝ π\_ref·exp(r\*/β) | 论文的闭式最优解 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_no\_shift | logits\[:, t\] 对 input\_ids\[:, t\] 打分 | T1、双重循环 |
| m2\_mask\_offbyone | 用 loss\_mask\[:, :-1\] 而不是 \[:, 1:\] | mask 语义 |
| m3\_naive\_logsigmoid | torch.log(torch.sigmoid(x)) | 数值稳定 |
| m4\_reward\_grad | 返回的 reward 没有 detach | 返回值约定 |
| m5\_mean\_not\_sum | 对 token 求平均 | T1、双重循环 |

**可视化**

- shift\_mask\_grid：一条样本的 token 网格，prompt 与回答分色。箭头表示"位置 t 的 logits → 标签 t+1"，被 mask 的部分加阴影；悬停显示每个 token 对序列 log-prob 的贡献。数据来自一个小例子，预计算。
- loss\_vs\_margin：L(h) 曲线，β 与 ε 用滑块调节。同时画出梯度权重 β·σ(−βh)，说明 DPO 主要在排错序的样本上用力。
- t3\_reward\_scatter：训练过程中隐式奖励与真实 r\* 的散点，用滑块切换训练步。

## R4 · GRPO（实现，M）

Paper：Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. arXiv:2402.03300（GRPO 一节）。KL 估计器：Schulman. *Approximating KL Divergence.* 博客，2020。

**讲解要点**

- 问题：PPO 需要一个和策略差不多大的 value 模型来算 advantage，显存与训练成本翻倍。
- 关键：对同一个 prompt 采 G 个回答，用组内奖励的均值和标准差做归一化，代替 value 模型。
- KL 不加在奖励里，而是直接加进 loss，用 k3 估计器，它无偏且逐 token 非负。
- 聚合方式：原文先对每条回答的 token 求平均，再对回答求平均。R5 Dr. GRPO 质疑的正是这一点，讲解里要埋下伏笔。
- 论文没写的（放进 notes.md）：std 用有偏还是无偏估计；组内奖励全相同时 std=0；padding mask。

**签名与约定**

```python
def grpo_advantages(
    rewards: Tensor,       # [B*G]，同一 prompt 的 G 个回答相邻排列
    group_size: int,       # G=1 时抛 ValueError
    scale_std: bool = True,
    eps: float = 1e-4,
) -> Tensor:               # [B*G]：(r - mean_g) / (std_g + eps)

def grpo_loss(
    logp: Tensor,          # [B*G, L] 当前策略逐 token log-prob（有梯度）
    old_logp: Tensor,      # [B*G, L] 采样时策略（常数）
    ref_logp: Tensor,      # [B*G, L] 参考策略（常数）
    advantages: Tensor,    # [B*G]
    mask: Tensor,          # [B*G, L]，1 为回答中的有效 token
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[Tensor, dict]:  # loss 标量；stats 含 kl_mean、clipfrac、ratio_mean
```

```latex
\rho_{i,t} = e^{\log\pi_\theta - \log\pi_{\text{old}}}, \qquad
k3_{i,t} = e^{\log\pi_{\text{ref}} - \log\pi_\theta} - (\log\pi_{\text{ref}} - \log\pi_\theta) - 1
```

```latex
L = -\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat A_i\right) - \beta\, k3_{i,t}\right]
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | shape 正确；G=1 抛 ValueError | 基本契约 |
| T1 | 两个函数都与参考解一致 | 约定已钉死 |
| T2 | advantage：每组和为 0；组内平移和正缩放不变；全相同奖励得到全 0 且无 NaN；组与组互不影响；scale\_std=False 时等于 r − mean | 组内归一化的定义 |
| T2 | logp = old\_logp 且 β=0 时，每个有效 token 的梯度为 −Â\_i / (\|o\_i\|·N)，clipfrac 为 0 | on-policy 退化 |
| T2 | logp = ref\_logp 时 kl\_mean = 0；advantage 全为 0 时 loss ≥ 0 | k3 非负 |
| T2 | 修改 mask=0 处的 logp、old\_logp、ref\_logp，loss 不变；这些位置的梯度为 0 | padding 隔离 |
| T2 | old\_logp、ref\_logp 不产生梯度 | 常数 |
| T2 | 把一条回答的有效 token 原样复制一倍，长度翻倍、逐 token 数值不变，loss 不变 | 先对 token 平均的聚合方式 |
| T3 | TablePolicyLM(max\_len=8, vocab=5，含 EOS)，目标串 "abc"，奖励为前 3 位的匹配比例。8 组 × G=8，β=0.04，参考策略为初始策略。150 步内平均奖励从约 0.2 升到 ≥ 0.9 | 方法能学会 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_std\_over\_batch | std 按整个 batch 而不是组内计算 | 组与组互不影响 |
| m2\_mask\_ignored | padding 也参与平均 | padding 隔离 |
| m3\_k1\_kl | KL 用 log π − log π\_ref（k1），可为负 | k3 非负 |
| m4\_old\_logp\_grad | old\_logp 没有 detach | 常数无梯度 |
| m5\_token\_mean | 对整个 batch 的有效 token 统一平均 | 聚合方式 |

m5 其实是 DAPO 的合法约定。notes.md 要讲清楚"mutant 不等于错，只是与本题约定不同"，这正是变体题存在的理由。

**可视化**

- group\_advantage：G=8 个回答卡片，各带奖励，下方画 advantage 柱状图。可拖动修改单个奖励；开关 std 归一化；用滑块整体缩放奖励，直观展示不变性。
- token\_weight\_heatmap：4 条长度分别为 3、6、12、24 的回答，advantage 符号相同，热图画每个 token 的梯度权重。可切换 GRPO（按回答平均）、DAPO（按 token 平均）、Dr. GRPO（常数归一），R5 复用这个组件。
- t3\_reward\_curve：平均奖励与平均长度随训练步的曲线，数据预计算。

## R5 · Dr. GRPO（变体题，前置 R4，M）

Paper：Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. arXiv:2503.20783。

**讲解要点**

- 论文指出 GRPO 有两个偏置。
- 长度偏置：每条回答都除以 |o\_i|。advantage 为负的回答越长，每个 token 受到的惩罚越小，所以错误回答倾向于变长；advantage 为正的回答则越短，每个 token 得到的奖励越大。
- 难度偏置：除以组内 std 后，几乎全对或几乎全错的题 std 很小，权重被放大。
- 修正：去掉 1/|o\_i|，改用常数归一（生成长度上限 L\_max）；去掉 std 归一。
- 变体题的教学目的：读出 follow-up 改了哪两处，再用测试证明改动确实产生了论文宣称的效果。

**签名与约定**

```python
def drgrpo_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """[B*G] -> [B*G]：r - mean_g，不除 std。"""

def drgrpo_loss(
    logp: Tensor, old_logp: Tensor, ref_logp: Tensor,  # 同 R4
    advantages: Tensor, mask: Tensor,
    max_len: int,            # 常数归一用的生成长度上限
    clip_eps: float = 0.2,
    beta: float = 0.04,      # 与 R4 保持一致，只隔离两处改动
) -> tuple[Tensor, dict]:
```

```latex
L = -\frac{1}{N \cdot L_{\max}}\sum_{i=1}^{N}\sum_{t=1}^{|o_i|}\left[\min\left(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t})\,\hat A_i\right) - \beta\, k3_{i,t}\right], \qquad \hat A_i = r_i - \mathrm{mean}_g(r)
```

starter.py 直接 import R4 参考解里的 ratio、clip、k3 辅助函数，做题者只需要改动 advantage 和聚合。

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 / T1 | shape；与参考解一致 | 基本契约 |
| T2 | 长度偏置：两条 advantage 相同且为负、长度分别为 5 和 20、逐 token 数值相同的回答，长短两条的逐 token 梯度之比为 1。测试同时用 R4 参考解算出 5/20 作对照，写进失败信息 | 论文的第一处修正 |
| T2 | 长度上限是常数：batch 里另一条回答 advantage 为 0 且 logp = ref，把它的长度从 10 换成 30，目标回答的梯度不变 | 归一化不依赖 batch 组成 |
| T2 | advantage：每组和为 0；奖励乘 k 时 advantage 也乘 k（与 R4 的缩放不变性相反） | 论文的第二处修正 |
| T3 | 仅展示、不判分：在带 EOS 的 toy 环境里比较训练过程中错误回答的平均长度，GRPO 与 Dr. GRPO 各画一条曲线 | 论文的经验结论 |

T3 先在参考解上用 5 个种子验证效应同向且稳定，确认后再升级为判分测试。

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_still\_seq\_mean | 仍然除以 \|o\_i\| | 长度偏置 |
| m2\_still\_std | 仍然除以 std | advantage 缩放 |
| m3\_batch\_max\_len | 用当前 batch 的最大长度代替常数 max\_len | 长度上限是常数 |

**可视化**

- 复用 R4 的 token\_weight\_heatmap，GRPO 与 Dr. GRPO 左右并排，用滑块调节回答长度。
- difficulty\_bias：G=8、奖励为 0/1 时，横轴为组内正确率 p，纵轴为正确回答的 |Â|。GRPO 为 √((1−p)/p)，Dr. GRPO 为 1−p，由前端直接计算，直观展示 std 归一会放大极端题的权重。
- t3\_length\_curves：错误回答长度随训练步变化的两条曲线，标注"展示用"。

# Diffusion 五题

## D1 · 噪声调度与前向加噪（实现，E）

Paper：Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239。Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. arXiv:2102.09672。

**讲解要点**

- 前向过程逐步加噪：q(x\_t | x\_{t−1}) = N(√(1−β\_t)·x\_{t−1}, β\_t·I)。它有闭式解 q(x\_t | x₀)，所以训练时可以一步采到任意 t。
- 方差保持：数据方差为 1 时，x\_t 的方差在所有 t 上都为 1。
- Improved DDPM 指出线性调度的问题：末段很多步几乎已是纯噪声，浪费了步数。cosine 调度让 ᾱ 在中段近似线性下降。
- 论文没写的（放进 notes.md）：下标从 0 还是从 1 开始；cosine 的 β 要截断在 0.999；线性调度在 T=1000 时 ᾱ\_T 约为 4e-5，终端 SNR 不为 0。最后这点是 Lin et al. 2024 的出发点，留给后续题。

**签名与约定**

```python
def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:  # [T] float64

def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> Tensor:  # [T] float64

def make_schedule(betas: Tensor) -> dict[str, Tensor]:
    """返回 alphas、alpha_bars、alpha_bars_prev（首项为 1）、sqrt_alpha_bars、
    sqrt_one_minus_alpha_bars、log_snr，全部为 [T] float64。"""

def q_sample(x0: Tensor, t: Tensor, noise: Tensor, alpha_bars: Tensor) -> Tensor:
    """x0 [B, ...]，t [B] long，每个样本用自己的 t；输出 dtype 与 x0 相同。"""
```

下标约定：数组下标 i = 0…T−1 对应论文的 t = i+1。

```latex
\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s), \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \mathrm{SNR}_t = \frac{\bar\alpha_t}{1-\bar\alpha_t}
```

```latex
f(\tau) = \cos^2\!\left(\frac{\tau + s}{1+s}\cdot\frac{\pi}{2}\right), \qquad
\beta_i = \min\!\left(1 - \frac{f((i+1)/T)}{f(i/T)},\ \beta_{\max}\right)
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | 调度张量为 \[T\] float64；q\_sample 保持 x0 的 dtype；x0 为 \[B, D\] 或 \[B, C, H, W\] 都能运行 | 基本契约 |
| T1 | T=1000 时两种调度都与参考解一致 | 约定已钉死 |
| T2 | ᾱ 严格递减且落在 (0, 1)；log-SNR 严格递减 | 加噪单调 |
| T2 | 线性调度首尾 β 分别为 1e-4 与 0.02；ᾱ\_T 约为 4e-5（相对误差 5% 内） | 终端 SNR 不为 0 |
| T2 | cosine：max β ≤ 0.999；下标 T/2 处 ᾱ 约为 0.5（±0.01） | 截断；中段近似线性 |
| T2 | noise=0 时 q\_sample 返回 √ᾱ·x0；x0=0 时返回 √(1−ᾱ)·noise | 闭式解的两部分 |
| T2 | batch 内每行用不同的 t，逐行检查 | 每个样本用自己的 t |
| T2 | x0、noise 都服从 N(0,1) 时，蒙特卡洛 10 万点下所有 t 的 Var(x\_t) 约为 1（±2%） | 方差保持 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_offbyone | 把 alpha\_bars\_prev 当作 alpha\_bars 返回 | T1、线性调度端点 |
| m2\_no\_sqrt | x\_t = ᾱ·x0 + (1−ᾱ)·ε | 方差保持 |
| m3\_cosine\_no\_clip | cosine 的 β 没有截断 | max β ≤ 0.999 |
| m4\_cumsum | ᾱ = 1 − cumsum(β) | T1、终端 ᾱ |
| m5\_shared\_t | 整个 batch 都用 t\[0\] | 每个样本用自己的 t |

**可视化**

- schedule\_curves：两个面板，分别画 ᾱ\_t 与 log-SNR 随 t/T 的变化，线性与 cosine 两条曲线。T（50–1000）与 s 用滑块调节，悬停读数，并标出终端 SNR。
- forward\_cloud：2000 个点的 8-高斯点云，两种调度左右并排；滑块拖动 t，由前端带种子的随机数实时加噪。让读者比较两种调度下结构消失的时刻。

## D2 · x₀ / ε / v 参数化互转（实现，E）

Paper：Ho et al. 2020（ε 参数化）。Salimans & Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022. arXiv:2202.00512（v 参数化）。

**讲解要点**

- 三种预测目标描述的是同一个量：给定 x\_t，知道 x₀ 与 ε 中的任意一个，就能推出另一个。
- 记 α\_t = √ᾱ\_t、σ\_t = √(1−ᾱ\_t)，则 α² + σ² = 1。令 α = cos φ、σ = sin φ，x\_t 就在 (x₀, ε) 平面的单位四分之一圆上转动，v = dx\_t/dφ 是它的切向量。
- ε → x₀ 要除以 α，t 接近 T 时会爆炸；x₀ → ε 要除以 σ，t 接近 0 时会爆炸。v 的互转不需要除法，两端都稳定，这是 progressive distillation 引入 v 的原因之一。
- 误差放大：同一个预测换算到不同参数化后，误差满足 ‖v̂ − v‖² = ‖x̂₀ − x₀‖² / σ² = ‖ε̂ − ε‖² / α²。也就是说，v-loss 等于 (1 + SNR) 加权的 x₀-loss。
- 延伸：rectified flow 的速度 x₁ − x₀ 是另一种插值路径下的"v"，在 D5 里会再次出现。

**签名与约定**

```python
def eps_to_x0(x_t, eps, alpha, sigma) -> Tensor   # (x_t - sigma*eps) / alpha
def x0_to_eps(x_t, x0, alpha, sigma) -> Tensor    # (x_t - alpha*x0) / sigma
def make_v(x0, eps, alpha, sigma) -> Tensor        # alpha*eps - sigma*x0
def v_to_x0(x_t, v, alpha, sigma) -> Tensor        # alpha*x_t - sigma*v
def v_to_eps(x_t, v, alpha, sigma) -> Tensor       # sigma*x_t + alpha*v
def convert(pred, src: str, dst: str, x_t, alpha, sigma) -> Tensor  # src、dst ∈ {"x0", "eps", "v"}
```

alpha、sigma 可广播到 x\_t 的形状（如 \[B, 1, 1, 1\]），由调用者保证 α² + σ² = 1。需要除法的转换对除数做 clamp(min=1e-8)，不抛错。

```latex
x_t = \alpha x_0 + \sigma\epsilon, \qquad v = \alpha\epsilon - \sigma x_0, \qquad x_0 = \alpha x_t - \sigma v, \qquad \epsilon = \sigma x_t + \alpha v
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | alpha 为 \[B, 1\] 时可广播；输出 shape 与 x\_t 相同 | 基本契约 |
| T1 | 与参考解一致 | 约定已钉死 |
| T2 | 6 种两两往返在 float64 下误差 < 1e-10，α ∈ \[0.01, 0.99\] | 互为逆变换 |
| T2 | 取 α = cos φ、σ = sin φ，x\_t 对 φ 的有限差分等于 make\_v（误差 1e-6） | v 是切向量 |
| T2 | α=0 或 σ=0 时，v 相关转换精确且有限；需要除法的转换不出 inf / NaN | 端点稳定性 |
| T2 | 对任一参数化下的随机错误预测，换算后的三种误差满足上面的恒等式（误差 1e-8） | loss 权重关系 |
| T2 | convert 直接换算的结果与"先转 x₀ 再转目标"一致 | 一致性 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_v\_sign | v = σ·x₀ − α·ε（符号反了） | 切向量、往返 |
| m2\_v\_to\_x0\_swapped | x₀ = σ·x\_t − α·v | 往返 |
| m3\_no\_clamp | 除法没有 clamp | 端点稳定性 |
| m4\_alpha\_squared | eps\_to\_x0 除以 α²（把 √ᾱ 与 ᾱ 混用） | 往返、T1 |

**可视化**

- param\_circle：横轴为 x₀ 方向、纵轴为 ε 方向。高维下两者近似正交，所以这张几何图是准确的。x\_t 是四分之一圆上的点，v 是切向量；拖动 φ（或 t），同时显示 x\_t 在两个轴上的分解。
- error\_amplification：在 cosine 调度下，用对数纵轴画三条放大系数曲线随 t 的变化：α/σ（x₀ 误差 → ε 误差）、1/σ（x₀ → v）、1/α（ε → x₀），并标出各自爆炸的区段。

## D3 · DDPM 后验与祖先采样（实现，M）

Paper：Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239。

**讲解要点**

- 反向过程要学 p(x\_{t−1} | x\_t)。训练目标来自可解析的后验 q(x\_{t−1} | x\_t, x₀)，它是一个高斯。
- 采样时先用预测的 ε 恢复 x̂₀，再代入后验均值。方差可取 β̃\_t（后验方差）或 β\_t，论文报告两者效果接近。
- 本题不训练：用高斯混合的解析 ε\* 代替网络。采样误差只来自实现与离散化，可以直接和真实分布比较。
- 论文没写的（放进 notes.md）：最后一步不加噪；β̃ 在第一步为 0，log 方差需要特殊处理（核对 improved-diffusion 的做法）；图像任务会把 x̂₀ clip 到 \[−1, 1\]，本题的 2D 数据默认不 clip。

**签名与约定**

```python
def q_posterior(x0, x_t, t, betas, alpha_bars, alpha_bars_prev) -> tuple[Tensor, Tensor]:
    """返回 (mean, var)，var 按 x_t 广播；t 为 [B] long。"""

def p_sample_step(eps_pred, x_t, t, betas, alpha_bars, alpha_bars_prev, noise,
                  var_type: str = "posterior") -> Tensor:
    """var_type ∈ {"posterior", "beta"}；t == 0 的样本不加噪。"""

def ddpm_sample(eps_model, shape, betas, generator, var_type="posterior", return_traj=False):
    """eps_model(x_t, t) -> eps；从 x_T ~ N(0, I) 开始，t 从 T-1 降到 0。"""
```

```latex
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
```

```latex
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad
x_{t-1} = \tilde\mu_t(x_t, \hat x_0) + \sigma_t z, \qquad \sigma_t^2 \in \{\tilde\beta_t,\ \beta_t\}
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | shape 正确；batch 内每行可用不同的 t | 基本契约 |
| T1 | 固定 noise 下，q\_posterior 与 p\_sample\_step 与参考解一致 | 约定已钉死 |
| T2 | q\_posterior 与测试里独立推导的结果一致：把 q(x\_{t−1} \| x₀) 与 q(x\_t \| x\_{t−1}) 两个高斯相乘，用精度相加的方式算均值和方差 | 后验的定义 |
| T2 | 所有 t 上后验方差 ≤ β\_t；下标 0 处方差为 0、均值等于 x₀ | 边界 |
| T2 | t=0 时传入两组不同的 noise，输出相同 | 最后一步不加噪 |
| T2 | 单点数据（std=1e-3）用 oracle ε\* 时，x̂₀ 约等于该点，且均值等于 μ̃(x\_t, x₀) | 预测与后验的衔接 |
| T2 | var\_type="beta" 时，有噪与无噪两次输出之差等于 √β\_t·z | 方差选项 |
| T3 | 8-高斯 GMM（半径 2，std 0.15），线性调度 T=1000，oracle ε\*，采 2000 个点。与 2000 个真实样本的 MMD² 低于阈值；每个模式占比 12.5% ± 3%，离群点 < 5%。两种 var\_type 都要通过 | 采样器正确 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_coef\_swapped | x₀ 与 x\_t 的系数写反 | 独立推导、T3 |
| m2\_noise\_at\_last | t=0 时仍加噪 | 最后一步不加噪 |
| m3\_var\_index | β̃ 的分子用 ᾱ\_t 而不是 ᾱ\_{t−1} | 独立推导 |
| m4\_x0\_no\_sqrt | x̂₀ 公式里的 ᾱ 没有开方 | oracle 衔接、T3 |
| m5\_ascending | t 从 0 升到 T−1 | T3 |

**可视化**

- reverse\_particles：500 个粒子从 N(0, I) 流向 8 个模式的动画，取 60 帧。背景画当前 t 的真实 p\_t 等高线，由前端按高斯混合公式实时计算。支持播放、暂停、拖动 t，并可切换 var\_type。
- variance\_compare：对数纵轴上 β\_t 与 β̃\_t 随 t 的变化，两者主要在小 t 处不同。
- 报告叠加：读取 report.json 中做题者的采样点，与真实样本叠在同一张图上，并显示 MMD 相对阈值的位置。

## D4 · DDIM（实现，M）

Paper：Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502。

**讲解要点**

- 关键观察：DDPM 的训练目标只依赖边缘分布 q(x\_t | x₀)。因此可以换一族非马尔可夫的前向过程，边缘不变，训练好的 ε 网络直接复用。
- η 控制随机性：η=1 恢复 DDPM 的后验方差；η=0 是确定性采样，对应概率流 ODE 的一种离散化。
- 跳步：更新式只依赖相邻两个时间点的 ᾱ，所以可以只在一个子序列上采样，比如 50 步。
- 确定性带来可反演性：η=0 时把更新式倒着跑，能把 x₀ 映射到 x\_T，这是很多编辑方法的基础。
- 论文没写的（放进 notes.md）：子序列怎么取（leading、trailing、linspace，Lin et al. 2024 讨论过差异）；最后一步要把 ᾱ\_prev 设为 1；步数少时反演误差很大。

**签名与约定**

```python
def ddim_step(x_t, eps_pred, ab_t, ab_prev, eta: float, noise) -> Tensor:
    """ab_t、ab_prev 可广播到 x_t；最后一步 ab_prev = 1。"""

def ddim_timesteps(T: int, num_steps: int) -> Tensor:
    """trailing 间隔：t_k = T-1 - k*(T // num_steps)，k = 0..num_steps-1。T=1000、n=50 时为 [999, 979, ..., 19]。"""

def ddim_sample(eps_model, x_T, alpha_bars, num_steps, eta=0.0, generator=None, return_traj=False): ...

def ddim_invert(eps_model, x0, alpha_bars, num_steps) -> Tensor:
    """η=0，按时间升序把 x0 映射到 x_T。"""
```

```latex
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad
\sigma = \eta\sqrt{\frac{1-\bar\alpha_{\text{prev}}}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{\text{prev}}}}
```

```latex
x_{\text{prev}} = \sqrt{\bar\alpha_{\text{prev}}}\,\hat x_0 + \sqrt{1-\bar\alpha_{\text{prev}}-\sigma^2}\;\hat\epsilon + \sigma z
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | shape 正确 | 基本契约 |
| T1 | 固定 noise，η 取 0、0.5、1 时 ddim\_step 与参考解一致 | 约定已钉死 |
| T2 | η=1 且相邻两步时，σ² 等于 DDPM 的 β̃\_t（其中 β\_t = 1 − ab\_t/ab\_prev） | η=1 恢复 DDPM |
| T2 | η=0 时输出与 noise 无关 | 确定性 |
| T2 | 单点数据（GMM 的 std 取 0）+ oracle ε\*，η=0 时 1 步和 50 步采样都精确落到该点（误差 1e-6） | x̂₀ 精确时一步到位 |
| T2 | ddim\_timesteps 长度为 n、严格降序、首项为 T−1、最小值 ≥ 0 | 时间表约定 |
| T2 | GMM oracle 下做反演再采样，往返误差随步数（10 → 100）单调下降，100 步时低于阈值 | 可反演 |
| T2 | 从同一个 x\_T 出发，η=0 下 50 步与 500 步的结果平均距离低于阈值 | ODE 离散化收敛 |
| T3 | GMM oracle，50 步、η=0：MMD² 低于阈值，模式覆盖达标。另记录同为 50 步的 DDPM（η=1）MMD 作对照，仅展示 | 少步采样质量 |

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_sigma\_one\_factor | σ 少了第二个开方因子 | η=1 恢复 DDPM |
| m2\_dir\_no\_sigma | 方向项用 √(1−ᾱ\_prev)，没有减 σ² | T1（η > 0） |
| m3\_no\_final\_jump | 停在最后一个子序列时间点，没有跳到 ᾱ = 1 | 单点数据一步到位 |
| m4\_x0\_uses\_prev | x̂₀ 用 ᾱ\_prev 计算 | T1、单点数据 |
| m5\_invert\_order | 反演时降序遍历，或用错了时间点的 ε | 反演往返 |

**可视化**

- eta\_steps\_grid：3 × 4 网格（η ∈ {0, 0.5, 1} × 步数 ∈ {10, 20, 50, 100}），每格画最终采样点并标出 MMD；点击任一格播放该设置下的轨迹（200 个粒子，预计算）。
- trajectories：从同一组 x\_T 出发的 30 个粒子，η=0 的平滑轨迹与 η=1 的抖动轨迹叠在一起。
- inversion\_demo：点击选中一个数据点，显示它反演得到的 x\_T 和重新生成的点，以及往返误差随步数变化的曲线。

## D5 · Flow Matching（实现，M）

Paper：Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747。Liu, Gong, Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. arXiv:2209.03003。

**讲解要点**

- 目标：学一个速度场 v(x, t)，让 ODE 把噪声分布推到数据分布。
- 关键技巧（conditional flow matching）：边缘速度场算不出来，但每一对 (x₀, x₁) 的条件速度可以算。回归条件速度的 loss 与回归边缘速度的 loss 梯度相同。
- 线性插值路径下，条件速度就是常数 x₁ − x₀。因此单点数据的路径是直线，Euler 一步就精确。
- CFM loss 的最小值不为 0：同一个 x\_t 可能来自多对 (x₀, x₁)，条件目标有不可约的方差。新手常把"loss 降不到 0"当成 bug，讲解里要单独说明。
- 时间方向是本题最大的坑：Lipman 把噪声放在 t=0，SD3 等工作把噪声放在 t=1。训练与采样方向不一致时，loss 照常下降，采样却完全错误。T3 专门抓这个问题。
- 论文没写的（放进 notes.md）：t 的采样分布（均匀或 SD3 的 logit-normal）；oracle 速度场在 t=1 处奇异，所以 Euler 只在 t < 1 处求值。

**签名与约定**

```python
def sample_t(batch: int, generator, dist: str = "uniform", m: float = 0.0, s: float = 1.0) -> Tensor:
    """[B]，取值在开区间 (0, 1)；logit_normal 为 sigmoid(N(m, s²))。"""

def interpolate(x0, x1, t) -> Tensor:   # (1-t)*x0 + t*x1；t 为 [B]，自动广播
def cfm_target(x0, x1) -> Tensor:       # x1 - x0
def cfm_loss(model, x1, x0, t) -> Tensor:  # mean ||model(x_t, t) - (x1 - x0)||²；x0 是噪声
def euler_sample(model, x0, num_steps, return_traj=False) -> Tensor:
    """t_k = k / num_steps，k = 0..num_steps-1，从 t=0 积分到 t=1。"""
```

模型接口：`model(x [B, D], t [B]) -> [B, D]`。约定：x₀ 为噪声、位于 t=0，x₁ 为数据、位于 t=1。

```latex
x_t = (1-t)\,x_0 + t\,x_1, \qquad
\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_1}\left\|v_\theta(x_t, t) - (x_1 - x_0)\right\|^2, \qquad
x_{k+1} = x_k + \Delta t\; v_\theta(x_k, t_k)
```

**测试**

| 层 | 测试 | 为什么成立 |
| --- | --- | --- |
| T0 | shape 正确；t 为 \[B\] 时能广播到 \[B, D\] | 基本契约 |
| T1 | 用一个固定的确定性线性模型，interpolate、cfm\_loss、euler\_sample 都与参考解一致 | 约定已钉死 |
| T2 | 单点数据 μ + point\_velocity\_oracle：随机 t 下 cfm\_loss 为 0（atol 1e-10） | 条件速度与最优速度一致 |
| T2 | 单点数据 + oracle：Euler 1 步就精确落到 μ | 直线路径 |
| T2 | 数据为 N(μ, s²I) + gaussian\_velocity\_oracle：Euler 200 步、2 万个点，样本均值约为 μ，标准差约为 s（误差 3%） | 采样方向与时间网格正确 |
| T2 | 包装模型记录被调用的 t：最大值 < 1 | 不在奇点处求值 |
| T2 | interpolate 在 t=0 返回 x₀、t=1 返回 x₁；sample\_t 的取值不含 0 和 1，logit-normal 的 logit 均值与标准差约为 m 与 s | 端点与时间采样 |
| T3 | common 里的 TimeMLP 在 eight\_gaussians 上训练：Adam lr 1e-3，batch 256，约 2000 步，均匀采 t。用 Euler 50 步采 2000 个点，MMD² 低于阈值，模式覆盖达标 | 能学会、方向一致 |

T3 若在 CI 上超过 10 s，降到 1000 步并重新标定阈值，结果写入 problem.yaml。

**Mutant**

| id | bug | 应被哪个测试抓到 |
| --- | --- | --- |
| m1\_target\_reversed | 目标写成 x₀ − x₁ | oracle 零 loss、T3 |
| m2\_sample\_reversed | 训练按 Lipman 约定，采样却从 t=1 积分到 0 | 高斯 oracle、T3 |
| m3\_interp\_swapped | x\_t = t·x₀ + (1−t)·x₁，目标仍是 x₁ − x₀ | oracle 零 loss |
| m4\_eval\_at\_one | Euler 网格取 t\_k = (k+1)/n，在 t=1 处求值 | 不在奇点处求值 |

**可视化**

- flow\_particles：1000 个粒子从高斯流向 8 个模式的动画，取 60 帧，叠加当前 t 下 20 × 20 网格的速度场箭头。支持播放、暂停、拖动 t。轨迹为预计算。
- steps\_compare：Euler 分别取 1、2、4、8、50 步时的最终样本并排展示；旁边对比单点数据（直线路径）与高斯数据（路径弯曲）的轨迹。
- convention\_bug：开关"采样方向反了"，左右对比正确结果与 m2 的结果，直观展示时间方向这个坑。
- loss\_curve：T3 的训练 loss 曲线，标出它收敛到非零值，并配一句说明原因。

## 可视化与网页

可视化分两类。一类是概念可视化，嵌在讲解里，不依赖做题者的代码。另一类是结果可视化：读取 report.json，把做题者自己的输出画出来，比如采样点和真实分布叠在一起。后者是本项目与现有题库最大的体验差异。

### 技术栈

- Vite + React + TypeScript，静态构建，部署到 GitHub Pages。
- Markdown 用 react-markdown + remark-math + rehype-katex 渲染；代码高亮用 Shiki。
- 曲线、散点、柱状图用 Observable Plot；粒子动画与向量场用 Canvas 2D，目标 60 fps，粒子数 ≤ 1000。
- 统一的 Slider、Toggle、PlayBar 控件。
- 数据：每题一个 `web/public/data/<problem>.json`，由 `paperleet vizdata` 生成，单题 ≤ 500 KB。能在前端用 TS 重算的（GAE、clip 曲线、调度、参数化几何）不预计算。

### 页面

1. 首页：RL 与 Diffusion 两条路径。每题一行，显示标题、paper、难度、题型和本地完成状态（存 localStorage）。
2. 题目页：顶部是 paper 卡片，包括标题、作者、会议、arXiv 链接和一行核心公式。主区标签依次为：题目、讲解、测试、坑点（通过后解锁）、参考解（剧透折叠）。可视化嵌在讲解中；宽屏时右侧固定一个"实验台"面板，显示当前题的主可视化。
3. 我的结果：拖入 report.json 后，显示 T0–T3 各层通过进度；每个失败测试显示 why 与 message；T3 artifacts 画成图，包括采样点与真实分布叠加、MMD 相对阈值的位置、训练曲线。

### 可视化组件清单

| 组件 | 题 | 数据来源 |
| --- | --- | --- |
| gae\_timeline | R1 | 前端重算 |
| bias\_variance\_vs\_lambda | R1 | 预计算 |
| clip\_objective | R2 | 前端重算 |
| trust\_region | R2 | 预计算 |
| shift\_mask\_grid | R3 | 预计算 |
| loss\_vs\_margin | R3 | 前端重算 |
| t3\_reward\_scatter | R3 | 预计算 |
| group\_advantage | R4 | 前端重算 |
| token\_weight\_heatmap | R4、R5 | 前端重算 |
| t3\_reward\_curve / t3\_length\_curves | R4 / R5 | 预计算 |
| difficulty\_bias | R5 | 前端重算 |
| schedule\_curves、forward\_cloud | D1 | 前端重算 |
| param\_circle、error\_amplification | D2 | 前端重算 |
| reverse\_particles | D3 | 预计算轨迹 + 前端算等高线 |
| variance\_compare | D3 | 前端重算 |
| eta\_steps\_grid、trajectories、inversion\_demo | D4 | 预计算 |
| flow\_particles、steps\_compare、convention\_bug、loss\_curve | D5 | 预计算 |
| report\_overlay | 所有带 T3 的题 | report.json |

### 设计要求

- 每个可视化配一句说明，写清"拖动什么、看什么"。
- 颜色语义固定：参考解与真实分布用中性灰，做题者的结果用强调色，错误用红色；亮色与暗色主题都要可读。
- 公式与图一一对应：讲解公式里的符号在图中用相同颜色标注，比如 δ\_t 与 Â\_t。
- 窄屏自适应：粒子动画在手机上降到 300 个粒子。
- 所有滑块可用键盘操作，每张图有文字替代描述。

## 里程碑与验收标准

先用一道最简单的题（D2）把框架从头到尾打通，再批量出题，最后做网页。这样框架的问题在第一道题上就会暴露，不会扩散到 10 道题。

1. M1 框架：仓库、common、judge CLI、report.json。用 D2 作模板题，走通 starter → 参考解 → 测试 → mutant → vizdata 全流程。验收：`paperleet ci` 对 D2 通过，4 个 mutant 全部被抓。
2. M2 Diffusion 其余 4 题：D1、D3、D4、D5，包括 T3 阈值标定。
3. M3 RL 5 题：R1–R5。
4. M4 网页：首页、题目页、全部可视化组件、报告视图。
5. M5 CI 与文档：GitHub Actions 运行 `paperleet ci` 并构建网页；CONTRIBUTING.md 写出题流程，直接引用本页的题目包规范。

### 验收清单

- [ ] 10 道题的参考解全部通过 T0–T3
- [ ] 每个 mutant 至少被一个测试抓到（测试强度 100%），且与参考解的 diff ≤ 10 行
- [ ] 全部测试重复 3 次，结果一致
- [ ] 每个 T3 在 2 核 CPU 上 ≤ 10 s；整套 `paperleet ci` ≤ 120 s
- [ ] 每题 README 覆盖模板的 7 节，至少包含 1 个 LaTeX 公式和 1 个交互可视化
- [ ] 每条失败信息都给出位置、期望值和实际值
- [ ] 网页构建通过，单题数据 ≤ 500 KB；拖入 report.json 能正确渲染
- [ ] 盲测：找一位没看过参考解的人，或新开一个只拿到 starter 和 README 的 agent，完成 D2 和 R4，记录用时与卡点

### 试点结束后要回答的问题

- 出一道题平均要多久？四层测试里哪一层最费时间？
- mutant 能否代表新手真实会犯的错？盲测者写出的错误有没有逃过测试？
- 盲测中哪些可视化被用到了，哪些没人看？
- T3 阈值在不同机器上是否稳定？

## 实现前需核对的点

本页的公式与约定已经足够实现。下面几项凭记忆写成，或依赖外部代码库的版本，code agent 动手前要先对照原文或源码核实，结果写进对应题的 problem.yaml 或 notes.md。

| 核对项 | 影响 | 去哪里核对 |
| --- | --- | --- |
| 各 paper 的公式编号、作者与会议 | README 与 problem.yaml 的 anchors | arXiv 原文 |
| GRPO 组内 std 用有偏还是无偏估计 | R4 的 T1 | TRL GRPOTrainer 源码，记录所用版本 |
| PPO 的 advantage 归一化与 value clipping 默认值 | R2 讲解里的对照表 | CleanRL ppo.py；Huang et al. 2022 博客 |
| DPO 的 label\_smoothing 对应的保守 DPO 出处 | R3 讲解 | TRL DPOTrainer 文档 |
| Dr. GRPO 常数归一的具体取值，以及是否保留 KL 项 | R5 讲解；β 的默认值 | 原文 2503.20783 及其开源代码 |
| 第一步 β̃ 为 0 时 log 方差的处理 | D3 notes | openai/improved-diffusion 的 gaussian\_diffusion.py |
| DDIM 时间子序列与 diffusers timestep\_spacing 各选项的对应 | D4 讲解里的对照表 | diffusers DDIMScheduler 源码 |
| Lipman、rectified flow、SD3 三者的时间方向约定 | D5 讲解里的对照表 | 三篇原文 |
| 两篇博客的稳定链接 | R2、R4 的 paper 卡片 | Huang et al. 2022（ICLR Blog Track）；Schulman 2020（joschu.net） |
| 所有 T3 阈值 | 判分稳定性 | 在 CI 机器上按"参考解实测 × 1.5"标定 |
