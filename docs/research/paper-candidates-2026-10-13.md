# Oral / Spotlight paper candidates · 2026-10-13

This evidence pool covers 35 papers from official ICLR, ICML, NeurIPS, and CVPR programs. Ten were selected for this implementation batch; OPSD and SDPO add two papers from the earlier distillation shortlist. Selection considers a distinctive implementable mechanism, an independent numerical oracle, and a small CPU experiment. Presentation status is separate from a best-paper award.

Program decisions, paper/code links, proposed modules, toy provenance, and harness notes are retained in the [structured evidence](paper-candidates-2026-10-13.json). A `/poster/` program URL can still correspond to an Oral decision. ICML 2026 Chebyshev Policies and MuonSSM are listed as Spotlight with an oral talk; do not promote that label to an award. NeurIPS 2026 is excluded because its program was not verified at this cutoff.

| Paper | Official selection | Status | Focus |
| --- | --- | --- | --- |
| [DistiLLM-2](https://arxiv.org/html/2503.07067) | [ICML 2025 · Oral](https://icml.cc/virtual/2025/poster/43884) | implemented | 教师回答上的 skew KL 与学生回答上的 skew reverse KL |
| [ADPA](https://arxiv.org/html/2502.17927) | [ICLR 2025 · Spotlight](https://iclr.cc/virtual/2025/poster/27759) | candidate | DPO 教师与 reference 教师的对数概率比优势 |
| [TROLL](https://arxiv.org/html/2510.03817) | [ICLR 2026 · Oral](https://iclr.cc/virtual/2026/poster/10008998) | candidate | 离散概率分布的 KL 信赖域投影 |
| [PKPO · Pass@K Policy Optimization](https://arxiv.org/html/2505.15201) | [NeurIPS 2025 · Spotlight](https://neurips.cc/virtual/2025/poster/117623) | implemented | 稳定的 pass@k 估计与样本奖励变换 |
| [REPA](https://arxiv.org/html/2410.06940v1) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/30467) | implemented | Patch-wise representation alignment loss |
| [Shortcut Models](https://arxiv.org/html/2410.12557v1) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/29802) | candidate | Compose two short steps into a long-step bootstrap target |
| [sCM / TrigFlow](https://arxiv.org/html/2410.11081v1) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/29963) | candidate | TrigFlow data/noise and velocity parameterization |
| [Rotation Trick](https://arxiv.org/html/2410.06424v1) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/30284) | implemented | Nearest-codebook assignment |
| [Differential Transformer](https://arxiv.org/html/2410.05258v1) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/29792) | implemented | Difference of two separately normalized attention maps |
| [SVDQuant](https://arxiv.org/html/2411.05007v1) | [ICLR 2025 · Spotlight](https://iclr.cc/virtual/2025/poster/27906) | implemented | Smoothing transform that preserves a linear layer |
| [Block Diffusion](https://arxiv.org/html/2503.09573v2) | [ICLR 2025 · Oral](https://iclr.cc/virtual/2025/poster/28004) | implemented | Clean/noisy block diffusion attention mask |
| [Adjoint Matching](https://arxiv.org/html/2409.08861v3) | [ICLR 2025 · Spotlight](https://iclr.cc/virtual/2025/poster/27782) | candidate | Backward lean-adjoint recurrence |
| [In-Place TTT](https://arxiv.org/html/2604.06169v1) | [ICLR 2026 · Oral](https://iclr.cc/virtual/2026/poster/10008404) | candidate | Chunkwise MLP fast-weight apply/update |
| [ParaRNN](https://arxiv.org/html/2510.21450v1) | [ICLR 2026 · Oral](https://iclr.cc/virtual/2026/poster/10007563) | candidate | Assemble nonlinear recurrence residuals and Jacobians |
| [DiffusionNFT](https://arxiv.org/html/2509.16117v1) | [ICLR 2026 · Oral](https://iclr.cc/virtual/2026/poster/10009149) | implemented | Implicit positive/negative velocity policies |
| [LLaDA](https://arxiv.org/abs/2502.09992) | [NeurIPS 2025 · Oral](https://neurips.cc/virtual/2025/oral/118609) | candidate | 随机 masking 与仅对被 mask 位置计算的加权交叉熵 |
| [LoRA-One](https://openreview.net/forum?id=KwIlvmLDLm) | [ICML 2025 · Oral](https://icml.cc/virtual/2025/oral/47237) | candidate | 对负 full gradient 做 SVD 的谱初始化 |
| [IMM](https://arxiv.org/abs/2503.07565) | [ICML 2025 · Oral](https://icml.cc/virtual/2025/oral/47188) | candidate | 多粒子 kernel moment matching / MMD 目标 |
| [TarFlow](https://arxiv.org/abs/2412.06329) | [ICML 2025 · Oral](https://icml.cc/virtual/2025/oral/47266) | candidate | 严格自回归 affine flow 与 log-Jacobian |
| [MaxSup](https://arxiv.org/abs/2502.15798) | [NeurIPS 2025 · Oral](https://neurips.cc/virtual/2025/oral/116897) | candidate | 把 Label Smoothing 拆成 CE 与 logit regularizer |
| [KVzip](https://arxiv.org/abs/2505.23416) | [NeurIPS 2025 · Oral](https://neurips.cc/virtual/2025/oral/118742) | candidate | 从 context reconstruction attention 提取原 cache 的重要性 |
| [Chebyshev Policies](https://arxiv.org/abs/2605.22305) | [ICML 2026 · Spotlight (with oral talk)](https://icml.cc/virtual/2026/oral/71071) | candidate | 稳定递推 Chebyshev basis |
| [BeyondMuon](https://arxiv.org/abs/2602.04669) | [ICML 2026 · Spotlight](https://icml.cc/virtual/2026/poster/60694) | candidate | 谱变换 U Sigma^p V^T (p=0,1/4,1/2,1) |
| [MuonSSM](https://arxiv.org/abs/2606.30461) | [ICML 2026 · Spotlight (with oral talk)](https://icml.cc/virtual/2026/oral/71058) | candidate | 对低秩 input injection 应用 Newton–Schulz conditioning |
| [ConceptAttention](https://arxiv.org/abs/2502.04320) | [ICML 2025 · Oral](https://icml.cc/virtual/2025/oral/47221) | candidate | concept stream 单向读取 image/concept tokens |
| [REG](https://arxiv.org/abs/2507.01467) | [NeurIPS 2025 · Oral](https://neurips.cc/virtual/2025/oral/116345) | candidate | image latent 与 semantic class token 同时间噪声插值 |
| [Unifloral / TD3-AWR](https://arxiv.org/abs/2504.11453) | [NeurIPS 2025 · Oral](https://neurips.cc/virtual/2025/oral/119623) | candidate | TD3-AWR 的 critic bootstrap target |
| [MoGe](https://arxiv.org/abs/2410.19115) | [CVPR 2025 · Oral](https://cvpr.thecvf.com/virtual/2025/oral/35291) | implemented | ROE scale + z-shift alignment：分段线性断点搜索与截断加权 L1 目标 |
| [DAPS](https://arxiv.org/abs/2407.01521) | [CVPR 2025 · Oral](https://cvpr.thecvf.com/virtual/2025/oral/35347) | implemented | 条件后验能量与Langevin更新 |
| [Go-with-the-Flow](https://arxiv.org/abs/2501.08331) | [CVPR 2025 · Oral](https://cvpr.thecvf.com/virtual/2025/oral/35273) | candidate | 双向flow构造噪声传输图：扩张/收缩/空洞 |
| [CUT3R](https://arxiv.org/abs/2501.12387) | [CVPR 2025 · Oral](https://cvpr.thecvf.com/virtual/2025/oral/35313) | candidate | 小型persistent-state读写步：观测更新、虚拟视角只读 |
| [ViT³](https://arxiv.org/abs/2512.01643) | [CVPR 2026 · Oral](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf#page=20) | candidate | 单步full-batch inner update与query readout |
| [DiDiCM](https://arxiv.org/abs/2511.20263) | [CVPR 2026 · Oral](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf#page=20) | candidate | categorical forward/reverse transition generator与概率推进 |
| [SSG](https://arxiv.org/abs/2604.08048) | [CVPR 2026 · Oral](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf#page=57) | candidate | 依据cosine dissimilarity选择token对并进行空间/通道swap |
| [SenCache](https://arxiv.org/abs/2602.24208) | [CVPR 2026 · Oral](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf#page=57) | candidate | latent/time敏感度的有限差分校准 |

The live batch uses fixed, documented teaching adaptations unless a package explicitly identifies a paper-derived experiment. In particular, PKPO uses its scalar Gaussian reward experiment, while Block Diffusion recreates the paper’s attention-mask example. No toy result stands in for a full model benchmark. The per-module `problem.yaml` and bilingual statements define the exact implemented scope and source anchors.

Remaining priority-A candidates include Shortcut Models, LLaDA, LoRA-One, IMM, TarFlow, MaxSup, KVzip, In-Place TTT, Chebyshev Policies, Go-with-the-Flow, ViT³, and DiDiCM. They remain candidates until implementation, tests, mutation checks, and review are complete.
