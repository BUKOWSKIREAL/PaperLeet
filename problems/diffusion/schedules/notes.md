# 通过后的复盘

- 累计 alpha 的首项不是 1；前一时间的累计 alpha 首项才是 1。`test_reference` 与 `test_linear_endpoints` 抓下标偏移。
- beta 表示方差，乘到噪声上的必须是标准差。忘记平方根会使中间时间方差收缩，见 `test_variance_preserving`。
- cosine 最后一项未经裁剪会得到 beta=1，让后续反演出现零除。`test_cosine_clip` 固定上限为 0.999。
- 每个 batch 样本拥有自己的时间；t 需要先索引，再扩展空间维度。`test_per_sample_time` 使用非连续、乱序时间避免错误广播碰巧通过。

核对 [Improved DDPM 式 16](https://arxiv.org/abs/2102.09672) 和 [betas_for_alpha_bar 官方实现](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py)，访问日期 2026-10-13。官方线性 beta 会按 1000/T 缩放，本题固定首尾值。
