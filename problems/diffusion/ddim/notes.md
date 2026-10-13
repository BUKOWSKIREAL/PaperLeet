# 通过后的复盘

- eta=1 恢复的是相邻时间的后验方差；方向项必须同步减去 sigma²。`test_eta_one_posterior` 和 `test_reference` 分别抓两个因子错误。
- 时间子序列最后一项通常不是 0，数组索引 0 也仍对应一次加噪。必须额外跳到 alpha_bar=1，见 `test_point_final_jump`。
- 本题 trailing 步距是 T//n。diffusers 官方 trailing 使用 T/n 后取整，T=1000、n=50 时一致，非整除时可能不同。`test_trailing_times` 明确覆盖 T=17、n=5。[官方源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) 于 2026-10-13 核对。
- 确定性不表示有限步反演严格可逆：epsilon 依赖当前状态，一阶反演只是局部近似。干净起点首次用索引零近似，`test_inversion_converges` 检验 10→100 步误差下降。
- `test_step_convergence` 固定同一初始噪声，避免把采样随机性误当离散化误差。50 步 eta=1 的对照仅展示，不影响 eta=0 的质量评分。
