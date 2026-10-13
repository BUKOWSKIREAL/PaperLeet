# 通过后的复盘

- 后验均值两项长得相近，却不能交换。`test_precision_product` 用高斯精度相加独立推导均值与方差，不依赖参考实现。
- 首个后验方差严格为零。improved-diffusion 在计算 log 方差前用第二项替换首项，避免 log(0)；本题直接使用方差，无需这一处理。[官方源码](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py) 于 2026-10-13 核对。
- 最后一步不加噪对 beta 方差选项也成立。只测试 posterior 会漏掉这一错误，`test_last_step_no_noise` 同时检查两种选项。
- 图像中的 x0 clipping 不能直接照搬到半径为 2 的点云。`test_gmm_sampling` 检查八个模式而非只检查点云中心。
- 采样必须降序；`test_sampler_time_order` 给出准确的错误时间位置，T3 再从分布层面检查。T3 MMD 阈值为参考实测的 1.5 倍，模式占比 12.5%±3%，半径 .5 外样本少于 5%。
