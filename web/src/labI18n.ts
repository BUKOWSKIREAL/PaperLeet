import { useI18n } from "./i18n";

const english: Record<string, string> = {
  "这道题暂未提供交互实验。你可以阅读题面并运行测试。":
    "An interactive lab is not available for this problem yet. Read the description and run the tests to explore it.",
  参考解二维采样点云: "Reference 2D samples",
  粒子: "particles",
  柱状图: "Bar chart",
  二维点云: "2D point cloud",
  回答: "Response",
  梯度: "gradient",
  旋转角: "Rotation angle",
  "v 是单位圆的切向量。": "v is tangent to the unit circle.",
  采样进度: "Sampling progress",
  暂停动画: "Pause animation",
  播放动画: "Play animation",
  重置动画: "Reset animation",
  "，叠加当前时间速度场": ", with the velocity field at the current time",
  "，背景为高斯混合分量的一倍与两倍标准差等高线":
    ", with one- and two-standard-deviation contours of the Gaussian mixture components",
  与: " and ",
  "让 TD 误差沿着时间传播": "Propagate TD errors through time",
  "拖动 γ 和 λ，观察优势估计。关闭 episode 隔离，检查误差怎样跨边界传播。":
    "Adjust γ and λ to inspect the advantage estimates. Disable episode isolation to see errors cross boundaries.",
  "γ · 折扣因子": "γ · discount factor",
  "λ · 轨迹权重": "λ · trace weight",
  "rₜ 奖励": "rₜ reward",
  "时间 t": "Time t",
  正常步: "regular step",
  "Âₜ 优势": "Âₜ advantage",
  "启用 episode 边界隔离": "Isolate episode boundaries",
  偏差与方差的交换: "The bias–variance tradeoff",
  "参考解在固定轨迹上的实测结果。λ 的两端体现偏差与方差之间的权衡。":
    "Measured reference results on fixed trajectories. The endpoints of λ illustrate the bias–variance tradeoff.",
  "实测 bias": "Measured bias",
  "实测 variance": "Measured variance",
  "同一个 batch，多轮更新后会发生什么":
    "What happens after multiple updates on one batch?",
  "比较参考实现与两个错误变体的 KL。下图来自同一组固定数据的真实更新。":
    "Compare KL for the reference and two faulty variants. These curves come from actual updates on the same fixed data.",
  "只 clip": "clip only",
  "max 替代 min": "max instead of min",
  最优臂概率: "Best-arm probability",
  实际序列的标签移位: "Label shifting on an actual sequence",
  "固定 token 样本。mask 为 0 的 prompt 不参与序列 log-prob 求和。":
    "A fixed token sample. Prompt tokens with mask 0 do not contribute to the sequence log-probability.",
  "有效 token log-prob": "Valid-token log-probability",
  偏好学习收敛到闭式最优策略:
    "Preference learning approaches the closed-form optimum",
  "拖动训练步数，查看隐式奖励与真实奖励的关系；对照最终策略与闭式解。":
    "Scrub through training to compare implicit and true rewards, then compare the final policy with the closed-form solution.",
  训练记录: "Training snapshot",
  "隐式 rewards": "Implicit rewards",
  "真实 rewards": "True rewards",
  选项: "Option",
  学到的策略: "Learned policy",
  闭式最优策略: "Closed-form optimum",
  概率: "Probability",
  归一化如何影响回答长度: "How normalization affects response length",
  策略更新的真实训练曲线: "Measured policy training curves",
  "切换奖励与长度，比较 GRPO 和 Dr. GRPO。长度偏置属于展示实验，不参与通过判定。":
    "Switch between rewards and lengths to compare GRPO and Dr. GRPO. This length-bias experiment is illustrative and does not affect acceptance.",
  "训练曲线来自固定随机种子的 toy 自回归策略。":
    "Training curves from a toy autoregressive policy with a fixed random seed.",
  回答长度: "Response length",
  错误回答长度: "Incorrect response length",
  "信号与噪声，在时间中的消长": "Signal and noise over time",
  "拖动离散时间 t，观察真实调度的累乘 ᾱₜ、log-SNR，以及固定高斯噪声下的二维点云。":
    "Adjust discrete time t to inspect cumulative ᾱₜ, log-SNR, and a 2D point cloud with fixed Gaussian noise.",
  "T · 调度总步数": "T · total schedule steps",
  "t · 离散时间": "t · discrete time",
  "灰色：固定 x₀ 与当前 xₜ": "Gray: fixed x₀ and current xₜ",
  "逆着噪声，恢复八个模式": "Reverse the noise to recover eight modes",
  "播放参考解的实际采样轨迹。切换方差策略；背景圈表示当前高斯混合分量的一倍与两倍标准差轮廓。":
    "Play the reference sampling trajectory and switch variance strategies. Background circles show one- and two-standard-deviation contours of the current mixture components.",
  "当前 t": "Current t ",
  "两种反向方差，在最后一步相遇":
    "Two reverse variances meet at the final step",
  "对数纵轴下比较 βₜ 与精确后验方差 β̃ₜ；t=0 的后验方差为零，因此未画入对数曲线。":
    "Compare βₜ with exact posterior variance β̃ₜ on a log scale. The zero posterior variance at t=0 is omitted from the log curve.",
  "t = 0：后验方差为 0，采样不添加噪声。":
    "t = 0: posterior variance is zero, so sampling adds no noise.",
  的粒子轨迹比较: " particle trajectories",
  "η × 步数：选一条生成路径": "η × steps: choose a generation path",
  "点击网格，播放该设置下由参考解预计算的真实轨迹。每格 MMD 来自独立质量评估。":
    "Select a grid cell to play the precomputed reference trajectory. Each cell's MMD comes from an independent quality evaluation.",
  步: " steps",
  "同样的起点，不同的随机性": "The same starting points, different randomness",
  "30 个粒子从相同 xT 出发；彩色为 η=0，灰色为 η=1 的真实轨迹。":
    "30 particles start from the same xT. Colored paths use η=0; gray paths use η=1.",
  "反演，再生成": "Invert, then generate",
  "选择数据点，查看它在噪声空间的位置和重建位置；增加步数会改变离散化误差。":
    "Choose a data point to see its noise-space position and reconstruction. Changing the step count changes discretization error.",
  点: "Point ",
  重建点: "Reconstructed",
  原始点: "Original",
  "反演 xT": "Inverted xT",
  "重建 L₂ 误差": "Reconstruction L₂ error",
  往返误差: "Round-trip error",
  粒子沿着学到的速度场流动: "Particles follow the learned velocity field",
  "参考模型的实际 Euler 采样轨迹。播放或拖动 t；箭头显示同一时刻的预测速度场。":
    "Actual Euler trajectories from the reference model. Play or scrub t; arrows show the predicted velocity field at that time.",
  "t · 噪声 → 数据": "t · noise → data",
  "Euler 用多少步，分布会怎样变化":
    "How the Euler step count changes the distribution",
  "切换步数，比较同一模型从相同噪声出发得到的最终样本。":
    "Switch step counts to compare final samples from the same model and initial noise.",
  "同一个模型，采样方向反了": "The same model, with sampling reversed",
  "切换到错误方向，查看真实 mutant 输出；训练 loss 下降并不保证采样方向正确。":
    "Switch to the wrong direction to inspect an actual faulty implementation. A falling training loss does not ensure the sampling direction is correct.",
  采样方向反了: "Reverse the sampling direction",
  "错误：从 t=1 积分到 t=0。": "Incorrect: integrate from t=1 to t=0.",
  "正确：从噪声 t=0 积分到数据 t=1。":
    "Correct: integrate from noise at t=0 to data at t=1.",
  直线与弯曲路径: "Straight and curved paths",
  "单点目标的最优流是直线；高斯目标的轨迹随方差变化而弯曲。下图均来自解析 oracle。":
    "The optimal flow to a point is straight; Gaussian trajectories curve as variance changes. Both plots use analytical oracles.",
  "单点 oracle": "Point oracle",
  "高斯 oracle": "Gaussian oracle",
  "彩色：单点": "Colored: point",
  "灰色：高斯": "Gray: Gaussian",
  "为什么训练 loss 不会降到零": "Why training loss does not reach zero",
  "实际 CFM 训练损失。同一个 xt 可以来自不同的样本配对，条件速度目标存在不可约方差。":
    "Measured CFM training loss. The same xt can arise from different sample pairs, creating irreducible variance in the conditional velocity target.",
  公式实验台: "Formula lab",
  "可交互 · 与判题独立": "Interactive · independent of grading",
  采样结果: "Samples",
  真实分布: "Target distribution",
  "Clipping，究竟截断了哪一边？": "Which side does clipping actually truncate?",
  "调整 ε 与概率比 r。阴影区域的有利更新被截断，目标梯度变为零。":
    "Adjust ε and probability ratio r. Favorable updates in the shaded region are clipped, making the objective gradient zero.",
  "ε · 截断半径": "ε · clip radius",
  "r · 概率比": "r · probability ratio",
  "当前 ∂L/∂r =": "Current ∂L/∂r =",
  从序列概率到偏好目标: "From sequence probabilities to preference objectives",
  "拖动 β 与标签平滑 ε，观察 loss 及实际梯度怎样变化；调整 prompt 长度，检查右移后的有效 token。":
    "Adjust β and label smoothing ε to inspect the loss and its actual gradient. Change prompt length to inspect valid tokens after shifting.",
  "β · 偏好温度": "β · preference temperature",
  "margin · 对数概率比差": "margin · log-probability ratio difference",
  "实际 loss 梯度": "Actual loss gradient",
  "当前 loss": "Current loss ",
  "ε=0 时，负梯度的大小是 βσ(−βh)；ε&gt;0 时，过大的正 margin 也会收到回拉梯度。loss 使用稳定的 softplus 计算。":
    "At ε=0, the negative gradient magnitude is βσ(−βh). With ε>0, an excessive positive margin also receives a restoring gradient. The loss uses stable softplus.",
  "prompt token 数": "Prompt token count",
  "logits[t−1] 预测 token[t]；只对 mask=1 的回答 token 求和。":
    "logits[t−1] predicts token[t]; sum only over response tokens with mask=1.",
  "去掉组标准化，看清长度偏置":
    "Remove group normalization to inspect length bias",
  "同一组回答，比较相对好坏": "Compare responses within a group",
  "调整 8 个回答各自的奖励，或整体缩放奖励。开关样本标准差归一化，观察 advantage 的变化。":
    "Adjust the rewards of eight responses or scale them together. Toggle sample-standard-deviation normalization to inspect the advantages.",
  奖励整体缩放: "Reward scale",
  "除以组内样本标准差（correction=1）":
    "Divide by group sample standard deviation (correction=1)",
  "G=8 组均值=": "G=8 group mean=",
  "样本 std=": "sample std=",
  "相同优势，不同长度：每个 token 分到多少梯度":
    "Equal advantages, different lengths: gradient per token",
  "4 条回答的 advantage 相同，以隔离聚合方式的差异。切换 GRPO、DAPO 与 Dr. GRPO，观察长度 3、6、12、24 的逐 token loss 梯度。":
    "Four responses share the same advantage to isolate aggregation effects. Switch between GRPO, DAPO, and Dr. GRPO to inspect per-token gradients at lengths 3, 6, 12, and 24.",
  "所有回答共享的 advantage A": "Shared response advantage A",
  "每个有效 token：": "Each valid token: ",
  "padding：梯度为零": "Padding: zero gradient",
  "这里只比较聚合；Dr. GRPO 去掉组标准差的影响见上方优势图。padding 的梯度始终为零。":
    "This compares aggregation only. The advantage plot above shows the effect of removing group-standard-deviation normalization in Dr. GRPO. Padding gradients are always zero.",
  "同一次正确回答，为什么难题的权重更大":
    "Why a correct answer to a hard question gets more weight",
  "固定 G=8、二元奖励。拖动组内正确率 p，比较正确回答的优势；切换标准差约定，查看样本修正系数。":
    "With G=8 and binary rewards, adjust group accuracy p to compare advantages of correct answers. Switch standard-deviation conventions to inspect the sample correction.",
  "p · 组内正确率": "p · group accuracy",
  "本题的样本 std（correction=1）":
    "Use this problem's sample std (correction=1)",
  "GRPO · 正确回答 |A|": "GRPO · correct-answer |A|",
  "组内正确率 p": "Group accuracy p",
  "正确回答 |A|": "Correct-answer |A|",
  "曲线为连续延拓，G=8 时非退化正确率为 1/8 至 7/8；全组奖励相同时优势为零。这里忽略 1e−4 稳定项，样本 std 的修正为 √((G−1)/G)。":
    "The curve is a continuous extension; with G=8, nondegenerate accuracy ranges from 1/8 to 7/8. Equal rewards give zero advantage. The 1e−4 stability term is omitted; sample std adds a √((G−1)/G) correction.",
  "三种参数化，同一个旋转坐标系":
    "Three parameterizations in one rotating coordinate system",
  "拖动 φ。xₜ 沿四分之一圆移动，v 是它的切向量；观察不同时间的误差放大。":
    "Adjust φ. xₜ moves along a quarter circle and v is its tangent vector. Inspect error amplification at different times.",
  "φ · 旋转角": "φ · rotation angle",
  "误差放大 · log scale": "Error amplification · log scale",
  你的采样结果与真实分布: "Your samples and the target distribution",
  你的实现: "Your implementation",
  阈值: "Threshold ",
  "MMD 相对通过阈值": "MMD relative to the acceptance threshold",
  在质量阈值以内: "Within the quality threshold",
  超过质量阈值: "Above the quality threshold",
};
export function useLabText() {
  const { locale } = useI18n();
  return (text: string) =>
    locale === "en"
      ? (english[text.replace(/\s+/g, " ").trim()] ?? text)
      : text;
}
