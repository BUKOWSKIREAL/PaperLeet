# Review: JiT: Clean Prediction in High Dimensions

For t away from the clamped endpoints, v=x-epsilon, x prediction maps via (x_hat-z)/(1-t), and epsilon prediction via (z-epsilon_hat)/t. All networks use the same velocity loss. Clamping follows official x-prediction code; exact cross-parameterization equivalence is only asserted outside the clamped boundary region. Broadcast t from [B] over any trailing dimensions.

Reimplements the fixed orthogonal 2D→D embedding idea in Figure 2 with synthetic spiral data. Default D=16, width=32, depth=3, 160 steps; the original uses D=2/8/16/512 and width=256, depth=5. Run `python problems/diffusion/jit/toy.py --impl problems/diffusion/jit/solution.py --full --steps 5000` for that dimension/architecture sweep; the step count and spiral distribution are our choices. Short-run comparisons are observations, not a guarantee of the paper ranking. No original toy source was found.

The deliberately incorrect implementations isolate wrong time direction, wrong x denominator, clean space loss. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.

In the experiment scaffold, depth counts all Linear layers including the output; `--full` therefore has five Linear layers with ReLU between them.
