# Reference notes

The official [training objective](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py) uses elementwise mean MSE and `pred_loss + lambda * sigreg_loss`. The paper's squared-norm notation suppresses reductions; this package pins the implementation's mean convention. SIGReg receives the transposed full embedding sequence. Its [single-GPU implementation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py) averages the empirical characteristic function over the batch at each time, then averages projection/time statistics.

With `e_t=2*z_t+a_t-z_(t+1)`, prediction gradients enter an interior embedding twice: positively through its role as context, negatively through its role as the previous step's target. The analytic oracle accumulates both contributions and the regularizer derivative. Detaching the target can leave the forward loss unchanged while corrupting this gradient.

For the rollout `z_(t+1)=c*z_t+k*a_t`, the final-state derivative with respect to action `a_j` is `k*c^(H-1-j)`. The test also checks that a later action cannot influence the first prediction. Reusing the initial state at every step passes a one-step example but fails both recurrence and gradient checks.

The supplied toy uses a scalar encoder and a Markov transition, not the paper's ViT and history-window Transformer. Its regularizer is actual fixed-projection SIGReg with the official finite quadrature, not a replacement variance penalty. All plotted trajectories are training examples. Four mutants stop target gradients, swap regularizer axes, reuse the initial state or substitute convex-combination weighting.
