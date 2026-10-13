# Review: Consistency Models: Boundary and Teacher Targets

At sigma_min the function must be exactly the identity, regardless of network output. The model receives c_in*x and the original sigma; this exercise supplies raw sigma conditioning instead of the official embedding. The skip/output coefficients follow the official boundary parameterization. The lower-noise teacher/EMA branch is frozen. The supplied Euler helper uses decreasing positive noise scales. This is consistency distillation with a teacher, not the separate teacher-free consistency-training algorithm.

A point-data teacher makes its probability-flow Euler step exact. Small MLP distillation exercises the boundary and target routing without pretrained models. Half the training pairs use a positive adjacent noise level sigma/2, so the EMA network affects the target; the other half anchor the exact sigma_min boundary.

The deliberately incorrect implementations isolate no boundary shift, unfrozen target, ema reversed, skip input scaling. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.
