# LeWM Joint Prediction and Latent Rollout

[LeWorldModel](https://arxiv.org/html/2603.19312v2) learns an encoder and an action-conditioned predictor together. Its teacher-forced prediction term compares a prediction from `(z_t, a_t)` with the next encoded observation `z_{t+1}`. **Both encoding paths receive gradients.** A supplied SIGReg term keeps the representation distribution from collapsing.

Implement two functions:

| Function | Input | Output |
|---|---|---|
| `lewm_objective(embeddings, actions, predictor, regularizer, weight=.1)` | Embeddings `(B,T,D)`, actions `(B,T-1,A)`, two callables | `(total, prediction, regularization)`, three scalar tensors |
| `latent_rollout(initial, actions, transition)` | Initial state `(B,D)`, actions `(B,H,A)`, transition callable | Predicted states `(B,H,D)` |

## Joint objective

Call `predictor(embeddings[:, :-1], actions)` once. The supplied predictor is causal and returns `(B,T-1,D)`; its architecture is outside this exercise. Align predictions with `embeddings[:, 1:]`. Use mean squared error across **all** batch, time and feature elements, matching the official `train.py` implementation:

$$L_{pred}=\frac{1}{B(T-1)D}\sum_{b,t,d}(\hat z_{b,t+1,d}-z_{b,t+1,d})^2.$$

Call `regularizer(embeddings.transpose(0,1))` once on **all T encoded frames**, with shape `(T,B,D)`. Its scalar output already averages its time-wise statistics. Return `prediction + weight * regularization`, along with the two components. Do not replace the prediction coefficient by `1-weight`, detach targets, use an EMA target, or flatten time into batch. `weight` is finite and nonnegative; zero is supported.

## Autoregressive rollout

For each action, call `transition(current_state, actions[:, t])`, then feed that prediction into the next step. Return the `H` predictions, excluding the initial state. The transition preserves `(B,D)`. Keep gradients through initial state, actions and previous predictions.

This interface isolates the **single-context-state case, N=1**. The paper's general predictor accepts a history window; implementing its Transformer, image encoder and history cache is outside this module. Training uses observed latent context, while rollout uses recursively predicted latent context.

All tensors use matching float32/float64 dtype and device. Require `B,D,A,H>=1` and `T>=2`; preserve input values. Input validators are provided in the starter.

For `z_0=2`, actions `[1,-2]`, and `transition(z,a)=.5*z+2*a`, the rollout is `[3,-2.5]`.

## Toy: learn a one-dimensional latent world

**Teaching adaptation.** A scalar encoder and a two-parameter linear predictor train on 24 four-transition sequences, using your objective and a supplied one-dimensional, fixed-direction SIGReg implementation. The latter follows the official quadrature on `[0,3]`; it is supplied so this exercise can focus on the world-model objective. Your rollout then predicts one of those training trajectories. Curves report prediction MSE, total loss and encoder scale.

This does not reproduce image training or evaluate held-out generalization. The analytical T2 tests separately check target-path gradients, recurrent gradients and temporal causality. T3 trains with the submitted functions; the preview uses the reference solution.
