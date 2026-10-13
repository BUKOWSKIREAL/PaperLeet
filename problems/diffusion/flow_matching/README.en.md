# Conditional Path and Velocity Regression

## 1. What the paper solves

Flow Matching learns a time-dependent velocity field that transports noise into data through an ordinary differential equation. Its conditional training objective regresses a directly computable velocity, without simulating an entire generation trajectory on every training step.

This module selects the linear conditional path and its CFM objective. Implement only `interpolate`, `cfm_target`, and `cfm_loss`; time sampling, the Euler solver, test model, and training loop are supplied.

## 2. Core equations

In this exercise, x0 is noise and x1 is data. Both have shape `[B,D]`; time t has shape `[B]`, with a separate interpolation weight for each sample:

$$x_t=(1-t)x_0+tx_1,\qquad u=x_1-x_0,\qquad\mathcal L=\operatorname{mean}_{b,d}\left(v_\theta(x_t,t)-u\right)^2.$$

The conditional velocity u is the displacement between a pair of endpoints, but the network sees only the current position and time. Different endpoint pairs can reach the same position, so the network learns an average conditional velocity. Even an optimal network generally has nonzero training loss: irreducible variance is not an implementation error. See [Flow Matching Equations 5, 9, 22, and 23](https://arxiv.org/html/2210.02747v2) for gradient equivalence and linear conditional paths.

A point distribution is a special case: $v^*(x,t)=(\mu-x)/(1-t)$. Substituting the interpolated position gives a constant target velocity, zero oracle loss, and exact arrival at the data point in one Euler step. Marginal trajectories for general mixtures can curve, making multiple integration steps useful.

## 3. From equations to code

Implement three functions around the same training objective:

```text
interpolate: broadcast t [B] to [B,...] and construct the noise-to-data path
cfm_target: return the path’s derivative with respect to time
cfm_loss: evaluate model(x,t) at the result of interpolate
Average the squared elementwise error against cfm_target
```

`interpolate` and `cfm_target` preserve endpoint shape and dtype. `cfm_loss` returns a scalar, averages over the batch and every coordinate, and preserves gradients to model parameters.

`sample_t` and `euler_sample` are provided. The test harness manages the model, optimizer, and training loop, calling your `cfm_loss` during training. Time sampling is uniform by default; the helper’s logit-normal option is available for later comparison.

## 4. Conventions

| Item | This module’s convention |
| --- | --- |
| Endpoints | `x0` is noise; `x1` is data |
| Time | `t` has shape `[B]` and runs from 0 to 1 |
| Conditional path | The paper’s conditional OT path with zero terminal noise |
| Pairing | Independent noise and data |
| Loss reduction | Mean over the batch and every coordinate |
| Test model | `model(x,t)` returns a velocity with the same shape as x |

Regression against conditional targets learns the marginal velocity field, without solving an ODE for each training example.

## 5. Interactive visualizations

<viz id="flow_particles" />

Play the particle flow and move through time to inspect velocity arrows at current positions. Change Euler step counts to compare straight oracle paths with learned curved trajectories.

<viz id="convention_bug" />

Toggle the reversed-sampling bug to see the same model generate incorrect results. Compare the loss curve to understand why decreasing training loss alone does not establish correct sampling.

## 6. Run your implementation

Open module **10** in the browser, complete `interpolate`, `cfm_target`, and `cfm_loss`, and run the tests.

T3 uses the supplied training loop on two CPU threads: TimeMLP with Adam, learning rate 1e-3, batch size 256, and 2,000 steps. It calls your loss during training, then uses the supplied Euler solver to generate 2,000 points and checks MMD, coverage of eight modes, and outliers. Thresholds and measured runtime are in `problem.yaml`.

## 7. Further exploration

The supplied Euler solver integrates forward in time, evaluating at left endpoints to avoid the point oracle’s singularity at t=1:

$$x_{k+1}=x_k+\frac1n v_\theta(x_k,k/n),\quad k=0,\ldots,n-1.$$

For optional exploration, read [Rectified Flow](https://arxiv.org/abs/2209.03003) on recoupling and [SD3 Equation 13](https://arxiv.org/html/2403.03206v1) on reversed time conventions and nonuniform time sampling. These comparisons add no required functions to this module.
