# Flow Matching Velocity Fields

## 1. What the paper solves

Flow Matching learns a time-dependent velocity field that transports noise into data through an ordinary differential equation. Its conditional training objective regresses a directly computable velocity, without simulating an entire generation trajectory on every training step.

## 2. Core equations

In this exercise, x0 is noise and x1 is data. Both have shape `[B,D]`; time t has shape `[B]`, with a separate interpolation weight for each sample:

$$x_t=(1-t)x_0+tx_1,\qquad u=x_1-x_0,\qquad\mathcal L=\operatorname{mean}_{b,d}\left(v_\theta(x_t,t)-u\right)^2.$$

Sampling starts from standard Gaussian noise and integrates forward in time:

$$x_{k+1}=x_k+\frac1n v_\theta(x_k,k/n),\quad k=0,\ldots,n-1.$$

The conditional velocity u is the displacement between a pair of endpoints, but the network sees only the current position and time. Different endpoint pairs can reach the same position, so the network learns an average conditional velocity. Even an optimal network generally has nonzero training loss: irreducible variance is not an implementation error. See [Flow Matching Equations 5, 9, 22, and 23](https://arxiv.org/html/2210.02747v2) for gradient equivalence and linear conditional paths.

A point distribution is a special case: $v^*(x,t)=(\mu-x)/(1-t)$. Substituting the interpolated position gives a constant target velocity, zero oracle loss, and exact arrival at the data point in one Euler step. Marginal trajectories for general mixtures can curve, making multiple integration steps useful.

## 3. From equations to code

```text
Training: independently sample x0, x1, and t [B]
Reshape t to [B,1] and construct interpolation [B,D]
Regress x1-x0; mean squared error over all elements → scalar
Sampling: start from x0 and iterate k = 0, ..., n-1
Call the model at the left endpoint k/n; advance by step size 1/n
```

Implement five functions covering time sampling, interpolation, targets, loss, and Euler sampling. `sample_t` supports uniform and logit-normal distributions and must return values in the open interval. The model interface is `model(x,t)`. Enabling trajectories returns `(samples, trajectory)`, including the initial noise and final samples.

## 4. Conventions

| Item | This exercise / Lipman | Rectified Flow | SD3 |
| --- | --- | --- | --- |
| Time-zero endpoint | Noise | Source distribution; noise for generation | Data |
| Time-one endpoint | Data | Target distribution; data for generation | Noise |
| Generation direction | 0 → 1 | 0 → 1 | 1 → 0 |
| Time sampling | Uniform by default | Uniform in the base objective | Studies logit-normal and other weightings |

See [SD3 Equation 13](https://arxiv.org/html/2403.03206v1) for its reversed convention. When adapting code, check the interpolation, target sign, and integration direction together.

## 5. Interactive visualizations

<viz id="flow_particles" />

Play the particle flow and move through time to inspect velocity arrows at current positions. Change Euler step counts to compare straight oracle paths with learned curved trajectories.

<viz id="convention_bug" />

Toggle the reversed-sampling bug to see the same model generate incorrect results. Compare the loss curve to understand why decreasing training loss alone does not establish correct sampling.

## 6. Run your implementation

Open problem **10**, complete the five functions in the browser editor, and click **Run tests**. View training and sampling diagnostics in the results panel.

T3 actually trains TimeMLP on two CPU threads using Adam, learning rate 1e-3, batch size 256, and 2,000 steps. It generates 2,000 points and checks MMD, coverage of all eight modes, and the outlier rate. Thresholds and measured runtime are recorded in `problem.yaml`.

## 7. Further exploration

Read [Rectified Flow](https://arxiv.org/abs/2209.03003) to explore how recoupling can straighten trajectories. This exercise pairs independent noise and data; it does not use reflow or a globally optimal transport coupling.
