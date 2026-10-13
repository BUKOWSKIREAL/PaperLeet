# x₀ / ε / v Parameterization Conversion

## 1. What the paper solves

A diffusion network can predict clean data, added noise, or velocity. Salimans and Ho study these parameterizations for distillation into fewer sampling steps. Changing the prediction target for the same training sample changes endpoint stability and the weighting of prediction errors.

## 2. Core equations

Let $x_0,\epsilon,x_t\in\mathbb R^{B\times D}$, with $\alpha=\sqrt{\bar\alpha_t}$ and $\sigma=\sqrt{1-\bar\alpha_t}$ broadcastable to the input shape, and $\alpha^2+\sigma^2=1$. Here alpha is a square-root signal coefficient, unlike the one-step alpha in problem **6**.

$$x_t=\alpha x_0+\sigma\epsilon,\qquad v=\alpha\epsilon-\sigma x_0.$$

$$x_0=\alpha x_t-\sigma v,\qquad\epsilon=\sigma x_t+\alpha v.$$

Set $\alpha=\cos\phi$ and $\sigma=\sin\phi$; differentiation gives $v=dx_t/d\phi$. The inverse is a two-dimensional orthogonal rotation and requires no division. In contrast, recovering data from noise divides by alpha, while recovering noise from data divides by sigma. Each amplifies prediction errors near the endpoint where its denominator approaches zero. For the same erroneous prediction expressed in all three representations:

$$\|\hat v-v\|^2=\frac{\|\hat x_0-x_0\|^2}{\sigma^2}=\frac{\|\hat\epsilon-\epsilon\|^2}{\alpha^2}.$$

The prediction target therefore also implies a loss weighting. See [Progressive Distillation, Appendix D, Equations 26 and 30](https://arxiv.org/html/2202.00512v2#A4).

## 3. From equations to code

Implement the six functions in the starter. Preserve input shape and dtype, without aggregating over the batch.

```text
Inputs pred, x_t: [B,...]; alpha, sigma: broadcastable coefficients
Recover x0 or epsilon according to src
Return the representation selected by dst
If src == dst, return the original prediction
For divisions, clamp the denominator to at least 1e-8
```

Both source and destination for `convert` must be one of `x0 / eps / v`; raise `ValueError` for unknown strings. Identity conversion should introduce no numerical error. Implement v-to-x0 and v-to-epsilon directly to preserve stability at both the pure-signal and pure-noise endpoints.

## 4. Conventions

| Item | This exercise | Common scheduler interfaces |
| --- | --- | --- |
| Input coefficients | Already square-rooted alpha and sigma | Often store alpha_cumprod and take square roots internally |
| Output | Return a Tensor directly | diffusers schedulers usually return a step-output object |
| Endpoint division | Clamp denominator to 1e-8 | Depends on the schedule and stability strategy |
| Data dimensions | Any broadcastable shape | Images usually use B×C×H×W |

## 5. Interactive visualizations

<viz id="param_circle" />

Adjust the rotation angle phi to inspect the position and tangent vectors. The rotation direction determines the tangent's sign. This plot shows the coefficient plane; when high-dimensional independent noise and data are approximately orthogonal, it also suggests a sample-space interpretation.

<viz id="error_amplification" />

Move through time to inspect amplification factors $1/\alpha$ and $1/\sigma$, and understand why a parameterization becomes harder near an endpoint.

## 6. Run your implementation

Open problem **7**, complete the six functions in the browser editor, and click **Run tests**. Use the failure details to locate discrepancies.

T2 checks all six round trips, finite differences, endpoints, and the error-weight identity. The positions of numerical errors help distinguish broadcasting mistakes from formula mistakes.

## 7. Further exploration

Next, implement the DDPM posterior to connect noise predictions to sampling. Problem **10** introduces a different velocity: its derivative is with respect to linear time, whereas this exercise differentiates with respect to a rotation angle. See [Ho et al., 2020](https://arxiv.org/abs/2006.11239) for the original noise parameterization.
