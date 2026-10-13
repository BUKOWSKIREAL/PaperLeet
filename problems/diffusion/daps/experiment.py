"""Teaching adaptation of one DAPS annealing transition with an analytic Gaussian oracle."""
import torch


def run(posterior_score, langevin_step, renoise):
    generator = torch.Generator().manual_seed(42)
    dtype = torch.float64
    batch = 4096
    # x0_hat stands for the supplied probability-flow ODE output at this noise level.
    anchor = torch.zeros(batch, 2, dtype=dtype)
    operator = torch.tensor([[1., .5]], dtype=dtype)
    measurement = torch.full((batch, 1), 1.5, dtype=dtype)
    r, beta, eta, sigma_next = 1., .5, .025, .3
    precision = torch.eye(2, dtype=dtype) / r ** 2 + operator.T @ operator / beta ** 2
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (operator.T @ measurement[0] / beta ** 2)
    # Unadjusted Langevin has finite-step stationary covariance H^-1(I-eta H/2)^-1.
    ula_cov = torch.linalg.inv(precision - eta * precision @ precision / 2)
    x = anchor.clone()
    times, residuals, mean_errors = [], [], []
    with torch.no_grad():
        for step in range(241):
            if step % 10 == 0:
                times.append(step)
                residuals.append((x @ operator.T - measurement).square().mean().item())
                mean_errors.append((x.mean(0) - mean).norm().item())
            if step < 240:
                noise = torch.randn(x.shape, generator=generator, dtype=dtype)
                x = langevin_step(x, posterior_score(x, anchor, operator, measurement, r, beta), eta, noise)
        fresh = torch.randn(x.shape, generator=generator, dtype=dtype)
        noisy = renoise(x, sigma_next, fresh)
    empirical_cov = (x - x.mean(0)).T @ (x - x.mean(0)) / batch
    displacement = noisy - x
    return {
        "kind": "toy_experiment", "title": "DAPS local posterior: Langevin then independent re-noising",
        "problem": "diffusion/daps",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2407.01521",
                       "anchor": "Eq. (7) and Appendix A Algorithm 1",
                       "adaptation": "One local annealing transition with a supplied ODE estimate and a 2D linear Gaussian likelihood. Analytic conditional moments replace image benchmarks and Appendix G's nonlinear toy."},
        "metrics": {"mean_error": (x.mean(0) - mean).abs().max().item(),
                    "ula_covariance_error": (empirical_cov - ula_cov).abs().max().item(),
                    "posterior_covariance_error": (empirical_cov - covariance).abs().max().item(),
                    "initial_measurement_mse": residuals[0], "final_measurement_mse": residuals[-1],
                    "renoise_variance_error": (displacement.var(0, unbiased=False) - sigma_next ** 2).abs().max().item(),
                    "renoise_formula_error": (displacement - sigma_next * fresh).abs().max().item()},
        "charts": [
            {"kind": "line", "title": "Measurement consistency", "x_label": "Langevin step",
             "x": times, "series": [{"name": "Measurement MSE", "values": residuals}]},
            {"kind": "line", "title": "Convergence to analytic conditional mean", "x_label": "Langevin step",
             "x": times, "series": [{"name": "Mean error", "values": mean_errors}]},
            {"kind": "line", "title": "Posterior covariance diagonal", "x_label": "Coordinate",
             "x": [0, 1], "series": [
                 {"name": "Exact Gaussian posterior", "values": covariance.diag().tolist()},
                 {"name": "Finite-step Langevin target", "values": ula_cov.diag().tolist()},
                 {"name": "Submitted implementation", "values": empirical_cov.diag().tolist()}]}],
    }
