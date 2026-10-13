"""Teaching adaptation: estimate a population from alternating tiny feature batches."""
import torch


def run(batch_moments, ema_population_stats):
    dtype = torch.float64
    # Two shifted components; each tiny batch contains only one component.
    offsets = torch.tensor([[-1., -.5], [-1., .5], [1., -.5], [1., .5]], dtype=dtype)
    shift = torch.tensor([2., 1.], dtype=dtype)
    components = [offsets - shift, offsets + shift]
    population = torch.cat(components)
    target_mean = population.mean(0)
    target_cov = torch.cov(population.T, correction=0)
    mean, second = batch_moments(population)
    history_mean, history_second = mean.detach(), second.detach()
    beta = .9
    batch_errors, ema_errors, mean_history, cov_history = [], [], [], []
    last_features, last_mean, last_cov = None, None, None
    for step in range(40):
        features = components[step % 2].clone().requires_grad_()
        batch_mean, raw_second = batch_moments(features)
        mean, second, covariance = ema_population_stats(features, history_mean, history_second, beta)
        batch_cov = raw_second - torch.outer(batch_mean, batch_mean)
        batch_errors.append(((batch_mean - target_mean).square().sum() + (batch_cov - target_cov).square().sum()).item())
        ema_errors.append(((mean - target_mean).square().sum() + (covariance - target_cov).square().sum()).item())
        mean_history.append(mean[0].item()); cov_history.append(covariance[0, 0].item())
        history_mean, history_second = mean.detach(), second.detach()
        last_features, last_mean, last_cov = features, mean, covariance
    objective = last_mean.square().sum() + last_cov.trace()
    gradient, = torch.autograd.grad(objective, last_features)
    # This objective equals trace(raw second), so its gradient has a scalar oracle.
    expected_gradient = 2 * (1 - beta) * last_features.detach() / last_features.shape[0]
    return {
        "kind": "toy_experiment", "title": "FD-loss: a population beyond the current microbatch",
        "problem": "diffusion/fd_population",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2604.28190",
            "anchor": "Section 3.2, Eqs. (3)-(5), Algorithm 1",
            "adaptation": "Forty deterministic microbatches alternate between two shifted 2D feature components. EMA starts from the known eight-point population; this illustrates estimator smoothing, not an ImageNet experiment or a guarantee under distribution shift."},
        "metrics": {"batch_average_error": sum(batch_errors) / len(batch_errors),
            "ema_average_error": sum(ema_errors) / len(ema_errors),
            "gradient_max_error": (gradient - expected_gradient).abs().max().item(),
            "final_mean_error": (last_mean - target_mean).norm().item(),
            "final_covariance_error": (last_cov - target_cov).norm().item()},
        "charts": [
            {"kind": "line", "title": "Moment estimation error", "x_label": "Microbatch",
                "x": list(range(40)), "series": [{"name": "Batch only", "values": batch_errors}, {"name": "EMA population", "values": ema_errors}]},
            {"kind": "line", "title": "Estimated first-coordinate mean", "x_label": "Microbatch",
                "x": list(range(40)), "series": [{"name": "EMA", "values": mean_history}, {"name": "Full population", "values": [0.] * 40}]},
            {"kind": "line", "title": "Estimated first-coordinate variance", "x_label": "Microbatch",
                "x": list(range(40)), "series": [{"name": "EMA", "values": cov_history}, {"name": "Full population", "values": [target_cov[0, 0].item()] * 40}]},
        ],
    }
