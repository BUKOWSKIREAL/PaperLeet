"""Teaching adaptation: FD post-training of a small affine Gaussian generator."""
import torch


def _fd2(mean, covariance, target_mean, target_covariance):
    # Independent 2x2 SPD trace-square-root identity, no eigendecomposition.
    trace_product = torch.trace(covariance @ target_covariance)
    trace_root = torch.sqrt(trace_product + 2 * torch.sqrt(torch.det(covariance) * torch.det(target_covariance)))
    return (mean - target_mean).square().sum() + covariance.trace() + target_covariance.trace() - 2 * trace_root


def run(frechet_distance, normalized_fd_loss):
    dtype = torch.float64
    target_mean = torch.tensor([1., -1.], dtype=dtype)
    target_covariance = torch.tensor([[1.5, .4], [.4, .7]], dtype=dtype)
    noise = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=dtype)
    parameters = torch.tensor([-1., 1., -.5, -.6, .4], dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=.02)
    raw_losses, normalized_losses, true_losses, mean_x = [], [], [], []
    old_mean, old_second = None, None
    final_covariance = None
    for step in range(201):
        lower = torch.stack((parameters[2].exp(), parameters.new_zeros(()), parameters[3], parameters[4].exp())).reshape(2, 2)
        generated = noise @ lower.T + parameters[:2]
        mean = generated.mean(0); second = generated.T @ generated / len(generated)
        covariance = second - torch.outer(mean, mean)
        if old_mean is None:
            old_mean, old_second = mean.detach(), second.detach()
        ema_mean = .8 * old_mean + .2 * mean
        ema_second = .8 * old_second + .2 * second
        ema_covariance = ema_second - torch.outer(ema_mean, ema_mean)
        distance = frechet_distance(ema_mean, ema_covariance, target_mean, target_covariance)
        loss = normalized_fd_loss(distance[None], distance.new_ones(1), .1)
        raw_losses.append(distance.item())
        normalized_losses.append(loss.item())
        true_losses.append(_fd2(mean, covariance, target_mean, target_covariance).item())
        mean_x.append(mean[0].item())
        old_mean, old_second = ema_mean.detach(), ema_second.detach()
        final_covariance = covariance.detach()
        if step < 200:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {
        "kind": "toy_experiment", "title": "FD-loss: post-train a two-dimensional generator",
        "problem": "diffusion/fd_loss",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2604.28190",
            "anchor": "Eqs. (2), (6); Algorithm 1; Appendix B.3",
            "adaptation": "A five-parameter affine generator uses four symmetric noise points, supplied EMA moments (beta=.8), and submitted differentiable FD and normalization. The reference covariance is correlated. This is a CPU teaching experiment, not image generation; eps=.1 and the optimizer are teaching settings."},
        "metrics": {"initial_fd": true_losses[0], "final_fd": true_losses[-1],
            "initial_ema_fd": raw_losses[0], "final_ema_fd": raw_losses[-1],
            "initial_normalized_objective": normalized_losses[0], "final_normalized_objective": normalized_losses[-1],
            "final_mean_error": (parameters[:2].detach() - target_mean).norm().item(),
            "final_covariance_error": (final_covariance - target_covariance).norm().item(),
            "minimum_covariance_eigenvalue": torch.linalg.eigvalsh(final_covariance).min().item()},
        "charts": [
            {"kind": "line", "title": "Fréchet distance during post-training", "x_label": "Gradient step",
                "x": list(range(0, 201, 4)), "series": [{"name": "EMA population FD", "values": raw_losses[::4]}, {"name": "Current generator FD (independent oracle)", "values": true_losses[::4]}]},
            {"kind": "line", "title": "Normalized training objective (stopped denominator)", "x_label": "Gradient step",
                "x": list(range(0, 201, 4)), "series": [{"name": "Normalized FD", "values": normalized_losses[::4]}]},
            {"kind": "line", "title": "Generated first-coordinate mean", "x_label": "Gradient step",
                "x": list(range(0, 201, 4)), "series": [{"name": "Generator", "values": mean_x[::4]}, {"name": "Reference", "values": [1.] * 51}]},
        ],
    }
