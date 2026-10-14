"""Teaching adaptation: fixed slices expose shape beyond first and second moments."""
import torch


def _oracle(features, directions, t):
    # Full symmetric-grid complex ECF oracle; the learner uses positive-grid real arithmetic.
    full_t = torch.cat((-t[1:].flip(0), t))
    cf = torch.exp(1j * (features @ directions).unsqueeze(-1) * full_t).mean(0)
    phi = torch.exp(-full_t.square() / 2)
    return (features.shape[0] * torch.trapezoid((cf - phi).abs().square() * phi, full_t, dim=-1)).mean()


def run(epps_pulley, sigreg_loss):
    dtype = torch.float64
    g = torch.Generator().manual_seed(17)
    features = (torch.randn(64, 2, generator=g, dtype=dtype) * torch.tensor([.3, 1.7], dtype=dtype) + torch.tensor([1., -.5], dtype=dtype)).requires_grad_()
    angles = torch.arange(16, dtype=dtype) * torch.pi / 16
    directions = torch.stack((angles.cos(), angles.sin()))
    t = torch.linspace(0, 3, 17, dtype=dtype)
    optimizer = torch.optim.Adam([features], lr=.05)
    losses, means, stds = [], [], []
    for step in range(101):
        loss = sigreg_loss(features, directions, t)
        losses.append(loss.item()); means.append(features.mean(0).norm().item())
        stds.append(features.std(0, correction=0).min().item())
        if step < 100:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Identical population mean=0, variance=1; different distribution shapes.
    quantiles = torch.erfinv(2 * (torch.arange(64, dtype=dtype) + .5) / 64 - 1) * 2 ** .5
    normal = quantiles / quantiles.std(correction=0)
    binary = torch.tensor([-1., 1.], dtype=dtype).repeat(32)
    scores = epps_pulley(torch.stack((normal, binary), dim=1), t)
    projected = features.detach() @ directions[:, :1]
    ecf = (projected * t).cos().mean(0)
    oracle = _oracle(features.detach(), directions, t)
    return {
        "kind": "toy_experiment", "title": "SIGReg: beyond matching moments", "problem": "representation/sigreg",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2511.08544",
            "anchor": "Sections 4.2.3 and 5.1: sliced Epps-Pulley regularization",
            "adaptation": "Optimize 64 free 2D embeddings using 16 fixed unit directions, and compare standardized Gaussian quantiles with balanced +/-1 samples sharing population mean and variance. This is a finite-grid illustration, not LeJEPA pretraining or a guarantee that fixed slices identify all distributions."},
        "metrics": {"initial_loss": losses[0], "final_loss": losses[-1], "mean_norm": means[-1], "minimum_std": stds[-1],
            "oracle_absolute_error": abs(oracle.item() - losses[-1]), "normal_quantile_statistic": scores[0].item(), "binary_statistic": scores[1].item()},
        "charts": [
            {"kind": "line", "title": "Submitted SIGReg objective", "x_label": "Step", "x": list(range(0, 101, 2)),
                "series": [{"name": "SIGReg", "values": losses[::2]}]},
            {"kind": "line", "title": "Embedding geometry (independent diagnostics)", "x_label": "Step", "x": list(range(0, 101, 2)),
                "series": [{"name": "Mean norm", "values": means[::2]}, {"name": "Minimum feature std", "values": stds[::2]}]},
            {"kind": "bar", "title": "Same mean and variance, different Epps-Pulley statistic", "x_label": "Distribution", "x": ["Gaussian quantiles", "Balanced +/-1"],
                "series": [{"name": "Statistic", "values": scores.detach().tolist()}]},
            {"kind": "line", "title": "First projection: real characteristic function", "x_label": "Frequency t", "x": t.tolist(),
                "series": [{"name": "Optimized", "values": ecf.tolist()}, {"name": "Standard normal", "values": torch.exp(-t.square() / 2).tolist()}]},
        ],
    }
