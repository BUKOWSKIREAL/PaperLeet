"""Teaching adaptation: recover spread from small, correlated, nonconstant embeddings."""
import torch


def run(vicreg_terms, vicreg_loss):
    angle = torch.arange(16, dtype=torch.float64) * (2 * torch.pi / 16)
    base = torch.stack((angle.cos(), angle.sin()), 1)
    x = (.08 * base @ base.new_tensor([[1., .9], [0., .25]])).requires_grad_()
    y = (x.detach() + .015 * base.roll(3, 0)).requires_grad_()
    optimizer = torch.optim.Adam([x, y], lr=.04)
    losses, stds, correlations = [], [], []
    for step in range(101):
        loss = vicreg_loss(x, y, 5., 5., 1.)
        with torch.no_grad():
            covariance = torch.cov(x.T)
            losses.append(loss.item())
            stds.append(x.std(0).min().item())
            correlations.append(covariance[0, 1].abs().item())
        if step < 100:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    final_terms = vicreg_terms(x, y)
    zero = torch.zeros_like(x)
    collapsed_terms = vicreg_terms(zero, zero)
    return {
        "kind": "toy_experiment", "title": "VICReg: spread and redundancy", "problem": "representation/vicreg",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2105.04906v3",
            "anchor": "Section 4.1: variance, invariance and covariance regularization",
            "adaptation": "Optimize two sets of 16 paired 2D embeddings from small, correlated but nonconstant values. This controlled geometry experiment is not image encoder pretraining; exact zero collapse is stationary and is not used as initialization."},
        "metrics": {"initial_loss": losses[0], "final_loss": losses[-1], "minimum_std": stds[-1],
            "off_diagonal_abs": correlations[-1], "final_invariance": final_terms[0].item(),
            "collapsed_invariance": collapsed_terms[0].item(), "collapsed_variance": collapsed_terms[1].item()},
        "charts": [
            {"kind": "line", "title": "Submitted VICReg objective", "x_label": "Step", "x": list(range(0, 101, 2)),
                "series": [{"name": "Loss", "values": losses[::2]}]},
            {"kind": "line", "title": "Embedding geometry (independent diagnostics)", "x_label": "Step", "x": list(range(0, 101, 2)),
                "series": [{"name": "Minimum feature std", "values": stds[::2]}, {"name": "Absolute off-diagonal covariance", "values": correlations[::2]}]},
            {"kind": "bar", "title": "Exact collapse has zero agreement error", "x_label": "Term", "x": ["Invariance", "Variance", "Covariance"],
                "series": [{"name": "Collapsed", "values": [v.item() for v in collapsed_terms]}, {"name": "Optimized", "values": [v.item() for v in final_terms]}]},
        ],
    }
