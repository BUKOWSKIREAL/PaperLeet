"""Teaching adaptation: optimize a fixed minibatch of forward-process velocity predictions."""
import torch


def run(implicit_velocities, nft_loss):
    dtype = torch.float64
    rewards = torch.tensor([0., .25, .5, .75, 1.], dtype=dtype)
    clean = torch.tensor([[1., .5]], dtype=dtype).expand(5, -1)
    noise = torch.tensor([[-.5, .2]], dtype=dtype).expand_as(clean)
    time = .6
    noisy = (1 - time) * clean + time * noise
    target = noise - clean  # Paper's data-to-noise time convention.
    old = torch.zeros_like(target)
    beta = .6
    expected = old + (2 * rewards[:, None] - 1) * (target - old) / beta
    current = old.clone().requires_grad_()
    losses, history = [], []
    for step in range(61):
        loss = nft_loss(current, old, target, rewards, beta)
        losses.append(loss.item())
        history.append(current[:, 0].detach().tolist())
        if step < 60:
            gradient, = torch.autograd.grad(loss, current)
            current = (current - 2 * gradient).detach().requires_grad_()
    with torch.no_grad():
        positive, negative = implicit_velocities(current, old, beta)
        ideal_loss = ((1 - (2 * rewards - 1).square()) * (target - old).square().sum(-1)).mean()
    return {
        "kind": "toy_experiment", "title": "DiffusionNFT: positive, negative and neutral forward targets",
        "problem": "diffusion/diffusion_nft",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2509.16117",
                       "anchor": "Section 3.2 Eq. (5); Section 3.3 Algorithm 1",
                       "adaptation": "A fixed five-example 2D forward-process minibatch with directly optimized velocity outputs and an analytic quadratic optimum. Not a full online rollout or image reward benchmark."},
        "metrics": {"initial_loss": losses[0], "final_loss": losses[-1],
                    "optimal_loss": ideal_loss.item(), "optimum_max_error": (current.detach() - expected).abs().max().item(),
                    "neutral_velocity_norm": current[2].detach().norm().item(),
                    "positive_endpoint_error": (positive[-1] - target[-1]).norm().item(),
                    "negative_endpoint_error": (negative[0] - target[0]).norm().item(),
                    "forward_path_reconstruction_error": (noisy - time * target - clean).abs().max().item()},
        "charts": [
            {"kind": "line", "title": "Dual flow-matching objective", "x_label": "Gradient step",
             "x": list(range(len(losses))), "series": [
                 {"name": "Submitted loss", "values": losses},
                 {"name": "Analytic minimum", "values": [ideal_loss.item()] * len(losses)}]},
            {"kind": "line", "title": "Current velocity by optimality probability", "x_label": "Gradient step",
             "x": list(range(len(history))), "series": [
                 {"name": f"r = {r:g}", "values": [row[i] for row in history]} for i, r in enumerate(rewards.tolist())]}],
    }
