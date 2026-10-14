"""LeetPaper ratio diagnostic and stale-rollout two-action bandit teaching toy."""
import torch


def run(weight_fn, loss_fn):
    ratios = torch.linspace(.25, 1.75, 61, dtype=torch.float64)
    scores = (ratios.log() - 1).reshape(-1, 1).requires_grad_()
    rollout = torch.full_like(scores, -1.)
    mask = torch.ones_like(scores)
    weights = weight_fn(scores, rollout, mask, .5, .5)
    loss = loss_fn(scores, rollout, torch.ones(len(ratios), dtype=scores.dtype), mask, .5, .5)
    gradients = torch.autograd.grad(loss, scores)[0].flatten()
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    rewards, steps = [], list(range(101))
    rewards.append((logits.softmax(0)[1] * 2 - 1).item())
    for step in range(100):
        if step % 5 == 0:
            behavior = logits.detach().softmax(0)
            # Deterministic finite rollout sample; behavior is five updates stale at most.
            count = max(1, min(127, round(behavior[1].item() * 128)))
            actions = torch.tensor([0] * (128 - count) + [1] * count)
            behavior_logp = behavior.log()[actions, None]
            advantages = actions.to(logits.dtype) * 2 - 1
        current = logits.log_softmax(0)[actions, None]
        value = loss_fn(current, behavior_logp, advantages, torch.ones_like(current), .5, .5)
        gradient = torch.autograd.grad(value, logits)[0]
        logits = (logits - .15 * gradient).detach().requires_grad_()
        rewards.append((logits.softmax(0)[1] * 2 - 1).item())
    expected = ratios * ((ratios > .5) & (ratios < 1.5))
    return {
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2602.15763v2#S4.SS1",
                       "adaptation": "Synthetic ratio sweep and two-action CPU bandit with rollouts refreshed every five steps; no coding-agent or large-model benchmark reproduction."},
        "charts": [
            {"kind": "line", "title": "Masking versus clamping importance weights", "x_label": "current / rollout ratio", "y_label": "weight", "x": ratios.tolist(),
             "series": [{"name": "submitted double-sided mask", "values": weights.detach().flatten().tolist()},
                        {"name": "clamped comparison", "values": ratios.clamp(.5, 1.5).tolist()}]},
            {"kind": "line", "title": "Measured score-function gradient", "x_label": "current / rollout ratio", "y_label": "d loss / d logp", "x": ratios.tolist(),
             "series": [{"name": "submitted gradient", "values": gradients.tolist()},
                        {"name": "analytic target", "values": (-expected / len(ratios)).tolist()}]},
            {"kind": "line", "title": "Stale-rollout bandit", "x_label": "update", "y_label": "expected reward", "x": steps,
             "series": [{"name": "submitted policy", "values": rewards}]}],
        "metrics": {"initial_reward": rewards[0], "final_reward": rewards[-1],
                    "max_gradient_error": (gradients + expected / len(ratios)).abs().max().item(),
                    "max_weight_error": (weights.flatten().detach() - expected).abs().max().item(),
                    "rollout_refresh_interval": 5},
    }
