"""PaperLeet quadratic teaching experiment, not the original CIFAR/ImageNet evaluation."""
import torch


def run(step_function):
    parameter = torch.tensor([4., -3.], dtype=torch.float64)
    first, second = torch.zeros_like(parameter), torch.zeros_like(parameter)
    target = torch.tensor([0.25, -0.5], dtype=parameter.dtype)
    curvature = torch.tensor([1., 16.], dtype=parameter.dtype)
    losses, trajectory = [], []
    for step in range(1, 201):
        delta = parameter - target
        losses.append((0.5 * curvature * delta.square()).sum().item())
        trajectory.append(parameter.detach().tolist())
        parameter, first, second = step_function(parameter, curvature * delta, first, second, step,
                                                  lr=.08, betas=(.8, .95), weight_decay=.1)
    final_loss = (0.5 * curvature * (parameter - target).square()).sum().item()
    return {"provenance": "PaperLeet teaching adaptation of AdamW Algorithm 2",
            "steps": 200, "losses": losses, "trajectory": trajectory,
            "target": target.tolist(), "final_parameter": parameter.detach().tolist(),
            "final_loss": final_loss, "initial_loss": losses[0]}
