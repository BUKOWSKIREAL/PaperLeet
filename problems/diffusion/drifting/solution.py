import math
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def drift_field(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor:
    """Official toy's doubly normalized distance kernel; generated self-pairs masked."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    if len(generated) == 0 or len(positive) == 0:
        raise ValueError('both sample sets must be nonempty')
    n = len(generated)
    targets = torch.cat((generated, positive))
    distance = torch.cdist(generated, targets)
    mask = torch.cat((torch.eye(n, dtype=torch.bool, device=generated.device), torch.zeros(n, len(positive), dtype=torch.bool, device=generated.device)), dim=1)
    log_kernel = (-distance / temperature).masked_fill(mask, -torch.inf)
    row_mass = torch.logsumexp(log_kernel, dim=1, keepdim=True)
    column_mass = torch.logsumexp(log_kernel, dim=0, keepdim=True)
    # A column containing only the masked self-pair has zero affinity.
    column_mass = torch.where(torch.isfinite(column_mass), column_mass, torch.zeros_like(column_mass))
    log_normalizer = .5 * (row_mass + column_mass)
    affinity = torch.exp(log_kernel - log_normalizer)
    negative, positive_weight = affinity[:, :n], affinity[:, n:]
    attraction = (positive_weight @ positive) * negative.sum(1, keepdim=True)
    repulsion = (negative @ generated) * positive_weight.sum(1, keepdim=True)
    return attraction - repulsion


def drifting_loss(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor:
    """MSE(g, stopgrad(g + V)); gradients flow only through the left-hand g."""
    field = drift_field(generated.detach(), positive.detach(), temperature)
    target = (generated + field).detach()
    return (generated - target).square().mean()
