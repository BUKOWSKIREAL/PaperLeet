# BUG: 返回标准差而不是方差
"""Terminal-goal scoring and diagonal-Gaussian CEM elite statistics."""
import torch


def _float_tensor(x, ndim, name):
    if not isinstance(x, torch.Tensor) or x.ndim != ndim or not x.numel() or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be finite, nonempty float32/float64 rank {ndim}')


def terminal_cost(rollouts, goal):
    """Score each candidate only at its final latent state: squared L2, not mean MSE."""
    _float_tensor(rollouts, 3, 'rollouts'); _float_tensor(goal, 1, 'goal')
    if goal.shape[0] != rollouts.shape[-1] or goal.dtype != rollouts.dtype or goal.device != rollouts.device:
        raise ValueError('goal must match latent dimension, dtype and device')
    return (rollouts[:, -1] - goal).square().sum(-1)


def cem_refit(candidates, costs, num_elites):
    """Return elite mean and population variance (H,A), plus selected indices (K,)."""
    _float_tensor(candidates, 3, 'candidates'); _float_tensor(costs, 1, 'costs')
    if costs.shape[0] != candidates.shape[0] or costs.dtype != candidates.dtype or costs.device != candidates.device:
        raise ValueError('one matching cost is required per candidate')
    if isinstance(num_elites, bool) or not isinstance(num_elites, int) or not 1 <= num_elites <= candidates.shape[0]:
        raise ValueError('num_elites must be an integer in [1,N]')
    indices = torch.argsort(costs, stable=True)[:num_elites]
    elites = candidates[indices]
    mean = elites.mean(0)
    variance = (elites - mean).square().mean(0)
    return mean, variance.sqrt(), indices
