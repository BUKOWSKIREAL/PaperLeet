# BUG: 停止未来编码器目标梯度
"""LeWM joint objective and a one-context-state latent rollout."""
import math
import torch


def _tensor(x, ndim, name):
    if not isinstance(x, torch.Tensor) or x.ndim != ndim or x.dtype not in (torch.float32, torch.float64) or not x.numel() or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be a finite nonempty float32/float64 rank-{ndim} tensor')


def lewm_objective(embeddings, actions, predictor, regularizer, weight=0.1):
    """Return (total, prediction, regularization); both embedding paths retain gradients."""
    _tensor(embeddings, 3, 'embeddings'); _tensor(actions, 3, 'actions')
    if embeddings.shape[1] < 2 or actions.shape[:2] != (embeddings.shape[0], embeddings.shape[1] - 1) or actions.dtype != embeddings.dtype or actions.device != embeddings.device:
        raise ValueError('actions must align with the T-1 transitions')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    predictions = predictor(embeddings[:, :-1], actions)
    if predictions.shape != embeddings[:, 1:].shape:
        raise ValueError('predictor must return (B,T-1,D)')
    prediction = (predictions - embeddings[:, 1:].detach()).square().mean()
    regularization = regularizer(embeddings.transpose(0, 1))
    if not isinstance(regularization, torch.Tensor) or regularization.ndim != 0:
        raise ValueError('regularizer must return a scalar tensor')
    return prediction + weight * regularization, prediction, regularization


def latent_rollout(initial, actions, transition):
    """Return z_1...z_H as (B,H,D), recursively feeding predictions to an N=1 model."""
    _tensor(initial, 2, 'initial'); _tensor(actions, 3, 'actions')
    if initial.shape[0] != actions.shape[0] or initial.dtype != actions.dtype or initial.device != actions.device:
        raise ValueError('initial and actions must share batch, dtype and device')
    state, states = initial, []
    for t in range(actions.shape[1]):
        state = transition(state, actions[:, t])
        if state.shape != initial.shape:
            raise ValueError('transition must preserve (B,D)')
        states.append(state)
    return torch.stack(states, dim=1)
