# BUG: Normalize across patches as well as features.
"""I-JEPA's released training objective and momentum-target operations."""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def _features(x, ndim=None):
    if not isinstance(x, Tensor) or (ndim is not None and x.ndim != ndim) or not x.numel() or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError('expected finite, nonempty float32/float64 tensors')


def _pair(a, b):
    _features(a); _features(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('tensor shape, dtype and device must match')


def _check_targets(features, masks, num_context, eps):
    _features(features, 3)
    if not isinstance(num_context, int) or isinstance(num_context, bool) or num_context < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('positive context count and finite epsilon required')
    if not isinstance(masks, (list, tuple)) or not masks:
        raise ValueError('at least one target block required')
    size = None
    for mask in masks:
        if not isinstance(mask, Tensor) or mask.ndim != 2 or mask.dtype != torch.int64 or mask.device != features.device or mask.shape[0] != features.shape[0] or mask.shape[1] < 1:
            raise ValueError('each mask must be int64 [B,M] on the feature device')
        if size is not None and mask.shape[1] != size:
            raise ValueError('target blocks must have equal lengths')
        if (mask < 0).any() or (mask >= features.shape[1]).any():
            raise ValueError('target indices out of range')
        size = mask.shape[1]


def masked_targets(features: Tensor, masks: list[Tensor], num_context: int = 1, eps: float = 1e-5) -> Tensor:
    """Return detached normalized targets [K*C*B,M,D], ordered by block/context/batch."""
    _check_targets(features, masks, num_context, eps)
    normalized = F.layer_norm(features.detach(), features.shape[1:], eps=eps)
    blocks = []
    for mask in masks:
        block = normalized.gather(1, mask[..., None].expand(-1, -1, features.shape[-1]))
        blocks.append(block.repeat(num_context, 1, 1))
    return torch.cat(blocks, dim=0)


def masked_prediction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Released I-JEPA code convention: smooth L1 with beta=1, mean over all entries."""
    _pair(prediction, target)
    return F.smooth_l1_loss(prediction, target.detach(), beta=1., reduction='mean')


def ema_update(target: Tensor, online: Tensor, momentum: float) -> Tensor:
    """Functional, detached equivalent of the author's in-place parameter EMA."""
    _pair(target, online)
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError('momentum must be finite and in [0,1]')
    return momentum * target.detach() + (1 - momentum) * online.detach()
