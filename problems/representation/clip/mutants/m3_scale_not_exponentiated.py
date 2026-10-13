# Bug: 将 log scale 当作 scale
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def contrastive_logits(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Features [B,D]; logit_scale is scalar log inverse temperature, capped at log(100)."""
    images = F.normalize(image_features, dim=-1, eps=1e-12)
    texts = F.normalize(text_features, dim=-1, eps=1e-12)
    scale = logit_scale.clamp(max=math.log(100))
    return scale * (images @ texts.T)


def clip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Average image-to-text and text-to-image cross entropy; diagonal pairs are positive."""
    logits = contrastive_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
