import math
import torch
from torch import Tensor
import torch.nn.functional as F


def contrastive_logits(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Required: L2-normalize [B,D] feature rows (eps=1e-12), dot products times
    exp(logit_scale), capped at 100. logit_scale is a scalar Tensor in log space.
    Return [B,B]; keep gradients through both encoders and unclipped temperature."""
    raise NotImplementedError()


def clip_loss(image_features: Tensor, text_features: Tensor, logit_scale: Tensor) -> Tensor:
    """Required: diagonal image/text pairs are positive; all other pairs negative.
    Average cross entropy over rows and over columns to give a scalar loss.
    Call contrastive_logits; do not detach features or temperature."""
    raise NotImplementedError()
