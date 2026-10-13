import math
import torch


def _validate(tau, direction):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be forward or reverse")


def clipped_distillation_loss(student_logits, teacher_logits, mask, tau, direction="forward"):
    """Upper-clip each vocabulary contribution, then sum and average per response."""
    _validate(tau, direction)
    s = student_logits.log_softmax(-1)
    t = teacher_logits.detach().log_softmax(-1)
    pointwise = t.exp() * (t - s) if direction == "forward" else s.exp() * (s - t)
    per_token = pointwise.clamp(max=tau).sum(-1)
    per_response = torch.where(mask, per_token, 0.).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_response.mean()
