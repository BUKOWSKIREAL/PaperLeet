import math
import torch


def _validate(tau, direction):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be forward or reverse")


def clipped_distillation_loss(student_logits, teacher_logits, mask, tau, direction="forward"):
    """Upper-clip each vocabulary contribution, then sum and average per response."""
    raise NotImplementedError("Implement this function")
