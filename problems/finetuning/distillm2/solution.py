import math
import torch
from torch import Tensor


def _coefficient(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("coefficients must be in [0, 1]")


def _log_pair(teacher_logits, student_logits, alpha):
    _coefficient(alpha)
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim < 1:
        raise ValueError("teacher/student logits must have equal nonempty vocabulary shapes")
    if teacher_logits.shape[-1] == 0:
        raise ValueError("vocabulary must not be empty")
    return teacher_logits.detach().log_softmax(-1), student_logits.log_softmax(-1)


def _log_mix(log_a, log_b, alpha):
    if alpha == 0:
        return log_b
    if alpha == 1:
        return log_a + log_b * 0.0
    return torch.logaddexp(log_a + math.log(alpha), log_b + math.log1p(-alpha))


def skew_kl(teacher_logits: Tensor, student_logits: Tensor, alpha: float) -> Tensor:
    """Per-position SKL(p || alpha*p + (1-alpha)*q), freezing the teacher."""
    lp, lq = _log_pair(teacher_logits, student_logits, alpha)
    lm = _log_mix(lp, lq, alpha)
    return (lp.exp() * (lp - lm)).sum(-1)


def skew_reverse_kl(teacher_logits: Tensor, student_logits: Tensor, alpha: float) -> Tensor:
    """Per-position SRKL(q || (1-alpha)*p + alpha*q), freezing the teacher."""
    lp, lq = _log_pair(teacher_logits, student_logits, alpha)
    lm = _log_mix(lq, lp, alpha)
    return (lq.exp() * (lq - lm)).sum(-1)


def distillm2_loss(teacher_t: Tensor, student_t: Tensor, mask_t: Tensor,
                  teacher_s: Tensor, student_s: Tensor, mask_s: Tensor,
                  alpha_t: float = 0.1, alpha_s: float = 0.1,
                  beta: float = 0.5) -> Tensor:
    """Eq. 2 reduction: sum response-token divergences, divide by 2*batch."""
    _coefficient(beta)
    for teacher, student, mask in ((teacher_t, student_t, mask_t), (teacher_s, student_s, mask_s)):
        if teacher.ndim != 3 or teacher.shape != student.shape or mask.shape != teacher.shape[:-1]:
            raise ValueError("expected paired [B,T,V] logits and [B,T] masks")
        if mask.dtype != torch.bool or teacher.shape[0] == 0:
            raise ValueError("masks must be boolean and batch must be nonempty")
    if teacher_t.shape[0] != teacher_s.shape[0]:
        raise ValueError("teacher and student response batches must be paired")
    t = skew_kl(teacher_t, student_t, alpha_t).masked_fill(~mask_t, 0).sum()
    s = skew_reverse_kl(teacher_s, student_s, alpha_s).masked_fill(~mask_s, 0).sum()
    return ((1 - beta) * t + beta * s) / (2 * teacher_t.shape[0])
