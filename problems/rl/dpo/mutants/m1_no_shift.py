# Bug: 标签未右移
import torch
from torch import Tensor
import torch.nn.functional as F


def sequence_logprob(logits: Tensor, input_ids: Tensor, loss_mask: Tensor) -> Tensor:
    """[B,T,V] predicts next token; shift labels/mask and sum answer log-probs."""
    scores = logits[:, :-1].log_softmax(-1)
    selected = scores.gather(-1, input_ids[:, :-1].unsqueeze(-1)).squeeze(-1)
    return (selected * loss_mask[:, 1:]).sum(-1)


def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor, ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float = 0.1, label_smoothing: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Sequence log-probs [B]; loss scalar and detached implicit rewards [B]."""
    chosen = beta * (pi_chosen - ref_chosen.detach())
    rejected = beta * (pi_rejected - ref_rejected.detach())
    margin = chosen - rejected
    loss = (-(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)).mean()
    return loss, chosen.detach(), rejected.detach()
