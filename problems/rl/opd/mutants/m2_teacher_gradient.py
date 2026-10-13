# Intentional bug: Allow gradients into the teacher.
import torch
from torch import Tensor


def sample_student(transition_logits: Tensor, batch_size: int, length: int,
                   generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Fixed-length Markov language model: row V is BOS, rows 0..V-1 are tokens."""
    if transition_logits.ndim != 2 or transition_logits.shape[0] != transition_logits.shape[1] + 1:
        raise ValueError('transition_logits must have shape [vocab + 1, vocab]')
    if batch_size < 1 or length < 1:
        raise ValueError('batch_size and length must be positive')
    with torch.no_grad():
        vocab = transition_logits.shape[-1]
        previous = torch.full((batch_size,), vocab, device=transition_logits.device, dtype=torch.long)
        tokens, prefixes = [], []
        for _ in range(length):
            prefixes.append(previous)
            probabilities = transition_logits[previous].softmax(-1)
            previous = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            tokens.append(previous)
        return torch.stack(tokens, -1), torch.stack(prefixes, -1)


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor,
                       direction: str = 'reverse') -> Tensor:
    """Full-vocabulary KL at student-generated prefixes; valid-token average."""
    if direction not in ('reverse', 'forward'):
        raise ValueError('direction must be reverse or forward')
    active = mask.bool().unsqueeze(-1)
    student = torch.where(active, student_logits, torch.zeros_like(student_logits)).log_softmax(-1)
    teacher = torch.where(active, teacher_logits, torch.zeros_like(teacher_logits)).log_softmax(-1)
    if direction == 'reverse':
        divergence = (student.exp() * (student - teacher)).sum(-1)
    else:
        divergence = (teacher.exp() * (teacher - student)).sum(-1)
    return (divergence * mask).sum() / mask.sum().clamp_min(1)
