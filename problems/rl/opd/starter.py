import torch
from torch import Tensor

def sample_student(transition_logits: Tensor, batch_size: int, length: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Fixed-length Markov language model: row V is BOS, rows 0..V-1 are tokens."""
    raise NotImplementedError('Implement sample_student')

def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, direction: str='reverse') -> Tensor:
    """Full-vocabulary KL at student-generated prefixes; valid-token average."""
    raise NotImplementedError('Implement distillation_loss')
