# 从教师而非学生选择 top-k
import torch


def topk_tail_reverse_kl(student_logits, teacher_logits, k):
    """Coarsen the vocabulary into student top-k tokens and one complementary bin."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= student_logits.shape[-1]:
        raise ValueError("k must be an integer in [1, vocabulary_size]")
    s = student_logits.log_softmax(-1)
    t = teacher_logits.detach().log_softmax(-1)
    indices = t.topk(k, dim=-1).indices
    sp, tp = s.gather(-1, indices), t.gather(-1, indices)
    result = (sp.exp() * (sp - tp)).sum(-1)
    if k < s.shape[-1]:
        selected = torch.zeros_like(s, dtype=torch.bool).scatter(-1, indices, True)
        sl = s.masked_fill(selected, -torch.inf).logsumexp(-1)
        tl = t.masked_fill(selected, -torch.inf).logsumexp(-1)
        result = result + sl.exp() * (sl - tl)
    return result
