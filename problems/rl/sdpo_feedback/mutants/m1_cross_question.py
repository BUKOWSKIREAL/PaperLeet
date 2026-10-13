# BUG: Ignore question identity when finding a successful solution.
import torch


def select_feedback(question_ids, success, rich_feedback):
    """Return solution index (-1 if absent) and usable-feedback availability."""
    count = len(question_ids)
    rows = torch.arange(count, device=question_ids.device)
    eligible = torch.ones((count, count), dtype=torch.bool, device=question_ids.device) & success[None, :]
    candidate = torch.where(eligible, rows[None, :], count).min(-1).values
    solution = torch.where(success, rows, candidate)
    solution = torch.where(solution < count, solution, -1)
    return solution, (solution >= 0) | rich_feedback


def feedback_reverse_kl(student_logits, teacher_logits, token_mask, feedback_available):
    """Average reverse KL over tokens of attempts with usable teacher feedback."""
    mask = token_mask & feedback_available[:, None]
    s, t = student_logits.log_softmax(-1), teacher_logits.detach().log_softmax(-1)
    loss = (s.exp() * (s - t)).sum(-1)
    return torch.where(mask, loss, 0.).sum() / mask.sum().clamp_min(1)
