# 教师复用学生上下文长度
import torch


def align_rollout_logits(student_logits, teacher_logits, student_prompt_lengths,
                         teacher_prompt_lengths, response_lengths):
    """Select next-token predictions on the SAME rollout under two contexts."""
    width = int(response_lengths.max().item())
    time = torch.arange(width, device=student_logits.device)
    mask = time[None, :] < response_lengths[:, None]
    rows = torch.arange(student_logits.shape[0], device=student_logits.device)[:, None]
    si = (student_prompt_lengths[:, None] - 1 + time).clamp(max=student_logits.shape[1] - 1)
    ti = (student_prompt_lengths[:, None] - 1 + time).clamp(max=teacher_logits.shape[1] - 1)
    student = student_logits[rows, si]
    teacher = teacher_logits.detach()[rows, ti]
    return (torch.where(mask[..., None], student, 0.),
            torch.where(mask[..., None], teacher, 0.), mask)
