import torch


def align_rollout_logits(student_logits, teacher_logits, student_prompt_lengths,
                         teacher_prompt_lengths, response_lengths):
    """Select next-token predictions on the SAME rollout under two contexts."""
    raise NotImplementedError("Implement this function")
