import torch


def select_feedback(question_ids, success, rich_feedback):
    """Return solution index (-1 if absent) and usable-feedback availability."""
    raise NotImplementedError("Implement select_feedback")


def feedback_reverse_kl(student_logits, teacher_logits, token_mask, feedback_available):
    """Average reverse KL over tokens of attempts with usable teacher feedback."""
    raise NotImplementedError("Implement feedback_reverse_kl")
