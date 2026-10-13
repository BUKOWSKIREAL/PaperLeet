"""A supplied right-padded, two-context categorical training experiment."""
import torch


def run(align_rollout_logits):
    lengths = torch.tensor([3, 1, 2])
    sp, tp = torch.tensor([2, 4, 1]), torch.tensor([5, 7, 4])
    targets = [[0, 1, 2], [2], [1, 0]]
    student = torch.zeros(3, 6, 3, dtype=torch.float64, requires_grad=True)
    teacher = torch.full((3, 10, 3), -4., dtype=student.dtype)
    # Distracting prefix predictions make the causal offset observable.
    teacher[:, :, 2] = 4.
    for b, sequence in enumerate(targets):
        for j, target in enumerate(sequence):
            teacher[b, tp[b] - 1 + j] = -4.
            teacher[b, tp[b] - 1 + j, target] = 4.
    history = []
    for step in range(41):
        with torch.no_grad():
            history.append(sum(student[b, sp[b] - 1 + j].softmax(-1)[target].item()
                               for b, seq in enumerate(targets) for j, target in enumerate(seq)) / 6)
        if step == 40:
            break
        s, t, mask = align_rollout_logits(student, teacher, sp, tp, lengths)
        loss = (-(t.softmax(-1) * s.log_softmax(-1)).sum(-1) * mask).sum() / mask.sum()
        grad, = torch.autograd.grad(loss, student)
        with torch.no_grad():
            student -= 2. * grad
    return {"kind": "toy_experiment", "title": "Same rollout, different context lengths",
            "problem": "finetuning/opsd_alignment",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2601.18734v3",
                           "anchor": "Section 3.2 and Algorithm 1",
                           "adaptation": "Right-padded categorical logits replace LLM calls; privileged targets are supplied. This tests context alignment, not reasoning benchmark performance."},
            "metrics": {"initial_correct_probability": history[0], "final_correct_probability": history[-1]},
            "charts": [{"kind": "line", "title": "Probability at the actual response positions", "x_label": "Update",
                        "x": list(range(41)), "series": [{"name": "Correct token", "values": history}]}]}
