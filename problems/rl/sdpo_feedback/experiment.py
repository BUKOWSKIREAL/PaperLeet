"""Six tiny program attempts, supplied tests, and a categorical feedback teacher."""
import torch


def run(select_feedback, feedback_reverse_kl):
    # Three safe, predefined candidate functions, evaluated on concrete unit tests.
    candidates = [lambda x: x + 1, lambda x: x * 2, lambda x: x * x]
    examples = [[(2, 3), (4, 5)], [(2, 4), (3, 6)], [(2, 4), (3, 9)]]
    question = torch.tensor([0, 0, 1, 1, 2, 2])
    attempts = [1, 0, 0, 1, 0, 1]
    success = torch.tensor([all(candidates[a](x) == y for x, y in examples[q]) for q, a in zip(question.tolist(), attempts)])
    rich = torch.tensor([False, False, False, False, True, False])
    peer, available = select_feedback(question, success, rich)
    teacher = torch.zeros(6, 1, 3, dtype=torch.float64)
    for b in range(6):
        if peer[b] >= 0:
            # Table 2 prioritizes successful solutions, including the attempt itself.
            target = attempts[peer[b].item()]
        elif rich[b]:
            # The supplied toy teacher interprets the rich unit-test examples exactly.
            target = question[b].item()
        else:
            target = 0  # Unavailable rows must never contribute to training.
        teacher[b, 0, target] = 3.
    student = torch.zeros_like(teacher, requires_grad=True)
    history = []
    truth_available = torch.tensor([True, True, True, True, True, False])
    for step in range(41):
        with torch.no_grad():
            probs = student.softmax(-1)[torch.arange(6), 0, question]
            history.append(probs[truth_available].mean().item())
        if step == 40:
            break
        loss = feedback_reverse_kl(student, teacher, torch.ones(6, 1, dtype=torch.bool), available)
        gradient, = torch.autograd.grad(loss, student)
        with torch.no_grad():
            student -= 1.5 * gradient
    unchanged = student.detach()[~truth_available].abs().max().item()
    return {"kind": "toy_experiment", "title": "Unit-test feedback on tiny program attempts",
            "problem": "rl/sdpo_feedback",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2601.20802v2",
                           "anchor": "Section 2, Algorithm 1, and teacher prompt Table 2",
                           "adaptation": "Six predefined program attempts with real unit-test outcomes; a categorical lookup teacher replaces the feedback-conditioned language model. Routing follows Table 2, including successful self-solutions; choosing the lowest-index successful peer is the deterministic exercise convention."},
            "metrics": {"feedback_count": available.sum().item(), "initial_correct_probability": history[0],
                        "final_correct_probability": history[-1], "unavailable_logit_change": unchanged},
            "charts": [{"kind": "line", "title": "Learning on attempts with feedback", "x_label": "Update",
                        "x": list(range(41)), "series": [{"name": "Correct program probability", "values": history}]}]}
