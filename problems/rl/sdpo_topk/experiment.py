import math
import torch


def run(topk_tail_reverse_kl):
    p, q = [.5, .25, .12, .08, .05], [.2, .2, .1, .2, .3]
    s, t = torch.tensor(p, dtype=torch.float64).log(), torch.tensor(q, dtype=torch.float64).log()
    actual = [topk_tail_reverse_kl(s, t, k).item() for k in range(1, 6)]
    expected = []
    for k in range(1, 6):
        total = sum(p[i] * math.log(p[i] / q[i]) for i in range(k))
        if k < 5:
            total += sum(p[k:]) * math.log(sum(p[k:]) / sum(q[k:]))
        expected.append(total)
    return {"kind": "toy_experiment", "title": "How much information does the tail bin retain?",
            "problem": "rl/sdpo_topk",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2601.20802v2",
                           "anchor": "Appendix A.3: Top-K KL Divergence",
                           "adaptation": "Five-category distributions demonstrate exact coarse-grained KL and information loss. No large-vocabulary memory benchmark is claimed."},
            "metrics": {"maximum_formula_error": max(abs(a - b) for a, b in zip(actual, expected)),
                        "top1_kl": actual[0], "full_kl": expected[-1]},
            "charts": [{"kind": "line", "title": "Student top-k plus tail", "x_label": "k", "x": list(range(1, 6)),
                        "series": [{"name": "Coarsened KL", "values": actual}, {"name": "Full KL", "values": [expected[-1]] * 5}]}]}
