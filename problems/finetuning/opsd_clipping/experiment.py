"""Small categorical example: upper-clipping signed vocabulary contributions."""
import math
import torch


def run(clipped_distillation_loss):
    p, q = [.03, .17, .8], [.8, .15, .05]
    s, t = torch.tensor(p, dtype=torch.float64).log()[None, None], torch.tensor(q, dtype=torch.float64).log()[None, None]
    thresholds = [.01, .03, .1, .3, 1., 3.]
    mask = torch.ones(1, 1, dtype=torch.bool)
    series, maximum_error = [], 0.
    for direction in ("forward", "reverse"):
        terms = [qt * math.log(qt / ps) if direction == "forward" else ps * math.log(ps / qt)
                 for ps, qt in zip(p, q)]
        actual = [clipped_distillation_loss(s, t, mask, tau, direction).item() for tau in thresholds]
        expected = [sum(min(term, tau) for term in terms) for tau in thresholds]
        maximum_error = max(maximum_error, max(abs(a - b) for a, b in zip(actual, expected)))
        series.append({"name": direction + " pointwise clipped", "values": actual})
        series.append({"name": direction + " aggregate clipped (comparison)", "values": [min(sum(terms), tau) for tau in thresholds]})
    return {"kind": "toy_experiment", "title": "Pointwise versus aggregate clipping",
            "problem": "finetuning/opsd_clipping",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2601.18734v3",
                           "anchor": "Section 3.2: Per-Token Pointwise Divergence Clipping",
                           "adaptation": "Three-category distributions isolate the clipping rule. Not a reproduction of the paper's Qwen experiment."},
            "metrics": {"maximum_formula_error": maximum_error, "small_tau_forward_loss": series[0]["values"][0]},
            "charts": [{"kind": "line", "title": "Signed loss versus clipping threshold", "x_label": "tau",
                        "x": thresholds, "series": series}]}
