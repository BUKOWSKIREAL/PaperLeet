"""Synthetic head-scale sweep. Teaching adaptation, not GLM-5 training metrics."""
import torch


def run(project):
    # Both heads span the same row space: joint projection must share its scale.
    base = torch.tensor([[1., 0., 0.], [0., 2., 0.]], dtype=torch.float64)
    exponents = list(range(-3, 4))
    split_small, split_fixed, global_small, global_fixed = [], [], [], []
    errors = []
    expected = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=base.dtype)
    with torch.no_grad():
        for exponent in exponents:
            scale = 10. ** exponent
            matrix = torch.cat([scale * base, base])
            actual = project(matrix, 2).reshape(2, 2, 3)
            # Independent closed form for the whole-matrix partial polar factor.
            denominator = (scale * scale + 1) ** .5
            whole = torch.stack([expected * scale / denominator, expected / denominator])
            split_small.append(actual[0].norm().item())
            split_fixed.append(actual[1].norm().item())
            global_small.append(whole[0].norm().item())
            global_fixed.append(whole[1].norm().item())
            errors.append((actual - expected).abs().max().item())
    return {
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/html/2602.15763v2#S2.SS1",
                       "adaptation": "Synthetic two-head scale sweep with exact partial polar factors; not a GLM training benchmark or production Muon implementation."},
        "charts": [{"kind": "line", "x": exponents, "y_label": "Frobenius norm", "title": "Per-head update norm versus input scale", "x_label": "log10(first-head scale)",
                    "series": [{"name": "split: first head", "values": split_small},
                               {"name": "split: second head", "values": split_fixed},
                               {"name": "whole matrix: first head", "values": global_small},
                               {"name": "whole matrix: second head", "values": global_fixed}]}],
        "metrics": {"max_headwise_error": max(errors), "first_head_norm_min": min(split_small),
                    "first_head_norm_max": max(split_small), "whole_matrix_first_norm_min": min(global_small)},
    }
