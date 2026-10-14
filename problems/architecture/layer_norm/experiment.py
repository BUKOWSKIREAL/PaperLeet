"""Teaching adaptation: token-local Add & Norm under varied offsets and scales."""
import math
import torch


def run(layer_norm, post_norm_residual):
    scales = torch.tensor([.02, .05, .2, .5, 1., 3., 10., 30.], dtype=torch.float64)
    offsets = torch.tensor([-12., 4., 0., 18., -5., 0., 40., -20.], dtype=scales.dtype)
    pattern = torch.tensor([-2., -1., 0., 1., 2., 4.], dtype=scales.dtype)
    x = (scales[:, None] * pattern + offsets[:, None])[None]
    sublayer = .15 * torch.tanh(x)
    summed = x + sublayer
    weight, bias = torch.ones(6, dtype=x.dtype), torch.zeros(6, dtype=x.dtype)
    eps = 1e-5
    output = post_norm_residual(x, sublayer, weight, bias, eps)
    # Independent scalar oracle: no reuse of the submitted normalization code.
    expected = []
    expected_std = []
    for row in summed[0].tolist():
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        expected.append([(value - mean) / math.sqrt(variance + eps) for value in row])
        expected_std.append(math.sqrt(variance / (variance + eps)))
    expected = torch.tensor(expected, dtype=x.dtype)[None]
    changed_x, changed_sublayer = x.clone(), sublayer.clone()
    changed_x[:, 1:] = changed_x[:, 1:] * 500 + 1000
    changed_sublayer[:, 1:] -= 700
    changed_output = post_norm_residual(changed_x, changed_sublayer, weight, bias, eps)
    normalized_sum = layer_norm(summed, weight, bias, eps)
    return {
        "provenance": "LeetPaper teaching adaptation of Transformer Section 3.1; dropout disabled; no training benchmark.",
        "charts": [
            {"title": "Per-token mean before and after Add & Norm", "kind": "line", "x": list(range(8)), "x_label": "token", "y_label": "mean",
             "series": [{"name": "residual sum", "values": summed.mean(-1).flatten().tolist()},
                        {"name": "Add & Norm", "values": output.mean(-1).flatten().detach().tolist()}]},
            {"title": "Per-token standard deviation", "kind": "line", "x": list(range(8)), "x_label": "token", "y_label": "population std",
             "series": [{"name": "residual sum", "values": summed.std(-1, correction=0).flatten().tolist()},
                        {"name": "Add & Norm", "values": output.std(-1, correction=0).flatten().detach().tolist()},
                        {"name": "analytical target", "values": expected_std}]},
        ],
        "metrics": {
            "maximum_output_error": (output - expected).abs().max().item(),
            "maximum_mean_magnitude": output.mean(-1).abs().max().item(),
            "other_token_influence": (changed_output[:, 0] - output[:, 0]).abs().max().item(),
            "composition_error": (output - normalized_sum).abs().max().item(),
        },
    }
