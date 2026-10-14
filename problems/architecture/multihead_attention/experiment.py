"""Two heads read different feature subspaces: a LeetPaper teaching adaptation."""
import math
import torch


def run(multi_head_attention):
    dtype = torch.float64
    query = torch.tensor([[[1., 0., 0., 1.], [0., 1., 1., 0.]]], dtype=dtype)
    key = torch.tensor([[[5., 0., 0., 0.], [0., 5., 5., 0.], [0., 0., 0., 5.]]], dtype=dtype)
    value = torch.tensor([[[2., -1., 5., 4.], [7., 3., -2., 6.], [-4., 8., 1., 9.]]], dtype=dtype)
    identity = torch.eye(4, dtype=dtype)
    output_projection = torch.tensor([[1., 0., .5, 0.], [0., 1., 0., -.5],
                                      [0., 0., 1., 0.], [.25, 0., 0., 1.]], dtype=dtype)
    output, weights = multi_head_attention(query, key, value, identity, identity,
                                           identity, output_projection, 2)
    # Analytic weights from the constructed logits; independent of learner code.
    high = math.exp(5 / math.sqrt(2))
    expected_weights = torch.tensor([[[[high, 1., 1.], [1., high, 1.]],
                                       [[1., 1., high], [1., high, 1.]]]], dtype=dtype) / (high + 2)
    expected_heads = [expected_weights[0, h] @ value[0, :, 2*h:2*h+2] for h in range(2)]
    expected = torch.cat(expected_heads, dim=-1)[None] @ output_projection
    actual_values = output[0, 0].detach().tolist()
    return {
        "provenance": "LeetPaper teaching adaptation of Transformer Section 3.2.2; fixed projections, no training.",
        "metrics": {"maximum_error": (output - expected).abs().max().item(),
                    "head_weight_error": (weights - expected_weights).abs().max().item(),
                    "head_separation_l1": (weights[0, 0, 0] - weights[0, 1, 0]).abs().sum().item()},
        "charts": [
            {"title": "Which key does each head read?", "kind": "bar", "x": [0, 1, 2],
             "x_label": "Key position", "series": [
                 {"name": "Head 1", "values": weights[0, 0, 0].detach().tolist()},
                 {"name": "Head 2", "values": weights[0, 1, 0].detach().tolist()}]},
            {"title": "First query after output projection", "kind": "bar", "x": [0, 1, 2, 3],
             "x_label": "Output feature", "series": [
                 {"name": "Implementation", "values": actual_values},
                 {"name": "Analytic target", "values": expected[0, 0].tolist()}]},
        ],
    }
