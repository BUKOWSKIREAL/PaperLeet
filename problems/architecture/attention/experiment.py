"""LeetPaper teaching adaptation: masked prefix averaging, not a paper experiment."""
import torch


def run(attention):
    values = torch.tensor([2., -1., 4., 7., -2., 5.], dtype=torch.float64)
    q = torch.zeros(1, 1, len(values), 3, dtype=values.dtype)
    mask = torch.ones(len(values), len(values), dtype=torch.bool).tril()
    output, weights = attention(q, q, values[None, None, :, None], mask)
    expected = values.cumsum(0) / torch.arange(1, len(values) + 1)
    changed = values.clone()
    changed[4:] += 1000
    changed_output, _ = attention(q, q, changed[None, None, :, None], mask)
    return {"provenance": "LeetPaper teaching adaptation of Transformer Section 3.2.1",
            "values": values.tolist(), "expected": expected.tolist(),
            "actual": output.flatten().detach().tolist(),
            "weights": weights[0, 0].detach().tolist(),
            "max_error": (output.flatten() - expected).abs().max().item(),
            "prefix_leakage": (changed_output.flatten()[:4] - output.flatten()[:4]).abs().max().item()}
