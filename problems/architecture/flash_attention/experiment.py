"""LeetPaper adaptation of FlashAttention Algorithm 1; CPU numerical experiment."""
import math
import torch


def run(attention):
    generator = torch.Generator().manual_seed(42)
    q, k = [torch.randn(37, 8, dtype=torch.float64, generator=generator) * 4 for _ in range(2)]
    v = torch.randn(37, 5, dtype=torch.float64, generator=generator)
    scores = q @ k.T / math.sqrt(8)
    expected = scores.softmax(-1) @ v
    expected_lse = scores.logsumexp(-1)
    rows = []
    for size in (1, 3, 8, 16, 37, 64):
        output, lse = attention(q, k, v, size)
        rows.append({"block_size": size,
                     "output_error": (output - expected).abs().max().item(),
                     "lse_error": (lse - expected_lse).abs().max().item(),
                     "score_tile_elements": min(size, 37) ** 2})
    return {"provenance": "LeetPaper teaching adaptation of FlashAttention Algorithm 1",
            "rows": rows, "dense_score_elements": 37 ** 2,
            "measurement": "Score tile elements are theoretical counts, not measured peak memory or GPU speed."}
