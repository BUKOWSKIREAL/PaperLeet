from problems.architecture.attention.solution import scaled_dot_product_attention
from problems.architecture.attention.experiment import run


def generate():
    result = run(scaled_dot_product_attention)
    return {"kind": "toy_experiment", "title": "Causal prefix averaging", "problem": "architecture/attention",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/1706.03762",
                           "adaptation": "Six scalar values; zero query/key vectors; no model training."},
            "samples": [[i, value] for i, value in enumerate(result["actual"])],
            "reference_samples": [[i, value] for i, value in enumerate(result["expected"])],
            "metrics": {"maximum_error": result["max_error"], "future_leakage": result["prefix_leakage"]},
            "weights": result["weights"]}
