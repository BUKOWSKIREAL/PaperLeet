from problems.architecture.layer_norm.solution import layer_norm, post_norm_residual
from problems.architecture.layer_norm.experiment import run


def generate():
    result = run(layer_norm, post_norm_residual)
    return {"kind": "toy_experiment", "title": "Token-local Add & Norm",
            "problem": "architecture/layer_norm",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/1706.03762",
                           "anchor": "Section 3.1: residual connections followed by layer normalization",
                           "adaptation": "Eight synthetic tokens with varied offsets and scales; dropout disabled; no model training."},
            "charts": result["charts"], "metrics": result["metrics"]}
