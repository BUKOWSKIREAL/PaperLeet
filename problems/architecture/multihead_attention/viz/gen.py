from problems.architecture.multihead_attention.solution import multi_head_attention
from problems.architecture.multihead_attention.experiment import run


def generate():
    result = run(multi_head_attention)
    return {**result, "kind": "toy_experiment", "problem": "architecture/multihead_attention",
            "title": "Two heads, two feature subspaces",
            "provenance": {"kind": "teaching_adaptation",
                           "source": "https://arxiv.org/abs/1706.03762",
                           "anchor": "Section 3.2.2, Figure 2: Multi-Head Attention",
                           "adaptation": "Two fixed heads attend to three synthetic tokens; independent analytic target, no paper benchmark reproduction."}}
