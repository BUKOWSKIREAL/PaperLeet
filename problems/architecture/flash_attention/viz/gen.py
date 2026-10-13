from problems.architecture.flash_attention.solution import block_attention
from problems.architecture.flash_attention.experiment import run


def generate():
    result = run(block_attention)
    return {"kind": "toy_experiment", "title": "Exact attention across tile sizes", "problem": "architecture/flash_attention",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2205.14135",
                           "adaptation": "37 synthetic tokens; CPU numerical equivalence, no GPU speed or measured-memory claim."},
            "samples": [[row["block_size"], row["output_error"]] for row in result["rows"]],
            "metrics": {"maximum_output_error": max(row["output_error"] for row in result["rows"]),
                        "maximum_lse_error": max(row["lse_error"] for row in result["rows"])},
            "rows": result["rows"]}
