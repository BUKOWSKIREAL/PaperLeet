from problems.optimization.adamw.solution import adamw_step
from problems.optimization.adamw.experiment import run


def generate():
    result = run(adamw_step)
    return {"kind": "toy_experiment", "title": "Anisotropic quadratic optimization", "problem": "optimization/adamw",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/1711.05101",
                           "adaptation": "Two coordinates and 200 CPU steps; not an image-classification benchmark."},
            "samples": result["trajectory"], "reference_samples": [result["target"]],
            "loss_curve": [[step, loss] for step, loss in enumerate(result["losses"])],
            "metrics": {"initial_loss": result["initial_loss"], "final_loss": result["final_loss"]}}
