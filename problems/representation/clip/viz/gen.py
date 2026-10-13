from problems.representation.clip.solution import clip_loss
from problems.representation.clip.experiment import run


def generate():
    result = run(clip_loss)
    return {"kind": "toy_experiment", "title": "Paired-vector retrieval", "problem": "representation/clip",
            "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2103.00020",
                           "adaptation": "Eight trainable vector pairs, 80 CPU steps; no image/text encoders or pretrained weights."},
            "loss_curve": [[step, loss] for step, loss in enumerate(result["losses"])],
            "metrics": {key: result[key] for key in ("initial_loss", "final_loss", "image_to_text_accuracy", "text_to_image_accuracy", "inverse_temperature")},
            "similarities": result["similarities"]}
