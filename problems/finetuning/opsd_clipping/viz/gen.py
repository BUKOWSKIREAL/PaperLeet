from problems.finetuning.opsd_clipping.experiment import run
from problems.finetuning.opsd_clipping.solution import clipped_distillation_loss

def generate():
    return run(clipped_distillation_loss)
