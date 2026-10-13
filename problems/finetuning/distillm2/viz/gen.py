from problems.finetuning.distillm2.experiment import run
from problems.finetuning.distillm2.solution import skew_kl, skew_reverse_kl, distillm2_loss

def generate():
    return run(skew_kl, skew_reverse_kl, distillm2_loss)
