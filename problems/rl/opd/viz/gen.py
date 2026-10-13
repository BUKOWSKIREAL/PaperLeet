from problems.rl.opd.solution import sample_student, distillation_loss
from problems.rl.opd.experiment import run


def generate():
    return run(sample_student, distillation_loss)
