from problems.rl.gspo.solution import sequence_ratio, gspo_loss
from problems.rl.gspo.experiment import run


def generate():
    return run(sequence_ratio, gspo_loss)
