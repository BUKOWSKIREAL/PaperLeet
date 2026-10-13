from problems.rl.dapo.solution import dynamic_group_mask, dapo_loss
from problems.rl.dapo.experiment import run


def generate():
    return run(dynamic_group_mask, dapo_loss)
