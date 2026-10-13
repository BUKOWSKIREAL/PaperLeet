from problems.rl.dr_grpo.solution import drgrpo_advantages,drgrpo_loss,grpo_reference_advantages,grpo_reference_loss
from problems.rl.dr_grpo.experiment import run


def generate():
    return dict(problem='rl/dr_grpo',lengths=[3,6,12,24],t3_length_curves=run(drgrpo_advantages,drgrpo_loss,grpo_reference_advantages,grpo_reference_loss))
