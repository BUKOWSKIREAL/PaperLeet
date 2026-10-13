from problems.rl.grpo.solution import grpo_advantages,grpo_loss
from problems.rl.grpo.experiment import run


def generate():
    return dict(problem='rl/grpo',rewards=[0.,.2,.4,.8,1.,.6,.2,0.],lengths=[3,6,12,24],
                t3_reward_curve=run(grpo_advantages,grpo_loss)['training'])
