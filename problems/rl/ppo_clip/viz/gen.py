from problems.rl.ppo_clip.solution import ppo_loss
from problems.rl.ppo_clip.mutants.m1_clip_only import ppo_loss as clip_only
from problems.rl.ppo_clip.mutants.m2_max_not_min import ppo_loss as max_not_min
from problems.rl.ppo_clip.experiment import run


def generate():
    reference=run(ppo_loss)
    return dict(problem='rl/ppo_clip',trust_region=dict(reference=reference['trust_region'],
                clip_only=run(clip_only)['trust_region'],max_not_min=run(max_not_min)['trust_region']),training=reference['training'])
