import torch
from problems.rl.gae.solution import compute_gae
from problems.rl.gae.experiment import run


def generate():
    generator=torch.Generator().manual_seed(42)
    rewards=torch.randn(20,generator=generator)*.4
    values=torch.sin(torch.linspace(0,5,20))*.5
    next_values=torch.cat((values[1:],torch.tensor([.2])))
    terminated=torch.zeros(20,dtype=torch.bool);terminated[9]=True
    truncated=torch.zeros(20,dtype=torch.bool);truncated[15]=True
    return dict(problem='rl/gae',timeline=dict(rewards=rewards.tolist(),values=values.tolist(),next_values=next_values.tolist(),terminated=terminated.tolist(),truncated=truncated.tolist()),
                bias_variance_vs_lambda=run(compute_gae,lambdas=[i/10 for i in range(11)]))
