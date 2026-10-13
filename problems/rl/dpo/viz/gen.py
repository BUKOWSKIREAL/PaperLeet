import torch
from problems.rl.dpo.solution import dpo_loss
from problems.rl.dpo.experiment import run


def generate():
    ids=torch.tensor([0,1,2,3,4,5,1]);mask=torch.tensor([0,0,0,1,1,1,1])
    logits=torch.randn(7,6,generator=torch.Generator().manual_seed(42))
    scores=logits[:-1].log_softmax(-1).gather(1,ids[1:,None]).squeeze(1)*mask[1:]
    return dict(problem='rl/dpo',shift_mask_grid=dict(tokens=['<bos>','Question',':','a','b','c','<eos>'],mask=mask.tolist(),contributions=[0.]+scores.tolist()),
                t3_reward_scatter=run(dpo_loss))
