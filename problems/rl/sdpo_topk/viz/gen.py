from problems.rl.sdpo_topk.experiment import run
from problems.rl.sdpo_topk.solution import topk_tail_reverse_kl

def generate():
    return run(topk_tail_reverse_kl)
