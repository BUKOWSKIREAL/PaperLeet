from problems.finetuning.opsd_alignment.experiment import run
from problems.finetuning.opsd_alignment.solution import align_rollout_logits

def generate():
    return run(align_rollout_logits)
