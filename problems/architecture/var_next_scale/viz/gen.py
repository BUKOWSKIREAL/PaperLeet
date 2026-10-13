from problems.architecture.var_next_scale.experiment import run
from problems.architecture.var_next_scale.solution import next_scale_inputs, scale_causal_mask


def generate():
    return run(next_scale_inputs, scale_causal_mask)
