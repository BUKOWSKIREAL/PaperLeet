from problems.finetuning.svdquant.experiment import run
from problems.finetuning.svdquant.solution import smooth_inputs, split_weight, svdquant_linear

def generate():
    return run(smooth_inputs, split_weight, svdquant_linear)
