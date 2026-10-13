from problems.representation.repa.experiment import run
from problems.representation.repa.solution import repa_alignment, repa_objective

def generate():
    return run(repa_alignment, repa_objective)
