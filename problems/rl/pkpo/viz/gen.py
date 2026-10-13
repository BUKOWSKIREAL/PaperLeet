from problems.rl.pkpo.experiment import run
from problems.rl.pkpo.solution import maxk_estimate, pkpo_weights, pkpo_loss

def generate():
    return run(maxk_estimate, pkpo_weights, pkpo_loss)
