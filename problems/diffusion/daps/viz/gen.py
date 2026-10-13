from problems.diffusion.daps.experiment import run
from problems.diffusion.daps.solution import posterior_score, langevin_step, renoise


def generate():
    return run(posterior_score, langevin_step, renoise)
