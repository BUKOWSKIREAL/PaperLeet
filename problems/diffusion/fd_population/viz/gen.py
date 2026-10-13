from problems.diffusion.fd_population.experiment import run
from problems.diffusion.fd_population.solution import batch_moments, ema_population_stats


def generate():
    return run(batch_moments, ema_population_stats)
