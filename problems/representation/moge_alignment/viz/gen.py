from problems.representation.moge_alignment.experiment import run
from problems.representation.moge_alignment.solution import optimal_scale_1d, roe_align


def generate():
    return run(optimal_scale_1d, roe_align)
