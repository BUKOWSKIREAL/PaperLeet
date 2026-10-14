from problems.representation.visreg.experiment import run
from problems.representation.visreg.solution import visreg_terms, view_invariance


def generate():
    return run(visreg_terms, view_invariance)
