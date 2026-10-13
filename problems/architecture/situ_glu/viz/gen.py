from problems.architecture.situ_glu.experiment import run
from problems.architecture.situ_glu.solution import softcap, situ_glu


def generate():
    return run(softcap, situ_glu)
