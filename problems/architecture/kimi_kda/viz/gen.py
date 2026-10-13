from problems.architecture.kimi_kda.solution import bounded_decay, recurrent_kda
from problems.architecture.kimi_kda.experiment import run


def generate():
    return run(bounded_decay, recurrent_kda)
