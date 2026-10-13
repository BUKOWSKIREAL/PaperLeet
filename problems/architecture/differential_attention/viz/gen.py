from problems.architecture.differential_attention.experiment import run
from problems.architecture.differential_attention.solution import differential_lambda, differential_attention

def generate():
    return run(differential_lambda, differential_attention)
