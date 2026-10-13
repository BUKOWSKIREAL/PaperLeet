from problems.representation.var_tokenizer.experiment import run
from problems.representation.var_tokenizer.solution import multiscale_quantize, decode_multiscale


def generate():
    return run(multiscale_quantize, decode_multiscale)
