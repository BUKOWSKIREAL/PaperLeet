from problems.finetuning.qlora.solution import nf4_quantize, nf4_dequantize, qlora_linear
from problems.finetuning.qlora.experiment import run


def generate():
    return run(nf4_quantize, nf4_dequantize, qlora_linear)
