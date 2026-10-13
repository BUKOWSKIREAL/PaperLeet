from problems.finetuning.lora.solution import lora_delta, lora_linear
from problems.finetuning.lora.experiment import run


def generate():
    return run(lora_delta, lora_linear)
