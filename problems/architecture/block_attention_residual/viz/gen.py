from problems.architecture.block_attention_residual.solution import assemble_sources, depth_attention
from problems.architecture.block_attention_residual.experiment import run


def generate():
    return run(assemble_sources, depth_attention)
