from problems.diffusion.block_diffusion.experiment import run
from problems.diffusion.block_diffusion.solution import block_diffusion_mask, block_diffusion_attention

def generate():
    return run(block_diffusion_mask, block_diffusion_attention)
