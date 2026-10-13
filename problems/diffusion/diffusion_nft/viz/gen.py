from problems.diffusion.diffusion_nft.experiment import run
from problems.diffusion.diffusion_nft.solution import implicit_velocities, nft_loss


def generate():
    return run(implicit_velocities, nft_loss)
