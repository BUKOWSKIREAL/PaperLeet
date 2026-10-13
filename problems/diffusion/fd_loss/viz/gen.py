from problems.diffusion.fd_loss.experiment import run
from problems.diffusion.fd_loss.solution import frechet_distance, normalized_fd_loss


def generate():
    return run(frechet_distance, normalized_fd_loss)
