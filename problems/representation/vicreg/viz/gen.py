from problems.representation.vicreg.experiment import run
from problems.representation.vicreg.solution import vicreg_terms, vicreg_loss


def generate():
    return run(vicreg_terms, vicreg_loss)
