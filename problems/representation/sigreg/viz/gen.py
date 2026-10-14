from problems.representation.sigreg.experiment import run
from problems.representation.sigreg.solution import epps_pulley, sigreg_loss


def generate():
    return run(epps_pulley, sigreg_loss)
