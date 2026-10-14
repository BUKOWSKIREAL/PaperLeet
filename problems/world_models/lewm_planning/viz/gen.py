from problems.world_models.lewm_planning.experiment import run
from problems.world_models.lewm_planning.solution import terminal_cost, cem_refit


def generate():
    return run(terminal_cost, cem_refit)
