from problems.world_models.lewm_prediction.experiment import run
from problems.world_models.lewm_prediction.solution import lewm_objective, latent_rollout


def generate():
    return run(lewm_objective, latent_rollout)
