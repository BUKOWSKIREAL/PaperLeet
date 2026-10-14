from problems.world_models.causal_jepa.experiment import run
from problems.world_models.causal_jepa.solution import object_mask_inputs, masked_prediction_loss


def generate():
    return run(object_mask_inputs, masked_prediction_loss)
