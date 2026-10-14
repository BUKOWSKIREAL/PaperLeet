from problems.representation.ijepa.experiment import run
from problems.representation.ijepa.solution import masked_targets, masked_prediction_loss, ema_update


def generate():
    return run(masked_targets, masked_prediction_loss, ema_update)
