from problems.rl.glm5_offpolicy.solution import masked_is_weights, offpolicy_loss
from problems.rl.glm5_offpolicy.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "GLM-5: masked off-policy updates", "problem": "rl/glm5_offpolicy",
            **run(masked_is_weights, offpolicy_loss)}
