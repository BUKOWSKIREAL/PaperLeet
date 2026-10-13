from problems.rl.grpo.experiment import run as train


def run(dr_advantage, dr_loss, grpo_advantage, grpo_loss, seed=42):
    return dict(display_only=True,
                grpo=train(grpo_advantage, grpo_loss, seed=seed, steps=100)['training'],
                dr_grpo=train(dr_advantage, dr_loss, seed=seed, dr=True, steps=100)['training'])
