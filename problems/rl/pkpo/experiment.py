"""Native Gaussian reward toy, with a bounded deterministic Monte Carlo budget."""
import math
import torch


def run(maxk_estimate, pkpo_weights, pkpo_loss):
    generator = torch.Generator().manual_seed(42)
    # Reuse independent groups at each theta to make curves easier to compare.
    noise = torch.randn(2048, 16, generator=generator, dtype=torch.float64)
    means = torch.linspace(.65, 1.0, 15, dtype=torch.float64)
    objectives, gradients = {}, {}
    gradient_identity_error = 0.
    for k in (1, 4, 8):
        objectives[k], gradients[k] = [], []
        for mean in means:
            theta = mean.clone().requires_grad_()
            actions = (theta.detach() + .1 * noise).detach()
            rewards = torch.where((actions >= 0) & (actions <= 1), actions.square(), 0.)
            log_probs = -.5 * ((actions - theta)/.1).square() - math.log(.1*math.sqrt(2*math.pi))
            estimate = maxk_estimate(rewards, k).mean()
            loss = pkpo_loss(log_probs, rewards, k)
            gradient = -torch.autograd.grad(loss, theta)[0]
            score_estimate = (pkpo_weights(rewards, k) * (actions-theta.detach())/.01).sum(-1).mean()
            gradient_identity_error = max(gradient_identity_error, abs(float(gradient-score_estimate)))
            objectives[k].append(float(estimate))
            gradients[k].append(float(gradient))
    return dict(kind='toy_experiment', title='PKPO Gaussian expected-maximum toy', problem='rl/pkpo',
                provenance=dict(kind='paper_derived', source='https://arxiv.org/html/2505.15201v5#S5.SS1',
                                anchor='Section 5.1 / Figure 1', adaptation='Original Normal(theta, std=0.1) and x^2 reward on [0,1]; 2048 groups of 16 samples, 15 means, k=1/4/8. Independently redrawn CPU curves; no paper figure pixels or LLM benchmark.'),
                metrics=dict(gradient_identity_error=gradient_identity_error,
                             optimum_k1=float(means[max(range(15),key=lambda i:objectives[1][i])]),
                             optimum_k8=float(means[max(range(15),key=lambda i:objectives[8][i])]),
                             maxk_gain_at_theta_1=objectives[8][-1]-objectives[1][-1]),
                charts=[dict(kind='line',title='Expected maximum reward',x_label='Gaussian mean',x=means.tolist(),series=[dict(name=f'k={k}',values=objectives[k]) for k in (1,4,8)]),
                        dict(kind='line',title='Policy gradient from submitted loss',x_label='Gaussian mean',x=means.tolist(),series=[dict(name=f'k={k}',values=gradients[k]) for k in (1,4,8)])])
