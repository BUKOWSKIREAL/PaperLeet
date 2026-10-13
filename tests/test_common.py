"""Independent checks for shared oracles: judges must not agree on a shared bug."""
import math
import torch

from paperleet.common.asserts import assert_close_located as close
from paperleet.common.metrics import mmd_rbf, mode_coverage
from paperleet.common.models import TimeMLP
from paperleet.common.oracles import gmm_eps_oracle
from paperleet.common.toydata import GaussianMixture


def test_gmm_density_score_and_diffusion():
    means = torch.tensor([[-1., .5], [2., -.5]], dtype=torch.float64)
    weights = torch.tensor([.2, .8], dtype=torch.float64)
    gmm = GaussianMixture(means, .3, weights)
    x = torch.randn(9, 2, dtype=torch.float64, requires_grad=True)
    components = torch.distributions.Independent(torch.distributions.Normal(means, .3), 1)
    independent_density = (components.log_prob(x[:, None]) + weights.log()).logsumexp(-1)
    close(gmm.log_prob(x), independent_density, atol=1e-10)
    gradient = torch.autograd.grad(independent_density.sum(), x)[0]
    close(gmm.score(x), gradient, atol=1e-10)
    diffused = gmm.diffused(.4)
    close(diffused.means, math.sqrt(.4) * means, atol=1e-10)
    assert abs(diffused.std**2 - (.4 * .3**2 + .6)) < 1e-12


def test_epsilon_oracle_matches_posterior_conditional_mean():
    # For a Gaussian prior, E[x0|xt] is available by precision addition.
    mean = torch.tensor([[1., -2.]], dtype=torch.float64)
    gmm = GaussianMixture(mean, .6)
    alpha_bars = torch.tensor([.1, .4, .9], dtype=torch.float64)
    x = torch.randn(3, 2, dtype=torch.float64)
    a = alpha_bars[:, None]
    posterior_var = 1 / (1 / .6**2 + a / (1 - a))
    posterior_mean = posterior_var * (mean / .6**2 + a.sqrt() * x / (1 - a))
    expected = (x - a.sqrt() * posterior_mean) / (1 - a).sqrt()
    close(gmm_eps_oracle(gmm, alpha_bars)(x, torch.arange(3)), expected, atol=1e-10)


def test_unbiased_mmd_matches_scalar_definition():
    x = torch.tensor([[0., 1.], [2., 0.], [1., -1.]], dtype=torch.float64)
    y = torch.tensor([[.3, .4], [.5, -.7]], dtype=torch.float64)
    def kernel(a, b):
        return sum(math.exp(-float((a-b).square().sum()) / (2*h*h)) for h in (.05, .1, .2, .5, 1.))
    xx = sum(kernel(x[i], x[j]) for i in range(3) for j in range(3) if i != j) / 6
    yy = sum(kernel(y[i], y[j]) for i in range(2) for j in range(2) if i != j) / 2
    xy = sum(kernel(a, b) for a in x for b in y) / 6
    assert abs(mmd_rbf(x, y) - (xx + yy - 2*xy)) < 1e-10
    assert mmd_rbf(x, x) < 0  # An unbiased estimate is allowed to be negative.


def test_coverage_counts_outliers_separately():
    centers = torch.tensor([[0., 0.], [2., 0.]])
    result = mode_coverage(torch.tensor([[0., .1], [2., .1], [50., 50.], [2.1, 0.]]), centers, .5)
    assert result == {"proportions": [.25, .5], "outlier_fraction": .25, "covered_modes": 2}


def test_model_initialization_reproducible_without_changing_rng():
    before = torch.random.get_rng_state().clone()
    first, second = TimeMLP(), TimeMLP()
    assert torch.equal(torch.random.get_rng_state(), before)
    x, t = torch.randn(8, 2), torch.rand(8)
    close(first(x, t), second(x, t), rtol=0, atol=0)
