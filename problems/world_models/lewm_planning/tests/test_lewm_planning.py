import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.world_models.lewm_planning.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_inputs(impl):
    """Costs, Gaussian statistics and elite indices have separate documented shapes."""
    for dtype in (torch.float32, torch.float64):
        x = torch.arange(30, dtype=dtype).reshape(5, 3, 2); before = x.clone()
        costs = impl.terminal_cost(x, torch.zeros(2, dtype=dtype))
        mean, variance, ids = impl.cem_refit(x, costs, 2)
        assert costs.shape == (5,) and mean.shape == variance.shape == (3, 2)
        assert ids.shape == (2,) and ids.dtype == torch.long
        assert mean.dtype == variance.dtype == dtype
        close(x, before)


@pytest.mark.t0
def test_invalid_contract(impl):
    """CEM requires nonempty finite candidates and a valid elite count."""
    x = torch.ones(3, 2, 1); costs = torch.arange(3.)
    for k in (0, 4, True, 1.5):
        with pytest.raises(ValueError): impl.cem_refit(x, costs, k)
    with pytest.raises(ValueError): impl.cem_refit(x, costs * float('nan'), 2)
    with pytest.raises(ValueError): impl.terminal_cost(x, torch.zeros(2))


@pytest.mark.t1
def test_terminal_l2_scalar_oracle(impl):
    """Early rollout errors do not contribute; latent dimensions sum rather than average."""
    x = torch.tensor([[[99., -40.], [2., 4.]], [[0., 0.], [5., -1.]]], dtype=torch.float64)
    goal = x.new_tensor([1., 2.])
    close(impl.terminal_cost(x, goal), x.new_tensor([5., 25.]))


@pytest.mark.t1
def test_elite_statistics_oracle(impl):
    """Select the lowest costs and compute each time/action coordinate independently."""
    x = torch.tensor([[[1., 2.], [3., 4.]], [[-2., 6.], [0., 2.]], [[5., 10.], [-3., 1.]], [[9., -4.], [7., 8.]]], dtype=torch.float64)
    costs = x.new_tensor([7., -1., 4., 2.])
    mean, variance, ids = impl.cem_refit(x, costs, 2)
    assert ids.tolist() == [1, 3]
    expected_mean = (x[1] + x[3]) / 2
    expected_variance = (x[1] - x[3]).square() / 4
    close(mean, expected_mean); close(variance, expected_variance)


@pytest.mark.t2
def test_ties_single_elite_and_all_elites(impl):
    """Stable ties favor smaller indices; K=1 yields zero population variance."""
    x = torch.tensor([[[3.]], [[-1.]], [[7.]], [[2.]]], dtype=torch.float64)
    costs = x.new_tensor([1., 0., 0., 2.])
    mean, variance, ids = impl.cem_refit(x, costs, 1)
    assert ids.tolist() == [1]; close(mean, x[1]); close(variance, torch.zeros_like(variance))
    mean, variance, ids = impl.cem_refit(x, costs, 4)
    close(mean, x.mean(0)); close(variance, torch.var(x, 0, correction=0))


@pytest.mark.t2
def test_affine_equivariance_and_monotone_costs(impl):
    """Translating/scaling actions transforms statistics, while order-preserving costs keep elites."""
    x = torch.randn(7, 4, 2, dtype=torch.float64); costs = torch.arange(7, dtype=x.dtype).flip(0)
    mean, variance, ids = impl.cem_refit(x, costs, 3)
    transformed = impl.cem_refit(2 * x - 3, 4 * costs + 10, 3)
    close(transformed[0], 2 * mean - 3); close(transformed[1], 4 * variance)
    assert torch.equal(ids, transformed[2])


@pytest.mark.t2
def test_goal_translation_and_prefix_invariance(impl):
    """A coordinate translation or arbitrary intermediate states cannot alter terminal ranking."""
    x = torch.randn(5, 4, 3, dtype=torch.float64); goal = torch.randn(3, dtype=x.dtype)
    original = impl.terminal_cost(x, goal)
    shifted = impl.terminal_cost(x + 10, goal + 10)
    close(original, shifted)
    x[:, :-1] = 1000
    close(impl.terminal_cost(x, goal), original)


@pytest.mark.t3
def test_cem_planning_experiment(impl, artifact):
    """The submitted cost and refit shrink goal error through the supplied deterministic CEM loop."""
    result = run(impl.terminal_cost, impl.cem_refit)
    artifact('lewm_cem_planning', result)
    m = result['metrics']
    assert m['final_mean_plan_cost'] < .001 * m['initial_cost']
    assert m['final_mean_variance'] < .02
