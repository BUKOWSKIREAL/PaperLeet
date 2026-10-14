import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.world_models.lewm_prediction.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_inputs(impl):
    """Joint outputs and rollouts preserve dtype and do not overwrite latent trajectories."""
    for dtype in (torch.float32, torch.float64):
        z = torch.arange(24, dtype=dtype).reshape(2, 4, 3) / 10
        a = torch.ones(2, 3, 1, dtype=dtype); saved = z.clone()
        got = impl.lewm_objective(z, a, lambda z, a: z + a, lambda z: z.square().mean())
        assert all(x.shape == () and x.dtype == dtype for x in got)
        rollout = impl.latent_rollout(z[:, 0], a, lambda z, a: z + a)
        assert rollout.shape == (2, 3, 3) and rollout.dtype == dtype
        close(z, saved)


@pytest.mark.t0
def test_invalid_contracts(impl):
    """Supplied guards reject misaligned actions, empty horizons and negative weights."""
    z = torch.ones(2, 3, 2); a = torch.ones(2, 2, 1)
    for weight in (-1., float('nan')):
        with pytest.raises(ValueError): impl.lewm_objective(z, a, lambda z, a: z, lambda z: z.mean(), weight)
    with pytest.raises(ValueError): impl.lewm_objective(z, a[:, :1], lambda z, a: z, lambda z: z.mean())
    with pytest.raises(ValueError): impl.latent_rollout(z[:, 0], a[:, :0], lambda z, a: z)


@pytest.mark.t1
def test_teacher_forcing_and_timewise_regularizer_oracle(impl):
    """Nonuniform transitions distinguish teacher-forcing, temporal offset and regularizer axes."""
    z = torch.tensor([[[1., 2.], [4., -1.], [3., 5.]], [[-2., 0.], [1., 6.], [2., -3.]]], dtype=torch.float64)
    a = torch.tensor([[[.5], [2.]], [[-1.], [3.]]], dtype=z.dtype)
    calls = []
    def reg(x):
        calls.append(x.clone()); return x.mean(1).square().mean()
    total, prediction, regularization = impl.lewm_objective(z, a, lambda x, u: 2 * x + u, reg, .3)
    errors = [(2 * z[b, t, d] + a[b, t, 0] - z[b, t + 1, d]).item() ** 2 for b in range(2) for t in range(2) for d in range(2)]
    expected_prediction = z.new_tensor(sum(errors) / len(errors))
    expected_regularization = z.mean(0).square().mean()
    close(prediction, expected_prediction); close(regularization, expected_regularization)
    close(total, expected_prediction + .3 * expected_regularization)
    assert len(calls) == 1; close(calls[0], z.transpose(0, 1))


@pytest.mark.t1
def test_action_rollout_closed_form(impl):
    """Each action is used once and each prediction becomes the next state."""
    initial = torch.tensor([[2.], [-1.]], dtype=torch.float64)
    actions = torch.tensor([[[1.], [-2.], [.5]], [[.3], [.7], [-1.]]], dtype=initial.dtype)
    got = impl.latent_rollout(initial, actions, lambda z, a: .5 * z + 2 * a)
    expected = initial.new_tensor([[[.5 ** (t + 1) * initial[b, 0].item() + sum(.5 ** (t - j) * 2 * actions[b, j, 0].item() for j in range(t + 1))] for t in range(3)] for b in range(2)])
    close(got, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_joint_encoder_gradients_analytic(impl):
    """Future targets must receive their direct gradient in addition to the context path."""
    z = torch.tensor([[[1.], [4.], [-2.]]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[[.5], [1.5]]], dtype=z.dtype, requires_grad=True)
    total, _, _ = impl.lewm_objective(z, a, lambda x, u: 2 * x + u, lambda x: x.square().mean(), .6)
    gz, ga = torch.autograd.grad(total, (z, a))
    errors = 2 * z.detach()[:, :-1] + a.detach() - z.detach()[:, 1:]
    expected = .4 * z.detach()
    expected[:, :-1] += 2 * errors
    expected[:, 1:] -= errors
    close(gz, expected, atol=1e-12, rtol=1e-12); close(ga, errors)


@pytest.mark.t2
def test_rollout_gradient_and_causality(impl):
    """Later actions cannot affect earlier states; gradients traverse every recurrent step."""
    initial = torch.tensor([[1.]], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[[.2], [-.3], [.4]]], dtype=initial.dtype, requires_grad=True)
    result = impl.latent_rollout(initial, actions, lambda z, a: .7 * z + 1.2 * a)
    gi, ga = torch.autograd.grad(result[:, -1].sum(), (initial, actions), retain_graph=True)
    close(gi, initial.new_tensor([[.7 ** 3]])); close(ga, actions.new_tensor([[[1.2 * .7 ** 2], [1.2 * .7], [1.2]]]))
    early, = torch.autograd.grad(result[:, 0].sum(), actions)
    close(early[:, 1:], torch.zeros_like(early[:, 1:]))


@pytest.mark.t2
def test_zero_regularization_and_batch_replication(impl):
    """Zero lambda reduces to next-step MSE; duplicating all examples preserves the mean objective."""
    z = torch.randn(3, 4, 2, dtype=torch.float64); a = torch.randn(3, 3, 1, dtype=z.dtype)
    p = lambda x, u: x + u
    r = lambda x: x.square().mean()
    first = impl.lewm_objective(z, a, p, r, 0.)
    close(first[0], first[1])
    second = impl.lewm_objective(z.repeat(2, 1, 1), a.repeat(2, 1, 1), p, r, 0.)
    for x, y in zip(first, second): close(x, y)


@pytest.mark.t3
def test_tiny_joint_training(impl, artifact):
    """Submitted objective trains both encoder and predictor; submitted rollout predicts training actions."""
    result = run(impl.lewm_objective, impl.latent_rollout)
    artifact('lewm_joint_training', result)
    m = result['metrics']
    assert m['final_prediction_loss'] < 1e-4 * m['initial_prediction_loss']
    assert m['rollout_mse'] < 1e-4 and .5 < m['encoder_scale'] < 3.
    assert abs(m['learned_state_coefficient'] - .8) < .02
