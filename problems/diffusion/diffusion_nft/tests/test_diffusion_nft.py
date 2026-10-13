import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion.diffusion_nft.experiment import run


@pytest.mark.t0
def test_interface_dtype_and_inputs(impl):
    """Both policy tensors preserve arbitrary feature dimensions and the scalar loss preserves dtype."""
    for dtype in (torch.float32, torch.float64):
        current = torch.randn(3, 2, 4, dtype=dtype)
        old, target, r = torch.randn_like(current), torch.randn_like(current), torch.tensor([0., .4, 1.], dtype=dtype)
        saved = [v.clone() for v in (current, old, target, r)]
        positive, negative = impl.implicit_velocities(current, old, .3)
        for value in (positive, negative):
            assert value.shape == current.shape and value.dtype == dtype and value.device == current.device
        loss = impl.nft_loss(current, old, target, r, .3)
        assert loss.shape == () and loss.dtype == dtype
        for actual, before in zip((current, old, target, r), saved): close(actual, before)


@pytest.mark.t0
def test_validation(impl):
    """Bounded per-example rewards and finite positive beta are enforced by supplied checks."""
    x = torch.ones(2, 3); r = torch.tensor([0., 1.])
    for beta in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError): impl.implicit_velocities(x, x, beta)
    for wrong in (torch.tensor([-.1, 1.]), torch.tensor([0., 1.1]), torch.tensor([0., float('nan')]), r[:, None]):
        with pytest.raises(ValueError): impl.nft_loss(x, x, x, wrong, .5)
    with pytest.raises(ValueError): impl.nft_loss(x, x, x[:, :2], r, .5)


@pytest.mark.t1
def test_independent_scalar_dual_objective(impl):
    """Scalar arithmetic checks beta, branch signs, reward broadcasting and sum-then-mean reduction."""
    current = torch.tensor([[2., -1.], [.5, 3.], [-2., .8]], dtype=torch.float64)
    old = current.new_tensor([[.2, 1.], [-1., .3], [2., -.1]])
    target = current.new_tensor([[1., .5], [2., -1.], [.1, .8]])
    rewards = current.new_tensor([0., .2, 1.]); beta = .4
    expected_p, expected_n, total = [], [], 0.
    for cur, frozen, goal, r in zip(current.tolist(), old.tolist(), target.tolist(), rewards.tolist()):
        p = [(1 - beta) * o + beta * c for c, o in zip(cur, frozen)]
        n = [(1 + beta) * o - beta * c for c, o in zip(cur, frozen)]
        total += sum(r * (vp - v) ** 2 + (1 - r) * (vn - v) ** 2 for vp, vn, v in zip(p, n, goal))
        expected_p.append(p); expected_n.append(n)
    p, n = impl.implicit_velocities(current, old, beta)
    close(p, current.new_tensor(expected_p)); close(n, current.new_tensor(expected_n))
    close(impl.nft_loss(current, old, target, rewards, beta), current.new_tensor(total / 3), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_reward_endpoints(impl):
    """Positive and negative examples select different implicit policies; neither is a signed MSE shortcut."""
    current = torch.tensor([[2., -3.]], dtype=torch.float64)
    old = current.new_tensor([[1., 1.]])
    target = current.new_tensor([[.5, -1.]])
    close(impl.nft_loss(current, old, target, current.new_tensor([1.]), .5), current.new_tensor(1.))
    close(impl.nft_loss(current, old, target, current.new_tensor([0.]), .5), current.new_tensor(16.))


@pytest.mark.t2
def test_exact_gradient_and_frozen_context(impl):
    """Analytic gradients flow only into the current policy, never old outputs, targets or rewards."""
    current = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
    old = torch.randn_like(current, requires_grad=True)
    target = torch.randn_like(current, requires_grad=True)
    r = torch.tensor([0., .2, .5, 1.], dtype=current.dtype, requires_grad=True)
    beta = .7
    loss = impl.nft_loss(current, old, target, r, beta)
    grads = torch.autograd.grad(loss, (current, old, target, r), allow_unused=True)
    expected = (2 * beta / 4) * (beta * (current.detach() - old.detach()) + (2 * r.detach()[:, None, None] - 1) * (old.detach() - target.detach()))
    close(grads[0], expected, atol=1e-12, rtol=1e-12)
    assert all(g is None or not g.any() for g in grads[1:])


@pytest.mark.t2
def test_neutral_reward_anchors_old_policy(impl):
    """Neutral reward yields zero gradient at old policy and quadratic pullback after displacement."""
    old = torch.randn(3, 4, dtype=torch.float64)
    current = old.clone().requires_grad_(); target = torch.randn_like(current)
    r = current.new_full((3,), .5)
    gradient, = torch.autograd.grad(impl.nft_loss(current, old, target, r, .3), current)
    close(gradient, torch.zeros_like(current), atol=1e-12, rtol=1e-12)
    shifted = (old + .7).requires_grad_()
    gradient, = torch.autograd.grad(impl.nft_loss(shifted, old, target, r, .3), shifted)
    close(gradient, current.new_full(current.shape, 2 * .3 ** 2 * .7 / 3), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_complement_symmetry_and_feature_reduction(impl):
    """Reflecting the policy and complementing rewards leaves the objective invariant; duplicating features doubles it."""
    current, old, target = [torch.randn(3, 4, dtype=torch.float64) for _ in range(3)]
    r = current.new_tensor([.1, .6, .8]); beta = .2
    loss = impl.nft_loss(current, old, target, r, beta)
    close(loss, impl.nft_loss(2 * old - current, old, target, 1 - r, beta), atol=1e-12, rtol=1e-12)
    close(2 * loss, impl.nft_loss(current.repeat(1, 2), old.repeat(1, 2), target.repeat(1, 2), r, beta))
    close(loss, impl.nft_loss(current.repeat(2, 1), old.repeat(2, 1), target.repeat(2, 1), r.repeat(2), beta))


@pytest.mark.t2
def test_closed_form_optimum_and_branch_gradients(impl):
    """The quadratic optimum and separate branch Jacobians rule out reciprocal beta and detached students."""
    old, target = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    r = old.new_tensor([0., .25, .5, .75, 1.]); beta = .6
    optimum = (old + (2 * r[:, None] - 1) * (target - old) / beta).requires_grad_()
    gradient, = torch.autograd.grad(impl.nft_loss(optimum, old, target, r, beta), optimum)
    close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=1e-12)
    p, n = impl.implicit_velocities(optimum, old, beta)
    dp, = torch.autograd.grad(p.sum(), optimum, retain_graph=True)
    dn, = torch.autograd.grad(n.sum(), optimum)
    close(dp, torch.full_like(dp, beta)); close(dn, torch.full_like(dn, -beta))


@pytest.mark.t3
def test_forward_process_velocity_training(impl, artifact):
    """Submitted loss trains five reward-conditioned velocity outputs to independently derived optima."""
    result = run(impl.implicit_velocities, impl.nft_loss)
    artifact('nft_forward_minibatch', result)
    m = result['metrics']
    assert m['optimum_max_error'] < 1e-7
    assert abs(m['final_loss'] - m['optimal_loss']) < 1e-10
    assert m['final_loss'] < .6 * m['initial_loss']
    assert m['neutral_velocity_norm'] < 1e-12
    assert m['positive_endpoint_error'] < 1e-7 and m['negative_endpoint_error'] < 1e-7
    assert m['forward_path_reconstruction_error'] < 1e-12
