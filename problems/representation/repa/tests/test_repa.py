import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.repa.experiment import run


def scalar_oracle(projected, teacher, eps=1e-8):
    values = []
    for p, t in zip(projected.detach().reshape(-1, projected.shape[-1]).tolist(), teacher.detach().reshape(-1, teacher.shape[-1]).tolist()):
        pn, tn = max(math.sqrt(sum(a*a for a in p)), eps), max(math.sqrt(sum(a*a for a in t)), eps)
        values.append(-sum(a*b for a,b in zip(p,t)) / (pn * tn))
    return sum(values) / len(values)


@pytest.mark.t0
def test_scalar_dtype_inputs_and_strides(impl):
    """Losses are scalar, preserve dtype/device, accept strided features and leave inputs intact."""
    for dtype in (torch.float32, torch.float64):
        p, t = torch.randn(2, 3, 4, dtype=dtype).transpose(0,1), torch.randn(2, 3, 4, dtype=dtype).transpose(0,1)
        before = p.clone(), t.clone()
        result = impl.repa_alignment(p,t)
        assert result.shape == () and result.dtype == dtype and result.device == p.device
        close(result, torch.tensor(scalar_oracle(p,t), dtype=dtype))
        close(p,before[0]); close(t,before[1])
        combined = impl.repa_objective(p,t,p,t)
        assert combined.shape == () and combined.dtype == dtype


@pytest.mark.t0
def test_supplied_validation(impl):
    """Invalid shape, empty patches, epsilon and loss weight are rejected."""
    x = torch.ones(1,2,3)
    for eps in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError): impl.repa_alignment(x,x,eps)
    for p,t in ((x,x[:,:,:2]), (x[0],x[0]), (x[:,:0],x[:,:0])):
        with pytest.raises(ValueError): impl.repa_alignment(p,t)
    for weight in (-1.,float('inf'),float('nan')):
        with pytest.raises(ValueError): impl.repa_objective(x,x,x,x,weight)
    with pytest.raises(ValueError): impl.repa_objective(x,x[:,:,:2],x,x)


@pytest.mark.t1
def test_patchwise_scalar_oracle_and_objective(impl):
    """Different patch magnitudes distinguish per-patch cosine from pooling or channel-wise normalization."""
    p = torch.tensor([[[1.,2.,-.5],[4.,-.2,1.]], [[-.1,2.,3.],[1.,-8.,2.]]], dtype=torch.float64)
    t = torch.tensor([[[2.,-.3,1.],[.5,4.,-1.]], [[2.,1.,-.2],[-3.,1.,2.]]], dtype=p.dtype)
    expected = scalar_oracle(p,t)
    close(impl.repa_alignment(p,t), p.new_tensor(expected), atol=1e-12, rtol=1e-12)
    pred, target = p*.3, t*.2
    mse = sum((a-b)**2 for a,b in zip(pred.flatten().tolist(),target.flatten().tolist())) / pred.numel()
    for weight in (0., .5, 2.3):
        close(impl.repa_objective(pred,target,p,t,weight), p.new_tensor(mse+weight*expected), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_zero_and_sub_epsilon_vectors(impl):
    """Norms are clamped independently; a zero teacher patch contributes zero without NaNs."""
    p = torch.tensor([[[0.,0.],[.002,-.003],[1.,2.]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[1.,2.],[.001,.004],[0.,0.]]], dtype=p.dtype)
    close(impl.repa_alignment(p,t,.01), p.new_tensor(scalar_oracle(p,t,.01)), atol=1e-12, rtol=1e-12)
    gradient, = torch.autograd.grad(impl.repa_alignment(p,t,.01), p)
    assert torch.isfinite(gradient).all()
    close(gradient[0,1], -t[0,1] / (.01**2 * 3), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_analytic_gradient_and_frozen_targets(impl):
    """Independent cosine derivatives reach projected features; neither fixed target receives gradients."""
    p = torch.tensor([[[1.,2.,-1.],[.5,-2.,3.]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[2.,-.5,1.],[-1.,1.,.2]]], dtype=p.dtype, requires_grad=True)
    pred = torch.randn(2,3,dtype=p.dtype,requires_grad=True)
    target = torch.randn(2,3,dtype=p.dtype,requires_grad=True)
    dp, dt, dv, dtarget = torch.autograd.grad(impl.repa_objective(pred,target,p,t,.7), (p,t,pred,target), allow_unused=True)
    expected = []
    for a,b in zip(p.detach()[0].tolist(),t.detach()[0].tolist()):
        pn,tn = math.sqrt(sum(v*v for v in a)),math.sqrt(sum(v*v for v in b))
        dot = sum(x*y for x,y in zip(a,b))
        expected.append([-.7/2 * (y/(pn*tn) - dot*x/(pn**3*tn)) for x,y in zip(a,b)])
    close(dp,p.new_tensor([expected]),atol=1e-12,rtol=1e-12)
    close(dv,2*(pred.detach()-target.detach())/pred.numel(),atol=1e-12,rtol=1e-12)
    assert dt is None or torch.count_nonzero(dt) == 0
    assert dtarget is None or torch.count_nonzero(dtarget) == 0


@pytest.mark.t2
def test_patch_scaling_permutation_and_sign(impl):
    """Independent positive patch scales preserve loss; matching/opposite patches reach -1/+1."""
    p,t = torch.randn(2,4,3,dtype=torch.float64),torch.randn(2,4,3,dtype=torch.float64)
    scales = torch.tensor([1.,4.,.1,7.],dtype=p.dtype)[None,:,None]
    close(impl.repa_alignment(p,t),impl.repa_alignment(p*scales,t/scales),atol=1e-12,rtol=1e-12)
    close(impl.repa_alignment(p,t),impl.repa_alignment(p[:,[2,0,3,1]],t[:,[2,0,3,1]]))
    close(impl.repa_alignment(p,p),p.new_tensor(-1.))
    close(impl.repa_alignment(p,-p),p.new_tensor(1.))


@pytest.mark.t3
def test_submitted_projector_training(impl, artifact):
    """The submitted losses train a real tiny projector; an independent cosine measures improvement."""
    result = run(impl.repa_alignment,impl.repa_objective)
    artifact('tiny_projector',result)
    metrics = result['metrics']
    assert metrics['final_cosine'] > .98
    assert metrics['final_cosine'] > metrics['initial_cosine'] + .2
    assert metrics['alignment_formula_error'] < 1e-10
