import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_shapes(impl):
    x=torch.randn(4,2,dtype=torch.float64);p=torch.randn(7,2,dtype=x.dtype)
    v=impl.drift_field(x,p,.6)
    assert v.shape==x.shape and v.dtype==x.dtype
    assert impl.drifting_loss(x,p,.6).ndim==0

@pytest.mark.t1
def test_scalar_pairwise_oracle(impl):
    """A scalar-loop oracle independently checks cross-masses and both normalizations."""
    x=torch.tensor([[0.,0.],[1.,0.],[-.5,.7]],dtype=torch.float64);p=torch.tensor([[.3,.1],[.8,1.]],dtype=x.dtype)
    targets=torch.cat((x,p));k=torch.empty(3,5,dtype=x.dtype)
    for i in range(3):
        for j in range(5):k[i,j]=0 if i==j else torch.exp(-torch.linalg.norm(x[i]-targets[j])/.8)
    a=k/(k.sum(1)[:,None]*k.sum(0)[None,:]).clamp_min(1e-12).sqrt()
    expected=torch.zeros_like(x)
    for i in range(3):
        for j in range(3):
            for h in range(2):expected[i]+=a[i,j]*a[i,3+h]*(p[h]-x[j])
    assert_close(impl.drift_field(x,p,.8),expected)

@pytest.mark.t2
def test_frozen_target_gradient(impl):
    x=torch.tensor([[.1,.2],[.8,-.5]],requires_grad=True);p=torch.tensor([[1.,1.],[.7,.9]],requires_grad=True)
    field=impl.drift_field(x.detach(),p.detach(),.7)
    loss=impl.drifting_loss(x,p,.7);loss.backward()
    assert_close(x.grad,-2*field/x.numel())
    assert p.grad is None or p.grad.eq(0).all()
    assert x.grad.norm()>0

@pytest.mark.t2
def test_permutation_translation(impl):
    x=torch.randn(5,2,dtype=torch.float64);p=torch.randn(7,2,dtype=x.dtype);order=torch.tensor([3,0,4,1,2])
    v=impl.drift_field(x,p,.8)
    assert_close(impl.drift_field(x[order],p.flip(0),.8),v[order])
    assert_close(impl.drift_field(x+2,p+2,.8),v,atol=1e-12,rtol=1e-8)

@pytest.mark.t2
def test_mask_and_validation(impl):
    # One generated sample has no negative partner after masking, hence zero field.
    assert_close(impl.drift_field(torch.zeros(1,2),torch.ones(3,2),.5),torch.zeros(1,2))
    with pytest.raises(ValueError):impl.drift_field(torch.zeros(2,2),torch.ones(2,2),0)

@pytest.mark.t3
def test_official_toy_adaptation(impl,artifact):
    """Train one-pass generators on reduced Swiss-roll and checkerboard experiments."""
    from problems.diffusion.drifting.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    for name in ('swiss_roll','checkerboard'):
        assert result['metrics'][name+'_final_mmd'] < result['metrics'][name+'_initial_mmd']*.7

@pytest.mark.t2
def test_distant_kernel_and_nonempty_contract(impl):
    # Every off-diagonal distance is about 1000: raw exp(-distance/.2) underflows.
    x=torch.tensor([[0.,0.],[1000.,0.]],dtype=torch.float64)
    positive=torch.tensor([[500.,500*3**.5]],dtype=x.dtype)
    field=impl.drift_field(x,positive,.2)
    assert torch.isfinite(field).all()
    assert (field[:,1]>100).all(), 'normalized affinity must survive common exponential underflow'
    for a,b in ((x[:0],positive),(x,positive[:0])):
        with pytest.raises(ValueError):impl.drift_field(a,b,.2)
    with pytest.raises(ValueError):impl.drift_field(x,positive,float('nan'))
