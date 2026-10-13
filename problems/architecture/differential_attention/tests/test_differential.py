import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.differential_attention.experiment import run


def scalar_probabilities(q,k,causal):
    # Nested scalar dot products and stable exp, independent of torch attention.
    n,d=q.shape
    rows=[]
    for i in range(n):
        keys=range(i+1) if causal else range(n)
        logits=[sum(float(q[i,c])*float(k[j,c]) for c in range(d))/math.sqrt(d) for j in keys]
        exps=[math.exp(a-max(logits)) for a in logits]
        rows.append([x/sum(exps) for x in exps]+[0.]*(n-len(exps)))
    return q.new_tensor(rows)


def scalar_output(q1,k1,q2,k2,v,coefficient,causal):
    output=torch.zeros_like(v)
    for b in range(q1.shape[0]):
        for h in range(q1.shape[1]):
            a=scalar_probabilities(q1[b,h],k1[b,h],causal).tolist()
            c=scalar_probabilities(q2[b,h],k2[b,h],causal).tolist()
            for i in range(v.shape[-2]):
                for d in range(v.shape[-1]):
                    output[b,h,i,d]=sum((a[i][j]-coefficient*c[i][j])*float(v[b,h,j,d]) for j in range(v.shape[-2]))
    return output


@pytest.mark.t0
def test_shapes_dtypes_strides_and_input_preservation(impl):
    """Multiple heads/batches and strided projected tensors preserve output shape, dtype, device and inputs."""
    for dtype in (torch.float32,torch.float64):
        tensors=[torch.randn(2,2,3,2,dtype=dtype).transpose(0,1) for _ in range(4)]
        value=torch.randn(2,2,3,4,dtype=dtype).transpose(0,1)
        before=[x.clone() for x in tensors+[value]]
        result=impl.differential_attention(*tensors,value,.3)
        assert result.shape==value.shape and result.dtype==dtype and result.device==value.device
        for a,b in zip(tensors+[value],before): close(a,b)
        params=[torch.randn(3,dtype=dtype) for _ in range(4)]
        coefficient=impl.differential_lambda(*params)
        assert coefficient.shape==() and coefficient.dtype==dtype


@pytest.mark.t0
def test_supplied_validation(impl):
    """Invalid projected dimensions and nonscalar/nonfinite coefficients are rejected."""
    q=torch.ones(1,1,2,2);v=torch.ones(1,1,2,4)
    with pytest.raises(ValueError):impl.differential_attention(q,q,q,q,v[:,:,:,:2],.5)
    with pytest.raises(ValueError):impl.differential_attention(q,q,q,q,v,torch.ones(2))
    with pytest.raises(ValueError):impl.differential_attention(q,q,q,q,v,float('nan'))
    with pytest.raises(ValueError):impl.differential_lambda(torch.ones(2),torch.ones(3),torch.ones(2),torch.ones(2))


@pytest.mark.t1
def test_independent_scalar_attention_oracle(impl):
    """Scalar softmax oracles cover distinct branch maps, scale, sign and masks for nondefault lambdas."""
    q1,k1,q2,k2=[torch.randn(2,2,3,2,dtype=torch.float64) for _ in range(4)]
    value=torch.randn(2,2,3,4,dtype=q1.dtype)
    for causal in (False,True):
        for coefficient in (-.4,0.,.7,1.2):
            expected=scalar_output(q1,k1,q2,k2,value,coefficient,causal)
            close(impl.differential_attention(q1,k1,q2,k2,value,coefficient,causal),expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t1
def test_lambda_parameterization_and_gradients(impl):
    """Independent exp/dot calculations and analytic derivatives check all four lambda parameter vectors."""
    params=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in ([.2,-.4],[.7,.3],[-.2,.8],[.4,-.5])]
    a=math.exp(sum(x*y for x,y in zip(params[0].detach().tolist(),params[1].detach().tolist())))
    b=math.exp(sum(x*y for x,y in zip(params[2].detach().tolist(),params[3].detach().tolist())))
    result=impl.differential_lambda(*params,.37)
    close(result,result.new_tensor(a-b+.37),atol=1e-12,rtol=1e-12)
    gradients=torch.autograd.grad(result,params)
    for actual,expected in zip(gradients,(a*params[1],a*params[0],-b*params[3],-b*params[2])):
        close(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_softmax_vjp_all_branches(impl):
    """Analytic softmax vector-Jacobian products validate Q/K/V and learnable lambda gradients."""
    q1,k1,q2,k2=[torch.randn(1,1,3,2,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    v=torch.randn(1,1,3,4,dtype=q1.dtype,requires_grad=True)
    coefficient=torch.tensor(.65,dtype=q1.dtype,requires_grad=True)
    upstream=torch.randn_like(v)
    for causal in (False,True):
        result=impl.differential_attention(q1,k1,q2,k2,v,coefficient,causal)
        actual=torch.autograd.grad((result*upstream).sum(),(q1,k1,q2,k2,v,coefficient))
        p1=scalar_probabilities(q1[0,0].detach(),k1[0,0].detach(),causal)
        p2=scalar_probabilities(q2[0,0].detach(),k2[0,0].detach(),causal)
        u=upstream[0,0] @ v[0,0].detach().T
        s1=p1*(u-(p1*u).sum(-1,keepdim=True))
        s2=-float(coefficient.detach())*p2*(u-(p2*u).sum(-1,keepdim=True))
        expected=(s1@k1[0,0].detach()/math.sqrt(2),s1.T@q1[0,0].detach()/math.sqrt(2),
                  s2@k2[0,0].detach()/math.sqrt(2),s2.T@q2[0,0].detach()/math.sqrt(2),
                  (p1-coefficient.detach()*p2).T@upstream[0,0],-(p2*u).sum())
        for a,b in zip(actual,expected):close(a.reshape(b.shape),b,atol=1e-11,rtol=1e-11)


@pytest.mark.t2
def test_common_mode_cancellation_causality_and_stability(impl):
    """Equal maps cancel at lambda=1; future inputs cannot leak; shared value offsets scale by 1-lambda."""
    q,k=torch.randn(1,1,4,2,dtype=torch.float64),torch.randn(1,1,4,2,dtype=torch.float64)
    v=torch.randn(1,1,4,4,dtype=q.dtype)
    close(impl.differential_attention(q,k,q,k,v,1.),torch.zeros_like(v),atol=1e-12,rtol=1e-12)
    q2,k2=q*.7,k.flip(-2)
    baseline=impl.differential_attention(q,k,q2,k2,v,.3)
    shift=torch.tensor([2.,-.2,3.,7.],dtype=q.dtype)
    close(impl.differential_attention(q,k,q2,k2,v+shift,.3)-baseline,torch.ones_like(v)*shift*.7,atol=1e-12,rtol=1e-12)
    changed=[x.clone() for x in (k,k2,v)]
    for x in changed:x[:,:,2:]+=50
    actual=impl.differential_attention(q,changed[0],q2,changed[1],changed[2],.3)
    close(actual[:,:,:2],baseline[:,:,:2],atol=1e-12,rtol=1e-12)
    assert torch.isfinite(impl.differential_attention(q*1000,k*1000,q2*1000,k2*1000,v,.3)).all()


@pytest.mark.t3
def test_submitted_common_mode_experiment(impl,artifact):
    """The submitted two-branch operator measures a lambda sweep with an independent common-offset law."""
    result=run(impl.differential_lambda,impl.differential_attention)
    artifact('common_mode_cancellation',result)
    m=result['metrics']
    assert m['common_mode_formula_error']<1e-10 and m['residual_at_lambda_one']<1e-10
    assert m['signal_at_lambda_one']>.5 and m['lambda_initialization_error']<1e-12
