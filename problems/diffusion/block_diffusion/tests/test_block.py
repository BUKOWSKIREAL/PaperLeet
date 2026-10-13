import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion.block_diffusion.experiment import run


def allowed_key_lists(length,block_size):
    # Construct local denoising contexts as lists, without a 2L-by-2L mask formula.
    blocks=[list(range(start,min(start+block_size,length))) for start in range(0,length,block_size)]
    contexts=[None]*(2*length)
    prefix=[]
    for block in blocks:
        for position in block:
            contexts[position]=block+[length+j for j in prefix]
            contexts[length+position]=[length+j for j in prefix+block]
        prefix+=block
    return contexts


def local_scalar_oracle(q,k,v,block_size):
    result=torch.zeros_like(v)
    contexts=allowed_key_lists(q.shape[-2]//2,block_size)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for i,keys in enumerate(contexts):
                logits=[sum(float(q[b,h,i,c])*float(k[b,h,j,c]) for c in range(q.shape[-1]))/math.sqrt(q.shape[-1]) for j in keys]
                exps=[math.exp(x-max(logits)) for x in logits]
                weights=[x/sum(exps) for x in exps]
                for d in range(v.shape[-1]):
                    result[b,h,i,d]=sum(w*float(v[b,h,j,d]) for w,j in zip(weights,keys))
    return result


@pytest.mark.t0
def test_mask_and_attention_interfaces(impl):
    """Masks are bool on the requested device; strided Q/K/V preserve shapes, dtype/device and values."""
    for dtype in (torch.float32,torch.float64):
        q,k=[torch.randn(2,2,10,3,dtype=dtype).transpose(0,1) for _ in range(2)]
        v=torch.randn(2,2,10,2,dtype=dtype).transpose(0,1)
        before=q.clone(),k.clone(),v.clone()
        mask=impl.block_diffusion_mask(5,2,device=q.device)
        assert mask.shape==(10,10) and mask.dtype==torch.bool and mask.device==q.device
        out=impl.block_diffusion_attention(q,k,v,2)
        assert out.shape==v.shape and out.dtype==dtype and out.device==v.device
        for x,y in zip((q,k,v),before):close(x,y)


@pytest.mark.t0
def test_supplied_validation(impl):
    """Reject nonpositive sizes, noninteger sizes and inconsistent concatenated attention shapes."""
    for length,block in ((0,2),(5,0),(-1,2),(5,1.5),(True,2)):
        with pytest.raises(ValueError):impl.block_diffusion_mask(length,block)
    q=torch.zeros(1,1,5,2)
    with pytest.raises(ValueError):impl.block_diffusion_attention(q,q,q,2)
    q=torch.zeros(1,1,6,2)
    with pytest.raises(ValueError):impl.block_diffusion_attention(q,q,q[:,:,:4],2)


@pytest.mark.t1
def test_paper_mask_and_ragged_block_contexts(impl):
    """Independent block-by-block contexts include noisy own-block keys and strictly earlier clean blocks."""
    for length,block in ((6,2),(5,2),(4,1),(4,4),(3,8)):
        actual=impl.block_diffusion_mask(length,block)
        expected=torch.zeros(2*length,2*length,dtype=torch.bool)
        for i,keys in enumerate(allowed_key_lists(length,block)):expected[i,keys]=True
        assert torch.equal(actual,expected)
    mask=impl.block_diffusion_mask(6,2)
    assert mask[0,1] and not mask[0,6] and mask[2,6] and not mask[2,8]
    assert mask[6,7] and not mask[6,0]


@pytest.mark.t1
def test_vectorized_attention_matches_individual_conditionals(impl):
    """Scalar per-block denoising conditionals match the vectorized pass, including unequal head/value widths."""
    for length,block in ((6,2),(5,2),(3,1),(3,8)):
        q,k=[torch.randn(1,2,2*length,3,dtype=torch.float64) for _ in range(2)]
        v=torch.randn(1,2,2*length,2,dtype=q.dtype)
        close(impl.block_diffusion_attention(q,k,v,block),local_scalar_oracle(q,k,v,block),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_forbidden_influence_and_analytic_vjp(impl):
    """Independent local softmax derivatives verify Q/K/V gradients; forbidden clean and noisy keys have zero influence."""
    q,k=[torch.randn(1,1,12,2,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    v=torch.randn(1,1,12,3,dtype=q.dtype,requires_grad=True)
    output=impl.block_diffusion_attention(q,k,v,2)
    upstream=v.new_tensor([.4,-.7,1.3])
    dq,dk,dv=torch.autograd.grad((output[0,0,2]*upstream).sum(),(q,k,v))
    keys=[2,3,6,7]  # The second noisy block and first clean block only.
    logits=[sum(float(q[0,0,2,c].detach())*float(k[0,0,j,c].detach()) for c in range(2))/math.sqrt(2) for j in keys]
    exps=[math.exp(x-max(logits)) for x in logits]
    weights=[x/sum(exps) for x in exps]
    value_dots=[sum(float(v[0,0,j,c].detach())*float(upstream[c]) for c in range(3)) for j in keys]
    baseline=sum(w*d for w,d in zip(weights,value_dots))
    expected_q=torch.zeros_like(q);expected_k=torch.zeros_like(k);expected_v=torch.zeros_like(v)
    for j,w,dot in zip(keys,weights,value_dots):
        ds=w*(dot-baseline)
        expected_q[0,0,2]+=ds*k[0,0,j].detach()/math.sqrt(2)
        expected_k[0,0,j]=ds*q[0,0,2].detach()/math.sqrt(2)
        expected_v[0,0,j]=w*upstream
    for actual,expected in ((dq,expected_q),(dk,expected_k),(dv,expected_v)):
        close(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_clean_stream_isolation_and_bidirectional_noisy_block(impl):
    """Clean queries cannot read noisy tokens; noisy tokens share their whole block, including later positions."""
    q,k=torch.zeros(1,1,12,2,dtype=torch.float64),torch.zeros(1,1,12,2,dtype=torch.float64)
    v=torch.arange(12,dtype=q.dtype).reshape(1,1,12,1)
    baseline=impl.block_diffusion_attention(q,k,v,2)
    changed=v.clone();changed[:,:,:6]+=100
    close(impl.block_diffusion_attention(q,k,changed,2)[:,:,6:],baseline[:,:,6:])
    changed=v.clone();changed[:,:,1]+=2
    difference=impl.block_diffusion_attention(q,k,changed,2)-baseline
    close(difference[0,0,0,0],q.new_tensor(1.))
    close(difference[0,0,2:6],torch.zeros_like(difference[0,0,2:6]))
    assert torch.isfinite(impl.block_diffusion_attention(q+1000,k+1000,v,2)).all()


@pytest.mark.t3
def test_paper_derived_six_token_information_flow(impl,artifact):
    """The submitted mask and attention recreate Figure 3 and measure all individual token perturbations."""
    result=run(impl.block_diffusion_mask,impl.block_diffusion_attention)
    artifact('six_token_block_dependencies',result)
    m=result['metrics']
    assert m['mask_mismatches']==0
    assert m['maximum_forbidden_influence']<1e-12
    assert m['minimum_allowed_influence']>.01
    assert m['attention_row_sum_error']<1e-12
