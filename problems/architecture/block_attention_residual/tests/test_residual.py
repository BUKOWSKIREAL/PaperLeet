import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.block_attention_residual.experiment import run


def inputs(gradients=False):
    rng=torch.Generator().manual_seed(343)
    sources=torch.randn(4,2,3,5,generator=rng,dtype=torch.float64)
    query=torch.randn(5,generator=rng,dtype=torch.float64)
    scale=torch.randn(5,generator=rng,dtype=torch.float64)
    return [x.requires_grad_(gradients) for x in (sources,query,scale)]


def scalar_oracle(sources,query,scale,eps=1e-6):
    # One token at a time; torch RMSNorm provides an independent normalization oracle.
    batch_outputs, batch_weights=[],[]
    for b in range(sources.shape[1]):
        token_outputs,token_weights=[],[]
        for t in range(sources.shape[2]):
            values=sources[:,b,t]
            keys=torch.nn.functional.rms_norm(values,(values.shape[-1],),weight=scale,eps=eps)
            logits=torch.stack([torch.dot(key,query) for key in keys])
            weights=logits.softmax(0)
            token_outputs.append(sum(weights[i]*values[i] for i in range(len(values))))
            token_weights.append(weights)
        batch_outputs.append(torch.stack(token_outputs))
        batch_weights.append(torch.stack(token_weights))
    return torch.stack(batch_outputs),torch.stack(batch_weights).permute(2,0,1)


@pytest.mark.t0
def test_source_order_shapes_and_preservation(impl):
    """The embedding is first, current partial block is last, and the caller's list is unchanged."""
    sources,query,scale=inputs()
    embedding,completed,partial=sources[0],[sources[1],sources[2]],sources[3]
    before=sources.clone()
    stacked=impl.assemble_sources(embedding,completed,partial)
    close(stacked,sources)
    assert len(completed)==2
    output,weights=impl.depth_attention(stacked,query,scale)
    assert output.shape==(2,3,5) and weights.shape==(4,2,3)
    assert output.dtype==weights.dtype==torch.float64 and output.device==sources.device
    close(sources,before)


@pytest.mark.t1
def test_independent_tokenwise_oracle(impl):
    """Only keys are RMS-normalized and logits normalize over depth without sqrt(d) scaling."""
    tensors=inputs()
    actual=impl.depth_attention(*tensors)
    expected=scalar_oracle(*tensors)
    for x,y in zip(actual,expected): close(x,y,atol=1e-10,rtol=1e-10)


@pytest.mark.t2
def test_missing_partial_and_real_zero_source(impl):
    """No partial block means no extra softmax slot; an explicit zero tensor remains a source."""
    sources,query,scale=inputs()
    embedding=sources[0]
    alone=impl.assemble_sources(embedding,[],None)
    assert alone.shape==(1,2,3,5)
    out,weights=impl.depth_attention(alone,query,scale)
    close(out,embedding)
    close(weights,torch.ones_like(weights))
    with_zero=impl.assemble_sources(embedding,[],torch.zeros_like(embedding))
    assert with_zero.shape==(2,2,3,5)
    out,weights=impl.depth_attention(with_zero,torch.zeros_like(query),scale)
    close(out,embedding/2)
    close(weights,torch.full_like(weights,.5))


@pytest.mark.t2
def test_uniform_query_and_original_value_magnitudes(impl):
    """A zero pseudo-query produces the arithmetic mean of original, unnormalized values."""
    sources,query,scale=inputs()
    sources=sources*torch.tensor([1.,10.,.1,5.],dtype=sources.dtype)[:,None,None,None]
    out,weights=impl.depth_attention(sources,torch.zeros_like(query),scale)
    close(out,sources.mean(dim=0))
    close(weights,torch.full_like(weights,.25))


@pytest.mark.t2
def test_gradients_through_assembly_norm_query_and_values(impl):
    """Source, pseudo-query and norm-scale gradients match an independent RMSNorm oracle."""
    tensors=inputs(gradients=True)
    independent=[x.detach().clone().requires_grad_() for x in tensors]
    source,query,scale=tensors
    stack=impl.assemble_sources(source[0],[source[1],source[2]],source[3])
    out,weights=impl.depth_attention(stack,query,scale)
    expected,expected_weights=scalar_oracle(*independent)
    objective=out.square().sum()+.2*weights.square().sum()
    oracle_objective=expected.square().sum()+.2*expected_weights.square().sum()
    for x,y in zip(torch.autograd.grad(objective,tensors),torch.autograd.grad(oracle_objective,independent)):
        close(x,y,atol=1e-8,rtol=1e-8)


@pytest.mark.t2
def test_depth_permutation_strides_and_zero_norm(impl):
    """Depth permutation only permutes weights; zero sources and large logits remain finite."""
    tensors=[x.transpose(-1,-2).contiguous().transpose(-1,-2) if x.ndim>1 else x for x in inputs()]
    sources,query,scale=tensors
    sources[1]=0
    query=query*1000
    out,weights=impl.depth_attention(sources,query,scale,eps=1e-4)
    assert torch.isfinite(out).all() and torch.isfinite(weights).all()
    permutation=torch.tensor([2,0,3,1])
    altered,permuted_weights=impl.depth_attention(sources[permutation],query,scale,eps=1e-4)
    close(altered,out)
    close(permuted_weights,weights[permutation])
    expected,expected_weights=scalar_oracle(sources,query,scale,eps=1e-4)
    close(out,expected)
    close(weights,expected_weights)


@pytest.mark.t3
def test_depth_selection_toy(impl,artifact):
    """Increasing query strength selects the useful embedding among synthetic depth sources."""
    result=run(impl.assemble_sources,impl.depth_attention)
    artifact('depth_selection',result)
    metrics=result['metrics']
    assert metrics['maximum_error']<1e-12
    assert metrics['selected_source_weight']>.999
    assert metrics['selected_mse']<1e-6
    assert metrics['uniform_mse']>.5
