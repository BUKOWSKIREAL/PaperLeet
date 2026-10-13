import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.svdquant.experiment import run


def quant_oracle(x,bits=4):
    data=x.tolist(); flat=x.flatten().tolist(); maximum=max(abs(v) for v in flat)
    scale=maximum/(2**(bits-1)-1) if maximum else 1.
    return torch.tensor([round(v/scale)*scale for v in flat],dtype=x.dtype).reshape(x.shape)


def known_matrix():
    # Distinct singular values with known rotated singular vectors.
    c,s=math.cos(.43),math.sin(.43)
    u=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.],[0.,0.,0.]],dtype=torch.float64)
    v=torch.tensor([[.8,-.6,0.],[.6,.8,0.],[0.,0.,1.]],dtype=torch.float64)
    singular=torch.tensor([9.,2.,.3],dtype=u.dtype)
    return u@torch.diag(singular)@v.T,u,v,singular


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    for dtype in (torch.float32,torch.float64):
        x=torch.arange(24,dtype=dtype).reshape(4,6)[:,::2]; w=torch.arange(15,dtype=dtype).reshape(5,3).T
        scale=torch.tensor([2.,.5,3.],dtype=dtype);before=[a.clone() for a in (x,w,scale)]
        xs,ws=impl.smooth_inputs(x,w,scale)
        left,right,residual=impl.split_weight(ws,2)
        assert xs.shape==x.shape and ws.shape==w.shape and xs.dtype==dtype and ws.dtype==dtype
        assert left.shape==(3,2) and right.shape==(2,5) and residual.shape==w.shape
        out=impl.svdquant_linear(x,w,scale,2)
        assert out.shape==(4,5) and out.dtype==dtype and out.device==x.device
        for a,b in zip((x,w,scale),before):close(a,b)


@pytest.mark.t0
def test_supplied_validation(impl):
    x=torch.ones(2,3);w=torch.ones(3,4)
    for scale in (torch.tensor([1.,0.,1.]),torch.tensor([1.,float('nan'),1.]),torch.ones(4)):
        with pytest.raises(ValueError):impl.smooth_inputs(x,w,scale)
    for rank in (-1,4,1.5):
        with pytest.raises(ValueError):impl.split_weight(w,rank)
    for bits in (1,9,2.5):
        with pytest.raises(ValueError):impl.svdquant_linear(x,w,torch.ones(3),1,bits)


@pytest.mark.t1
def test_smoothing_orientation_and_linear_identity(impl):
    x=torch.tensor([[1.,3.,-2.],[.2,-.5,1.]],dtype=torch.float64)
    w=torch.tensor([[1.,2.],[-3.,.5],[.8,-2.]],dtype=x.dtype);scale=torch.tensor([2.,.25,4.],dtype=x.dtype)
    xs,ws=impl.smooth_inputs(x,w,scale)
    expected_x=torch.tensor([[float(x[i,j])/float(scale[j]) for j in range(3)] for i in range(2)],dtype=x.dtype)
    expected_w=torch.tensor([[float(w[i,j])*float(scale[i]) for j in range(2)] for i in range(3)],dtype=x.dtype)
    close(xs,expected_x);close(ws,expected_w);close(xs@ws,x@w,atol=1e-12,rtol=1e-12)


@pytest.mark.t1
def test_known_spectrum_optimal_lowrank_and_residual(impl):
    w,u,v,s=known_matrix()
    for rank in range(4):
        left,right,residual=impl.split_weight(w,rank)
        expected=u[:,:rank]@torch.diag(s[:rank])@v[:,:rank].T
        close(left@right,expected,atol=1e-11,rtol=1e-11)
        close(residual,w-expected,atol=1e-11,rtol=1e-11)
        assert abs(float(residual.square().sum())-float(s[rank:].square().sum()))<1e-10


@pytest.mark.t2
def test_two_branches_against_scalar_quantization_oracle(impl):
    w,u,v,s=known_matrix();scale=torch.tensor([2.,.7,3.,.5],dtype=w.dtype)
    # Choose original W so that smoothed W has the independently known SVD.
    original_w=w/scale[:,None]
    x=torch.tensor([[.37,-1.8,.4,2.1],[1.1,.63,-.2,-.9]],dtype=w.dtype)*scale
    xs=x/scale
    for rank in (0,1,2,3):
        low=u[:,:rank]@torch.diag(s[:rank])@v[:,:rank].T
        for bits in (3,4,8):
            expected=xs@low+quant_oracle(xs,bits)@quant_oracle(w-low,bits)
            close(impl.svdquant_linear(x,original_w,scale,rank,bits),expected,atol=1e-10,rtol=1e-10)


@pytest.mark.t2
def test_full_rank_retains_unquantized_activation(impl):
    w=torch.tensor([[7.,1.2],[.3,2.1]],dtype=torch.float64)
    x=torch.tensor([[.031,2.7],[1.123,-.041]],dtype=w.dtype);scale=torch.tensor([2.,.3],dtype=w.dtype)
    actual=impl.svdquant_linear(x,w,scale,2)
    close(actual,x@w,atol=1e-11,rtol=1e-11)
    assert (actual-quant_oracle(x/scale)@(w*scale[:,None])).abs().max()>.05


@pytest.mark.t2
def test_zero_weights_zero_activations_and_rank_zero(impl):
    x=torch.tensor([[1.,-.3,2.],[.4,.7,-1.]],dtype=torch.float64)
    w=torch.zeros(3,2,dtype=x.dtype);scale=torch.tensor([1.,2.,.3],dtype=x.dtype)
    for rank in (0,1,2):
        close(impl.svdquant_linear(x,w,scale,rank),torch.zeros(2,2,dtype=x.dtype))
        close(impl.svdquant_linear(torch.zeros_like(x),torch.ones_like(w),scale,rank),torch.zeros(2,2,dtype=x.dtype))
    w=torch.tensor([[1.,.3],[-.7,2.1],[1.8,-.3]],dtype=x.dtype)
    close(impl.svdquant_linear(x,w,scale,0),quant_oracle(x/scale)@quant_oracle(w*scale[:,None]))


@pytest.mark.t2
def test_repeated_singular_values_compare_products_not_signs(impl):
    w=torch.diag(torch.tensor([3.,3.,1.],dtype=torch.float64))
    left,right,residual=impl.split_weight(w,1)
    close(left@right+residual,w)
    assert abs(float(residual.square().sum())-10.)<1e-10
    assert torch.linalg.matrix_rank(left@right).item()==1


@pytest.mark.t3
def test_outlier_channel_toy(impl,artifact):
    result=run(impl.smooth_inputs,impl.split_weight,impl.svdquant_linear)
    artifact('outlier_migration',result)
    m=result['metrics']
    assert m['smoothing_identity_error']<1e-10
    assert m['full_rank_relative_error']<1e-12
    assert m['rank1_relative_error']<m['naive_relative_error']*.3
    assert m['residual_absmax']<m['weight_absmax']*.2
