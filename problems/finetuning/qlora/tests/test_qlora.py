import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.qlora.experiment import run


@pytest.mark.t0
def test_shapes_tail_and_zero_block(impl):
    """Unpacked codes retain shape, zero blocks decode exactly, and tail blocks have their own scale."""
    weight = torch.zeros(3, 5, dtype=torch.float64)
    weight[-1, -1] = -2.
    codes, scales = impl.nf4_quantize(weight, 4)
    assert codes.shape == weight.shape and codes.dtype == torch.uint8
    assert scales.shape == (4,) and scales.dtype == weight.dtype
    assert torch.all(codes.flatten()[:12] == 7)
    close(impl.nf4_dequantize(codes, scales, 4), weight)
    with pytest.raises(ValueError):
        impl.nf4_quantize(weight, 0)


@pytest.mark.t1
def test_reference(impl, ref):
    """Non-square weights and uneven blocks agree in quantization, decoding and adapter forward."""
    weight = torch.randn(3, 5, dtype=torch.float64)
    codes, scales = impl.nf4_quantize(weight, 4)
    ec, es = ref.nf4_quantize(weight, 4)
    assert torch.equal(codes, ec)
    close(scales, es)
    close(impl.nf4_dequantize(codes, scales, 4), ref.nf4_dequantize(ec, es, 4))
    x, a, b = torch.randn(2, 5, dtype=weight.dtype), torch.randn(2, 5, dtype=weight.dtype), torch.randn(3, 2, dtype=weight.dtype)
    close(impl.qlora_linear(x, codes, scales, a, b, 3., 4), ref.qlora_linear(x, codes, scales, a, b, 3., 4))


@pytest.mark.t2
def test_nf4_asymmetry_exact_zero_and_scale(impl):
    """NF4 has an exact zero and asymmetric nearest-neighbor bins; each block uses absmax."""
    values = torch.tensor([[-1., -.09, 0., .08, 1., 0., 0., 0.],
                           [-10., -.9, 0., .8, 10., 0., 0., 0.]], dtype=torch.float64)
    codes, scales = impl.nf4_quantize(values, 8)
    assert codes[0].tolist() == [0, 6, 7, 8, 15, 7, 7, 7]
    assert torch.equal(codes[0], codes[1])
    close(scales, torch.tensor([1., 10.], dtype=values.dtype))
    decoded = impl.nf4_dequantize(codes, scales, 8)
    assert decoded[0, 2] == 0
    close(decoded[1], 10 * decoded[0])
    assert abs(float(decoded[0, 1] + decoded[0, 3])) > .01


@pytest.mark.t2
def test_block_locality_and_positive_rescaling(impl):
    """Changing one block cannot alter another block's indices or scale."""
    w = torch.randn(12, dtype=torch.float64)
    c, s = impl.nf4_quantize(w, 4)
    changed = w.clone(); changed[4:8] *= 100
    dc, ds = impl.nf4_quantize(changed, 4)
    assert torch.equal(c, dc)
    close(ds, s * torch.tensor([1., 100., 1.], dtype=s.dtype))
    reconstructed = impl.nf4_dequantize(c, s, 4)
    close(impl.nf4_dequantize(dc, ds, 4)[:4], reconstructed[:4])


@pytest.mark.t2
def test_frozen_base_and_adapter_gradients(impl):
    """Quantized weights, scales and bias stay frozen; adapters and inputs retain their analytical gradients."""
    w = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    codes, scales = impl.nf4_quantize(w, 4)
    assert not scales.requires_grad
    scales = scales.detach().requires_grad_()
    x = torch.randn(4, 5, dtype=w.dtype, requires_grad=True)
    a = torch.randn(2, 5, dtype=w.dtype, requires_grad=True)
    b = torch.randn(3, 2, dtype=w.dtype, requires_grad=True)
    bias = torch.randn(3, dtype=w.dtype, requires_grad=True)
    impl.qlora_linear(x, codes, scales, a, b, 6., 4, bias).sum().backward()
    assert w.grad is None
    assert scales.grad is None or torch.count_nonzero(scales.grad) == 0
    assert bias.grad is None or torch.count_nonzero(bias.grad) == 0
    base = impl.nf4_dequantize(codes, scales, 4)
    close(x.grad, (base.detach() + 3 * b.detach() @ a.detach()).sum(0).expand_as(x))
    close(a.grad, 3 * b.detach().sum(0)[:, None] * x.detach().sum(0)[None, :])
    close(b.grad, 3 * (x.detach() @ a.detach().T).sum(0).expand_as(b))


@pytest.mark.t3
def test_quantized_adapter_regression(impl, artifact):
    """Learning on a frozen NF4 map recovers a low-rank target change on the submitted implementation."""
    out = run(impl.nf4_quantize, impl.nf4_dequantize, impl.qlora_linear)
    artifact('nf4_adapter_regression', out)
    assert out['final_mse'] < 1e-5, f"Regression MSE: {out['final_mse']}"
    assert out['quantization_rmse'] < .05
