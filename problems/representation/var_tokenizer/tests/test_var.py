import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.representation.var_tokenizer.experiment import run


def fixture(dtype=torch.float64):
    g = torch.Generator().manual_seed(42)
    features = torch.randn(2, 2, 3, 3, generator=g, dtype=dtype) * .4 + .3
    codes = torch.tensor([[-.7, -.4], [0., 0.], [.3, .8], [.9, -.2], [-.2, .5]], dtype=dtype)
    projections = torch.tensor([[[.8, .1], [-.2, 1.]], [[1.1, -.3], [.1, .9]], [[1., .1], [0., .8]]], dtype=dtype)
    return features, codes, [1, 2, 3], projections


def scalar_oracle(features, codes, scales, projections):
    """Explicit candidate distances and channel sums, independent of vectorized reference kernels."""
    residual = features.detach().clone()
    cumulative = torch.zeros_like(residual)
    all_tokens, all_images = [], []
    batch, channels, full, _ = features.shape
    for k, n in enumerate(scales):
        reduced = F.adaptive_avg_pool2d(residual, (n, n))
        indices = torch.empty(batch, n, n, dtype=torch.long)
        lookup = torch.empty(batch, channels, n, n, dtype=features.dtype)
        for b in range(batch):
            for y in range(n):
                for x in range(n):
                    distances = [sum(float(reduced[b, c, y, x] - code[c])**2 for c in range(channels)) for code in codes]
                    idx = min(range(len(codes)), key=lambda v: (distances[v], v))
                    indices[b, y, x] = idx
                    lookup[b, :, y, x] = codes[idx]
        up = F.interpolate(lookup, size=(full, full), mode='bicubic', align_corners=False)
        contribution = torch.stack([sum(projections[k, out, inc] * up[:, inc] for inc in range(channels)) for out in range(channels)], dim=1)
        cumulative = sum([cumulative, contribution])
        residual = features - cumulative
        all_tokens.append(indices)
        all_images.append(cumulative.clone())
    return all_tokens, all_images


@pytest.mark.t0
def test_shapes_frozen_outputs_and_no_input_mutation(impl):
    """Frozen target generation preserves input dtype/device and independent reconstruction snapshots."""
    for dtype in (torch.float32, torch.float64):
        features, codes, scales, projections = fixture(dtype)
        features = features.transpose(-1, -2).requires_grad_()
        codes.requires_grad_(); projections.requires_grad_()
        saved = features.detach().clone(), codes.detach().clone(), projections.detach().clone()
        tokens, images = impl.multiscale_quantize(features, codes, scales, projections)
        assert len(tokens) == len(images) == len(scales)
        for n, token, image in zip(scales, tokens, images):
            assert token.shape == (2, n, n) and token.dtype == torch.long
            assert image.shape == features.shape and image.dtype == dtype and image.device == features.device
            assert not image.requires_grad
        assert len({x.data_ptr() for x in images}) == len(scales)
        close(features, saved[0]); close(codes, saved[1]); close(projections, saved[2])
        assert not impl.decode_multiscale(tokens, codes, projections).requires_grad


@pytest.mark.t0
def test_domain_validation(impl):
    """The final resolution, scale order, projection shape and token range are checked."""
    f, c, s, p = fixture()
    for bad in ([], [1, 2], [1, 3, 2], [1, 1, 3], [2, 3]):
        with pytest.raises(ValueError): impl.multiscale_quantize(f, c, bad, p)
    with pytest.raises(ValueError): impl.multiscale_quantize(f, c, s, p[:1])
    with pytest.raises(ValueError): impl.decode_multiscale([torch.tensor([[[len(c)]]])], c, p[:1])


@pytest.mark.t1
def test_independent_scalar_residual_vq_oracle(impl):
    """Nonuniform scales and nonidentity projections exercise residual update, nearest code and full-scale accumulation."""
    features, codes, scales, projections = fixture()
    want_tokens, want_images = scalar_oracle(features, codes, scales, projections)
    got_tokens, got_images = impl.multiscale_quantize(features, codes, scales, projections)
    for got, want in zip(got_tokens, want_tokens): close(got, want, atol=0, rtol=0)
    for got, want in zip(got_images, want_images): close(got, want, atol=1e-12, rtol=1e-12)
    close(impl.decode_multiscale(want_tokens, codes, projections), want_images[-1], atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_hand_computed_residual_tokens(impl):
    """A coarse .5 code leaves signed .25 residuals; quantizing the original map yields different indices."""
    f = torch.tensor([[[[.25, .75], [.25, .75]]]], dtype=torch.float64)
    c = torch.tensor([[-.25], [0.], [.25], [.5], [.75]], dtype=f.dtype)
    p = torch.ones(2, 1, 1, dtype=f.dtype)
    tokens, images = impl.multiscale_quantize(f, c, [1, 2], p)
    close(tokens[0], torch.tensor([[[3]]]))
    close(tokens[1], torch.tensor([[[0, 2], [0, 2]]]))
    close(images[0], torch.full_like(f, .5)); close(images[1], f)
    close(impl.decode_multiscale(tokens, c, p), f)


@pytest.mark.t2
def test_ties_single_scale_and_zero_residual(impl):
    """Euclidean ties choose the first code; exact reconstruction leaves a zero residual at later scales."""
    c = torch.tensor([[-1.], [1.], [0.]], dtype=torch.float64)
    f = torch.tensor([[[[.5]]]], dtype=torch.float64)
    tokens, images = impl.multiscale_quantize(f, c, [1], torch.ones(1, 1, 1, dtype=f.dtype))
    close(tokens[0], torch.tensor([[[1]]]))
    close(images[0], torch.ones_like(f))
    close(impl.decode_multiscale(tokens, c, torch.ones(1, 1, 1, dtype=f.dtype)), images[0])
    f = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    tokens, images = impl.multiscale_quantize(f, c, [1, 2, 4], torch.ones(3, 1, 1, dtype=f.dtype))
    close(tokens[0], torch.ones(1, 1, 1, dtype=torch.long))
    for token in tokens[1:]: close(token, torch.full_like(token, 2))
    for image in images: close(image, f)


@pytest.mark.t2
def test_decode_adds_each_scale_and_respects_channels(impl):
    """Decoding arbitrary codes is an additive lifted sum, with no averaging or spatial size loss."""
    _, c, s, p = fixture()
    maps = [torch.full((1, n, n), k, dtype=torch.long) for k, n in enumerate(s)]
    expected_pixel = sum(p[k] @ c[k] for k in range(3))
    expected = expected_pixel[None, :, None, None].expand(1, 2, 3, 3)
    close(impl.decode_multiscale(maps, c, p), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_batch_independence_and_final_quantization_bound(impl):
    """Independent examples stay independent; a dense scalar codebook bounds the last-scale residual."""
    features, c, s, p = fixture()
    all_tokens, all_images = impl.multiscale_quantize(features, c, s, p)
    for b in range(2):
        tokens, images = impl.multiscale_quantize(features[b:b+1], c, s, p)
        for a, z in zip(tokens, all_tokens): close(a, z[b:b+1])
        for a, z in zip(images, all_images): close(a, z[b:b+1])
    c = torch.linspace(-2., 2., 81, dtype=torch.float64)[:, None]
    f = torch.linspace(-1., 1., 16, dtype=torch.float64).reshape(1, 1, 4, 4)
    _, images = impl.multiscale_quantize(f, c, [1, 2, 4], torch.ones(3, 1, 1, dtype=f.dtype))
    assert (images[-1] - f).abs().max() <= .025 + 1e-12


@pytest.mark.t3
def test_toy_residual_detail_reconstruction(impl, artifact):
    """The learner's quantizer and decoder recover a synthetic latent's residual detail."""
    result = run(impl.multiscale_quantize, impl.decode_multiscale)
    artifact('var_residual_latent', result)
    assert result['metrics']['final_mse'] < .004
    assert result['metrics']['final_mse'] < result['metrics']['coarse_mse'] * .2
    assert result['metrics']['encode_decode_error'] < 1e-12
