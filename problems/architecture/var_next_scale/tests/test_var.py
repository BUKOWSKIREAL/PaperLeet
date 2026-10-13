import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.var_next_scale.experiment import run


def fixture(dtype=torch.float64):
    tokens = [torch.tensor([[[1]], [[2]]]), torch.tensor([[[0, 1], [2, 3]], [[3, 2], [1, 0]]]), (torch.arange(32) % 4).reshape(2, 4, 4)]
    codes = torch.tensor([[0., .5], [1., -.3], [-.8, .1], [.4, 1.2]], dtype=dtype)
    projections = torch.tensor([[[1., .2], [-.1, .9]], [[.8, -.3], [.4, 1.1]], [[1., 0.], [0., 1.]]], dtype=dtype)
    return tokens, codes, projections


def oracle(maps, codes, projections):
    """Scalar code lookup/channel sums; reconstruct each prefix independently."""
    full, batch, channels = maps[-1].shape[-1], maps[0].shape[0], codes.shape[1]
    outputs = []
    for target in range(1, len(maps)):
        terms = []
        for prior in range(target):
            n = maps[prior].shape[-1]
            feature = codes.new_empty(batch, channels, n, n)
            for b in range(batch):
                for c in range(channels):
                    for y in range(n):
                        for x in range(n):
                            feature[b, c, y, x] = codes[int(maps[prior][b, y, x]), c]
            enlarged = F.interpolate(feature, size=(full, full), mode='bicubic', align_corners=False)
            projected = torch.stack([sum(projections[prior, out, inc] * enlarged[:, inc] for inc in range(channels)) for out in range(channels)], dim=1)
            terms.append(projected)
        n = maps[target].shape[-1]
        outputs.append(F.adaptive_avg_pool2d(sum(terms), (n, n)).flatten(2).transpose(1, 2))
    return torch.cat(outputs, dim=1) if outputs else codes.new_empty(batch, 0, channels)


@pytest.mark.t0
def test_shapes_dtypes_and_frozen_inputs(impl):
    """Teacher features preserve dtype/device and input values; tokenizer parameters stay frozen."""
    for dtype in (torch.float32, torch.float64):
        maps, codes, proj = fixture(dtype)
        maps = [m.transpose(-1, -2) for m in maps]
        codes.requires_grad_(); proj.requires_grad_()
        before = [m.clone() for m in maps]
        out = impl.next_scale_inputs(maps, codes, proj)
        assert out.shape == (2, 20, 2) and out.dtype == dtype and out.device == codes.device
        assert not out.requires_grad
        for a, b in zip(maps, before): close(a, b)
        mask = impl.scale_causal_mask([1, 2, 4])
        assert mask.shape == (21, 21) and mask.dtype == torch.bool


@pytest.mark.t0
def test_validation(impl):
    """Invalid scale schedules and out-of-vocabulary token targets fail explicitly."""
    for scales in ([], [2, 4], [1, 1], [1, 0], [1, 2.5]):
        with pytest.raises(ValueError): impl.scale_causal_mask(scales)
    maps, codes, proj = fixture()
    with pytest.raises(ValueError): impl.next_scale_inputs(maps, codes, proj[:1])
    maps[-1][0, 0, 0] = len(codes)
    with pytest.raises(ValueError): impl.next_scale_inputs(maps, codes, proj)


@pytest.mark.t1
def test_independent_prefix_reconstruction_oracle(impl):
    """Each scale consumes the area-resized sum of all earlier lifted residual codes."""
    for final_size in (3, 4):
        maps, codes, proj = fixture()
        maps[-1] = maps[-1][:, :final_size, :final_size]
        close(impl.next_scale_inputs(maps, codes, proj), oracle(maps, codes, proj), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_exact_scale_visibility(impl):
    """Every pair within a scale is visible, including higher raster positions; later scales are blocked."""
    levels = [0] + [1]*4 + [2]*9
    expected = torch.tensor([[key <= query for key in levels] for query in levels])
    close(impl.scale_causal_mask([1, 2, 3]), expected)
    assert bool(impl.scale_causal_mask([1, 2])[1, 4])


@pytest.mark.t2
def test_target_and_future_perturbations_do_not_change_own_inputs(impl):
    """Changing a target map affects only strictly finer scale teacher inputs."""
    maps, codes, proj = fixture()
    base = impl.next_scale_inputs(maps, codes, proj)
    for k, unaffected in ((1, 4), (2, 20)):
        changed = [m.clone() for m in maps]; changed[k] = (changed[k]+1) % len(codes)
        out = impl.next_scale_inputs(changed, codes, proj)
        close(out[:, :unaffected], base[:, :unaffected], atol=0, rtol=0)
        if k == 1: assert (out[:, 4:] - base[:, 4:]).abs().max() > .1


@pytest.mark.t2
def test_all_prefix_scales_contribute(impl):
    """The coarse 1×1 code remains in the 4×4 input after 2×2 residual codes are added."""
    maps, codes, proj = fixture()
    changed = [m.clone() for m in maps]; changed[0] = (changed[0]+1) % len(codes)
    a = impl.next_scale_inputs(maps, codes, proj)
    b = impl.next_scale_inputs(changed, codes, proj)
    assert (a[:, 4:] - b[:, 4:]).abs().max() > .1
    close(b, oracle(changed, codes, proj), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_single_scale_and_batch_independence(impl):
    """A lone start scale has an empty tokenizer prefix; batching cannot mix examples."""
    maps, codes, proj = fixture()
    out = impl.next_scale_inputs(maps[:1], codes, proj[:1])
    assert out.shape == (2, 0, 2)
    close(impl.scale_causal_mask([1]), torch.ones(1, 1, dtype=torch.bool))
    full = impl.next_scale_inputs(maps, codes, proj)
    for b in range(2): close(full[b:b+1], impl.next_scale_inputs([m[b:b+1] for m in maps], codes, proj))


@pytest.mark.t3
def test_toy_attention_has_no_target_leakage(impl, artifact):
    """The learner's inputs and mask jointly prevent current/future target leakage through attention."""
    result = run(impl.next_scale_inputs, impl.scale_causal_mask)
    artifact('var_scale_causality', result)
    assert result['metrics']['forbidden_target_influence'] < 1e-12
    assert result['metrics']['minimum_prefix_influence'] > .05
    assert result['metrics']['within_scale_2_visible_pairs'] == 16
