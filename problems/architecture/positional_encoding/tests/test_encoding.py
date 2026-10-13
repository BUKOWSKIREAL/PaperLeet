import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.positional_encoding.experiment import run


@pytest.mark.t0
def test_shape_dtype_and_empty_sequence(impl):
    """Requested shape, device and precision are retained, including a zero-length sequence."""
    for dtype in (torch.float32, torch.float64):
        for length, width in ((4, 8), (3, 5), (2, 1), (0, 3)):
            got = impl.sinusoidal_encoding(length, width, dtype=dtype, device=torch.device('cpu'))
            assert got.shape == (length, width)
            assert got.dtype == dtype and got.device.type == 'cpu'
            assert torch.isfinite(got).all()


@pytest.mark.t1
def test_scalar_oracle_and_pair_frequencies(impl):
    """Every channel agrees with Python's scalar sine/cosine at the paper's 2i/d frequency."""
    for width, start in ((1, 0), (3, 5), (8, 0), (8, 10000)):
        expected = [[(math.sin if dim % 2 == 0 else math.cos)(position / 10000.0 ** (2 * (dim // 2) / width))
                     for dim in range(width)] for position in range(start, start + 4)]
        got = impl.sinusoidal_encoding(4, width, start=start, dtype=torch.float64)
        close(got, torch.tensor(expected, dtype=torch.float64), atol=2e-12, rtol=2e-12)


@pytest.mark.t2
def test_origin_and_unit_pair_norm(impl):
    """Position zero is [0,1,...]; each complete sine/cosine pair has unit squared norm."""
    got = impl.sinusoidal_encoding(9, 7, dtype=torch.float64)
    close(got[0], torch.tensor([0., 1., 0., 1., 0., 1., 0.], dtype=torch.float64))
    pairs = got[:, :6].reshape(9, 3, 2)
    close(pairs.square().sum(-1), torch.ones(9, 3, dtype=torch.float64))


@pytest.mark.t2
def test_offset_matches_sliced_full_sequence(impl):
    """Chunked generation with an offset agrees with the corresponding full-sequence slice."""
    full = impl.sinusoidal_encoding(23, 9, dtype=torch.float64)
    chunk = impl.sinusoidal_encoding(7, 9, start=11, dtype=torch.float64)
    close(chunk, full[11:18], atol=0, rtol=0)


@pytest.mark.t2
def test_position_embeddings_preserve_input_gradient(impl):
    """Adding fixed features broadcasts across batches and preserves gradients to embeddings."""
    encoding = impl.sinusoidal_encoding(5, 6, dtype=torch.float64)
    assert not encoding.requires_grad
    embeddings = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(embeddings)
    combined = embeddings + encoding
    close(combined[0] - embeddings[0], combined[1] - embeddings[1])
    close(torch.autograd.grad((combined * weights).sum(), embeddings)[0], weights)


@pytest.mark.t2
def test_invalid_dimensions_and_nonfloating_dtype(impl):
    """Negative lengths/offsets, zero width and unsupported integer output raise ValueError."""
    for kwargs in ({'length': -1, 'd_model': 4}, {'length': 3, 'd_model': 0},
                   {'length': 3, 'd_model': 4, 'start': -2},
                   {'length': 3, 'd_model': 4, 'dtype': torch.int64}):
        with pytest.raises(ValueError):
            impl.sinusoidal_encoding(**kwargs)


@pytest.mark.t3
def test_relative_offset_rotation_experiment(impl, artifact):
    """A fixed offset rotates every frequency pair; all 32 toy positions remain distinct."""
    result = run(impl.sinusoidal_encoding)
    artifact('relative_offset_rotation', result)
    assert result['metrics']['maximum_rotation_error'] < 1e-12
    assert result['metrics']['minimum_position_distance'] > .5
