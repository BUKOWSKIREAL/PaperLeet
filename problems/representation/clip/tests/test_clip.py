import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.clip.experiment import run


@pytest.mark.t0
def test_shapes_single_pair_and_zero_features(impl):
    """Single-pair loss is zero; zero feature rows remain finite under epsilon normalization."""
    images, texts = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    scale = torch.tensor(1., dtype=images.dtype)
    logits = impl.contrastive_logits(images, texts, scale)
    loss = impl.clip_loss(images, texts, scale)
    assert logits.shape == (1, 1) and loss.shape == () and logits.dtype == images.dtype
    close(logits, torch.zeros_like(logits))
    close(loss, torch.zeros_like(loss))


@pytest.mark.t1
def test_orthogonal_analytic_loss(impl):
    """Two orthogonal matched pairs have logits 2I and loss log(1+exp(-2))."""
    features = torch.eye(2, dtype=torch.float64)
    scale = torch.tensor(math.log(2), dtype=features.dtype)
    close(impl.contrastive_logits(features, features, scale), 2 * features)
    close(impl.clip_loss(features, features, scale), torch.tensor(math.log1p(math.exp(-2)), dtype=features.dtype))


@pytest.mark.t2
def test_bidirectional_loss_independent_logsumexp(impl):
    """Non-symmetric batches require both row and column log-partition functions."""
    images, texts = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    scale = torch.tensor(math.log(3), dtype=images.dtype)
    unit_i = images / images.norm(dim=-1, keepdim=True)
    unit_t = texts / texts.norm(dim=-1, keepdim=True)
    logits = 3 * unit_i @ unit_t.T
    expected = (logits.logsumexp(1).mean() + logits.logsumexp(0).mean()) / 2 - logits.diag().mean()
    close(impl.clip_loss(images, texts, scale), expected)
    close(impl.clip_loss(images, texts, scale), impl.clip_loss(texts, images, scale))


@pytest.mark.t2
def test_feature_scale_invariance_and_pair_permutation(impl):
    """Positive feature norms carry no similarity information; jointly permuting pairs preserves loss."""
    images, texts = torch.randn(4, 3), torch.randn(4, 3)
    scale = torch.tensor(.7)
    expected = impl.clip_loss(images, texts, scale)
    norms = torch.tensor([[.1], [2.], [10.], [50.]])
    close(impl.clip_loss(images * norms, texts / norms, scale), expected)
    order = torch.tensor([3, 1, 0, 2])
    close(impl.clip_loss(images[order], texts[order], scale), expected)


@pytest.mark.t2
def test_temperature_cap_and_gradients(impl):
    """The inverse temperature is learned below the cap and cannot overflow or grow beyond 100."""
    features = torch.eye(3, dtype=torch.float64)
    scale = torch.tensor(math.log(2), dtype=features.dtype, requires_grad=True)
    loss = impl.clip_loss(features, features, scale)
    gradient, = torch.autograd.grad(loss, scale)
    close(gradient, torch.tensor(-4 / (math.exp(2) + 2), dtype=features.dtype))
    large_scale = torch.tensor(1000., dtype=features.dtype, requires_grad=True)
    logits = impl.contrastive_logits(features, features, large_scale)
    close(logits, 100 * features)
    close(torch.autograd.grad(logits.sum(), large_scale)[0], torch.zeros_like(large_scale))


@pytest.mark.t2
def test_encoder_gradients_against_independent_formula(impl):
    """Both modality encoders and temperature receive gradients of the symmetric objective."""
    images, texts = [torch.randn(3, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    scale = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    args = (images, texts, scale)
    i, t, s = [x.detach().clone().requires_grad_() for x in args]
    logits = s.exp() * ((i / i.norm(dim=-1, keepdim=True)) @ (t / t.norm(dim=-1, keepdim=True)).T)
    expected = (logits.logsumexp(1).mean() + logits.logsumexp(0).mean()) / 2 - logits.diag().mean()
    for got, want in zip(torch.autograd.grad(impl.clip_loss(*args), args),
                          torch.autograd.grad(expected, (i, t, s))):
        close(got, want, atol=1e-10, rtol=1e-10)


@pytest.mark.t3
def test_learn_pair_retrieval(impl, artifact):
    """Eight random paired vectors learn bidirectional retrieval using the learner's contrastive loss."""
    result = run(impl.clip_loss)
    artifact("paired_retrieval", result)
    assert result["image_to_text_accuracy"] == 1
    assert result["text_to_image_accuracy"] == 1
    assert result["final_loss"] < .05
    assert result["final_loss"] < .1 * result["initial_loss"]
