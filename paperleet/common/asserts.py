"""Tensor comparisons with actionable location information."""
import torch


def assert_close_located(got, exp, rtol=1e-5, atol=1e-8):
    got, exp = torch.as_tensor(got), torch.as_tensor(exp)
    assert got.shape == exp.shape, f"at shape: expected {tuple(exp.shape)}, got {tuple(got.shape)}"
    expected = exp.to(dtype=got.dtype, device=got.device)
    close = torch.isclose(got, expected, rtol=rtol, atol=atol)
    if bool(close.all()):
        return
    error = (got - expected).abs().to(torch.float64)
    error = torch.where(close, torch.full_like(error, -1), error)
    error = torch.nan_to_num(error, nan=float("inf"), posinf=float("inf"))
    flat = int(error.reshape(-1).argmax())
    index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), got.shape)) if got.ndim else ()
    raise AssertionError(f"max abs diff {error[index].item():.7g} at {index}: expected {expected[index].item()}, got {got[index].item()} (rtol={rtol}, atol={atol})")
