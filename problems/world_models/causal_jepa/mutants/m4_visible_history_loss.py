# BUG: 历史损失错误地包含可见对象
"""Object trajectory masking with identity anchors and frozen latent targets."""
import torch


def _float_tensor(x, ndim, name):
    if not isinstance(x, torch.Tensor) or x.ndim != ndim or not x.numel() or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be finite, nonempty float32/float64 rank {ndim}')


def _mask(masked_slots, batch, slots, device):
    if not isinstance(masked_slots, torch.Tensor) or masked_slots.shape != (batch, slots) or masked_slots.dtype != torch.bool or masked_slots.device != device:
        raise ValueError('masked_slots must be bool (B,S) on the input device')


def object_mask_inputs(history, masked_slots, anchor_projection, time_embeddings, mask_token):
    """Return query/context tensor (B,T,S,D) and bool target mask (B,T,S)."""
    _float_tensor(history, 4, 'history'); _float_tensor(anchor_projection, 2, 'anchor_projection')
    _float_tensor(time_embeddings, 2, 'time_embeddings'); _float_tensor(mask_token, 1, 'mask_token')
    b, h, s, d = history.shape
    _mask(masked_slots, b, s, history.device)
    if anchor_projection.shape != (d, d) or time_embeddings.shape[1] != d or time_embeddings.shape[0] <= h or mask_token.shape != (d,):
        raise ValueError('projection must be (D,D), time (T,D) with T>H, token (D,)')
    if any(x.dtype != history.dtype or x.device != history.device for x in (anchor_projection, time_embeddings, mask_token)):
        raise ValueError('floating tensors must share dtype and device')
    t = time_embeddings.shape[0]
    target_mask = torch.ones((b, t, s), dtype=torch.bool, device=history.device)
    target_mask[:, 0] = False
    target_mask[:, 1:h] = masked_slots[:, None, :]
    anchors = history[:, 0] @ anchor_projection.T
    queries = anchors[:, None] + time_embeddings[None, :, None] + mask_token
    visible = torch.cat((history, history.new_zeros((b, t - h, s, d))), dim=1)
    visible = visible + time_embeddings[None, :, None]
    return torch.where(target_mask[..., None], queries, visible), target_mask


def masked_prediction_loss(predictions, targets, history_length, masked_slots):
    """Separate mean MSE for masked non-anchor history and all future tokens; freeze targets."""
    _float_tensor(predictions, 4, 'predictions'); _float_tensor(targets, 4, 'targets')
    if predictions.shape != targets.shape or predictions.dtype != targets.dtype or predictions.device != targets.device:
        raise ValueError('predictions and targets must match')
    b, t, s, d = predictions.shape
    _mask(masked_slots, b, s, predictions.device)
    if isinstance(history_length, bool) or not isinstance(history_length, int) or not 1 <= history_length < t:
        raise ValueError('history_length must be in [1,T-1]')
    errors = (predictions - targets.detach()).square()
    selected = masked_slots[:, None, :, None].expand(b, history_length - 1, s, d)
    history_error = errors[:, 1:history_length].reshape(-1)
    history_loss = history_error.mean() if history_error.numel() else errors[:, :0].sum()
    future_loss = errors[:, history_length:].mean()
    return history_loss + future_loss, history_loss, future_loss
