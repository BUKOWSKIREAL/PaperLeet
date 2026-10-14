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
    raise NotImplementedError("Implement object_mask_inputs")


def masked_prediction_loss(predictions, targets, history_length, masked_slots):
    """Separate mean MSE for masked non-anchor history and all future tokens; freeze targets."""
    _float_tensor(predictions, 4, 'predictions'); _float_tensor(targets, 4, 'targets')
    if predictions.shape != targets.shape or predictions.dtype != targets.dtype or predictions.device != targets.device:
        raise ValueError('predictions and targets must match')
    b, t, s, d = predictions.shape
    _mask(masked_slots, b, s, predictions.device)
    if isinstance(history_length, bool) or not isinstance(history_length, int) or not 1 <= history_length < t:
        raise ValueError('history_length must be in [1,T-1]')
    raise NotImplementedError("Implement masked_prediction_loss")
