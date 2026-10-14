# Reference notes

The target encoder sees the unmasked image, so target preparation starts from its full token feature map. The context encoder's masking and the predictor's positional queries are upstream of this exercise. Supplying features does not reproduce representation learning by itself.

The author's `forward_target` runs without gradients, normalizes over the last feature dimension, applies target masks and invokes `repeat_interleave_batch`. That helper's name can mislead: it repeats a **whole batch for each block**, not each individual image. The target mask list order and index order are both meaningful.

The official `loss_fn` uses `F.smooth_l1_loss` with its default beta 1 and mean reduction. The paper's L2 description is not the numeric contract of this released implementation. The derivative with respect to each prediction is `clamp(prediction-target,-1,1)/numel`. Both target preparation and the loss independently enforce the teacher boundary, making accidental training of externally supplied targets detectable.

The target encoder update uses the old target with weight `m`. For fixed online parameters, after T updates, `target_T = m**T * target_0 + (1-m**T) * online`. The toy checks this identity in a separate trace; it does not pretend that online parameters are fixed during real I-JEPA training.

Sources: [paper](https://arxiv.org/abs/2301.08243), [src/train.py](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py), [mask gathering](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/utils.py), [batch repetition](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/tensors.py). The functional EMA output and deterministic mask inputs are deliberate teaching adaptations.
