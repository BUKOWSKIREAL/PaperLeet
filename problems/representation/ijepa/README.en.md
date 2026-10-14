# I-JEPA Masked Latent Targets and EMA

[I-JEPA](https://arxiv.org/abs/2301.08243) predicts masked target representations from a context representation. Implement the target preparation, prediction loss and momentum update around that prediction. Target features and masks are supplied; the encoder and predictor are outside this exercise.

## Implement

`masked_targets(features, masks, num_context=1, eps=1e-5)` receives teacher features `[B,N,D]` and a list of `K` target masks, each integer `[B,M]`. It must:

1. Detach the teacher features and apply layer normalization independently to each token's **D features**, without affine parameters. Use population variance and `sqrt(variance + eps)`.
2. Gather each batch item's target indices in the given order.
3. Repeat each target block's whole B-item batch `C=num_context` times, then concatenate target blocks. Return `[K*C*B,M,D]` in **target-block, context, batch** order.

For two target blocks, two contexts and two images, the leading-axis order is `(block0,context0,image0)`, `(block0,context0,image1)`, `(block0,context1,image0)`, `(block0,context1,image1)`, then block1 in the same order. Repeating each image consecutively would misalign the predictor's targets.

`masked_prediction_loss(prediction, target)` returns the mean smooth-L1 objective with `beta=1`:

$$\ell(r)=\begin{cases}\tfrac12r^2,&|r|<1\\ |r|-\tfrac12,&|r|\geq1,\end{cases}\qquad
L=\operatorname{mean}\ell(prediction-\operatorname{sg}(target)).$$

This follows the **released author's code** in `src/train.py`; the paper describes average L2 prediction distance. Both are identified explicitly here, and the judge requires smooth L1. The target is stopped even if the caller passes a tensor requiring gradients. Predictions remain differentiable. Residuals `[-2,0.5]` give loss `(1.5+0.125)/2 = 0.8125`.

`ema_update(target, online, momentum)` returns

$$\theta_{target}^{new}=m\operatorname{sg}(\theta_{target})+(1-m)\operatorname{sg}(\theta_{online}).$$

Return a new detached tensor and leave both inputs unchanged. This is a functional equivalent of the author's in-place parameter update. In training, the caller invokes it after the optimizer step, for each encoder parameter; the predictor is not included in the target-encoder EMA.

## Contract

Finite CPU float32/float64 tensors, equal dtype/device for paired tensors; preserve input values. Features are nonempty `[B,N,D]`; masks are int64 on the same device, have equal nonzero M, and contain valid token indices. Context count is a positive integer and epsilon positive and finite. Loss and EMA tensor pairs have identical nonempty shapes. Momentum is finite in `[0,1]`. Noncontiguous inputs are supported. Validators are supplied.

## Toy experiment

**Teaching adaptation.** Two synthetic six-token images supply latent features. Two target masks and two contexts exercise ordering. A small prediction table learns the selected targets through your loss. A separate fixed-online EMA example displays how target parameters approach online parameters. Scalar normalization/gather and the EMA closed form independently check the outputs. This does not train an image encoder or reproduce ImageNet pretraining. The preview uses the reference; T3 invokes all three submitted functions.

[Author training code](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py).
