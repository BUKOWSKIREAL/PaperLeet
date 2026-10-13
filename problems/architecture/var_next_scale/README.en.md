# VAR Next-Scale Prediction

[VAR](https://arxiv.org/html/2404.02905v2) predicts a whole resolution level at a time. Implement two functions that prepare frozen tokenizer features and the training attention mask. The [official implementation](https://github.com/FoundationVision/VAR/blob/main/models/quant.py) constructs each input from a cumulative reconstruction of **earlier** token maps.

## Implement

`next_scale_inputs(token_maps, codebook, projections) -> Tensor`

The list contains `K` integer target maps `[B,n_k,n_k]`. Let `lift_codes` be the supplied lookup → bicubic upsample → channel projection helper, and let `H=n_K`. For the input predicting scale `k>0`, compute

$$F_{<k}=\sum_{j<k}\operatorname{lift\_codes}(r_j,Z,P_j,H),\qquad x_k=\operatorname{area}_{n_k}(F_{<k}).$$

Flatten each `x_k` in row-major spatial order into `[B,n_k²,C]`, then concatenate for `k=1,…,K−1`. The result is `[B,Σ_{k>0}n_k²,C]`. The first scale uses a separate class/start embedding and contributes no row to this function. For `K=1`, return `[B,0,C]`. The last target map is never used as an input.

`scale_causal_mask(scales, device=None) -> Tensor`

Return boolean `[L,L]`, with `L=Σn_k²` and `True` meaning **visible**. Query at scale `i` may read a key at scale `j` exactly when `j≤i`. Thus every same-scale pair is visible, even in reverse raster order. [The official mask](https://github.com/FoundationVision/VAR/blob/main/models/var.py) follows this rule. Same-scale attention is safe because those positions contain lower-scale features, not their target tokens.

## Contract

Scales are positive, strictly increasing integers beginning at `1`; all maps are square and share `B>0`. Token maps are int64 with indices in `[0,V)`. The shared codebook is `[V,C]` and the supplied projections are `[K,C,C]`. Inputs are finite CPU float32/float64 tensors with matching floating dtype/device; strided tensors are valid. Preserve values, dtype and device. The tokenizer is frozen: the supplied `no_grad` wrapper is intentional.

`lift_codes` is already implemented. It uses `align_corners=False` bicubic interpolation to the largest resolution, followed by `P_k @ feature` at each pixel. In the original system `Phi` contains learned spatial convolutions; fixed channel matrices keep this exercise focused. Position embeddings, class conditioning, codebook training and the Transformer are outside this contract.

## Toy

A teaching adaptation uses 1×1, 2×2 and 4×4 target maps and a fixed attention layer. Your two functions control the entire information path. Changing a target map must not change predictions at its own or any coarser scale; earlier maps must still influence finer predictions. The chart also compares visible-key counts with raster causal attention. This is an information-flow experiment, not an image-generation benchmark.
