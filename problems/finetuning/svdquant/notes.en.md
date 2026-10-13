# Solution notes

With the stated matrix layout, scale input channels by `1/lambda` and weight **rows** by `lambda`. Truncated SVD may use `left=U[:,:r]*singular[:r]`, `right=Vh[:r]`; another equivalent factorization is fine. Preserve the residual before fake quantization. In Eq. (5) the high-precision path sees xs, while the low-bit path sees Q(xs). A quantized input in both paths would lose the benefit even at full rank.

## Source and numerical scope

[The v1 text in §4.2](https://arxiv.org/html/2411.05007v1) writes the transformed weight as `W diag(lambda)` despite defining W[M,N]. For that layout the dimensionally consistent identity is `diag(lambda) W`, used here. The test's nonsquare matrix prevents this transpose ambiguity from passing unnoticed. The output follows Eq. (5) and tests compare products rather than singular-vector signs.

The [author implementation](https://github.com/mit-han-lab/deepcompressor/) is a calibration and inference system. This exercise independently implements one-shot SVD with supplied smoothing, and chooses an explicit per-tensor symmetric integer fake quantizer from Eq. (1). It does not model the full production calibration configuration, iterative low-rank refinement, grouped/FP4 variants, storage packing or Nunchaku fused kernels. CPU float32/float64 stand in for the high-precision branch; no hardware speed or compression claim follows.

The controlled outlier toy measures actual output error and residual spectrum. It demonstrates one deliberately chosen matrix; increasing rank or applying SVDQuant is not guaranteed to improve every arbitrary quantized matrix monotonically. Independent tests use known rotated singular vectors and Python scalar rounding, including rank-zero/full-rank and zero tensors. Mutants target reversed smoothing, quantized high-precision activations, quantizing the full weight twice, and omitted singular values.
