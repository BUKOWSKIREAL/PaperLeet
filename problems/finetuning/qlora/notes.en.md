# Implementation notes — QLoRA NF4 Blocks and Trainable Adapters

Implement block quantization, dequantization and a frozen-base adapter forward. The **provided `NF4` tuple** is the pinned asymmetric 16-value codebook: exact zero is index 7, positive and negative nonzero levels are not mirror images. Flatten the weight in row-major order, split into blocks, divide by each block's absmax, and choose the nearest level. Ties choose the lower code index. A zero block has scale zero and all code indices 7. A final short block has its own scale.

Return uint8 **unpacked indices** with the original weight shape and a flat floating-point scale vector. Decoding repeats each scale for its block. Weight quantization is frozen; dequantization freezes scales. The forward accepts A[r,in], B[out,r] and alpha/r scaling, preserves input/adapter gradients and freezes optional bias. `block_size` defaults to 64; tests use smaller blocks for inspection. NF4 constants are supplied in the starter; they are not something to memorize.

The toy trains adapters on a frozen quantized map. Storage here consumes one byte per index, not half a byte. It intentionally omits nibble packing, double quantization, paged optimizers and CUDA kernels, so its metrics cannot establish the paper's full memory claims. See the [official QLoRA training repository](https://github.com/artidoro/qlora) for those larger-system components.

## Deliberately wrong variants

- Replace NF4 with uniform levels.
- Use global absmax for each block.
- Allow gradients into frozen scales.
- Forget adapter rank normalization.

Each mutant differs from the reference in a small, intentional change. T2 includes independent formula or gradient expectations, so equality with the reference is not the only acceptance evidence. T3 produces artifacts from the supplied implementation.

Source: [2305.14314v1](https://arxiv.org/html/2305.14314v1#S3); [Official quantization library, pinned 0.43.3 NF4 codebook](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/functional.py). Toy provenance: PaperLeet synthetic regression with unpacked NF4 indices; does not measure full QLoRA memory savings.
