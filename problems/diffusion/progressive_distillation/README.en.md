# 26. Progressive Distillation: Two Teacher Steps to One

Derive the exact clean prediction that makes one student DDIM step match two teacher steps, then learn it.

## Contract

- `ddim_step(noisy: Tensor, clean_prediction: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distilled_target(noisy: Tensor, teacher_endpoint: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor`
- `distillation_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor`
- `teacher_pair(teacher, noisy: Tensor, t: Tensor, midpoint: Tensor, end: Tensor) -> Tensor` (provided)

Inputs alpha and sigma are amplitudes, with alpha²+sigma²=1; they are not cumulative variance alpha_bar. Require sigma_t>0 and a strictly lower end time so alpha_s-(sigma_s/sigma_t)alpha_t is nonzero. The teacher performs two denoising evaluations, including a new prediction at the intermediate state. Freeze its solved target. Weight is supplied per sample (the toy uses 1); parameterization rotations remain in module #7.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** A fixed affine denoiser teacher takes two cosine-schedule DDIM steps; a tiny linear student learns the solved target and matches the endpoint in one evaluation. Demonstrates the exact distillation algebra, not image quality or repeated multi-stage halving.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2202.00512
- https://github.com/google-research/google-research/tree/0a1b0fe7298a58bca6dcb47112379d829bc0c0a5/diffusion_distillation

Anchors: arXiv v1 §3 Progressive distillation; Algorithm 2 and Appendix B distillation target.
