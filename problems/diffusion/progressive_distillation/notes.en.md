# Review: Progressive Distillation: Two Teacher Steps to One

Inputs alpha and sigma are amplitudes, with alpha²+sigma²=1; they are not cumulative variance alpha_bar. Require sigma_t>0 and a strictly lower end time so alpha_s-(sigma_s/sigma_t)alpha_t is nonzero. The teacher performs two denoising evaluations, including a new prediction at the intermediate state. Freeze its solved target. Weight is supplied per sample (the toy uses 1); parameterization rotations remain in module #7.

A fixed affine denoiser teacher takes two cosine-schedule DDIM steps; a tiny linear student learns the solved target and matches the endpoint in one evaluation. Demonstrates the exact distillation algebra, not image quality or repeated multi-stage halving.

The deliberately incorrect implementations isolate omits ratio term, wrong ddim sign, target has gradient, drops loss weight. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.
