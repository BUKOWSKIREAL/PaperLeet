# Implementation notes — On-policy Distillation at Student Prefixes

Implement `sample_student` and `distillation_loss`. The sampling table has shape `[V+1,V]`; row `V` is BOS and row `k` is the distribution after token `k`. Return `(tokens, prefixes)`, both `[B,L]` integer tensors. The first prefix is BOS; every subsequent prefix is the previous sampled token. Use the supplied generator and categorical sampling at temperature 1, without a gradient through discrete sampling. Sequences have fixed length and no EOS in this toy.

The loss inputs are `[B,L,V]` logits evaluated for **both models at those same student-generated prefixes**, plus `[B,L]` binary mask. `reverse` means KL(student || teacher); `forward` means KL(teacher || student). Sum over the entire vocabulary, average only valid tokens, freeze the teacher, and use log-softmax. All-padding gives differentiable zero. Padded logits may be nonfinite and must be neutralized before log-softmax. Valid logits are finite.

This is the lambda=1 GKD variant. It is not sampled-token OPD, a policy-gradient estimator, or teacher-generated training data. The supplied training loop learns four conditional distributions with a three-token model. Its evaluation metric is mean forward KL over all four states, while the training objective is reverse KL on fresh student prefixes.

## Deliberately wrong variants

- Reverse KL uses the teacher distribution as weights.
- Allow gradients into the teacher.
- Sample every token from BOS instead of the student prefix.
- Normalize over padding too.

Each mutant differs from the reference in a small, intentional change. T2 includes independent formula or gradient expectations, so equality with the reference is not the only acceptance evidence. T3 produces artifacts from the supplied implementation.

Source: [2306.13649v3](https://arxiv.org/html/2306.13649v3#S3); [TRL implementation (independent of the paper authors; not the reference for this exercise)](https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py). Toy provenance: PaperLeet fixed-length categorical Markov model, not the original T5 or Gaussian-mixture experiment.
