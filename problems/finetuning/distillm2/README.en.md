# DistiLLM-2 Two-source Asymmetric Distillation

Implement the asymmetric objective from [DistiLLM-2 v2, §2.2 and Eq. (2)](https://arxiv.org/html/2503.07067v2). The teacher and student are evaluated on **two different response sources**: teacher responses (`_t`) and student responses (`_s`).

## Functions

For teacher probabilities `p = softmax(teacher_logits)` and student probabilities `q = softmax(student_logits)`, implement per-position vocabulary divergences:

$$\mathrm{SKL}_\alpha(p,q)=\sum_v p_v\log\frac{p_v}{\alpha p_v+(1-\alpha)q_v},$$
$$\mathrm{SRKL}_\alpha(p,q)=\sum_v q_v\log\frac{q_v}{(1-\alpha)p_v+\alpha q_v}.$$

1. `skew_kl(teacher_logits, student_logits, alpha)` returns `[...]` from equal-shaped `[..., V]` logits.
2. `skew_reverse_kl(...)` has the same interface, but the reversed source and mixture above.
3. `distillm2_loss(teacher_t, student_t, mask_t, teacher_s, student_s, mask_s, alpha_t=.1, alpha_s=.1, beta=.5)` returns a scalar:

$$L=\frac{(1-\beta)\sum_{b,t}m^t_{bt}\,\mathrm{SKL}_{\alpha_t}(p^t_{bt},q^t_{bt})+\beta\sum_{b,t}m^s_{bt}\,\mathrm{SRKL}_{\alpha_s}(p^s_{bt},q^s_{bt})}{2B}.$$

Each pair has shape `[B,T,V]`; the two sources may have different `T` and `V` but share nonzero `B`. Masks are boolean `[B,T]`, with `True` for response tokens. Sum tokens and vocabulary, then divide by `2B`, **not by valid-token count**. An empty/all-masked response contributes zero. Coefficients are finite and in `[0,1]`. Vocabulary is nonempty; logits are finite, same dtype/device in each pair. Preserve float32/float64, strided inputs and input values. Both student branches retain autograd; teacher logits are frozen. Stable log-space helpers and validation are supplied. No model loading or response generation is required.

## Toy experiment

Optimize two separate sets of synthetic three-token-vocabulary distributions with your loss. T3 plots the actual objective and both divergences over 60 updates. This is a **teaching adaptation** of the two-source objective; it does not reproduce the paper's long-tail Figure 1 data or a language-model benchmark. Coefficients stay fixed. This exercise uses full conditional-vocabulary KL at each supplied context and pins the `1/(2B)` factor printed in Eq. (2); see the notes for the scope boundary.
