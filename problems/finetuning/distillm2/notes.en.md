# Solution notes

Use `log_softmax` and a log-space mixture. The teacher must be detached before constructing either mixture. In reverse KL, gradients flow through both the student probability multiplier and the student component of the mixture; detaching either changes the learning rule. At alpha zero the two losses become forward/reverse KL; at alpha one both vanish.

Apply SKL to the teacher-response pair and SRKL to the student-response pair. Mask token losses, sum each source, then apply the beta weights and `1/(2B)`. A token mean changes the relative weight of differently sized responses. The tests independently compute scalar probabilities and central finite differences; they do not call the reference implementation.

## Paper and implementation boundary

The source is [arXiv v2](https://arxiv.org/html/2503.07067v2), not a floating latest version. Eq. (2) includes a factor one half, retained here. The paper's Eq. (1) is written along observed response tokens; this exercise explicitly computes full-vocabulary conditional divergences at each supplied context. It is a local distribution-matching exercise and does not assert equality to a full autoregressive sequence KL under arbitrary context sampling. Fixed alpha/beta replace the paper's curriculum and response curation; the response contexts are inputs. No claim of bitwise equivalence to the [author training repository](https://github.com/jongwooko/distillm-2) is made.

T3's independent synthetic distributions exercise both sources, including the gradient update. It does not copy Figure 1 or establish the paper's model-quality gains. Four mutants cover identical losses on both sources, incorrect SRKL mixture orientation, missing Eq. (2) scaling, and detached students.
