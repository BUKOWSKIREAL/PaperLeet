# Reference notes

Table 2 of [SDPO v2](https://arxiv.org/html/2601.20802v2) uses a successful original attempt as its own correct solution. Otherwise a same-question successful solution takes precedence over environment output. Environment feedback is used only when no successful solution is available. Build a same-question success matrix, choose the first candidate, then override successful rows with their own index. Picking the smallest index among several peers makes the exercise deterministic; that tie-break is not prescribed by the paper.

The returned solution index can therefore point to the row itself. Index -1 means no successful solution, and does not by itself mean unavailable: rich feedback can still supervise that row. Tests include both sources being available so that priority is observable.

Detach teacher logits and apply availability to both numerator and denominator. This module uses full-vocabulary per-token reverse KL at supplied student prefixes, averaged over available tokens; the next module implements Appendix A.3 coarsening. The toy replaces language-model reprompting with a categorical lookup teacher. It retains the feedback routing, but does not reproduce the paper's training or benchmark results.
