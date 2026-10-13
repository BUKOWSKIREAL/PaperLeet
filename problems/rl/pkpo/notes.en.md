# Solution notes

An item contributes to Eq. (33) only when it is the unique largest reward in its subset. If its sorted index is i and the second-largest index is j<i, the remaining k-2 elements can be chosen in `C(j,k-2)` ways. Summing these weighted gaps gives the prefix formula. Two cumulative sums, one of coefficients and one of reward-weighted coefficients, give all weights after sorting. Equal values produce zero gaps and permutation-equivariant results.

For k=1 we choose zero baseline, recovering REINFORCE with reward/N. For k=N only the gap between the largest and second-largest reward survives; a tied maximum contributes zero. For k>=2 adding the same constant to every reward leaves these weights unchanged. An average of transformed rewards is not the leave-one-out baseline, since each coefficient already depends on the group.

The weights contain `1/C(N,k)` and must be summed against log probabilities. Only independent groups are averaged. Detach the sampled rewards and actions for a score-function estimator; otherwise a pathwise gradient changes the estimator.

## Source version and toy

The problem is associated with the NeurIPS 2025 paper and pins the [corrected v5 Eq. (33)](https://arxiv.org/html/2505.15201v5); it uses the subset definition rather than earlier recurrences with version-dependent corrections. The prefix implementation is independently derived, not copied from the paper's appendix code. The k=1 zero-baseline convention is stated explicitly because Eq. (33)'s max over an empty set needs a convention. We do not implement PPO clipping or the complete LLM optimizer.

The Gaussian toy retains the paper's standard deviation and reward support. It measures the objective and gradients on a bounded grid, not convergence or a quality benchmark. Independent subset enumeration checks finite groups; exhaustive Bernoulli sample enumeration verifies the unbiased pass@k gradient against `k*(1-p)^(k-1)`. Mutants catch mean reward, untransformed rewards, sorted output order and a second normalization.
