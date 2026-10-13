# Parameterization conversion: Review after passing

- Reversing v's sign preserves every shape but reverses motion. `test_velocity_tangent` determines the correct sign with finite differences.
- Alpha is the square root of cumulative signal. Squaring it again amplifies errors at high noise levels. `test_six_roundtrips` and `test_reference` catch this mistake.
- At alpha=0, noise cannot uniquely determine x0. Clamping produces finite values but cannot recover missing information. The inverse rotation from v remains exact; see `test_endpoint_stability`.
- Losses for different parameterizations imply different time weights. `test_error_weight_identity` converts the same erroneous prediction into all three representations, avoiding comparisons between unrelated prediction errors.

The original exercise checked [Salimans and Ho, Appendix D, Equations 26 and 30](https://arxiv.org/html/2202.00512v2#A4) on 2026-10-13.
