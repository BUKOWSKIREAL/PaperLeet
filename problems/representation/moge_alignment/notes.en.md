# Reference notes

For a single residual, its center breakpoint changes the slope upward, while its two truncation boundaries change the slope downward. A minimum can therefore be represented by a zero-residual breakpoint. Constant terms from zero coefficients still contribute to the objective. Zero weights cannot supply useful anchors.

The point-map objective has only two unknowns: a shared xyz scale and a z shift. Anchoring one depth removes the shift; its inverse-depth weight is still computed from the original target depth. The implementation evaluates every active ratio per anchor, which has cubic worst-case cost in the tiny point count. MoGe's production solver uses sorted slope changes and batching. No L2 fit, SVD, rotation, or free xyz translation appears here.

Tests compare the achieved objective with a separate arrangement solver that includes center and saturation lines. This avoids requiring a particular anchor when several optima exist. The scalar contract deliberately specifies tie behavior; the full alignment contract accepts any minimizer.

The optional coordinate truncation follows the author code's `min(weight*abs(residual), trunc)` convention. Using `weight*min(abs(residual), trunc)` is a different estimator. This distinction is covered with unequal weights. Masking precedes finite/depth checks, so invalid camera pixels do not contaminate valid fitting.

The toy reports inlier recovery and objective curves, not monocular reconstruction quality. This package is based on CVPR 2025 MoGe's alignment core, even though the official repository now also contains later MoGe variants.
