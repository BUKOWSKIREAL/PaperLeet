# Reference notes

Bias correction uses the current one-based step after updating both moments. Epsilon belongs after the square root of the corrected second moment. No in-place operations are needed.

The decay is computed from the old parameter. Its contribution remains proportional to the parameter even if the adaptive denominator differs across coordinates. This is the paper's distinction from Adam with L2 loss regularization. Tests vary decay while holding the gradient and state fixed to isolate this mechanism.

The exercise uses `lr * weight_decay` as PyTorch does. Comparing raw coefficients against the paper's separate schedule notation requires matching parameterization. The toy illustrates optimization only; it is not evidence about downstream generalization.
