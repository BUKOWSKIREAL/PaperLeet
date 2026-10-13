# Reference notes

A matrix multiply on the final axis automatically shares the same network across arbitrary leading dimensions. Both biases broadcast over these dimensions. No sequence reduction belongs in the FFN.

The gradient test uses `torch.nn.functional.linear` as an independent expression and checks the input, both matrices and both biases. Explicitly negative preactivations verify the zero derivative of the closed ReLU gate away from its nondifferentiable origin.

The toy labels are independent analytic absolute-value/hinge functions. The held-out set evaluates the learned nonlinear mapping beyond the training tokens. It is a small deterministic teaching experiment; the paper's machine-translation benchmarks are outside this module.
