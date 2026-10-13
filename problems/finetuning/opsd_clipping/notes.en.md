# Reference notes

Use log_softmax for numerical stability. Finite-difference checks avoid clipping boundaries. The clipped objective is not guaranteed nonnegative; a concrete probability example tests this distinction. Per-response normalization gives long and short responses equal batch weight.
