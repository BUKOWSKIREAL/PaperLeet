# Reference notes

The key-by-value state is `[B,H,K,V]`. Decay multiplies its key rows, not value columns. Compute the prediction from the already decayed state. Then use the same token's updated state for the query read. This order matters even for a one-token sequence.

A normalized repeated key with beta=1 replaces its old predicted value instead of adding another copy. With beta=0, the state still decays. Different key channels can have different retention factors; reducing alpha to a head-wise scalar changes the operator.

In K3's gate, `exp(A)` multiplies the logit *inside* the sigmoid. The fixed negative lower bound multiplies that sigmoid output. Floating-point saturation can yield retention endpoints; use inclusive numerical bounds in tests. A log-decay bound alone does not claim arbitrary-length products are free of underflow.

The scan avoids in-place changes so gradients also reach an initial state passed from a previous chunk. Concatenating outputs from two scans must agree with a single longer scan. No q scaling is implicit in this task.

Figure 3's curve redraw is paper-derived; the four-token memory example is a teaching adaptation with independent hand-derived targets. Neither is a full K3 training or throughput experiment.
