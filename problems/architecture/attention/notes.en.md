# Reference notes

The scale uses key width `D`, not sequence length or value width. Softmax is invariant to adding a common scalar to one row; shifting every key by the same vector therefore preserves outputs. Jointly permuting keys and values also preserves outputs.

An empty mask row requires care before softmax. Replacing output NaNs afterward can leave NaNs in backward. The reference first replaces the invalid row's scores, then zeros its weights.

The independent gradient test covers q, k and v. The six-value toy demonstrates causal information flow without introducing a training objective unrelated to the paper. Its experimental provenance is teaching adaptation.
