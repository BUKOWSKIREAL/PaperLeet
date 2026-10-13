# Reference notes

Compute one frequency per sine/cosine pair using `exp(-log(10000) * 2i / d_model)`. The sine slice receives all frequencies. For an odd width, truncate only the cosine slice; changing the frequency denominator changes every channel.

Creating positions and frequencies directly in the requested dtype avoids precision loss from casting float32 results to float64 afterward. The offset belongs inside the trigonometric argument.

The toy uses the angle-addition identity to validate each frequency pair independently. Distinctness alone would accept many incorrect encodings, so T1 also checks the exact scalar formula. Odd dimensions, zero length and explicit offsets are documented interface extensions, not claims about the original model settings.
