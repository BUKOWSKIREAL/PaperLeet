# Reference notes

Compose softcap on the gate's linear factor, sigmoid on the raw gate, and a separate softcap on the up branch. The two cap values are distinct.

Tests compare gradients with scalar analytic derivatives. The softcap derivative is `1 - tanh(x / beta)^2`; unlike a clamp, it does not suddenly become zero at `|x| = beta`. Extreme floating-point saturation is allowed.

Figure 4 evaluates equal branch inputs. Independent tests also use unequal magnitudes and opposite signs so an implementation cannot pass merely by matching that diagonal slice. The activation bound applies before any FFN output projection. This exercise does not validate full-model training stability.
