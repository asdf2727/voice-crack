"""
Everything here follows the tensor shape contract `(B?, T, C)` where:

- B: optional batch size
- T: time axis
- C: channels

Note: This is the transpose of the usual CNN shape convention `(B, C, T)`.
"""