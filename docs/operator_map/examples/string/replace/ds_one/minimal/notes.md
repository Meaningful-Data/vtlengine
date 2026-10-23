Without `pattern2`, "all occurrences of `pattern1` are removed" (String pattern replacement).

- `"hello world"` has three `l`; without them it is `"heo word"`.
- `"say"` has none, so it is kept.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
