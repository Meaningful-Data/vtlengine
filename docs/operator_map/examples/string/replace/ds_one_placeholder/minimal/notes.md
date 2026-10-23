`replace(DS_1, "a", _)`: the trailing `_` stands for the omitted `pattern2`, so this is
`replace(DS_1, "a")` with the underscore written out.
The manual's syntax conventions say that "a non-specified optional operand is substituted by
an underscore or, if it is in the end of the invocation, can be omitted" (Overview and
conventions, syntax conventions).
With `pattern2` omitted "all occurrences of `pattern1` are removed" (String pattern replacement).

- `"banana"` has three `a`; without them it is `"bnn"`.
- `"xyz"` has none, so it is kept.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
