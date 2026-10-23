`substr(DS_1, 2, _)`: the trailing `_` stands for the omitted `length`, so this is
`substr(DS_1, 2)` with the underscore written out.
The manual's syntax conventions say that "a non-specified optional operand is substituted by
an underscore or, if it is in the end of the invocation, can be omitted" (Overview and
conventions, syntax conventions).
With `length` omitted "the substring ends at the end of the input string" (Sub-string
extraction), so each value loses its first character.

- `"hello"` gives `"ello"` and `"ab"` gives `"b"`.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
