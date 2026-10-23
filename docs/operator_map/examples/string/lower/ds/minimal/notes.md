`lower` converts the characters of each value to lower case.

- `"HELLO"` becomes `"hello"` and `"Hi There"` becomes `"hi there"`.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
