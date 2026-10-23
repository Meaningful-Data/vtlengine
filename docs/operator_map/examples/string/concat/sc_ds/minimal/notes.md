Checks `sc || DS`: the scalar is the left operand, so it comes first in each result value.

- `"Dear " || "Ann"` is `"Dear Ann"` and `"Dear " || "Bob"` is `"Dear Bob"`.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
