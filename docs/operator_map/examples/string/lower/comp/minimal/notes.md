`lower` inside `calc`: `Me_2` is a new measure holding the lower-case `Me_1`, which is kept.

- `"HELLO"` becomes `"hello"` and `"Hi There"` becomes `"hi there"`.
- Row 3 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
