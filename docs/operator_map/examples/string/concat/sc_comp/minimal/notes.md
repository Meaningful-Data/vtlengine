Checks `sc || comp` inside `calc`: `Me_2` is a new measure and `Me_1` is kept.

- `"Hello " || "World"` is `"Hello World"` and `"Hello " || "VTL"` is `"Hello VTL"`.
- Row 3 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
