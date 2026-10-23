Checks `DS || sc`: the scalar is composed with every measure of the dataset ("An operand Scalar
value is composed with all the Measures of the other operand", Typical behaviours).

- `"hello" || "!"` is `"hello!"` and `"world" || "!"` is `"world!"`.
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
