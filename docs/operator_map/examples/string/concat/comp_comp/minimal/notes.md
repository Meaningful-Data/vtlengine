Checks `comp || comp` inside `calc`. `Me_3` is a new component, so it is added to the dataset as a
measure (Typical behaviours, operations on Data Set Components), and `Me_1` and `Me_2` are kept.

- `"foo" || "bar"` is `"foobar"` and `"ab" || "cd"` is `"abcd"`.
- Row 3 has a null `Me_2`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
