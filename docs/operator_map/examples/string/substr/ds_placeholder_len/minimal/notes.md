The `_` placeholder leaves `start` out: "If `start` is omitted, the substring starts from the 1st
position" (Sub-string extraction). With `length` 3 each value keeps its first three characters.

- `"hello world"` gives `"hel"`.
- `"ab"` gives `"ab"`: "If `length` is omitted or overcomes the length of the input string,
  the substring ends at the end of the input string".
- Row 3 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
