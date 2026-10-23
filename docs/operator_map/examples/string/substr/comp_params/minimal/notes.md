`substr(comp, comp, comp)`: `start` and `length` are components, not constants. The manual types
them as `component<integer [value >= 1]>` and `component<integer [value >= 0]>` as well as
integers (Sub-string extraction, Input parameters type). Each data point uses its own values:
`Me_2` is the start and `Me_3` the length of the substring of `Me_1` that `calc` stores in the new
measure `Me_4`.

- `"hello world"`, start 7, length 5: the 5 characters from the 7th one (h=1, e=2, l=3, l=4, o=5,
  space=6, w=7), that is `"world"`.
- `"abcdef"`, start 2, length 3: `"bcd"`.
- `"abc"`, start 2, length 10: only `"bc"` is left from the 2nd character, and "If `length` is
  omitted or overcomes the length of the input string, the substring ends at the end of the input
  string".
- Row 4 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
