`replace(comp, comp, comp)`: `pattern1` and `pattern2` are components, not constants. The manual
types both as `component<string>` as well as `string` (String pattern replacement, Input
parameters type). Each data point uses its own patterns: `Me_2` is the pattern to be replaced and
`Me_3` the replacing one, and `calc` stores the result in the new measure `Me_4`.

- `"hello world"`, `"world"` by `"VTL"`: `"hello VTL"`.
- `"banana"`, `"an"` by `"AN"`: "all the occurrences" are replaced (the `an` at 2 and the one at
  4), so the result is `"bANANa"`.
- `"abc"`, `"x"` by `"y"`: there is no occurrence, so the value is kept.
- Row 4 has a null `Me_1`. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
