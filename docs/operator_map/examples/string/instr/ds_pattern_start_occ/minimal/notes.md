`instr(DS_1, "o", 2, 2)`: the search starts from the 2nd character and the result is the position
of the 2nd occurrence of `"o"` from there, or 0 when there is none (String pattern location).

- `"hello world"` (`o` at 5 and 8): the 2nd occurrence is the one at 8.
- `"o-o-o"` (`o` at 1, 3 and 5): the `o` at 1 is before the 2nd character, so the occurrences found
  are the ones at 3 (the 1st) and 5 (the 2nd), and the result is 5. Ignoring `start`, it would
  be 3.
- `"hello"` (`o` at 5 only): there is no 2nd occurrence, so the result is 0.
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
