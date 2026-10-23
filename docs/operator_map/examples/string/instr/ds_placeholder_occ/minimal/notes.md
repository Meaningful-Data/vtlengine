`instr(DS_1, "o", _, 2)`: the `_` leaves `start` out, so the search starts from the 1st position
("If `start` is omitted, the search starts from the 1st position"), and the result is the position
of the 2nd occurrence of `"o"` (String pattern location).

- `"hello world"` (`o` at 5 and 8): 8.
- `"o-o-o"` (`o` at 1, 3 and 5): the 2nd occurrence is the one at 3.
- `"hello"` (`o` at 5 only): there is no 2nd occurrence, so the result is 0.
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
