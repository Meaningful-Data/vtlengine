`instr(DS_1, "o", 6)`: the search for `"o"` starts from the 6th character and the result is the
position in the whole string (String pattern location). At dataset level the result is a single
measure `int_var` of type integer.

- `"hello world"` (h=1, e=2, l=3, l=4, o=5, space=6, w=7, o=8, r=9, l=10, d=11): the `o` at 5 is
  before the 6th character, so the first one found is the `o` at 8.
- `"boot camp"` (b=1, o=2, o=3, t=4, space=5, c=6, a=7, m=8, p=9): both `o` come before the 6th
  character, so none is found and the result is 0.
- `"abcdeo"` (a=1, b=2, c=3, d=4, e=5, o=6): the `o` is exactly at the 6th character, where the
  search starts, so it is found and the result is 6. The page says that "The search starts from
  the *start*th character of the input string", and that with `start` omitted "the search starts
  from the 1st position", where a pattern has to be found too, so an occurrence at the `start`
  position counts.
- Row 4 has a null measure. A null operand gives a null result: the User Manual says that
  "most of the scalar operations (arithmetic, logical, and others) return null when any of
  their arguments is null", and the manual's own `length` and `instr` examples show it.
